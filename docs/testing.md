# Testing

Native validation covers API profile, buffer/memory, descriptor/pipeline,
command/queue/fence, shader capabilities, query/timestamp behavior, async
submission, fence-less cleanup and readback lifetime. Structural tests also
validate release dates, malformed source manifests, deterministic amalgamation,
ABI guard exception mapping, translator-header independence, and the four-symbol shared
driver export boundary. Capability-dependent
fp16, subgroup, integer-dot and cooperative-matrix paths skip when Dawn does
not expose the required feature. The default test build also compiles every
Vulkan frontend module with `VK_USE_64_BIT_PTR_DEFINES=0`.

Compiler-only regressions `test_nikki_cooperative_matrix_translation` and
`test_nikki_packed_integer_dot` validate real SPIR-V through Tint to WGSL without
requiring GPU feature support. The matrix test covers packed storage offsets,
strides and arithmetic; the dot test checks saturation against a wide-integer
oracle. The GPU compute test additionally checks 12 packed-dot results, including
all six opcodes, saturation boundaries and repeated accumulation. Both compiler
tests also run under Node in the wasm32/wasm64 CI lanes. These checks do
not replace execution on matrix-capable GPU hardware. The matrix executable
optionally accepts specialized SPIR-V files and prints their generated WGSL.

The compute test also checks scalar and `vec2` read/write/read sequences with
matching and disjoint dynamic indices, untouched guard values, and adjacent
loads ending at an odd-sized buffer's final float. Aliased `VkBuffer` handles
are tested across dispatches separated by a shader-write to shader-read memory
barrier. These cases adapt momoten's `buffer_read_write.comp` and
`coalesced_buffer_load.comp`; they do not assume a write through one descriptor
is visible through another descriptor in the same invocation. The fixtures use
the legacy SPIR-V `GLSL450` memory model, not `VulkanKHR`, and have no coherent
decorations or shader memory barriers. No runtime alias policy is changed.

```bash
cmake -S . -B build-native \
  -DNIKKI_BUILD_DRIVER=ON \
  -DNIKKI_BUILD_TESTS=ON
cmake --build build-native
ctest --test-dir build-native --output-on-failure
```

Repeat with `NIKKI_BUILD_AMALGAMATION=ON` to compile the generated source.
The generated file lives in the build tree and is not tracked by Git.
Use `tools/amalgamate.py --check build/dist/nikki.cpp --version-header build/include/nikki/version.h` to verify freshness;
`--stdout` is available for release tooling without creating a source-tree
artifact.

The browser lane requires the pinned emsdk/Emdawnwebgpu versions from the
top-level `CMakeLists.txt` and an actual browser WebGPU implementation. Configure
with `emcmake`, build `test_nikki_compute`, and run it in Chromium/SwiftShader
through the shared `tests/browser/run_webgpu_test.mjs` harness. A native Dawn
pass or successful wasm link does not substitute for this lane. The shared
harness is also used by the optional ncnn browser integration.

See [wasm-builds.md](wasm-builds.md) for the CI artifacts and browser test recipe.

ncnn integration tests are opt-in through `integrations/ncnn`; Native driver
and browser harness details live there rather than in ncnn. They are not part
of the generic driver inventory.

Installation can be checked without Dawn leaking into the consumer ABI:

```bash
cmake --install build-native --prefix /tmp/nikki-install
cmake -S tests/cmake/package_consumer -B build-package-consumer \
  -DCMAKE_PREFIX_PATH=/tmp/nikki-install
```

The installed native package uses an exported `nikki::driver` target. It does not
construct an imported library with `find_library()` and does not export the
static target or its private Dawn/Tint/SPIRV-Tools dependencies.
The [WASM SDK](wasm-builds.md) instead exports `nikki::static` with its complete
static link dependencies and is tested with a relocated consumer build.

# nikki

nikki is an experimental implementation of a restricted subset of the Vulkan API over WebGPU.

## Experimental and non-conformant implementation

nikki implements a restricted, buffer-oriented subset of the Vulkan API over WebGPU. It is not a complete Vulkan implementation, has not passed the Khronos Vulkan Conformance Test Suite, and is not a Khronos conformant product.

This project is intended for research, testing, and development only. It is not affiliated with, endorsed by, or certified by the Khronos Group.

Applications can submit compute commands and Vulkan Shader SPIR-V from this restricted API subset; nikki translates compute entry points through Tint to WGSL and replays the command stream with Dawn WebGPU.

The implementation is independent of ncnn. ncnn is one consumer: native builds can load the shared library through simplevk, while Emscripten builds link the static implementation without adding nikki runtime types to ncnn.

## Repository layout

```text
src/translator/         SPIR-V preparation, Tint legalization and WGSL output
src/backend/            Dawn objects, device profile, dispatch and async replay
src/*.cpp               internal impl_* Vulkan modules split by responsibility
src/vulkan_procs.inc    canonical bootstrap and proc-registry declarations
src/entrypoints.cpp     guarded Vulkan ABI wrappers and proc availability policy
src/source_manifest.txt canonical modular/amalgamation source order
third_party/dawn/       pinned Dawn source submodule
tools/                  Dawn bootstrap and amalgamation tools
nikkiConfig.cmake.in installed CMake package configuration template
tests/                  API, runtime, translator and structural tests
integrations/ncnn/      optional native and browser ncnn integration
docs/                   architecture, profile and testing contracts
```

The files under `src/` are the only development sources. `src/source_manifest.txt` is the single ordered inventory used by modular builds and `tools/amalgamate.py`. The generated single-source distribution is a build/release artifact and is not tracked by Git. It combines nikki's own sources, but it still requires the matching Dawn, Tint, SPIRV-Tools and header dependencies when compiled.

The native shared library directly exports only these Vulkan bootstrap entry points:

```text
vkGetInstanceProcAddr
vkEnumerateInstanceExtensionProperties
vkEnumerateInstanceLayerProperties
vkCreateInstance
```

All other supported commands are obtained through Vulkan proc lookup, with scope, core-version and enabled-extension policy declared in `src/vulkan_procs.inc`. This bootstrap-only native export boundary is intentional; the static target exposes the implementation to its link-time consumer instead.

## Standalone build

The source build requires CMake 3.22 or newer. Clone the project with its pinned Dawn submodule, bootstrap Dawn's dependencies, then build:

```bash
git clone --recursive https://github.com/nihui/nikki.git
cd nikki
python3 tools/bootstrap_dawn.py
cmake -S . -B build \
  -DNIKKI_BUILD_STATIC=ON \
  -DNIKKI_BUILD_DRIVER=ON \
  -DNIKKI_BUILD_TESTS=ON
cmake --build build
ctest --test-dir build --output-on-failure
```

`NIKKI_BUILD_STATIC`, `NIKKI_BUILD_DRIVER`, `NIKKI_BUILD_TESTS`, `NIKKI_BUILD_AMALGAMATION`, and `NIKKI_INSTALL` control the corresponding targets. The old `NIKKI_BUILD_STATIC_LIBRARY`, `NIKKI_BUILD_NATIVE_DRIVER`, and `NIKKI_USE_AMALGAMATION` names remain deprecated aliases. Native builds may choose a `STATIC` or `SHARED` Dawn monolithic library with `NIKKI_DAWN_LIBRARY_TYPE`.

Build-tree targets are:

```text
nikki::static
nikki::driver
```

The existing `nikki::nikki` alias remains available for `nikki::static`. When every library and test target is disabled, adding this directory does not locate or configure Dawn. An installation containing the native package exports `nikki::driver` without exposing Dawn, Tint or SPIRV-Tools as package dependencies. The static target remains build-tree-only.

## Single-source distribution

Generate and compile the single-source form with:

```bash
cmake -S . -B build -DNIKKI_BUILD_AMALGAMATION=ON
cmake --build build --target nikki_amalgamate
```

The output is `build/dist/nikki.cpp`. Amalgamation builds and installation generate it automatically; normal modular builds do not. Dependency revisions are frozen in the top-level `CMakeLists.txt`. `third_party/dawn` is the default Dawn source; `NIKKI_DAWN_SOURCE_DIR` remains available for parent-controlled builds.

## Run ncnn

Native ncnn selects the shared library through simplevk:

```bash
NCNN_VULKAN_DRIVER=/absolute/path/to/libnikki.so ./your_ncnn_program
```

Emscripten/browser integrations link `nikki::static`; they do not load a native shared-library implementation. Current ncnn detects its buffer-only profile through `GpuInfo::support_image_storage()`. nikki reports image formats as unsupported, so ncnn automatically avoids image allocation, descriptors and sampler setup. No nikki-specific ncnn source hook or buffer-only build option is required.

See [integrations/ncnn/README.md](integrations/ncnn/README.md) for sibling, embedded and browser builds.

## Tests

Generic tests cover the Vulkan API profile, ABI guards, buffer compute, asynchronous lifetime, translator/header boundaries, 32-bit handles, source manifest and native shared-library exports. Capability-dependent shader tests skip only when the selected Dawn adapter does not expose the required feature.

See [docs/testing.md](docs/testing.md) for native, amalgamated, browser, package and ncnn validation.

## Supported profile

nikki reports API version 1.1 within a deliberately narrow, buffer-oriented compute profile. Conditional shader support includes fp16, subgroups, packed integer dot product and cooperative matrices only when the selected Dawn adapter and Tint translation path provide the required behavior. Images, samplers and graphics pipelines are not supported; unsupported Vulkan or SPIR-V features fail explicitly.

See [docs/architecture.md](docs/architecture.md) for the implementation boundaries and [docs/vulkan-profile.md](docs/vulkan-profile.md) for the current feature contract.

## Development note

GPT was used extensively during the development of this project, including for initial code generation, implementation experiments, and test development. The resulting code was manually reviewed, corrected, and validated against the full ncnn test suite.

## License

nikki is licensed under the Apache License 2.0. See [LICENSE.txt](LICENSE.txt).

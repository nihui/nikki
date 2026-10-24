# Browser WASM builds

nikki's browser implementation is linked statically into its Emscripten
consumer. It is not a standalone Vulkan driver or a dynamically loadable wasm
library. The CI artifacts intentionally separate the intermediate static
archive from the linked browser smoke-test module.

## Reproducible build

The version pins are in the top-level `CMakeLists.txt`: emsdk 5.0.6, the
Dawn/Tint commit, and the SHA-verified Emdawnwebgpu port. Initialize only the
top-level Dawn submodule; recursive submodule checkout traverses unrelated,
sometimes inaccessible Chromium repositories.

```sh
git submodule update --init third_party/dawn
python3 tools/bootstrap_dawn.py
emcmake cmake -S . -B build-wasm -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DNIKKI_BUILD_STATIC=ON \
  -DNIKKI_BUILD_DRIVER=OFF \
  -DNIKKI_BUILD_TESTS=ON \
  -DNIKKI_INSTALL=OFF
cmake --build build-wasm --target nikki test_nikki_compute --parallel 2
```

## Artifacts and API contract

* `nikki-emscripten-static` contains `libnikki.a`, license notices, and
  compiler/version metadata. The archive is **not** a standalone SDK: the
  consumer must use compatible Tint/SPIRV-Tools objects and link Emdawnwebgpu
  at the final application link. Prefer `add_subdirectory(nikki)` and
  `target_link_libraries(<consumer> PRIVATE nikki::static)`.
* `nikki-webgpu-smoke` contains `test_nikki_compute.js`,
  `test_nikki_compute.wasm`, `webgpu_test.html`, and
  `webgpu_test_worker.mjs`, plus notices and build statistics. This is a
  runnable browser **test**, not a public JavaScript API.

The CI verifies artifact file lists, wasm/archive magic bytes, a usable
ES-module export in Chromium, successful Vulkan compute and readback,
the absence of WebGPU validation errors/device loss, and the worker's
asynchronous progression. It records raw and gzip sizes rather than
enforcing an arbitrary size threshold.

## Browser smoke test

Install the pinned Puppeteer dependency and run with a Chromium executable
that supports WebGPU:

```sh
npm ci --prefix tests/browser
NIKKI_WEBGPU_TEST_CHROME=/path/to/chrome \
NIKKI_WEBGPU_TEST_ADAPTER=swiftshader \
  node tests/browser/run_webgpu_test.mjs \
  build-wasm/tests/test_nikki_compute.js --require-webgpu
```

`swiftshader` uses the headless Chromium software adapter, so this lane
checks WebGPU execution without relying on the GitHub runner's graphics
hardware. Native ncnn integration uses the same harness and may also select
a hardware adapter. Browser support and optional WebGPU features depend on
the target browser/adapter, independently of compilation success.

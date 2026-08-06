# ncnn integration

Native ncnn loads nikki through simplevk and `NCNN_VULKAN_DRIVER`; no ncnn
source hook or nikki-specific API is required. Emscripten builds link the
`nikki::static` CMake target from the external integration project; ncnn
does not look up or name a particular Vulkan implementation.

For sibling checkouts, use a small parent project:

```cmake
cmake_minimum_required(VERSION 3.22)
project(ncnn_nikki_superbuild LANGUAGES C CXX)
set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NIKKI_BUILD_DRIVER ON CACHE BOOL "" FORCE)
set(NIKKI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(/path/to/ncnn ncnn)
add_subdirectory(/path/to/nikki nikki)
include(/path/to/nikki/integrations/ncnn/ncnn_nikki.cmake)
ncnn_nikki_register_ncnn_tests()
```

For an embedded checkout use relative `add_subdirectory(ncnn)` and
`add_subdirectory(nikki)` paths. Initialize nikki's Dawn submodule and
run `tools/bootstrap_dawn.py`, then build `nikki_native` and
`nikki_ncnn_model_test`. The adapter's tiny in-memory model compares CPU and
Vulkan outputs and returns 77 when no WebGPU adapter is available. Generic
nikki tests remain independent of ncnn.

Current ncnn detects buffer-only drivers through
`GpuInfo::support_image_storage()`. nikki returns
`VK_ERROR_FORMAT_NOT_SUPPORTED` from Vulkan image-format queries, so ncnn
automatically avoids image allocation, image descriptors and sampler setup.
No nikki-specific ncnn build option is required.

## Browser tests

The Emscripten ncnn integration owns its browser runner under `browser/` so
ncnn does not carry Puppeteer, HTML, Worker, or nikki validation support.
Install the local runner dependency with:

```bash
npm install --prefix integrations/ncnn/browser
```

Configure the browser superbuild after initializing Dawn:

```bash
emcmake cmake -S integrations/ncnn/browser -B build-ncnn-webgpu \
    -DNIKKI_NCNN_SOURCE_DIR=/path/to/ncnn \
    -DNCNN_BUILD_TESTS=ON \
    -DNCNN_BUILD_EXAMPLES=OFF \
    -DNCNN_BUILD_TOOLS=OFF
cmake --build build-ncnn-webgpu --target webgpu_tests -j8
```

ncnn keeps its normal `ncnn_add_test()` and `cmake/run_test.cmake` path. Run
CTest with the existing executable-loader environment variables; the runner
accepts the test module path as its positional argument:

```bash
cmake -E env \
    TESTS_EXECUTABLE_LOADER=/path/to/node \
    TESTS_EXECUTABLE_LOADER_ARGUMENTS=$PWD/integrations/ncnn/browser/run_webgpu_test.mjs \
    NIKKI_NCNN_TEST_CHROME=/path/to/chrome \
    NIKKI_NCNN_TEST_ADAPTER=swiftshader \
    ctest --test-dir build-ncnn-webgpu --output-on-failure --timeout 14400
```

`NIKKI_NCNN_TEST_ADAPTER` accepts `swiftshader` or `hardware`. The optional
`NIKKI_NCNN_TEST_CHROME_VERSION` verifies the browser version. Browser-only
target options, model preloading, Puppeteer, and page assets remain entirely in
the nikki integration.

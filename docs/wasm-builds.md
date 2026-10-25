# Browser WASM builds

nikki's browser implementation is linked statically into its Emscripten
consumer. It is not a standalone Vulkan driver, dynamically loadable WASM
library, or JavaScript SDK.

## Consume a prebuilt package

Use emsdk **6.0.12** and extract `nikki-YYYYMMDD-wasm32-emscripten.tar.gz`.
The archive contains one directory with the same basename. Point CMake at
that directory, not at a nikki source checkout:

```sh
emcmake cmake -S my-app -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -Dnikki_DIR=/path/to/nikki-YYYYMMDD-wasm32-emscripten/lib/cmake/nikki
cmake --build build
```

```cmake
find_package(nikki CONFIG REQUIRED)
target_link_libraries(my_app PRIVATE nikki::static)
```

Use the explicit `nikki_DIR`: Emscripten searches packages only under its
sysroot by default, so `CMAKE_PREFIX_PATH` alone is not sufficient.

The package includes `lib/libnikki.a`, the complete reachable static link
closure (Tint, SPIRV-Tools, Dawn utilities and Abseil), Vulkan headers,
the pinned Emdawnwebgpu runtime package (local port, JavaScript, C/C++ headers
and notices), Closure externs, and relocatable CMake package files. No
nikki/Dawn source or producer build tree is needed, and linking requires no
dependency downloads, even with an empty ports cache. Install emsdk 6.0.12
separately: this is not an offline toolchain bundle.

`nikki::static` propagates the port, ASYNCIFY, 65536-byte Asyncify stack,
1048576-byte WASM stack, and Release Closure options. Consumers choose their
own module exports, JavaScript interface, environment, heap and filesystem
options. The package is wasm32, single-threaded, and does not target MEMORY64
or pthreads. It checks the exact compiler version and bundled local port hash.

Root metadata files are `release-version.txt`, `source-revision.txt`,
`dawn-revision.txt` and `emsdk-version.txt`. The CMake package also exposes
`nikki_RELEASE_VERSION`, `nikki_EMSDK_VERSION`, and `nikki_DAWN_REVISION`.
Pin a successful producer workflow run and its full source revision when
consuming Actions artifacts. The artifact is named
`nikki-YYYYMMDD-wasm32-emscripten` and contains the single `.tar.gz` file.
Do not substitute the old `nikki-emscripten-static` artifact: it contains
only an intermediate archive and cannot satisfy this consumer contract.

## Build and package

The dependency pins remain in the top-level `CMakeLists.txt`: emsdk 6.0.12,
the Dawn/Tint commit, and the SHA-verified Emdawnwebgpu port and runtime
archive. Initialize only the top-level Dawn submodule; recursive checkout traverses unrelated,
sometimes inaccessible Chromium repositories.

When upgrading Emscripten, use fresh build and staging directories and rebuild
every object and static dependency. Do not reconfigure a 5.x build tree or reuse
its archives under 6.0.12, even if the compiler path is unchanged: cross-version
ABI compatibility is not guaranteed. The example below uses versioned directories.

```sh
git submodule update --init third_party/dawn
python3 tools/bootstrap_dawn.py
VERSION=$(date -u +'%Y%m%d')
export BINARYEN_CORES=1 # Bound link-time memory use, without changing optimization flags.
emcmake cmake -S . -B build-wasm-6.0.12 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_LIBDIR=lib \
  -DNIKKI_RELEASE_VERSION="$VERSION" \
  -DNIKKI_BUILD_STATIC=ON -DNIKKI_BUILD_DRIVER=OFF \
  -DNIKKI_BUILD_TESTS=ON -DNIKKI_INSTALL=ON
cmake --build build-wasm-6.0.12 --target nikki nikki_amalgamate test_nikki_compute --parallel 2
cmake --install build-wasm-6.0.12 --prefix "$PWD/stage-wasm-6.0.12"
python3 tools/package_wasm.py --stage stage-wasm-6.0.12 --output release --version "$VERSION"
```

When configuring a WASM installation, CMake downloads and SHA512-verifies
the frozen Emdawnwebgpu release package for inclusion in the SDK. Subsequent
configures reuse the verified archive in the build directory. Native builds
and source-build port selection are unchanged.

`NIKKI_RELEASE_VERSION` defaults to the UTC configure date and is cached.
It must be a real date in 2000–2099. The Vulkan `driverVersion` uses the same
encoding as momoten: `(year - 2000).month.day`, so `20261024` becomes
`26.10.24`. `driverInfo` contains the release date; registered `driverID` 33
and the experimental/non-conformant status are unchanged. The API project
version remains 0.1.0. The generated amalgamation embeds its configured
version header.

The WASM workflow selects its date once, builds and packages the SDK,
extracts it to a different directory, then hides the entire producer
checkout while compiling a clean consumer. The consumer has a fresh ports
cache and no cached Emdawn library. The workflow verifies that the ports
cache stays empty and the bundled runtime is rebuilt, proving no dependency
port was downloaded or reused from the producer. It runs that consumer's
Vulkan compute and readback test in Chromium SwiftShader, checking driver ID/date,
WebGPU validation errors, device loss and asynchronous worker progression.
This catches missing static/runtime dependencies, accidental downloads and
absolute build-tree paths.
The `nikki-YYYYMMDD-webgpu-smoke` artifact is the linked browser test,
not a public JavaScript API. It records raw/gzip JS and WASM sizes.

Pushes and PRs only build artifacts. A manual `wasm` workflow run publishes
a date-tagged release only when its `publish` input is explicitly enabled,
after all package and browser checks pass. Existing date tags are never
repointed to a different source commit.

## Browser smoke test

```sh
npm ci --prefix tests/browser
NIKKI_WEBGPU_TEST_CHROME=/path/to/chrome \
NIKKI_WEBGPU_TEST_ADAPTER=swiftshader \
  node tests/browser/run_webgpu_test.mjs \
  build-wasm-6.0.12/tests/test_nikki_compute.js --require-webgpu
```

SwiftShader tests WebGPU execution without assuming physical GPU access.
Browser/adapter support and optional features are distinct from compile
and link success. Serve browser artifacts over HTTP, not `file://`.

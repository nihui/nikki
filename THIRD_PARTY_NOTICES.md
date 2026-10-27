# Third-party notices

nikki uses Khronos Vulkan-Headers, SPIRV-Headers and SPIRV-Tools, and Google
Dawn/Tint. Their pinned revisions are recorded in the top-level `CMakeLists.txt` and
their license files remain in their respective source checkouts. The frozen
Emdawnwebgpu port under `ports/` is derived from the matching Dawn/emsdk
release. These dependencies are not covered by nikki's Apache-2.0 license;
their respective upstream licenses continue to apply.

The WASM SDK redistributes the Emdawnwebgpu package with its upstream notices
in `share/nikki/ports/emdawnwebgpu_pkg`, including `webgpu/src/LICENSE`
(Emscripten's MIT and University of Illinois/NCSA licenses),
`webgpu_cpp/LICENSE` (BSD 3-Clause), and the local port's license header.
Other bundled dependency licenses are in `share/nikki/licenses`.

The pinned Dawn/Tint source is built with nikki's downstream
`patches/dawn-wgsl-matrix-i8.patch`, which adds missing WGSL writer emission for
existing signed/unsigned 8-bit subgroup-matrix element types. It does not add a
standalone scalar 8-bit WGSL extension. Bootstrap checks the upstream revision
and exact pre/post-patch source hashes. SDK archives record the upstream base
revision in `dawn-revision.txt` and the applied patch digest in
`dawn-patch-sha256.txt`.

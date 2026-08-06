# Architecture

The exported Vulkan ABI is the only runtime consumer boundary. The four
bootstrap wrappers are listed in `src/vulkan_procs.inc` together with every
proc-resolved command. Function pointers in that registry lead through
`VulkanAbiGuard` to the `impl_*` frontend functions. Dependencies flow in one
direction:

```text
Vulkan entry points and object/state validation
                    |
                    v
       backend replay and Dawn lifetime
                    |
                    v
              Dawn WebGPU API

pipeline creation -> translator stages -> Tint/SPIRV-Tools
```

`src/vulkan_internal.h` contains lightweight Vulkan handle wrappers, handle
tables, checked arithmetic and frontend declarations. It intentionally
contains no WebGPU, backend, or Tint header. `src/backend/objects.h` owns Vulkan
objects backed by WGPU state. Operations are separated into `device.h`,
`runtime.h`, `dispatch.h`, and `shader_module.h`, so object definitions do not
act as an umbrella for every backend operation. Translator types live in
`src/translator/translator_internal.h`; stage-only Tint declarations live in
`translator_stages.h`.

Adapter capabilities are represented at three boundaries. `DeviceProfile`
records physical WebGPU capabilities, `EnabledDeviceProfile` records the
features enabled for a Vulkan logical device, and `ShaderTranslationProfile`
is an owning value snapshot consumed by the translator. The translator does
not accept `Device`, `PhysicalDevice`, `PipelineLayout`, WGPU handles, or
backend resource types.

Vulkan implementation files follow object responsibility: device, memory,
pipeline, command, queue, sync and entrypoints. `src/backend/dispatch.cpp`
encodes compute dispatches, aliases and internal compute kernels;
`src/backend/runtime.cpp` owns command replay and asynchronous
submission/readback processing. Translation is separated into SPIR-V
preparation, immediate data, invocation builtins, subgroup operations,
cooperative matrices, reflection and WGSL generation. WebGPU shader-module
creation remains in the backend.

Application dispatch dimensions are not flattened, split or passed to the
shader through hidden push-constant data. A dispatch that exceeds the WebGPU
device limit is rejected while recording the Vulkan command buffer. The
llvmpipe-only workgroup reshape is a separate pipeline compatibility path: it
is enabled only for shaders proven independent of local invocation layout and
scales the physical dispatch by the inverse workgroup-size change.

`src/source_manifest.txt` is the sole ordered source inventory for both CMake
module targets and `tools/amalgamate.py`. Its global order is the amalgamation
order; group names describe responsibility and do not reorder entries. The
generated file is tested for byte-for-byte freshness and behavior but is never
the development source.

This project and momoten intentionally use a similar project skeleton and
navigation model. They do not share runtime objects, command replay, memory
semantics, or shader translation implementations.

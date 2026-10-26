# Restricted API compute profile

nikki is an experimental, non-conformant implementation of a deliberately
narrow compute-only subset of the Vulkan API. It reports API version 1.1
within that subset and implements instance/device discovery, one
compute/transfer queue, buffers and mapped memory, descriptor layouts and push
descriptors, update templates,
specialization and push constants, compute pipelines, command recording,
buffer copy, dispatch, barriers, submission, fences, timestamps and
pipeline-cache compatibility objects.

Core Vulkan 1.1 entry points and their KHR aliases resolve to the same
implementation when the required extension was enabled. Proc lookup respects
global/instance/device scope and enabled-extension state. Unsupported image and
graphics entry points are not exposed.

`VK_KHR_driver_properties` reports conformance version `0.0.0.0`.
`driverName` is `nikki`, and `deviceName` prefixes the adapter name with
`nikki: `. `driverInfo` and the startup warning retain the experimental,
non-conformant status.

Shader support includes ncnn's storage-buffer compute subset plus conditional
fp16, subgroup, packed integer-dot and cooperative-matrix translation when the
selected Dawn adapter and Tint path support the complete behavior. Unsupported
SPIR-V is rejected; it is never silently approximated.

Packed 4x8 integer dots support signed, unsigned and mixed-sign inputs, including
saturating accumulation. Mixed-sign dots and saturation are lowered to exact
integer operations, not advertised as hardware-accelerated instructions.

Cooperative FP16 matrices can use packed 32-bit storage without changing its
byte layout. Matrix/matrix element-wise addition, subtraction and multiplication
use per-subgroup workgroup scratch; this fallback requires full subgroups and
entry-point-level operations without conditional early returns. Unsupported
control flow and explicit array strides that cannot be preserved fail explicitly.

Command buffers use explicit initial, recording, executable, pending and
invalid states. Recording errors are deferred to `vkEndCommandBuffer`, and
submission resources remain alive until Dawn completion/readback processing.

// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_TEST_COOPERATIVE_MATRIX_BINDING_SPIRV_H
#define NIKKI_TEST_COOPERATIVE_MATRIX_BINDING_SPIRV_H

// A 16x16 FP16 accumulator is stored in a packed float[] storage buffer.
// It requires 512 bytes despite the runtime array's ordinary 4-byte minimum.
// Equivalent SPIR-V assembly (assemble for SPIR-V 1.3):
// OpCapability Shader
// OpCapability Float16
// OpCapability VulkanMemoryModel
// OpCapability CooperativeMatrixKHR
// OpExtension "SPV_KHR_cooperative_matrix"
// OpExtension "SPV_KHR_vulkan_memory_model"
// OpMemoryModel Logical Vulkan
// OpEntryPoint GLCompute %main "main"
// OpExecutionMode %main LocalSize 32 1 1
// OpDecorate %array ArrayStride 4
// OpDecorate %block Block
// OpMemberDecorate %block 0 Offset 0
// OpDecorate %buffer DescriptorSet 0
// OpDecorate %buffer Binding 0
// %void = OpTypeVoid
// %function = OpTypeFunction %void
// %u32 = OpTypeInt 32 0
// %f32 = OpTypeFloat 32
// %f16 = OpTypeFloat 16
// %zero = OpConstant %u32 0
// %two = OpConstant %u32 2
// %scope = OpConstant %u32 3
// %size = OpConstant %u32 16
// %stride = OpConstant %u32 8
// %one = OpConstant %f16 1
// %array = OpTypeRuntimeArray %f32
// %block = OpTypeStruct %array
// %block_ptr = OpTypePointer StorageBuffer %block
// %element_ptr = OpTypePointer StorageBuffer %f32
// %buffer = OpVariable %block_ptr StorageBuffer
// %matrix = OpTypeCooperativeMatrixKHR %f16 %scope %size %size %two
// %main = OpFunction %void None %function
// %entry = OpLabel
// %ptr = OpAccessChain %element_ptr %buffer %zero %zero
// %value = OpCompositeConstruct %matrix %one
// OpCooperativeMatrixStoreKHR %ptr %value %zero %stride
// OpReturn
// OpFunctionEnd

static const uint32_t cooperative_matrix_binding_spv[] = {
    0x07230203, 0x00010300, 0x00000000, 0x00000016, 0x00000000, 0x00020011, 0x00000001, 0x00020011,
    0x00000009, 0x00020011, 0x000014e1, 0x00020011, 0x00001786, 0x0008000a, 0x5f565053, 0x5f52484b,
    0x706f6f63, 0x74617265, 0x5f657669, 0x7274616d, 0x00007869, 0x0008000a, 0x5f565053, 0x5f52484b,
    0x6b6c7576, 0x6d5f6e61, 0x726f6d65, 0x6f6d5f79, 0x006c6564, 0x0003000e, 0x00000000, 0x00000003,
    0x0005000f, 0x00000005, 0x00000012, 0x6e69616d, 0x00000000, 0x00060010, 0x00000012, 0x00000011,
    0x00000020, 0x00000001, 0x00000001, 0x00040047, 0x0000000c, 0x00000006, 0x00000004, 0x00030047,
    0x0000000d, 0x00000002, 0x00050048, 0x0000000d, 0x00000000, 0x00000023, 0x00000000, 0x00040047,
    0x00000010, 0x00000022, 0x00000000, 0x00040047, 0x00000010, 0x00000021, 0x00000000, 0x00020013,
    0x00000001, 0x00030021, 0x00000002, 0x00000001, 0x00040015, 0x00000003, 0x00000020, 0x00000000,
    0x00030016, 0x00000004, 0x00000020, 0x00030016, 0x00000005, 0x00000010, 0x0004002b, 0x00000003,
    0x00000006, 0x00000000, 0x0004002b, 0x00000003, 0x00000007, 0x00000002, 0x0004002b, 0x00000003,
    0x00000008, 0x00000003, 0x0004002b, 0x00000003, 0x00000009, 0x00000010, 0x0004002b, 0x00000003,
    0x0000000a, 0x00000008, 0x0004002b, 0x00000005, 0x0000000b, 0x00003c00, 0x0003001d, 0x0000000c,
    0x00000004, 0x0003001e, 0x0000000d, 0x0000000c, 0x00040020, 0x0000000e, 0x0000000c, 0x0000000d,
    0x00040020, 0x0000000f, 0x0000000c, 0x00000004, 0x0004003b, 0x0000000e, 0x00000010, 0x0000000c,
    0x00071168, 0x00000011, 0x00000005, 0x00000008, 0x00000009, 0x00000009, 0x00000007, 0x00050036,
    0x00000001, 0x00000012, 0x00000000, 0x00000002, 0x000200f8, 0x00000013, 0x00060041, 0x0000000f,
    0x00000014, 0x00000010, 0x00000006, 0x00000006, 0x00040050, 0x00000011, 0x00000015, 0x0000000b,
    0x0005116a, 0x00000014, 0x00000015, 0x00000006, 0x0000000a, 0x000100fd, 0x00010038,
};

#endif

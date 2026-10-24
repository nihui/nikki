// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

int normalize_spirv(std::vector<uint32_t>& spirv)
{
    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
        return -1;

    const uint32_t bound = spirv[3];
    uint32_t local_size_ids[3] = {0, 0, 0};
    uint32_t workgroup_size_builtin_id = 0;
    std::vector<int32_t> spec_ids(bound, -1);
    std::vector<uint32_t> workgroup_size_members;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if (opcode == spv::Op::OpExecutionModeId && word_count >= 6 && spirv[i + 2] == (uint32_t)spv::ExecutionMode::LocalSizeId)
        {
            local_size_ids[0] = spirv[i + 3];
            local_size_ids[1] = spirv[i + 4];
            local_size_ids[2] = spirv[i + 5];
        }
        else if (opcode == spv::Op::OpDecorate && word_count >= 4 && spirv[i + 1] < bound)
        {
            if (spirv[i + 2] == (uint32_t)spv::Decoration::SpecId)
                spec_ids[spirv[i + 1]] = spirv[i + 3];
            else if (spirv[i + 2] == (uint32_t)spv::Decoration::BuiltIn && spirv[i + 3] == (uint32_t)spv::BuiltIn::WorkgroupSize)
                workgroup_size_builtin_id = spirv[i + 1];
        }
        else if (opcode == spv::Op::OpSpecConstantComposite && word_count == 6 && spirv[i + 2] == workgroup_size_builtin_id)
        {
            workgroup_size_members.assign(spirv.begin() + i + 3, spirv.begin() + i + 6);
        }

        i += word_count;
    }

    bool has_workgroup_size_builtin = workgroup_size_members.size() == 3;
    for (int i = 0; i < 3 && has_workgroup_size_builtin; i++)
    {
        const uint32_t id = workgroup_size_members[i];
        if (id >= bound || spec_ids[id] != 233 + i)
            has_workgroup_size_builtin = false;
    }

    const bool has_local_size_id_mode = local_size_ids[0] != 0 && local_size_ids[1] != 0 && local_size_ids[2] != 0;

    std::vector<uint32_t> normalized;
    normalized.reserve(spirv.size());
    normalized.insert(normalized.end(), spirv.begin(), spirv.begin() + 5);

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        const bool non_readable_decorate = opcode == spv::Op::OpDecorate && word_count >= 3 && spirv[i + 2] == (uint32_t)spv::Decoration::NonReadable;
        const bool non_readable_member_decorate = opcode == spv::Op::OpMemberDecorate && word_count >= 4 && spirv[i + 3] == (uint32_t)spv::Decoration::NonReadable;
        const bool literal_local_size = has_workgroup_size_builtin && opcode == spv::Op::OpExecutionMode && word_count == 6 && spirv[i + 2] == (uint32_t)spv::ExecutionMode::LocalSize;
        bool duplicate_local_size_spec_id = false;
        if (has_local_size_id_mode && opcode == spv::Op::OpDecorate && word_count >= 4 && spirv[i + 2] == (uint32_t)spv::Decoration::SpecId)
        {
            const uint32_t spec_id = spirv[i + 3];
            if (spec_id >= 233 && spec_id <= 235 && spirv[i + 1] != local_size_ids[spec_id - 233])
                duplicate_local_size_spec_id = true;
        }

        if (!non_readable_decorate && !non_readable_member_decorate && !literal_local_size && !duplicate_local_size_spec_id)
            normalized.insert(normalized.end(), spirv.begin() + i, spirv.begin() + i + word_count);

        i += word_count;
    }

    spirv.swap(normalized);
    return 0;
}

int specialize_spirv(std::vector<uint32_t>& spirv, const VkSpecializationInfo* info)
{
    if (info && info->dataSize != 0 && !info->pData)
        return -1;
    if (info && info->mapEntryCount != 0 && !info->pMapEntries)
        return -1;

    std::unordered_map<uint32_t, std::vector<uint32_t> > values;
    if (info)
    {
        for (uint32_t i = 0; i < info->mapEntryCount; i++)
        {
            const VkSpecializationMapEntry& entry = info->pMapEntries[i];
            if (entry.size != 4 || entry.offset > info->dataSize
                || entry.size > info->dataSize - entry.offset)
            {
                log_error("specialization id %u has invalid offset=%u size=%llu data-size=%llu",
                          entry.constantID, entry.offset, (unsigned long long)entry.size,
                          (unsigned long long)info->dataSize);
                return -1;
            }

            if (values.find(entry.constantID) != values.end())
            {
                log_error("specialization id %u is specified more than once", entry.constantID);
                return -1;
            }

            std::vector<uint32_t>& value = values[entry.constantID];
            value.resize(1);
            memcpy(value.data(), (const unsigned char*)info->pData + entry.offset, entry.size);
        }
    }

    spvtools::Optimizer optimizer(SPV_ENV_VULKAN_1_1);
    optimizer.SetMessageConsumer([](spv_message_level_t, const char*, const spv_position_t& position, const char* message) {
        log_error("SPIR-V optimizer word=%zu: %s", position.index, message ? message : "");
    });
    if (!values.empty())
        optimizer.RegisterPass(spvtools::CreateSetSpecConstantDefaultValuePass(values));
    optimizer.RegisterPass(spvtools::CreateFreezeSpecConstantValuePass());
    optimizer.RegisterPass(spvtools::CreateFoldSpecConstantOpAndCompositePass());
    optimizer.RegisterPass(spvtools::CreateDeadBranchElimPass());
    optimizer.RegisterPass(spvtools::CreateAggressiveDCEPass(false));
    optimizer.RegisterPass(spvtools::CreateRemoveUnusedInterfaceVariablesPass());

    std::vector<uint32_t> specialized;
    if (!optimizer.Run(spirv.data(), spirv.size(), &specialized))
    {
        log_error("SPIR-V specialization failed");
        return -1;
    }

    simplify_bounded_integer_loops(specialized);
    spirv.swap(specialized);
    return 0;
}

int legalize_non_finite_fp16_constants(std::vector<uint32_t>& spirv)
{
    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
        return -1;

    std::unordered_set<uint32_t> fp16_types;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if (opcode == spv::Op::OpTypeFloat && word_count == 3 && spirv[i + 2] == 16)
            fp16_types.insert(spirv[i + 1]);

        i += word_count;
    }

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if ((opcode == spv::Op::OpConstant || opcode == spv::Op::OpSpecConstant)
            && word_count == 4 && fp16_types.count(spirv[i + 1]) != 0)
        {
            const uint16_t bits = (uint16_t)spirv[i + 3];
            if ((bits & 0x7c00) == 0x7c00)
            {
                if ((bits & 0x03ff) != 0)
                {
                    log_error("non-finite f16 NaN constants are not supported by WGSL");
                    return -1;
                }

                // WGSL shader-creation constants must be finite. Match its
                // runtime f32-to-f16 conversion by clamping overflow.
                spirv[i + 3] = (uint32_t)((bits & 0x8000) | 0x7bff);
            }
        }

        i += word_count;
    }

    return 0;
}

std::string spirv_literal_string(const uint32_t* words, size_t word_count)
{
    std::string value;
    value.reserve(word_count * 4);
    for (size_t i = 0; i < word_count; i++)
    {
        for (int j = 0; j < 4; j++)
        {
            const char ch = (char)((words[i] >> (j * 8)) & 0xff);
            if (ch == 0)
                return value;
            value.push_back(ch);
        }
    }

    return value;
}

// Tint's SPIR-V reader does not accept SPV_KHR_integer_dot_product yet. Replace
// packed dot instructions with a distinctive reverseBits(reverseBits(x)) ^ y
// sequence that Tint can read, then restore the matching WGSL builtins in IR.
int legalize_packed_integer_dot_spirv(std::vector<uint32_t>& spirv,
                                      size_t& packed_integer_dot_count)
{
    packed_integer_dot_count = 0;
    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
        return -1;

    bool has_integer_dot_declaration = false;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if (opcode == spv::Op::OpSDot || opcode == spv::Op::OpUDot)
        {
            if (word_count != 6 || spirv[i + 5] != (uint32_t)spv::PackedVectorFormat::PackedVectorFormat4x8Bit)
            {
                log_error("only packed 4x8 OpSDot and OpUDot are supported");
                return -1;
            }

            packed_integer_dot_count++;
        }
        else if ((opcode == spv::Op::OpExtension
                  && spirv_literal_string(&spirv[i + 1], word_count - 1) == "SPV_KHR_integer_dot_product")
                 || (opcode == spv::Op::OpCapability && word_count == 2
                     && (spirv[i + 1] == (uint32_t)spv::Capability::DotProductInput4x8BitPacked
                         || spirv[i + 1] == (uint32_t)spv::Capability::DotProduct)))
        {
            has_integer_dot_declaration = true;
        }
        else if (opcode == spv::Op::OpSUDot
                 || opcode == spv::Op::OpSDotAccSat
                 || opcode == spv::Op::OpUDotAccSat
                 || opcode == spv::Op::OpSUDotAccSat)
        {
            log_error("mixed-signedness and accumulating-saturating integer dot products are not supported");
            return -1;
        }

        i += word_count;
    }

    if (packed_integer_dot_count == 0 && !has_integer_dot_declaration)
        return 0;
    if (packed_integer_dot_count > (UINT32_MAX - spirv[3]) / 2)
        return -1;

    std::vector<uint32_t> legalized;
    legalized.reserve(spirv.size() + packed_integer_dot_count * 8);
    legalized.insert(legalized.end(), spirv.begin(), spirv.begin() + 5);
    uint32_t next_id = spirv[3];
    legalized[3] += (uint32_t)packed_integer_dot_count * 2;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);

        const bool integer_dot_extension = opcode == spv::Op::OpExtension
                                           && spirv_literal_string(&spirv[i + 1], word_count - 1) == "SPV_KHR_integer_dot_product";
        const bool integer_dot_capability = opcode == spv::Op::OpCapability && word_count == 2
                                            && (spirv[i + 1] == (uint32_t)spv::Capability::DotProductInput4x8BitPacked
                                                || spirv[i + 1] == (uint32_t)spv::Capability::DotProduct);
        if (opcode == spv::Op::OpSDot || opcode == spv::Op::OpUDot)
        {
            const uint32_t first_reverse_id = next_id++;
            const uint32_t second_reverse_id = next_id++;
            legalized.push_back((4u << 16) | (uint32_t)spv::Op::OpBitReverse);
            legalized.push_back(spirv[i + 1]);
            legalized.push_back(first_reverse_id);
            legalized.push_back(spirv[i + 3]);
            legalized.push_back((4u << 16) | (uint32_t)spv::Op::OpBitReverse);
            legalized.push_back(spirv[i + 1]);
            legalized.push_back(second_reverse_id);
            legalized.push_back(first_reverse_id);
            legalized.push_back((5u << 16) | (uint32_t)spv::Op::OpBitwiseXor);
            legalized.push_back(spirv[i + 1]);
            legalized.push_back(spirv[i + 2]);
            legalized.push_back(second_reverse_id);
            legalized.push_back(spirv[i + 4]);
        }
        else if (!integer_dot_extension && !integer_dot_capability)
        {
            legalized.insert(legalized.end(), spirv.begin() + i, spirv.begin() + i + word_count);
        }

        i += word_count;
    }

    if (next_id != legalized[3])
        return -1;

    spirv.swap(legalized);
    return 0;
}

bool spirv_instruction_has_result_type(spv::Op opcode)
{
    switch (opcode)
    {
    case spv::Op::OpUndef:
    case spv::Op::OpConstantTrue:
    case spv::Op::OpConstantFalse:
    case spv::Op::OpConstant:
    case spv::Op::OpConstantComposite:
    case spv::Op::OpConstantNull:
    case spv::Op::OpSpecConstantTrue:
    case spv::Op::OpSpecConstantFalse:
    case spv::Op::OpSpecConstant:
    case spv::Op::OpSpecConstantComposite:
    case spv::Op::OpSpecConstantOp:
    case spv::Op::OpVariable:
    case spv::Op::OpLoad:
    case spv::Op::OpAccessChain:
    case spv::Op::OpInBoundsAccessChain:
    case spv::Op::OpPtrAccessChain:
    case spv::Op::OpInBoundsPtrAccessChain:
    case spv::Op::OpCompositeConstruct:
    case spv::Op::OpCompositeExtract:
    case spv::Op::OpCompositeInsert:
    case spv::Op::OpCopyObject:
    case spv::Op::OpTranspose:
    case spv::Op::OpConvertFToU:
    case spv::Op::OpConvertFToS:
    case spv::Op::OpConvertSToF:
    case spv::Op::OpConvertUToF:
    case spv::Op::OpUConvert:
    case spv::Op::OpSConvert:
    case spv::Op::OpFConvert:
    case spv::Op::OpBitcast:
    case spv::Op::OpSNegate:
    case spv::Op::OpFNegate:
    case spv::Op::OpIAdd:
    case spv::Op::OpFAdd:
    case spv::Op::OpISub:
    case spv::Op::OpFSub:
    case spv::Op::OpIMul:
    case spv::Op::OpFMul:
    case spv::Op::OpUDiv:
    case spv::Op::OpSDiv:
    case spv::Op::OpFDiv:
    case spv::Op::OpPhi:
    case spv::Op::OpSelect:
    case spv::Op::OpFunctionParameter:
    case spv::Op::OpFunctionCall:
    case spv::Op::OpCooperativeMatrixLoadKHR:
    case spv::Op::OpCooperativeMatrixMulAddKHR:
        return true;
    default:
        return false;
    }
}

} // namespace nikki_detail

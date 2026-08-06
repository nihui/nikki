// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

// Tint requires uniform control flow for some GlobalInvocationId.z uses.
// Reconstruct z from WorkgroupId when local_size_z is known to be one.
int legalize_uniform_global_invocation_z(std::vector<uint32_t>& spirv)
{
    if (spirv.size() < 5 || spirv[0] != (uint32_t)spv::MagicNumber)
        return -1;

    const uint32_t bound = spirv[3];
    uint32_t global_invocation_id = 0;
    uint32_t workgroup_id = 0;
    uint32_t global_invocation_pointer_type = 0;
    uint32_t local_size_z = 0;
    uint32_t local_size_z_id = 0;
    std::vector<uint32_t> constant_values(bound, 0);
    std::vector<unsigned char> constant_known(bound, 0);

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return -1;

        if (opcode == spv::Op::OpDecorate && word_count >= 4
            && spirv[i + 2] == (uint32_t)spv::Decoration::BuiltIn)
        {
            if (spirv[i + 3] == (uint32_t)spv::BuiltIn::GlobalInvocationId)
                global_invocation_id = spirv[i + 1];
            else if (spirv[i + 3] == (uint32_t)spv::BuiltIn::WorkgroupId)
                workgroup_id = spirv[i + 1];
        }
        else if (opcode == spv::Op::OpConstant && word_count == 4 && spirv[i + 2] < bound)
        {
            constant_values[spirv[i + 2]] = spirv[i + 3];
            constant_known[spirv[i + 2]] = 1;
        }
        else if (opcode == spv::Op::OpExecutionMode && word_count >= 6
                 && spirv[i + 2] == (uint32_t)spv::ExecutionMode::LocalSize)
        {
            local_size_z = spirv[i + 5];
        }
        else if (opcode == spv::Op::OpExecutionModeId && word_count >= 6
                 && spirv[i + 2] == (uint32_t)spv::ExecutionMode::LocalSizeId)
        {
            local_size_z_id = spirv[i + 5];
        }

        i += word_count;
    }

    if (local_size_z == 0 && local_size_z_id < bound && constant_known[local_size_z_id])
        local_size_z = constant_values[local_size_z_id];

    if (global_invocation_id == 0 || local_size_z != 1)
        return 0;

    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (opcode == spv::Op::OpVariable && word_count >= 4 && spirv[i + 2] == global_invocation_id)
        {
            global_invocation_pointer_type = spirv[i + 1];
            break;
        }
        i += word_count;
    }
    if (global_invocation_pointer_type == 0)
        return -1;

    bool rewrite_needed = false;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if ((opcode == spv::Op::OpAccessChain || opcode == spv::Op::OpInBoundsAccessChain)
            && word_count == 5 && spirv[i + 3] == global_invocation_id
            && spirv[i + 4] < bound && constant_known[spirv[i + 4]]
            && constant_values[spirv[i + 4]] == 2)
        {
            rewrite_needed = true;
            break;
        }
        i += word_count;
    }
    if (!rewrite_needed)
        return 0;

    const bool create_workgroup_id = workgroup_id == 0;
    if (create_workgroup_id)
        workgroup_id = bound;
    std::vector<uint32_t> legalized;
    legalized.reserve(spirv.size() + 16);
    legalized.insert(legalized.end(), spirv.begin(), spirv.begin() + 5);
    if (create_workgroup_id)
        legalized[3] = bound + 1;

    bool decorate_inserted = !create_workgroup_id;
    bool variable_inserted = !create_workgroup_id;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);

        if (!decorate_inserted && opcode >= spv::Op::OpTypeVoid && opcode <= spv::Op::OpTypeForwardPointer)
        {
            legalized.push_back((4u << 16) | (uint32_t)spv::Op::OpDecorate);
            legalized.push_back(workgroup_id);
            legalized.push_back((uint32_t)spv::Decoration::BuiltIn);
            legalized.push_back((uint32_t)spv::BuiltIn::WorkgroupId);
            decorate_inserted = true;
        }
        if (!variable_inserted && opcode == spv::Op::OpFunction)
        {
            legalized.push_back((4u << 16) | (uint32_t)spv::Op::OpVariable);
            legalized.push_back(global_invocation_pointer_type);
            legalized.push_back(workgroup_id);
            legalized.push_back((uint32_t)spv::StorageClass::Input);
            variable_inserted = true;
        }

        if (opcode == spv::Op::OpEntryPoint && create_workgroup_id)
        {
            legalized.push_back(((word_count + 1) << 16) | (uint32_t)opcode);
            legalized.insert(legalized.end(), spirv.begin() + i + 1, spirv.begin() + i + word_count);
            legalized.push_back(workgroup_id);
        }
        else if ((opcode == spv::Op::OpAccessChain || opcode == spv::Op::OpInBoundsAccessChain)
                 && word_count == 5 && spirv[i + 3] == global_invocation_id
                 && spirv[i + 4] < bound && constant_known[spirv[i + 4]]
                 && constant_values[spirv[i + 4]] == 2)
        {
            legalized.insert(legalized.end(), spirv.begin() + i, spirv.begin() + i + word_count);
            legalized[legalized.size() - 2] = workgroup_id;
        }
        else
        {
            legalized.insert(legalized.end(), spirv.begin() + i, spirv.begin() + i + word_count);
        }

        i += word_count;
    }

    if (!decorate_inserted || !variable_inserted)
        return -1;

    spirv.swap(legalized);
    return 0;
}

bool can_reshape_workgroup(const std::vector<uint32_t>& spirv)
{
    uint32_t workgroup_size_id = 0;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return false;

        if (opcode == spv::Op::OpDecorate && word_count >= 4
            && spirv[i + 2] == (uint32_t)spv::Decoration::BuiltIn)
        {
            const spv::BuiltIn builtin = (spv::BuiltIn)spirv[i + 3];
            if (builtin == spv::BuiltIn::WorkgroupSize)
                workgroup_size_id = spirv[i + 1];
            else if (builtin != spv::BuiltIn::GlobalInvocationId)
                return false;
        }
        else if (opcode == spv::Op::OpVariable && word_count >= 4
                 && spirv[i + 3] == (uint32_t)spv::StorageClass::Workgroup)
        {
            return false;
        }
        else if (opcode == spv::Op::OpControlBarrier || opcode == spv::Op::OpMemoryBarrier)
        {
            return false;
        }
        else if (opcode == spv::Op::OpCapability && word_count >= 2)
        {
            const spv::Capability capability = (spv::Capability)spirv[i + 1];
            if ((capability >= spv::Capability::GroupNonUniform
                 && capability <= spv::Capability::GroupNonUniformQuad)
                || capability == spv::Capability::GroupNonUniformPartitionedEXT
                || capability == spv::Capability::GroupNonUniformRotateKHR)
                return false;
        }

        i += word_count;
    }

    bool inside_function = false;
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (opcode == spv::Op::OpFunction)
            inside_function = true;
        else if (opcode == spv::Op::OpFunctionEnd)
            inside_function = false;
        else if (inside_function && workgroup_size_id != 0)
        {
            for (uint32_t j = 1; j < word_count; j++)
            {
                if (spirv[i + j] == workgroup_size_id)
                    return false;
            }
        }

        i += word_count;
    }

    return true;
}

} // namespace nikki_detail

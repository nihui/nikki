// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

static const char* wgsl_builtin_name(tint::core::BuiltinValue builtin)
{
    switch (builtin)
    {
    case tint::core::BuiltinValue::kLocalInvocationIndex:
        return "local_invocation_index";
    case tint::core::BuiltinValue::kGlobalInvocationIndex:
        return "global_invocation_index";
    case tint::core::BuiltinValue::kWorkgroupIndex:
        return "workgroup_index";
    case tint::core::BuiltinValue::kSubgroupSize:
        return "subgroup_size";
    case tint::core::BuiltinValue::kSubgroupId:
        return "subgroup_id";
    case tint::core::BuiltinValue::kNumSubgroups:
        return "num_subgroups";
    default:
        return 0;
    }
}

int legalize_subgroup_id_builtins(tint::core::ir::Module& module,
                                  std::vector<SubgroupBuiltinSubstitution>& substitutions,
                                  bool& uses_linear_indexing_placeholder)
{
    // Tint recognizes these builtins but its IR-to-WGSL writer does not emit them yet.
    // Use an unused u32 compute builtin while writing and restore the WGSL attribute later.
    std::unordered_set<tint::core::BuiltinValue> used_builtins;
    for (size_t i = 0; i < module.functions.Length(); i++)
    {
        const tint::VectorRef<tint::core::ir::FunctionParam*> params = module.functions[i]->Params();
        for (size_t j = 0; j < params.Length(); j++)
        {
            if (params[j]->Builtin().has_value())
                used_builtins.insert(params[j]->Builtin().value());
        }
    }

    const tint::core::BuiltinValue placeholders[] = {
        tint::core::BuiltinValue::kSubgroupSize,
        tint::core::BuiltinValue::kLocalInvocationIndex,
        tint::core::BuiltinValue::kGlobalInvocationIndex,
        tint::core::BuiltinValue::kWorkgroupIndex};

    uses_linear_indexing_placeholder = false;
    for (size_t i = 0; i < module.functions.Length(); i++)
    {
        const tint::VectorRef<tint::core::ir::FunctionParam*> params = module.functions[i]->Params();
        for (size_t j = 0; j < params.Length(); j++)
        {
            const std::optional<tint::core::BuiltinValue> builtin = params[j]->Builtin();
            if (!builtin.has_value()
                || (builtin.value() != tint::core::BuiltinValue::kSubgroupId
                    && builtin.value() != tint::core::BuiltinValue::kNumSubgroups))
                continue;

            size_t placeholder_index = 0;
            for (; placeholder_index < sizeof(placeholders) / sizeof(placeholders[0]); placeholder_index++)
            {
                if (used_builtins.count(placeholders[placeholder_index]) == 0)
                    break;
            }
            if (placeholder_index == sizeof(placeholders) / sizeof(placeholders[0]))
            {
                log_error("Tint subgroup builtin legalization has no unused u32 compute builtin");
                return -1;
            }

            const tint::core::BuiltinValue placeholder = placeholders[placeholder_index];
            tint::core::IOAttributes attributes = params[j]->Attributes();
            attributes.builtin = placeholder;
            params[j]->SetAttributes(attributes);
            used_builtins.insert(placeholder);

            SubgroupBuiltinSubstitution substitution;
            substitution.placeholder = placeholder;
            substitution.builtin = builtin.value();
            substitutions.push_back(substitution);

            if (placeholder == tint::core::BuiltinValue::kGlobalInvocationIndex
                || placeholder == tint::core::BuiltinValue::kWorkgroupIndex)
                uses_linear_indexing_placeholder = true;
        }
    }

    return 0;
}

int restore_subgroup_id_builtins(std::string& wgsl,
                                 const std::vector<SubgroupBuiltinSubstitution>& substitutions)
{
    for (size_t i = 0; i < substitutions.size(); i++)
    {
        const char* placeholder_name = wgsl_builtin_name(substitutions[i].placeholder);
        const char* builtin_name = wgsl_builtin_name(substitutions[i].builtin);
        if (!placeholder_name || !builtin_name)
            return -1;

        const std::string placeholder = std::string("@builtin(") + placeholder_name + ")";
        const std::string builtin = std::string("@builtin(") + builtin_name + ")";
        const size_t position = wgsl.find(placeholder);
        if (position == std::string::npos || wgsl.find(placeholder, position + placeholder.size()) != std::string::npos)
        {
            log_error("Tint WGSL subgroup builtin placeholder %s is missing or duplicated", placeholder_name);
            return -1;
        }
        wgsl.replace(position, placeholder.size(), builtin);
    }

    return 0;
}

bool spirv_uses_subgroup(const std::vector<uint32_t>& spirv)
{
    for (size_t i = 5; i < spirv.size();)
    {
        const uint32_t word_count = spirv[i] >> 16;
        const spv::Op opcode = (spv::Op)(spirv[i] & 0xffff);
        if (word_count == 0 || i + word_count > spirv.size())
            return false;

        if (opcode >= spv::Op::OpGroupNonUniformElect
            && opcode <= spv::Op::OpGroupNonUniformQuadSwap)
            return true;

        if (opcode == spv::Op::OpDecorate && word_count >= 4
            && spirv[i + 2] == (uint32_t)spv::Decoration::BuiltIn)
        {
            const spv::BuiltIn builtin = (spv::BuiltIn)spirv[i + 3];
            if (builtin == spv::BuiltIn::SubgroupSize
                || builtin == spv::BuiltIn::NumSubgroups
                || builtin == spv::BuiltIn::SubgroupId
                || builtin == spv::BuiltIn::SubgroupLocalInvocationId)
                return true;
        }

        i += word_count;
    }

    return false;
}

} // namespace nikki_detail

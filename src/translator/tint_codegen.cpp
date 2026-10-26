// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "translator_stages.h"

namespace nikki_detail {

int translate_shader(const TranslationInput& input, TranslationResult& translated)
try
{
    translated = TranslationResult();
    if (input.entry_point.empty() || !input.translator)
        return -1;

    std::vector<uint32_t> spirv = input.spirv;
    if (normalize_spirv(spirv) != 0)
    {
        log_error("SPIR-V normalization failed");
        return -1;
    }
    VkSpecializationInfo specialization_info;
    specialization_info.mapEntryCount = (uint32_t)input.specialization.entries.size();
    specialization_info.pMapEntries = input.specialization.entries.empty() ? 0 : input.specialization.entries.data();
    specialization_info.dataSize = input.specialization.data.size();
    specialization_info.pData = input.specialization.data.empty() ? 0 : input.specialization.data.data();
    if (specialize_spirv(spirv, &specialization_info) != 0)
        return -1;
    if (legalize_storage_buffer_16bit_integer_spirv(spirv) != 0)
    {
        log_error("SPIR-V 16-bit integer storage legalization failed");
        return -1;
    }
    const bool uses_cooperative_matrix = spirv_uses_cooperative_matrix(spirv);
    if (uses_cooperative_matrix && !input.shader_profile.cooperative_matrix_enabled)
    {
        log_error("SPIR-V cooperative matrix operations require enabled VK_KHR_cooperative_matrix support");
        return -1;
    }
    translated.uses_subgroup = spirv_uses_subgroup(spirv) || uses_cooperative_matrix;
    if (translated.uses_subgroup && !input.shader_profile.subgroup_enabled)
    {
        log_error("SPIR-V subgroup operations are not supported by this WebGPU device");
        return -1;
    }
    translated.workgroup_reshape_safe = can_reshape_workgroup(spirv);
    size_t packed_integer_dot_count = 0;
    bool uses_integer_dot = false;
    if (legalize_packed_integer_dot_spirv(spirv, packed_integer_dot_count, &uses_integer_dot) != 0)
    {
        log_error("SPIR-V packed integer dot legalization failed");
        return -1;
    }
    if (uses_integer_dot && !input.shader_profile.packed_integer_dot_enabled)
    {
        log_error("SPIR-V packed integer dot product requires VK_KHR_shader_integer_dot_product");
        return -1;
    }
    CooperativeMatrixLegalization cooperative_matrix_info;
    if (uses_cooperative_matrix
        && legalize_cooperative_matrix_spirv(spirv, cooperative_matrix_info) != 0)
    {
        log_error("SPIR-V cooperative matrix legalization failed");
        return -1;
    }
    if (legalize_uniform_global_invocation_z(spirv) != 0)
    {
        log_error("SPIR-V uniform control-flow legalization failed");
        return -1;
    }
    if (legalize_non_finite_fp16_constants(spirv) != 0)
    {
        log_error("SPIR-V non-finite f16 constant legalization failed");
        return -1;
    }

    std::call_once(g_tint_initialize_once, []() {
        tint::Initialize();
    });

    tint::Result<tint::core::ir::Module> ir_result = tint::spirv::reader::ReadIR(spirv);
    if (ir_result != tint::Success)
    {
        log_error("Tint SPIR-V ReadIR failed: %s", ir_result.Failure().reason.c_str());
        return -1;
    }

    tint::core::ir::Module& module = ir_result.Get();
    if (strengthen_atomic_compare_exchange_ir(module) != 0)
    {
        log_error("SPIR-V strong atomic compare-exchange legalization failed");
        return -1;
    }
    tint::SubstituteOverridesConfig substitute_overrides_config;
    tint::Result<tint::SuccessType> result = tint::core::ir::transform::SubstituteOverrides(module, substitute_overrides_config);
    if (result != tint::Success)
    {
        log_error("Tint specialization substitution failed: %s", result.Failure().reason.c_str());
        return -1;
    }
    const uint32_t cooperative_matrix_subgroup_size = input.required_subgroup_size != 0
                                                          ? input.required_subgroup_size
                                                          : input.shader_profile.subgroup_default_size;
    if (restore_cooperative_matrix_ir(module, cooperative_matrix_info,
                                      cooperative_matrix_subgroup_size)
        != 0)
    {
        log_error("Tint cooperative matrix restoration failed");
        return -1;
    }
    if (restore_packed_integer_dot_ir(module, packed_integer_dot_count) != 0)
        return -1;
    result = tint::core::ir::transform::SingleEntryPoint(module, input.entry_point);
    if (result != tint::Success)
    {
        log_error("Tint single entry point failed: %s", result.Failure().reason.c_str());
        return -1;
    }

    result = tint::core::ir::transform::DeadCodeElimination(module);
    if (result != tint::Success)
    {
        log_error("Tint dead code elimination failed: %s", result.Failure().reason.c_str());
        return -1;
    }

    translated.requires_push_constants = uses_push_constants(module);

    std::vector<SubgroupBuiltinSubstitution> subgroup_builtin_substitutions;
    bool uses_linear_indexing_placeholder = false;
    if (legalize_subgroup_id_builtins(module, subgroup_builtin_substitutions, uses_linear_indexing_placeholder) != 0)
        return -1;

    if (translated.uses_subgroup && input.shader_profile.subgroup_size_control_supported)
    {
        translated.fixed_subgroup_size = input.required_subgroup_size != 0
                                             ? input.required_subgroup_size
                                             : input.shader_profile.subgroup_default_size;

        tint::core::ir::Function* compute_entry_point = 0;
        for (size_t i = 0; i < module.functions.Length(); i++)
        {
            if (module.functions[i]->IsCompute())
            {
                compute_entry_point = module.functions[i];
                break;
            }
        }
        if (!compute_entry_point || !compute_entry_point->WorkgroupSizeAsConst().has_value())
        {
            log_error("Tint subgroup compute entry point has no constant workgroup size");
            return -1;
        }

        const std::array<uint32_t, 3> workgroup_size = compute_entry_point->WorkgroupSizeAsConst().value();
        if (workgroup_size[0] % translated.fixed_subgroup_size != 0)
        {
            log_error("workgroup_size.x=%u is not a multiple of subgroup size=%u",
                      workgroup_size[0], translated.fixed_subgroup_size);
            return -1;
        }
    }

    std::unordered_map<uint64_t, ShaderBufferAccess> storage_access;
    std::unordered_map<uint64_t, std::vector<tint::core::ir::Instruction*> > storage_writes;
    collect_storage_access(module, storage_access, storage_writes);
    collect_exclusive_write_binding_pairs(storage_writes, translated.exclusive_write_binding_pairs);

    const uint32_t max_immediate_size = input.shader_profile.max_immediate_size;
    const bool immediate_supported = input.shader_profile.immediate_supported;
    const bool use_uniform = translated.requires_push_constants
                             && (!immediate_supported || input.push_constant_size > max_immediate_size);
    if (use_uniform)
    {
        tint::core::ir::transform::ChangeImmediateToUniformConfig config;
        config.immediate_binding_point = tint::BindingPoint{1, 0};
        result = tint::core::ir::transform::ChangeImmediateToUniform(module, config);
        if (result != tint::Success)
        {
            log_error("Tint immediate to uniform failed: %s", result.Failure().reason.c_str());
            return -1;
        }
    }

    tint::wgsl::writer::Options writer_options;
    writer_options.allowed_features = input.translator->wgsl_allowed_features;
    if (uses_linear_indexing_placeholder)
        writer_options.allowed_features.features.insert(tint::wgsl::LanguageFeature::kLinearIndexing);
    if (input.shader_profile.shader_f16_enabled)
        writer_options.allowed_features.extensions.insert(tint::wgsl::Extension::kF16);
    if (translated.uses_subgroup)
        writer_options.allowed_features.extensions.insert(tint::wgsl::Extension::kSubgroups);
    if (translated.fixed_subgroup_size != 0)
        writer_options.allowed_features.extensions.insert(tint::wgsl::Extension::kSubgroupSizeControl);
    if (uses_cooperative_matrix)
        writer_options.allowed_features.extensions.insert(tint::wgsl::Extension::kChromiumExperimentalSubgroupMatrix);
    tint::Result<tint::Program> program_result = tint::wgsl::writer::ProgramFromIR(module, writer_options);
    if (program_result != tint::Success)
    {
        log_error("Tint IR to program failed: %s", program_result.Failure().reason.c_str());
        return -1;
    }

    tint::Program program = program_result.Move();
    if (reflect_shader(program, input.entry_point.c_str(), storage_access, input, use_uniform, translated) != 0)
        return -1;

    tint::Result<tint::wgsl::writer::Output> wgsl_result = tint::wgsl::writer::Generate(program, writer_options);
    if (wgsl_result != tint::Success)
    {
        log_error("Tint WGSL generation failed: %s", wgsl_result.Failure().reason.c_str());
        return -1;
    }

    translated.wgsl = wgsl_result->wgsl;
    if (restore_cooperative_matrix_wgsl_types(translated.wgsl, cooperative_matrix_info) != 0)
        return -1;
    if (restore_subgroup_id_builtins(translated.wgsl, subgroup_builtin_substitutions) != 0)
        return -1;
    if (translated.fixed_subgroup_size != 0)
    {
        const char enable_subgroups[] = "enable subgroups;\n";
        size_t enable_subgroups_position = translated.wgsl.find(enable_subgroups);
        if (enable_subgroups_position == std::string::npos)
        {
            translated.wgsl.insert(0, enable_subgroups);
            enable_subgroups_position = 0;
        }
        translated.wgsl.insert(enable_subgroups_position + strlen(enable_subgroups),
                               "enable subgroup_size_control;\n");

        const size_t compute_attribute = translated.wgsl.find("@compute");
        if (compute_attribute == std::string::npos)
        {
            log_error("Tint WGSL subgroup entry point has no @compute attribute");
            return -1;
        }

        char subgroup_attribute[48];
        snprintf(subgroup_attribute, sizeof(subgroup_attribute), " @subgroup_size(%uu)", translated.fixed_subgroup_size);
        translated.wgsl.insert(compute_attribute + strlen("@compute"), subgroup_attribute);
    }
    return 0;
}
catch (const std::bad_alloc&)
{
    translated = TranslationResult();
    log_error("shader translation failed: out of host memory");
    return -2;
}
catch (const std::exception& exception)
{
    translated = TranslationResult();
    log_error("shader translation failed: %s", exception.what());
    return -1;
}
catch (...)
{
    translated = TranslationResult();
    log_error("shader translation failed with an unknown exception");
    return -1;
}

} // namespace nikki_detail

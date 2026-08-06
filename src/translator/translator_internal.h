// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_TRANSLATOR_INTERNAL_H
#define NIKKI_TRANSLATOR_INTERNAL_H

#include "../common.h"

#include <vulkan/vulkan.h>

#include "src/tint/lang/wgsl/allowed_features.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

namespace nikki_detail {

struct TranslatorInstanceState
{
    tint::wgsl::AllowedFeatures wgsl_allowed_features;
};

enum ShaderBufferAccess
{
    SHADER_BUFFER_ACCESS_READ,
    SHADER_BUFFER_ACCESS_READ_WRITE
};

struct SpecializationData
{
    std::vector<VkSpecializationMapEntry> entries;
    std::vector<unsigned char> data;
};

struct DescriptorBindingLayout
{
    uint32_t set;
    uint32_t binding;
    VkDescriptorType descriptor_type;
    uint32_t descriptor_count;
    VkShaderStageFlags stage_flags;
};

struct ShaderTranslationProfile
{
    bool shader_f16_enabled;
    bool subgroup_enabled;
    bool subgroup_size_control_supported;
    bool cooperative_matrix_enabled;
    bool packed_integer_dot_enabled;
    bool immediate_supported;
    uint32_t subgroup_default_size;
    uint32_t max_immediate_size;
    uint64_t max_uniform_buffer_binding_size;
};

struct TranslationInput
{
    std::vector<uint32_t> spirv;
    std::string entry_point;
    SpecializationData specialization;
    std::vector<DescriptorBindingLayout> descriptor_layout;
    uint32_t push_constant_size;
    ShaderTranslationProfile shader_profile;
    uint32_t required_subgroup_size;
    std::shared_ptr<TranslatorInstanceState> translator;
};

struct ResourceBindingABI
{
    uint32_t group;
    uint32_t binding;
    VkDescriptorType descriptor_type;
    ShaderBufferAccess access;
    ShaderBufferAccess layout_access;
    uint64_t min_binding_size;
    bool internal_uniform;
};

struct TranslationResult
{
    std::string wgsl;
    std::vector<ResourceBindingABI> bindings;
    std::unordered_set<uint64_t> exclusive_write_binding_pairs;
    uint32_t immediate_size;
    bool push_constant_uniform;
    uint32_t push_constant_data_size;
    uint32_t push_constant_uniform_size;
    bool requires_push_constants;
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;
    bool workgroup_reshape_safe;
    bool uses_subgroup;
    uint32_t fixed_subgroup_size;
};

int translate_shader(const TranslationInput& input, TranslationResult& translated);

bool cooperative_matrix_translation_probe();

} // namespace nikki_detail

#endif // NIKKI_TRANSLATOR_INTERNAL_H

// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_internal.h"
#include "backend/runtime.h"
#include "backend/shader_module.h"
#include "translator/translator_internal.h"

#include "spirv/unified1/spirv.hpp11"

namespace nikki_detail {

ComputePipeline::~ComputePipeline()
{
    if (pipeline)
        wgpuComputePipelineRelease(pipeline);
    if (pipeline_layout)
        wgpuPipelineLayoutRelease(pipeline_layout);
    if (immediate_bind_group_layout)
        wgpuBindGroupLayoutRelease(immediate_bind_group_layout);
    if (bind_group_layout)
        wgpuBindGroupLayoutRelease(bind_group_layout);
    if (shader_module)
        wgpuShaderModuleRelease(shader_module);
}

// descriptor layouts, shader modules and pipelines

static VkResult validate_descriptor_set_layout_create_info(
    Device* device_impl, const VkDescriptorSetLayoutCreateInfo* create_info,
    std::vector<DescriptorBinding>* bindings, bool* push_descriptor)
{
    if (!device_impl || !create_info
        || create_info->sType != VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (create_info->pNext)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    const VkDescriptorSetLayoutCreateFlags supported_flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    if ((create_info->flags & ~supported_flags) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->bindingCount != 0 && !create_info->pBindings)
        return VK_ERROR_INITIALIZATION_FAILED;

    const bool is_push_descriptor = (create_info->flags & VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR) != 0;
    if (is_push_descriptor
        && device_impl->enabled_extensions.find("VK_KHR_push_descriptor") == device_impl->enabled_extensions.end())
        return VK_ERROR_EXTENSION_NOT_PRESENT;

    std::vector<DescriptorBinding> validated_bindings;
    validated_bindings.reserve(create_info->bindingCount);

    std::unordered_set<uint32_t> seen_bindings;
    uint32_t storage_buffer_count = 0;
    uint32_t uniform_buffer_count = 0;
    for (uint32_t i = 0; i < create_info->bindingCount; i++)
    {
        const VkDescriptorSetLayoutBinding& source = create_info->pBindings[i];
        if ((source.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
             && source.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            || source.descriptorCount != 1
            || source.stageFlags != VK_SHADER_STAGE_COMPUTE_BIT
            || source.pImmutableSamplers
            || source.binding >= device_impl->limits.maxBindingsPerBindGroup
            || !seen_bindings.insert(source.binding).second)
            return VK_ERROR_FEATURE_NOT_PRESENT;

        if (source.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER)
            storage_buffer_count++;
        else
            uniform_buffer_count++;

        DescriptorBinding binding;
        binding.binding = source.binding;
        binding.descriptor_type = source.descriptorType;
        binding.descriptor_count = source.descriptorCount;
        binding.stage_flags = source.stageFlags;
        validated_bindings.push_back(binding);
    }

    if (storage_buffer_count > device_impl->limits.maxStorageBuffersPerShaderStage
        || uniform_buffer_count > device_impl->limits.maxUniformBuffersPerShaderStage
        || create_info->bindingCount > device_impl->limits.maxBindingsPerBindGroup)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    if (bindings)
        bindings->swap(validated_bindings);
    if (push_descriptor)
        *push_descriptor = is_push_descriptor;
    return VK_SUCCESS;
}

VkResult impl_create_descriptor_set_layout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* create_info,
                                           const VkAllocationCallbacks* allocator, VkDescriptorSetLayout* set_layout)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !set_layout)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<DescriptorSetLayout> set_layout_impl = std::make_shared<DescriptorSetLayout>();
    set_layout_impl->owner = device_impl;
    const VkResult result = validate_descriptor_set_layout_create_info(
        device_impl, create_info, &set_layout_impl->bindings, &set_layout_impl->push_descriptor);
    if (result != VK_SUCCESS)
        return result;

    *set_layout = make_handle<VkDescriptorSetLayout>(g_descriptor_set_layouts, set_layout_impl);
    return VK_SUCCESS;
}

void impl_get_descriptor_set_layout_support(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo* create_info,
    VkDescriptorSetLayoutSupport* support)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !support || support->sType != VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT)
        return;

    support->supported = VK_FALSE;
    if (device_failed(device_impl))
        return;

    support->supported = validate_descriptor_set_layout_create_info(
                             device_impl, create_info, 0, 0)
                                 == VK_SUCCESS
                             ? VK_TRUE
                             : VK_FALSE;
}

void impl_destroy_descriptor_set_layout(VkDevice device, VkDescriptorSetLayout set_layout,
                                        const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DescriptorSetLayout> set_layout_impl = get_handle(g_descriptor_set_layouts, set_layout);
    if (!device_impl || !set_layout_impl || set_layout_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    erase_handle(g_descriptor_set_layouts, set_layout);
}

VkResult impl_create_pipeline_layout(VkDevice device, const VkPipelineLayoutCreateInfo* create_info,
                                     const VkAllocationCallbacks* allocator, VkPipelineLayout* pipeline_layout)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO || !pipeline_layout)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0 || create_info->setLayoutCount > 1
        || (create_info->setLayoutCount != 0 && !create_info->pSetLayouts)
        || (create_info->pushConstantRangeCount != 0 && !create_info->pPushConstantRanges))
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<PipelineLayout> pipeline_layout_impl = std::make_shared<PipelineLayout>();
    pipeline_layout_impl->owner = device_impl;
    for (uint32_t i = 0; i < create_info->setLayoutCount; i++)
    {
        std::shared_ptr<DescriptorSetLayout> set_layout = get_handle(g_descriptor_set_layouts, create_info->pSetLayouts[i]);
        if (!set_layout || set_layout->owner != device_impl)
            return VK_ERROR_INITIALIZATION_FAILED;
        pipeline_layout_impl->set_layouts.push_back(set_layout);
    }

    for (uint32_t i = 0; i < create_info->pushConstantRangeCount; i++)
    {
        const VkPushConstantRange& source = create_info->pPushConstantRanges[i];
        uint64_t end = 0;
        if (source.size == 0 || source.offset % 4 != 0 || source.size % 4 != 0
            || source.stageFlags != VK_SHADER_STAGE_COMPUTE_BIT
            || !checked_add(source.offset, source.size, end)
            || end > std::min<uint64_t>(device_impl->limits.maxUniformBufferBindingSize, 4096))
            return VK_ERROR_FEATURE_NOT_PRESENT;

        for (size_t j = 0; j < pipeline_layout_impl->push_constant_ranges.size(); j++)
        {
            const PushConstantRange& other = pipeline_layout_impl->push_constant_ranges[j];
            const uint64_t other_end = (uint64_t)other.offset + other.size;
            if ((uint64_t)source.offset < other_end && (uint64_t)other.offset < end)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }

        PushConstantRange range;
        range.stage_flags = source.stageFlags;
        range.offset = source.offset;
        range.size = source.size;
        pipeline_layout_impl->push_constant_ranges.push_back(range);
        pipeline_layout_impl->push_constant_size = std::max<uint32_t>(pipeline_layout_impl->push_constant_size, (uint32_t)end);
    }

    *pipeline_layout = make_handle<VkPipelineLayout>(g_pipeline_layouts, pipeline_layout_impl);
    return VK_SUCCESS;
}

void impl_destroy_pipeline_layout(VkDevice device, VkPipelineLayout pipeline_layout,
                                  const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<PipelineLayout> pipeline_layout_impl = get_handle(g_pipeline_layouts, pipeline_layout);
    if (!device_impl || !pipeline_layout_impl || pipeline_layout_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    erase_handle(g_pipeline_layouts, pipeline_layout);
}

VkResult impl_create_shader_module(VkDevice device, const VkShaderModuleCreateInfo* create_info,
                                   const VkAllocationCallbacks* allocator, VkShaderModule* shader_module)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO || !shader_module)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0
        || !create_info->pCode || create_info->codeSize < 20 || create_info->codeSize % 4 != 0)
        return VK_ERROR_INVALID_SHADER_NV;

    std::shared_ptr<ShaderModule> shader_module_impl = std::make_shared<ShaderModule>();
    shader_module_impl->owner = device_impl;
    shader_module_impl->spirv.assign(create_info->pCode, create_info->pCode + create_info->codeSize / 4);
    if (shader_module_impl->spirv[0] != (uint32_t)spv::MagicNumber)
        return VK_ERROR_INVALID_SHADER_NV;

    *shader_module = make_handle<VkShaderModule>(g_shader_modules, shader_module_impl);
    return VK_SUCCESS;
}

void impl_destroy_shader_module(VkDevice device, VkShaderModule shader_module, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<ShaderModule> shader_module_impl = get_handle(g_shader_modules, shader_module);
    if (!device_impl || !shader_module_impl || shader_module_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    erase_handle(g_shader_modules, shader_module);
}

const DescriptorBinding* find_layout_binding(const DescriptorSetLayout& layout, uint32_t binding)
{
    for (size_t i = 0; i < layout.bindings.size(); i++)
    {
        if (layout.bindings[i].binding == binding)
            return &layout.bindings[i];
    }

    return 0;
}

static uint32_t power_of_two_divisor(uint32_t value)
{
    return value & (~value + 1);
}

static void reshape_llvmpipe_workgroup(const Device& device, TranslationResult& translated,
                                       ComputePipeline& pipeline)
{
    if (!translated.workgroup_reshape_safe || !device.physical_device
        || device.physical_device->backend_type != WGPUBackendType_Vulkan
        || device.physical_device->device_name.find("llvmpipe") == std::string::npos)
        return;

    const uint32_t old_size[3] = {
        translated.workgroup_size_x,
        translated.workgroup_size_y,
        translated.workgroup_size_z};
    const uint32_t new_size[3] = {
        power_of_two_divisor(old_size[0]),
        power_of_two_divisor(old_size[1]),
        power_of_two_divisor(old_size[2])};
    if (new_size[0] == 0 || new_size[1] == 0 || new_size[2] == 0
        || (new_size[0] == old_size[0] && new_size[1] == old_size[1] && new_size[2] == old_size[2]))
        return;

    char old_attribute[96];
    char new_attribute[96];
    snprintf(old_attribute, sizeof(old_attribute), "@workgroup_size(%uu, %uu, %uu)",
             old_size[0], old_size[1], old_size[2]);
    snprintf(new_attribute, sizeof(new_attribute), "@workgroup_size(%uu, %uu, %uu)",
             new_size[0], new_size[1], new_size[2]);
    const size_t position = translated.wgsl.find(old_attribute);
    if (position == std::string::npos)
        return;

    translated.wgsl.replace(position, strlen(old_attribute), new_attribute);
    pipeline.workgroup_size_x = new_size[0];
    pipeline.workgroup_size_y = new_size[1];
    pipeline.workgroup_size_z = new_size[2];
    pipeline.dispatch_scale_x = old_size[0] / new_size[0];
    pipeline.dispatch_scale_y = old_size[1] / new_size[1];
    pipeline.dispatch_scale_z = old_size[2] / new_size[2];
}

static VkResult make_translation_input(const ShaderModule& shader, const char* entry_point,
                                       const VkSpecializationInfo* specialization_info,
                                       const PipelineLayout& layout, const Device& device,
                                       uint32_t required_subgroup_size,
                                       TranslationInput& input)
{
    if (!entry_point || !entry_point[0] || !device.instance || !device.physical_device
        || !device.instance->translator)
        return VK_ERROR_INITIALIZATION_FAILED;

    input.spirv = shader.spirv;
    input.entry_point = entry_point;
    input.push_constant_size = layout.push_constant_size;
    input.required_subgroup_size = required_subgroup_size;
    input.translator = device.instance->translator;

    if (specialization_info)
    {
        if ((specialization_info->mapEntryCount != 0 && !specialization_info->pMapEntries)
            || (specialization_info->dataSize != 0 && !specialization_info->pData))
            return VK_ERROR_INITIALIZATION_FAILED;

        for (uint32_t i = 0; i < specialization_info->mapEntryCount; i++)
        {
            const VkSpecializationMapEntry& entry = specialization_info->pMapEntries[i];
            uint64_t end = 0;
            if (entry.size == 0 || !checked_add(entry.offset, entry.size, end)
                || end > specialization_info->dataSize)
                return VK_ERROR_INITIALIZATION_FAILED;
            input.specialization.entries.push_back(entry);
        }
        if (specialization_info->dataSize != 0)
        {
            const unsigned char* data = (const unsigned char*)specialization_info->pData;
            input.specialization.data.assign(data, data + specialization_info->dataSize);
        }
    }

    for (size_t set = 0; set < layout.set_layouts.size(); set++)
    {
        const DescriptorSetLayout& set_layout = *layout.set_layouts[set];
        for (size_t i = 0; i < set_layout.bindings.size(); i++)
        {
            DescriptorBindingLayout binding;
            binding.set = (uint32_t)set;
            binding.binding = set_layout.bindings[i].binding;
            binding.descriptor_type = set_layout.bindings[i].descriptor_type;
            binding.descriptor_count = set_layout.bindings[i].descriptor_count;
            binding.stage_flags = set_layout.bindings[i].stage_flags;
            input.descriptor_layout.push_back(binding);
        }
    }

    input.shader_profile.shader_f16_enabled = device.enabled_profile.shader.shader_f16;
    input.shader_profile.subgroup_enabled = device.enabled_profile.shader.subgroup;
    input.shader_profile.subgroup_size_control_supported = device.physical_device->profile.shader.subgroup.size_control_supported;
    input.shader_profile.cooperative_matrix_enabled = device.enabled_profile.shader.cooperative_matrix;
    input.shader_profile.packed_integer_dot_enabled = device.enabled_profile.shader.packed_integer_dot;
    input.shader_profile.immediate_supported = device.instance->immediate_supported;
    input.shader_profile.subgroup_default_size = device.physical_device->profile.shader.subgroup.default_size;
    input.shader_profile.max_immediate_size = device.limits.maxImmediateSize;
    input.shader_profile.max_uniform_buffer_binding_size = device.limits.maxUniformBufferBindingSize;
    return VK_SUCCESS;
}

static VkResult create_compute_pipeline(Device* device_impl, const VkComputePipelineCreateInfo& create_info,
                                        std::shared_ptr<ComputePipeline>& pipeline)
{
    if (create_info.sType != VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO
        || create_info.pNext
        || create_info.flags != 0
        || create_info.stage.sType != VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO
        || create_info.stage.stage != VK_SHADER_STAGE_COMPUTE_BIT
        || create_info.stage.flags != 0
        || create_info.basePipelineHandle != VK_NULL_HANDLE)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    uint32_t required_subgroup_size = 0;
    if (create_info.stage.pNext)
    {
        const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT* subgroup = (const VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT*)create_info.stage.pNext;
        if (subgroup->sType != VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT
            || subgroup->pNext
            || !device_impl->enabled_profile.shader.subgroup_size_control
            || device_impl->enabled_extensions.count("VK_EXT_subgroup_size_control") == 0
            || !is_power_of_two(subgroup->requiredSubgroupSize)
            || subgroup->requiredSubgroupSize < device_impl->physical_device->profile.shader.subgroup.min_size
            || subgroup->requiredSubgroupSize > device_impl->physical_device->profile.shader.subgroup.max_size)
            return VK_ERROR_FEATURE_NOT_PRESENT;

        required_subgroup_size = subgroup->requiredSubgroupSize;
    }

    std::shared_ptr<ShaderModule> shader = get_handle(g_shader_modules, create_info.stage.module);
    std::shared_ptr<PipelineLayout> layout = get_handle(g_pipeline_layouts, create_info.layout);
    if (!shader || !layout || shader->owner != device_impl || layout->owner != device_impl)
        return VK_ERROR_INITIALIZATION_FAILED;

    TranslationInput translation_input;
    const VkResult input_result = make_translation_input(
        *shader, create_info.stage.pName, create_info.stage.pSpecializationInfo,
        *layout, *device_impl, required_subgroup_size, translation_input);
    if (input_result != VK_SUCCESS)
        return input_result;

    TranslationResult translated;
    const int translate_result = translate_shader(translation_input, translated);
    if (translate_result == -2)
        return VK_ERROR_OUT_OF_HOST_MEMORY;
    if (translate_result != 0)
        return VK_ERROR_INVALID_SHADER_NV;

    wgpuDevicePushErrorScope(device_impl->device, WGPUErrorFilter_Validation);

    pipeline = std::make_shared<ComputePipeline>();
    pipeline->owner = device_impl;
    pipeline->vk_pipeline_layout = layout;
    pipeline->active_bindings.reserve(translated.bindings.size());
    for (size_t i = 0; i < translated.bindings.size(); i++)
    {
        const ResourceBindingABI& source = translated.bindings[i];
        ActiveBinding binding;
        binding.group = source.group;
        binding.binding = source.binding;
        binding.descriptor_type = source.descriptor_type;
        binding.access = source.access == SHADER_BUFFER_ACCESS_READ
                             ? BUFFER_ACCESS_READ
                             : BUFFER_ACCESS_READ_WRITE;
        binding.layout_access = source.layout_access == SHADER_BUFFER_ACCESS_READ
                                    ? BUFFER_ACCESS_READ
                                    : BUFFER_ACCESS_READ_WRITE;
        binding.min_binding_size = source.min_binding_size;
        binding.internal_uniform = source.internal_uniform;
        pipeline->active_bindings.push_back(binding);
    }
    pipeline->exclusive_write_binding_pairs = translated.exclusive_write_binding_pairs;
    pipeline->immediate_size = translated.immediate_size;
    pipeline->push_constant_uniform = translated.push_constant_uniform;
    pipeline->push_constant_data_size = translated.push_constant_data_size;
    pipeline->push_constant_uniform_size = translated.push_constant_uniform_size;
    pipeline->requires_push_constants = translated.requires_push_constants;
    pipeline->workgroup_size_x = translated.workgroup_size_x;
    pipeline->workgroup_size_y = translated.workgroup_size_y;
    pipeline->workgroup_size_z = translated.workgroup_size_z;
    // Mesa llvmpipe may crash its JIT for Tint-generated compute shaders with
    // odd workgroup dimensions. If workgroup boundaries are unobservable,
    // move odd factors into the dispatch count without changing invocation ids.
    reshape_llvmpipe_workgroup(*device_impl, translated, *pipeline);
    pipeline->shader_module = create_shader_module(device_impl, translated.wgsl);
    if (!pipeline->shader_module)
    {
        pop_error_scope(device_impl, "shader module");
        return VK_ERROR_INVALID_SHADER_NV;
    }

    std::vector<WGPUBindGroupLayoutEntry> group_entries;
    for (size_t i = 0; i < translated.bindings.size(); i++)
    {
        const ResourceBindingABI& binding = translated.bindings[i];
        if (binding.internal_uniform)
            continue;
        if (layout->set_layouts.empty())
        {
            log_error("pipeline active binding=%u has no Vulkan descriptor set layout", binding.binding);
            pop_error_scope(device_impl, "pipeline descriptor layout");
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        const DescriptorBinding* layout_binding = find_layout_binding(*layout->set_layouts[0], binding.binding);
        if (!layout_binding || layout_binding->descriptor_type != binding.descriptor_type)
        {
            log_error("pipeline active binding=%u is missing from Vulkan descriptor set layout", binding.binding);
            pop_error_scope(device_impl, "pipeline descriptor binding");
            return VK_ERROR_INITIALIZATION_FAILED;
        }

        WGPUBindGroupLayoutEntry entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entry.binding = binding.binding;
        entry.visibility = WGPUShaderStage_Compute;
        entry.buffer = WGPU_BUFFER_BINDING_LAYOUT_INIT;
        entry.buffer.type = binding.descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                ? WGPUBufferBindingType_Uniform
                            : binding.layout_access == SHADER_BUFFER_ACCESS_READ
                                ? WGPUBufferBindingType_ReadOnlyStorage
                                : WGPUBufferBindingType_Storage;
        entry.buffer.minBindingSize = binding.min_binding_size;
        group_entries.push_back(entry);
    }

    WGPUBindGroupLayoutDescriptor group_descriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    group_descriptor.entryCount = group_entries.size();
    group_descriptor.entries = group_entries.empty() ? 0 : group_entries.data();
    pipeline->bind_group_layout = wgpuDeviceCreateBindGroupLayout(device_impl->device, &group_descriptor);
    if (!pipeline->bind_group_layout)
    {
        pop_error_scope(device_impl, "bind group layout");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    WGPUBindGroupLayout layouts[2] = {pipeline->bind_group_layout, 0};
    uint32_t layout_count = 1;
    if (translated.push_constant_uniform)
    {
        WGPUBindGroupLayoutEntry uniform_entry = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        uniform_entry.binding = 0;
        uniform_entry.visibility = WGPUShaderStage_Compute;
        uniform_entry.buffer = WGPU_BUFFER_BINDING_LAYOUT_INIT;
        uniform_entry.buffer.type = WGPUBufferBindingType_Uniform;
        uniform_entry.buffer.hasDynamicOffset = WGPU_FALSE;
        uniform_entry.buffer.minBindingSize = translated.push_constant_uniform_size;

        WGPUBindGroupLayoutDescriptor uniform_descriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
        uniform_descriptor.entryCount = 1;
        uniform_descriptor.entries = &uniform_entry;
        pipeline->immediate_bind_group_layout = wgpuDeviceCreateBindGroupLayout(device_impl->device, &uniform_descriptor);
        if (!pipeline->immediate_bind_group_layout)
        {
            pop_error_scope(device_impl, "push uniform bind group layout");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        layouts[1] = pipeline->immediate_bind_group_layout;
        layout_count = 2;
    }

    WGPUPipelineLayoutDescriptor pipeline_layout_descriptor = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_descriptor.bindGroupLayoutCount = layout_count;
    pipeline_layout_descriptor.bindGroupLayouts = layouts;
    pipeline_layout_descriptor.immediateSize = translated.immediate_size;
    pipeline->pipeline_layout = wgpuDeviceCreatePipelineLayout(device_impl->device, &pipeline_layout_descriptor);
    if (!pipeline->pipeline_layout)
    {
        pop_error_scope(device_impl, "pipeline layout");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    WGPUComputePipelineDescriptor pipeline_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipeline_descriptor.layout = pipeline->pipeline_layout;
    pipeline_descriptor.compute.module = pipeline->shader_module;
    pipeline_descriptor.compute.entryPoint.data = create_info.stage.pName;
    pipeline_descriptor.compute.entryPoint.length = strlen(create_info.stage.pName);
    pipeline->pipeline = wgpuDeviceCreateComputePipeline(device_impl->device, &pipeline_descriptor);
    if (pop_error_scope(device_impl, "compute pipeline") != 0 || !pipeline->pipeline)
        return VK_ERROR_INITIALIZATION_FAILED;

    // Dispatch only needs the compute pipeline and the bind group layouts.
    // Releasing construction-only objects here avoids retaining one shader
    // module and one pipeline layout for every client pipeline.
    wgpuPipelineLayoutRelease(pipeline->pipeline_layout);
    pipeline->pipeline_layout = 0;
    wgpuShaderModuleRelease(pipeline->shader_module);
    pipeline->shader_module = 0;

    return VK_SUCCESS;
}

VkResult impl_create_compute_pipelines(VkDevice device, VkPipelineCache pipeline_cache, uint32_t create_info_count,
                                       const VkComputePipelineCreateInfo* create_infos,
                                       const VkAllocationCallbacks* allocator, VkPipeline* pipelines)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || create_info_count == 0 || !create_infos || !pipelines)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (pipeline_cache != VK_NULL_HANDLE)
    {
        std::shared_ptr<PipelineCache> cache = get_handle(g_pipeline_caches, pipeline_cache);
        if (!cache || cache->owner != device_impl)
            return VK_ERROR_INITIALIZATION_FAILED;
    }

    for (uint32_t i = 0; i < create_info_count; i++)
        pipelines[i] = VK_NULL_HANDLE;

    for (uint32_t i = 0; i < create_info_count; i++)
    {
        std::shared_ptr<ComputePipeline> pipeline;
        VkResult result = create_compute_pipeline(device_impl, create_infos[i], pipeline);
        if (result != VK_SUCCESS)
        {
            std::shared_ptr<ShaderModule> shader = get_handle(g_shader_modules, create_infos[i].stage.module);
            std::shared_ptr<PipelineLayout> layout = get_handle(g_pipeline_layouts, create_infos[i].layout);
            log_error("device %llu vkCreateComputePipelines index=%u shader=%llu layout=%llu entry=%s failed result=%d",
                      (unsigned long long)device_impl->id, i,
                      shader ? (unsigned long long)shader->id : 0,
                      layout ? (unsigned long long)layout->id : 0,
                      create_infos[i].stage.pName ? create_infos[i].stage.pName : "<null>",
                      (int)result);
            return result;
        }

        pipelines[i] = make_handle<VkPipeline>(g_compute_pipelines, pipeline);
    }

    return VK_SUCCESS;
}

void impl_destroy_pipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<ComputePipeline> pipeline_impl = get_handle(g_compute_pipelines, pipeline);
    if (!device_impl || !pipeline_impl || pipeline_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    if (pipeline_in_flight(device_impl, pipeline_impl.get()))
    {
        log_error("device %llu pipeline %llu destroyed while in flight",
                  (unsigned long long)device_impl->id, (unsigned long long)pipeline_impl->id);
        return;
    }

    pipeline_impl->live = false;
    erase_handle(g_compute_pipelines, pipeline);
}

VkResult impl_create_descriptor_update_template(VkDevice device,
                                                const VkDescriptorUpdateTemplateCreateInfo* create_info,
                                                const VkAllocationCallbacks* allocator,
                                                VkDescriptorUpdateTemplate* descriptor_update_template)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || !descriptor_update_template
        || create_info->sType != VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0
        || (create_info->descriptorUpdateEntryCount != 0 && !create_info->pDescriptorUpdateEntries))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (create_info->templateType != VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR
        || create_info->pipelineBindPoint != VK_PIPELINE_BIND_POINT_COMPUTE
        || create_info->set != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<PipelineLayout> pipeline_layout = get_handle(g_pipeline_layouts, create_info->pipelineLayout);
    std::shared_ptr<DescriptorSetLayout> set_layout = get_handle(g_descriptor_set_layouts, create_info->descriptorSetLayout);
    if (!pipeline_layout || !set_layout
        || pipeline_layout->owner != device_impl || set_layout->owner != device_impl
        || !set_layout->push_descriptor || pipeline_layout->set_layouts.empty()
        || pipeline_layout->set_layouts[0].get() != set_layout.get())
        return VK_ERROR_INITIALIZATION_FAILED;

    std::shared_ptr<DescriptorUpdateTemplate> descriptor_update_template_impl = std::make_shared<DescriptorUpdateTemplate>();
    descriptor_update_template_impl->owner = device_impl;
    descriptor_update_template_impl->template_type = create_info->templateType;
    descriptor_update_template_impl->bind_point = create_info->pipelineBindPoint;
    descriptor_update_template_impl->set = create_info->set;
    descriptor_update_template_impl->set_layout = set_layout;
    descriptor_update_template_impl->pipeline_layout = pipeline_layout;

    std::unordered_set<uint32_t> updated_bindings;
    for (uint32_t i = 0; i < create_info->descriptorUpdateEntryCount; i++)
    {
        const VkDescriptorUpdateTemplateEntry& source = create_info->pDescriptorUpdateEntries[i];
        const DescriptorBinding* layout_binding = find_layout_binding(*set_layout, source.dstBinding);
        if ((source.descriptorType != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
             && source.descriptorType != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
            || source.descriptorCount != 1
            || source.dstArrayElement != 0 || !layout_binding
            || layout_binding->descriptor_type != source.descriptorType
            || !updated_bindings.insert(source.dstBinding).second)
            return VK_ERROR_FEATURE_NOT_PRESENT;

        DescriptorTemplateEntry entry;
        entry.binding = source.dstBinding;
        entry.array_element = source.dstArrayElement;
        entry.descriptor_count = source.descriptorCount;
        entry.descriptor_type = source.descriptorType;
        entry.offset = source.offset;
        entry.stride = source.stride;
        descriptor_update_template_impl->entries.push_back(entry);
    }

    *descriptor_update_template = make_handle<VkDescriptorUpdateTemplate>(g_descriptor_update_templates, descriptor_update_template_impl);
    return VK_SUCCESS;
}

void impl_destroy_descriptor_update_template(VkDevice device,
                                             VkDescriptorUpdateTemplate descriptor_update_template,
                                             const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DescriptorUpdateTemplate> descriptor_update_template_impl = get_handle(g_descriptor_update_templates, descriptor_update_template);
    if (!device_impl || !descriptor_update_template_impl || descriptor_update_template_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    erase_handle(g_descriptor_update_templates, descriptor_update_template);
}

} // namespace nikki_detail

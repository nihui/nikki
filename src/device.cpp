// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_internal.h"
#include "backend/device.h"
#include "backend/runtime.h"

namespace nikki_detail {

static VkResult validate_device_create_info(const PhysicalDevice* physical_device,
                                            const VkDeviceCreateInfo* create_info,
                                            const VkAllocationCallbacks* allocator)
{
    if (!physical_device || !create_info || create_info->sType != VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (allocator || create_info->flags != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->enabledLayerCount != 0)
        return VK_ERROR_LAYER_NOT_PRESENT;
    if ((create_info->enabledExtensionCount != 0 && !create_info->ppEnabledExtensionNames)
        || (create_info->enabledLayerCount != 0 && !create_info->ppEnabledLayerNames))
        return VK_ERROR_INITIALIZATION_FAILED;

    if (create_info->pEnabledFeatures)
    {
        const VkBool32* features = (const VkBool32*)create_info->pEnabledFeatures;
        for (size_t i = 0; i < sizeof(VkPhysicalDeviceFeatures) / sizeof(VkBool32); i++)
        {
            if (features[i] != VK_FALSE)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }
    }

    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
    {
        if (!device_extension_supported(physical_device, create_info->ppEnabledExtensionNames[i]))
            return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    bool subgroup_size_control_extension_enabled = false;
    bool subgroup_extended_types_extension_enabled = false;
    bool cooperative_matrix_extension_enabled = false;
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
    {
        if (strcmp(create_info->ppEnabledExtensionNames[i], VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME) == 0)
            subgroup_size_control_extension_enabled = true;
        if (strcmp(create_info->ppEnabledExtensionNames[i], VK_KHR_SHADER_SUBGROUP_EXTENDED_TYPES_EXTENSION_NAME) == 0)
            subgroup_extended_types_extension_enabled = true;
        if (strcmp(create_info->ppEnabledExtensionNames[i], VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME) == 0)
            cooperative_matrix_extension_enabled = true;
    }

    struct StructureHeader
    {
        VkStructureType sType;
        const void* pNext;
    };
    const StructureHeader* next = (const StructureHeader*)create_info->pNext;
    while (next)
    {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR)
        {
            const VkPhysicalDevice16BitStorageFeaturesKHR* storage = (const VkPhysicalDevice16BitStorageFeaturesKHR*)next;
            if ((storage->storageBuffer16BitAccess == VK_TRUE && !physical_device->profile.shader.shader_f16)
                || storage->uniformAndStorageBuffer16BitAccess == VK_TRUE
                || storage->storagePushConstant16 == VK_TRUE
                || storage->storageInputOutput16 == VK_TRUE)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR)
        {
            const VkPhysicalDeviceFloat16Int8FeaturesKHR* arithmetic = (const VkPhysicalDeviceFloat16Int8FeaturesKHR*)next;
            if ((arithmetic->shaderFloat16 == VK_TRUE && !physical_device->profile.shader.shader_f16)
                || arithmetic->shaderInt8 == VK_TRUE)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR)
        {
            const VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR* dot_product = (const VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR*)next;
            if (dot_product->shaderIntegerDotProduct == VK_TRUE
                && !physical_device->profile.shader.packed_integer_dot)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT)
        {
            const VkPhysicalDeviceSubgroupSizeControlFeaturesEXT* subgroup = (const VkPhysicalDeviceSubgroupSizeControlFeaturesEXT*)next;
            if ((subgroup->subgroupSizeControl == VK_TRUE
                 && (!physical_device->profile.shader.subgroup.size_control_supported || !subgroup_size_control_extension_enabled))
                || subgroup->computeFullSubgroups == VK_TRUE)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR)
        {
            const VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR* subgroup = (const VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR*)next;
            if (subgroup->shaderSubgroupExtendedTypes == VK_TRUE
                && (!subgroup_extended_types_extension_enabled
                    || !physical_device->profile.shader.shader_f16
                    || !physical_device->profile.shader.subgroup.operations_supported))
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR)
        {
            const VkPhysicalDeviceCooperativeMatrixFeaturesKHR* cooperative_matrix = (const VkPhysicalDeviceCooperativeMatrixFeaturesKHR*)next;
            if ((cooperative_matrix->cooperativeMatrix == VK_TRUE
                 && (!cooperative_matrix_extension_enabled || !physical_device->profile.shader.cooperative_matrix.supported))
                || cooperative_matrix->cooperativeMatrixRobustBufferAccess == VK_TRUE)
                return VK_ERROR_FEATURE_NOT_PRESENT;
        }
        else
        {
            return VK_ERROR_FEATURE_NOT_PRESENT;
        }

        next = (const StructureHeader*)next->pNext;
    }

    if (create_info->queueCreateInfoCount != 1 || !create_info->pQueueCreateInfos
        || create_info->pQueueCreateInfos[0].queueFamilyIndex != 0
        || create_info->pQueueCreateInfos[0].queueCount != 1
        || create_info->pQueueCreateInfos[0].sType != VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO
        || create_info->pQueueCreateInfos[0].pNext
        || create_info->pQueueCreateInfos[0].flags != 0
        || !create_info->pQueueCreateInfos[0].pQueuePriorities
        || create_info->pQueueCreateInfos[0].pQueuePriorities[0] < 0.f
        || create_info->pQueueCreateInfos[0].pQueuePriorities[0] > 1.f)
        return VK_ERROR_INITIALIZATION_FAILED;

    return VK_SUCCESS;
}

VkResult impl_create_device(VkPhysicalDevice physical_device, const VkDeviceCreateInfo* create_info,
                            const VkAllocationCallbacks* allocator, VkDevice* device)
{
    PhysicalDevice* physical_device_impl = unwrap(physical_device);
    if (!device)
        return VK_ERROR_INITIALIZATION_FAILED;
    const VkResult validate_result = validate_device_create_info(physical_device_impl, create_info, allocator);
    if (validate_result != VK_SUCCESS)
        return validate_result;

    std::shared_ptr<Device> device_impl = std::make_shared<Device>();
    device_impl->instance = physical_device_impl->owner->shared_from_this();
    device_impl->physical_device = physical_device->impl;
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
        device_impl->enabled_extensions.insert(create_info->ppEnabledExtensionNames[i]);

    WGPULimits required_limits = WGPU_LIMITS_INIT;
    required_limits.maxBufferSize = physical_device_impl->limits.maxBufferSize;
    required_limits.maxStorageBufferBindingSize = physical_device_impl->limits.maxStorageBufferBindingSize;
    required_limits.maxStorageBuffersPerShaderStage = physical_device_impl->limits.maxStorageBuffersPerShaderStage;
    required_limits.minStorageBufferOffsetAlignment = physical_device_impl->limits.minStorageBufferOffsetAlignment;
    required_limits.maxUniformBufferBindingSize = physical_device_impl->limits.maxUniformBufferBindingSize;
    required_limits.maxUniformBuffersPerShaderStage = physical_device_impl->limits.maxUniformBuffersPerShaderStage;
    required_limits.minUniformBufferOffsetAlignment = physical_device_impl->limits.minUniformBufferOffsetAlignment;
    required_limits.maxBindGroups = 2;
    required_limits.maxBindingsPerBindGroup = physical_device_impl->limits.maxBindingsPerBindGroup;
    required_limits.maxComputeWorkgroupStorageSize = physical_device_impl->limits.maxComputeWorkgroupStorageSize;
    required_limits.maxComputeInvocationsPerWorkgroup = physical_device_impl->limits.maxComputeInvocationsPerWorkgroup;
    required_limits.maxComputeWorkgroupSizeX = physical_device_impl->limits.maxComputeWorkgroupSizeX;
    required_limits.maxComputeWorkgroupSizeY = physical_device_impl->limits.maxComputeWorkgroupSizeY;
    required_limits.maxComputeWorkgroupSizeZ = physical_device_impl->limits.maxComputeWorkgroupSizeZ;
    required_limits.maxComputeWorkgroupsPerDimension = physical_device_impl->limits.maxComputeWorkgroupsPerDimension;
    if (physical_device_impl->owner->immediate_supported)
        required_limits.maxImmediateSize = std::min(64u, physical_device_impl->limits.maxImmediateSize);

    WGPUDeviceDescriptor descriptor = WGPU_DEVICE_DESCRIPTOR_INIT;
    bool require_shader_f16 = false;
    bool enable_subgroup_size_control = false;
    bool enable_subgroup_extended_types = false;
    bool enable_cooperative_matrix = false;
    struct StructureHeader
    {
        VkStructureType sType;
        const void* pNext;
    };
    const StructureHeader* next = (const StructureHeader*)create_info->pNext;
    while (next)
    {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR)
        {
            const VkPhysicalDevice16BitStorageFeaturesKHR* storage = (const VkPhysicalDevice16BitStorageFeaturesKHR*)next;
            if (storage->storageBuffer16BitAccess == VK_TRUE)
                require_shader_f16 = true;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR)
        {
            const VkPhysicalDeviceFloat16Int8FeaturesKHR* arithmetic = (const VkPhysicalDeviceFloat16Int8FeaturesKHR*)next;
            if (arithmetic->shaderFloat16 == VK_TRUE)
                require_shader_f16 = true;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT)
        {
            const VkPhysicalDeviceSubgroupSizeControlFeaturesEXT* subgroup = (const VkPhysicalDeviceSubgroupSizeControlFeaturesEXT*)next;
            if (subgroup->subgroupSizeControl == VK_TRUE)
                enable_subgroup_size_control = true;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR)
        {
            const VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR* subgroup = (const VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR*)next;
            if (subgroup->shaderSubgroupExtendedTypes == VK_TRUE)
                enable_subgroup_extended_types = true;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR)
        {
            const VkPhysicalDeviceCooperativeMatrixFeaturesKHR* cooperative_matrix = (const VkPhysicalDeviceCooperativeMatrixFeaturesKHR*)next;
            if (cooperative_matrix->cooperativeMatrix == VK_TRUE)
                enable_cooperative_matrix = true;
        }

        next = (const StructureHeader*)next->pNext;
    }

    std::vector<WGPUFeatureName> required_features;
    if (require_shader_f16)
        required_features.push_back(WGPUFeatureName_ShaderF16);
    if (physical_device_impl->profile.shader.subgroup.operations_supported)
        required_features.push_back(WGPUFeatureName_Subgroups);
    if (physical_device_impl->profile.shader.subgroup.size_control_supported)
        required_features.push_back(WGPUFeatureName_SubgroupSizeControl);
    if (physical_device_impl->profile.timestamp_query)
        required_features.push_back(WGPUFeatureName_TimestampQuery);
    if (enable_cooperative_matrix)
        required_features.push_back(cooperative_matrix_feature_name());
    descriptor.requiredFeatureCount = required_features.size();
    descriptor.requiredFeatures = required_features.empty() ? 0 : required_features.data();
    device_impl->enabled_profile.shader.shader_f16 = require_shader_f16;
    device_impl->enabled_profile.shader.packed_integer_dot = device_impl->enabled_extensions.count(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME) != 0;
    device_impl->enabled_profile.shader.subgroup = physical_device_impl->profile.shader.subgroup.operations_supported;
    device_impl->enabled_profile.shader.subgroup_size_control = enable_subgroup_size_control;
    device_impl->enabled_profile.shader.subgroup_extended_types = enable_subgroup_extended_types;
    device_impl->enabled_profile.timestamp_query = physical_device_impl->profile.timestamp_query;
    device_impl->enabled_profile.shader.cooperative_matrix = enable_cooperative_matrix;
    descriptor.requiredLimits = &required_limits;
    const VkResult backend_result = request_backend_device(device_impl, physical_device_impl, descriptor);
    if (backend_result != VK_SUCCESS)
        return backend_result;

    VkDevice_T* handle = new VkDevice_T;
    handle->impl = device_impl;
    VkQueue_T* queue_handle = new VkQueue_T;
    queue_handle->device = device_impl.get();
    device_impl->queue_handle = queue_handle;
    *device = handle;
    return VK_SUCCESS;
}

void impl_destroy_device(VkDevice device, const VkAllocationCallbacks* allocator)
{
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    Device* device_impl = unwrap(device);
    if (device_impl)
    {
        const VkResult result = wait_device_submissions(device_impl);
        if (result != VK_SUCCESS)
            log_error("device %llu destroy wait failed result=%d", (unsigned long long)device_impl->id, (int)result);
        release_device_objects(device_impl);
    }

    delete device;
}

void impl_get_device_queue(VkDevice device, uint32_t queue_family_index, uint32_t queue_index, VkQueue* queue)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !queue)
        return;

    *queue = queue_family_index == 0 && queue_index == 0 ? device_impl->queue_handle : VK_NULL_HANDLE;
}

} // namespace nikki_detail

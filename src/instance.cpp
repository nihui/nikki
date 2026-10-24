// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_internal.h"
#include "nikki/version.h"
#include "backend/device.h"
#include "translator/translator_internal.h"

namespace nikki_detail {

static VkResult enumerate_properties(uint32_t available_count, const VkExtensionProperties* available,
                                     uint32_t* property_count, VkExtensionProperties* properties)
{
    if (!property_count)
        return VK_ERROR_INITIALIZATION_FAILED;

    if (!properties)
    {
        *property_count = available_count;
        return VK_SUCCESS;
    }

    const uint32_t requested_count = *property_count;
    const uint32_t copy_count = std::min(requested_count, available_count);
    if (copy_count != 0)
        memcpy(properties, available, copy_count * sizeof(VkExtensionProperties));
    *property_count = copy_count;
    return copy_count < available_count ? VK_INCOMPLETE : VK_SUCCESS;
}

static const VkExtensionProperties g_instance_extensions[] = {
    {"VK_KHR_get_physical_device_properties2", 2}};

static bool instance_extension_supported(const char* name)
{
    if (!name)
        return false;

    for (size_t i = 0; i < sizeof(g_instance_extensions) / sizeof(g_instance_extensions[0]); i++)
    {
        if (strcmp(name, g_instance_extensions[i].extensionName) == 0)
            return true;
    }

    return false;
}

VkResult impl_enumerate_instance_extension_properties(const char* layer_name, uint32_t* property_count, VkExtensionProperties* properties)
{
    if (layer_name)
        return VK_ERROR_LAYER_NOT_PRESENT;

    return enumerate_properties(sizeof(g_instance_extensions) / sizeof(g_instance_extensions[0]),
                                g_instance_extensions, property_count, properties);
}

VkResult impl_enumerate_instance_layer_properties(uint32_t* property_count, VkLayerProperties*)
{
    if (!property_count)
        return VK_ERROR_INITIALIZATION_FAILED;

    *property_count = 0;
    return VK_SUCCESS;
}

VkResult impl_enumerate_instance_version(uint32_t* api_version)
{
    if (!api_version)
        return VK_ERROR_INITIALIZATION_FAILED;

    *api_version = VK_MAKE_VERSION(1, 1, 0);
    return VK_SUCCESS;
}

VkResult impl_create_instance(const VkInstanceCreateInfo* create_info, const VkAllocationCallbacks* allocator, VkInstance* instance)
{
    static std::once_flag non_conformant_warning_once;
    std::call_once(non_conformant_warning_once, []() {
        fprintf(stderr, "WARNING: nikki is not a conformant Vulkan implementation, testing use only.\n");
    });

    if (!create_info || create_info->sType != VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO || !instance)
        return VK_ERROR_INITIALIZATION_FAILED;
    *instance = VK_NULL_HANDLE;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->enabledLayerCount != 0)
        return VK_ERROR_LAYER_NOT_PRESENT;
    if (create_info->enabledExtensionCount != 0 && !create_info->ppEnabledExtensionNames)
        return VK_ERROR_INITIALIZATION_FAILED;
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
    {
        if (!instance_extension_supported(create_info->ppEnabledExtensionNames[i]))
            return VK_ERROR_EXTENSION_NOT_PRESENT;
    }

    if (create_info->pApplicationInfo)
    {
        const VkApplicationInfo& application_info = *create_info->pApplicationInfo;
        if (application_info.sType != VK_STRUCTURE_TYPE_APPLICATION_INFO || application_info.pNext)
            return VK_ERROR_INITIALIZATION_FAILED;
        const uint32_t api_major = VK_VERSION_MAJOR(application_info.apiVersion);
        const uint32_t api_minor = VK_VERSION_MINOR(application_info.apiVersion);
        if (api_major > 1 || (api_major == 1 && api_minor > 1))
            return VK_ERROR_INCOMPATIBLE_DRIVER;
    }

    WGPUInstanceFeatureName instance_feature = WGPUInstanceFeatureName_TimedWaitAny;
    WGPUInstanceLimits instance_limits = WGPU_INSTANCE_LIMITS_INIT;
    instance_limits.timedWaitAnyMaxCount = 1;
    WGPUInstanceDescriptor descriptor = WGPU_INSTANCE_DESCRIPTOR_INIT;
    descriptor.requiredFeatureCount = 1;
    descriptor.requiredFeatures = &instance_feature;
    descriptor.requiredLimits = &instance_limits;
    std::shared_ptr<Instance> instance_impl = std::make_shared<Instance>();
    instance_impl->translator = std::make_shared<TranslatorInstanceState>();
    instance_impl->api_version = create_info->pApplicationInfo && create_info->pApplicationInfo->apiVersion
                                     ? create_info->pApplicationInfo->apiVersion
                                     : VK_API_VERSION_1_0;
    for (uint32_t i = 0; i < create_info->enabledExtensionCount; i++)
        instance_impl->enabled_extensions.insert(create_info->ppEnabledExtensionNames[i]);
    instance_impl->instance = wgpuCreateInstance(&descriptor);
    if (!instance_impl->instance)
        return VK_ERROR_INITIALIZATION_FAILED;

    const auto allow_wgsl_feature = [&](WGPUWGSLLanguageFeatureName webgpu_feature,
                                        tint::wgsl::LanguageFeature tint_feature) {
        if (wgpuInstanceHasWGSLLanguageFeature(instance_impl->instance, webgpu_feature) == WGPU_TRUE)
            instance_impl->translator->wgsl_allowed_features.features.insert(tint_feature);
    };
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_ReadonlyAndReadwriteStorageTextures,
                       tint::wgsl::LanguageFeature::kReadonlyAndReadwriteStorageTextures);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_Packed4x8IntegerDotProduct,
                       tint::wgsl::LanguageFeature::kPacked4X8IntegerDotProduct);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_UnrestrictedPointerParameters,
                       tint::wgsl::LanguageFeature::kUnrestrictedPointerParameters);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_PointerCompositeAccess,
                       tint::wgsl::LanguageFeature::kPointerCompositeAccess);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_UniformBufferStandardLayout,
                       tint::wgsl::LanguageFeature::kUniformBufferStandardLayout);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_SubgroupId,
                       tint::wgsl::LanguageFeature::kSubgroupId);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_TextureAndSamplerLet,
                       tint::wgsl::LanguageFeature::kTextureAndSamplerLet);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_SubgroupUniformity,
                       tint::wgsl::LanguageFeature::kSubgroupUniformity);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_TextureFormatsTier1,
                       tint::wgsl::LanguageFeature::kTextureFormatsTier1);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_LinearIndexing,
                       tint::wgsl::LanguageFeature::kLinearIndexing);
    allow_wgsl_feature(WGPUWGSLLanguageFeatureName_ImmediateAddressSpace,
                       tint::wgsl::LanguageFeature::kImmediateAddressSpace);

    instance_impl->packed_4x8_integer_dot_product_supported = instance_impl->translator->wgsl_allowed_features.features.count(
                                                                  tint::wgsl::LanguageFeature::kPacked4X8IntegerDotProduct)
                                                              != 0;
    instance_impl->immediate_supported = instance_impl->translator->wgsl_allowed_features.features.count(
                                             tint::wgsl::LanguageFeature::kImmediateAddressSpace)
                                         != 0;
    instance_impl->cooperative_matrix_translation_supported = cooperative_matrix_translation_probe();

    VkInstance_T* handle = new VkInstance_T;
    handle->impl = instance_impl;
    *instance = handle;
    return VK_SUCCESS;
}

void impl_destroy_instance(VkInstance instance, const VkAllocationCallbacks* allocator)
{
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    delete instance;
}

VkResult impl_enumerate_physical_devices(VkInstance instance, uint32_t* physical_device_count, VkPhysicalDevice* physical_devices)
{
    Instance* instance_impl = unwrap(instance);
    if (!instance_impl || !physical_device_count)
        return VK_ERROR_INITIALIZATION_FAILED;

    VkResult result = ensure_physical_devices(instance_impl);
    if (result != VK_SUCCESS)
        return result;

    const uint32_t available_count = (uint32_t)instance_impl->physical_devices.size();
    if (!physical_devices)
    {
        *physical_device_count = available_count;
        return VK_SUCCESS;
    }

    const uint32_t write_count = std::min(*physical_device_count, available_count);
    for (uint32_t i = 0; i < write_count; i++)
        physical_devices[i] = instance_impl->physical_devices[i];

    *physical_device_count = write_count;
    return write_count < available_count ? VK_INCOMPLETE : VK_SUCCESS;
}

void impl_get_physical_device_features(VkPhysicalDevice physical_device, VkPhysicalDeviceFeatures* features)
{
    if (!unwrap(physical_device) || !features)
        return;

    memset(features, 0, sizeof(VkPhysicalDeviceFeatures));
}

static VkPhysicalDeviceType physical_device_type(WGPUAdapterType adapter_type)
{
    if (adapter_type == WGPUAdapterType_DiscreteGPU)
        return VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
    if (adapter_type == WGPUAdapterType_IntegratedGPU)
        return VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
    if (adapter_type == WGPUAdapterType_CPU)
        return VK_PHYSICAL_DEVICE_TYPE_CPU;

    return VK_PHYSICAL_DEVICE_TYPE_OTHER;
}

void impl_get_physical_device_properties(VkPhysicalDevice physical_device, VkPhysicalDeviceProperties* properties)
{
    PhysicalDevice* physical_device_impl = unwrap(physical_device);
    if (!physical_device_impl || !properties)
        return;

    memset(properties, 0, sizeof(VkPhysicalDeviceProperties));
    properties->apiVersion = VK_MAKE_VERSION(1, 1, 0);
    properties->driverVersion = VK_MAKE_VERSION(NIKKI_DRIVER_VERSION_MAJOR,
        NIKKI_DRIVER_VERSION_MINOR, NIKKI_DRIVER_VERSION_PATCH);
    properties->vendorID = physical_device_impl->vendor_id;
    properties->deviceID = physical_device_impl->device_id;
    properties->deviceType = physical_device_type(physical_device_impl->adapter_type);
    const std::string device_name = "nikki experimental non-conformant: " + physical_device_impl->device_name;
    strncpy(properties->deviceName, device_name.c_str(), VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1);
    properties->deviceName[VK_MAX_PHYSICAL_DEVICE_NAME_SIZE - 1] = '\0';

    VkPhysicalDeviceLimits& limits = properties->limits;
    limits.maxImageDimension1D = 0;
    limits.maxImageDimension2D = 0;
    limits.maxImageDimension3D = 0;
    limits.maxImageDimensionCube = 0;
    limits.maxImageArrayLayers = 0;
    limits.maxTexelBufferElements = 0;
    limits.maxUniformBufferRange = (uint32_t)std::min<uint64_t>(physical_device_impl->limits.maxUniformBufferBindingSize, UINT32_MAX);
    limits.maxStorageBufferRange = (uint32_t)std::min<uint64_t>(physical_device_impl->limits.maxStorageBufferBindingSize, UINT32_MAX);
    limits.maxPushConstantsSize = (uint32_t)std::max<uint64_t>(128, std::min<uint64_t>(physical_device_impl->limits.maxUniformBufferBindingSize, 4096));
    limits.maxMemoryAllocationCount = 4096;
    limits.maxSamplerAllocationCount = 0;
    limits.bufferImageGranularity = 1;
    limits.sparseAddressSpaceSize = 0;
    limits.maxBoundDescriptorSets = 1;
    limits.maxPerStageDescriptorStorageBuffers = physical_device_impl->limits.maxStorageBuffersPerShaderStage;
    limits.maxDescriptorSetStorageBuffers = physical_device_impl->limits.maxStorageBuffersPerShaderStage;
    limits.maxPerStageDescriptorUniformBuffers = physical_device_impl->limits.maxUniformBuffersPerShaderStage;
    limits.maxDescriptorSetUniformBuffers = physical_device_impl->limits.maxUniformBuffersPerShaderStage;
    limits.maxPerStageResources = (uint32_t)std::min<uint64_t>(
        physical_device_impl->limits.maxBindingsPerBindGroup,
        (uint64_t)limits.maxPerStageDescriptorStorageBuffers
            + limits.maxPerStageDescriptorUniformBuffers);
    limits.maxComputeSharedMemorySize = physical_device_impl->limits.maxComputeWorkgroupStorageSize;
    limits.maxComputeWorkGroupCount[0] = physical_device_impl->limits.maxComputeWorkgroupsPerDimension;
    limits.maxComputeWorkGroupCount[1] = physical_device_impl->limits.maxComputeWorkgroupsPerDimension;
    limits.maxComputeWorkGroupCount[2] = physical_device_impl->limits.maxComputeWorkgroupsPerDimension;
    limits.maxComputeWorkGroupInvocations = physical_device_impl->limits.maxComputeInvocationsPerWorkgroup;
    limits.maxComputeWorkGroupSize[0] = physical_device_impl->limits.maxComputeWorkgroupSizeX;
    limits.maxComputeWorkGroupSize[1] = physical_device_impl->limits.maxComputeWorkgroupSizeY;
    limits.maxComputeWorkGroupSize[2] = physical_device_impl->limits.maxComputeWorkgroupSizeZ;
    limits.minMemoryMapAlignment = 8;
    limits.minTexelBufferOffsetAlignment = 1;
    limits.minUniformBufferOffsetAlignment = physical_device_impl->limits.minUniformBufferOffsetAlignment;
    limits.minStorageBufferOffsetAlignment = physical_device_impl->limits.minStorageBufferOffsetAlignment;
    limits.timestampComputeAndGraphics = physical_device_impl->profile.timestamp_query ? VK_TRUE : VK_FALSE;
    limits.timestampPeriod = physical_device_impl->profile.timestamp_query ? 1.f : 0.f;
    limits.nonCoherentAtomSize = 4;
}

void impl_get_physical_device_queue_family_properties(VkPhysicalDevice physical_device, uint32_t* property_count,
                                                      VkQueueFamilyProperties* properties)
{
    PhysicalDevice* physical_device_impl = unwrap(physical_device);
    if (!physical_device_impl || !property_count)
        return;

    if (!properties)
    {
        *property_count = 1;
        return;
    }

    if (*property_count == 0)
        return;

    memset(&properties[0], 0, sizeof(VkQueueFamilyProperties));
    properties[0].queueFlags = VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT;
    properties[0].queueCount = 1;
    properties[0].timestampValidBits = physical_device_impl->profile.timestamp_query ? 64 : 0;
    properties[0].minImageTransferGranularity.width = 1;
    properties[0].minImageTransferGranularity.height = 1;
    properties[0].minImageTransferGranularity.depth = 1;
    *property_count = 1;
}

void impl_get_physical_device_memory_properties(VkPhysicalDevice physical_device, VkPhysicalDeviceMemoryProperties* properties)
{
    PhysicalDevice* physical_device_impl = unwrap(physical_device);
    if (!physical_device_impl || !properties)
        return;

    memset(properties, 0, sizeof(VkPhysicalDeviceMemoryProperties));
    properties->memoryTypeCount = 2;
    properties->memoryTypes[0].propertyFlags = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    properties->memoryTypes[0].heapIndex = 0;
    properties->memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    properties->memoryTypes[1].heapIndex = 0;
    properties->memoryHeapCount = 1;
    properties->memoryHeaps[0].size = physical_device_impl->limits.maxBufferSize;
    properties->memoryHeaps[0].flags = VK_MEMORY_HEAP_DEVICE_LOCAL_BIT;
}

void impl_get_physical_device_format_properties(VkPhysicalDevice physical_device, VkFormat, VkFormatProperties* properties)
{
    if (!unwrap(physical_device) || !properties)
        return;

    memset(properties, 0, sizeof(VkFormatProperties));
}

VkResult impl_get_physical_device_image_format_properties(VkPhysicalDevice, VkFormat, VkImageType, VkImageTiling,
                                                          VkImageUsageFlags, VkImageCreateFlags, VkImageFormatProperties*)
{
    return VK_ERROR_FORMAT_NOT_SUPPORTED;
}

void impl_get_physical_device_features2(VkPhysicalDevice physical_device, VkPhysicalDeviceFeatures2KHR* features)
{
    if (!features || features->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR)
        return;

    PhysicalDevice* physical_device_impl = unwrap(physical_device);
    impl_get_physical_device_features(physical_device, &features->features);

    struct StructureHeader
    {
        VkStructureType sType;
        void* pNext;
    };

    StructureHeader* next = (StructureHeader*)features->pNext;
    while (next)
    {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR)
        {
            VkPhysicalDevice16BitStorageFeaturesKHR* storage = (VkPhysicalDevice16BitStorageFeaturesKHR*)next;
            storage->storageBuffer16BitAccess = physical_device_impl && physical_device_impl->profile.shader.shader_f16 ? VK_TRUE : VK_FALSE;
            storage->uniformAndStorageBuffer16BitAccess = VK_FALSE;
            storage->storagePushConstant16 = VK_FALSE;
            storage->storageInputOutput16 = VK_FALSE;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR)
        {
            VkPhysicalDeviceFloat16Int8FeaturesKHR* arithmetic = (VkPhysicalDeviceFloat16Int8FeaturesKHR*)next;
            arithmetic->shaderFloat16 = physical_device_impl && physical_device_impl->profile.shader.shader_f16 ? VK_TRUE : VK_FALSE;
            arithmetic->shaderInt8 = VK_FALSE;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR)
        {
            VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR* dot_product = (VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR*)next;
            dot_product->shaderIntegerDotProduct = physical_device_impl && physical_device_impl->profile.shader.packed_integer_dot ? VK_TRUE : VK_FALSE;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT)
        {
            VkPhysicalDeviceSubgroupSizeControlFeaturesEXT* subgroup = (VkPhysicalDeviceSubgroupSizeControlFeaturesEXT*)next;
            subgroup->subgroupSizeControl = physical_device_impl && physical_device_impl->profile.shader.subgroup.size_control_supported ? VK_TRUE : VK_FALSE;
            subgroup->computeFullSubgroups = VK_FALSE;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR)
        {
            VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR* subgroup = (VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR*)next;
            subgroup->shaderSubgroupExtendedTypes = physical_device_impl
                                                            && physical_device_impl->profile.shader.shader_f16
                                                            && physical_device_impl->profile.shader.subgroup.operations_supported
                                                        ? VK_TRUE
                                                        : VK_FALSE;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR)
        {
            VkPhysicalDeviceCooperativeMatrixFeaturesKHR* cooperative_matrix = (VkPhysicalDeviceCooperativeMatrixFeaturesKHR*)next;
            cooperative_matrix->cooperativeMatrix = physical_device_impl && physical_device_impl->profile.shader.cooperative_matrix.supported ? VK_TRUE : VK_FALSE;
            cooperative_matrix->cooperativeMatrixRobustBufferAccess = VK_FALSE;
        }

        next = (StructureHeader*)next->pNext;
    }
}

void impl_get_physical_device_properties2(VkPhysicalDevice physical_device, VkPhysicalDeviceProperties2KHR* properties)
{
    if (!properties || properties->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR)
        return;

    PhysicalDevice* physical_device_impl = unwrap(physical_device);
    impl_get_physical_device_properties(physical_device, &properties->properties);

    struct StructureHeader
    {
        VkStructureType sType;
        void* pNext;
    };

    StructureHeader* next = (StructureHeader*)properties->pNext;
    while (next)
    {
        if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES)
        {
            VkPhysicalDeviceDriverProperties* driver = (VkPhysicalDeviceDriverProperties*)next;
            // VK_DRIVER_ID_NIKKI, added in Vulkan-Headers 1.4.366.
            driver->driverID = static_cast<VkDriverId>(33);
            strncpy(driver->driverName, "nikki experimental non-conformant", VK_MAX_DRIVER_NAME_SIZE - 1);
            driver->driverName[VK_MAX_DRIVER_NAME_SIZE - 1] = '\0';
            const std::string driver_info = "nikki " NIKKI_RELEASE_VERSION_STRING " experimental non-conformant; WebGPU backend: "
                                            + physical_device_impl->driver_name;
            strncpy(driver->driverInfo, driver_info.c_str(), VK_MAX_DRIVER_INFO_SIZE - 1);
            driver->driverInfo[VK_MAX_DRIVER_INFO_SIZE - 1] = '\0';
            driver->conformanceVersion.major = 0;
            driver->conformanceVersion.minor = 0;
            driver->conformanceVersion.subminor = 0;
            driver->conformanceVersion.patch = 0;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES)
        {
            VkPhysicalDeviceSubgroupProperties* subgroup = (VkPhysicalDeviceSubgroupProperties*)next;
            const bool supported = physical_device_impl && physical_device_impl->profile.shader.subgroup.operations_supported;
            subgroup->subgroupSize = supported ? physical_device_impl->profile.shader.subgroup.default_size : 4;
            subgroup->supportedStages = supported ? VK_SHADER_STAGE_COMPUTE_BIT : 0;
            subgroup->supportedOperations = supported
                                                ? VK_SUBGROUP_FEATURE_BASIC_BIT
                                                      | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT
                                                      | VK_SUBGROUP_FEATURE_VOTE_BIT
                                                      | VK_SUBGROUP_FEATURE_BALLOT_BIT
                                                      | VK_SUBGROUP_FEATURE_SHUFFLE_BIT
                                                      | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT
                                                      | VK_SUBGROUP_FEATURE_QUAD_BIT
                                                : 0;
            subgroup->quadOperationsInAllStages = supported ? VK_TRUE : VK_FALSE;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT)
        {
            VkPhysicalDeviceSubgroupSizeControlPropertiesEXT* subgroup = (VkPhysicalDeviceSubgroupSizeControlPropertiesEXT*)next;
            const uint32_t min_subgroup_size = physical_device_impl && physical_device_impl->profile.shader.subgroup.size_control_supported
                                                   ? physical_device_impl->profile.shader.subgroup.min_size
                                               : physical_device_impl && physical_device_impl->profile.shader.subgroup.operations_supported
                                                   ? physical_device_impl->profile.shader.subgroup.default_size
                                                   : 4;
            const uint32_t max_subgroup_size = physical_device_impl && physical_device_impl->profile.shader.subgroup.size_control_supported
                                                   ? physical_device_impl->profile.shader.subgroup.max_size
                                               : physical_device_impl && physical_device_impl->profile.shader.subgroup.operations_supported
                                                   ? physical_device_impl->profile.shader.subgroup.default_size
                                                   : 128;
            subgroup->minSubgroupSize = min_subgroup_size;
            subgroup->maxSubgroupSize = max_subgroup_size;
            subgroup->maxComputeWorkgroupSubgroups = physical_device_impl ? std::max(physical_device_impl->limits.maxComputeInvocationsPerWorkgroup / min_subgroup_size, 1u) : 1;
            subgroup->requiredSubgroupSizeStages = physical_device_impl && physical_device_impl->profile.shader.subgroup.size_control_supported
                                                       ? VK_SHADER_STAGE_COMPUTE_BIT
                                                       : 0;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES)
        {
            VkPhysicalDeviceMaintenance3Properties* maintenance3 = (VkPhysicalDeviceMaintenance3Properties*)next;
            maintenance3->maxPerSetDescriptors = physical_device_impl ? physical_device_impl->limits.maxBindingsPerBindGroup : 0;
            maintenance3->maxMemoryAllocationSize = physical_device_impl ? physical_device_impl->limits.maxBufferSize : 0;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR)
        {
            VkPhysicalDeviceCooperativeMatrixPropertiesKHR* cooperative_matrix = (VkPhysicalDeviceCooperativeMatrixPropertiesKHR*)next;
            cooperative_matrix->cooperativeMatrixSupportedStages = physical_device_impl && physical_device_impl->profile.shader.cooperative_matrix.supported
                                                                       ? VK_SHADER_STAGE_COMPUTE_BIT
                                                                       : 0;
        }
        else if (next->sType == VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR)
        {
            VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR* dot_product = (VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR*)next;
            const VkBool32 packed = physical_device_impl && physical_device_impl->profile.shader.packed_integer_dot ? VK_TRUE : VK_FALSE;
            dot_product->integerDotProduct8BitUnsignedAccelerated = VK_FALSE;
            dot_product->integerDotProduct8BitSignedAccelerated = VK_FALSE;
            dot_product->integerDotProduct8BitMixedSignednessAccelerated = VK_FALSE;
            dot_product->integerDotProduct4x8BitPackedUnsignedAccelerated = packed;
            dot_product->integerDotProduct4x8BitPackedSignedAccelerated = packed;
            dot_product->integerDotProduct4x8BitPackedMixedSignednessAccelerated = VK_FALSE;
            dot_product->integerDotProduct16BitUnsignedAccelerated = VK_FALSE;
            dot_product->integerDotProduct16BitSignedAccelerated = VK_FALSE;
            dot_product->integerDotProduct16BitMixedSignednessAccelerated = VK_FALSE;
            dot_product->integerDotProduct32BitUnsignedAccelerated = VK_FALSE;
            dot_product->integerDotProduct32BitSignedAccelerated = VK_FALSE;
            dot_product->integerDotProduct32BitMixedSignednessAccelerated = VK_FALSE;
            dot_product->integerDotProduct64BitUnsignedAccelerated = VK_FALSE;
            dot_product->integerDotProduct64BitSignedAccelerated = VK_FALSE;
            dot_product->integerDotProduct64BitMixedSignednessAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating8BitUnsignedAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating8BitSignedAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating8BitMixedSignednessAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating4x8BitPackedMixedSignednessAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating16BitUnsignedAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating16BitSignedAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating16BitMixedSignednessAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating32BitUnsignedAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating32BitSignedAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating32BitMixedSignednessAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating64BitUnsignedAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating64BitSignedAccelerated = VK_FALSE;
            dot_product->integerDotProductAccumulatingSaturating64BitMixedSignednessAccelerated = VK_FALSE;
        }

        next = (StructureHeader*)next->pNext;
    }
}

VkResult impl_get_physical_device_cooperative_matrix_properties(
    VkPhysicalDevice physical_device, uint32_t* property_count, VkCooperativeMatrixPropertiesKHR* properties)
{
    PhysicalDevice* physical_device_impl = unwrap(physical_device);
    if (!physical_device_impl || !property_count)
        return VK_ERROR_INITIALIZATION_FAILED;

    const uint32_t available_count = physical_device_impl->profile.shader.cooperative_matrix.supported
                                         ? (uint32_t)physical_device_impl->profile.shader.cooperative_matrix.configs.size()
                                         : 0;
    if (!properties)
    {
        *property_count = available_count;
        return VK_SUCCESS;
    }

    const uint32_t write_count = std::min(*property_count, available_count);
    for (uint32_t i = 0; i < write_count; i++)
    {
        if (properties[i].sType != VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR)
            return VK_ERROR_INITIALIZATION_FAILED;

        const CooperativeMatrixConfig& config = physical_device_impl->profile.shader.cooperative_matrix.configs[i];
        properties[i].MSize = config.M;
        properties[i].NSize = config.N;
        properties[i].KSize = config.K;
        properties[i].AType = config.component_type;
        properties[i].BType = config.component_type;
        properties[i].CType = config.result_component_type;
        properties[i].ResultType = config.result_component_type;
        properties[i].saturatingAccumulation = VK_FALSE;
        properties[i].scope = VK_SCOPE_SUBGROUP_KHR;
    }

    *property_count = write_count;
    return write_count < available_count ? VK_INCOMPLETE : VK_SUCCESS;
}

void impl_get_physical_device_format_properties2(VkPhysicalDevice physical_device, VkFormat format,
                                                 VkFormatProperties2KHR* properties)
{
    if (!properties || properties->sType != VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR)
        return;

    impl_get_physical_device_format_properties(physical_device, format, &properties->formatProperties);
}

VkResult impl_get_physical_device_image_format_properties2(
    VkPhysicalDevice physical_device, const VkPhysicalDeviceImageFormatInfo2KHR* format_info,
    VkImageFormatProperties2KHR* properties)
{
    if (!format_info || format_info->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR
        || !properties || properties->sType != VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR)
        return VK_ERROR_INITIALIZATION_FAILED;

    return impl_get_physical_device_image_format_properties(physical_device, format_info->format, format_info->type,
                                                            format_info->tiling, format_info->usage, format_info->flags,
                                                            &properties->imageFormatProperties);
}

void impl_get_physical_device_queue_family_properties2(
    VkPhysicalDevice physical_device, uint32_t* property_count, VkQueueFamilyProperties2KHR* properties)
{
    if (!properties)
    {
        impl_get_physical_device_queue_family_properties(physical_device, property_count, 0);
        return;
    }
    if (!property_count || *property_count == 0
        || properties[0].sType != VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR)
        return;

    VkQueueFamilyProperties queue_properties;
    impl_get_physical_device_queue_family_properties(physical_device, property_count, &queue_properties);
    if (*property_count != 0)
        properties[0].queueFamilyProperties = queue_properties;
}

void impl_get_physical_device_memory_properties2(
    VkPhysicalDevice physical_device, VkPhysicalDeviceMemoryProperties2KHR* properties)
{
    if (!properties || properties->sType != VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR)
        return;

    impl_get_physical_device_memory_properties(physical_device, &properties->memoryProperties);
}

// VK_KHR_bind_memory2 and VK_KHR_get_memory_requirements2 also require image
// entry points. The compute profile exposes their Vulkan 1.1 buffer commands,
// but does not advertise the incomplete KHR extensions.
static const VkExtensionProperties g_device_extensions[] = {
    {"VK_KHR_descriptor_update_template", 1},
    {"VK_KHR_driver_properties", 1},
    {"VK_KHR_maintenance1", 2},
    {"VK_KHR_maintenance3", 1},
    {"VK_KHR_push_descriptor", 2},
    {"VK_KHR_storage_buffer_storage_class", 1}};

static const VkExtensionProperties g_subgroup_size_control_extension = {
    "VK_EXT_subgroup_size_control", 2};

static const VkExtensionProperties g_shader_subgroup_extended_types_extension = {
    "VK_KHR_shader_subgroup_extended_types", 1};

static const VkExtensionProperties g_16bit_storage_extension = {
    "VK_KHR_16bit_storage", 1};

static const VkExtensionProperties g_shader_float16_int8_extension = {
    "VK_KHR_shader_float16_int8", 1};

static const VkExtensionProperties g_shader_integer_dot_product_extension = {
    "VK_KHR_shader_integer_dot_product", 1};

static const VkExtensionProperties g_cooperative_matrix_extension = {
    "VK_KHR_cooperative_matrix", 2};

VkResult impl_enumerate_device_extension_properties(VkPhysicalDevice physical_device, const char* layer_name,
                                                    uint32_t* property_count, VkExtensionProperties* properties)
{
    if (!unwrap(physical_device))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (layer_name)
        return VK_ERROR_LAYER_NOT_PRESENT;

    PhysicalDevice* physical_device_impl = unwrap(physical_device);
    VkExtensionProperties extensions[sizeof(g_device_extensions) / sizeof(g_device_extensions[0]) + 6];
    memcpy(extensions, g_device_extensions, sizeof(g_device_extensions));
    uint32_t extension_count = sizeof(g_device_extensions) / sizeof(g_device_extensions[0]);
    if (physical_device_impl->profile.shader.shader_f16)
    {
        extensions[extension_count++] = g_16bit_storage_extension;
        extensions[extension_count++] = g_shader_float16_int8_extension;
    }
    extensions[extension_count++] = g_subgroup_size_control_extension;
    if (physical_device_impl->profile.shader.shader_f16 && physical_device_impl->profile.shader.subgroup.operations_supported)
        extensions[extension_count++] = g_shader_subgroup_extended_types_extension;
    if (physical_device_impl->profile.shader.packed_integer_dot)
        extensions[extension_count++] = g_shader_integer_dot_product_extension;
    if (physical_device_impl->profile.shader.cooperative_matrix.supported)
        extensions[extension_count++] = g_cooperative_matrix_extension;

    return enumerate_properties(extension_count, extensions, property_count, properties);
}

VkResult impl_enumerate_device_layer_properties(VkPhysicalDevice physical_device, uint32_t* property_count, VkLayerProperties*)
{
    if (!unwrap(physical_device) || !property_count)
        return VK_ERROR_INITIALIZATION_FAILED;

    *property_count = 0;
    return VK_SUCCESS;
}

bool device_extension_supported(const PhysicalDevice* physical_device, const char* name)
{
    if (!name)
        return false;

    for (size_t i = 0; i < sizeof(g_device_extensions) / sizeof(g_device_extensions[0]); i++)
    {
        if (strcmp(name, g_device_extensions[i].extensionName) == 0)
            return true;
    }

    if (physical_device
        && strcmp(name, g_subgroup_size_control_extension.extensionName) == 0)
        return true;

    if (physical_device && physical_device->profile.shader.shader_f16
        && physical_device->profile.shader.subgroup.operations_supported
        && strcmp(name, g_shader_subgroup_extended_types_extension.extensionName) == 0)
        return true;

    if (physical_device && physical_device->profile.shader.shader_f16
        && (strcmp(name, g_16bit_storage_extension.extensionName) == 0
            || strcmp(name, g_shader_float16_int8_extension.extensionName) == 0))
        return true;

    if (physical_device && physical_device->profile.shader.packed_integer_dot
        && strcmp(name, g_shader_integer_dot_product_extension.extensionName) == 0)
        return true;

    if (physical_device && physical_device->profile.shader.cooperative_matrix.supported
        && strcmp(name, g_cooperative_matrix_extension.extensionName) == 0)
        return true;

    return false;
}

} // namespace nikki_detail

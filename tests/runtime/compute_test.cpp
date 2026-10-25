// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

#include "../shaders/compute_spirv.h"
#include "../shaders/buffer_access_spirv.h"

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* name);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char* layer_name, uint32_t* property_count, VkExtensionProperties* properties);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo* create_info, const VkAllocationCallbacks* allocator, VkInstance* instance);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t* property_count, VkLayerProperties* properties);

// Derive a uniform-buffer variant of shader_spv without adding a generated
// shader artifact. The readonly input BufferBlock becomes a std140 Block with
// a fixed float[128] array; the storage output and push constants are unchanged.
static std::vector<uint32_t> make_uniform_input_shader_spv()
{
    std::vector<uint32_t> spirv(shader_spv, shader_spv + sizeof(shader_spv) / sizeof(shader_spv[0]));
    if (spirv.size() < 5)
        return std::vector<uint32_t>();

    const uint32_t input_array_type_id = 0x23;
    const uint32_t input_block_type_id = 0x24;
    const uint32_t float_type_id = 0x11;
    const uint32_t uint_type_id = 0x06;
    const uint32_t array_length_id = spirv[3];
    bool array_stride_patched = false;
    bool block_patched = false;
    bool array_type_patched = false;

    for (size_t offset = 5; offset < spirv.size();)
    {
        const uint32_t instruction = spirv[offset];
        const uint32_t word_count = instruction >> 16;
        const uint32_t opcode = instruction & 0xffff;
        if (word_count == 0 || offset + word_count > spirv.size())
            return std::vector<uint32_t>();

        // OpDecorate %input_array ArrayStride 4 -> 16.
        if (opcode == 71 && word_count == 4
            && spirv[offset + 1] == input_array_type_id
            && spirv[offset + 2] == 6)
        {
            spirv[offset + 3] = 16;
            array_stride_patched = true;
        }
        // OpDecorate %input_block BufferBlock -> Block.
        else if (opcode == 71 && word_count == 3
                 && spirv[offset + 1] == input_block_type_id
                 && spirv[offset + 2] == 3)
        {
            spirv[offset + 2] = 2;
            block_patched = true;
        }
        // Replace OpTypeRuntimeArray with an OpConstant length followed by
        // OpTypeArray. The old header bound is the first unused result id.
        else if (opcode == 29 && word_count == 3
                 && spirv[offset + 1] == input_array_type_id
                 && spirv[offset + 2] == float_type_id)
        {
            const uint32_t replacement[] = {
                0x0004002b, uint_type_id, array_length_id, 128,
                0x0004001c, input_array_type_id, float_type_id, array_length_id};
            spirv.erase(spirv.begin() + offset, spirv.begin() + offset + word_count);
            spirv.insert(spirv.begin() + offset,
                         replacement, replacement + sizeof(replacement) / sizeof(replacement[0]));
            offset += sizeof(replacement) / sizeof(replacement[0]);
            array_type_patched = true;
            continue;
        }

        offset += word_count;
    }

    if (!array_stride_patched || !block_patched || !array_type_patched
        || array_length_id == UINT32_MAX)
        return std::vector<uint32_t>();

    spirv[3] = array_length_id + 1;
    return spirv;
}

static int has_extension(const VkExtensionProperties* properties, uint32_t count, const char* name)
{
    for (uint32_t i = 0; i < count; i++)
    {
        if (strcmp(properties[i].extensionName, name) == 0)
            return 1;
    }

    return 0;
}

static int fail(const char* operation, VkResult result)
{
    fprintf(stderr, "%s failed %d\n", operation, result);
    return 1;
}

#define LOAD_INSTANCE_PROC(name)                                                   \
    PFN_vk##name name = (PFN_vk##name)vkGetInstanceProcAddr(instance, "vk" #name); \
    if (!name)                                                                     \
    {                                                                              \
        fprintf(stderr, "vkGetInstanceProcAddr returned null for vk%s\n", #name);  \
        return 1;                                                                  \
    }

#define LOAD_DEVICE_PROC(name)                                                  \
    PFN_vk##name name = (PFN_vk##name)GetDeviceProcAddr(device, "vk" #name);    \
    if (!name)                                                                  \
    {                                                                           \
        fprintf(stderr, "vkGetDeviceProcAddr returned null for vk%s\n", #name); \
        return 1;                                                               \
    }

#define CHECK_VK(operation)                  \
    do                                       \
    {                                        \
        VkResult result = (operation);       \
        if (result != VK_SUCCESS)            \
            return fail(#operation, result); \
    } while (0)

int main()
{
    if (vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateDevice") != 0
        || vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkQueueSubmit") != 0)
    {
        fprintf(stderr, "vkGetInstanceProcAddr resolved a non-global command without an instance\n");
        return 1;
    }

    PFN_vkEnumerateInstanceVersion EnumerateInstanceVersion = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    if (!EnumerateInstanceVersion)
    {
        fprintf(stderr, "vkGetInstanceProcAddr returned null for vkEnumerateInstanceVersion\n");
        return 1;
    }
    uint32_t instance_version = 0;
    CHECK_VK(EnumerateInstanceVersion(&instance_version));
    if (instance_version < VK_API_VERSION_1_1)
    {
        fprintf(stderr, "unexpected instance version %u\n", instance_version);
        return 1;
    }

    uint32_t layer_count = 1;
    CHECK_VK(vkEnumerateInstanceLayerProperties(&layer_count, 0));
    if (layer_count != 0)
    {
        fprintf(stderr, "unexpected instance layer count %u\n", layer_count);
        return 1;
    }

    uint32_t instance_extension_count = 0;
    CHECK_VK(vkEnumerateInstanceExtensionProperties(0, &instance_extension_count, 0));
    if (instance_extension_count != 1)
    {
        fprintf(stderr, "unexpected instance extension count %u\n", instance_extension_count);
        return 1;
    }
    VkExtensionProperties instance_extensions[1];
    CHECK_VK(vkEnumerateInstanceExtensionProperties(0, &instance_extension_count, instance_extensions));
    if (!has_extension(instance_extensions, instance_extension_count, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
    {
        fprintf(stderr, "VK_KHR_get_physical_device_properties2 is missing\n");
        return 1;
    }

    VkApplicationInfo application_info;
    memset(&application_info, 0, sizeof(application_info));
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.pApplicationName = "test_nikki_compute";
    application_info.apiVersion = VK_MAKE_VERSION(1, 1, 123);

    const char* instance_extension_names[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
    VkInstanceCreateInfo instance_create_info;
    memset(&instance_create_info, 0, sizeof(instance_create_info));
    instance_create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_create_info.pApplicationInfo = &application_info;
    instance_create_info.enabledExtensionCount = 1;
    instance_create_info.ppEnabledExtensionNames = instance_extension_names;

    VkInstance instance = VK_NULL_HANDLE;
    CHECK_VK(vkCreateInstance(&instance_create_info, 0, &instance));
    if (!instance)
    {
        fprintf(stderr, "vkCreateInstance returned a null handle\n");
        return 1;
    }
    if (vkGetInstanceProcAddr(instance, "vkCreateImage") != 0)
    {
        fprintf(stderr, "unsupported vkCreateImage unexpectedly resolved\n");
        return 1;
    }

    LOAD_INSTANCE_PROC(DestroyInstance);
    LOAD_INSTANCE_PROC(EnumeratePhysicalDevices);
    LOAD_INSTANCE_PROC(EnumerateDeviceExtensionProperties);
    LOAD_INSTANCE_PROC(EnumerateDeviceLayerProperties);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceFeatures);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceFeatures2KHR);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceProperties2KHR);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceProperties);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceFormatProperties);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceFormatProperties2KHR);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceImageFormatProperties);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceImageFormatProperties2KHR);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceMemoryProperties);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceMemoryProperties2KHR);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceQueueFamilyProperties);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceQueueFamilyProperties2KHR);
    LOAD_INSTANCE_PROC(GetPhysicalDeviceCooperativeMatrixPropertiesKHR);
    LOAD_INSTANCE_PROC(CreateDevice);
    LOAD_INSTANCE_PROC(GetDeviceProcAddr);

    if (vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2") != vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR")
        || vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2") != vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2KHR")
        || vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties2") != vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceMemoryProperties2KHR")
        || vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyProperties2") != vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceQueueFamilyProperties2KHR"))
    {
        fprintf(stderr, "Vulkan 1.1 physical device query aliases do not match\n");
        return 1;
    }

    uint32_t physical_device_count = 0;
    CHECK_VK(EnumeratePhysicalDevices(instance, &physical_device_count, 0));
    if (physical_device_count == 0)
    {
        fprintf(stderr, "unexpected physical device count %u\n", physical_device_count);
        return 1;
    }
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    CHECK_VK(EnumeratePhysicalDevices(instance, &physical_device_count, physical_devices.data()));
    VkPhysicalDevice physical_device = physical_devices[0];

    VkPhysicalDeviceFeatures features;
    VkPhysicalDeviceFeatures zero_features;
    memset(&features, 0xff, sizeof(features));
    memset(&zero_features, 0, sizeof(zero_features));
    GetPhysicalDeviceFeatures(physical_device, &features);
    if (memcmp(&features, &zero_features, sizeof(features)) != 0)
    {
        fprintf(stderr, "unexpected enabled physical device feature\n");
        return 1;
    }

    VkPhysicalDeviceFeatures2KHR features2;
    memset(&features2, 0, sizeof(features2));
    features2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2_KHR;
    GetPhysicalDeviceFeatures2KHR(physical_device, &features2);

    VkPhysicalDeviceProperties physical_device_properties;
    GetPhysicalDeviceProperties(physical_device, &physical_device_properties);
    if (physical_device_properties.apiVersion < VK_API_VERSION_1_1)
    {
        fprintf(stderr, "unexpected physical device version %u\n", physical_device_properties.apiVersion);
        return 1;
    }
    if (!strstr(physical_device_properties.deviceName, "experimental")
        || !strstr(physical_device_properties.deviceName, "non-conformant"))
    {
        fprintf(stderr, "physical device name does not disclose non-conformant status: %s\n",
                physical_device_properties.deviceName);
        return 1;
    }
    if (physical_device_properties.limits.maxComputeWorkGroupInvocations < 64)
    {
        fprintf(stderr, "maxComputeWorkGroupInvocations is too small\n");
        return 1;
    }
    uint32_t queue_family_count = 0;
    GetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, 0);
    if (queue_family_count != 1)
    {
        fprintf(stderr, "unexpected queue family count %u\n", queue_family_count);
        return 1;
    }
    VkQueueFamilyProperties queue_family_properties;
    GetPhysicalDeviceQueueFamilyProperties(physical_device, &queue_family_count, &queue_family_properties);
    if ((queue_family_properties.queueFlags & (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT))
        != (VK_QUEUE_COMPUTE_BIT | VK_QUEUE_TRANSFER_BIT))
    {
        fprintf(stderr, "queue family does not support compute and transfer\n");
        return 1;
    }
    const bool timestamp_query_supported = physical_device_properties.limits.timestampComputeAndGraphics == VK_TRUE;
    if (timestamp_query_supported
            ? (queue_family_properties.timestampValidBits != 64 || physical_device_properties.limits.timestampPeriod != 1.f)
            : (queue_family_properties.timestampValidBits != 0 || physical_device_properties.limits.timestampPeriod != 0.f))
    {
        fprintf(stderr, "unexpected WebGPU timestamp query profile bits=%u period=%f\n",
                queue_family_properties.timestampValidBits, physical_device_properties.limits.timestampPeriod);
        return 1;
    }

    VkQueueFamilyProperties2KHR queue_family_properties2;
    memset(&queue_family_properties2, 0, sizeof(queue_family_properties2));
    queue_family_properties2.sType = VK_STRUCTURE_TYPE_QUEUE_FAMILY_PROPERTIES_2_KHR;
    queue_family_count = 1;
    GetPhysicalDeviceQueueFamilyProperties2KHR(
        physical_device, &queue_family_count, &queue_family_properties2);
    if (queue_family_count != 1
        || queue_family_properties2.queueFamilyProperties.queueFlags != queue_family_properties.queueFlags)
    {
        fprintf(stderr, "queue family properties2 mismatch\n");
        return 1;
    }

    VkPhysicalDeviceMemoryProperties memory_properties;
    GetPhysicalDeviceMemoryProperties(physical_device, &memory_properties);
    VkPhysicalDeviceMemoryProperties2KHR memory_properties2;
    memset(&memory_properties2, 0, sizeof(memory_properties2));
    memory_properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2_KHR;
    GetPhysicalDeviceMemoryProperties2KHR(physical_device, &memory_properties2);
    if (memory_properties2.memoryProperties.memoryTypeCount != memory_properties.memoryTypeCount
        || memory_properties2.memoryProperties.memoryHeapCount != memory_properties.memoryHeapCount)
    {
        fprintf(stderr, "memory properties2 mismatch\n");
        return 1;
    }

    VkFormatProperties format_properties;
    memset(&format_properties, 0xff, sizeof(format_properties));
    GetPhysicalDeviceFormatProperties(physical_device, VK_FORMAT_R32_SFLOAT, &format_properties);
    if (format_properties.linearTilingFeatures != 0 || format_properties.optimalTilingFeatures != 0
        || format_properties.bufferFeatures != 0)
    {
        fprintf(stderr, "buffer-only profile exposed format features\n");
        return 1;
    }

    VkFormatProperties2KHR format_properties2;
    memset(&format_properties2, 0, sizeof(format_properties2));
    format_properties2.sType = VK_STRUCTURE_TYPE_FORMAT_PROPERTIES_2_KHR;
    GetPhysicalDeviceFormatProperties2KHR(physical_device, VK_FORMAT_R32_SFLOAT, &format_properties2);
    if (format_properties2.formatProperties.linearTilingFeatures != 0
        || format_properties2.formatProperties.optimalTilingFeatures != 0
        || format_properties2.formatProperties.bufferFeatures != 0)
    {
        fprintf(stderr, "buffer-only profile exposed format features2\n");
        return 1;
    }

    VkImageFormatProperties image_format_properties;
    if (GetPhysicalDeviceImageFormatProperties(
            physical_device, VK_FORMAT_R32_SFLOAT, VK_IMAGE_TYPE_2D, VK_IMAGE_TILING_OPTIMAL,
            VK_IMAGE_USAGE_STORAGE_BIT, 0, &image_format_properties)
        != VK_ERROR_FORMAT_NOT_SUPPORTED)
    {
        fprintf(stderr, "buffer-only profile unexpectedly supports an image format\n");
        return 1;
    }

    VkPhysicalDeviceImageFormatInfo2KHR image_format_info2;
    memset(&image_format_info2, 0, sizeof(image_format_info2));
    image_format_info2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_IMAGE_FORMAT_INFO_2_KHR;
    image_format_info2.format = VK_FORMAT_R32_SFLOAT;
    image_format_info2.type = VK_IMAGE_TYPE_2D;
    image_format_info2.tiling = VK_IMAGE_TILING_OPTIMAL;
    image_format_info2.usage = VK_IMAGE_USAGE_STORAGE_BIT;
    VkImageFormatProperties2KHR image_format_properties2;
    memset(&image_format_properties2, 0, sizeof(image_format_properties2));
    image_format_properties2.sType = VK_STRUCTURE_TYPE_IMAGE_FORMAT_PROPERTIES_2_KHR;
    if (GetPhysicalDeviceImageFormatProperties2KHR(
            physical_device, &image_format_info2, &image_format_properties2)
        != VK_ERROR_FORMAT_NOT_SUPPORTED)
    {
        fprintf(stderr, "buffer-only profile unexpectedly supports an image format2\n");
        return 1;
    }

    uint32_t device_layer_count = 1;
    CHECK_VK(EnumerateDeviceLayerProperties(physical_device, &device_layer_count, 0));
    if (device_layer_count != 0)
    {
        fprintf(stderr, "unexpected device layer count %u\n", device_layer_count);
        return 1;
    }

    uint32_t device_extension_count = 0;
    CHECK_VK(EnumerateDeviceExtensionProperties(physical_device, 0, &device_extension_count, 0));
    if (device_extension_count < 6 || device_extension_count > 12)
    {
        fprintf(stderr, "unexpected device extension count %u\n", device_extension_count);
        return 1;
    }
    std::vector<VkExtensionProperties> device_extensions(device_extension_count);
    CHECK_VK(EnumerateDeviceExtensionProperties(
        physical_device, 0, &device_extension_count, device_extensions.data()));
    if (!has_extension(device_extensions.data(), device_extension_count, VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME)
        || !has_extension(device_extensions.data(), device_extension_count, VK_KHR_MAINTENANCE1_EXTENSION_NAME)
        || !has_extension(device_extensions.data(), device_extension_count, VK_KHR_MAINTENANCE3_EXTENSION_NAME)
        || !has_extension(device_extensions.data(), device_extension_count, VK_KHR_DRIVER_PROPERTIES_EXTENSION_NAME)
        || !has_extension(device_extensions.data(), device_extension_count, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME))
    {
        fprintf(stderr, "required device extension is missing\n");
        return 1;
    }

    const bool subgroup_size_control_extension_supported = has_extension(device_extensions.data(), device_extension_count, "VK_EXT_subgroup_size_control");
    const bool subgroup_extended_types_supported = has_extension(device_extensions.data(), device_extension_count, "VK_KHR_shader_subgroup_extended_types");
    const bool shader_f16_supported = has_extension(device_extensions.data(), device_extension_count, VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    const bool packed_integer_dot_supported = has_extension(device_extensions.data(), device_extension_count, VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    const bool cooperative_matrix_supported = has_extension(device_extensions.data(), device_extension_count, "VK_KHR_cooperative_matrix");
    if (shader_f16_supported != (has_extension(device_extensions.data(), device_extension_count, VK_KHR_16BIT_STORAGE_EXTENSION_NAME) != 0))
    {
        fprintf(stderr, "WebGPU f16 extensions were not advertised together\n");
        return 1;
    }

    VkPhysicalDevice16BitStorageFeaturesKHR storage_16bit_features;
    memset(&storage_16bit_features, 0, sizeof(storage_16bit_features));
    storage_16bit_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES_KHR;
    VkPhysicalDeviceFloat16Int8FeaturesKHR float16_int8_features;
    memset(&float16_int8_features, 0, sizeof(float16_int8_features));
    float16_int8_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR;
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroup_features;
    memset(&subgroup_features, 0, sizeof(subgroup_features));
    subgroup_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT;
    VkPhysicalDeviceShaderSubgroupExtendedTypesFeaturesKHR subgroup_extended_types_features;
    memset(&subgroup_extended_types_features, 0, sizeof(subgroup_extended_types_features));
    subgroup_extended_types_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_SUBGROUP_EXTENDED_TYPES_FEATURES_KHR;
    VkPhysicalDeviceShaderIntegerDotProductFeaturesKHR integer_dot_features;
    memset(&integer_dot_features, 0, sizeof(integer_dot_features));
    integer_dot_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_FEATURES_KHR;
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR cooperative_matrix_features;
    memset(&cooperative_matrix_features, 0, sizeof(cooperative_matrix_features));
    cooperative_matrix_features.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR;
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subgroup_size_properties;
    memset(&subgroup_size_properties, 0, sizeof(subgroup_size_properties));
    subgroup_size_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT;
    if (shader_f16_supported || subgroup_size_control_extension_supported || subgroup_extended_types_supported
        || packed_integer_dot_supported || cooperative_matrix_supported)
    {
        void* feature_chain = 0;
        if (subgroup_extended_types_supported)
        {
            subgroup_extended_types_features.pNext = feature_chain;
            feature_chain = &subgroup_extended_types_features;
        }
        if (subgroup_size_control_extension_supported)
        {
            subgroup_features.pNext = feature_chain;
            feature_chain = &subgroup_features;
        }
        if (packed_integer_dot_supported)
        {
            integer_dot_features.pNext = feature_chain;
            feature_chain = &integer_dot_features;
        }
        if (cooperative_matrix_supported)
        {
            cooperative_matrix_features.pNext = feature_chain;
            feature_chain = &cooperative_matrix_features;
        }
        if (shader_f16_supported)
        {
            float16_int8_features.pNext = feature_chain;
            storage_16bit_features.pNext = &float16_int8_features;
            feature_chain = &storage_16bit_features;
        }
        features2.pNext = feature_chain;
        GetPhysicalDeviceFeatures2KHR(physical_device, &features2);
    }
    const bool subgroup_size_control_supported = subgroup_features.subgroupSizeControl == VK_TRUE;
    if (shader_f16_supported)
    {
        if (!storage_16bit_features.storageBuffer16BitAccess
            || storage_16bit_features.uniformAndStorageBuffer16BitAccess
            || storage_16bit_features.storagePushConstant16
            || storage_16bit_features.storageInputOutput16
            || !float16_int8_features.shaderFloat16
            || float16_int8_features.shaderInt8)
        {
            fprintf(stderr, "unexpected WebGPU f16 feature profile\n");
            return 1;
        }
    }
    if (packed_integer_dot_supported && !integer_dot_features.shaderIntegerDotProduct)
    {
        fprintf(stderr, "packed integer dot product feature is missing\n");
        return 1;
    }
    if (cooperative_matrix_features.cooperativeMatrix != (cooperative_matrix_supported ? VK_TRUE : VK_FALSE)
        || cooperative_matrix_features.cooperativeMatrixRobustBufferAccess)
    {
        fprintf(stderr, "unexpected WebGPU cooperative matrix feature profile\n");
        return 1;
    }

    VkPhysicalDeviceShaderIntegerDotProductPropertiesKHR integer_dot_properties;
    memset(&integer_dot_properties, 0, sizeof(integer_dot_properties));
    integer_dot_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_INTEGER_DOT_PRODUCT_PROPERTIES_KHR;
    VkPhysicalDeviceMaintenance3Properties maintenance3_properties;
    memset(&maintenance3_properties, 0, sizeof(maintenance3_properties));
    maintenance3_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES;
    VkPhysicalDeviceCooperativeMatrixPropertiesKHR cooperative_matrix_device_properties;
    memset(&cooperative_matrix_device_properties, 0, sizeof(cooperative_matrix_device_properties));
    cooperative_matrix_device_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
    VkPhysicalDeviceSubgroupProperties subgroup_properties;
    memset(&subgroup_properties, 0, sizeof(subgroup_properties));
    subgroup_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES;
    VkPhysicalDeviceDriverProperties driver_properties;
    memset(&driver_properties, 0, sizeof(driver_properties));
    driver_properties.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRIVER_PROPERTIES;
    void* property_chain = 0;
    if (cooperative_matrix_supported)
    {
        cooperative_matrix_device_properties.pNext = property_chain;
        property_chain = &cooperative_matrix_device_properties;
    }
    subgroup_size_properties.pNext = property_chain;
    property_chain = &subgroup_size_properties;
    if (packed_integer_dot_supported)
    {
        integer_dot_properties.pNext = property_chain;
        property_chain = &integer_dot_properties;
    }
    subgroup_properties.pNext = property_chain;
    property_chain = &subgroup_properties;
    driver_properties.pNext = property_chain;
    property_chain = &driver_properties;
    maintenance3_properties.pNext = property_chain;
    VkPhysicalDeviceProperties2KHR properties2;
    memset(&properties2, 0, sizeof(properties2));
    properties2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2_KHR;
    properties2.pNext = &maintenance3_properties;
    GetPhysicalDeviceProperties2KHR(physical_device, &properties2);
    if (maintenance3_properties.maxPerSetDescriptors == 0
        || maintenance3_properties.maxMemoryAllocationSize == 0)
    {
        fprintf(stderr, "invalid maintenance3 properties\n");
        return 1;
    }
    if (driver_properties.driverID != static_cast<VkDriverId>(33))
    {
        fprintf(stderr, "unexpected nikki driver ID: %u\n", (unsigned int)driver_properties.driverID);
        return 1;
    }
    if (!strstr(driver_properties.driverName, "experimental")
        || !strstr(driver_properties.driverName, "non-conformant")
        || !strstr(driver_properties.driverInfo, "experimental")
        || !strstr(driver_properties.driverInfo, "non-conformant")
        || driver_properties.conformanceVersion.major != 0
        || driver_properties.conformanceVersion.minor != 0
        || driver_properties.conformanceVersion.subminor != 0
        || driver_properties.conformanceVersion.patch != 0)
    {
        fprintf(stderr, "driver properties do not disclose zero-conformance status\n");
        return 1;
    }

    if (subgroup_size_control_supported)
    {
        if (subgroup_size_properties.minSubgroupSize == 0
            || subgroup_size_properties.maxSubgroupSize < subgroup_size_properties.minSubgroupSize)
        {
            fprintf(stderr, "invalid WebGPU subgroup size range %u~%u\n",
                    subgroup_size_properties.minSubgroupSize, subgroup_size_properties.maxSubgroupSize);
            return 1;
        }
        if (!subgroup_features.subgroupSizeControl || subgroup_features.computeFullSubgroups
            || subgroup_size_properties.requiredSubgroupSizeStages != VK_SHADER_STAGE_COMPUTE_BIT)
        {
            fprintf(stderr, "unexpected WebGPU subgroup size control profile\n");
            return 1;
        }
    }
    else if (subgroup_size_properties.minSubgroupSize != 4
             || subgroup_size_properties.maxSubgroupSize != 128
             || subgroup_size_properties.requiredSubgroupSizeStages != 0)
    {
        fprintf(stderr, "unexpected fallback WebGPU subgroup size range %u~%u\n",
                subgroup_size_properties.minSubgroupSize, subgroup_size_properties.maxSubgroupSize);
        return 1;
    }
    uint32_t cooperative_matrix_property_count = 0;
    CHECK_VK(GetPhysicalDeviceCooperativeMatrixPropertiesKHR(
        physical_device, &cooperative_matrix_property_count, 0));
    if (cooperative_matrix_supported)
    {
        if (!cooperative_matrix_features.cooperativeMatrix
            || cooperative_matrix_device_properties.cooperativeMatrixSupportedStages != VK_SHADER_STAGE_COMPUTE_BIT
            || cooperative_matrix_property_count == 0)
        {
            fprintf(stderr, "invalid WebGPU cooperative matrix profile\n");
            return 1;
        }

        std::vector<VkCooperativeMatrixPropertiesKHR> cooperative_matrix_properties(cooperative_matrix_property_count);
        for (uint32_t i = 0; i < cooperative_matrix_property_count; i++)
        {
            memset(&cooperative_matrix_properties[i], 0, sizeof(cooperative_matrix_properties[i]));
            cooperative_matrix_properties[i].sType = VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR;
        }
        CHECK_VK(GetPhysicalDeviceCooperativeMatrixPropertiesKHR(
            physical_device, &cooperative_matrix_property_count, cooperative_matrix_properties.data()));
        for (uint32_t i = 0; i < cooperative_matrix_property_count; i++)
        {
            const VkCooperativeMatrixPropertiesKHR& matrix = cooperative_matrix_properties[i];
            if (matrix.MSize == 0 || matrix.NSize == 0 || matrix.KSize == 0
                || matrix.AType != matrix.BType || matrix.CType != matrix.ResultType
                || matrix.saturatingAccumulation || matrix.scope != VK_SCOPE_SUBGROUP_KHR)
            {
                fprintf(stderr, "invalid WebGPU cooperative matrix configuration\n");
                return 1;
            }
        }
    }
    else if (cooperative_matrix_property_count != 0
             || cooperative_matrix_device_properties.cooperativeMatrixSupportedStages != 0)
    {
        fprintf(stderr, "unsupported WebGPU cooperative matrix profile is non-empty\n");
        return 1;
    }

    const VkSubgroupFeatureFlags required_subgroup_operations = VK_SUBGROUP_FEATURE_BASIC_BIT
                                                                | VK_SUBGROUP_FEATURE_ARITHMETIC_BIT
                                                                | VK_SUBGROUP_FEATURE_VOTE_BIT
                                                                | VK_SUBGROUP_FEATURE_BALLOT_BIT
                                                                | VK_SUBGROUP_FEATURE_SHUFFLE_BIT
                                                                | VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT
                                                                | VK_SUBGROUP_FEATURE_QUAD_BIT;
    const bool subgroup_operations_supported = (subgroup_properties.supportedOperations & required_subgroup_operations) == required_subgroup_operations;
    if (subgroup_extended_types_supported != (shader_f16_supported && subgroup_operations_supported)
        || subgroup_extended_types_features.shaderSubgroupExtendedTypes != (subgroup_extended_types_supported ? VK_TRUE : VK_FALSE))
    {
        fprintf(stderr, "unexpected WebGPU subgroup extended types profile\n");
        return 1;
    }
    if (subgroup_operations_supported)
    {
        if (subgroup_properties.subgroupSize == 0
            || (subgroup_properties.subgroupSize & (subgroup_properties.subgroupSize - 1)) != 0
            || subgroup_properties.supportedStages != VK_SHADER_STAGE_COMPUTE_BIT
            || !subgroup_properties.quadOperationsInAllStages)
        {
            fprintf(stderr, "invalid WebGPU subgroup properties\n");
            return 1;
        }
        if (subgroup_size_control_supported
            && (subgroup_properties.subgroupSize < subgroup_size_properties.minSubgroupSize
                || subgroup_properties.subgroupSize > subgroup_size_properties.maxSubgroupSize))
        {
            fprintf(stderr, "WebGPU default subgroup size is outside the controlled range\n");
            return 1;
        }
    }
    else if (subgroup_properties.subgroupSize != 4
             || subgroup_properties.supportedStages != 0
             || subgroup_properties.supportedOperations != 0
             || subgroup_properties.quadOperationsInAllStages)
    {
        fprintf(stderr, "unexpected unsupported WebGPU subgroup profile\n");
        return 1;
    }
    if (packed_integer_dot_supported)
    {
        if (!integer_dot_properties.integerDotProduct4x8BitPackedUnsignedAccelerated
            || !integer_dot_properties.integerDotProduct4x8BitPackedSignedAccelerated
            || integer_dot_properties.integerDotProduct4x8BitPackedMixedSignednessAccelerated
            || integer_dot_properties.integerDotProductAccumulatingSaturating4x8BitPackedUnsignedAccelerated
            || integer_dot_properties.integerDotProductAccumulatingSaturating4x8BitPackedSignedAccelerated)
        {
            fprintf(stderr, "unexpected WebGPU packed integer dot product profile\n");
            return 1;
        }
    }

    float queue_priority = 1.f;
    VkDeviceQueueCreateInfo queue_create_info;
    memset(&queue_create_info, 0, sizeof(queue_create_info));
    queue_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_create_info.queueFamilyIndex = 0;
    queue_create_info.queueCount = 1;
    queue_create_info.pQueuePriorities = &queue_priority;

    std::vector<const char*> device_extension_names;
    device_extension_names.push_back(VK_KHR_DESCRIPTOR_UPDATE_TEMPLATE_EXTENSION_NAME);
    device_extension_names.push_back(VK_KHR_MAINTENANCE1_EXTENSION_NAME);
    device_extension_names.push_back(VK_KHR_MAINTENANCE3_EXTENSION_NAME);
    device_extension_names.push_back(VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    device_extension_names.push_back(VK_KHR_STORAGE_BUFFER_STORAGE_CLASS_EXTENSION_NAME);
    if (shader_f16_supported)
    {
        device_extension_names.push_back(VK_KHR_16BIT_STORAGE_EXTENSION_NAME);
        device_extension_names.push_back(VK_KHR_SHADER_FLOAT16_INT8_EXTENSION_NAME);
    }
    if (subgroup_size_control_extension_supported)
        device_extension_names.push_back("VK_EXT_subgroup_size_control");
    if (subgroup_extended_types_supported)
        device_extension_names.push_back("VK_KHR_shader_subgroup_extended_types");
    if (packed_integer_dot_supported)
        device_extension_names.push_back(VK_KHR_SHADER_INTEGER_DOT_PRODUCT_EXTENSION_NAME);
    if (cooperative_matrix_supported)
        device_extension_names.push_back("VK_KHR_cooperative_matrix");
    VkDeviceCreateInfo device_create_info;
    memset(&device_create_info, 0, sizeof(device_create_info));
    device_create_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_create_info.pNext = features2.pNext;
    device_create_info.queueCreateInfoCount = 1;
    device_create_info.pQueueCreateInfos = &queue_create_info;
    device_create_info.enabledExtensionCount = device_extension_names.size();
    device_create_info.ppEnabledExtensionNames = device_extension_names.data();

    VkDevice device = VK_NULL_HANDLE;
    CHECK_VK(CreateDevice(physical_device, &device_create_info, 0, &device));

    if (GetDeviceProcAddr(device, "vkCreateDescriptorUpdateTemplate") != GetDeviceProcAddr(device, "vkCreateDescriptorUpdateTemplateKHR")
        || GetDeviceProcAddr(device, "vkDestroyDescriptorUpdateTemplate") != GetDeviceProcAddr(device, "vkDestroyDescriptorUpdateTemplateKHR")
        || GetDeviceProcAddr(device, "vkTrimCommandPool") != GetDeviceProcAddr(device, "vkTrimCommandPoolKHR")
        || GetDeviceProcAddr(device, "vkGetDescriptorSetLayoutSupport") != GetDeviceProcAddr(device, "vkGetDescriptorSetLayoutSupportKHR"))
    {
        fprintf(stderr, "Vulkan 1.1 device aliases do not match\n");
        return 1;
    }
    if (GetDeviceProcAddr(device, "vkBindBufferMemory2KHR") != 0
        || GetDeviceProcAddr(device, "vkGetBufferMemoryRequirements2KHR") != 0)
    {
        fprintf(stderr, "an unadvertised image-bearing KHR extension unexpectedly resolved\n");
        return 1;
    }

    LOAD_DEVICE_PROC(DestroyDevice);
    LOAD_DEVICE_PROC(GetDeviceQueue);
    LOAD_DEVICE_PROC(CreateBuffer);
    LOAD_DEVICE_PROC(DestroyBuffer);
    LOAD_DEVICE_PROC(GetBufferMemoryRequirements);
    LOAD_DEVICE_PROC(GetBufferMemoryRequirements2);
    LOAD_DEVICE_PROC(BindBufferMemory);
    LOAD_DEVICE_PROC(BindBufferMemory2);
    LOAD_DEVICE_PROC(AllocateMemory);
    LOAD_DEVICE_PROC(FreeMemory);
    LOAD_DEVICE_PROC(MapMemory);
    LOAD_DEVICE_PROC(UnmapMemory);
    LOAD_DEVICE_PROC(FlushMappedMemoryRanges);
    LOAD_DEVICE_PROC(InvalidateMappedMemoryRanges);
    LOAD_DEVICE_PROC(GetDeviceMemoryCommitment);
    LOAD_DEVICE_PROC(CreateDescriptorSetLayout);
    LOAD_DEVICE_PROC(DestroyDescriptorSetLayout);
    LOAD_DEVICE_PROC(GetDescriptorSetLayoutSupport);
    LOAD_DEVICE_PROC(CreatePipelineLayout);
    LOAD_DEVICE_PROC(DestroyPipelineLayout);
    LOAD_DEVICE_PROC(CreateShaderModule);
    LOAD_DEVICE_PROC(DestroyShaderModule);
    LOAD_DEVICE_PROC(CreateComputePipelines);
    LOAD_DEVICE_PROC(DestroyPipeline);
    LOAD_DEVICE_PROC(CreateDescriptorUpdateTemplateKHR);
    LOAD_DEVICE_PROC(DestroyDescriptorUpdateTemplateKHR);
    LOAD_DEVICE_PROC(CreateCommandPool);
    LOAD_DEVICE_PROC(DestroyCommandPool);
    LOAD_DEVICE_PROC(AllocateCommandBuffers);
    LOAD_DEVICE_PROC(FreeCommandBuffers);
    LOAD_DEVICE_PROC(ResetCommandPool);
    LOAD_DEVICE_PROC(TrimCommandPool);
    LOAD_DEVICE_PROC(BeginCommandBuffer);
    LOAD_DEVICE_PROC(EndCommandBuffer);
    LOAD_DEVICE_PROC(ResetCommandBuffer);
    LOAD_DEVICE_PROC(CmdBindPipeline);
    LOAD_DEVICE_PROC(CmdPushDescriptorSetKHR);
    LOAD_DEVICE_PROC(CmdPushDescriptorSetWithTemplateKHR);
    LOAD_DEVICE_PROC(CmdPushConstants);
    LOAD_DEVICE_PROC(CmdDispatch);
    LOAD_DEVICE_PROC(CmdCopyBuffer);
    LOAD_DEVICE_PROC(CmdResetQueryPool);
    LOAD_DEVICE_PROC(CmdWriteTimestamp);
    LOAD_DEVICE_PROC(CmdCopyQueryPoolResults);
    LOAD_DEVICE_PROC(CmdPipelineBarrier);
    LOAD_DEVICE_PROC(CreateFence);
    LOAD_DEVICE_PROC(DestroyFence);
    LOAD_DEVICE_PROC(GetFenceStatus);
    LOAD_DEVICE_PROC(ResetFences);
    LOAD_DEVICE_PROC(QueueSubmit);
    LOAD_DEVICE_PROC(QueueWaitIdle);
    LOAD_DEVICE_PROC(DeviceWaitIdle);
    LOAD_DEVICE_PROC(WaitForFences);
    LOAD_DEVICE_PROC(CreatePipelineCache);
    LOAD_DEVICE_PROC(DestroyPipelineCache);
    LOAD_DEVICE_PROC(GetPipelineCacheData);
    LOAD_DEVICE_PROC(MergePipelineCaches);
    LOAD_DEVICE_PROC(CreateQueryPool);
    LOAD_DEVICE_PROC(DestroyQueryPool);
    LOAD_DEVICE_PROC(GetQueryPoolResults);

    VkQueue queue = VK_NULL_HANDLE;
    GetDeviceQueue(device, 0, 0, &queue);
    if (!queue)
    {
        fprintf(stderr, "vkGetDeviceQueue returned a null queue\n");
        return 1;
    }

    VkPipelineCacheCreateInfo pipeline_cache_create_info;
    memset(&pipeline_cache_create_info, 0, sizeof(pipeline_cache_create_info));
    pipeline_cache_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO;
    VkPipelineCache pipeline_caches[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    CHECK_VK(CreatePipelineCache(device, &pipeline_cache_create_info, 0, &pipeline_caches[0]));
    CHECK_VK(CreatePipelineCache(device, &pipeline_cache_create_info, 0, &pipeline_caches[1]));
    size_t pipeline_cache_data_size = 1;
    CHECK_VK(GetPipelineCacheData(device, pipeline_caches[0], &pipeline_cache_data_size, 0));
    if (pipeline_cache_data_size != 0)
    {
        fprintf(stderr, "pipeline cache data is not empty\n");
        return 1;
    }
    CHECK_VK(MergePipelineCaches(device, pipeline_caches[0], 1, &pipeline_caches[1]));

    uint32_t host_memory_type = UINT32_MAX;
    for (uint32_t i = 0; i < memory_properties.memoryTypeCount; i++)
    {
        const VkMemoryPropertyFlags required = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((memory_properties.memoryTypes[i].propertyFlags & required) == required)
        {
            host_memory_type = i;
            break;
        }
    }
    if (host_memory_type == UINT32_MAX)
    {
        fprintf(stderr, "host-visible coherent memory type is missing\n");
        return 1;
    }

    const uint32_t element_count = 127;
    const VkDeviceSize buffer_size = element_count * sizeof(float);
    VkBuffer buffers[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkDeviceMemory memories[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    for (uint32_t i = 0; i < 2; i++)
    {
        VkBufferCreateInfo buffer_create_info;
        memset(&buffer_create_info, 0, sizeof(buffer_create_info));
        buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_create_info.size = buffer_size;
        buffer_create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        CHECK_VK(CreateBuffer(device, &buffer_create_info, 0, &buffers[i]));

        VkMemoryRequirements memory_requirements;
        GetBufferMemoryRequirements(device, buffers[i], &memory_requirements);
        if ((memory_requirements.memoryTypeBits & (1u << host_memory_type)) == 0)
        {
            fprintf(stderr, "host-visible memory type is incompatible with the buffer\n");
            return 1;
        }

        VkMemoryDedicatedRequirements dedicated_requirements;
        memset(&dedicated_requirements, 0, sizeof(dedicated_requirements));
        dedicated_requirements.sType = VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS;
        VkMemoryRequirements2 memory_requirements2;
        memset(&memory_requirements2, 0, sizeof(memory_requirements2));
        memory_requirements2.sType = VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2;
        memory_requirements2.pNext = &dedicated_requirements;
        VkBufferMemoryRequirementsInfo2 requirements_info2;
        memset(&requirements_info2, 0, sizeof(requirements_info2));
        requirements_info2.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2;
        requirements_info2.buffer = buffers[i];
        GetBufferMemoryRequirements2(device, &requirements_info2, &memory_requirements2);
        if (memory_requirements2.memoryRequirements.size != memory_requirements.size
            || memory_requirements2.memoryRequirements.alignment != memory_requirements.alignment
            || memory_requirements2.memoryRequirements.memoryTypeBits != memory_requirements.memoryTypeBits
            || dedicated_requirements.prefersDedicatedAllocation
            || dedicated_requirements.requiresDedicatedAllocation)
        {
            fprintf(stderr, "buffer memory requirements2 mismatch\n");
            return 1;
        }

        VkMemoryAllocateInfo allocate_info;
        memset(&allocate_info, 0, sizeof(allocate_info));
        allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocate_info.allocationSize = memory_requirements.size;
        allocate_info.memoryTypeIndex = host_memory_type;
        CHECK_VK(AllocateMemory(device, &allocate_info, 0, &memories[i]));
        VkDeviceSize committed_memory = 0;
        GetDeviceMemoryCommitment(device, memories[i], &committed_memory);
        if (committed_memory != allocate_info.allocationSize)
        {
            fprintf(stderr, "unexpected committed memory size %llu\n",
                    (unsigned long long)committed_memory);
            return 1;
        }
        if (i == 0)
        {
            CHECK_VK(BindBufferMemory(device, buffers[i], memories[i], 0));
        }
        else
        {
            VkBindBufferMemoryInfo bind_info;
            memset(&bind_info, 0, sizeof(bind_info));
            bind_info.sType = VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO;
            bind_info.buffer = buffers[i];
            bind_info.memory = memories[i];
            CHECK_VK(BindBufferMemory2(device, 1, &bind_info));
        }
    }

    void* mapped = 0;
    CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
    float* input = (float*)mapped;
    for (uint32_t i = 0; i < element_count; i++)
        input[i] = (int)(i % 17) - 8.25f;
    VkMappedMemoryRange input_range;
    memset(&input_range, 0, sizeof(input_range));
    input_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    input_range.memory = memories[0];
    input_range.offset = 0;
    input_range.size = buffer_size;
    CHECK_VK(FlushMappedMemoryRanges(device, 1, &input_range));
    UnmapMemory(device, memories[0]);

    VkDescriptorSetLayoutBinding layout_bindings[2];
    memset(layout_bindings, 0, sizeof(layout_bindings));
    for (uint32_t i = 0; i < 2; i++)
    {
        layout_bindings[i].binding = i;
        layout_bindings[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        layout_bindings[i].descriptorCount = 1;
        layout_bindings[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo descriptor_layout_create_info;
    memset(&descriptor_layout_create_info, 0, sizeof(descriptor_layout_create_info));
    descriptor_layout_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
    descriptor_layout_create_info.flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR;
    descriptor_layout_create_info.bindingCount = 2;
    descriptor_layout_create_info.pBindings = layout_bindings;

    VkDescriptorSetLayoutSupport descriptor_layout_support;
    memset(&descriptor_layout_support, 0, sizeof(descriptor_layout_support));
    descriptor_layout_support.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_SUPPORT;
    GetDescriptorSetLayoutSupport(
        device, &descriptor_layout_create_info, &descriptor_layout_support);
    if (!descriptor_layout_support.supported)
    {
        fprintf(stderr, "push descriptor set layout was reported unsupported\n");
        return 1;
    }
    layout_bindings[0].descriptorCount = 2;
    descriptor_layout_support.supported = VK_TRUE;
    GetDescriptorSetLayoutSupport(
        device, &descriptor_layout_create_info, &descriptor_layout_support);
    if (descriptor_layout_support.supported)
    {
        fprintf(stderr, "unsupported descriptor array layout was accepted\n");
        return 1;
    }
    layout_bindings[0].descriptorCount = 1;

    VkDescriptorSetLayout descriptor_layout = VK_NULL_HANDLE;
    CHECK_VK(CreateDescriptorSetLayout(device, &descriptor_layout_create_info, 0, &descriptor_layout));

    VkPushConstantRange push_constant_range;
    push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    push_constant_range.offset = 0;
    push_constant_range.size = 8;
    VkPipelineLayoutCreateInfo pipeline_layout_create_info;
    memset(&pipeline_layout_create_info, 0, sizeof(pipeline_layout_create_info));
    pipeline_layout_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
    pipeline_layout_create_info.setLayoutCount = 1;
    pipeline_layout_create_info.pSetLayouts = &descriptor_layout;
    pipeline_layout_create_info.pushConstantRangeCount = 1;
    pipeline_layout_create_info.pPushConstantRanges = &push_constant_range;

    VkPipelineLayout pipeline_layout = VK_NULL_HANDLE;
    CHECK_VK(CreatePipelineLayout(device, &pipeline_layout_create_info, 0, &pipeline_layout));

    VkShaderModuleCreateInfo shader_module_create_info;
    memset(&shader_module_create_info, 0, sizeof(shader_module_create_info));
    shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    shader_module_create_info.codeSize = sizeof(shader_spv);
    shader_module_create_info.pCode = shader_spv;

    VkShaderModule shader_module = VK_NULL_HANDLE;
    CHECK_VK(CreateShaderModule(device, &shader_module_create_info, 0, &shader_module));

    VkPipelineShaderStageCreateInfo stage_create_info;
    memset(&stage_create_info, 0, sizeof(stage_create_info));
    stage_create_info.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    stage_create_info.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage_create_info.module = shader_module;
    stage_create_info.pName = "main";
    VkComputePipelineCreateInfo pipeline_create_info;
    memset(&pipeline_create_info, 0, sizeof(pipeline_create_info));
    pipeline_create_info.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
    pipeline_create_info.stage = stage_create_info;
    pipeline_create_info.layout = pipeline_layout;
    pipeline_create_info.basePipelineIndex = -1;

    VkPipeline pipeline = VK_NULL_HANDLE;
    CHECK_VK(CreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_create_info, 0, &pipeline));

    struct DescriptorData
    {
        VkDescriptorBufferInfo input;
        VkDescriptorBufferInfo output;
    };
    VkDescriptorUpdateTemplateEntry template_entries[2];
    memset(template_entries, 0, sizeof(template_entries));
    template_entries[0].dstBinding = 0;
    template_entries[0].descriptorCount = 1;
    template_entries[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    template_entries[0].offset = offsetof(DescriptorData, input);
    template_entries[0].stride = sizeof(VkDescriptorBufferInfo);
    template_entries[1].dstBinding = 1;
    template_entries[1].descriptorCount = 1;
    template_entries[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    template_entries[1].offset = offsetof(DescriptorData, output);
    template_entries[1].stride = sizeof(VkDescriptorBufferInfo);

    VkDescriptorUpdateTemplateCreateInfoKHR template_create_info;
    memset(&template_create_info, 0, sizeof(template_create_info));
    template_create_info.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO_KHR;
    template_create_info.descriptorUpdateEntryCount = 2;
    template_create_info.pDescriptorUpdateEntries = template_entries;
    template_create_info.templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR;
    template_create_info.descriptorSetLayout = descriptor_layout;
    template_create_info.pipelineBindPoint = VK_PIPELINE_BIND_POINT_COMPUTE;
    template_create_info.pipelineLayout = pipeline_layout;
    template_create_info.set = 0;

    VkDescriptorUpdateTemplateKHR descriptor_template = VK_NULL_HANDLE;
    CHECK_VK(CreateDescriptorUpdateTemplateKHR(device, &template_create_info, 0, &descriptor_template));

    VkCommandPoolCreateInfo command_pool_create_info;
    memset(&command_pool_create_info, 0, sizeof(command_pool_create_info));
    command_pool_create_info.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    command_pool_create_info.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    command_pool_create_info.queueFamilyIndex = 0;

    VkCommandPool command_pool = VK_NULL_HANDLE;
    CHECK_VK(CreateCommandPool(device, &command_pool_create_info, 0, &command_pool));

    VkCommandBufferAllocateInfo command_buffer_allocate_info;
    memset(&command_buffer_allocate_info, 0, sizeof(command_buffer_allocate_info));
    command_buffer_allocate_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    command_buffer_allocate_info.commandPool = command_pool;
    command_buffer_allocate_info.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    command_buffer_allocate_info.commandBufferCount = 1;

    VkCommandBuffer command_buffer = VK_NULL_HANDLE;
    CHECK_VK(AllocateCommandBuffers(device, &command_buffer_allocate_info, &command_buffer));
    VkCommandBufferBeginInfo begin_info;
    memset(&begin_info, 0, sizeof(begin_info));
    begin_info.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    begin_info.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);

    DescriptorData descriptor_data;
    descriptor_data.input.buffer = buffers[0];
    descriptor_data.input.offset = 0;
    descriptor_data.input.range = buffer_size;
    descriptor_data.output.buffer = buffers[1];
    descriptor_data.output.offset = 0;
    descriptor_data.output.range = buffer_size;
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);

    struct PushConstants
    {
        uint32_t count;
        float bias;
    };
    PushConstants push_constants;
    push_constants.count = element_count;
    push_constants.bias = 0.5f;
    CmdPushConstants(
        command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, 8, &push_constants);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);

    VkWriteDescriptorSet descriptor_writes[2];
    memset(descriptor_writes, 0, sizeof(descriptor_writes));
    descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[0].dstBinding = 0;
    descriptor_writes[0].descriptorCount = 1;
    descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[0].pBufferInfo = &descriptor_data.input;
    descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    descriptor_writes[1].dstBinding = 1;
    descriptor_writes[1].descriptorCount = 1;
    descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    descriptor_writes[1].pBufferInfo = &descriptor_data.output;
    CmdPushDescriptorSetKHR(
        command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline_layout, 0, 2, descriptor_writes);

    VkMemoryBarrier dispatch_barrier;
    memset(&dispatch_barrier, 0, sizeof(dispatch_barrier));
    dispatch_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    dispatch_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    dispatch_barrier.dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    CmdPipelineBarrier(
        command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &dispatch_barrier, 0, 0, 0, 0);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));

    VkFenceCreateInfo fence_create_info;
    memset(&fence_create_info, 0, sizeof(fence_create_info));
    fence_create_info.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;

    VkFence fence = VK_NULL_HANDLE;
    CHECK_VK(CreateFence(device, &fence_create_info, 0, &fence));
    if (GetFenceStatus(device, fence) != VK_NOT_READY)
    {
        fprintf(stderr, "new fence is not unsignaled\n");
        return 1;
    }

    VkFenceCreateInfo signaled_fence_create_info = fence_create_info;
    signaled_fence_create_info.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VkFence signaled_fence = VK_NULL_HANDLE;
    CHECK_VK(CreateFence(device, &signaled_fence_create_info, 0, &signaled_fence));
    VkFence wait_any_fences[2] = {fence, signaled_fence};
    CHECK_VK(WaitForFences(device, 2, wait_any_fences, VK_FALSE, 0));
    DestroyFence(device, signaled_fence, 0);

    VkSubmitInfo submit_info;
    memset(&submit_info, 0, sizeof(submit_info));
    submit_info.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    submit_info.commandBufferCount = 1;
    submit_info.pCommandBuffers = &command_buffer;
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    VkMappedMemoryRange output_range;
    memset(&output_range, 0, sizeof(output_range));
    output_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    output_range.memory = memories[1];
    output_range.offset = 0;
    output_range.size = buffer_size;
    void* invalidated_output = 0;
    CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &invalidated_output));
    CHECK_VK(InvalidateMappedMemoryRanges(device, 1, &output_range));
    UnmapMemory(device, memories[1]);
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    const auto check_output = [&](float bias, float input_offset) -> int {
        void* output_mapped = 0;
        VkResult map_result = MapMemory(device, memories[1], 0, buffer_size, 0, &output_mapped);
        if (map_result != VK_SUCCESS)
            return fail("MapMemory output", map_result);

        const float* output = (const float*)output_mapped;
        for (uint32_t i = 0; i < element_count; i++)
        {
            const float source = (int)(i % 17) - 8.25f + input_offset;
            const float expected = fabsf(source) + bias;
            if (fabsf(output[i] - expected) > 0.0001f)
            {
                fprintf(stderr, "output mismatch at %u expected=%f actual=%f\n", i, expected, output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);
        return 0;
    };

    if (check_output(push_constants.bias, 0.f) != 0)
        return 1;

    // Dispatch dimensions must stay within the limit reported by Vulkan.
    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
    CmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &push_constants);
    CmdDispatch(command_buffer, physical_device_properties.limits.maxComputeWorkGroupCount[0] + 1, 1, 1);
    if (EndCommandBuffer(command_buffer) != VK_ERROR_INITIALIZATION_FAILED)
    {
        fprintf(stderr, "oversized dispatch was not rejected\n");
        return 1;
    }

    CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
    input = (float*)mapped;
    for (uint32_t i = 0; i < element_count; i++)
        input[i] = (int)(i % 17) - 8.25f + 1.25f;
    UnmapMemory(device, memories[0]);

    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
    push_constants.bias = 1.5f;
    CmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &push_constants);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    if (check_output(push_constants.bias, 1.25f) != 0)
        return 1;

    descriptor_data.output.buffer = buffers[0];
    push_constants.bias = 0.75f;
    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
    CmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &push_constants);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
    const float* inplace_output = (const float*)mapped;
    for (uint32_t i = 0; i < element_count; i++)
    {
        const float source = (int)(i % 17) - 8.25f + 1.25f;
        const float expected = fabsf(source) + push_constants.bias;
        if (fabsf(inplace_output[i] - expected) > 0.0001f)
        {
            fprintf(stderr, "in-place output mismatch at %u expected=%f actual=%f\n",
                    i, expected, inplace_output[i]);
            return 1;
        }
    }
    UnmapMemory(device, memories[0]);

    if (subgroup_operations_supported)
    {
        VkShaderModuleCreateInfo subgroup_shader_module_create_info = shader_module_create_info;
        subgroup_shader_module_create_info.codeSize = sizeof(subgroup_shader_spv);
        subgroup_shader_module_create_info.pCode = subgroup_shader_spv;
        VkShaderModule subgroup_shader_module = VK_NULL_HANDLE;
        CHECK_VK(CreateShaderModule(device, &subgroup_shader_module_create_info, 0, &subgroup_shader_module));

        VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT required_subgroup_size;
        memset(&required_subgroup_size, 0, sizeof(required_subgroup_size));
        required_subgroup_size.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
        required_subgroup_size.requiredSubgroupSize = subgroup_properties.subgroupSize;

        VkPipelineShaderStageCreateInfo subgroup_stage_create_info = stage_create_info;
        subgroup_stage_create_info.pNext = subgroup_size_control_supported ? &required_subgroup_size : 0;
        subgroup_stage_create_info.module = subgroup_shader_module;
        VkComputePipelineCreateInfo subgroup_pipeline_create_info = pipeline_create_info;
        subgroup_pipeline_create_info.stage = subgroup_stage_create_info;
        VkPipeline subgroup_pipeline = VK_NULL_HANDLE;
        CHECK_VK(CreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &subgroup_pipeline_create_info, 0, &subgroup_pipeline));

        descriptor_data.output.buffer = buffers[1];
        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, subgroup_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, 1, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        const float* subgroup_output = (const float*)mapped;
        for (uint32_t i = 0; i < 64; i++)
        {
            const float expected = (float)(subgroup_properties.subgroupSize
                                           + i / subgroup_properties.subgroupSize);
            if (subgroup_output[i] != expected)
            {
                fprintf(stderr, "subgroup output mismatch at %u expected=%f actual=%f\n",
                        i, expected, subgroup_output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, subgroup_pipeline, 0);
        DestroyShaderModule(device, subgroup_shader_module, 0);

        VkShaderModuleCreateInfo feature_shader_module_create_info = shader_module_create_info;
        feature_shader_module_create_info.codeSize = sizeof(subgroup_vote_ballot_quad_shader_spv);
        feature_shader_module_create_info.pCode = subgroup_vote_ballot_quad_shader_spv;
        VkShaderModule feature_shader_module = VK_NULL_HANDLE;
        CHECK_VK(CreateShaderModule(device, &feature_shader_module_create_info, 0, &feature_shader_module));

        VkPipelineShaderStageCreateInfo feature_stage_create_info = stage_create_info;
        feature_stage_create_info.pNext = subgroup_size_control_supported ? &required_subgroup_size : 0;
        feature_stage_create_info.module = feature_shader_module;
        VkComputePipelineCreateInfo feature_pipeline_create_info = pipeline_create_info;
        feature_pipeline_create_info.stage = feature_stage_create_info;
        VkPipeline feature_pipeline = VK_NULL_HANDLE;
        CHECK_VK(CreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &feature_pipeline_create_info, 0, &feature_pipeline));

        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, feature_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, 1, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        const uint32_t* feature_output = (const uint32_t*)mapped;
        for (uint32_t i = 0; i < 64; i++)
        {
            const uint32_t lane = i % subgroup_properties.subgroupSize;
            const uint32_t expected = 3 + subgroup_properties.subgroupSize / 2 * 4
                                      + (lane & ~3u) * 256;
            if (feature_output[i] != expected)
            {
                fprintf(stderr, "subgroup vote/ballot/quad mismatch at %u expected=%u actual=%u\n",
                        i, expected, feature_output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, feature_pipeline, 0);
        DestroyShaderModule(device, feature_shader_module, 0);
    }

    if (subgroup_extended_types_supported)
    {
        VkShaderModuleCreateInfo fp16_subgroup_shader_module_create_info = shader_module_create_info;
        fp16_subgroup_shader_module_create_info.codeSize = sizeof(fp16_subgroup_shader_spv);
        fp16_subgroup_shader_module_create_info.pCode = fp16_subgroup_shader_spv;
        VkShaderModule fp16_subgroup_shader_module = VK_NULL_HANDLE;
        CHECK_VK(CreateShaderModule(device, &fp16_subgroup_shader_module_create_info, 0, &fp16_subgroup_shader_module));

        VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT required_subgroup_size;
        memset(&required_subgroup_size, 0, sizeof(required_subgroup_size));
        required_subgroup_size.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT;
        required_subgroup_size.requiredSubgroupSize = subgroup_properties.subgroupSize;

        VkPipelineShaderStageCreateInfo fp16_subgroup_stage_create_info = stage_create_info;
        fp16_subgroup_stage_create_info.pNext = subgroup_size_control_supported ? &required_subgroup_size : 0;
        fp16_subgroup_stage_create_info.module = fp16_subgroup_shader_module;
        VkComputePipelineCreateInfo fp16_subgroup_pipeline_create_info = pipeline_create_info;
        fp16_subgroup_pipeline_create_info.stage = fp16_subgroup_stage_create_info;
        VkPipeline fp16_subgroup_pipeline = VK_NULL_HANDLE;
        CHECK_VK(CreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &fp16_subgroup_pipeline_create_info, 0, &fp16_subgroup_pipeline));

        descriptor_data.output.buffer = buffers[1];
        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, fp16_subgroup_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, 1, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        const float* fp16_subgroup_output = (const float*)mapped;
        for (uint32_t i = 0; i < 64; i++)
        {
            const float expected = (float)(subgroup_properties.subgroupSize
                                           + i / subgroup_properties.subgroupSize);
            if (fp16_subgroup_output[i] != expected)
            {
                fprintf(stderr, "fp16 subgroup output mismatch at %u expected=%f actual=%f\n",
                        i, expected, fp16_subgroup_output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, fp16_subgroup_pipeline, 0);
        DestroyShaderModule(device, fp16_subgroup_shader_module, 0);
    }

    if (shader_f16_supported)
    {
        static const uint16_t fp16_input_values[4] = {0x0000, 0x3c00, 0x4000, 0xbc00};
        static const uint16_t fp16_expected_values[4] = {0x3c00, 0x4000, 0x4200, 0x0000};
        CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
        uint16_t* fp16_input = (uint16_t*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
            fp16_input[i] = fp16_input_values[i % 4];
        UnmapMemory(device, memories[0]);

        VkShaderModuleCreateInfo fp16_shader_module_create_info = shader_module_create_info;
        fp16_shader_module_create_info.codeSize = sizeof(fp16_shader_spv);
        fp16_shader_module_create_info.pCode = fp16_shader_spv;
        VkShaderModule fp16_shader_module = VK_NULL_HANDLE;
        CHECK_VK(CreateShaderModule(device, &fp16_shader_module_create_info, 0, &fp16_shader_module));

        VkPipelineShaderStageCreateInfo fp16_stage_create_info = stage_create_info;
        fp16_stage_create_info.module = fp16_shader_module;
        VkComputePipelineCreateInfo fp16_pipeline_create_info = pipeline_create_info;
        fp16_pipeline_create_info.stage = fp16_stage_create_info;
        VkPipeline fp16_pipeline = VK_NULL_HANDLE;
        CHECK_VK(CreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &fp16_pipeline_create_info, 0, &fp16_pipeline));

        descriptor_data.output.buffer = buffers[1];
        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, fp16_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, element_count, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        const uint16_t* fp16_output = (const uint16_t*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
        {
            if (fp16_output[i] != fp16_expected_values[i % 4])
            {
                fprintf(stderr, "fp16 output mismatch at %u expected=0x%04x actual=0x%04x\n",
                        i, fp16_expected_values[i % 4], fp16_output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, fp16_pipeline, 0);
        DestroyShaderModule(device, fp16_shader_module, 0);

        fp16_shader_module_create_info.codeSize = sizeof(fp16_specialization_shader_spv);
        fp16_shader_module_create_info.pCode = fp16_specialization_shader_spv;
        CHECK_VK(CreateShaderModule(device, &fp16_shader_module_create_info, 0, &fp16_shader_module));

        const float specialized_fp16_value = 2.5f;
        VkSpecializationMapEntry fp16_specialization_entry;
        memset(&fp16_specialization_entry, 0, sizeof(fp16_specialization_entry));
        fp16_specialization_entry.constantID = 0;
        fp16_specialization_entry.size = sizeof(specialized_fp16_value);
        VkSpecializationInfo fp16_specialization_info;
        memset(&fp16_specialization_info, 0, sizeof(fp16_specialization_info));
        fp16_specialization_info.mapEntryCount = 1;
        fp16_specialization_info.pMapEntries = &fp16_specialization_entry;
        fp16_specialization_info.dataSize = sizeof(specialized_fp16_value);
        fp16_specialization_info.pData = &specialized_fp16_value;

        fp16_stage_create_info.module = fp16_shader_module;
        fp16_stage_create_info.pSpecializationInfo = &fp16_specialization_info;
        fp16_pipeline_create_info.stage = fp16_stage_create_info;
        CHECK_VK(CreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &fp16_pipeline_create_info, 0, &fp16_pipeline));

        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, fp16_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, element_count, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        fp16_output = (const uint16_t*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
        {
            if (fp16_output[i] != 0x4100)
            {
                fprintf(stderr, "fp16 specialization output mismatch at %u expected=0x4100 actual=0x%04x\n",
                        i, fp16_output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, fp16_pipeline, 0);
        DestroyShaderModule(device, fp16_shader_module, 0);

        fp16_shader_module_create_info.codeSize = sizeof(fp16_overflow_shader_spv);
        fp16_shader_module_create_info.pCode = fp16_overflow_shader_spv;
        CHECK_VK(CreateShaderModule(device, &fp16_shader_module_create_info, 0, &fp16_shader_module));

        fp16_stage_create_info.module = fp16_shader_module;
        fp16_stage_create_info.pSpecializationInfo = 0;
        fp16_pipeline_create_info.stage = fp16_stage_create_info;
        CHECK_VK(CreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &fp16_pipeline_create_info, 0, &fp16_pipeline));

        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, fp16_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, element_count, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        fp16_output = (const uint16_t*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
        {
            if (fp16_output[i] != 0xfbff)
            {
                fprintf(stderr, "fp16 overflow output mismatch at %u expected=0xfbff actual=0x%04x\n",
                        i, fp16_output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, fp16_pipeline, 0);
        DestroyShaderModule(device, fp16_shader_module, 0);

        CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
        uint16_t* storage_int16_input = (uint16_t*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
            storage_int16_input[i] = (uint16_t)(i * 997u + 0x7ff0u);
        UnmapMemory(device, memories[0]);

        VkShaderModuleCreateInfo storage_int16_shader_module_create_info = shader_module_create_info;
        storage_int16_shader_module_create_info.codeSize = sizeof(storage_int16_shader_spv);
        storage_int16_shader_module_create_info.pCode = storage_int16_shader_spv;
        VkShaderModule storage_int16_shader_module = VK_NULL_HANDLE;
        CHECK_VK(CreateShaderModule(device, &storage_int16_shader_module_create_info, 0, &storage_int16_shader_module));

        VkPipelineShaderStageCreateInfo storage_int16_stage_create_info = stage_create_info;
        storage_int16_stage_create_info.module = storage_int16_shader_module;
        VkComputePipelineCreateInfo storage_int16_pipeline_create_info = pipeline_create_info;
        storage_int16_pipeline_create_info.stage = storage_int16_stage_create_info;
        VkPipeline storage_int16_pipeline = VK_NULL_HANDLE;
        CHECK_VK(CreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &storage_int16_pipeline_create_info, 0, &storage_int16_pipeline));

        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, storage_int16_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, element_count, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        const uint16_t* storage_int16_output = (const uint16_t*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
        {
            const uint16_t input = (uint16_t)(i * 997u + 0x7ff0u);
            const uint16_t expected = (uint16_t)(input + 0x1234u);
            if (storage_int16_output[i] != expected)
            {
                fprintf(stderr, "16-bit integer storage mismatch at %u expected=0x%04x actual=0x%04x\n",
                        i, expected, storage_int16_output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, storage_int16_pipeline, 0);
        DestroyShaderModule(device, storage_int16_shader_module, 0);

        CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
        uint16_t* storage_int16_pack_input = (uint16_t*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
            storage_int16_pack_input[i] = (uint16_t)(i * 1453u + 0x4ff0u);
        UnmapMemory(device, memories[0]);

        storage_int16_shader_module_create_info.codeSize = sizeof(storage_int16_pack_shader_spv);
        storage_int16_shader_module_create_info.pCode = storage_int16_pack_shader_spv;
        CHECK_VK(CreateShaderModule(device, &storage_int16_shader_module_create_info, 0, &storage_int16_shader_module));
        storage_int16_stage_create_info.module = storage_int16_shader_module;
        storage_int16_pipeline_create_info.stage = storage_int16_stage_create_info;
        CHECK_VK(CreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &storage_int16_pipeline_create_info, 0, &storage_int16_pipeline));

        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, storage_int16_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, element_count / 2, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        const uint16_t* storage_int16_pack_output = (const uint16_t*)mapped;
        const uint32_t storage_int16_pack_element_count = element_count / 2 * 2;
        for (uint32_t i = 0; i < storage_int16_pack_element_count; i++)
        {
            const uint16_t input = (uint16_t)(i * 1453u + 0x4ff0u);
            const uint16_t expected = (uint16_t)(input ^ ((i & 1) == 0 ? 0x5678u : 0x1234u));
            if (storage_int16_pack_output[i] != expected)
            {
                fprintf(stderr, "packed 16-bit integer storage mismatch at %u expected=0x%04x actual=0x%04x\n",
                        i, expected, storage_int16_pack_output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, storage_int16_pipeline, 0);
        DestroyShaderModule(device, storage_int16_shader_module, 0);

        CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
        int16_t* storage_int16_signed_input = (int16_t*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
            storage_int16_signed_input[i] = (int16_t)(i * 1777u + 0x6ff0u);
        UnmapMemory(device, memories[0]);

        storage_int16_shader_module_create_info.codeSize = sizeof(storage_int16_signed_shader_spv);
        storage_int16_shader_module_create_info.pCode = storage_int16_signed_shader_spv;
        CHECK_VK(CreateShaderModule(device, &storage_int16_shader_module_create_info, 0, &storage_int16_shader_module));
        storage_int16_stage_create_info.module = storage_int16_shader_module;
        storage_int16_pipeline_create_info.stage = storage_int16_stage_create_info;
        CHECK_VK(CreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &storage_int16_pipeline_create_info, 0, &storage_int16_pipeline));

        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, storage_int16_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, element_count, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        const int16_t* storage_int16_signed_output = (const int16_t*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
        {
            const int16_t input = (int16_t)(i * 1777u + 0x6ff0u);
            const int16_t expected = (int16_t)((int)input - 12345);
            if (storage_int16_signed_output[i] != expected)
            {
                fprintf(stderr, "signed 16-bit integer storage mismatch at %u expected=%d actual=%d\n",
                        i, expected, storage_int16_signed_output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, storage_int16_pipeline, 0);
        DestroyShaderModule(device, storage_int16_shader_module, 0);

        CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
        int16_t* storage_int16_vector_input = (int16_t*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
            storage_int16_vector_input[i] = (int16_t)(i * 2341u + 0x5ff0u);
        UnmapMemory(device, memories[0]);

        storage_int16_shader_module_create_info.codeSize = sizeof(storage_int16_vector_shader_spv);
        storage_int16_shader_module_create_info.pCode = storage_int16_vector_shader_spv;
        CHECK_VK(CreateShaderModule(device, &storage_int16_shader_module_create_info, 0, &storage_int16_shader_module));
        storage_int16_stage_create_info.module = storage_int16_shader_module;
        storage_int16_pipeline_create_info.stage = storage_int16_stage_create_info;
        CHECK_VK(CreateComputePipelines(
            device, VK_NULL_HANDLE, 1, &storage_int16_pipeline_create_info, 0, &storage_int16_pipeline));

        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, storage_int16_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, element_count / 4, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        const int16_t* storage_int16_vector_output = (const int16_t*)mapped;
        const uint32_t storage_int16_vector_element_count = element_count / 4 * 4;
        for (uint32_t i = 0; i < storage_int16_vector_element_count; i++)
        {
            const int16_t input = (int16_t)(i * 2341u + 0x5ff0u);
            const int16_t expected = (int16_t)((int)input + (int)(i % 4) + 1);
            if (storage_int16_vector_output[i] != expected)
            {
                fprintf(stderr, "vector 16-bit integer storage mismatch at %u expected=%d actual=%d\n",
                        i, expected, storage_int16_vector_output[i]);
                UnmapMemory(device, memories[1]);
                return 1;
            }
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, storage_int16_pipeline, 0);
        DestroyShaderModule(device, storage_int16_shader_module, 0);

        CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
        float* restored_input = (float*)mapped;
        for (uint32_t i = 0; i < element_count; i++)
        {
            const float source = (int)(i % 17) - 8.25f + 1.25f;
            restored_input[i] = fabsf(source) + push_constants.bias;
        }
        UnmapMemory(device, memories[0]);
    }

    const uint32_t uniform_element_count = 128;
    const VkDeviceSize uniform_stride = 16;
    const VkDeviceSize uniform_buffer_size = uniform_element_count * uniform_stride;
    VkBufferCreateInfo uniform_buffer_create_info;
    memset(&uniform_buffer_create_info, 0, sizeof(uniform_buffer_create_info));
    uniform_buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    uniform_buffer_create_info.size = uniform_buffer_size;
    uniform_buffer_create_info.usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    uniform_buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer uniform_buffer = VK_NULL_HANDLE;
    CHECK_VK(CreateBuffer(device, &uniform_buffer_create_info, 0, &uniform_buffer));

    VkMemoryRequirements uniform_memory_requirements;
    GetBufferMemoryRequirements(device, uniform_buffer, &uniform_memory_requirements);
    VkMemoryAllocateInfo uniform_memory_allocate_info;
    memset(&uniform_memory_allocate_info, 0, sizeof(uniform_memory_allocate_info));
    uniform_memory_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    uniform_memory_allocate_info.allocationSize = uniform_memory_requirements.size;
    uniform_memory_allocate_info.memoryTypeIndex = host_memory_type;
    VkDeviceMemory uniform_memory = VK_NULL_HANDLE;
    CHECK_VK(AllocateMemory(device, &uniform_memory_allocate_info, 0, &uniform_memory));
    CHECK_VK(BindBufferMemory(device, uniform_buffer, uniform_memory, 0));

    void* uniform_mapped = 0;
    CHECK_VK(MapMemory(device, uniform_memory, 0, uniform_buffer_size, 0, &uniform_mapped));
    memset(uniform_mapped, 0, uniform_buffer_size);
    float* uniform_values = (float*)uniform_mapped;
    for (uint32_t i = 0; i < element_count; i++)
        uniform_values[i * uniform_stride / sizeof(float)] = (int)(i % 13) - 6.5f;
    VkMappedMemoryRange uniform_range;
    memset(&uniform_range, 0, sizeof(uniform_range));
    uniform_range.sType = VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE;
    uniform_range.memory = uniform_memory;
    uniform_range.offset = 0;
    uniform_range.size = uniform_buffer_size;
    CHECK_VK(FlushMappedMemoryRanges(device, 1, &uniform_range));
    UnmapMemory(device, uniform_memory);

    VkDescriptorSetLayoutBinding uniform_layout_bindings[2] = {
        layout_bindings[0], layout_bindings[1]};
    uniform_layout_bindings[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    VkDescriptorSetLayoutCreateInfo uniform_descriptor_layout_create_info = descriptor_layout_create_info;
    uniform_descriptor_layout_create_info.pBindings = uniform_layout_bindings;
    VkDescriptorSetLayout uniform_descriptor_layout = VK_NULL_HANDLE;
    CHECK_VK(CreateDescriptorSetLayout(
        device, &uniform_descriptor_layout_create_info, 0, &uniform_descriptor_layout));

    VkPipelineLayoutCreateInfo uniform_pipeline_layout_create_info = pipeline_layout_create_info;
    uniform_pipeline_layout_create_info.pSetLayouts = &uniform_descriptor_layout;
    VkPipelineLayout uniform_pipeline_layout = VK_NULL_HANDLE;
    CHECK_VK(CreatePipelineLayout(device, &uniform_pipeline_layout_create_info, 0, &uniform_pipeline_layout));

    std::vector<uint32_t> uniform_shader_spirv = make_uniform_input_shader_spv();
    if (uniform_shader_spirv.empty())
    {
        fprintf(stderr, "failed to derive uniform input shader\n");
        return 1;
    }
    VkShaderModuleCreateInfo uniform_shader_module_create_info = shader_module_create_info;
    uniform_shader_module_create_info.codeSize = uniform_shader_spirv.size() * sizeof(uint32_t);
    uniform_shader_module_create_info.pCode = uniform_shader_spirv.data();
    VkShaderModule uniform_shader_module = VK_NULL_HANDLE;
    CHECK_VK(CreateShaderModule(device, &uniform_shader_module_create_info, 0, &uniform_shader_module));

    VkPipelineShaderStageCreateInfo uniform_stage_create_info = stage_create_info;
    uniform_stage_create_info.module = uniform_shader_module;
    VkComputePipelineCreateInfo uniform_pipeline_create_info = pipeline_create_info;
    uniform_pipeline_create_info.stage = uniform_stage_create_info;
    uniform_pipeline_create_info.layout = uniform_pipeline_layout;
    VkPipeline uniform_pipeline = VK_NULL_HANDLE;
    CHECK_VK(CreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &uniform_pipeline_create_info, 0, &uniform_pipeline));

    VkDescriptorBufferInfo uniform_input_info;
    uniform_input_info.buffer = uniform_buffer;
    uniform_input_info.offset = 0;
    uniform_input_info.range = uniform_buffer_size;
    VkDescriptorBufferInfo uniform_output_info;
    uniform_output_info.buffer = buffers[1];
    uniform_output_info.offset = 0;
    uniform_output_info.range = buffer_size;
    VkWriteDescriptorSet uniform_descriptor_writes[2];
    memset(uniform_descriptor_writes, 0, sizeof(uniform_descriptor_writes));
    uniform_descriptor_writes[0].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    uniform_descriptor_writes[0].dstBinding = 0;
    uniform_descriptor_writes[0].descriptorCount = 1;
    uniform_descriptor_writes[0].descriptorType = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    uniform_descriptor_writes[0].pBufferInfo = &uniform_input_info;
    uniform_descriptor_writes[1].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
    uniform_descriptor_writes[1].dstBinding = 1;
    uniform_descriptor_writes[1].descriptorCount = 1;
    uniform_descriptor_writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    uniform_descriptor_writes[1].pBufferInfo = &uniform_output_info;

    push_constants.bias = 2.25f;
    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, uniform_pipeline);
    CmdPushDescriptorSetKHR(
        command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, uniform_pipeline_layout,
        0, 2, uniform_descriptor_writes);
    CmdPushConstants(
        command_buffer, uniform_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(push_constants), &push_constants);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
    const float* uniform_output = (const float*)mapped;
    for (uint32_t i = 0; i < element_count; i++)
    {
        const float source = (int)(i % 13) - 6.5f;
        const float expected = fabsf(source) + push_constants.bias;
        if (fabsf(uniform_output[i] - expected) > 0.0001f)
        {
            fprintf(stderr, "uniform input output mismatch at %u expected=%f actual=%f\n",
                    i, expected, uniform_output[i]);
            return 1;
        }
    }
    UnmapMemory(device, memories[1]);

    CHECK_VK(ResetCommandBuffer(command_buffer, VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT));
    DestroyPipeline(device, uniform_pipeline, 0);
    DestroyShaderModule(device, uniform_shader_module, 0);
    DestroyPipelineLayout(device, uniform_pipeline_layout, 0);
    DestroyDescriptorSetLayout(device, uniform_descriptor_layout, 0);
    DestroyBuffer(device, uniform_buffer, 0);
    FreeMemory(device, uniform_memory, 0);

    VkShaderModuleCreateInfo specialization_shader_module_create_info;
    memset(&specialization_shader_module_create_info, 0, sizeof(specialization_shader_module_create_info));
    specialization_shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    specialization_shader_module_create_info.codeSize = sizeof(specialization_shader_spv);
    specialization_shader_module_create_info.pCode = specialization_shader_spv;
    VkShaderModule specialization_shader_module = VK_NULL_HANDLE;
    CHECK_VK(CreateShaderModule(
        device, &specialization_shader_module_create_info, 0, &specialization_shader_module));

    int use_alternative = 0;
    VkSpecializationMapEntry specialization_map_entry;
    specialization_map_entry.constantID = 0;
    specialization_map_entry.offset = 0;
    specialization_map_entry.size = sizeof(use_alternative);
    VkSpecializationInfo specialization_info;
    specialization_info.mapEntryCount = 1;
    specialization_info.pMapEntries = &specialization_map_entry;
    specialization_info.dataSize = sizeof(use_alternative);
    specialization_info.pData = &use_alternative;

    stage_create_info.module = specialization_shader_module;
    stage_create_info.pSpecializationInfo = &specialization_info;
    pipeline_create_info.stage = stage_create_info;

    VkPipeline specialization_pipeline = VK_NULL_HANDLE;
    CHECK_VK(CreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &pipeline_create_info, 0, &specialization_pipeline));

    descriptor_data.output.buffer = buffers[1];
    push_constants.bias = 2.f;
    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, specialization_pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
    CmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 8, &push_constants);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
    const float* output = (const float*)mapped;
    for (uint32_t i = 0; i < element_count; i++)
    {
        const float original = (int)(i % 17) - 8.25f + 1.25f;
        const float expected = fabsf(original) + 0.75f + push_constants.bias;
        if (fabsf(output[i] - expected) > 0.0001f)
        {
            fprintf(stderr, "specialization output mismatch at %u expected=%f actual=%f\n",
                    i, expected, output[i]);
            return 1;
        }
    }
    UnmapMemory(device, memories[1]);

    VkPushConstantRange large_push_constant_range;
    large_push_constant_range.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    large_push_constant_range.offset = 0;
    large_push_constant_range.size = 84;
    pipeline_layout_create_info.pPushConstantRanges = &large_push_constant_range;
    VkPipelineLayout large_pipeline_layout = VK_NULL_HANDLE;
    CHECK_VK(CreatePipelineLayout(device, &pipeline_layout_create_info, 0, &large_pipeline_layout));

    VkShaderModuleCreateInfo large_shader_module_create_info;
    memset(&large_shader_module_create_info, 0, sizeof(large_shader_module_create_info));
    large_shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    large_shader_module_create_info.codeSize = sizeof(large_push_shader_spv);
    large_shader_module_create_info.pCode = large_push_shader_spv;
    VkShaderModule large_shader_module = VK_NULL_HANDLE;
    CHECK_VK(CreateShaderModule(device, &large_shader_module_create_info, 0, &large_shader_module));

    stage_create_info.module = large_shader_module;
    stage_create_info.pSpecializationInfo = 0;
    pipeline_create_info.stage = stage_create_info;
    pipeline_create_info.layout = large_pipeline_layout;
    VkPipeline large_pipeline = VK_NULL_HANDLE;
    CHECK_VK(CreateComputePipelines(device, VK_NULL_HANDLE, 1, &pipeline_create_info, 0, &large_pipeline));

    template_create_info.pipelineLayout = large_pipeline_layout;
    VkDescriptorUpdateTemplateKHR large_descriptor_template = VK_NULL_HANDLE;
    CHECK_VK(CreateDescriptorUpdateTemplateKHR(
        device, &template_create_info, 0, &large_descriptor_template));

    struct LargePushConstants
    {
        float values[20];
        uint32_t count;
    };
    static_assert(sizeof(LargePushConstants) == 84, "unexpected large push constant layout");
    LargePushConstants large_push_constants;
    for (uint32_t i = 0; i < 20; i++)
        large_push_constants.values[i] = (float)i * 0.25f;
    large_push_constants.values[19] = -3.5f;
    large_push_constants.count = element_count;

    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, large_pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, large_descriptor_template, large_pipeline_layout, 0, &descriptor_data);
    CmdPushConstants(
        command_buffer, large_pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(large_push_constants), &large_push_constants);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
    output = (const float*)mapped;
    for (uint32_t i = 0; i < element_count; i++)
    {
        const float original = (int)(i % 17) - 8.25f + 1.25f;
        const float input_value = fabsf(original) + 0.75f;
        const float expected = input_value + large_push_constants.values[19];
        if (fabsf(output[i] - expected) > 0.0001f)
        {
            fprintf(stderr, "large push output mismatch at %u expected=%f actual=%f\n",
                    i, expected, output[i]);
            return 1;
        }
    }
    UnmapMemory(device, memories[1]);

    VkShaderModuleCreateInfo writable_shader_module_create_info;
    memset(&writable_shader_module_create_info, 0, sizeof(writable_shader_module_create_info));
    writable_shader_module_create_info.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
    writable_shader_module_create_info.codeSize = sizeof(writable_alias_shader_spv);
    writable_shader_module_create_info.pCode = writable_alias_shader_spv;
    VkShaderModule writable_shader_module = VK_NULL_HANDLE;
    CHECK_VK(CreateShaderModule(
        device, &writable_shader_module_create_info, 0, &writable_shader_module));

    stage_create_info.module = writable_shader_module;
    pipeline_create_info.stage = stage_create_info;
    pipeline_create_info.layout = pipeline_layout;
    VkPipeline writable_pipeline = VK_NULL_HANDLE;
    CHECK_VK(CreateComputePipelines(
        device, VK_NULL_HANDLE, 1, &pipeline_create_info, 0, &writable_pipeline));

    VkBuffer alias_buffers[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
    VkBufferCreateInfo alias_buffer_create_info;
    memset(&alias_buffer_create_info, 0, sizeof(alias_buffer_create_info));
    alias_buffer_create_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    alias_buffer_create_info.size = buffer_size;
    alias_buffer_create_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    alias_buffer_create_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    CHECK_VK(CreateBuffer(device, &alias_buffer_create_info, 0, &alias_buffers[0]));
    CHECK_VK(CreateBuffer(device, &alias_buffer_create_info, 0, &alias_buffers[1]));

    VkMemoryRequirements alias_memory_requirements;
    GetBufferMemoryRequirements(device, alias_buffers[0], &alias_memory_requirements);
    const VkDeviceSize alias_second_offset = (alias_memory_requirements.size + alias_memory_requirements.alignment - 1)
                                             / alias_memory_requirements.alignment * alias_memory_requirements.alignment;
    VkMemoryAllocateInfo alias_allocate_info;
    memset(&alias_allocate_info, 0, sizeof(alias_allocate_info));
    alias_allocate_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alias_allocate_info.allocationSize = alias_second_offset + alias_memory_requirements.size;
    alias_allocate_info.memoryTypeIndex = host_memory_type;
    VkDeviceMemory alias_memory = VK_NULL_HANDLE;
    CHECK_VK(AllocateMemory(device, &alias_allocate_info, 0, &alias_memory));
    CHECK_VK(BindBufferMemory(device, alias_buffers[0], alias_memory, 0));
    CHECK_VK(BindBufferMemory(device, alias_buffers[1], alias_memory, alias_second_offset));

    CHECK_VK(MapMemory(device, alias_memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    float* alias_first = (float*)mapped;
    float* alias_second = (float*)((unsigned char*)mapped + alias_second_offset);
    for (uint32_t i = 0; i < element_count; i++)
    {
        alias_first[i] = -(float)(i + 1);
        alias_second[i] = -1.f;
    }
    UnmapMemory(device, alias_memory);

    descriptor_data.input.buffer = alias_buffers[0];
    descriptor_data.input.offset = 0;
    descriptor_data.input.range = buffer_size;
    descriptor_data.output.buffer = alias_buffers[1];
    descriptor_data.output.offset = 0;
    descriptor_data.output.range = buffer_size;
    push_constants.count = element_count;
    push_constants.bias = 0.25f;
    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
    CmdPushConstants(
        command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(push_constants), &push_constants);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    CHECK_VK(MapMemory(device, alias_memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    alias_first = (float*)mapped;
    alias_second = (float*)((unsigned char*)mapped + alias_second_offset);
    for (uint32_t i = 0; i < element_count; i++)
    {
        const float expected = (float)(i + 1) + push_constants.bias;
        if (alias_second[i] != expected)
        {
            fprintf(stderr, "disjoint readonly/writable alias mismatch at %u\n", i);
            return 1;
        }
    }

    for (uint32_t i = 0; i < element_count; i++)
    {
        alias_first[i] = (float)i;
        alias_second[i] = (float)i * 2.f;
    }
    UnmapMemory(device, alias_memory);

    descriptor_data.input.buffer = alias_buffers[0];
    descriptor_data.input.offset = 0;
    descriptor_data.input.range = buffer_size;
    descriptor_data.output.buffer = alias_buffers[1];
    descriptor_data.output.offset = 0;
    descriptor_data.output.range = buffer_size;
    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, writable_pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
    CmdPushConstants(
        command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(element_count), &element_count);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    CHECK_VK(MapMemory(device, alias_memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    alias_first = (float*)mapped;
    alias_second = (float*)((unsigned char*)mapped + alias_second_offset);
    for (uint32_t i = 0; i < element_count; i++)
    {
        if (alias_first[i] != (float)i + 1.f || alias_second[i] != (float)i * 2.f + 2.f)
        {
            fprintf(stderr, "disjoint writable alias mismatch at %u\n", i);
            return 1;
        }
    }
    UnmapMemory(device, alias_memory);

    // WebGPU rejects a native copy when both ends are the same GPUBuffer,
    // even when the ranges are disjoint. nikki implements this legal
    // Vulkan copy with its embedded single-binding WGSL copy shader.
    CHECK_VK(MapMemory(device, alias_memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    alias_first = (float*)mapped;
    alias_second = (float*)((unsigned char*)mapped + alias_second_offset);
    for (uint32_t i = 0; i < element_count; i++)
    {
        alias_first[i] = 3000.f + (float)i;
        alias_second[i] = -1.f;
    }
    UnmapMemory(device, alias_memory);

    const VkDeviceSize first_same_memory_copy_size = buffer_size / 2 / 4 * 4;
    VkBufferCopy same_memory_copy_regions[2];
    same_memory_copy_regions[0].srcOffset = 0;
    same_memory_copy_regions[0].dstOffset = 0;
    same_memory_copy_regions[0].size = first_same_memory_copy_size;
    same_memory_copy_regions[1].srcOffset = first_same_memory_copy_size;
    same_memory_copy_regions[1].dstOffset = first_same_memory_copy_size;
    same_memory_copy_regions[1].size = buffer_size - first_same_memory_copy_size;
    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdCopyBuffer(command_buffer, alias_buffers[0], alias_buffers[1], 2, same_memory_copy_regions);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    CHECK_VK(MapMemory(device, alias_memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    alias_second = (float*)((unsigned char*)mapped + alias_second_offset);
    for (uint32_t i = 0; i < element_count; i++)
    {
        if (alias_second[i] != 3000.f + (float)i)
        {
            fprintf(stderr, "same-memory vkCmdCopyBuffer mismatch at %u\n", i);
            return 1;
        }
    }
    UnmapMemory(device, alias_memory);

    // A readback for the first bound range must not discard pending host writes
    // in a disjoint range of the same VkDeviceMemory allocation.
    CHECK_VK(MapMemory(device, alias_memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    alias_first = (float*)mapped;
    alias_second = (float*)((unsigned char*)mapped + alias_second_offset);
    for (uint32_t i = 0; i < element_count; i++)
    {
        alias_first[i] = -(float)i;
        alias_second[i] = 9000.f + (float)i;
    }
    UnmapMemory(device, alias_memory);

    descriptor_data.input.buffer = buffers[0];
    descriptor_data.output.buffer = alias_buffers[0];
    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
    CmdPushConstants(
        command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(push_constants), &push_constants);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    VkBufferCopy preserved_dirty_copy;
    preserved_dirty_copy.srcOffset = 0;
    preserved_dirty_copy.dstOffset = 0;
    preserved_dirty_copy.size = buffer_size;
    CmdCopyBuffer(command_buffer, alias_buffers[1], buffers[0], 1, &preserved_dirty_copy);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

    CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
    const float* preserved_dirty_output = (const float*)mapped;
    for (uint32_t i = 0; i < element_count; i++)
    {
        if (preserved_dirty_output[i] != 9000.f + (float)i)
        {
            fprintf(stderr, "disjoint host dirty range was lost at %u\n", i);
            return 1;
        }
    }
    UnmapMemory(device, memories[0]);

    VkBuffer overlapping_buffer = VK_NULL_HANDLE;
    CHECK_VK(CreateBuffer(device, &alias_buffer_create_info, 0, &overlapping_buffer));
    CHECK_VK(BindBufferMemory(device, overlapping_buffer, alias_memory, 0));

    // Read -> write -> read through the same storage reference must observe
    // the write when the dynamic indices match, for scalar and vec2 loads.
    // Different descriptor bindings are not a same-invocation visibility
    // oracle: these fixtures deliberately keep data and output disjoint.
    const uint32_t* read_write_shaders[] = {buffer_read_write_scalar_spv, buffer_read_write_vec2_spv};
    const size_t read_write_shader_sizes[] = {sizeof(buffer_read_write_scalar_spv), sizeof(buffer_read_write_vec2_spv)};
    for (uint32_t variant = 0; variant < 2; variant++)
    {
        VkShaderModuleCreateInfo access_shader_info = shader_module_create_info;
        access_shader_info.codeSize = read_write_shader_sizes[variant];
        access_shader_info.pCode = read_write_shaders[variant];
        VkShaderModule access_shader = VK_NULL_HANDLE;
        CHECK_VK(CreateShaderModule(device, &access_shader_info, 0, &access_shader));
        VkComputePipelineCreateInfo access_pipeline_info = pipeline_create_info;
        access_pipeline_info.stage.module = access_shader;
        VkPipeline access_pipeline = VK_NULL_HANDLE;
        CHECK_VK(CreateComputePipelines(device, VK_NULL_HANDLE, 1, &access_pipeline_info, 0, &access_pipeline));

        for (uint32_t read_index = 0; read_index < 2; read_index++)
        {
            for (uint32_t write_index = 0; write_index < 2; write_index++)
            {
                CHECK_VK(MapMemory(device, alias_memory, 0, VK_WHOLE_SIZE, 0, &mapped));
                alias_first = (float*)mapped;
                alias_second = (float*)((unsigned char*)mapped + alias_second_offset);
                for (uint32_t i = 0; i < element_count; i++)
                {
                    alias_first[i] = i < 8 ? 32.f + (float)i : -17.f;
                    alias_second[i] = -33.f;
                }
                UnmapMemory(device, alias_memory);

                descriptor_data.input.buffer = alias_buffers[0];
                descriptor_data.input.range = 8 * sizeof(float);
                descriptor_data.output.buffer = alias_buffers[1];
                descriptor_data.output.range = 8 * sizeof(float);
                const uint32_t indices[] = {read_index, write_index};
                CHECK_VK(ResetCommandBuffer(command_buffer, 0));
                CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
                CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, access_pipeline);
                CmdPushDescriptorSetWithTemplateKHR(
                    command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
                CmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                                 0, sizeof(indices), indices);
                CmdDispatch(command_buffer, 1, 1, 1);
                CHECK_VK(EndCommandBuffer(command_buffer));
                CHECK_VK(ResetFences(device, 1, &fence));
                CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
                CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

                CHECK_VK(MapMemory(device, alias_memory, 0, VK_WHOLE_SIZE, 0, &mapped));
                alias_first = (float*)mapped;
                alias_second = (float*)((unsigned char*)mapped + alias_second_offset);
                for (uint32_t i = 0; i < element_count; i++)
                {
                    float expected_data = i < 8 ? 32.f + (float)i : -17.f;
                    if (i >= write_index * 4 && i < write_index * 4 + 4)
                        expected_data = (float)(i % 4 + 1);
                    float expected_output = -33.f;
                    if (i < 8)
                        expected_output = i >= 4 && read_index == write_index
                                              ? (float)(i % 4 + 1)
                                              : 32.f + (float)(read_index * 4 + i % 4);
                    if (alias_first[i] != expected_data || alias_second[i] != expected_output)
                    {
                        fprintf(stderr, "buffer read/write mismatch variant=%u read=%u write=%u at %u data=%f expected=%f output=%f expected=%f\n",
                                variant, read_index, write_index, i, alias_first[i], expected_data,
                                alias_second[i], expected_output);
                        UnmapMemory(device, alias_memory);
                        return 1;
                    }
                }
                UnmapMemory(device, alias_memory);
            }
        }
        DestroyPipeline(device, access_pipeline, 0);
        DestroyShaderModule(device, access_shader, 0);
    }

    // A write via one VkBuffer becomes visible through an overlapping buffer
    // in a later dispatch after an API memory barrier. Adjacent scalar loads
    // and the final unpaired float use an exact 127-float descriptor range.
    VkShaderModuleCreateInfo tail_shader_info = shader_module_create_info;
    tail_shader_info.codeSize = sizeof(buffer_load_tail_spv);
    tail_shader_info.pCode = buffer_load_tail_spv;
    VkShaderModule tail_shader = VK_NULL_HANDLE;
    CHECK_VK(CreateShaderModule(device, &tail_shader_info, 0, &tail_shader));
    VkComputePipelineCreateInfo tail_pipeline_info = pipeline_create_info;
    tail_pipeline_info.stage.module = tail_shader;
    VkPipeline tail_pipeline = VK_NULL_HANDLE;
    CHECK_VK(CreateComputePipelines(device, VK_NULL_HANDLE, 1, &tail_pipeline_info, 0, &tail_pipeline));

    CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &mapped));
    for (uint32_t i = 0; i < element_count; i++)
        ((float*)mapped)[i] = -(float)i - 0.25f;
    UnmapMemory(device, memories[0]);
    descriptor_data.input.buffer = buffers[0];
    descriptor_data.input.range = buffer_size;
    descriptor_data.output.buffer = alias_buffers[0];
    descriptor_data.output.range = buffer_size;
    push_constants.count = element_count;
    push_constants.bias = 0.25f;
    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
    CmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                     0, sizeof(push_constants), &push_constants);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    dispatch_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    dispatch_barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
    CmdPipelineBarrier(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &dispatch_barrier, 0, 0, 0, 0);
    descriptor_data.input.buffer = overlapping_buffer;
    descriptor_data.output.buffer = alias_buffers[1];
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, tail_pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
    CmdDispatch(command_buffer, 2, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    CHECK_VK(MapMemory(device, alias_memory, 0, VK_WHOLE_SIZE, 0, &mapped));
    alias_first = (float*)mapped;
    alias_second = (float*)((unsigned char*)mapped + alias_second_offset);
    for (uint32_t i = 0; i < element_count; i++)
    {
        const float expected = (float)i + 0.5f;
        if (alias_first[i] != expected || alias_second[i] != expected)
        {
            fprintf(stderr, "synchronized alias/tail load mismatch at %u expected=%f data=%f output=%f\n",
                    i, expected, alias_first[i], alias_second[i]);
            UnmapMemory(device, alias_memory);
            return 1;
        }
    }
    UnmapMemory(device, alias_memory);
    DestroyPipeline(device, tail_pipeline, 0);
    DestroyShaderModule(device, tail_shader, 0);

    descriptor_data.input.buffer = alias_buffers[0];
    descriptor_data.output.buffer = overlapping_buffer;
    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, writable_pipeline);
    CmdPushDescriptorSetWithTemplateKHR(
        command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
    CmdPushConstants(
        command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
        0, sizeof(element_count), &element_count);
    CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    VkResult overlapping_result = QueueSubmit(queue, 1, &submit_info, fence);
    if (overlapping_result != VK_ERROR_INITIALIZATION_FAILED)
    {
        fprintf(stderr, "overlapping writable alias returned %d\n", overlapping_result);
        return 1;
    }

    CHECK_VK(ResetCommandBuffer(command_buffer, 0));
    CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
    VkMemoryBarrier memory_barrier;
    memset(&memory_barrier, 0, sizeof(memory_barrier));
    memory_barrier.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER;
    memory_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    memory_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    VkBufferMemoryBarrier buffer_barrier;
    memset(&buffer_barrier, 0, sizeof(buffer_barrier));
    buffer_barrier.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER;
    buffer_barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    buffer_barrier.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    buffer_barrier.srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    buffer_barrier.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    buffer_barrier.buffer = buffers[1];
    buffer_barrier.offset = 0;
    buffer_barrier.size = VK_WHOLE_SIZE;
    CmdPipelineBarrier(
        command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
        VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &memory_barrier, 1, &buffer_barrier, 0, 0);
    VkBufferCopy copy_region;
    copy_region.srcOffset = 0;
    copy_region.dstOffset = 0;
    copy_region.size = buffer_size;
    CmdCopyBuffer(command_buffer, buffers[1], buffers[0], 1, &copy_region);
    CHECK_VK(EndCommandBuffer(command_buffer));
    CHECK_VK(ResetFences(device, 1, &fence));
    CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
    CHECK_VK(QueueWaitIdle(queue));
    CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));
    CHECK_VK(GetFenceStatus(device, fence));
    CHECK_VK(DeviceWaitIdle(device));

    // Fence-less submissions must not retain their replay resources forever.
    // The second submit opportunistically polls the first, and QueueWaitIdle
    // completes whichever submissions are still pending.
    CHECK_VK(QueueSubmit(queue, 0, 0, VK_NULL_HANDLE));
    CHECK_VK(QueueSubmit(queue, 0, 0, VK_NULL_HANDLE));
    CHECK_VK(QueueWaitIdle(queue));

    void* copy_src_mapped = 0;
    void* copy_dst_mapped = 0;
    CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &copy_src_mapped));
    CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, &copy_dst_mapped));
    if (memcmp(copy_src_mapped, copy_dst_mapped, buffer_size) != 0)
    {
        fprintf(stderr, "vkCmdCopyBuffer output mismatch\n");
        return 1;
    }
    UnmapMemory(device, memories[0]);
    UnmapMemory(device, memories[1]);

    if (packed_integer_dot_supported)
    {
        uint32_t* integer_dot_input = 0;
        CHECK_VK(MapMemory(device, memories[0], 0, buffer_size, 0, (void**)&integer_dot_input));
        integer_dot_input[0] = 0xfffefdfc;
        integer_dot_input[1] = 0xfbfaf9f8;
        CHECK_VK(FlushMappedMemoryRanges(device, 1, &input_range));
        UnmapMemory(device, memories[0]);

        uint32_t* integer_dot_output = 0;
        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, (void**)&integer_dot_output));
        integer_dot_output[0] = 0;
        integer_dot_output[1] = 0;
        CHECK_VK(FlushMappedMemoryRanges(device, 1, &output_range));
        UnmapMemory(device, memories[1]);

        VkShaderModuleCreateInfo integer_dot_shader_create_info = shader_module_create_info;
        integer_dot_shader_create_info.codeSize = sizeof(packed_integer_dot_shader_spv);
        integer_dot_shader_create_info.pCode = packed_integer_dot_shader_spv;
        VkShaderModule integer_dot_shader = VK_NULL_HANDLE;
        CHECK_VK(CreateShaderModule(device, &integer_dot_shader_create_info, 0, &integer_dot_shader));

        VkComputePipelineCreateInfo integer_dot_pipeline_create_info = pipeline_create_info;
        integer_dot_pipeline_create_info.stage.module = integer_dot_shader;
        VkPipeline integer_dot_pipeline = VK_NULL_HANDLE;
        CHECK_VK(CreateComputePipelines(device, VK_NULL_HANDLE, 1, &integer_dot_pipeline_create_info, 0, &integer_dot_pipeline));

        descriptor_data.input.buffer = buffers[0];
        descriptor_data.input.offset = 0;
        descriptor_data.input.range = buffer_size;
        descriptor_data.output.buffer = buffers[1];
        descriptor_data.output.offset = 0;
        descriptor_data.output.range = buffer_size;
        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, integer_dot_pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdDispatch(command_buffer, 1, 1, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, (void**)&integer_dot_output));
        CHECK_VK(InvalidateMappedMemoryRanges(device, 1, &output_range));
        if ((int32_t)integer_dot_output[0] != 70 || integer_dot_output[1] != 252998)
        {
            fprintf(stderr, "packed integer dot mismatch signed=%d unsigned=%u\n",
                    (int32_t)integer_dot_output[0], integer_dot_output[1]);
            return 1;
        }
        UnmapMemory(device, memories[1]);

        DestroyPipeline(device, integer_dot_pipeline, 0);
        DestroyShaderModule(device, integer_dot_shader, 0);
    }

    VkQueryPool timestamp_query_pool = VK_NULL_HANDLE;
    if (timestamp_query_supported)
    {
        VkQueryPoolCreateInfo query_pool_create_info;
        memset(&query_pool_create_info, 0, sizeof(query_pool_create_info));
        query_pool_create_info.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO;
        query_pool_create_info.queryType = VK_QUERY_TYPE_TIMESTAMP;
        query_pool_create_info.queryCount = 2;
        CHECK_VK(CreateQueryPool(device, &query_pool_create_info, 0, &timestamp_query_pool));

        uint64_t timestamps[2] = {UINT64_MAX, UINT64_MAX};
        if (GetQueryPoolResults(device, timestamp_query_pool, 0, 2, sizeof(timestamps),
                                timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT)
            != VK_NOT_READY)
        {
            fprintf(stderr, "new timestamp query pool is unexpectedly available\n");
            return 1;
        }

        descriptor_data.input.buffer = buffers[0];
        descriptor_data.input.offset = 0;
        descriptor_data.input.range = buffer_size;
        descriptor_data.output.buffer = buffers[1];
        descriptor_data.output.offset = 0;
        descriptor_data.output.range = buffer_size;
        push_constants.count = element_count;
        push_constants.bias = 0.25f;
        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdResetQueryPool(command_buffer, timestamp_query_pool, 0, 2);
        CmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestamp_query_pool, 0);
        CmdBindPipeline(command_buffer, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
        CmdPushDescriptorSetWithTemplateKHR(
            command_buffer, descriptor_template, pipeline_layout, 0, &descriptor_data);
        CmdPushConstants(command_buffer, pipeline_layout, VK_SHADER_STAGE_COMPUTE_BIT,
                         0, sizeof(push_constants), &push_constants);
        CmdDispatch(command_buffer, (element_count + 63) / 64, 1, 1);
        CmdWriteTimestamp(command_buffer, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, timestamp_query_pool, 1);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(GetQueryPoolResults(device, timestamp_query_pool, 0, 2, sizeof(timestamps),
                                     timestamps, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT));
        if (timestamps[1] < timestamps[0])
        {
            fprintf(stderr, "timestamp query order is invalid begin=%llu end=%llu\n",
                    (unsigned long long)timestamps[0], (unsigned long long)timestamps[1]);
            return 1;
        }

        CHECK_VK(ResetCommandBuffer(command_buffer, 0));
        CHECK_VK(BeginCommandBuffer(command_buffer, &begin_info));
        CmdCopyQueryPoolResults(command_buffer, timestamp_query_pool, 0, 2,
                                buffers[1], 0, sizeof(uint64_t), VK_QUERY_RESULT_64_BIT);
        CHECK_VK(EndCommandBuffer(command_buffer));
        CHECK_VK(ResetFences(device, 1, &fence));
        CHECK_VK(QueueSubmit(queue, 1, &submit_info, fence));
        CHECK_VK(WaitForFences(device, 1, &fence, VK_TRUE, UINT64_MAX));

        CHECK_VK(MapMemory(device, memories[1], 0, buffer_size, 0, &mapped));
        const uint64_t* copied_timestamps = (const uint64_t*)mapped;
        if (copied_timestamps[0] != timestamps[0] || copied_timestamps[1] != timestamps[1])
        {
            fprintf(stderr, "copied timestamp results mismatch\n");
            return 1;
        }
        UnmapMemory(device, memories[1]);
    }

    TrimCommandPool(device, command_pool, 0);
    CHECK_VK(ResetCommandPool(device, command_pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));

    DestroyBuffer(device, overlapping_buffer, 0);
    DestroyBuffer(device, alias_buffers[0], 0);
    DestroyBuffer(device, alias_buffers[1], 0);
    FreeMemory(device, alias_memory, 0);
    DestroyPipeline(device, writable_pipeline, 0);
    DestroyShaderModule(device, writable_shader_module, 0);

    FreeCommandBuffers(device, command_pool, 1, &command_buffer);
    DestroyFence(device, fence, 0);
    DestroyCommandPool(device, command_pool, 0);
    DestroyDescriptorUpdateTemplateKHR(device, large_descriptor_template, 0);
    DestroyPipeline(device, large_pipeline, 0);
    DestroyShaderModule(device, large_shader_module, 0);
    DestroyPipelineLayout(device, large_pipeline_layout, 0);
    DestroyPipeline(device, specialization_pipeline, 0);
    DestroyShaderModule(device, specialization_shader_module, 0);
    DestroyDescriptorUpdateTemplateKHR(device, descriptor_template, 0);
    DestroyPipeline(device, pipeline, 0);
    DestroyShaderModule(device, shader_module, 0);
    DestroyPipelineLayout(device, pipeline_layout, 0);
    DestroyDescriptorSetLayout(device, descriptor_layout, 0);
    for (uint32_t i = 0; i < 2; i++)
    {
        DestroyBuffer(device, buffers[i], 0);
        FreeMemory(device, memories[i], 0);
    }
    DestroyPipelineCache(device, pipeline_caches[1], 0);
    DestroyPipelineCache(device, pipeline_caches[0], 0);
    if (timestamp_query_pool)
        DestroyQueryPool(device, timestamp_query_pool, 0);
    DestroyDevice(device, 0);
    DestroyInstance(instance, 0);

    return 0;
}

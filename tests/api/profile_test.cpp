// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

#include <cstdio>
#include <cstring>
#include <vector>

extern "C" VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* name);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char* layer_name, uint32_t* property_count, VkExtensionProperties* properties);
extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo* create_info, const VkAllocationCallbacks* allocator, VkInstance* instance);

static bool has_extension(const std::vector<VkExtensionProperties>& extensions, const char* name)
{
    for (size_t i = 0; i < extensions.size(); i++)
    {
        if (strcmp(extensions[i].extensionName, name) == 0)
            return true;
    }
    return false;
}

int main()
{
    if (vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkCreateDevice") != 0
        || vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkQueueSubmit") != 0)
    {
        fprintf(stderr, "global proc lookup exposed a non-global command\n");
        return 1;
    }

    PFN_vkEnumerateInstanceVersion enumerate_instance_version = (PFN_vkEnumerateInstanceVersion)vkGetInstanceProcAddr(VK_NULL_HANDLE, "vkEnumerateInstanceVersion");
    uint32_t api_version = 0;
    if (!enumerate_instance_version
        || enumerate_instance_version(&api_version) != VK_SUCCESS
        || api_version < VK_API_VERSION_1_1)
    {
        fprintf(stderr, "Vulkan 1.1 global profile is unavailable\n");
        return 1;
    }

    uint32_t extension_count = 0;
    if (vkEnumerateInstanceExtensionProperties(0, &extension_count, 0) != VK_SUCCESS)
        return 1;
    std::vector<VkExtensionProperties> extensions(extension_count);
    if (vkEnumerateInstanceExtensionProperties(0, &extension_count, extensions.data()) != VK_SUCCESS
        || !has_extension(extensions, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
    {
        fprintf(stderr, "required instance extension is unavailable\n");
        return 1;
    }

    const char* enabled_extensions[] = {
        VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME};
    VkApplicationInfo application_info = {};
    application_info.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    application_info.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo create_info = {};
    create_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    create_info.pApplicationInfo = &application_info;
    create_info.enabledExtensionCount = 1;
    create_info.ppEnabledExtensionNames = enabled_extensions;

    VkInstance instance = VK_NULL_HANDLE;
    if (vkCreateInstance(&create_info, 0, &instance) != VK_SUCCESS || !instance)
        return 1;

    PFN_vkVoidFunction features2 = vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2");
    PFN_vkVoidFunction features2_khr = vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceFeatures2KHR");
    PFN_vkDestroyInstance destroy_instance = (PFN_vkDestroyInstance)vkGetInstanceProcAddr(instance, "vkDestroyInstance");
    const bool valid = features2 && features2 == features2_khr
                       && vkGetInstanceProcAddr(instance, "vkCreateImage") == 0
                       && destroy_instance;
    if (destroy_instance)
        destroy_instance(instance, 0);
    if (!valid)
    {
        fprintf(stderr, "instance proc scope, alias, or unsupported API profile mismatch\n");
        return 1;
    }
    return 0;
}

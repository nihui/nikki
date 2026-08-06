// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_internal.h"
#include "backend/objects.h"
#include "abi_guard.h"

#if defined(_WIN32)
#define NIKKI_EXPORT __declspec(dllexport)
#else
#define NIKKI_EXPORT __attribute__((visibility("default")))
#endif

namespace nikki_detail {

enum ProcScope
{
    PROC_SCOPE_GLOBAL,
    PROC_SCOPE_INSTANCE,
    PROC_SCOPE_DEVICE
};

struct VulkanProc
{
    const char* name;
    PFN_vkVoidFunction function;
    ProcScope scope;
    const char* required_extension;
    uint32_t required_api_version;
};

#define NIKKI_BOOTSTRAP_PROC(api_name, function, scope, extension, version) \
    {"vk" #api_name, (PFN_vkVoidFunction)function, scope, extension, version},
#define NIKKI_PROC(api_name, impl_name, scope, extension, version) \
    {"vk" #api_name, abi_function<decltype(&impl_##impl_name), &impl_##impl_name>(), scope, extension, version},

static const VulkanProc g_vulkan_procs[] = {
#include "vulkan_procs.inc"
};

#undef NIKKI_PROC
#undef NIKKI_BOOTSTRAP_PROC

static const VulkanProc* find_vulkan_proc(const char* name, ProcScope minimum_scope,
                                          ProcScope maximum_scope)
{
    if (!name)
        return 0;
    for (size_t i = 0; i < sizeof(g_vulkan_procs) / sizeof(g_vulkan_procs[0]); i++)
    {
        if (g_vulkan_procs[i].scope >= minimum_scope
            && g_vulkan_procs[i].scope <= maximum_scope
            && strcmp(g_vulkan_procs[i].name, name) == 0)
            return &g_vulkan_procs[i];
    }
    return 0;
}

static bool proc_available(uint32_t api_version,
                           const std::unordered_set<std::string>& enabled_extensions,
                           const VulkanProc& proc)
{
    if (proc.required_api_version && api_version < proc.required_api_version)
        return false;
    if (proc.required_extension
        && enabled_extensions.find(proc.required_extension) == enabled_extensions.end())
        return false;
    return true;
}

PFN_vkVoidFunction impl_get_device_proc_addr(VkDevice device, const char* name)
{
    Device* device_impl = unwrap(device);
    const VulkanProc* proc = find_vulkan_proc(name, PROC_SCOPE_DEVICE, PROC_SCOPE_DEVICE);
    if (!device_impl || !proc
        || !proc_available(device_impl->instance->api_version, device_impl->enabled_extensions, *proc))
        return 0;
    return proc->function;
}

PFN_vkVoidFunction impl_get_instance_proc_addr(VkInstance instance, const char* name)
{
    const VulkanProc* proc = find_vulkan_proc(
        name, PROC_SCOPE_GLOBAL, instance ? PROC_SCOPE_DEVICE : PROC_SCOPE_GLOBAL);
    if (!proc)
        return 0;
    if (proc->scope == PROC_SCOPE_GLOBAL)
        return proc->function;

    Instance* instance_impl = unwrap(instance);
    if (!instance_impl)
        return 0;
    if (proc->scope == PROC_SCOPE_DEVICE)
    {
        if (proc->required_api_version
            && instance_impl->api_version < proc->required_api_version
            && !proc->required_extension)
            return 0;
        return proc->function;
    }
    if (proc->required_extension
        && strcmp(proc->required_extension, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME) != 0)
        return proc->function;
    if (!proc_available(instance_impl->api_version, instance_impl->enabled_extensions, *proc))
        return 0;

    return proc->function;
}

} // namespace nikki_detail

extern "C" NIKKI_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceExtensionProperties(
    const char* layer_name, uint32_t* property_count, VkExtensionProperties* properties)
{
    return nikki_detail::VulkanAbiGuard<
        decltype(&nikki_detail::impl_enumerate_instance_extension_properties),
        &nikki_detail::impl_enumerate_instance_extension_properties>::call(layer_name, property_count, properties);
}

extern "C" NIKKI_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo* create_info, const VkAllocationCallbacks* allocator, VkInstance* instance)
{
    return nikki_detail::VulkanAbiGuard<decltype(&nikki_detail::impl_create_instance),
                                        &nikki_detail::impl_create_instance>::call(create_info, allocator, instance);
}

extern "C" NIKKI_EXPORT VKAPI_ATTR VkResult VKAPI_CALL vkEnumerateInstanceLayerProperties(
    uint32_t* property_count, VkLayerProperties* properties)
{
    return nikki_detail::VulkanAbiGuard<
        decltype(&nikki_detail::impl_enumerate_instance_layer_properties),
        &nikki_detail::impl_enumerate_instance_layer_properties>::call(property_count, properties);
}

extern "C" NIKKI_EXPORT VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL vkGetInstanceProcAddr(VkInstance instance, const char* name)
{
    return nikki_detail::VulkanAbiGuard<decltype(&nikki_detail::impl_get_instance_proc_addr),
                                        &nikki_detail::impl_get_instance_proc_addr>::call(instance, name);
}

#undef NIKKI_EXPORT

// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_ABI_GUARD_H
#define NIKKI_ABI_GUARD_H

#include "common.h"

#include <vulkan/vulkan.h>

#include <exception>
#include <new>

namespace nikki_detail {

template<typename Function, Function function>
struct VulkanAbiGuard;

template<typename... Args, VkResult (*function)(Args...)>
struct VulkanAbiGuard<VkResult (*)(Args...), function>
{
    static VKAPI_ATTR VkResult VKAPI_CALL call(Args... args) noexcept
    {
        try
        {
            return function(args...);
        }
        catch (const std::bad_alloc&)
        {
            return VK_ERROR_OUT_OF_HOST_MEMORY;
        }
        catch (const std::exception& exception)
        {
            log_error("Vulkan entry point failed: %s", exception.what());
            return VK_ERROR_INITIALIZATION_FAILED;
        }
        catch (...)
        {
            log_error("Vulkan entry point failed with an unknown exception");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
};

template<typename... Args, void (*function)(Args...)>
struct VulkanAbiGuard<void (*)(Args...), function>
{
    static VKAPI_ATTR void VKAPI_CALL call(Args... args) noexcept
    {
        try
        {
            function(args...);
        }
        catch (const std::exception& exception)
        {
            log_error("Vulkan entry point failed: %s", exception.what());
        }
        catch (...)
        {
            log_error("Vulkan entry point failed with an unknown exception");
        }
    }
};

template<typename... Args, PFN_vkVoidFunction (*function)(Args...)>
struct VulkanAbiGuard<PFN_vkVoidFunction (*)(Args...), function>
{
    static VKAPI_ATTR PFN_vkVoidFunction VKAPI_CALL call(Args... args) noexcept
    {
        try
        {
            return function(args...);
        }
        catch (const std::exception& exception)
        {
            log_error("Vulkan proc lookup failed: %s", exception.what());
            return 0;
        }
        catch (...)
        {
            log_error("Vulkan proc lookup failed with an unknown exception");
            return 0;
        }
    }
};

template<typename Function, Function function>
inline PFN_vkVoidFunction abi_function()
{
    return (PFN_vkVoidFunction)&VulkanAbiGuard<Function, function>::call;
}

} // namespace nikki_detail

#endif // NIKKI_ABI_GUARD_H

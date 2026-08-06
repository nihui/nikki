// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#define VK_NO_PROTOTYPES
#include <vulkan/vulkan.h>

extern "C" VKAPI_ATTR VkResult VKAPI_CALL vkCreateInstance(
    const VkInstanceCreateInfo* create_info, const VkAllocationCallbacks* allocator,
    VkInstance* instance);

int main()
{
    VkInstance instance = VK_NULL_HANDLE;
    for (int i = 0; i < 2; i++)
    {
        if (vkCreateInstance(0, 0, &instance) != VK_ERROR_INITIALIZATION_FAILED)
            return 1;
    }
    return 0;
}

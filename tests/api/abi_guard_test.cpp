// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "../../src/abi_guard.h"

#include <new>
#include <stdexcept>

namespace {

VkResult throw_bad_alloc()
{
    throw std::bad_alloc();
}

VkResult throw_result_exception()
{
    throw std::runtime_error("result");
}

void throw_void_exception(int* called)
{
    *called = 1;
    throw std::runtime_error("void");
}

PFN_vkVoidFunction throw_proc_exception()
{
    throw std::runtime_error("proc");
}

} // namespace

int main()
{
    using namespace nikki_detail;

    if (VulkanAbiGuard<decltype(&throw_bad_alloc), &throw_bad_alloc>::call()
        != VK_ERROR_OUT_OF_HOST_MEMORY)
        return 1;
    if (VulkanAbiGuard<decltype(&throw_result_exception), &throw_result_exception>::call()
        != VK_ERROR_INITIALIZATION_FAILED)
        return 1;

    int void_called = 0;
    VulkanAbiGuard<decltype(&throw_void_exception), &throw_void_exception>::call(&void_called);
    if (!void_called)
        return 1;

    if (VulkanAbiGuard<decltype(&throw_proc_exception), &throw_proc_exception>::call() != 0)
        return 1;

    return 0;
}

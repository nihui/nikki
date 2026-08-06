// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_internal.h"
#include "backend/runtime.h"

namespace nikki_detail {

VkResult wait_device_submissions(Device* device)
{
    if (!device)
        return VK_ERROR_DEVICE_LOST;

    std::vector<std::shared_ptr<Submission> > submissions;
    {
        std::lock_guard<std::mutex> lock(device->mutex);
        submissions = device->submissions;
    }

    VkResult first_error = device->lost || device->error ? VK_ERROR_DEVICE_LOST : VK_SUCCESS;
    for (size_t i = 0; i < submissions.size(); i++)
    {
        const VkResult result = process_submission(submissions[i], UINT64_MAX);
        if (result != VK_SUCCESS && first_error == VK_SUCCESS)
            first_error = result;
    }
    return first_error;
}

VkResult impl_queue_wait_idle(VkQueue queue)
{
    return wait_device_submissions(unwrap(queue));
}

VkResult impl_device_wait_idle(VkDevice device)
{
    return wait_device_submissions(unwrap(device));
}

VkResult impl_create_fence(VkDevice device, const VkFenceCreateInfo* create_info,
                           const VkAllocationCallbacks* allocator, VkFence* fence)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_FENCE_CREATE_INFO || !fence)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || (create_info->flags & ~VK_FENCE_CREATE_SIGNALED_BIT) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<Fence> fence_impl = std::make_shared<Fence>();
    fence_impl->owner = device_impl;
    fence_impl->signaled = (create_info->flags & VK_FENCE_CREATE_SIGNALED_BIT) != 0;
    *fence = make_handle<VkFence>(g_fences, fence_impl);
    return VK_SUCCESS;
}

void impl_destroy_fence(VkDevice device, VkFence fence, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<Fence> fence_impl = get_handle(g_fences, fence);
    if (!device_impl || !fence_impl || fence_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    if (fence_impl->submission && !fence_impl->submission->processed)
    {
        log_error("device %llu fence %llu destroyed while pending",
                  (unsigned long long)device_impl->id, (unsigned long long)fence_impl->id);
        return;
    }

    erase_handle(g_fences, fence);
}

VkResult impl_get_fence_status(VkDevice device, VkFence fence)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<Fence> fence_impl = get_handle(g_fences, fence);
    if (!device_impl || !fence_impl || fence_impl->owner != device_impl)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_impl->lost || device_impl->error)
    {
        fence_impl->failed = true;
        return VK_ERROR_DEVICE_LOST;
    }
    if (fence_impl->failed)
        return VK_ERROR_DEVICE_LOST;
    if (fence_impl->signaled)
        return VK_SUCCESS;
    if (fence_impl->submission)
    {
        const VkResult result = process_submission(fence_impl->submission, 0);
        if (result == VK_SUCCESS)
        {
            fence_impl->signaled = true;
            return VK_SUCCESS;
        }
        if (result != VK_TIMEOUT)
        {
            fence_impl->failed = true;
            return result;
        }
    }

    return VK_NOT_READY;
}

VkResult impl_reset_fences(VkDevice device, uint32_t fence_count, const VkFence* fences)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || fence_count == 0 || !fences)
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < fence_count; i++)
    {
        std::shared_ptr<Fence> fence_impl = get_handle(g_fences, fences[i]);
        if (!fence_impl || fence_impl->owner != device_impl || (fence_impl->submission && !fence_impl->submission->processed))
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    for (uint32_t i = 0; i < fence_count; i++)
    {
        std::shared_ptr<Fence> fence_impl = get_handle(g_fences, fences[i]);
        fence_impl->signaled = false;
        fence_impl->failed = false;
        fence_impl->submission.reset();
    }
    return VK_SUCCESS;
}

VkResult impl_wait_for_fences(VkDevice device, uint32_t fence_count, const VkFence* fences,
                              VkBool32 wait_all, uint64_t timeout)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || fence_count == 0 || !fences)
        return VK_ERROR_INITIALIZATION_FAILED;

    std::vector<std::shared_ptr<Fence> > fence_impls;
    for (uint32_t i = 0; i < fence_count; i++)
    {
        std::shared_ptr<Fence> fence_impl = get_handle(g_fences, fences[i]);
        if (!fence_impl || fence_impl->owner != device_impl)
            return VK_ERROR_INITIALIZATION_FAILED;
        fence_impls.push_back(fence_impl);
    }

    if (device_impl->lost || device_impl->error)
    {
        for (size_t i = 0; i < fence_impls.size(); i++)
            fence_impls[i]->failed = true;
        return VK_ERROR_DEVICE_LOST;
    }

    const uint64_t wait_begin = monotonic_time_ns();
    if (!wait_all)
    {
        for (;;)
        {
            std::vector<WGPUFutureWaitInfo> wait_infos;
            for (size_t i = 0; i < fence_impls.size(); i++)
            {
                std::shared_ptr<Fence>& fence = fence_impls[i];
                if (fence->signaled)
                    return VK_SUCCESS;
                if (fence->failed)
                    return VK_ERROR_DEVICE_LOST;
                if (!fence->submission)
                    continue;

                const VkResult result = process_submission(fence->submission, 0);
                if (result == VK_SUCCESS)
                {
                    fence->signaled = true;
                    return VK_SUCCESS;
                }
                if (result != VK_TIMEOUT)
                {
                    fence->failed = true;
                    return result;
                }

                WGPUFuture future = {};
                if (!fence->submission->completed)
                {
                    future = fence->submission->future;
                }
                else
                {
                    for (size_t j = 0; j < fence->submission->readbacks.size(); j++)
                    {
                        const Readback& readback = fence->submission->readbacks[j];
                        if (!readback.copied && readback.map_started
                            && readback.map_result && !readback.map_result->completed)
                        {
                            future = readback.future;
                            break;
                        }
                    }
                    if (future.id == 0)
                    {
                        for (size_t j = 0; j < fence->submission->query_readbacks.size(); j++)
                        {
                            const QueryReadback& readback = fence->submission->query_readbacks[j];
                            if (!readback.copied && readback.map_started
                                && readback.map_result && !readback.map_result->completed)
                            {
                                future = readback.future;
                                break;
                            }
                        }
                    }
                }

                if (future.id != 0)
                {
                    bool duplicate = false;
                    for (size_t j = 0; j < wait_infos.size(); j++)
                    {
                        if (wait_infos[j].future.id == future.id)
                        {
                            duplicate = true;
                            break;
                        }
                    }
                    if (!duplicate)
                    {
                        WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
                        wait_info.future = future;
                        wait_infos.push_back(wait_info);
                    }
                }
            }

            const uint64_t remaining = remaining_timeout_ns(wait_begin, timeout);
            if (remaining == 0 || wait_infos.empty())
                return VK_TIMEOUT;

            WGPUWaitStatus status;
            if (wait_any(device_impl->instance.get(), wait_infos.size(), wait_infos.data(),
                         remaining, "fence-wait-any", status)
                != 0)
            {
                device_impl->lost = true;
                for (size_t i = 0; i < fence_impls.size(); i++)
                    fence_impls[i]->failed = true;
                return VK_ERROR_DEVICE_LOST;
            }
            if (status == WGPUWaitStatus_TimedOut)
                return VK_TIMEOUT;
            if (status != WGPUWaitStatus_Success)
            {
                device_impl->lost = true;
                for (size_t i = 0; i < fence_impls.size(); i++)
                    fence_impls[i]->failed = true;
                return VK_ERROR_DEVICE_LOST;
            }

            bool completed = false;
            for (size_t i = 0; i < wait_infos.size(); i++)
                completed = completed || wait_infos[i].completed == WGPU_TRUE;
            if (!completed)
            {
                device_impl->lost = true;
                for (size_t i = 0; i < fence_impls.size(); i++)
                    fence_impls[i]->failed = true;
                return VK_ERROR_DEVICE_LOST;
            }
        }
    }

    for (size_t i = 0; i < fence_impls.size(); i++)
    {
        if (fence_impls[i]->signaled)
            continue;
        if (fence_impls[i]->failed)
            return VK_ERROR_DEVICE_LOST;
        if (!fence_impls[i]->submission)
            return VK_TIMEOUT;

        const uint64_t remaining = remaining_timeout_ns(wait_begin, timeout);
        const VkResult result = process_submission(fence_impls[i]->submission, remaining);
        if (result != VK_SUCCESS)
        {
            if (result != VK_TIMEOUT)
                fence_impls[i]->failed = true;
            return result;
        }
        fence_impls[i]->signaled = true;
    }

    return VK_SUCCESS;
}

// Pipeline cache objects are accepted for Vulkan compatibility.
// WebGPU pipelines are not serialized.

} // namespace nikki_detail

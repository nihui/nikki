// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_internal.h"
#include "backend/dispatch.h"
#include "backend/shader_module.h"

namespace nikki_detail {

VkResult impl_queue_submit(VkQueue queue, uint32_t submit_count, const VkSubmitInfo* submits, VkFence fence)
{
    Device* device = unwrap(queue);
    if (!device || (submit_count != 0 && !submits))
        return VK_ERROR_DEVICE_LOST;
    if (device->lost || device->error)
        return VK_ERROR_DEVICE_LOST;

    const VkResult reap_result = reap_device_submissions(device);
    if (reap_result != VK_SUCCESS)
        return reap_result;
    if (device->lost || device->error)
        return VK_ERROR_DEVICE_LOST;

    std::shared_ptr<Fence> fence_impl;
    if (fence != VK_NULL_HANDLE)
    {
        fence_impl = get_handle(g_fences, fence);
        if (!fence_impl || fence_impl->owner != device || fence_impl->signaled || fence_impl->submission)
            return VK_ERROR_INITIALIZATION_FAILED;
    }

    std::shared_ptr<Submission> submission = std::make_shared<Submission>();
    submission->owner = device;
    std::vector<MemoryRange> readback_ranges;
    std::vector<MemoryRange> upload_ranges;
    std::vector<std::shared_ptr<CommandBuffer> > command_buffers;
    std::vector<unsigned char> push_uniform_data;
    const uint64_t push_uniform_alignment = std::max<uint64_t>(4, device->limits.minUniformBufferOffsetAlignment);
    VkResult result = collect_submit_command_buffers(device, submit_count, submits, command_buffers);
    if (result != VK_SUCCESS)
        return result;

    result = collect_submission_data(device, command_buffers, submission, upload_ranges,
                                     push_uniform_data, push_uniform_alignment);
    if (result != VK_SUCCESS)
        return result;
    upload_memory_ranges(device, upload_ranges);

    WGPUBuffer push_uniform_buffer = 0;
    result = create_push_uniform_buffer(device, push_uniform_data, submission, push_uniform_buffer);
    if (result != VK_SUCCESS)
        return result;

    wgpuDevicePushErrorScope(device->device, WGPUErrorFilter_Validation);
    WGPUCommandEncoderDescriptor encoder_descriptor = WGPU_COMMAND_ENCODER_DESCRIPTOR_INIT;
    WGPUCommandEncoder encoder = wgpuDeviceCreateCommandEncoder(device->device, &encoder_descriptor);
    if (!encoder)
    {
        pop_error_scope(device, "command encoder");
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    ReplayState replay;
    replay.encoder = encoder;
    replay.push_uniform_buffer = push_uniform_buffer;
    replay.push_uniform_size = push_uniform_data.size();
    replay.push_uniform_alignment = push_uniform_alignment;
    for (size_t i = 0; i < command_buffers.size(); i++)
    {
        if (encode_command_buffer(device, replay, command_buffers[i], submission, readback_ranges) != 0)
        {
            end_compute_pass(replay);
            wgpuCommandEncoderRelease(encoder);
            pop_error_scope(device, "command replay");
            return VK_ERROR_INITIALIZATION_FAILED;
        }
    }
    end_compute_pass(replay);

    result = encode_query_readbacks(device, encoder, submission);
    if (result != VK_SUCCESS)
    {
        wgpuCommandEncoderRelease(encoder);
        pop_error_scope(device, "query readback allocation");
        return result;
    }

    result = encode_readback_ranges(device, encoder, readback_ranges, submission);
    if (result != VK_SUCCESS)
    {
        wgpuCommandEncoderRelease(encoder);
        pop_error_scope(device, "readback allocation");
        return result;
    }

    WGPUCommandBufferDescriptor command_buffer_descriptor = WGPU_COMMAND_BUFFER_DESCRIPTOR_INIT;
    WGPUCommandBuffer webgpu_command_buffer = wgpuCommandEncoderFinish(encoder, &command_buffer_descriptor);
    wgpuCommandEncoderRelease(encoder);
    if (!webgpu_command_buffer || pop_error_scope(device, "command encoding") != 0)
    {
        if (webgpu_command_buffer)
            wgpuCommandBufferRelease(webgpu_command_buffer);
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    wgpuQueueSubmit(device->queue, 1, &webgpu_command_buffer);
    wgpuCommandBufferRelease(webgpu_command_buffer);

    for (size_t i = 0; i < submission->query_uses.size(); i++)
    {
        std::lock_guard<std::mutex> lock(submission->query_uses[i].query_pool->mutex);
        submission->query_uses[i].query_pool->pending_count++;
    }

    WGPUQueueWorkDoneCallbackInfo callback_info = WGPU_QUEUE_WORK_DONE_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = queue_work_done_callback;
    std::shared_ptr<Submission>* callback_context = new std::shared_ptr<Submission>(submission);
    callback_info.userdata1 = callback_context;
    submission->future = wgpuQueueOnSubmittedWorkDone(device->queue, callback_info);
    if (submission->future.id == 0)
    {
        delete callback_context;
        device->lost = true;
        for (size_t i = 0; i < submission->command_buffers.size(); i++)
        {
            submission->command_buffers[i]->state = COMMAND_BUFFER_INVALID;
            submission->command_buffers[i]->error = VK_ERROR_DEVICE_LOST;
        }
        if (fence_impl)
        {
            fence_impl->signaled = false;
            fence_impl->failed = true;
        }
        {
            // Queue work has already been submitted, but without a completion
            // future there is no safe point at which its resources can be
            // released. Keep them alive until wgpuDeviceDestroy().
            std::lock_guard<std::mutex> lock(device->mutex);
            device->abandoned_submissions.push_back(submission);
        }
        log_error("device %llu submission %llu returned an invalid completion future",
                  (unsigned long long)device->id, (unsigned long long)submission->id);
        return VK_ERROR_DEVICE_LOST;
    }

    for (size_t i = 0; i < submission->command_buffers.size(); i++)
    {
        CommandBuffer& command_buffer = *submission->command_buffers[i];
        command_buffer.pending_count++;
        command_buffer.state = COMMAND_BUFFER_PENDING;
    }

    if (fence_impl)
    {
        submission->fence = fence_impl;
        fence_impl->submission = submission;
        fence_impl->signaled = false;
        fence_impl->failed = false;
    }
    {
        std::lock_guard<std::mutex> lock(device->mutex);
        device->submissions.push_back(submission);
    }

    return VK_SUCCESS;
}

} // namespace nikki_detail

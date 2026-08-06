// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"

namespace nikki_detail {

Submission::~Submission()
{
    release_resources();
}

void Submission::release_resources()
{
    for (size_t i = 0; i < bind_groups.size(); i++)
        wgpuBindGroupRelease(bind_groups[i]);
    bind_groups.clear();
    for (size_t i = 0; i < temporary_buffers.size(); i++)
        wgpuBufferRelease(temporary_buffers[i]);
    temporary_buffers.clear();
    for (size_t i = 0; i < readbacks.size(); i++)
        wgpuBufferRelease(readbacks[i].buffer);
    readbacks.clear();
    for (size_t i = 0; i < query_readbacks.size(); i++)
        wgpuBufferRelease(query_readbacks[i].buffer);
    query_readbacks.clear();
    memories.clear();
    buffers.clear();
    pipelines.clear();
    query_uses.clear();
    command_buffers.clear();
}

void end_compute_pass(ReplayState& replay)
{
    if (!replay.compute_pass)
        return;

    wgpuComputePassEncoderEnd(replay.compute_pass);
    wgpuComputePassEncoderRelease(replay.compute_pass);
    replay.compute_pass = 0;
    replay.compute_pass_buffer_access.clear();
}

static QueryUse& get_query_use(const std::shared_ptr<Submission>& submission,
                               const std::shared_ptr<QueryPool>& query_pool)
{
    for (size_t i = 0; i < submission->query_uses.size(); i++)
    {
        if (submission->query_uses[i].query_pool.get() == query_pool.get())
            return submission->query_uses[i];
    }

    QueryUse use;
    use.query_pool = query_pool;
    use.written.resize(query_pool->values.size(), 0);
    submission->query_uses.push_back(use);
    return submission->query_uses.back();
}

static int encode_timestamp(ReplayState& replay, const std::shared_ptr<QueryPool>& query_pool,
                            uint32_t query)
{
    end_compute_pass(replay);

    WGPUPassTimestampWrites timestamp_writes = WGPU_PASS_TIMESTAMP_WRITES_INIT;
    timestamp_writes.querySet = query_pool->query_set;
    timestamp_writes.beginningOfPassWriteIndex = query * 2;
    timestamp_writes.endOfPassWriteIndex = query * 2 + 1;

    WGPUComputePassDescriptor descriptor = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    descriptor.timestampWrites = &timestamp_writes;
    WGPUComputePassEncoder compute_pass = wgpuCommandEncoderBeginComputePass(replay.encoder, &descriptor);
    if (!compute_pass)
        return -1;

    wgpuComputePassEncoderEnd(compute_pass);
    wgpuComputePassEncoderRelease(compute_pass);
    return 0;
}

int encode_command_buffer(Device* device, ReplayState& replay,
                          const std::shared_ptr<CommandBuffer>& command_buffer,
                          const std::shared_ptr<Submission>& submission,
                          std::vector<MemoryRange>& readback_ranges)
{
    for (size_t i = 0; i < command_buffer->commands.size(); i++)
    {
        const Command& command = command_buffer->commands[i];
        if (command.type == COMMAND_DISPATCH)
        {
            if (encode_dispatch(device, replay, command, submission) != 0)
                return -1;

            for (size_t j = 0; j < command.bindings.size(); j++)
            {
                if (command.bindings[j].access == BUFFER_ACCESS_READ_WRITE
                    && !command.bindings[j].memory->host_shadow.empty()
                    && !append_memory_transfer_range(readback_ranges,
                                                     command.bindings[j].memory,
                                                     command.bindings[j].offset,
                                                     command.bindings[j].size))
                    return -1;
            }
        }
        else if (command.type == COMMAND_COPY_BUFFER)
        {
            end_compute_pass(replay);
            for (size_t j = 0; j < command.copy_regions.size(); j++)
            {
                const CopyRegion& region = command.copy_regions[j];
                const uint64_t src_offset = command.src_buffer->memory_offset + region.src_offset;
                const uint64_t dst_offset = command.dst_buffer->memory_offset + region.dst_offset;
                if (command.src_buffer->memory.get() == command.dst_buffer->memory.get())
                {
                    SameBufferCopyPlan plan;
                    if (make_same_buffer_copy_plan(device, command, region, plan))
                    {
                        if (encode_same_buffer_copy(device, replay, command, plan, submission) != 0)
                            return -1;
                    }
                    else
                    {
                        end_compute_pass(replay);
                        WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
                        descriptor.size = align_up(region.size, 4);
                        descriptor.usage = WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;
                        WGPUBuffer temporary = wgpuDeviceCreateBuffer(device->device, &descriptor);
                        if (!temporary)
                            return -1;
                        submission->temporary_buffers.push_back(temporary);

                        wgpuCommandEncoderCopyBufferToBuffer(
                            replay.encoder, command.src_buffer->memory->buffer, src_offset, temporary, 0, region.size);
                        wgpuCommandEncoderCopyBufferToBuffer(
                            replay.encoder, temporary, 0, command.dst_buffer->memory->buffer, dst_offset, region.size);
                    }
                }
                else
                {
                    end_compute_pass(replay);
                    wgpuCommandEncoderCopyBufferToBuffer(
                        replay.encoder,
                        command.src_buffer->memory->buffer, src_offset,
                        command.dst_buffer->memory->buffer, dst_offset,
                        region.size);
                }
            }

            append_unique(submission->buffers, command.src_buffer);
            append_unique(submission->buffers, command.dst_buffer);
            append_unique(submission->memories, command.src_buffer->memory);
            append_unique(submission->memories, command.dst_buffer->memory);
            if (!command.dst_buffer->memory->host_shadow.empty())
            {
                for (size_t j = 0; j < command.copy_regions.size(); j++)
                {
                    uint64_t dst_offset = 0;
                    if (!checked_add(command.dst_buffer->memory_offset,
                                     command.copy_regions[j].dst_offset, dst_offset)
                        || !append_memory_transfer_range(readback_ranges,
                                                         command.dst_buffer->memory,
                                                         dst_offset,
                                                         command.copy_regions[j].size))
                        return -1;
                }
            }
        }
        else if (command.type == COMMAND_RESET_QUERY_POOL)
        {
            end_compute_pass(replay);
            QueryUse& use = get_query_use(submission, command.query_pool);
            std::lock_guard<std::mutex> lock(command.query_pool->mutex);
            for (uint32_t j = 0; j < command.query_count; j++)
            {
                const uint32_t query = command.first_query + j;
                use.written[query] = 0;
                command.query_pool->available[query] = 0;
            }
        }
        else if (command.type == COMMAND_WRITE_TIMESTAMP)
        {
            QueryUse& use = get_query_use(submission, command.query_pool);
            const uint32_t query = command.first_query;
            if (use.written[query] || encode_timestamp(replay, command.query_pool, query) != 0)
                return -1;

            use.written[query] = 1;
            std::lock_guard<std::mutex> lock(command.query_pool->mutex);
            command.query_pool->available[query] = 0;
        }
        else if (command.type == COMMAND_COPY_QUERY_POOL_RESULTS)
        {
            end_compute_pass(replay);
            get_query_use(submission, command.query_pool);

            const uint64_t size = (uint64_t)(command.query_count - 1) * 256 + sizeof(uint64_t);
            WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
            descriptor.size = size;
            descriptor.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
            WGPUBuffer resolved = wgpuDeviceCreateBuffer(device->device, &descriptor);
            if (!resolved)
                return -1;
            submission->temporary_buffers.push_back(resolved);

            const uint64_t dst_offset = command.query_result_buffer->memory_offset + command.query_result_offset;
            for (uint32_t j = 0; j < command.query_count; j++)
            {
                const uint64_t resolve_offset = (uint64_t)j * 256;
                wgpuCommandEncoderResolveQuerySet(replay.encoder, command.query_pool->query_set,
                                                  (command.first_query + j) * 2, 1, resolved, resolve_offset);
                wgpuCommandEncoderCopyBufferToBuffer(replay.encoder, resolved, resolve_offset,
                                                     command.query_result_buffer->memory->buffer,
                                                     dst_offset + (uint64_t)j * sizeof(uint64_t), sizeof(uint64_t));
            }
            append_unique(submission->buffers, command.query_result_buffer);
            append_unique(submission->memories, command.query_result_buffer->memory);
            if (!command.query_result_buffer->memory->host_shadow.empty()
                && !append_memory_transfer_range(readback_ranges,
                                                 command.query_result_buffer->memory,
                                                 dst_offset, (uint64_t)command.query_count * sizeof(uint64_t)))
                return -1;
        }
        else if (command.type == COMMAND_BARRIER)
        {
            end_compute_pass(replay);
            for (size_t j = 0; j < command.barrier_buffers.size(); j++)
            {
                append_unique(submission->buffers, command.barrier_buffers[j]);
                append_unique(submission->memories, command.barrier_buffers[j]->memory);
            }
        }
    }

    append_unique(submission->command_buffers, command_buffer);
    return 0;
}

void queue_work_done_callback(WGPUQueueWorkDoneStatus status, WGPUStringView message,
                              void* userdata1, void*)
{
    std::shared_ptr<Submission>* context = (std::shared_ptr<Submission>*)userdata1;
    if (!context)
        return;
    std::shared_ptr<Submission> submission = *context;
    delete context;
    if (!submission)
        return;

    submission->status = status;
    submission->completed = true;
    if (status != WGPUQueueWorkDoneStatus_Success)
        log_error("submission %llu failed status=%d: %.*s", (unsigned long long)submission->id,
                  (int)status, (int)message.length, message.data ? message.data : "");
}

static void map_callback(WGPUMapAsyncStatus status, WGPUStringView message, void* userdata1, void*)
{
    std::shared_ptr<MapResult>* context = (std::shared_ptr<MapResult>*)userdata1;
    if (!context)
        return;
    std::shared_ptr<MapResult> result = *context;
    delete context;
    if (!result)
        return;

    result->status = status;
    result->completed = true;
    if (status != WGPUMapAsyncStatus_Success)
        log_error("readback map failed status=%d: %.*s", (int)status,
                  (int)message.length, message.data ? message.data : "");
}

static VkResult finish_submission(const std::shared_ptr<Submission>& submission, VkResult result)
{
    if (!submission)
        return result;
    if (submission->processed)
        return submission->result;

    for (size_t i = 0; i < submission->command_buffers.size(); i++)
    {
        CommandBuffer& command_buffer = *submission->command_buffers[i];
        if (command_buffer.pending_count != 0)
            command_buffer.pending_count--;
        if (command_buffer.pending_count == 0)
        {
            command_buffer.state = (command_buffer.usage_flags & VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT)
                                       ? COMMAND_BUFFER_INVALID
                                       : COMMAND_BUFFER_EXECUTABLE;
        }
    }

    for (size_t i = 0; i < submission->query_uses.size(); i++)
    {
        std::lock_guard<std::mutex> lock(submission->query_uses[i].query_pool->mutex);
        if (submission->query_uses[i].query_pool->pending_count != 0)
            submission->query_uses[i].query_pool->pending_count--;
    }

    submission->result = result;
    submission->processed = true;

    std::shared_ptr<Fence> fence = submission->fence.lock();
    if (fence)
    {
        fence->signaled = result == VK_SUCCESS;
        fence->failed = result != VK_SUCCESS;
    }

    submission->release_resources();
    if (submission->owner)
    {
        std::lock_guard<std::mutex> lock(submission->owner->mutex);
        std::vector<std::shared_ptr<Submission> >& submissions = submission->owner->submissions;
        submissions.erase(
            std::remove_if(
                submissions.begin(), submissions.end(),
                [&](const std::shared_ptr<Submission>& value) {
                    return value.get() == submission.get();
                }),
            submissions.end());
    }

    return result;
}

static VkResult abandon_submission(const std::shared_ptr<Submission>& submission, VkResult result)
{
    if (!submission)
        return result;
    if (submission->processed)
        return submission->result;

    for (size_t i = 0; i < submission->command_buffers.size(); i++)
    {
        CommandBuffer& command_buffer = *submission->command_buffers[i];
        if (command_buffer.pending_count != 0)
            command_buffer.pending_count--;
        command_buffer.state = COMMAND_BUFFER_INVALID;
        command_buffer.error = result;
    }

    submission->result = result;
    submission->processed = true;

    std::shared_ptr<Fence> fence = submission->fence.lock();
    if (fence)
    {
        fence->signaled = false;
        fence->failed = true;
    }

    if (submission->owner)
    {
        std::lock_guard<std::mutex> lock(submission->owner->mutex);
        std::vector<std::shared_ptr<Submission> >& submissions = submission->owner->submissions;
        submissions.erase(
            std::remove_if(
                submissions.begin(), submissions.end(),
                [&](const std::shared_ptr<Submission>& value) {
                    return value.get() == submission.get();
                }),
            submissions.end());

        // A failed WaitAny does not prove that WebGPU has stopped using the
        // recorded resources. Keep them alive until the device is destroyed.
        submission->owner->abandoned_submissions.push_back(submission);
    }

    return result;
}

VkResult process_submission(const std::shared_ptr<Submission>& submission, uint64_t timeout)
{
    if (!submission)
        return VK_ERROR_DEVICE_LOST;
    if (submission->processed)
        return submission->result;

    const uint64_t wait_begin = monotonic_time_ns();
    if (!submission->completed)
    {
        const int wait_result = wait_future(
            submission->owner->instance.get(), submission->future, timeout, "queue-completion");
        if (wait_result == 1)
            return VK_TIMEOUT;
        if (wait_result != 0)
        {
            submission->owner->lost = true;
            return abandon_submission(submission, VK_ERROR_DEVICE_LOST);
        }
    }
    if (submission->status != WGPUQueueWorkDoneStatus_Success)
    {
        submission->owner->lost = true;
        return finish_submission(submission, VK_ERROR_DEVICE_LOST);
    }

    for (size_t i = 0; i < submission->readbacks.size(); i++)
    {
        Readback& readback = submission->readbacks[i];
        if (readback.copied)
            continue;

        if (!readback.map_started)
        {
            readback.map_result = std::make_shared<MapResult>();
            std::shared_ptr<MapResult>* callback_context = new std::shared_ptr<MapResult>(readback.map_result);
            WGPUBufferMapCallbackInfo callback_info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
            callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
            callback_info.callback = map_callback;
            callback_info.userdata1 = callback_context;
            readback.future = wgpuBufferMapAsync(readback.buffer, WGPUMapMode_Read, 0, readback.size, callback_info);
            if (readback.future.id == 0)
            {
                delete callback_context;
                submission->owner->error = true;
                return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
            }
            readback.map_started = true;
        }

        if (!readback.map_result->completed)
        {
            const uint64_t remaining = remaining_timeout_ns(wait_begin, timeout);
            const int wait_result = wait_future(submission->owner->instance.get(), readback.future, remaining, "readback-map");
            if (wait_result == 1)
                return VK_TIMEOUT;
            if (wait_result != 0)
            {
                submission->owner->error = true;
                return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
            }
        }
        if (!readback.map_result->completed
            || readback.map_result->status != WGPUMapAsyncStatus_Success)
        {
            submission->owner->error = true;
            return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
        }

        const void* mapped = wgpuBufferGetConstMappedRange(readback.buffer, 0, readback.size);
        if (!mapped)
        {
            wgpuBufferUnmap(readback.buffer);
            submission->owner->error = true;
            return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
        }
        memcpy(readback.memory->host_shadow.data() + readback.offset, mapped, readback.size);
        subtract_byte_range(readback.memory->dirty_ranges, readback.offset, readback.size);
        wgpuBufferUnmap(readback.buffer);
        readback.copied = true;
    }

    for (size_t i = 0; i < submission->query_readbacks.size(); i++)
    {
        QueryReadback& readback = submission->query_readbacks[i];
        if (readback.copied)
            continue;

        const uint64_t size = (uint64_t)readback.written.size() * 2 * sizeof(uint64_t);
        if (!readback.map_started)
        {
            readback.map_result = std::make_shared<MapResult>();
            std::shared_ptr<MapResult>* callback_context = new std::shared_ptr<MapResult>(readback.map_result);
            WGPUBufferMapCallbackInfo callback_info = WGPU_BUFFER_MAP_CALLBACK_INFO_INIT;
            callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
            callback_info.callback = map_callback;
            callback_info.userdata1 = callback_context;
            readback.future = wgpuBufferMapAsync(readback.buffer, WGPUMapMode_Read, 0, size, callback_info);
            if (readback.future.id == 0)
            {
                delete callback_context;
                submission->owner->error = true;
                return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
            }
            readback.map_started = true;
        }

        if (!readback.map_result->completed)
        {
            const uint64_t remaining = remaining_timeout_ns(wait_begin, timeout);
            const int wait_result = wait_future(submission->owner->instance.get(), readback.future, remaining, "query-readback-map");
            if (wait_result == 1)
                return VK_TIMEOUT;
            if (wait_result != 0)
            {
                submission->owner->error = true;
                return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
            }
        }
        if (!readback.map_result->completed || readback.map_result->status != WGPUMapAsyncStatus_Success)
        {
            submission->owner->error = true;
            return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
        }

        const void* mapped = wgpuBufferGetConstMappedRange(readback.buffer, 0, size);
        if (!mapped)
        {
            wgpuBufferUnmap(readback.buffer);
            submission->owner->error = true;
            return finish_submission(submission, VK_ERROR_MEMORY_MAP_FAILED);
        }

        const uint64_t* values = (const uint64_t*)mapped;
        {
            std::lock_guard<std::mutex> lock(readback.query_pool->mutex);
            for (size_t j = 0; j < readback.written.size(); j++)
            {
                if (readback.written[j])
                {
                    readback.query_pool->values[j] = values[j * 2];
                    readback.query_pool->available[j] = 1;
                }
            }
        }
        wgpuBufferUnmap(readback.buffer);
        readback.copied = true;
    }

    return finish_submission(submission, VK_SUCCESS);
}

VkResult reap_device_submissions(Device* device)
{
    if (!device)
        return VK_ERROR_DEVICE_LOST;

    std::vector<std::shared_ptr<Submission> > submissions;
    {
        std::lock_guard<std::mutex> lock(device->mutex);
        submissions = device->submissions;
    }

    for (size_t i = 0; i < submissions.size(); i++)
    {
        const VkResult result = process_submission(submissions[i], 0);
        if (result != VK_SUCCESS && result != VK_TIMEOUT)
            return result;
    }

    return VK_SUCCESS;
}

static bool command_resources_live(const Command& command)
{
    if (command.type == COMMAND_DISPATCH)
    {
        if (!command.pipeline || !command.pipeline->live)
            return false;
        for (size_t i = 0; i < command.bindings.size(); i++)
        {
            if (!command.bindings[i].buffer || !command.bindings[i].buffer->live
                || !command.bindings[i].memory || !command.bindings[i].memory->live)
                return false;
        }
        return true;
    }
    if (command.type == COMMAND_COPY_BUFFER)
    {
        return command.src_buffer && command.src_buffer->live
               && command.src_buffer->memory && command.src_buffer->memory->live
               && command.dst_buffer && command.dst_buffer->live
               && command.dst_buffer->memory && command.dst_buffer->memory->live;
    }
    if (command.type == COMMAND_BARRIER)
    {
        for (size_t i = 0; i < command.barrier_buffers.size(); i++)
        {
            if (!command.barrier_buffers[i] || !command.barrier_buffers[i]->live
                || !command.barrier_buffers[i]->memory || !command.barrier_buffers[i]->memory->live)
                return false;
        }
        return true;
    }
    if (command.type == COMMAND_RESET_QUERY_POOL || command.type == COMMAND_WRITE_TIMESTAMP)
        return command.query_pool && command.query_pool->live;
    if (command.type == COMMAND_COPY_QUERY_POOL_RESULTS)
    {
        return command.query_pool && command.query_pool->live
               && command.query_result_buffer && command.query_result_buffer->live
               && command.query_result_buffer->memory && command.query_result_buffer->memory->live;
    }

    return false;
}

VkResult collect_submit_command_buffers(Device* device, uint32_t submit_count,
                                        const VkSubmitInfo* submits,
                                        std::vector<std::shared_ptr<CommandBuffer> >& command_buffers)
{
    std::unordered_set<CommandBuffer*> unique_command_buffers;
    for (uint32_t i = 0; i < submit_count; i++)
    {
        if (submits[i].sType != VK_STRUCTURE_TYPE_SUBMIT_INFO || submits[i].pNext
            || submits[i].waitSemaphoreCount != 0 || submits[i].signalSemaphoreCount != 0
            || (submits[i].commandBufferCount != 0 && !submits[i].pCommandBuffers))
            return VK_ERROR_FEATURE_NOT_PRESENT;

        for (uint32_t j = 0; j < submits[i].commandBufferCount; j++)
        {
            VkCommandBuffer handle = submits[i].pCommandBuffers[j];
            CommandBuffer* command_buffer = unwrap(handle);
            if (!command_buffer || command_buffer->owner != device
                || command_buffer->state != COMMAND_BUFFER_EXECUTABLE || command_buffer->error != VK_SUCCESS)
                return VK_ERROR_INITIALIZATION_FAILED;

            const bool first_use = unique_command_buffers.insert(command_buffer).second;
            if (!first_use || command_buffer->pending_count != 0)
                return VK_ERROR_INITIALIZATION_FAILED;
            for (size_t k = 0; k < command_buffer->commands.size(); k++)
            {
                if (!command_resources_live(command_buffer->commands[k]))
                {
                    log_error("device %llu command buffer %llu submit uses destroyed resource at command %llu",
                              (unsigned long long)device->id,
                              (unsigned long long)command_buffer->id,
                              (unsigned long long)k);
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
                if (command_buffer->commands[k].query_pool
                    && query_pool_in_flight(command_buffer->commands[k].query_pool.get()))
                {
                    log_error("device %llu command buffer %llu submit reuses pending query pool %llu",
                              (unsigned long long)device->id,
                              (unsigned long long)command_buffer->id,
                              (unsigned long long)command_buffer->commands[k].query_pool->id);
                    return VK_ERROR_INITIALIZATION_FAILED;
                }
            }
            command_buffers.push_back(handle->impl);
        }
    }

    return VK_SUCCESS;
}

static bool append_uniform_data(std::vector<unsigned char>& uniform_data,
                                uint64_t alignment, uint64_t max_buffer_size,
                                const void* data, uint64_t data_size, uint64_t allocation_size)
{
    const uint64_t offset = align_up(uniform_data.size(), alignment);
    uint64_t end = 0;
    if (allocation_size == 0 || data_size > allocation_size || (data_size != 0 && !data)
        || (offset == 0 && !uniform_data.empty())
        || !checked_add(offset, allocation_size, end)
        || end > max_buffer_size
        || end > std::numeric_limits<size_t>::max())
        return false;

    uniform_data.resize((size_t)end, 0);
    if (data_size != 0)
        memcpy(uniform_data.data() + offset, data, data_size);
    return true;
}

VkResult collect_submission_data(Device* device,
                                 const std::vector<std::shared_ptr<CommandBuffer> >& command_buffers,
                                 const std::shared_ptr<Submission>& submission,
                                 std::vector<MemoryRange>& upload_ranges,
                                 std::vector<unsigned char>& push_uniform_data,
                                 uint64_t push_uniform_alignment)
{
    for (size_t i = 0; i < command_buffers.size(); i++)
    {
        const std::vector<Command>& commands = command_buffers[i]->commands;
        for (size_t j = 0; j < commands.size(); j++)
        {
            const Command& command = commands[j];
            if (command.type == COMMAND_DISPATCH)
            {
                for (size_t k = 0; k < command.bindings.size(); k++)
                {
                    append_unique(submission->memories, command.bindings[k].memory);
                    if (!append_host_upload_ranges(upload_ranges,
                                                   command.bindings[k].memory,
                                                   command.bindings[k].offset,
                                                   command.bindings[k].size))
                        return VK_ERROR_INITIALIZATION_FAILED;
                }

                if (command.pipeline->push_constant_uniform)
                {
                    const uint64_t allocation_size = align_up(command.pipeline->push_constant_uniform_size, 4);
                    if (command.push_constants.size() < command.pipeline->push_constant_data_size
                        || !append_uniform_data(push_uniform_data,
                                                push_uniform_alignment, device->limits.maxBufferSize,
                                                command.push_constants.data(),
                                                command.pipeline->push_constant_data_size,
                                                allocation_size))
                        return VK_ERROR_OUT_OF_HOST_MEMORY;
                }
            }
            else if (command.type == COMMAND_COPY_BUFFER)
            {
                append_unique(submission->memories, command.src_buffer->memory);
                append_unique(submission->memories, command.dst_buffer->memory);
                for (size_t k = 0; k < command.copy_regions.size(); k++)
                {
                    SameBufferCopyPlan plan;
                    if (make_same_buffer_copy_plan(device, command, command.copy_regions[k], plan))
                    {
                        const uint64_t max_words = same_buffer_copy_max_words(device);
                        if (max_words == 0)
                            return VK_ERROR_INITIALIZATION_FAILED;

                        for (uint64_t base_word = 0; base_word < plan.word_count;)
                        {
                            const uint64_t chunk_words = std::min<uint64_t>(plan.word_count - base_word, max_words);
                            SameBufferCopyParams params;
                            params.src_word = plan.src_word;
                            params.dst_word = plan.dst_word;
                            params.word_count = plan.word_count;
                            params.base_word = (uint32_t)base_word;
                            if (!append_uniform_data(push_uniform_data,
                                                     push_uniform_alignment, device->limits.maxBufferSize,
                                                     &params, sizeof(params), sizeof(params)))
                                return VK_ERROR_OUT_OF_HOST_MEMORY;
                            base_word += chunk_words;
                        }
                    }

                    uint64_t src_offset = 0;
                    if (!checked_add(command.src_buffer->memory_offset,
                                     command.copy_regions[k].src_offset, src_offset)
                        || !append_host_upload_ranges(upload_ranges,
                                                      command.src_buffer->memory,
                                                      src_offset,
                                                      command.copy_regions[k].size))
                        return VK_ERROR_INITIALIZATION_FAILED;
                }
            }
        }
    }

    return VK_SUCCESS;
}

void upload_memory_ranges(Device* device, std::vector<MemoryRange>& upload_ranges)
{
    for (size_t i = 0; i < upload_ranges.size(); i++)
    {
        MemoryRange& range = upload_ranges[i];
        wgpuQueueWriteBuffer(device->queue, range.memory->buffer, range.offset,
                             range.memory->host_shadow.data() + range.offset, range.size);
        subtract_byte_range(range.memory->dirty_ranges, range.offset, range.size);
    }
}

VkResult create_push_uniform_buffer(Device* device,
                                    const std::vector<unsigned char>& push_uniform_data,
                                    const std::shared_ptr<Submission>& submission,
                                    WGPUBuffer& push_uniform_buffer)
{
    push_uniform_buffer = 0;
    if (push_uniform_data.empty())
        return VK_SUCCESS;

    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.size = align_up(push_uniform_data.size(), 4);
    descriptor.usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    push_uniform_buffer = wgpuDeviceCreateBuffer(device->device, &descriptor);
    if (!push_uniform_buffer)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    submission->temporary_buffers.push_back(push_uniform_buffer);
    wgpuQueueWriteBuffer(device->queue, push_uniform_buffer, 0,
                         push_uniform_data.data(), push_uniform_data.size());
    return VK_SUCCESS;
}

VkResult encode_readback_ranges(Device* device, WGPUCommandEncoder encoder,
                                const std::vector<MemoryRange>& readback_ranges,
                                const std::shared_ptr<Submission>& submission)
{
    for (size_t i = 0; i < readback_ranges.size(); i++)
    {
        const MemoryRange& range = readback_ranges[i];
        WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        descriptor.size = range.size;
        descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        WGPUBuffer readback_buffer = wgpuDeviceCreateBuffer(device->device, &descriptor);
        if (!readback_buffer)
            return VK_ERROR_OUT_OF_HOST_MEMORY;

        wgpuCommandEncoderCopyBufferToBuffer(encoder, range.memory->buffer, range.offset,
                                             readback_buffer, 0, range.size);
        Readback readback;
        readback.buffer = readback_buffer;
        readback.memory = range.memory;
        readback.offset = range.offset;
        readback.size = range.size;
        submission->readbacks.push_back(readback);
    }

    return VK_SUCCESS;
}

VkResult encode_query_readbacks(Device* device, WGPUCommandEncoder encoder,
                                const std::shared_ptr<Submission>& submission)
{
    for (size_t i = 0; i < submission->query_uses.size(); i++)
    {
        const QueryUse& use = submission->query_uses[i];
        if (std::find(use.written.begin(), use.written.end(), (unsigned char)1) == use.written.end())
            continue;

        const uint64_t size = (uint64_t)use.written.size() * 2 * sizeof(uint64_t);
        WGPUBufferDescriptor resolve_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        resolve_descriptor.size = size;
        resolve_descriptor.usage = WGPUBufferUsage_QueryResolve | WGPUBufferUsage_CopySrc;
        WGPUBuffer resolve_buffer = wgpuDeviceCreateBuffer(device->device, &resolve_descriptor);
        if (!resolve_buffer)
            return VK_ERROR_OUT_OF_DEVICE_MEMORY;
        submission->temporary_buffers.push_back(resolve_buffer);

        WGPUBufferDescriptor readback_descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        readback_descriptor.size = size;
        readback_descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_MapRead;
        WGPUBuffer readback_buffer = wgpuDeviceCreateBuffer(device->device, &readback_descriptor);
        if (!readback_buffer)
            return VK_ERROR_OUT_OF_HOST_MEMORY;

        wgpuCommandEncoderResolveQuerySet(encoder, use.query_pool->query_set,
                                          0, (uint32_t)use.written.size() * 2, resolve_buffer, 0);
        wgpuCommandEncoderCopyBufferToBuffer(encoder, resolve_buffer, 0, readback_buffer, 0, size);

        QueryReadback readback;
        readback.buffer = readback_buffer;
        readback.query_pool = use.query_pool;
        readback.written = use.written;
        submission->query_readbacks.push_back(readback);
    }

    return VK_SUCCESS;
}

} // namespace nikki_detail

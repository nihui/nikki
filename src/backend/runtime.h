// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_BACKEND_RUNTIME_H
#define NIKKI_BACKEND_RUNTIME_H

#include "device.h"

namespace nikki_detail {

inline bool append_memory_transfer_range(std::vector<MemoryRange>& ranges,
                                         const std::shared_ptr<DeviceMemory>& memory,
                                         uint64_t offset, uint64_t size)
{
    if (!memory || size == 0)
        return false;

    uint64_t end = 0;
    if (!checked_add(offset, size, end) || end > memory->size)
        return false;

    const uint64_t aligned_offset = offset - offset % 4;
    const uint64_t aligned_end = align_up(end, 4);
    if (aligned_end == 0 || aligned_end > memory->padded_size)
        return false;

    uint64_t merged_offset = aligned_offset;
    uint64_t merged_end = aligned_end;
    for (size_t i = 0; i < ranges.size();)
    {
        if (ranges[i].memory.get() != memory.get())
        {
            i++;
            continue;
        }

        uint64_t range_end = 0;
        if (!checked_add(ranges[i].offset, ranges[i].size, range_end))
            return false;
        if (merged_end < ranges[i].offset || range_end < merged_offset)
        {
            i++;
            continue;
        }

        merged_offset = std::min(merged_offset, ranges[i].offset);
        merged_end = std::max(merged_end, range_end);
        ranges.erase(ranges.begin() + i);
        i = 0;
    }

    MemoryRange range;
    range.memory = memory;
    range.offset = merged_offset;
    range.size = merged_end - merged_offset;
    ranges.push_back(range);
    return true;
}

inline bool append_host_upload_ranges(std::vector<MemoryRange>& ranges,
                                      const std::shared_ptr<DeviceMemory>& memory,
                                      uint64_t offset, uint64_t size)
{
    if (!memory || memory->host_shadow.empty())
        return true;

    uint64_t end = 0;
    if (size == 0 || !checked_add(offset, size, end) || end > memory->size)
        return false;

    if (memory->mapped)
    {
        uint64_t mapped_end = 0;
        if (!checked_add(memory->mapped_offset, memory->mapped_size, mapped_end))
            return false;
        const uint64_t intersection_offset = std::max<uint64_t>(offset, memory->mapped_offset);
        const uint64_t intersection_end = std::min<uint64_t>(end, mapped_end);
        if (intersection_offset < intersection_end
            && !append_memory_transfer_range(ranges, memory, intersection_offset,
                                             intersection_end - intersection_offset))
            return false;
    }

    for (size_t i = 0; i < memory->dirty_ranges.size(); i++)
    {
        uint64_t dirty_end = 0;
        if (!checked_add(memory->dirty_ranges[i].offset, memory->dirty_ranges[i].size, dirty_end))
            return false;
        const uint64_t intersection_offset = std::max<uint64_t>(offset, memory->dirty_ranges[i].offset);
        const uint64_t intersection_end = std::min<uint64_t>(end, dirty_end);
        if (intersection_offset < intersection_end
            && !append_memory_transfer_range(ranges, memory, intersection_offset,
                                             intersection_end - intersection_offset))
            return false;
    }

    return true;
}

inline uint64_t buffer_memory_alignment(const Device& device, const Buffer& buffer)
{
    uint64_t alignment = 4;
    if ((buffer.usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) != 0)
        alignment = std::max<uint64_t>(alignment, device.limits.minStorageBufferOffsetAlignment);
    if ((buffer.usage & VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT) != 0)
        alignment = std::max<uint64_t>(alignment, device.limits.minUniformBufferOffsetAlignment);
    return alignment;
}

template<typename T>
inline bool contains_object(const std::vector<std::shared_ptr<T> >& objects, const T* object)
{
    for (size_t i = 0; i < objects.size(); i++)
    {
        if (objects[i].get() == object)
            return true;
    }
    return false;
}

template<typename T>
inline void append_unique(std::vector<std::shared_ptr<T> >& values, const std::shared_ptr<T>& value)
{
    if (!value)
        return;

    for (size_t i = 0; i < values.size(); i++)
    {
        if (values[i].get() == value.get())
            return;
    }
    values.push_back(value);
}

inline bool memory_in_flight(Device* device, const DeviceMemory* memory)
{
    std::lock_guard<std::mutex> lock(device->mutex);
    for (size_t i = 0; i < device->submissions.size(); i++)
    {
        if (!device->submissions[i]->processed
            && contains_object(device->submissions[i]->memories, memory))
            return true;
    }
    for (size_t i = 0; i < device->abandoned_submissions.size(); i++)
    {
        if (contains_object(device->abandoned_submissions[i]->memories, memory))
            return true;
    }
    return false;
}

inline bool buffer_in_flight(Device* device, const Buffer* buffer)
{
    std::lock_guard<std::mutex> lock(device->mutex);
    for (size_t i = 0; i < device->submissions.size(); i++)
    {
        if (!device->submissions[i]->processed
            && contains_object(device->submissions[i]->buffers, buffer))
            return true;
    }
    for (size_t i = 0; i < device->abandoned_submissions.size(); i++)
    {
        if (contains_object(device->abandoned_submissions[i]->buffers, buffer))
            return true;
    }
    return false;
}

inline bool pipeline_in_flight(Device* device, const ComputePipeline* pipeline)
{
    std::lock_guard<std::mutex> lock(device->mutex);
    for (size_t i = 0; i < device->submissions.size(); i++)
    {
        if (!device->submissions[i]->processed
            && contains_object(device->submissions[i]->pipelines, pipeline))
            return true;
    }
    for (size_t i = 0; i < device->abandoned_submissions.size(); i++)
    {
        if (contains_object(device->abandoned_submissions[i]->pipelines, pipeline))
            return true;
    }
    return false;
}

inline bool query_pool_in_flight(QueryPool* query_pool)
{
    if (!query_pool)
        return false;

    std::lock_guard<std::mutex> lock(query_pool->mutex);
    return query_pool->pending_count != 0;
}

VkResult wait_device_submissions(Device* device);
void release_device_objects(Device* device);
void end_compute_pass(ReplayState& replay);
int encode_command_buffer(Device* device, ReplayState& replay,
                          const std::shared_ptr<CommandBuffer>& command_buffer,
                          const std::shared_ptr<Submission>& submission,
                          std::vector<MemoryRange>& readback_ranges);
void queue_work_done_callback(WGPUQueueWorkDoneStatus status, WGPUStringView message,
                              void* userdata1, void* userdata2);
VkResult process_submission(const std::shared_ptr<Submission>& submission, uint64_t timeout);
VkResult reap_device_submissions(Device* device);
VkResult collect_submit_command_buffers(
    Device* device, uint32_t submit_count, const VkSubmitInfo* submits,
    std::vector<std::shared_ptr<CommandBuffer> >& command_buffers);
VkResult collect_submission_data(
    Device* device, const std::vector<std::shared_ptr<CommandBuffer> >& command_buffers,
    const std::shared_ptr<Submission>& submission,
    std::vector<MemoryRange>& upload_ranges,
    std::vector<unsigned char>& push_uniform_data, uint64_t push_uniform_alignment);
void upload_memory_ranges(Device* device, std::vector<MemoryRange>& upload_ranges);
VkResult create_push_uniform_buffer(
    Device* device, const std::vector<unsigned char>& push_uniform_data,
    const std::shared_ptr<Submission>& submission, WGPUBuffer& push_uniform_buffer);
VkResult encode_readback_ranges(
    Device* device, WGPUCommandEncoder encoder,
    const std::vector<MemoryRange>& readback_ranges,
    const std::shared_ptr<Submission>& submission);
VkResult encode_query_readbacks(Device* device, WGPUCommandEncoder encoder,
                                const std::shared_ptr<Submission>& submission);

} // namespace nikki_detail

#endif // NIKKI_BACKEND_RUNTIME_H

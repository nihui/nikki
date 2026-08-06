// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_internal.h"
#include "backend/runtime.h"

namespace nikki_detail {

DeviceMemory::~DeviceMemory()
{
    if (buffer)
        wgpuBufferRelease(buffer);
}

// memory and buffer

VkResult impl_allocate_memory(VkDevice device, const VkMemoryAllocateInfo* allocate_info,
                              const VkAllocationCallbacks* allocator, VkDeviceMemory* memory)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !allocate_info || allocate_info->sType != VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO || !memory)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (allocate_info->pNext)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (allocate_info->allocationSize == 0 || allocate_info->memoryTypeIndex >= 2)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (allocate_info->allocationSize > device_impl->limits.maxBufferSize)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    const uint64_t padded_size = align_up(allocate_info->allocationSize, 4);
    if (padded_size == 0 || padded_size > device_impl->limits.maxBufferSize)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
    descriptor.size = padded_size;
    descriptor.usage = WGPUBufferUsage_Storage | WGPUBufferUsage_Uniform
                       | WGPUBufferUsage_CopySrc | WGPUBufferUsage_CopyDst;

    std::shared_ptr<DeviceMemory> memory_impl = std::make_shared<DeviceMemory>();
    memory_impl->owner = device_impl;
    memory_impl->size = allocate_info->allocationSize;
    memory_impl->padded_size = padded_size;
    memory_impl->memory_type_index = allocate_info->memoryTypeIndex;
    memory_impl->buffer = wgpuDeviceCreateBuffer(device_impl->device, &descriptor);
    if (!memory_impl->buffer)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    if (allocate_info->memoryTypeIndex == 1)
        memory_impl->host_shadow.resize(padded_size, 0);

    *memory = make_handle<VkDeviceMemory>(g_memories, memory_impl);
    return VK_SUCCESS;
}

void impl_free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DeviceMemory> memory_impl = get_handle(g_memories, memory);
    if (!device_impl || !memory_impl || memory_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    if (memory_in_flight(device_impl, memory_impl.get()))
    {
        log_error("device %llu memory %llu freed while in flight",
                  (unsigned long long)device_impl->id, (unsigned long long)memory_impl->id);
        return;
    }

    memory_impl->live = false;
    memory_impl->mapped = false;
    memory_impl->mapped_offset = 0;
    memory_impl->mapped_size = 0;
    memory_impl->dirty_ranges.clear();
    erase_handle(g_memories, memory);
}

VkResult impl_map_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
                         VkMemoryMapFlags flags, void** data)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DeviceMemory> memory_impl = get_handle(g_memories, memory);
    if (!device_impl || !memory_impl || memory_impl->owner != device_impl || !data || flags != 0 || memory_impl->host_shadow.empty())
        return VK_ERROR_MEMORY_MAP_FAILED;
    if (memory_impl->mapped)
        return VK_ERROR_MEMORY_MAP_FAILED;

    const uint64_t map_size = size == VK_WHOLE_SIZE ? memory_impl->size - std::min<uint64_t>(offset, memory_impl->size) : size;
    uint64_t end = 0;
    if (map_size == 0 || offset >= memory_impl->size || offset % 8 != 0
        || !checked_add(offset, map_size, end) || end > memory_impl->size)
        return VK_ERROR_MEMORY_MAP_FAILED;

    memory_impl->mapped = true;
    memory_impl->mapped_offset = offset;
    memory_impl->mapped_size = map_size;
    *data = memory_impl->host_shadow.data() + offset;
    return VK_SUCCESS;
}

void impl_unmap_memory(VkDevice device, VkDeviceMemory memory)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DeviceMemory> memory_impl = get_handle(g_memories, memory);
    if (!device_impl || !memory_impl || memory_impl->owner != device_impl || !memory_impl->mapped)
        return;

    if (!add_byte_range(memory_impl->dirty_ranges, memory_impl->mapped_offset, memory_impl->mapped_size))
    {
        device_impl->error = true;
        log_error("device %llu memory %llu unmap range overflow",
                  (unsigned long long)device_impl->id, (unsigned long long)memory_impl->id);
    }
    memory_impl->mapped = false;
    memory_impl->mapped_offset = 0;
    memory_impl->mapped_size = 0;
}

VkResult impl_flush_mapped_memory_ranges(VkDevice device, uint32_t range_count, const VkMappedMemoryRange* ranges)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || (range_count != 0 && !ranges))
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < range_count; i++)
    {
        std::shared_ptr<DeviceMemory> memory_impl = get_handle(g_memories, ranges[i].memory);
        if (ranges[i].sType != VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE || ranges[i].pNext
            || !memory_impl || memory_impl->owner != device_impl || memory_impl->host_shadow.empty() || !memory_impl->mapped)
            return VK_ERROR_MEMORY_MAP_FAILED;

        const uint64_t range_size = ranges[i].size == VK_WHOLE_SIZE ? memory_impl->size - std::min<uint64_t>(ranges[i].offset, memory_impl->size) : ranges[i].size;
        uint64_t end = 0;
        uint64_t mapped_end = 0;
        if (range_size == 0 || ranges[i].offset >= memory_impl->size
            || !checked_add(ranges[i].offset, range_size, end) || end > memory_impl->size
            || !checked_add(memory_impl->mapped_offset, memory_impl->mapped_size, mapped_end)
            || ranges[i].offset < memory_impl->mapped_offset || end > mapped_end)
            return VK_ERROR_MEMORY_MAP_FAILED;

        if (!add_byte_range(memory_impl->dirty_ranges, ranges[i].offset, range_size))
            return VK_ERROR_MEMORY_MAP_FAILED;
    }

    return VK_SUCCESS;
}

VkResult impl_invalidate_mapped_memory_ranges(VkDevice device, uint32_t range_count, const VkMappedMemoryRange* ranges)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || (range_count != 0 && !ranges))
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < range_count; i++)
    {
        std::shared_ptr<DeviceMemory> memory_impl = get_handle(g_memories, ranges[i].memory);
        if (ranges[i].sType != VK_STRUCTURE_TYPE_MAPPED_MEMORY_RANGE || ranges[i].pNext
            || !memory_impl || memory_impl->owner != device_impl || memory_impl->host_shadow.empty() || !memory_impl->mapped)
            return VK_ERROR_MEMORY_MAP_FAILED;

        const uint64_t range_size = ranges[i].size == VK_WHOLE_SIZE ? memory_impl->size - std::min<uint64_t>(ranges[i].offset, memory_impl->size) : ranges[i].size;
        uint64_t end = 0;
        uint64_t mapped_end = 0;
        if (range_size == 0 || ranges[i].offset >= memory_impl->size
            || !checked_add(ranges[i].offset, range_size, end) || end > memory_impl->size
            || !checked_add(memory_impl->mapped_offset, memory_impl->mapped_size, mapped_end)
            || ranges[i].offset < memory_impl->mapped_offset || end > mapped_end)
            return VK_ERROR_MEMORY_MAP_FAILED;
    }

    const VkResult wait_result = wait_device_submissions(device_impl);
    if (wait_result != VK_SUCCESS)
        return wait_result;

    return VK_SUCCESS;
}

void impl_get_device_memory_commitment(VkDevice device, VkDeviceMemory memory, VkDeviceSize* committed_memory)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<DeviceMemory> memory_impl = get_handle(g_memories, memory);
    if (!device_impl || !memory_impl || memory_impl->owner != device_impl || !committed_memory)
        return;

    *committed_memory = memory_impl->size;
}

VkResult impl_create_buffer(VkDevice device, const VkBufferCreateInfo* create_info,
                            const VkAllocationCallbacks* allocator, VkBuffer* buffer)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO || !buffer)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->size == 0 || create_info->size > device_impl->limits.maxBufferSize)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    if (create_info->sharingMode != VK_SHARING_MODE_EXCLUSIVE)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    const VkBufferUsageFlags supported_usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT
                                               | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT;
    if (create_info->usage == 0 || (create_info->usage & ~supported_usage) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<Buffer> buffer_impl = std::make_shared<Buffer>();
    buffer_impl->owner = device_impl;
    buffer_impl->size = create_info->size;
    buffer_impl->usage = create_info->usage;
    *buffer = make_handle<VkBuffer>(g_buffers, buffer_impl);
    return VK_SUCCESS;
}

void impl_destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<Buffer> buffer_impl = get_handle(g_buffers, buffer);
    if (!device_impl || !buffer_impl || buffer_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    if (buffer_in_flight(device_impl, buffer_impl.get()))
    {
        log_error("device %llu buffer %llu destroyed while in flight",
                  (unsigned long long)device_impl->id, (unsigned long long)buffer_impl->id);
        return;
    }

    buffer_impl->live = false;
    erase_handle(g_buffers, buffer);
}

void impl_get_buffer_memory_requirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<Buffer> buffer_impl = get_handle(g_buffers, buffer);
    if (!device_impl || !buffer_impl || buffer_impl->owner != device_impl || !requirements)
        return;

    requirements->size = buffer_impl->size;
    requirements->alignment = buffer_memory_alignment(*device_impl, *buffer_impl);
    requirements->memoryTypeBits = 3;
}

void impl_get_buffer_memory_requirements2(VkDevice device,
                                          const VkBufferMemoryRequirementsInfo2* info,
                                          VkMemoryRequirements2* requirements)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !info || info->sType != VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2
        || info->pNext || !requirements
        || requirements->sType != VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2)
        return;

    std::shared_ptr<Buffer> buffer_impl = get_handle(g_buffers, info->buffer);
    if (!buffer_impl || buffer_impl->owner != device_impl || !buffer_impl->live)
        return;

    impl_get_buffer_memory_requirements(device, info->buffer, &requirements->memoryRequirements);

    struct StructureHeader
    {
        VkStructureType sType;
        void* pNext;
    };

    StructureHeader* next = (StructureHeader*)requirements->pNext;
    while (next)
    {
        if (next->sType == VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS)
        {
            VkMemoryDedicatedRequirements* dedicated = (VkMemoryDedicatedRequirements*)next;
            dedicated->prefersDedicatedAllocation = VK_FALSE;
            dedicated->requiresDedicatedAllocation = VK_FALSE;
        }
        next = (StructureHeader*)next->pNext;
    }
}

static VkResult validate_buffer_memory_binding(Device* device_impl,
                                               const std::shared_ptr<Buffer>& buffer_impl,
                                               const std::shared_ptr<DeviceMemory>& memory_impl,
                                               VkDeviceSize memory_offset)
{
    if (!device_impl || !buffer_impl || !memory_impl
        || buffer_impl->owner != device_impl || memory_impl->owner != device_impl)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (!buffer_impl->live || !memory_impl->live || buffer_impl->memory)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (memory_offset % buffer_memory_alignment(*device_impl, *buffer_impl) != 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    uint64_t end = 0;
    if (!checked_add(memory_offset, buffer_impl->size, end) || end > memory_impl->size)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;

    return VK_SUCCESS;
}

VkResult impl_bind_buffer_memory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memory_offset)
{
    Device* device_impl = unwrap(device);
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;

    std::shared_ptr<Buffer> buffer_impl = get_handle(g_buffers, buffer);
    std::shared_ptr<DeviceMemory> memory_impl = get_handle(g_memories, memory);
    const VkResult result = validate_buffer_memory_binding(device_impl, buffer_impl, memory_impl, memory_offset);
    if (result != VK_SUCCESS)
        return result;

    buffer_impl->memory = memory_impl;
    buffer_impl->memory_offset = memory_offset;
    return VK_SUCCESS;
}

VkResult impl_bind_buffer_memory2(VkDevice device, uint32_t bind_info_count,
                                  const VkBindBufferMemoryInfo* bind_infos)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || (bind_info_count != 0 && !bind_infos))
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;

    struct PendingBinding
    {
        std::shared_ptr<Buffer> buffer;
        std::shared_ptr<DeviceMemory> memory;
        VkDeviceSize memory_offset;
    };

    std::vector<PendingBinding> pending;
    pending.reserve(bind_info_count);
    std::unordered_set<Buffer*> seen_buffers;
    for (uint32_t i = 0; i < bind_info_count; i++)
    {
        const VkBindBufferMemoryInfo& info = bind_infos[i];
        if (info.sType != VK_STRUCTURE_TYPE_BIND_BUFFER_MEMORY_INFO || info.pNext)
            return VK_ERROR_FEATURE_NOT_PRESENT;

        PendingBinding binding;
        binding.buffer = get_handle(g_buffers, info.buffer);
        binding.memory = get_handle(g_memories, info.memory);
        binding.memory_offset = info.memoryOffset;
        if (!binding.buffer || !seen_buffers.insert(binding.buffer.get()).second)
            return VK_ERROR_INITIALIZATION_FAILED;

        const VkResult result = validate_buffer_memory_binding(
            device_impl, binding.buffer, binding.memory, binding.memory_offset);
        if (result != VK_SUCCESS)
            return result;
        pending.push_back(binding);
    }

    for (size_t i = 0; i < pending.size(); i++)
    {
        pending[i].buffer->memory = pending[i].memory;
        pending[i].buffer->memory_offset = pending[i].memory_offset;
    }

    return VK_SUCCESS;
}

} // namespace nikki_detail

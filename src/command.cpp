// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_internal.h"
#include "backend/runtime.h"

namespace nikki_detail {

QueryPool::~QueryPool()
{
    if (query_set)
        wgpuQuerySetRelease(query_set);
}

// command pools, command buffers and command recording

static void reset_command_buffer(CommandBuffer& command_buffer, bool release_resources = false)
{
    command_buffer.state = COMMAND_BUFFER_INITIAL;
    command_buffer.pending_count = 0;
    command_buffer.usage_flags = 0;
    command_buffer.error = VK_SUCCESS;
    command_buffer.current_pipeline.reset();
    command_buffer.descriptor_pipeline_layout.reset();
    command_buffer.descriptors.clear();
    command_buffer.push_constant_pipeline_layout.reset();
    command_buffer.push_constants.clear();
    command_buffer.commands.clear();

    if (release_resources)
    {
        std::unordered_map<uint32_t, DescriptorValue>().swap(command_buffer.descriptors);
        std::vector<unsigned char>().swap(command_buffer.push_constants);
        std::vector<Command>().swap(command_buffer.commands);
    }
}

template<typename T>
static std::vector<std::shared_ptr<T> > take_device_objects(HandleTable<T>& table, Device* device)
{
    return table.take_if([device](const std::shared_ptr<T>& object) {
        return object->owner == device;
    });
}

void release_device_objects(Device* device)
{
    if (!device)
        return;

    size_t object_count = 0;

    std::vector<std::shared_ptr<Fence> > fences = take_device_objects(g_fences, device);
    object_count += fences.size();

    std::vector<std::shared_ptr<CommandPool> > command_pools = take_device_objects(g_command_pools, device);
    object_count += command_pools.size();
    for (size_t i = 0; i < command_pools.size(); i++)
    {
        std::vector<VkCommandBuffer> command_buffers;
        {
            std::lock_guard<std::mutex> lock(command_pools[i]->mutex);
            command_buffers.assign(command_pools[i]->command_buffers.begin(), command_pools[i]->command_buffers.end());
            command_pools[i]->command_buffers.clear();
        }
        object_count += command_buffers.size();
        for (size_t j = 0; j < command_buffers.size(); j++)
            delete command_buffers[j];
    }

    std::vector<std::shared_ptr<PipelineCache> > pipeline_caches = take_device_objects(g_pipeline_caches, device);
    object_count += pipeline_caches.size();
    std::vector<std::shared_ptr<QueryPool> > query_pools = take_device_objects(g_query_pools, device);
    object_count += query_pools.size();
    std::vector<std::shared_ptr<ComputePipeline> > compute_pipelines = take_device_objects(g_compute_pipelines, device);
    object_count += compute_pipelines.size();
    std::vector<std::shared_ptr<DescriptorUpdateTemplate> > descriptor_update_templates = take_device_objects(g_descriptor_update_templates, device);
    object_count += descriptor_update_templates.size();
    std::vector<std::shared_ptr<ShaderModule> > shader_modules = take_device_objects(g_shader_modules, device);
    object_count += shader_modules.size();
    std::vector<std::shared_ptr<PipelineLayout> > pipeline_layouts = take_device_objects(g_pipeline_layouts, device);
    object_count += pipeline_layouts.size();
    std::vector<std::shared_ptr<DescriptorSetLayout> > descriptor_set_layouts = take_device_objects(g_descriptor_set_layouts, device);
    object_count += descriptor_set_layouts.size();
    std::vector<std::shared_ptr<Buffer> > buffers = take_device_objects(g_buffers, device);
    object_count += buffers.size();
    std::vector<std::shared_ptr<DeviceMemory> > memories = take_device_objects(g_memories, device);
    object_count += memories.size();

    if (object_count != 0)
        log_error("device %llu destroyed with %llu live child objects",
                  (unsigned long long)device->id, (unsigned long long)object_count);
}

// timestamp query pools

VkResult impl_create_query_pool(VkDevice device, const VkQueryPoolCreateInfo* create_info,
                                const VkAllocationCallbacks* allocator, VkQueryPool* query_pool)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO || !query_pool)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator || create_info->pNext || create_info->flags != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (!device_impl->enabled_profile.timestamp_query || create_info->queryType != VK_QUERY_TYPE_TIMESTAMP
        || create_info->queryCount == 0 || create_info->pipelineStatistics != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    WGPUQuerySetDescriptor descriptor = WGPU_QUERY_SET_DESCRIPTOR_INIT;
    descriptor.type = WGPUQueryType_Timestamp;
    if (create_info->queryCount > UINT32_MAX / 2)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    // Emdawnwebgpu currently serializes both pass timestamp indices even when
    // one is WGPU_QUERY_SET_INDEX_UNDEFINED. Reserve an internal end index for
    // every Vulkan query so the browser receives two valid indices.
    descriptor.count = create_info->queryCount * 2;

    std::shared_ptr<QueryPool> query_pool_impl = std::make_shared<QueryPool>();
    query_pool_impl->owner = device_impl;
    query_pool_impl->query_set = wgpuDeviceCreateQuerySet(device_impl->device, &descriptor);
    if (!query_pool_impl->query_set)
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    query_pool_impl->values.resize(create_info->queryCount, 0);
    query_pool_impl->available.resize(create_info->queryCount, 0);

    *query_pool = make_handle<VkQueryPool>(g_query_pools, query_pool_impl);
    return VK_SUCCESS;
}

void impl_destroy_query_pool(VkDevice device, VkQueryPool query_pool, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<QueryPool> query_pool_impl = get_handle(g_query_pools, query_pool);
    if (!device_impl || !query_pool_impl || query_pool_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }
    if (query_pool_in_flight(query_pool_impl.get()))
    {
        log_error("device %llu query pool %llu destroyed while in flight",
                  (unsigned long long)device_impl->id, (unsigned long long)query_pool_impl->id);
        return;
    }

    query_pool_impl->live = false;
    erase_handle(g_query_pools, query_pool);
}

VkResult impl_get_query_pool_results(VkDevice device, VkQueryPool query_pool,
                                     uint32_t first_query, uint32_t query_count,
                                     size_t data_size, void* data, VkDeviceSize stride,
                                     VkQueryResultFlags flags)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<QueryPool> query_pool_impl = get_handle(g_query_pools, query_pool);
    const VkQueryResultFlags supported_flags = VK_QUERY_RESULT_64_BIT
                                               | VK_QUERY_RESULT_WAIT_BIT
                                               | VK_QUERY_RESULT_WITH_AVAILABILITY_BIT
                                               | VK_QUERY_RESULT_PARTIAL_BIT;
    if (!device_impl || !query_pool_impl || query_pool_impl->owner != device_impl || !query_pool_impl->live
        || (query_count != 0 && !data) || (flags & ~supported_flags) != 0)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;

    const uint64_t query_end = (uint64_t)first_query + query_count;
    const uint64_t value_size = (flags & VK_QUERY_RESULT_64_BIT) ? sizeof(uint64_t) : sizeof(uint32_t);
    const uint64_t entry_size = value_size * ((flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) ? 2 : 1);
    uint64_t required_size = 0;
    if (query_end > query_pool_impl->values.size()
        || (query_count != 0 && (stride < entry_size || (uint64_t)(query_count - 1) > (UINT64_MAX - entry_size) / stride || !checked_add((uint64_t)(query_count - 1) * stride, entry_size, required_size) || required_size > data_size)))
        return VK_ERROR_INITIALIZATION_FAILED;

    if ((flags & VK_QUERY_RESULT_WAIT_BIT) != 0)
    {
        const VkResult result = wait_device_submissions(device_impl);
        if (result != VK_SUCCESS)
            return result;
    }

    bool all_available = true;
    std::lock_guard<std::mutex> lock(query_pool_impl->mutex);
    for (uint32_t i = 0; i < query_count; i++)
    {
        const uint32_t query = first_query + i;
        const bool available = query_pool_impl->available[query] != 0;
        unsigned char* output = (unsigned char*)data + (uint64_t)i * stride;
        all_available = all_available && available;

        if (available || (flags & VK_QUERY_RESULT_PARTIAL_BIT) != 0)
        {
            if ((flags & VK_QUERY_RESULT_64_BIT) != 0)
            {
                const uint64_t value = available ? query_pool_impl->values[query] : 0;
                memcpy(output, &value, sizeof(value));
            }
            else
            {
                const uint32_t value = available ? (uint32_t)query_pool_impl->values[query] : 0;
                memcpy(output, &value, sizeof(value));
            }
        }

        if ((flags & VK_QUERY_RESULT_WITH_AVAILABILITY_BIT) != 0)
        {
            if ((flags & VK_QUERY_RESULT_64_BIT) != 0)
            {
                const uint64_t value = available ? 1 : 0;
                memcpy(output + value_size, &value, sizeof(value));
            }
            else
            {
                const uint32_t value = available ? 1 : 0;
                memcpy(output + value_size, &value, sizeof(value));
            }
        }
    }

    return all_available ? VK_SUCCESS : VK_NOT_READY;
}

VkResult impl_create_command_pool(VkDevice device, const VkCommandPoolCreateInfo* create_info,
                                  const VkAllocationCallbacks* allocator, VkCommandPool* command_pool)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO || !command_pool)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->queueFamilyIndex != 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    const VkCommandPoolCreateFlags supported_flags = VK_COMMAND_POOL_CREATE_TRANSIENT_BIT | VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    if ((create_info->flags & ~supported_flags) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<CommandPool> command_pool_impl = std::make_shared<CommandPool>();
    command_pool_impl->owner = device_impl;
    command_pool_impl->flags = create_info->flags;
    command_pool_impl->queue_family_index = create_info->queueFamilyIndex;
    *command_pool = make_handle<VkCommandPool>(g_command_pools, command_pool_impl);
    return VK_SUCCESS;
}

void impl_destroy_command_pool(VkDevice device, VkCommandPool command_pool, const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<CommandPool> command_pool_impl = get_handle(g_command_pools, command_pool);
    if (!device_impl || !command_pool_impl || command_pool_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    std::vector<VkCommandBuffer> command_buffers;
    {
        std::lock_guard<std::mutex> lock(command_pool_impl->mutex);
        for (std::unordered_set<VkCommandBuffer>::const_iterator it = command_pool_impl->command_buffers.begin();
             it != command_pool_impl->command_buffers.end(); ++it)
        {
            CommandBuffer* command_buffer = unwrap(*it);
            if (command_buffer && command_buffer->pending_count != 0)
            {
                log_error("command pool %llu destroyed with pending command buffer %llu",
                          (unsigned long long)command_pool_impl->id, (unsigned long long)command_buffer->id);
                return;
            }
        }
        command_buffers.assign(command_pool_impl->command_buffers.begin(), command_pool_impl->command_buffers.end());
        command_pool_impl->command_buffers.clear();
    }
    for (size_t i = 0; i < command_buffers.size(); i++)
        delete command_buffers[i];

    erase_handle(g_command_pools, command_pool);
}

VkResult impl_allocate_command_buffers(VkDevice device, const VkCommandBufferAllocateInfo* allocate_info,
                                       VkCommandBuffer* command_buffers)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !allocate_info || allocate_info->sType != VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO
        || allocate_info->commandBufferCount == 0 || !command_buffers)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocate_info->pNext || allocate_info->level != VK_COMMAND_BUFFER_LEVEL_PRIMARY)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    std::shared_ptr<CommandPool> pool = get_handle(g_command_pools, allocate_info->commandPool);
    if (!pool || pool->owner != device_impl)
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < allocate_info->commandBufferCount; i++)
        command_buffers[i] = VK_NULL_HANDLE;

    for (uint32_t i = 0; i < allocate_info->commandBufferCount; i++)
    {
        std::shared_ptr<CommandBuffer> command_buffer_impl = std::make_shared<CommandBuffer>();
        command_buffer_impl->owner = device_impl;
        command_buffer_impl->pool = pool;
        VkCommandBuffer_T* handle = new VkCommandBuffer_T;
        handle->impl = command_buffer_impl;
        command_buffers[i] = handle;

        std::lock_guard<std::mutex> lock(pool->mutex);
        pool->command_buffers.insert(handle);
    }

    return VK_SUCCESS;
}

void impl_free_command_buffers(VkDevice device, VkCommandPool command_pool, uint32_t command_buffer_count,
                               const VkCommandBuffer* command_buffers)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<CommandPool> pool = get_handle(g_command_pools, command_pool);
    if (!device_impl || !pool || pool->owner != device_impl || (command_buffer_count != 0 && !command_buffers))
        return;

    for (uint32_t i = 0; i < command_buffer_count; i++)
    {
        CommandBuffer* command_buffer = unwrap(command_buffers[i]);
        if (!command_buffer || command_buffer->owner != device_impl || command_buffer->pool.get() != pool.get()
            || command_buffer->pending_count != 0)
            continue;

        {
            std::lock_guard<std::mutex> lock(pool->mutex);
            pool->command_buffers.erase(command_buffers[i]);
        }
        delete command_buffers[i];
    }
}

VkResult impl_reset_command_pool(VkDevice device, VkCommandPool command_pool, VkCommandPoolResetFlags flags)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<CommandPool> pool = get_handle(g_command_pools, command_pool);
    const VkCommandPoolResetFlags supported_flags = VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT;
    if (!device_impl || !pool || pool->owner != device_impl || (flags & ~supported_flags) != 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    std::lock_guard<std::mutex> lock(pool->mutex);
    for (std::unordered_set<VkCommandBuffer>::const_iterator it = pool->command_buffers.begin();
         it != pool->command_buffers.end(); ++it)
    {
        CommandBuffer* command_buffer = unwrap(*it);
        if (!command_buffer || command_buffer->pending_count != 0)
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    for (std::unordered_set<VkCommandBuffer>::const_iterator it = pool->command_buffers.begin();
         it != pool->command_buffers.end(); ++it)
        reset_command_buffer(*unwrap(*it), (flags & VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT) != 0);

    return VK_SUCCESS;
}

void impl_trim_command_pool(VkDevice device, VkCommandPool command_pool, VkCommandPoolTrimFlags flags)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<CommandPool> pool = get_handle(g_command_pools, command_pool);
    if (!device_impl || !pool || pool->owner != device_impl || flags != 0)
    {
        log_error("trim command pool received invalid arguments");
        return;
    }

    std::lock_guard<std::mutex> lock(pool->mutex);
    for (std::unordered_set<VkCommandBuffer>::const_iterator it = pool->command_buffers.begin();
         it != pool->command_buffers.end(); ++it)
    {
        CommandBuffer* command_buffer = unwrap(*it);
        if (!command_buffer)
            continue;
        command_buffer->push_constants.shrink_to_fit();
        command_buffer->commands.shrink_to_fit();
        command_buffer->descriptors.rehash(command_buffer->descriptors.size());
    }
}

VkResult impl_reset_command_buffer(VkCommandBuffer command_buffer, VkCommandBufferResetFlags flags)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    const VkCommandBufferResetFlags supported_flags = VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT;
    if (!command_buffer_impl || (flags & ~supported_flags) != 0
        || command_buffer_impl->pending_count != 0 || !command_buffer_impl->pool
        || (command_buffer_impl->pool->flags & VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT) == 0)
        return VK_ERROR_INITIALIZATION_FAILED;

    reset_command_buffer(*command_buffer_impl, (flags & VK_COMMAND_BUFFER_RESET_RELEASE_RESOURCES_BIT) != 0);
    return VK_SUCCESS;
}

VkResult impl_begin_command_buffer(VkCommandBuffer command_buffer, const VkCommandBufferBeginInfo* begin_info)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    if (!command_buffer_impl || !begin_info || begin_info->sType != VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO
        || begin_info->pNext || begin_info->pInheritanceInfo || command_buffer_impl->pending_count != 0
        || (command_buffer_impl->state != COMMAND_BUFFER_INITIAL && command_buffer_impl->state != COMMAND_BUFFER_EXECUTABLE))
        return VK_ERROR_INITIALIZATION_FAILED;

    const VkCommandBufferUsageFlags supported_flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    if ((begin_info->flags & ~supported_flags) != 0)
        return VK_ERROR_FEATURE_NOT_PRESENT;

    reset_command_buffer(*command_buffer_impl);
    command_buffer_impl->state = COMMAND_BUFFER_RECORDING;
    command_buffer_impl->usage_flags = begin_info->flags;
    return VK_SUCCESS;
}

VkResult impl_end_command_buffer(VkCommandBuffer command_buffer)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (command_buffer_impl->error != VK_SUCCESS)
    {
        command_buffer_impl->state = COMMAND_BUFFER_INVALID;
        return command_buffer_impl->error;
    }

    command_buffer_impl->state = COMMAND_BUFFER_EXECUTABLE;
    return VK_SUCCESS;
}

static void set_command_error(CommandBuffer* command_buffer, VkResult error, const char* operation)
{
    if (!command_buffer || command_buffer->error != VK_SUCCESS)
        return;

    command_buffer->error = error;
    log_error("command buffer %llu %s failed", command_buffer ? (unsigned long long)command_buffer->id : 0, operation);
}

void impl_cmd_bind_pipeline(VkCommandBuffer command_buffer, VkPipelineBindPoint pipeline_bind_point, VkPipeline pipeline)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    std::shared_ptr<ComputePipeline> pipeline_impl = get_handle(g_compute_pipelines, pipeline);
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING || pipeline_bind_point != VK_PIPELINE_BIND_POINT_COMPUTE
        || !pipeline_impl || pipeline_impl->owner != command_buffer_impl->owner || !pipeline_impl->live)
    {
        set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "bind pipeline");
        return;
    }

    command_buffer_impl->current_pipeline = pipeline_impl;
}

static int update_descriptor(CommandBuffer* command_buffer, const DescriptorSetLayout& set_layout,
                             uint32_t binding, VkDescriptorType descriptor_type,
                             const VkDescriptorBufferInfo* buffer_info)
{
    if (!command_buffer
        || (descriptor_type != VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
            && descriptor_type != VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER)
        || !buffer_info)
        return -1;

    const DescriptorBinding* layout_binding = find_layout_binding(set_layout, binding);
    std::shared_ptr<Buffer> buffer = get_handle(g_buffers, buffer_info->buffer);
    if (!layout_binding || layout_binding->descriptor_type != descriptor_type
        || layout_binding->descriptor_count != 1
        || !buffer || buffer->owner != command_buffer->owner || !buffer->live
        || !buffer->memory || !buffer->memory->live
        || buffer_info->range == 0 || buffer_info->offset >= buffer->size)
        return -1;

    const bool uniform = descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
    const VkBufferUsageFlags required_usage = uniform
                                                  ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                                  : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    if ((buffer->usage & required_usage) == 0)
        return -1;

    const uint64_t range = buffer_info->range == VK_WHOLE_SIZE
                               ? buffer->size - buffer_info->offset
                               : buffer_info->range;
    const uint64_t max_binding_size = uniform
                                          ? command_buffer->owner->limits.maxUniformBufferBindingSize
                                          : command_buffer->owner->limits.maxStorageBufferBindingSize;
    const uint64_t offset_alignment = uniform
                                          ? command_buffer->owner->limits.minUniformBufferOffsetAlignment
                                          : command_buffer->owner->limits.minStorageBufferOffsetAlignment;
    uint64_t end = 0;
    if (!checked_add(buffer_info->offset, range, end) || end > buffer->size
        || range > max_binding_size
        || buffer_info->offset % std::max<uint64_t>(4, offset_alignment) != 0)
        return -1;

    DescriptorValue value;
    value.descriptor_type = descriptor_type;
    value.buffer_info = *buffer_info;
    command_buffer->descriptors[binding] = value;
    return 0;
}

void impl_cmd_push_descriptor_set_with_template(VkCommandBuffer command_buffer,
                                                VkDescriptorUpdateTemplate descriptor_update_template,
                                                VkPipelineLayout pipeline_layout, uint32_t set,
                                                const void* data)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    std::shared_ptr<DescriptorUpdateTemplate> template_impl = get_handle(g_descriptor_update_templates, descriptor_update_template);
    std::shared_ptr<PipelineLayout> layout_impl = get_handle(g_pipeline_layouts, pipeline_layout);
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING || !template_impl || !layout_impl
        || (!data && !template_impl->entries.empty())
        || template_impl->owner != command_buffer_impl->owner || layout_impl->owner != command_buffer_impl->owner
        || template_impl->pipeline_layout.get() != layout_impl.get()
        || template_impl->set != set)
    {
        set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "push descriptor template");
        return;
    }

    if (!template_impl->entries.empty())
    {
        if (command_buffer_impl->descriptor_pipeline_layout.get() != layout_impl.get())
            command_buffer_impl->descriptors.clear();
        command_buffer_impl->descriptor_pipeline_layout = layout_impl;
    }

    const unsigned char* bytes = (const unsigned char*)data;
    for (size_t i = 0; i < template_impl->entries.size(); i++)
    {
        const DescriptorTemplateEntry& entry = template_impl->entries[i];
        VkDescriptorBufferInfo buffer_info;
        memcpy(&buffer_info, bytes + entry.offset, sizeof(buffer_info));
        if (update_descriptor(command_buffer_impl, *template_impl->set_layout,
                              entry.binding, entry.descriptor_type, &buffer_info)
            != 0)
        {
            set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "push descriptor template value");
            return;
        }
    }
}

void impl_cmd_push_descriptor_set(VkCommandBuffer command_buffer, VkPipelineBindPoint pipeline_bind_point,
                                  VkPipelineLayout pipeline_layout, uint32_t set, uint32_t descriptor_write_count,
                                  const VkWriteDescriptorSet* descriptor_writes)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    std::shared_ptr<PipelineLayout> layout_impl = get_handle(g_pipeline_layouts, pipeline_layout);
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING || pipeline_bind_point != VK_PIPELINE_BIND_POINT_COMPUTE
        || !layout_impl || layout_impl->owner != command_buffer_impl->owner || set != 0
        || layout_impl->set_layouts.empty() || !layout_impl->set_layouts[0]->push_descriptor
        || (descriptor_write_count != 0 && !descriptor_writes))
    {
        set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "push descriptor set");
        return;
    }

    if (descriptor_write_count != 0)
    {
        if (command_buffer_impl->descriptor_pipeline_layout.get() != layout_impl.get())
            command_buffer_impl->descriptors.clear();
        command_buffer_impl->descriptor_pipeline_layout = layout_impl;
    }

    for (uint32_t i = 0; i < descriptor_write_count; i++)
    {
        const VkWriteDescriptorSet& write = descriptor_writes[i];
        if (write.sType != VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET || write.pNext
            || write.descriptorCount != 1 || write.dstArrayElement != 0
            || update_descriptor(command_buffer_impl, *layout_impl->set_layouts[0],
                                 write.dstBinding, write.descriptorType, write.pBufferInfo)
                   != 0)
        {
            set_command_error(command_buffer_impl, VK_ERROR_FEATURE_NOT_PRESENT, "push descriptor set value");
            return;
        }
    }
}

static bool pipeline_layout_supports_push_constant_range(const PipelineLayout& layout,
                                                         VkShaderStageFlags stage_flags,
                                                         uint32_t offset, uint32_t size)
{
    if (stage_flags != VK_SHADER_STAGE_COMPUTE_BIT || size == 0)
        return false;

    uint64_t end = 0;
    if (!checked_add(offset, size, end))
        return false;

    uint64_t covered = offset;
    while (covered < end)
    {
        uint64_t next = covered;
        for (size_t i = 0; i < layout.push_constant_ranges.size(); i++)
        {
            const PushConstantRange& range = layout.push_constant_ranges[i];
            const uint64_t range_end = (uint64_t)range.offset + range.size;
            if ((range.stage_flags & stage_flags) == stage_flags
                && range.offset <= covered && range_end > next)
                next = range_end;
        }
        if (next == covered)
            return false;
        covered = std::min<uint64_t>(next, end);
    }

    return true;
}

void impl_cmd_push_constants(VkCommandBuffer command_buffer, VkPipelineLayout pipeline_layout,
                             VkShaderStageFlags stage_flags, uint32_t offset, uint32_t size, const void* values)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    std::shared_ptr<PipelineLayout> layout = get_handle(g_pipeline_layouts, pipeline_layout);
    uint64_t end = 0;
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING || !layout || layout->owner != command_buffer_impl->owner
        || !values
        || offset % 4 != 0 || size % 4 != 0 || !checked_add(offset, size, end)
        || end > layout->push_constant_size
        || !pipeline_layout_supports_push_constant_range(*layout, stage_flags, offset, size))
    {
        set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "push constants");
        return;
    }

    if (command_buffer_impl->push_constant_pipeline_layout.get() != layout.get())
        command_buffer_impl->push_constants.clear();
    command_buffer_impl->push_constant_pipeline_layout = layout;

    if (command_buffer_impl->push_constants.size() < layout->push_constant_size)
        command_buffer_impl->push_constants.resize(layout->push_constant_size, 0);
    memcpy(command_buffer_impl->push_constants.data() + offset, values, size);
}

static int resolve_dispatch_binding(CommandBuffer& command_buffer, const ActiveBinding& active,
                                    ResolvedBinding& resolved)
{
    std::unordered_map<uint32_t, DescriptorValue>::const_iterator it = command_buffer.descriptors.find(active.binding);
    if (it == command_buffer.descriptors.end())
        return -1;
    if (it->second.descriptor_type != active.descriptor_type)
        return -1;

    const VkDescriptorBufferInfo& info = it->second.buffer_info;
    std::shared_ptr<Buffer> buffer = get_handle(g_buffers, info.buffer);
    if (!buffer || buffer->owner != command_buffer.owner || !buffer->live
        || !buffer->memory || !buffer->memory->live)
        return -1;
    if (info.offset > buffer->size)
        return -1;

    const uint64_t size = info.range == VK_WHOLE_SIZE ? buffer->size - info.offset : info.range;
    uint64_t buffer_end = 0;
    uint64_t memory_offset = 0;
    uint64_t memory_end = 0;
    if (size < active.min_binding_size
        || !checked_add(info.offset, size, buffer_end) || buffer_end > buffer->size
        || !checked_add(buffer->memory_offset, info.offset, memory_offset)
        || !checked_add(memory_offset, size, memory_end) || memory_end > buffer->memory->size)
        return -1;

    resolved.binding = active.binding;
    resolved.descriptor_type = active.descriptor_type;
    resolved.access = active.access;
    resolved.layout_access = active.layout_access;
    resolved.min_binding_size = active.min_binding_size;
    resolved.buffer = buffer;
    resolved.memory = buffer->memory;
    resolved.offset = memory_offset;
    resolved.size = size;
    return 0;
}

static bool scale_dispatch_grid(const ComputePipeline& pipeline, uint32_t limit,
                                uint32_t group_count_x, uint32_t group_count_y,
                                uint32_t group_count_z, uint32_t physical[3])
{
    const uint64_t scaled_x = (uint64_t)group_count_x * pipeline.dispatch_scale_x;
    const uint64_t scaled_y = (uint64_t)group_count_y * pipeline.dispatch_scale_y;
    const uint64_t scaled_z = (uint64_t)group_count_z * pipeline.dispatch_scale_z;
    if (scaled_x > limit || scaled_y > limit || scaled_z > limit)
        return false;

    physical[0] = (uint32_t)scaled_x;
    physical[1] = (uint32_t)scaled_y;
    physical[2] = (uint32_t)scaled_z;
    return true;
}

void impl_cmd_dispatch(VkCommandBuffer command_buffer, uint32_t group_count_x,
                       uint32_t group_count_y, uint32_t group_count_z)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING || !command_buffer_impl->current_pipeline)
    {
        set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "dispatch pipeline");
        return;
    }

    const PipelineLayout& pipeline_layout = *command_buffer_impl->current_pipeline->vk_pipeline_layout;
    bool has_external_binding = false;
    for (size_t i = 0; i < command_buffer_impl->current_pipeline->active_bindings.size(); i++)
    {
        if (!command_buffer_impl->current_pipeline->active_bindings[i].internal_uniform)
        {
            has_external_binding = true;
            break;
        }
    }
    if (has_external_binding
        && command_buffer_impl->descriptor_pipeline_layout.get() != &pipeline_layout)
    {
        set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "dispatch descriptor layout");
        return;
    }
    if (command_buffer_impl->current_pipeline->requires_push_constants
        && command_buffer_impl->push_constant_pipeline_layout.get() != &pipeline_layout)
    {
        set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "dispatch push constant layout");
        return;
    }

    const WGPULimits& limits = command_buffer_impl->owner->limits;
    uint32_t physical_group_count[3];
    if (!scale_dispatch_grid(*command_buffer_impl->current_pipeline,
                             limits.maxComputeWorkgroupsPerDimension,
                             group_count_x, group_count_y, group_count_z,
                             physical_group_count))
    {
        command_buffer_impl->error = VK_ERROR_INITIALIZATION_FAILED;
        log_error("command buffer %llu pipeline %llu dispatch size failed groups=%u,%u,%u limit=%u local=%u,%u,%u",
                  (unsigned long long)command_buffer_impl->id,
                  (unsigned long long)command_buffer_impl->current_pipeline->id,
                  group_count_x, group_count_y, group_count_z,
                  limits.maxComputeWorkgroupsPerDimension,
                  command_buffer_impl->current_pipeline->workgroup_size_x,
                  command_buffer_impl->current_pipeline->workgroup_size_y,
                  command_buffer_impl->current_pipeline->workgroup_size_z);
        return;
    }

    Command command;
    command.type = COMMAND_DISPATCH;
    command.pipeline = command_buffer_impl->current_pipeline;
    command.group_count_x = physical_group_count[0];
    command.group_count_y = physical_group_count[1];
    command.group_count_z = physical_group_count[2];

    for (size_t i = 0; i < command_buffer_impl->current_pipeline->active_bindings.size(); i++)
    {
        const ActiveBinding& active = command_buffer_impl->current_pipeline->active_bindings[i];
        if (active.internal_uniform)
            continue;

        ResolvedBinding binding;
        if (resolve_dispatch_binding(*command_buffer_impl, active, binding) != 0)
        {
            set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "dispatch descriptor");
            return;
        }
        command.bindings.push_back(binding);
    }

    command.push_constants.resize(command_buffer_impl->current_pipeline->push_constant_data_size, 0);
    const size_t push_size = std::min(command.push_constants.size(), command_buffer_impl->push_constants.size());
    if (push_size != 0)
        memcpy(command.push_constants.data(), command_buffer_impl->push_constants.data(), push_size);

    command_buffer_impl->commands.push_back(command);
}

void impl_cmd_copy_buffer(VkCommandBuffer command_buffer, VkBuffer src_buffer, VkBuffer dst_buffer,
                          uint32_t region_count, const VkBufferCopy* regions)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    std::shared_ptr<Buffer> src = get_handle(g_buffers, src_buffer);
    std::shared_ptr<Buffer> dst = get_handle(g_buffers, dst_buffer);
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING || !src || !dst
        || src->owner != command_buffer_impl->owner || dst->owner != command_buffer_impl->owner
        || !src->live || !dst->live
        || !src->memory || !dst->memory || !src->memory->live || !dst->memory->live
        || region_count == 0 || !regions)
    {
        set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "copy buffer");
        return;
    }

    Command command;
    command.type = COMMAND_COPY_BUFFER;
    command.src_buffer = src;
    command.dst_buffer = dst;
    for (uint32_t i = 0; i < region_count; i++)
    {
        uint64_t src_end = 0;
        uint64_t dst_end = 0;
        if (regions[i].size == 0
            || !checked_add(regions[i].srcOffset, regions[i].size, src_end) || src_end > src->size
            || !checked_add(regions[i].dstOffset, regions[i].size, dst_end) || dst_end > dst->size
            || regions[i].srcOffset % 4 != 0 || regions[i].dstOffset % 4 != 0 || regions[i].size % 4 != 0)
        {
            set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "copy buffer range");
            return;
        }

        CopyRegion region;
        region.src_offset = regions[i].srcOffset;
        region.dst_offset = regions[i].dstOffset;
        region.size = regions[i].size;
        command.copy_regions.push_back(region);
    }

    if (src->memory.get() == dst->memory.get())
    {
        for (size_t i = 0; i < command.copy_regions.size(); i++)
        {
            uint64_t src_offset = 0;
            if (!checked_add(src->memory_offset, command.copy_regions[i].src_offset, src_offset))
            {
                set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "copy buffer source range");
                return;
            }

            for (size_t j = 0; j < command.copy_regions.size(); j++)
            {
                uint64_t dst_offset = 0;
                if (!checked_add(dst->memory_offset, command.copy_regions[j].dst_offset, dst_offset)
                    || ranges_overlap(src_offset, command.copy_regions[i].size,
                                      dst_offset, command.copy_regions[j].size))
                {
                    set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "copy buffer overlap");
                    return;
                }
            }
        }
    }

    command_buffer_impl->commands.push_back(command);
}

void impl_cmd_reset_query_pool(VkCommandBuffer command_buffer, VkQueryPool query_pool,
                               uint32_t first_query, uint32_t query_count)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    std::shared_ptr<QueryPool> query_pool_impl = get_handle(g_query_pools, query_pool);
    const uint64_t query_end = (uint64_t)first_query + query_count;
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING
        || !query_pool_impl || query_pool_impl->owner != command_buffer_impl->owner || !query_pool_impl->live
        || query_count == 0 || query_end > query_pool_impl->values.size())
    {
        set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "reset query pool");
        return;
    }

    Command command;
    command.type = COMMAND_RESET_QUERY_POOL;
    command.query_pool = query_pool_impl;
    command.first_query = first_query;
    command.query_count = query_count;
    command_buffer_impl->commands.push_back(command);
}

void impl_cmd_write_timestamp(VkCommandBuffer command_buffer, VkPipelineStageFlagBits pipeline_stage,
                              VkQueryPool query_pool, uint32_t query)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    std::shared_ptr<QueryPool> query_pool_impl = get_handle(g_query_pools, query_pool);
    const VkPipelineStageFlags supported_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                  | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                                  | VK_PIPELINE_STAGE_TRANSFER_BIT
                                                  | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
                                                  | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING
        || !query_pool_impl || query_pool_impl->owner != command_buffer_impl->owner || !query_pool_impl->live
        || query >= query_pool_impl->values.size() || (pipeline_stage & supported_stages) == 0
        || ((VkPipelineStageFlags)pipeline_stage & ~supported_stages) != 0)
    {
        set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "write timestamp");
        return;
    }

    Command command;
    command.type = COMMAND_WRITE_TIMESTAMP;
    command.query_pool = query_pool_impl;
    command.first_query = query;
    command.query_count = 1;
    command_buffer_impl->commands.push_back(command);
}

void impl_cmd_copy_query_pool_results(VkCommandBuffer command_buffer, VkQueryPool query_pool,
                                      uint32_t first_query, uint32_t query_count,
                                      VkBuffer dst_buffer, VkDeviceSize dst_offset,
                                      VkDeviceSize stride, VkQueryResultFlags flags)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    std::shared_ptr<QueryPool> query_pool_impl = get_handle(g_query_pools, query_pool);
    std::shared_ptr<Buffer> dst_buffer_impl = get_handle(g_buffers, dst_buffer);
    const uint64_t query_end = (uint64_t)first_query + query_count;
    uint64_t result_size = 0;
    uint64_t result_end = 0;
    bool size_valid = query_count != 0 && stride >= sizeof(uint64_t)
                      && (uint64_t)(query_count - 1) <= (UINT64_MAX - sizeof(uint64_t)) / stride;
    if (size_valid)
    {
        result_size = (uint64_t)(query_count - 1) * stride + sizeof(uint64_t);
        size_valid = checked_add(dst_offset, result_size, result_end);
    }
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING
        || !query_pool_impl || query_pool_impl->owner != command_buffer_impl->owner || !query_pool_impl->live
        || !dst_buffer_impl || dst_buffer_impl->owner != command_buffer_impl->owner || !dst_buffer_impl->live
        || !dst_buffer_impl->memory || !dst_buffer_impl->memory->live
        || query_end > query_pool_impl->values.size() || !size_valid || result_end > dst_buffer_impl->size
        || flags != VK_QUERY_RESULT_64_BIT || dst_offset % 8 != 0 || stride != sizeof(uint64_t))
    {
        set_command_error(command_buffer_impl, VK_ERROR_FEATURE_NOT_PRESENT, "copy query pool results");
        return;
    }

    Command command;
    command.type = COMMAND_COPY_QUERY_POOL_RESULTS;
    command.query_pool = query_pool_impl;
    command.first_query = first_query;
    command.query_count = query_count;
    command.query_result_buffer = dst_buffer_impl;
    command.query_result_offset = dst_offset;
    command.query_result_stride = stride;
    command.query_result_flags = flags;
    command_buffer_impl->commands.push_back(command);
}

void impl_cmd_pipeline_barrier(VkCommandBuffer command_buffer,
                               VkPipelineStageFlags src_stage_mask,
                               VkPipelineStageFlags dst_stage_mask,
                               VkDependencyFlags dependency_flags,
                               uint32_t memory_barrier_count,
                               const VkMemoryBarrier* memory_barriers, uint32_t buffer_memory_barrier_count,
                               const VkBufferMemoryBarrier* buffer_memory_barriers,
                               uint32_t image_memory_barrier_count, const VkImageMemoryBarrier*)
{
    CommandBuffer* command_buffer_impl = unwrap(command_buffer);
    const VkPipelineStageFlags supported_stages = VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT
                                                  | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT
                                                  | VK_PIPELINE_STAGE_TRANSFER_BIT
                                                  | VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT
                                                  | VK_PIPELINE_STAGE_HOST_BIT
                                                  | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT;
    const VkAccessFlags supported_access = VK_ACCESS_UNIFORM_READ_BIT
                                           | VK_ACCESS_SHADER_READ_BIT
                                           | VK_ACCESS_SHADER_WRITE_BIT
                                           | VK_ACCESS_TRANSFER_READ_BIT
                                           | VK_ACCESS_TRANSFER_WRITE_BIT
                                           | VK_ACCESS_HOST_READ_BIT
                                           | VK_ACCESS_HOST_WRITE_BIT
                                           | VK_ACCESS_MEMORY_READ_BIT
                                           | VK_ACCESS_MEMORY_WRITE_BIT;
    if (!command_buffer_impl || command_buffer_impl->state != COMMAND_BUFFER_RECORDING
        || src_stage_mask == 0 || dst_stage_mask == 0
        || (src_stage_mask & ~supported_stages) != 0
        || (dst_stage_mask & ~supported_stages) != 0
        || dependency_flags != 0
        || (memory_barrier_count != 0 && !memory_barriers)
        || (buffer_memory_barrier_count != 0 && !buffer_memory_barriers)
        || image_memory_barrier_count != 0)
    {
        set_command_error(command_buffer_impl, VK_ERROR_FEATURE_NOT_PRESENT, "pipeline barrier");
        return;
    }

    Command command;
    command.type = COMMAND_BARRIER;
    for (uint32_t i = 0; i < memory_barrier_count; i++)
    {
        if (memory_barriers[i].sType != VK_STRUCTURE_TYPE_MEMORY_BARRIER
            || memory_barriers[i].pNext
            || (memory_barriers[i].srcAccessMask & ~supported_access) != 0
            || (memory_barriers[i].dstAccessMask & ~supported_access) != 0)
        {
            set_command_error(command_buffer_impl, VK_ERROR_FEATURE_NOT_PRESENT, "memory barrier");
            return;
        }
    }
    for (uint32_t i = 0; i < buffer_memory_barrier_count; i++)
    {
        const VkBufferMemoryBarrier& barrier = buffer_memory_barriers[i];
        std::shared_ptr<Buffer> buffer = get_handle(g_buffers, barrier.buffer);
        if (barrier.sType != VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER
            || barrier.pNext
            || !buffer || buffer->owner != command_buffer_impl->owner || !buffer->live
            || !buffer->memory || !buffer->memory->live
            || (barrier.srcAccessMask & ~supported_access) != 0
            || (barrier.dstAccessMask & ~supported_access) != 0
            || barrier.srcQueueFamilyIndex != barrier.dstQueueFamilyIndex
            || (barrier.srcQueueFamilyIndex != VK_QUEUE_FAMILY_IGNORED
                && barrier.srcQueueFamilyIndex != 0))
        {
            set_command_error(command_buffer_impl, VK_ERROR_FEATURE_NOT_PRESENT, "buffer barrier");
            return;
        }

        const uint64_t size = barrier.size == VK_WHOLE_SIZE
                                  ? buffer->size - std::min<uint64_t>(barrier.offset, buffer->size)
                                  : barrier.size;
        uint64_t end = 0;
        if (size == 0 || !checked_add(barrier.offset, size, end) || end > buffer->size)
        {
            set_command_error(command_buffer_impl, VK_ERROR_INITIALIZATION_FAILED, "buffer barrier range");
            return;
        }
        command.barrier_buffers.push_back(buffer);
    }
    command_buffer_impl->commands.push_back(command);
}

} // namespace nikki_detail

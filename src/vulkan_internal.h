// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_VULKAN_INTERNAL_H
#define NIKKI_VULKAN_INTERNAL_H

#include "common.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace nikki_detail {

// common helpers

inline std::atomic<uint64_t> g_next_object_id(1);

inline uint64_t next_object_id()
{
    return g_next_object_id.fetch_add(1, std::memory_order_relaxed);
}

inline uint64_t monotonic_time_ns()
{
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
               std::chrono::steady_clock::now().time_since_epoch())
        .count();
}

inline uint64_t remaining_timeout_ns(uint64_t begin, uint64_t timeout)
{
    if (timeout == UINT64_MAX)
        return UINT64_MAX;

    const uint64_t now = monotonic_time_ns();
    const uint64_t elapsed = now >= begin ? now - begin : 0;
    return elapsed >= timeout ? 0 : timeout - elapsed;
}

inline uint64_t align_up(uint64_t value, uint64_t alignment)
{
    if (alignment <= 1)
        return value;

    const uint64_t remainder = value % alignment;
    if (remainder == 0)
        return value;
    if (value > std::numeric_limits<uint64_t>::max() - (alignment - remainder))
        return 0;

    return value + alignment - remainder;
}

inline bool checked_add(uint64_t a, uint64_t b, uint64_t& result)
{
    if (a > std::numeric_limits<uint64_t>::max() - b)
        return false;

    result = a + b;
    return true;
}

inline bool is_power_of_two(uint32_t value)
{
    return value != 0 && (value & (value - 1)) == 0;
}

inline bool ranges_overlap(uint64_t a_offset, uint64_t a_size, uint64_t b_offset, uint64_t b_size)
{
    uint64_t a_end = 0;
    uint64_t b_end = 0;
    return checked_add(a_offset, a_size, a_end) && checked_add(b_offset, b_size, b_end)
           && a_offset < b_end && b_offset < a_end;
}

struct ByteRange
{
    uint64_t offset;
    uint64_t size;
};

inline bool add_byte_range(std::vector<ByteRange>& ranges, uint64_t offset, uint64_t size)
{
    uint64_t end = 0;
    if (size == 0 || !checked_add(offset, size, end))
        return false;

    for (size_t i = 0; i < ranges.size();)
    {
        uint64_t range_end = 0;
        if (!checked_add(ranges[i].offset, ranges[i].size, range_end))
            return false;
        if (end < ranges[i].offset || range_end < offset)
        {
            i++;
            continue;
        }

        offset = std::min(offset, ranges[i].offset);
        end = std::max(end, range_end);
        ranges.erase(ranges.begin() + i);
        i = 0;
    }

    ByteRange range;
    range.offset = offset;
    range.size = end - offset;
    ranges.push_back(range);
    return true;
}

inline void subtract_byte_range(std::vector<ByteRange>& ranges, uint64_t offset, uint64_t size)
{
    uint64_t end = 0;
    if (size == 0 || !checked_add(offset, size, end))
        return;

    std::vector<ByteRange> remaining;
    for (size_t i = 0; i < ranges.size(); i++)
    {
        uint64_t range_end = 0;
        if (!checked_add(ranges[i].offset, ranges[i].size, range_end))
            continue;
        if (end <= ranges[i].offset || range_end <= offset)
        {
            remaining.push_back(ranges[i]);
            continue;
        }

        if (ranges[i].offset < offset)
        {
            ByteRange left;
            left.offset = ranges[i].offset;
            left.size = offset - ranges[i].offset;
            remaining.push_back(left);
        }
        if (range_end > end)
        {
            ByteRange right;
            right.offset = end;
            right.size = range_end - end;
            remaining.push_back(right);
        }
    }
    ranges.swap(remaining);
}

template<typename Handle>
inline uint64_t handle_to_id(Handle handle)
{
    if constexpr (std::is_pointer<Handle>::value)
        return (uint64_t)(uintptr_t)handle;
    else
        return (uint64_t)handle;
}

template<typename Handle>
inline Handle id_to_handle(uint64_t id)
{
    if constexpr (std::is_pointer<Handle>::value)
        return (Handle)(uintptr_t)id;
    else
        return (Handle)id;
}

template<typename T>
class HandleTable
{
public:
    uint64_t insert(const std::shared_ptr<T>& object)
    {
        const uint64_t id = next_object_id();
        std::lock_guard<std::mutex> lock(mutex);
        objects[id] = object;
        return id;
    }

    std::shared_ptr<T> get(uint64_t id) const
    {
        if (id == 0)
            return std::shared_ptr<T>();

        std::lock_guard<std::mutex> lock(mutex);
        typename std::unordered_map<uint64_t, std::shared_ptr<T> >::const_iterator it = objects.find(id);
        if (it == objects.end())
            return std::shared_ptr<T>();

        return it->second;
    }

    void erase(uint64_t id)
    {
        if (id == 0)
            return;

        std::lock_guard<std::mutex> lock(mutex);
        objects.erase(id);
    }

    template<typename Predicate>
    std::vector<std::shared_ptr<T> > take_if(Predicate predicate)
    {
        std::vector<std::shared_ptr<T> > values;
        std::lock_guard<std::mutex> lock(mutex);
        typename std::unordered_map<uint64_t, std::shared_ptr<T> >::iterator it = objects.begin();
        while (it != objects.end())
        {
            if (!predicate(it->second))
            {
                ++it;
                continue;
            }

            values.push_back(it->second);
            it = objects.erase(it);
        }
        return values;
    }

private:
    mutable std::mutex mutex;
    std::unordered_map<uint64_t, std::shared_ptr<T> > objects;
};

enum BufferAccess
{
    BUFFER_ACCESS_READ,
    BUFFER_ACCESS_READ_WRITE
};

enum CommandBufferState
{
    COMMAND_BUFFER_INITIAL,
    COMMAND_BUFFER_RECORDING,
    COMMAND_BUFFER_EXECUTABLE,
    COMMAND_BUFFER_PENDING,
    COMMAND_BUFFER_INVALID
};

enum CommandType
{
    COMMAND_DISPATCH,
    COMMAND_COPY_BUFFER,
    COMMAND_BARRIER,
    COMMAND_RESET_QUERY_POOL,
    COMMAND_WRITE_TIMESTAMP,
    COMMAND_COPY_QUERY_POOL_RESULTS
};

struct Instance;
struct TranslatorInstanceState;
struct PhysicalDevice;
struct Device;
struct DeviceMemory;
struct Buffer;
struct DescriptorSetLayout;
struct DescriptorUpdateTemplate;
struct PipelineLayout;
struct ShaderModule;
struct ComputePipeline;
struct CommandPool;
struct CommandBuffer;
struct Fence;
struct PipelineCache;
struct QueryPool;
struct Submission;

} // namespace nikki_detail

struct VkInstance_T
{
    std::shared_ptr<nikki_detail::Instance> impl;
};

struct VkPhysicalDevice_T
{
    std::shared_ptr<nikki_detail::PhysicalDevice> impl;
};

struct VkDevice_T
{
    std::shared_ptr<nikki_detail::Device> impl;
};

struct VkQueue_T
{
    nikki_detail::Device* device;
};

struct VkCommandBuffer_T
{
    std::shared_ptr<nikki_detail::CommandBuffer> impl;
};

namespace nikki_detail {

VkResult impl_enumerate_instance_extension_properties(const char* layer_name, uint32_t* property_count, VkExtensionProperties* properties);

VkResult impl_enumerate_instance_layer_properties(uint32_t* property_count, VkLayerProperties*);

VkResult impl_enumerate_instance_version(uint32_t* api_version);

VkResult impl_create_instance(const VkInstanceCreateInfo* create_info, const VkAllocationCallbacks* allocator, VkInstance* instance);

void impl_destroy_instance(VkInstance instance, const VkAllocationCallbacks* allocator);

VkResult impl_enumerate_physical_devices(VkInstance instance, uint32_t* physical_device_count, VkPhysicalDevice* physical_devices);

void impl_get_physical_device_features(VkPhysicalDevice physical_device, VkPhysicalDeviceFeatures* features);

void impl_get_physical_device_properties(VkPhysicalDevice physical_device, VkPhysicalDeviceProperties* properties);

void impl_get_physical_device_queue_family_properties(VkPhysicalDevice physical_device, uint32_t* property_count,
                                                      VkQueueFamilyProperties* properties);

void impl_get_physical_device_memory_properties(VkPhysicalDevice physical_device, VkPhysicalDeviceMemoryProperties* properties);

void impl_get_physical_device_format_properties(VkPhysicalDevice physical_device, VkFormat, VkFormatProperties* properties);

VkResult impl_get_physical_device_image_format_properties(VkPhysicalDevice, VkFormat, VkImageType, VkImageTiling,
                                                          VkImageUsageFlags, VkImageCreateFlags, VkImageFormatProperties*);

void impl_get_physical_device_features2(VkPhysicalDevice physical_device, VkPhysicalDeviceFeatures2KHR* features);

void impl_get_physical_device_properties2(VkPhysicalDevice physical_device, VkPhysicalDeviceProperties2KHR* properties);

VkResult impl_get_physical_device_cooperative_matrix_properties(
    VkPhysicalDevice physical_device, uint32_t* property_count, VkCooperativeMatrixPropertiesKHR* properties);

void impl_get_physical_device_format_properties2(VkPhysicalDevice physical_device, VkFormat format,
                                                 VkFormatProperties2KHR* properties);

VkResult impl_get_physical_device_image_format_properties2(
    VkPhysicalDevice physical_device, const VkPhysicalDeviceImageFormatInfo2KHR* format_info,
    VkImageFormatProperties2KHR* properties);

void impl_get_physical_device_queue_family_properties2(
    VkPhysicalDevice physical_device, uint32_t* property_count, VkQueueFamilyProperties2KHR* properties);

void impl_get_physical_device_memory_properties2(
    VkPhysicalDevice physical_device, VkPhysicalDeviceMemoryProperties2KHR* properties);

VkResult impl_enumerate_device_extension_properties(VkPhysicalDevice physical_device, const char* layer_name,
                                                    uint32_t* property_count, VkExtensionProperties* properties);

VkResult impl_enumerate_device_layer_properties(VkPhysicalDevice physical_device, uint32_t* property_count, VkLayerProperties*);

VkResult impl_create_device(VkPhysicalDevice physical_device, const VkDeviceCreateInfo* create_info,
                            const VkAllocationCallbacks* allocator, VkDevice* device);

void impl_destroy_device(VkDevice device, const VkAllocationCallbacks* allocator);

void impl_get_device_queue(VkDevice device, uint32_t queue_family_index, uint32_t queue_index, VkQueue* queue);

VkResult impl_allocate_memory(VkDevice device, const VkMemoryAllocateInfo* allocate_info,
                              const VkAllocationCallbacks* allocator, VkDeviceMemory* memory);

void impl_free_memory(VkDevice device, VkDeviceMemory memory, const VkAllocationCallbacks* allocator);

VkResult impl_map_memory(VkDevice device, VkDeviceMemory memory, VkDeviceSize offset, VkDeviceSize size,
                         VkMemoryMapFlags flags, void** data);

void impl_unmap_memory(VkDevice device, VkDeviceMemory memory);

VkResult impl_flush_mapped_memory_ranges(VkDevice device, uint32_t range_count, const VkMappedMemoryRange* ranges);

VkResult impl_invalidate_mapped_memory_ranges(VkDevice device, uint32_t range_count, const VkMappedMemoryRange* ranges);

void impl_get_device_memory_commitment(VkDevice device, VkDeviceMemory memory, VkDeviceSize* committed_memory);

VkResult impl_create_buffer(VkDevice device, const VkBufferCreateInfo* create_info,
                            const VkAllocationCallbacks* allocator, VkBuffer* buffer);

void impl_destroy_buffer(VkDevice device, VkBuffer buffer, const VkAllocationCallbacks* allocator);

void impl_get_buffer_memory_requirements(VkDevice device, VkBuffer buffer, VkMemoryRequirements* requirements);

void impl_get_buffer_memory_requirements2(VkDevice device,
                                          const VkBufferMemoryRequirementsInfo2* info,
                                          VkMemoryRequirements2* requirements);

VkResult impl_bind_buffer_memory(VkDevice device, VkBuffer buffer, VkDeviceMemory memory, VkDeviceSize memory_offset);

VkResult impl_bind_buffer_memory2(VkDevice device, uint32_t bind_info_count,
                                  const VkBindBufferMemoryInfo* bind_infos);

VkResult impl_create_descriptor_set_layout(VkDevice device, const VkDescriptorSetLayoutCreateInfo* create_info,
                                           const VkAllocationCallbacks* allocator, VkDescriptorSetLayout* set_layout);

void impl_get_descriptor_set_layout_support(
    VkDevice device, const VkDescriptorSetLayoutCreateInfo* create_info,
    VkDescriptorSetLayoutSupport* support);

void impl_destroy_descriptor_set_layout(VkDevice device, VkDescriptorSetLayout set_layout,
                                        const VkAllocationCallbacks* allocator);

VkResult impl_create_pipeline_layout(VkDevice device, const VkPipelineLayoutCreateInfo* create_info,
                                     const VkAllocationCallbacks* allocator, VkPipelineLayout* pipeline_layout);

void impl_destroy_pipeline_layout(VkDevice device, VkPipelineLayout pipeline_layout,
                                  const VkAllocationCallbacks* allocator);

VkResult impl_create_shader_module(VkDevice device, const VkShaderModuleCreateInfo* create_info,
                                   const VkAllocationCallbacks* allocator, VkShaderModule* shader_module);

void impl_destroy_shader_module(VkDevice device, VkShaderModule shader_module, const VkAllocationCallbacks* allocator);

VkResult impl_create_compute_pipelines(VkDevice device, VkPipelineCache pipeline_cache, uint32_t create_info_count,
                                       const VkComputePipelineCreateInfo* create_infos,
                                       const VkAllocationCallbacks* allocator, VkPipeline* pipelines);

void impl_destroy_pipeline(VkDevice device, VkPipeline pipeline, const VkAllocationCallbacks* allocator);

VkResult impl_create_descriptor_update_template(VkDevice device,
                                                const VkDescriptorUpdateTemplateCreateInfo* create_info,
                                                const VkAllocationCallbacks* allocator,
                                                VkDescriptorUpdateTemplate* descriptor_update_template);

void impl_destroy_descriptor_update_template(VkDevice device,
                                             VkDescriptorUpdateTemplate descriptor_update_template,
                                             const VkAllocationCallbacks* allocator);

VkResult impl_create_query_pool(VkDevice device, const VkQueryPoolCreateInfo* create_info,
                                const VkAllocationCallbacks* allocator, VkQueryPool* query_pool);

void impl_destroy_query_pool(VkDevice device, VkQueryPool query_pool, const VkAllocationCallbacks* allocator);

VkResult impl_get_query_pool_results(VkDevice device, VkQueryPool query_pool,
                                     uint32_t first_query, uint32_t query_count,
                                     size_t data_size, void* data, VkDeviceSize stride,
                                     VkQueryResultFlags flags);

VkResult impl_create_command_pool(VkDevice device, const VkCommandPoolCreateInfo* create_info,
                                  const VkAllocationCallbacks* allocator, VkCommandPool* command_pool);

void impl_destroy_command_pool(VkDevice device, VkCommandPool command_pool, const VkAllocationCallbacks* allocator);

VkResult impl_allocate_command_buffers(VkDevice device, const VkCommandBufferAllocateInfo* allocate_info,
                                       VkCommandBuffer* command_buffers);

void impl_free_command_buffers(VkDevice device, VkCommandPool command_pool, uint32_t command_buffer_count,
                               const VkCommandBuffer* command_buffers);

VkResult impl_reset_command_pool(VkDevice device, VkCommandPool command_pool, VkCommandPoolResetFlags flags);

void impl_trim_command_pool(VkDevice device, VkCommandPool command_pool, VkCommandPoolTrimFlags flags);

VkResult impl_reset_command_buffer(VkCommandBuffer command_buffer, VkCommandBufferResetFlags flags);

VkResult impl_begin_command_buffer(VkCommandBuffer command_buffer, const VkCommandBufferBeginInfo* begin_info);

VkResult impl_end_command_buffer(VkCommandBuffer command_buffer);

void impl_cmd_bind_pipeline(VkCommandBuffer command_buffer, VkPipelineBindPoint pipeline_bind_point, VkPipeline pipeline);

void impl_cmd_push_descriptor_set_with_template(VkCommandBuffer command_buffer,
                                                VkDescriptorUpdateTemplate descriptor_update_template,
                                                VkPipelineLayout pipeline_layout, uint32_t set,
                                                const void* data);

void impl_cmd_push_descriptor_set(VkCommandBuffer command_buffer, VkPipelineBindPoint pipeline_bind_point,
                                  VkPipelineLayout pipeline_layout, uint32_t set, uint32_t descriptor_write_count,
                                  const VkWriteDescriptorSet* descriptor_writes);

void impl_cmd_push_constants(VkCommandBuffer command_buffer, VkPipelineLayout pipeline_layout,
                             VkShaderStageFlags stage_flags, uint32_t offset, uint32_t size, const void* values);

void impl_cmd_dispatch(VkCommandBuffer command_buffer, uint32_t group_count_x,
                       uint32_t group_count_y, uint32_t group_count_z);

void impl_cmd_copy_buffer(VkCommandBuffer command_buffer, VkBuffer src_buffer, VkBuffer dst_buffer,
                          uint32_t region_count, const VkBufferCopy* regions);

void impl_cmd_reset_query_pool(VkCommandBuffer command_buffer, VkQueryPool query_pool,
                               uint32_t first_query, uint32_t query_count);

void impl_cmd_write_timestamp(VkCommandBuffer command_buffer, VkPipelineStageFlagBits pipeline_stage,
                              VkQueryPool query_pool, uint32_t query);

void impl_cmd_copy_query_pool_results(VkCommandBuffer command_buffer, VkQueryPool query_pool,
                                      uint32_t first_query, uint32_t query_count,
                                      VkBuffer dst_buffer, VkDeviceSize dst_offset,
                                      VkDeviceSize stride, VkQueryResultFlags flags);

void impl_cmd_pipeline_barrier(VkCommandBuffer command_buffer,
                               VkPipelineStageFlags src_stage_mask,
                               VkPipelineStageFlags dst_stage_mask,
                               VkDependencyFlags dependency_flags,
                               uint32_t memory_barrier_count,
                               const VkMemoryBarrier* memory_barriers, uint32_t buffer_memory_barrier_count,
                               const VkBufferMemoryBarrier* buffer_memory_barriers,
                               uint32_t image_memory_barrier_count, const VkImageMemoryBarrier*);

VkResult impl_queue_submit(VkQueue queue, uint32_t submit_count, const VkSubmitInfo* submits, VkFence fence);

VkResult impl_queue_wait_idle(VkQueue queue);

VkResult impl_device_wait_idle(VkDevice device);

VkResult impl_create_fence(VkDevice device, const VkFenceCreateInfo* create_info,
                           const VkAllocationCallbacks* allocator, VkFence* fence);

void impl_destroy_fence(VkDevice device, VkFence fence, const VkAllocationCallbacks* allocator);

VkResult impl_get_fence_status(VkDevice device, VkFence fence);

VkResult impl_reset_fences(VkDevice device, uint32_t fence_count, const VkFence* fences);

VkResult impl_wait_for_fences(VkDevice device, uint32_t fence_count, const VkFence* fences,
                              VkBool32 wait_all, uint64_t timeout);

VkResult impl_create_pipeline_cache(VkDevice device, const VkPipelineCacheCreateInfo* create_info,
                                    const VkAllocationCallbacks* allocator, VkPipelineCache* pipeline_cache);

void impl_destroy_pipeline_cache(VkDevice device, VkPipelineCache pipeline_cache,
                                 const VkAllocationCallbacks* allocator);

VkResult impl_get_pipeline_cache_data(VkDevice device, VkPipelineCache pipeline_cache, size_t* data_size, void*);

VkResult impl_merge_pipeline_caches(VkDevice device, VkPipelineCache dst_cache,
                                    uint32_t src_cache_count, const VkPipelineCache* src_caches);

PFN_vkVoidFunction impl_get_device_proc_addr(VkDevice device, const char* name);

} // namespace nikki_detail

#endif // NIKKI_VULKAN_INTERNAL_H

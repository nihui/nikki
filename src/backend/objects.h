// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_BACKEND_OBJECTS_H
#define NIKKI_BACKEND_OBJECTS_H

#include "../vulkan_internal.h"
#include "device_profile.h"

#include <webgpu/webgpu.h>

namespace nikki_detail {

struct Instance : public std::enable_shared_from_this<Instance>
{
    Instance()
        : id(next_object_id()),
          instance(0),
          api_version(VK_API_VERSION_1_0),
          physical_devices_enumerated(false),
          packed_4x8_integer_dot_product_supported(false),
          cooperative_matrix_translation_supported(false),
          immediate_supported(false)
    {
    }

    ~Instance();

    uint64_t id;
    WGPUInstance instance;
    uint32_t api_version;
    std::unordered_set<std::string> enabled_extensions;
    std::vector<VkPhysicalDevice> physical_devices;
    bool physical_devices_enumerated;
    std::shared_ptr<TranslatorInstanceState> translator;
    bool packed_4x8_integer_dot_product_supported;
    bool cooperative_matrix_translation_supported;
    bool immediate_supported;
};

struct PhysicalDevice
{
    PhysicalDevice()
        : id(next_object_id()),
          owner(0),
          adapter(0),
          vendor_id(0),
          device_id(0),
          backend_type(WGPUBackendType_Undefined),
          adapter_type(WGPUAdapterType_Unknown)
    {
        limits = WGPU_LIMITS_INIT;
    }

    ~PhysicalDevice();

    uint64_t id;
    Instance* owner;
    WGPUAdapter adapter;
    WGPULimits limits;
    uint32_t vendor_id;
    uint32_t device_id;
    WGPUBackendType backend_type;
    WGPUAdapterType adapter_type;
    DeviceProfile profile;
    std::string device_name;
    std::string driver_name;
    std::string vendor_name;
    std::string architecture_name;
};

struct Device
{
    Device()
        : id(next_object_id()),
          device(0),
          queue(0),
          queue_handle(0),
          alias_merge_bind_group_layout(0),
          alias_merge_pipeline(0),
          same_buffer_copy_bind_group_layout(0),
          same_buffer_copy_pipeline(0),
          lost(false),
          error(false)
    {
        limits = WGPU_LIMITS_INIT;
    }

    ~Device();

    uint64_t id;
    std::shared_ptr<Instance> instance;
    std::shared_ptr<PhysicalDevice> physical_device;
    WGPUDevice device;
    WGPUQueue queue;
    WGPULimits limits;
    VkQueue queue_handle;
    EnabledDeviceProfile enabled_profile;
    WGPUBindGroupLayout alias_merge_bind_group_layout;
    WGPUComputePipeline alias_merge_pipeline;
    std::mutex alias_merge_mutex;
    WGPUBindGroupLayout same_buffer_copy_bind_group_layout;
    WGPUComputePipeline same_buffer_copy_pipeline;
    std::mutex same_buffer_copy_mutex;
    std::atomic<bool> lost;
    std::atomic<bool> error;
    std::unordered_set<std::string> enabled_extensions;
    std::mutex mutex;
    std::vector<std::shared_ptr<Submission> > submissions;
    std::vector<std::shared_ptr<Submission> > abandoned_submissions;
};

struct DeviceMemory
{
    DeviceMemory()
        : id(next_object_id()),
          owner(0),
          size(0),
          padded_size(0),
          memory_type_index(0),
          buffer(0),
          mapped(false),
          mapped_offset(0),
          mapped_size(0),
          live(true)
    {
    }

    ~DeviceMemory();

    uint64_t id;
    Device* owner;
    VkDeviceSize size;
    VkDeviceSize padded_size;
    uint32_t memory_type_index;
    WGPUBuffer buffer;
    std::vector<unsigned char> host_shadow;
    bool mapped;
    VkDeviceSize mapped_offset;
    VkDeviceSize mapped_size;
    bool live;
    std::vector<ByteRange> dirty_ranges;
};

struct MemoryRange
{
    std::shared_ptr<DeviceMemory> memory;
    uint64_t offset;
    uint64_t size;
};

struct ReplayState
{
    ReplayState()
        : encoder(0),
          compute_pass(0),
          push_uniform_buffer(0),
          push_uniform_offset(0),
          push_uniform_size(0),
          push_uniform_alignment(4)
    {
    }

    WGPUCommandEncoder encoder;
    WGPUComputePassEncoder compute_pass;
    std::unordered_map<uint64_t, BufferAccess> compute_pass_buffer_access;
    WGPUBuffer push_uniform_buffer;
    uint64_t push_uniform_offset;
    uint64_t push_uniform_size;
    uint64_t push_uniform_alignment;
};

struct Buffer
{
    Buffer()
        : id(next_object_id()),
          owner(0),
          size(0),
          usage(0),
          memory_offset(0),
          live(true)
    {
    }

    uint64_t id;
    Device* owner;
    VkDeviceSize size;
    VkBufferUsageFlags usage;
    std::shared_ptr<DeviceMemory> memory;
    VkDeviceSize memory_offset;
    bool live;
};

struct DescriptorBinding
{
    uint32_t binding;
    VkDescriptorType descriptor_type;
    uint32_t descriptor_count;
    VkShaderStageFlags stage_flags;
};

struct DescriptorSetLayout
{
    DescriptorSetLayout()
        : id(next_object_id()),
          owner(0),
          push_descriptor(false)
    {
    }

    uint64_t id;
    Device* owner;
    bool push_descriptor;
    std::vector<DescriptorBinding> bindings;
};

struct DescriptorTemplateEntry
{
    uint32_t binding;
    uint32_t array_element;
    uint32_t descriptor_count;
    VkDescriptorType descriptor_type;
    size_t offset;
    size_t stride;
};

struct DescriptorUpdateTemplate
{
    DescriptorUpdateTemplate()
        : id(next_object_id()),
          owner(0),
          template_type(VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_DESCRIPTOR_SET),
          bind_point(VK_PIPELINE_BIND_POINT_COMPUTE),
          set(0)
    {
    }

    uint64_t id;
    Device* owner;
    VkDescriptorUpdateTemplateType template_type;
    VkPipelineBindPoint bind_point;
    uint32_t set;
    std::shared_ptr<DescriptorSetLayout> set_layout;
    std::shared_ptr<PipelineLayout> pipeline_layout;
    std::vector<DescriptorTemplateEntry> entries;
};

struct PushConstantRange
{
    VkShaderStageFlags stage_flags;
    uint32_t offset;
    uint32_t size;
};

struct PipelineLayout
{
    PipelineLayout()
        : id(next_object_id()),
          owner(0),
          push_constant_size(0)
    {
    }

    uint64_t id;
    Device* owner;
    std::vector<std::shared_ptr<DescriptorSetLayout> > set_layouts;
    std::vector<PushConstantRange> push_constant_ranges;
    uint32_t push_constant_size;
};

struct ShaderModule
{
    ShaderModule()
        : id(next_object_id()),
          owner(0)
    {
    }

    uint64_t id;
    Device* owner;
    std::vector<uint32_t> spirv;
};

struct ActiveBinding
{
    uint32_t group;
    uint32_t binding;
    VkDescriptorType descriptor_type;
    BufferAccess access;
    BufferAccess layout_access;
    uint64_t min_binding_size;
    bool internal_uniform;
};

struct ComputePipeline
{
    ComputePipeline()
        : id(next_object_id()),
          owner(0),
          shader_module(0),
          bind_group_layout(0),
          immediate_bind_group_layout(0),
          pipeline_layout(0),
          pipeline(0),
          immediate_size(0),
          push_constant_uniform(false),
          push_constant_data_size(0),
          push_constant_uniform_size(0),
          requires_push_constants(false),
          workgroup_size_x(1),
          workgroup_size_y(1),
          workgroup_size_z(1),
          dispatch_scale_x(1),
          dispatch_scale_y(1),
          dispatch_scale_z(1),
          live(true)
    {
    }

    ~ComputePipeline();

    uint64_t id;
    Device* owner;
    std::shared_ptr<PipelineLayout> vk_pipeline_layout;
    WGPUShaderModule shader_module;
    WGPUBindGroupLayout bind_group_layout;
    WGPUBindGroupLayout immediate_bind_group_layout;
    WGPUPipelineLayout pipeline_layout;
    WGPUComputePipeline pipeline;
    std::vector<ActiveBinding> active_bindings;
    std::unordered_set<uint64_t> exclusive_write_binding_pairs;
    uint32_t immediate_size;
    bool push_constant_uniform;
    uint32_t push_constant_data_size;
    uint32_t push_constant_uniform_size;
    bool requires_push_constants;
    uint32_t workgroup_size_x;
    uint32_t workgroup_size_y;
    uint32_t workgroup_size_z;
    uint32_t dispatch_scale_x;
    uint32_t dispatch_scale_y;
    uint32_t dispatch_scale_z;
    bool live;
};

struct ResolvedBinding
{
    uint32_t binding;
    VkDescriptorType descriptor_type;
    BufferAccess access;
    BufferAccess layout_access;
    uint64_t min_binding_size;
    std::shared_ptr<Buffer> buffer;
    std::shared_ptr<DeviceMemory> memory;
    uint64_t offset;
    uint64_t size;
};

struct CopyRegion
{
    uint64_t src_offset;
    uint64_t dst_offset;
    uint64_t size;
};

struct Command
{
    Command()
        : type(COMMAND_BARRIER),
          group_count_x(0),
          group_count_y(0),
          group_count_z(0),
          first_query(0),
          query_count(0),
          query_result_offset(0),
          query_result_stride(0),
          query_result_flags(0)
    {
    }

    CommandType type;
    std::shared_ptr<ComputePipeline> pipeline;
    std::vector<ResolvedBinding> bindings;
    std::vector<unsigned char> push_constants;
    uint32_t group_count_x;
    uint32_t group_count_y;
    uint32_t group_count_z;
    std::shared_ptr<Buffer> src_buffer;
    std::shared_ptr<Buffer> dst_buffer;
    std::vector<CopyRegion> copy_regions;
    std::vector<std::shared_ptr<Buffer> > barrier_buffers;
    std::shared_ptr<QueryPool> query_pool;
    uint32_t first_query;
    uint32_t query_count;
    std::shared_ptr<Buffer> query_result_buffer;
    uint64_t query_result_offset;
    uint64_t query_result_stride;
    VkQueryResultFlags query_result_flags;
};

struct CommandPool
{
    CommandPool()
        : id(next_object_id()),
          owner(0),
          flags(0),
          queue_family_index(0)
    {
    }

    uint64_t id;
    Device* owner;
    VkCommandPoolCreateFlags flags;
    uint32_t queue_family_index;
    std::mutex mutex;
    std::unordered_set<VkCommandBuffer> command_buffers;
};

struct DescriptorValue
{
    VkDescriptorType descriptor_type;
    VkDescriptorBufferInfo buffer_info;
};

struct CommandBuffer
{
    CommandBuffer()
        : id(next_object_id()),
          owner(0),
          state(COMMAND_BUFFER_INITIAL),
          pending_count(0),
          usage_flags(0),
          error(VK_SUCCESS)
    {
    }

    uint64_t id;
    Device* owner;
    std::shared_ptr<CommandPool> pool;
    CommandBufferState state;
    uint32_t pending_count;
    VkCommandBufferUsageFlags usage_flags;
    VkResult error;
    std::shared_ptr<ComputePipeline> current_pipeline;
    std::shared_ptr<PipelineLayout> descriptor_pipeline_layout;
    std::unordered_map<uint32_t, DescriptorValue> descriptors;
    std::shared_ptr<PipelineLayout> push_constant_pipeline_layout;
    std::vector<unsigned char> push_constants;
    std::vector<Command> commands;
};

struct PipelineCache
{
    PipelineCache()
        : id(next_object_id()),
          owner(0)
    {
    }

    uint64_t id;
    Device* owner;
};

struct QueryPool
{
    QueryPool()
        : id(next_object_id()),
          owner(0),
          query_set(0),
          live(true),
          pending_count(0)
    {
    }

    ~QueryPool();

    uint64_t id;
    Device* owner;
    WGPUQuerySet query_set;
    std::vector<uint64_t> values;
    std::vector<unsigned char> available;
    bool live;
    uint32_t pending_count;
    std::mutex mutex;
};

struct MapResult
{
    MapResult()
        : status(WGPUMapAsyncStatus_Error),
          completed(false)
    {
    }

    WGPUMapAsyncStatus status;
    bool completed;
};

struct Readback
{
    Readback()
        : buffer(0),
          offset(0),
          size(0),
          map_started(false),
          copied(false)
    {
        future.id = 0;
    }

    WGPUBuffer buffer;
    std::shared_ptr<DeviceMemory> memory;
    uint64_t offset;
    uint64_t size;
    WGPUFuture future;
    std::shared_ptr<MapResult> map_result;
    bool map_started;
    bool copied;
};

struct QueryReadback
{
    QueryReadback()
        : buffer(0),
          map_started(false),
          copied(false)
    {
        future.id = 0;
    }

    WGPUBuffer buffer;
    std::shared_ptr<QueryPool> query_pool;
    std::vector<unsigned char> written;
    WGPUFuture future;
    std::shared_ptr<MapResult> map_result;
    bool map_started;
    bool copied;
};

struct QueryUse
{
    std::shared_ptr<QueryPool> query_pool;
    std::vector<unsigned char> written;
};

struct Submission
{
    Submission()
        : id(next_object_id()),
          owner(0),
          completed(false),
          processed(false),
          status(WGPUQueueWorkDoneStatus_Error),
          result(VK_NOT_READY)
    {
        future.id = 0;
    }

    ~Submission();

    void release_resources();

    uint64_t id;
    Device* owner;
    WGPUFuture future;
    bool completed;
    bool processed;
    WGPUQueueWorkDoneStatus status;
    VkResult result;
    std::weak_ptr<Fence> fence;
    std::vector<WGPUBindGroup> bind_groups;
    std::vector<WGPUBuffer> temporary_buffers;
    std::vector<Readback> readbacks;
    std::vector<QueryReadback> query_readbacks;
    std::vector<std::shared_ptr<DeviceMemory> > memories;
    std::vector<std::shared_ptr<Buffer> > buffers;
    std::vector<std::shared_ptr<ComputePipeline> > pipelines;
    std::vector<QueryUse> query_uses;
    std::vector<std::shared_ptr<CommandBuffer> > command_buffers;
};

struct Fence
{
    Fence()
        : id(next_object_id()),
          owner(0),
          signaled(false),
          failed(false)
    {
    }

    uint64_t id;
    Device* owner;
    bool signaled;
    bool failed;
    std::shared_ptr<Submission> submission;
};

inline HandleTable<DeviceMemory> g_memories;
inline HandleTable<Buffer> g_buffers;
inline HandleTable<DescriptorSetLayout> g_descriptor_set_layouts;
inline HandleTable<DescriptorUpdateTemplate> g_descriptor_update_templates;
inline HandleTable<PipelineLayout> g_pipeline_layouts;
inline HandleTable<ShaderModule> g_shader_modules;
inline HandleTable<ComputePipeline> g_compute_pipelines;
inline HandleTable<CommandPool> g_command_pools;
inline HandleTable<Fence> g_fences;
inline HandleTable<PipelineCache> g_pipeline_caches;
inline HandleTable<QueryPool> g_query_pools;

const DescriptorBinding* find_layout_binding(const DescriptorSetLayout& layout, uint32_t binding);

template<typename Handle, typename T>
inline Handle make_handle(HandleTable<T>& table, const std::shared_ptr<T>& object)
{
    return id_to_handle<Handle>(table.insert(object));
}

template<typename T, typename Handle>
inline std::shared_ptr<T> get_handle(const HandleTable<T>& table, Handle handle)
{
    return table.get(handle_to_id(handle));
}

template<typename T, typename Handle>
inline void erase_handle(HandleTable<T>& table, Handle handle)
{
    table.erase(handle_to_id(handle));
}

inline Instance* unwrap(VkInstance instance)
{
    return instance && instance->impl ? instance->impl.get() : 0;
}

inline PhysicalDevice* unwrap(VkPhysicalDevice physical_device)
{
    return physical_device && physical_device->impl ? physical_device->impl.get() : 0;
}

inline Device* unwrap(VkDevice device)
{
    return device && device->impl ? device->impl.get() : 0;
}

inline Device* unwrap(VkQueue queue)
{
    return queue ? queue->device : 0;
}

inline bool device_failed(const Device* device)
{
    return device && (device->lost || device->error);
}

inline CommandBuffer* unwrap(VkCommandBuffer command_buffer)
{
    return command_buffer && command_buffer->impl ? command_buffer->impl.get() : 0;
}

} // namespace nikki_detail

#endif // NIKKI_BACKEND_OBJECTS_H

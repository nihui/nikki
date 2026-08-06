// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "dispatch.h"
#include "shader_module.h"

namespace nikki_detail {

uint64_t binding_pair_key(uint32_t a, uint32_t b)
{
    if (a > b)
        std::swap(a, b);
    return ((uint64_t)a << 32) | b;
}

static bool compute_pass_compatible(const ReplayState& replay,
                                    const std::vector<WGPUBuffer>& buffers,
                                    const std::vector<ResolvedBinding>& bindings)
{
    for (size_t i = 0; i < buffers.size(); i++)
    {
        const uint64_t buffer_id = handle_to_id(buffers[i]);
        std::unordered_map<uint64_t, BufferAccess>::const_iterator it = replay.compute_pass_buffer_access.find(buffer_id);
        if (it != replay.compute_pass_buffer_access.end() && it->second != bindings[i].layout_access)
            return false;
    }

    return true;
}

static int begin_compute_pass(ReplayState& replay)
{
    if (replay.compute_pass)
        return 0;

    WGPUComputePassDescriptor descriptor = WGPU_COMPUTE_PASS_DESCRIPTOR_INIT;
    replay.compute_pass = wgpuCommandEncoderBeginComputePass(replay.encoder, &descriptor);
    return replay.compute_pass ? 0 : -1;
}

static_assert(sizeof(SameBufferCopyParams) == 16, "unexpected same-buffer copy parameter layout");

bool make_same_buffer_copy_plan(Device* device, const Command& command,
                                const CopyRegion& region, SameBufferCopyPlan& plan)
{
    if (!device || command.src_buffer->memory.get() != command.dst_buffer->memory.get())
        return false;

    uint64_t src_offset = 0;
    uint64_t dst_offset = 0;
    uint64_t src_end = 0;
    uint64_t dst_end = 0;
    if (!checked_add(command.src_buffer->memory_offset, region.src_offset, src_offset)
        || !checked_add(command.dst_buffer->memory_offset, region.dst_offset, dst_offset)
        || !checked_add(src_offset, region.size, src_end)
        || !checked_add(dst_offset, region.size, dst_end))
        return false;

    const uint64_t alignment = std::max<uint64_t>(device->limits.minStorageBufferOffsetAlignment, 4);
    const uint64_t binding_offset_unaligned = std::min(src_offset, dst_offset);
    const uint64_t binding_offset = binding_offset_unaligned - binding_offset_unaligned % alignment;
    const uint64_t binding_end = std::max(src_end, dst_end);
    const uint64_t binding_size = binding_end - binding_offset;
    const uint64_t src_word = (src_offset - binding_offset) / 4;
    const uint64_t dst_word = (dst_offset - binding_offset) / 4;
    const uint64_t word_count = region.size / 4;
    if (region.size == 0 || region.size % 4 != 0
        || src_offset % 4 != 0 || dst_offset % 4 != 0
        || binding_size == 0 || binding_size % 4 != 0
        || binding_end > command.src_buffer->memory->padded_size
        || binding_size > device->limits.maxStorageBufferBindingSize
        || src_word > UINT32_MAX || dst_word > UINT32_MAX
        || word_count == 0 || word_count > UINT32_MAX
        || src_word + word_count > (uint64_t)UINT32_MAX + 1
        || dst_word + word_count > (uint64_t)UINT32_MAX + 1)
        return false;

    plan.binding_offset = binding_offset;
    plan.binding_size = binding_size;
    plan.src_word = (uint32_t)src_word;
    plan.dst_word = (uint32_t)dst_word;
    plan.word_count = (uint32_t)word_count;
    return true;
}

uint64_t same_buffer_copy_max_words(const Device* device)
{
    return device ? (uint64_t)device->limits.maxComputeWorkgroupsPerDimension * 64 : 0;
}

static int create_internal_compute_pipeline(Device* device,
                                            const char* source_code, size_t source_code_size,
                                            const WGPUBindGroupLayoutEntry* entries, size_t entry_count,
                                            WGPUBindGroupLayout& bind_group_layout,
                                            WGPUComputePipeline& pipeline)
{
    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code.data = source_code;
    source.code.length = source_code_size;
    WGPUShaderModuleDescriptor shader_descriptor = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    shader_descriptor.nextInChain = &source.chain;
    WGPUShaderModule shader = wgpuDeviceCreateShaderModule(device->device, &shader_descriptor);
    if (!shader)
        return -1;

    WGPUBindGroupLayoutDescriptor bind_group_layout_descriptor = WGPU_BIND_GROUP_LAYOUT_DESCRIPTOR_INIT;
    bind_group_layout_descriptor.entryCount = entry_count;
    bind_group_layout_descriptor.entries = entries;
    bind_group_layout = wgpuDeviceCreateBindGroupLayout(device->device, &bind_group_layout_descriptor);
    if (!bind_group_layout)
    {
        wgpuShaderModuleRelease(shader);
        return -1;
    }

    WGPUPipelineLayoutDescriptor pipeline_layout_descriptor = WGPU_PIPELINE_LAYOUT_DESCRIPTOR_INIT;
    pipeline_layout_descriptor.bindGroupLayoutCount = 1;
    pipeline_layout_descriptor.bindGroupLayouts = &bind_group_layout;
    WGPUPipelineLayout pipeline_layout = wgpuDeviceCreatePipelineLayout(device->device, &pipeline_layout_descriptor);
    if (!pipeline_layout)
    {
        wgpuBindGroupLayoutRelease(bind_group_layout);
        bind_group_layout = 0;
        wgpuShaderModuleRelease(shader);
        return -1;
    }

    WGPUComputePipelineDescriptor pipeline_descriptor = WGPU_COMPUTE_PIPELINE_DESCRIPTOR_INIT;
    pipeline_descriptor.layout = pipeline_layout;
    pipeline_descriptor.compute.module = shader;
    pipeline_descriptor.compute.entryPoint.data = "main";
    pipeline_descriptor.compute.entryPoint.length = 4;
    pipeline = wgpuDeviceCreateComputePipeline(device->device, &pipeline_descriptor);
    wgpuPipelineLayoutRelease(pipeline_layout);
    wgpuShaderModuleRelease(shader);
    if (!pipeline)
    {
        wgpuBindGroupLayoutRelease(bind_group_layout);
        bind_group_layout = 0;
        return -1;
    }

    return 0;
}

static int ensure_same_buffer_copy_pipeline(Device* device)
{
    std::lock_guard<std::mutex> lock(device->same_buffer_copy_mutex);
    if (device->same_buffer_copy_pipeline)
        return 0;

    // WebGPU rejects a native copy with the same GPUBuffer at both ends.
    // Bind the arena once so the shader also avoids writable binding aliases.
    static const char same_buffer_copy_wgsl[] = "struct CopyParams {\n"
                                                "    src_word : u32,\n"
                                                "    dst_word : u32,\n"
                                                "    word_count : u32,\n"
                                                "    base_word : u32,\n"
                                                "}\n"
                                                "@group(0) @binding(0) var<storage, read_write> arena : array<u32>;\n"
                                                "@group(0) @binding(1) var<uniform> params : CopyParams;\n"
                                                "@compute @workgroup_size(64)\n"
                                                "fn main(@builtin(global_invocation_id) id : vec3<u32>) {\n"
                                                "    if (id.x < params.word_count - params.base_word) {\n"
                                                "        let i = params.base_word + id.x;\n"
                                                "        arena[params.dst_word + i] = arena[params.src_word + i];\n"
                                                "    }\n"
                                                "}\n";

    WGPUBindGroupLayoutEntry entries[2];
    entries[0] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[0].binding = 0;
    entries[0].visibility = WGPUShaderStage_Compute;
    entries[0].buffer = WGPU_BUFFER_BINDING_LAYOUT_INIT;
    entries[0].buffer.type = WGPUBufferBindingType_Storage;
    entries[0].buffer.minBindingSize = 4;
    entries[1] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
    entries[1].binding = 1;
    entries[1].visibility = WGPUShaderStage_Compute;
    entries[1].buffer = WGPU_BUFFER_BINDING_LAYOUT_INIT;
    entries[1].buffer.type = WGPUBufferBindingType_Uniform;
    entries[1].buffer.minBindingSize = sizeof(SameBufferCopyParams);

    return create_internal_compute_pipeline(device,
                                            same_buffer_copy_wgsl, sizeof(same_buffer_copy_wgsl) - 1,
                                            entries, 2,
                                            device->same_buffer_copy_bind_group_layout,
                                            device->same_buffer_copy_pipeline);
}

int encode_same_buffer_copy(Device* device, ReplayState& replay,
                            const Command& command,
                            const SameBufferCopyPlan& plan,
                            const std::shared_ptr<Submission>& submission)
{
    const uint64_t max_words = same_buffer_copy_max_words(device);
    if (max_words == 0 || ensure_same_buffer_copy_pipeline(device) != 0)
        return -1;

    for (uint64_t base_word = 0; base_word < plan.word_count;)
    {
        const uint64_t chunk_words = std::min<uint64_t>(plan.word_count - base_word, max_words);
        const uint64_t uniform_offset = align_up(replay.push_uniform_offset, replay.push_uniform_alignment);
        uint64_t uniform_end = 0;
        if (!replay.push_uniform_buffer
            || (uniform_offset == 0 && replay.push_uniform_offset != 0)
            || !checked_add(uniform_offset, sizeof(SameBufferCopyParams), uniform_end)
            || uniform_end > replay.push_uniform_size)
            return -1;

        std::unordered_map<uint64_t, BufferAccess>::const_iterator arena_access = replay.compute_pass_buffer_access.find(handle_to_id(command.src_buffer->memory->buffer));
        std::unordered_map<uint64_t, BufferAccess>::const_iterator uniform_access = replay.compute_pass_buffer_access.find(handle_to_id(replay.push_uniform_buffer));
        if (replay.compute_pass
            && ((arena_access != replay.compute_pass_buffer_access.end()
                 && arena_access->second != BUFFER_ACCESS_READ_WRITE)
                || (uniform_access != replay.compute_pass_buffer_access.end()
                    && uniform_access->second != BUFFER_ACCESS_READ)))
            end_compute_pass(replay);
        if (begin_compute_pass(replay) != 0)
            return -1;

        WGPUBindGroupEntry entries[2];
        entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[0].binding = 0;
        entries[0].buffer = command.src_buffer->memory->buffer;
        entries[0].offset = plan.binding_offset;
        entries[0].size = plan.binding_size;
        entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[1].binding = 1;
        entries[1].buffer = replay.push_uniform_buffer;
        entries[1].offset = uniform_offset;
        entries[1].size = sizeof(SameBufferCopyParams);

        WGPUBindGroupDescriptor descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        descriptor.layout = device->same_buffer_copy_bind_group_layout;
        descriptor.entryCount = 2;
        descriptor.entries = entries;
        WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device->device, &descriptor);
        if (!bind_group)
            return -1;
        submission->bind_groups.push_back(bind_group);

        wgpuComputePassEncoderSetPipeline(replay.compute_pass, device->same_buffer_copy_pipeline);
        wgpuComputePassEncoderSetBindGroup(replay.compute_pass, 0, bind_group, 0, 0);
        wgpuComputePassEncoderDispatchWorkgroups(
            replay.compute_pass, (uint32_t)((chunk_words + 63) / 64), 1, 1);
        replay.compute_pass_buffer_access[handle_to_id(command.src_buffer->memory->buffer)] = BUFFER_ACCESS_READ_WRITE;
        replay.compute_pass_buffer_access[handle_to_id(replay.push_uniform_buffer)] = BUFFER_ACCESS_READ;
        replay.push_uniform_offset = uniform_end;
        base_word += chunk_words;
    }

    return 0;
}

static int ensure_alias_merge_pipeline(Device* device)
{
    std::lock_guard<std::mutex> lock(device->alias_merge_mutex);
    if (device->alias_merge_pipeline)
        return 0;

    static const char alias_merge_wgsl[] = "@group(0) @binding(0) var<storage, read_write> destination : array<u32>;\n"
                                           "@group(0) @binding(1) var<storage, read> baseline : array<u32>;\n"
                                           "@group(0) @binding(2) var<storage, read> candidate : array<u32>;\n"
                                           "@compute @workgroup_size(64)\n"
                                           "fn main(@builtin(global_invocation_id) id : vec3<u32>) {\n"
                                           "    let i = id.x;\n"
                                           "    if (i < arrayLength(&destination)) {\n"
                                           "        let value = candidate[i];\n"
                                           "        if (value != baseline[i]) {\n"
                                           "            destination[i] = value;\n"
                                           "        }\n"
                                           "    }\n"
                                           "}\n";

    WGPUBindGroupLayoutEntry entries[3];
    for (uint32_t i = 0; i < 3; i++)
    {
        entries[i] = WGPU_BIND_GROUP_LAYOUT_ENTRY_INIT;
        entries[i].binding = i;
        entries[i].visibility = WGPUShaderStage_Compute;
        entries[i].buffer = WGPU_BUFFER_BINDING_LAYOUT_INIT;
        entries[i].buffer.type = i == 0 ? WGPUBufferBindingType_Storage : WGPUBufferBindingType_ReadOnlyStorage;
    }

    return create_internal_compute_pipeline(device,
                                            alias_merge_wgsl, sizeof(alias_merge_wgsl) - 1,
                                            entries, 3,
                                            device->alias_merge_bind_group_layout,
                                            device->alias_merge_pipeline);
}

struct WritableAliasGroup
{
    std::vector<size_t> bindings;
    WGPUBuffer baseline;
};

enum SnapshotMode
{
    SNAPSHOT_NONE,
    SNAPSHOT_READ,
    SNAPSHOT_WRITE
};

static int encode_alias_merge(Device* device, ReplayState& replay,
                              const std::vector<ResolvedBinding>& bindings,
                              const std::vector<WGPUBuffer>& bind_buffers,
                              const std::vector<WritableAliasGroup>& groups,
                              const std::shared_ptr<Submission>& submission)
{
    if (groups.empty())
        return 0;

    end_compute_pass(replay);
    if (ensure_alias_merge_pipeline(device) != 0 || begin_compute_pass(replay) != 0)
        return -1;

    wgpuComputePassEncoderSetPipeline(replay.compute_pass, device->alias_merge_pipeline);
    for (size_t i = 0; i < groups.size(); i++)
    {
        const WritableAliasGroup& group = groups[i];
        if (group.bindings.size() < 2 || !group.baseline)
            return -1;

        const ResolvedBinding& destination = bindings[group.bindings[0]];
        const uint64_t merge_size = align_up(destination.size, 4);
        if (merge_size == 0)
            return -1;

        const uint64_t max_dispatch_size = (uint64_t)device->limits.maxComputeWorkgroupsPerDimension * 64 * 4;
        uint64_t max_chunk_size = std::min<uint64_t>(device->limits.maxStorageBufferBindingSize, max_dispatch_size);
        const uint64_t alignment = std::max<uint64_t>(device->limits.minStorageBufferOffsetAlignment, 4);
        max_chunk_size -= max_chunk_size % alignment;
        if (max_chunk_size == 0)
            return -1;

        for (size_t j = 0; j < group.bindings.size(); j++)
        {
            const size_t binding_index = group.bindings[j];
            for (uint64_t offset = 0; offset < merge_size;)
            {
                const uint64_t size = std::min<uint64_t>(merge_size - offset, max_chunk_size);
                WGPUBindGroupEntry entries[3];
                entries[0] = WGPU_BIND_GROUP_ENTRY_INIT;
                entries[0].binding = 0;
                entries[0].buffer = destination.memory->buffer;
                entries[0].offset = destination.offset + offset;
                entries[0].size = size;
                entries[1] = WGPU_BIND_GROUP_ENTRY_INIT;
                entries[1].binding = 1;
                entries[1].buffer = group.baseline;
                entries[1].offset = offset;
                entries[1].size = size;
                entries[2] = WGPU_BIND_GROUP_ENTRY_INIT;
                entries[2].binding = 2;
                entries[2].buffer = bind_buffers[binding_index];
                entries[2].offset = offset;
                entries[2].size = size;

                WGPUBindGroupDescriptor descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
                descriptor.layout = device->alias_merge_bind_group_layout;
                descriptor.entryCount = 3;
                descriptor.entries = entries;
                WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device->device, &descriptor);
                if (!bind_group)
                    return -1;
                submission->bind_groups.push_back(bind_group);

                wgpuComputePassEncoderSetBindGroup(replay.compute_pass, 0, bind_group, 0, 0);
                wgpuComputePassEncoderDispatchWorkgroups(replay.compute_pass, (uint32_t)((size + 255) / 256), 1, 1);
                offset += size;
            }
        }

        replay.compute_pass_buffer_access[handle_to_id(destination.memory->buffer)] = BUFFER_ACCESS_READ_WRITE;
        replay.compute_pass_buffer_access[handle_to_id(group.baseline)] = BUFFER_ACCESS_READ;
        for (size_t j = 0; j < group.bindings.size(); j++)
            replay.compute_pass_buffer_access[handle_to_id(bind_buffers[group.bindings[j]])] = BUFFER_ACCESS_READ;
    }

    return 0;
}

static int analyze_dispatch_aliases(const Command& command,
                                    const std::vector<ResolvedBinding>& bindings,
                                    std::vector<SnapshotMode>& snapshots,
                                    std::vector<WritableAliasGroup>& writable_alias_groups)
{
    // WebGPU rejects overlapping writable bindings in one dispatch. Preserve
    // Vulkan shader behavior with snapshots when the writes are exclusive.
    std::vector<int> writable_alias_group_indices(bindings.size(), -1);
    for (size_t i = 0; i < bindings.size(); i++)
    {
        if (bindings[i].access != BUFFER_ACCESS_READ_WRITE || writable_alias_group_indices[i] >= 0)
            continue;

        std::vector<size_t> aliases;
        aliases.push_back(i);
        for (size_t j = i + 1; j < bindings.size(); j++)
        {
            if (bindings[j].access != BUFFER_ACCESS_READ_WRITE
                || bindings[i].memory.get() != bindings[j].memory.get()
                || bindings[i].offset != bindings[j].offset
                || bindings[i].size != bindings[j].size)
                continue;

            bool exclusive_with_group = true;
            for (size_t k = 0; k < aliases.size(); k++)
            {
                if (command.pipeline->exclusive_write_binding_pairs.count(
                        binding_pair_key(bindings[aliases[k]].binding, bindings[j].binding))
                    == 0)
                {
                    exclusive_with_group = false;
                    break;
                }
            }
            if (exclusive_with_group)
                aliases.push_back(j);
        }
        if (aliases.size() < 2)
            continue;

        WritableAliasGroup group;
        group.bindings = aliases;
        group.baseline = 0;
        const int group_index = (int)writable_alias_groups.size();
        writable_alias_groups.push_back(group);
        for (size_t j = 0; j < aliases.size(); j++)
        {
            writable_alias_group_indices[aliases[j]] = group_index;
            snapshots[aliases[j]] = SNAPSHOT_WRITE;
        }
    }

    for (size_t i = 0; i < bindings.size(); i++)
    {
        for (size_t j = i + 1; j < bindings.size(); j++)
        {
            if (bindings[i].memory.get() != bindings[j].memory.get())
                continue;

            const bool layout_write_i = bindings[i].layout_access == BUFFER_ACCESS_READ_WRITE;
            const bool layout_write_j = bindings[j].layout_access == BUFFER_ACCESS_READ_WRITE;
            if (!layout_write_i && !layout_write_j)
                continue;

            const bool overlap = ranges_overlap(bindings[i].offset, bindings[i].size,
                                                bindings[j].offset, bindings[j].size);
            const bool actual_write_i = bindings[i].access == BUFFER_ACCESS_READ_WRITE;
            const bool actual_write_j = bindings[j].access == BUFFER_ACCESS_READ_WRITE;
            if (actual_write_i && actual_write_j)
            {
                if (!overlap)
                    continue;

                if (writable_alias_group_indices[i] >= 0
                    && writable_alias_group_indices[i] == writable_alias_group_indices[j])
                    continue;

                log_error("dispatch writable alias binding=%u and binding=%u memory=%llu",
                          bindings[i].binding, bindings[j].binding,
                          (unsigned long long)bindings[i].memory->id);
                return -1;
            }

            if (layout_write_i != layout_write_j)
            {
                snapshots[layout_write_i ? j : i] = SNAPSHOT_READ;
                continue;
            }

            if (overlap)
                snapshots[actual_write_i ? j : i] = SNAPSHOT_READ;
        }
    }

    return 0;
}

static int prepare_dispatch_bindings(Device* device, ReplayState& replay,
                                     const std::vector<ResolvedBinding>& bindings,
                                     const std::vector<SnapshotMode>& snapshots,
                                     std::vector<WritableAliasGroup>& writable_alias_groups,
                                     const std::shared_ptr<Submission>& submission,
                                     std::vector<WGPUBuffer>& bind_buffers,
                                     std::vector<uint64_t>& bind_offsets)
{
    for (size_t i = 0; i < snapshots.size(); i++)
    {
        if (snapshots[i] != SNAPSHOT_NONE)
        {
            end_compute_pass(replay);
            break;
        }
    }

    bind_buffers.assign(bindings.size(), 0);
    bind_offsets.assign(bindings.size(), 0);
    for (size_t i = 0; i < bindings.size(); i++)
    {
        if (snapshots[i] == SNAPSHOT_NONE)
        {
            bind_buffers[i] = bindings[i].memory->buffer;
            bind_offsets[i] = bindings[i].offset;
            continue;
        }

        const uint64_t copy_size = align_up(bindings[i].size, 4);
        if (copy_size == 0 || copy_size > bindings[i].memory->padded_size
            || bindings[i].offset > bindings[i].memory->padded_size - copy_size)
            return -1;

        WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        descriptor.size = copy_size;
        descriptor.usage = WGPUBufferUsage_CopyDst
                           | (snapshots[i] == SNAPSHOT_WRITE ? WGPUBufferUsage_CopySrc : WGPUBufferUsage_None)
                           | (bindings[i].descriptor_type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                  ? WGPUBufferUsage_Uniform
                                  : WGPUBufferUsage_Storage);
        WGPUBuffer temporary = wgpuDeviceCreateBuffer(device->device, &descriptor);
        if (!temporary)
            return -1;

        wgpuCommandEncoderCopyBufferToBuffer(replay.encoder, bindings[i].memory->buffer, bindings[i].offset,
                                             temporary, 0, copy_size);
        submission->temporary_buffers.push_back(temporary);
        bind_buffers[i] = temporary;
    }

    for (size_t i = 0; i < writable_alias_groups.size(); i++)
    {
        const ResolvedBinding& binding = bindings[writable_alias_groups[i].bindings[0]];
        const uint64_t copy_size = align_up(binding.size, 4);
        if (copy_size == 0 || copy_size > binding.memory->padded_size
            || binding.offset > binding.memory->padded_size - copy_size)
            return -1;

        WGPUBufferDescriptor descriptor = WGPU_BUFFER_DESCRIPTOR_INIT;
        descriptor.size = copy_size;
        descriptor.usage = WGPUBufferUsage_CopyDst | WGPUBufferUsage_Storage;
        WGPUBuffer baseline = wgpuDeviceCreateBuffer(device->device, &descriptor);
        if (!baseline)
            return -1;

        wgpuCommandEncoderCopyBufferToBuffer(replay.encoder, binding.memory->buffer, binding.offset,
                                             baseline, 0, copy_size);
        submission->temporary_buffers.push_back(baseline);
        writable_alias_groups[i].baseline = baseline;
    }

    return 0;
}

int encode_dispatch(Device* device, ReplayState& replay, const Command& command,
                    const std::shared_ptr<Submission>& submission)
{
    const std::vector<ResolvedBinding>& bindings = command.bindings;
    std::vector<SnapshotMode> snapshots(bindings.size(), SNAPSHOT_NONE);
    std::vector<WritableAliasGroup> writable_alias_groups;
    if (analyze_dispatch_aliases(command, bindings, snapshots, writable_alias_groups) != 0)
        return -1;

    std::vector<WGPUBuffer> bind_buffers;
    std::vector<uint64_t> bind_offsets;
    if (prepare_dispatch_bindings(device, replay, bindings, snapshots, writable_alias_groups,
                                  submission, bind_buffers, bind_offsets)
        != 0)
        return -1;

    std::vector<WGPUBindGroupEntry> entries(bindings.size());
    for (size_t i = 0; i < bindings.size(); i++)
    {
        entries[i] = WGPU_BIND_GROUP_ENTRY_INIT;
        entries[i].binding = bindings[i].binding;
        entries[i].buffer = bind_buffers[i];
        entries[i].offset = bind_offsets[i];
        entries[i].size = bindings[i].size;
    }

    WGPUBindGroupDescriptor bind_group_descriptor = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
    bind_group_descriptor.layout = command.pipeline->bind_group_layout;
    bind_group_descriptor.entryCount = entries.size();
    bind_group_descriptor.entries = entries.empty() ? 0 : entries.data();
    WGPUBindGroup bind_group = wgpuDeviceCreateBindGroup(device->device, &bind_group_descriptor);
    if (!bind_group)
        return -1;
    submission->bind_groups.push_back(bind_group);

    WGPUBindGroup push_bind_group = 0;
    if (command.pipeline->push_constant_uniform)
    {
        const uint64_t size = align_up(command.pipeline->push_constant_uniform_size, 4);
        const uint64_t offset = align_up(replay.push_uniform_offset, replay.push_uniform_alignment);
        uint64_t end = 0;
        if (!replay.push_uniform_buffer
            || size == 0
            || command.push_constants.size() < command.pipeline->push_constant_data_size
            || !checked_add(offset, size, end)
            || end > replay.push_uniform_size)
            return -1;

        WGPUBindGroupEntry entry = WGPU_BIND_GROUP_ENTRY_INIT;
        entry.binding = 0;
        entry.buffer = replay.push_uniform_buffer;
        entry.offset = offset;
        entry.size = command.pipeline->push_constant_uniform_size;
        WGPUBindGroupDescriptor descriptor_group = WGPU_BIND_GROUP_DESCRIPTOR_INIT;
        descriptor_group.layout = command.pipeline->immediate_bind_group_layout;
        descriptor_group.entryCount = 1;
        descriptor_group.entries = &entry;
        push_bind_group = wgpuDeviceCreateBindGroup(device->device, &descriptor_group);
        if (!push_bind_group)
            return -1;
        submission->bind_groups.push_back(push_bind_group);
        replay.push_uniform_offset = end;
    }

    if (replay.compute_pass && !compute_pass_compatible(replay, bind_buffers, bindings))
        end_compute_pass(replay);
    if (begin_compute_pass(replay) != 0)
        return -1;

    wgpuComputePassEncoderSetPipeline(replay.compute_pass, command.pipeline->pipeline);
    wgpuComputePassEncoderSetBindGroup(replay.compute_pass, 0, bind_group, 0, 0);
    if (push_bind_group)
        wgpuComputePassEncoderSetBindGroup(replay.compute_pass, 1, push_bind_group, 0, 0);
    if (command.pipeline->immediate_size != 0)
    {
        if (command.push_constants.size() < command.pipeline->immediate_size)
            return -1;
        wgpuComputePassEncoderSetImmediates(replay.compute_pass, 0, command.push_constants.data(), command.pipeline->immediate_size);
    }
    if (command.group_count_x > device->limits.maxComputeWorkgroupsPerDimension
        || command.group_count_y > device->limits.maxComputeWorkgroupsPerDimension
        || command.group_count_z > device->limits.maxComputeWorkgroupsPerDimension)
        return -1;
    wgpuComputePassEncoderDispatchWorkgroups(replay.compute_pass,
                                             command.group_count_x,
                                             command.group_count_y,
                                             command.group_count_z);

    for (size_t i = 0; i < bind_buffers.size(); i++)
        replay.compute_pass_buffer_access[handle_to_id(bind_buffers[i])] = bindings[i].layout_access;

    if (encode_alias_merge(device, replay, bindings, bind_buffers, writable_alias_groups, submission) != 0)
        return -1;

    append_unique(submission->pipelines, command.pipeline);
    for (size_t i = 0; i < command.bindings.size(); i++)
    {
        append_unique(submission->buffers, command.bindings[i].buffer);
        append_unique(submission->memories, command.bindings[i].memory);
    }
    return 0;
}

} // namespace nikki_detail

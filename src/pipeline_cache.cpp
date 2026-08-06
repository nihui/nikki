// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "vulkan_internal.h"
#include "backend/objects.h"

namespace nikki_detail {

// Pipeline cache objects are accepted for Vulkan compatibility.
// WebGPU pipelines are not serialized.

VkResult impl_create_pipeline_cache(VkDevice device, const VkPipelineCacheCreateInfo* create_info,
                                    const VkAllocationCallbacks* allocator, VkPipelineCache* pipeline_cache)
{
    Device* device_impl = unwrap(device);
    if (!device_impl || !create_info || create_info->sType != VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO || !pipeline_cache)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (device_failed(device_impl))
        return VK_ERROR_DEVICE_LOST;
    if (allocator)
        return VK_ERROR_FEATURE_NOT_PRESENT;
    if (create_info->pNext || create_info->flags != 0
        || (create_info->initialDataSize != 0 && !create_info->pInitialData))
        return VK_ERROR_INITIALIZATION_FAILED;

    std::shared_ptr<PipelineCache> pipeline_cache_impl = std::make_shared<PipelineCache>();
    pipeline_cache_impl->owner = device_impl;
    *pipeline_cache = make_handle<VkPipelineCache>(g_pipeline_caches, pipeline_cache_impl);
    return VK_SUCCESS;
}

void impl_destroy_pipeline_cache(VkDevice device, VkPipelineCache pipeline_cache,
                                 const VkAllocationCallbacks* allocator)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<PipelineCache> pipeline_cache_impl = get_handle(g_pipeline_caches, pipeline_cache);
    if (!device_impl || !pipeline_cache_impl || pipeline_cache_impl->owner != device_impl)
        return;
    if (allocator)
    {
        log_error("custom allocation callbacks are not supported");
        return;
    }

    erase_handle(g_pipeline_caches, pipeline_cache);
}

VkResult impl_get_pipeline_cache_data(VkDevice device, VkPipelineCache pipeline_cache, size_t* data_size, void*)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<PipelineCache> pipeline_cache_impl = get_handle(g_pipeline_caches, pipeline_cache);
    if (!device_impl || !pipeline_cache_impl || pipeline_cache_impl->owner != device_impl || !data_size)
        return VK_ERROR_INITIALIZATION_FAILED;

    *data_size = 0;
    return VK_SUCCESS;
}

VkResult impl_merge_pipeline_caches(VkDevice device, VkPipelineCache dst_cache,
                                    uint32_t src_cache_count, const VkPipelineCache* src_caches)
{
    Device* device_impl = unwrap(device);
    std::shared_ptr<PipelineCache> dst = get_handle(g_pipeline_caches, dst_cache);
    if (!device_impl || !dst || dst->owner != device_impl || (src_cache_count != 0 && !src_caches))
        return VK_ERROR_INITIALIZATION_FAILED;

    for (uint32_t i = 0; i < src_cache_count; i++)
    {
        std::shared_ptr<PipelineCache> src = get_handle(g_pipeline_caches, src_caches[i]);
        if (!src || src->owner != device_impl)
            return VK_ERROR_INITIALIZATION_FAILED;
    }
    return VK_SUCCESS;
}

} // namespace nikki_detail

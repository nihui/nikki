// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_BACKEND_DEVICE_H
#define NIKKI_BACKEND_DEVICE_H

#include "objects.h"

namespace nikki_detail {

VkResult ensure_physical_devices(Instance* instance);
WGPUFeatureName cooperative_matrix_feature_name();
VkResult request_backend_device(const std::shared_ptr<Device>& device,
                                PhysicalDevice* physical_device,
                                WGPUDeviceDescriptor descriptor);

inline std::string string_from_webgpu(WGPUStringView value)
{
    if (!value.data)
        return std::string();

    return std::string(value.data, value.length);
}

// Synchronous Vulkan waits are implemented with Emdawnwebgpu WaitAny.
// Reject nested waits so an Asyncify resume cannot recursively re-enter Wasm.
inline thread_local uint32_t g_wait_depth = 0;

inline int wait_any(Instance* instance, size_t future_count, WGPUFutureWaitInfo* wait_infos,
                    uint64_t timeout, const char* stage, WGPUWaitStatus& status)
{
    if (!instance || !instance->instance || future_count == 0 || !wait_infos)
        return -1;
    if (g_wait_depth != 0)
    {
        log_error("instance %llu nested WaitAny stage=%s depth=%u",
                  (unsigned long long)instance->id, stage ? stage : "unknown", g_wait_depth);
        return -1;
    }

    g_wait_depth++;
    status = wgpuInstanceWaitAny(instance->instance, future_count, wait_infos, timeout);
    g_wait_depth--;
    return 0;
}

inline int wait_future(Instance* instance, WGPUFuture future, uint64_t timeout, const char* stage)
{
    if (!instance || !instance->instance || future.id == 0)
        return -1;

    WGPUFutureWaitInfo wait_info = WGPU_FUTURE_WAIT_INFO_INIT;
    wait_info.future = future;
    WGPUWaitStatus status;
    if (wait_any(instance, 1, &wait_info, timeout, stage, status) != 0)
        return -1;
    if (status == WGPUWaitStatus_TimedOut)
        return 1;
    if (status != WGPUWaitStatus_Success || wait_info.completed != WGPU_TRUE)
        return -1;

    return 0;
}

} // namespace nikki_detail

#endif // NIKKI_BACKEND_DEVICE_H

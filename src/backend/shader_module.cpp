// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "shader_module.h"

namespace nikki_detail {

struct ErrorScopeResult
{
    ErrorScopeResult()
        : status(WGPUPopErrorScopeStatus_Error),
          type(WGPUErrorType_Unknown),
          completed(false)
    {
    }

    WGPUPopErrorScopeStatus status;
    WGPUErrorType type;
    bool completed;
    std::string message;
};

static void error_scope_callback(WGPUPopErrorScopeStatus status, WGPUErrorType type, WGPUStringView message, void* userdata1, void*)
{
    std::shared_ptr<ErrorScopeResult>* context = (std::shared_ptr<ErrorScopeResult>*)userdata1;
    if (!context)
        return;
    std::shared_ptr<ErrorScopeResult> result = *context;
    delete context;
    if (!result)
        return;

    result->status = status;
    result->type = type;
    result->completed = true;
    result->message = string_from_webgpu(message);
}

int pop_error_scope(Device* device, const char* stage)
{
    std::shared_ptr<ErrorScopeResult> result = std::make_shared<ErrorScopeResult>();
    WGPUPopErrorScopeCallbackInfo callback_info = WGPU_POP_ERROR_SCOPE_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = error_scope_callback;
    std::shared_ptr<ErrorScopeResult>* callback_context = new std::shared_ptr<ErrorScopeResult>(result);
    callback_info.userdata1 = callback_context;

    WGPUFuture future = wgpuDevicePopErrorScope(device->device, callback_info);
    if (future.id == 0)
    {
        delete callback_context;
        device->error = true;
        log_error("device %llu %s returned an invalid future",
                  (unsigned long long)device->id, stage);
        return -1;
    }
    if (wait_future(device->instance.get(), future, UINT64_MAX, stage) != 0
        || !result->completed
        || result->status != WGPUPopErrorScopeStatus_Success)
    {
        device->error = true;
        log_error("device %llu %s failed status=%d type=%d: %s",
                  (unsigned long long)device->id, stage,
                  (int)result->status, (int)result->type, result->message.c_str());
        return -1;
    }
    if (result->type != WGPUErrorType_NoError)
    {
        // The scope captured this operation's error and the WebGPU device is
        // still usable. Propagate failure to the Vulkan caller without
        // converting a validation error into permanent device loss.
        log_error("device %llu %s failed type=%d: %s",
                  (unsigned long long)device->id, stage,
                  (int)result->type, result->message.c_str());
        return -1;
    }

    return 0;
}

WGPUShaderModule create_shader_module(Device* device, const std::string& wgsl)
{
    WGPUShaderSourceWGSL source = WGPU_SHADER_SOURCE_WGSL_INIT;
    source.code.data = wgsl.data();
    source.code.length = wgsl.size();

    WGPUShaderModuleDescriptor descriptor = WGPU_SHADER_MODULE_DESCRIPTOR_INIT;
    descriptor.nextInChain = &source.chain;
    return wgpuDeviceCreateShaderModule(device->device, &descriptor);
}

} // namespace nikki_detail

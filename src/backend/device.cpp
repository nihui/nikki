// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "../vulkan_internal.h"
#include "device.h"
#include "runtime.h"
#include "../translator/translator_internal.h"

#if defined(__EMSCRIPTEN__)
#include <emscripten/emscripten.h>
#endif

#if defined(__EMSCRIPTEN__)
// clang-format off
EM_JS(int, nikki_read_adapter_info,
      (uintptr_t adapter,
       char* vendor, uint32_t vendor_size,
       char* architecture, uint32_t architecture_size,
       char* device, uint32_t device_size,
       char* description, uint32_t description_size,
       uint32_t* subgroup_min_size, uint32_t* subgroup_max_size),
{
    try
    {
        var js_adapter = WebGPU.getJsObject(adapter);
        var info = js_adapter && js_adapter.info;
        if (!info)
            return 0;

        stringToUTF8(info.vendor || "", vendor, vendor_size);
        stringToUTF8(info.architecture || "", architecture, architecture_size);
        stringToUTF8(info.device || "", device, device_size);
        stringToUTF8(info.description || "", description, description_size);

        var subgroup_min = Number(info.subgroupMinSize);
        var subgroup_max = Number(info.subgroupMaxSize);
        HEAPU32[subgroup_min_size >> 2] = Number.isFinite(subgroup_min) && subgroup_min >= 0 ? subgroup_min : 0;
        HEAPU32[subgroup_max_size >> 2] = Number.isFinite(subgroup_max) && subgroup_max >= 0 ? subgroup_max : 0;
        return info.isFallbackAdapter ? 2 : 1;
    }
    catch (error)
    {
        return 0;
    }
});

EM_JS(int, nikki_read_subgroup_matrix_configs,
      (uintptr_t adapter, uint32_t* configs, uint32_t capacity),
{
    try
    {
        var feature_name = "chromium-experimental-subgroup-matrix";
        var feature_value = 0x00050034;

        // Emdawnwebgpu does not generate Dawn-only extension enums, while
        // Chrome exposes this experimental feature directly to JavaScript.
        // Add the missing sparse enum entry so requestDevice() can forward it.
        WebGPU.FeatureName[feature_value] = feature_name;

        var js_adapter = WebGPU.getJsObject(adapter);
        if (!js_adapter || !js_adapter.features || !js_adapter.features.has(feature_name))
            return -1;

        var info = js_adapter.info;
        var matrix_configs = info && info["subgroupMatrixConfigs"];
        if (!matrix_configs)
            return 0;

        var component_types = {
            "f16": 1,
            "f32": 2,
            "i8": 3,
            "u8": 4,
            "i32": 5,
            "u32": 6
        };
        var count = matrix_configs.length;
        var write_count = Math.min(count, capacity);
        var offset = configs >> 2;
        for (var i = 0; i < write_count; i++)
        {
            var config = matrix_configs[i];
            HEAPU32[offset + i * 5] = component_types[config["componentType"]] || 0;
            HEAPU32[offset + i * 5 + 1] = component_types[config["resultComponentType"]] || 0;
            HEAPU32[offset + i * 5 + 2] = Number(config["M"]);
            HEAPU32[offset + i * 5 + 3] = Number(config["N"]);
            HEAPU32[offset + i * 5 + 4] = Number(config["K"]);
        }
        return count;
    }
    catch (error)
    {
        return 0;
    }
});
// clang-format on
#endif
namespace nikki_detail {

Instance::~Instance()
{
    for (size_t i = 0; i < physical_devices.size(); i++)
        delete physical_devices[i];
    physical_devices.clear();
    if (instance)
        wgpuInstanceRelease(instance);
}

PhysicalDevice::~PhysicalDevice()
{
    if (adapter)
        wgpuAdapterRelease(adapter);
}

Device::~Device()
{
    if (queue_handle)
    {
        delete queue_handle;
        queue_handle = 0;
    }
    if (alias_merge_pipeline)
        wgpuComputePipelineRelease(alias_merge_pipeline);
    if (alias_merge_bind_group_layout)
        wgpuBindGroupLayoutRelease(alias_merge_bind_group_layout);
    if (same_buffer_copy_pipeline)
        wgpuComputePipelineRelease(same_buffer_copy_pipeline);
    if (same_buffer_copy_bind_group_layout)
        wgpuBindGroupLayoutRelease(same_buffer_copy_bind_group_layout);
    if (device)
        wgpuDeviceDestroy(device);
    submissions.clear();
    abandoned_submissions.clear();
    if (queue)
        wgpuQueueRelease(queue);
    if (device)
        wgpuDeviceRelease(device);
}

struct AdapterResult
{
    AdapterResult()
        : status(WGPURequestAdapterStatus_Error),
          adapter(0)
    {
    }

    ~AdapterResult()
    {
        if (adapter)
            wgpuAdapterRelease(adapter);
    }

    WGPURequestAdapterStatus status;
    WGPUAdapter adapter;
    std::string message;
};

struct DeviceResult
{
    DeviceResult()
        : status(WGPURequestDeviceStatus_Error),
          device(0)
    {
    }

    ~DeviceResult()
    {
        if (device)
            wgpuDeviceRelease(device);
    }

    WGPURequestDeviceStatus status;
    WGPUDevice device;
    std::shared_ptr<Device> owner;
};

static void request_adapter_callback(WGPURequestAdapterStatus status, WGPUAdapter adapter, WGPUStringView message, void* userdata1, void*)
{
    std::shared_ptr<AdapterResult>* context = (std::shared_ptr<AdapterResult>*)userdata1;
    if (!context)
    {
        if (adapter)
            wgpuAdapterRelease(adapter);
        return;
    }
    std::shared_ptr<AdapterResult> result = *context;
    delete context;
    if (!result)
    {
        if (adapter)
            wgpuAdapterRelease(adapter);
        return;
    }

    result->status = status;
    result->adapter = adapter;
    result->message = string_from_webgpu(message);
}

static void request_device_callback(WGPURequestDeviceStatus status, WGPUDevice device, WGPUStringView message, void* userdata1, void*)
{
    std::shared_ptr<DeviceResult>* context = (std::shared_ptr<DeviceResult>*)userdata1;
    if (!context)
    {
        if (device)
            wgpuDeviceRelease(device);
        return;
    }
    std::shared_ptr<DeviceResult> result = *context;
    delete context;
    if (!result)
    {
        if (device)
            wgpuDeviceRelease(device);
        return;
    }

    result->status = status;
    result->device = device;

    if (status != WGPURequestDeviceStatus_Success)
        log_error("request device failed: %.*s", (int)message.length, message.data ? message.data : "");
}

static void device_lost_callback(const WGPUDevice*, WGPUDeviceLostReason reason, WGPUStringView message, void* userdata1, void*)
{
    Device* device = (Device*)userdata1;
    if (!device || reason == WGPUDeviceLostReason_Destroyed)
        return;

    device->lost = true;
    log_error("device %llu lost reason=%d: %.*s", (unsigned long long)device->id, (int)reason, (int)message.length, message.data ? message.data : "");
}

static void uncaptured_error_callback(const WGPUDevice*, WGPUErrorType type, WGPUStringView message, void* userdata1, void*)
{
    Device* device = (Device*)userdata1;
    if (device)
        device->error = true;

    log_error("device %llu uncaptured error type=%d: %.*s",
              device ? (unsigned long long)device->id : 0,
              (int)type, (int)message.length, message.data ? message.data : "");
}

static bool same_physical_device(const PhysicalDevice& a, const PhysicalDevice& b)
{
    return a.vendor_id == b.vendor_id
           && a.device_id == b.device_id
           && a.backend_type == b.backend_type
           && a.adapter_type == b.adapter_type
           && a.device_name == b.device_name
           && a.driver_name == b.driver_name
           && a.vendor_name == b.vendor_name
           && a.architecture_name == b.architecture_name;
}

#if defined(__EMSCRIPTEN__)
static std::optional<VkComponentTypeKHR> cooperative_matrix_component_type(uint32_t type)
{
    if (type == 1)
        return VK_COMPONENT_TYPE_FLOAT16_KHR;
    if (type == 2)
        return VK_COMPONENT_TYPE_FLOAT32_KHR;
    if (type == 3)
        return VK_COMPONENT_TYPE_SINT8_KHR;
    if (type == 4)
        return VK_COMPONENT_TYPE_UINT8_KHR;
    if (type == 5)
        return VK_COMPONENT_TYPE_SINT32_KHR;
    if (type == 6)
        return VK_COMPONENT_TYPE_UINT32_KHR;

    return std::nullopt;
}

WGPUFeatureName cooperative_matrix_feature_name()
{
    return (WGPUFeatureName)0x00050034;
}
#else
static std::optional<VkComponentTypeKHR> cooperative_matrix_component_type(WGPUSubgroupMatrixComponentType type)
{
    if (type == WGPUSubgroupMatrixComponentType_F16)
        return VK_COMPONENT_TYPE_FLOAT16_KHR;
    if (type == WGPUSubgroupMatrixComponentType_F32)
        return VK_COMPONENT_TYPE_FLOAT32_KHR;
    if (type == WGPUSubgroupMatrixComponentType_I8)
        return VK_COMPONENT_TYPE_SINT8_KHR;
    if (type == WGPUSubgroupMatrixComponentType_U8)
        return VK_COMPONENT_TYPE_UINT8_KHR;
    if (type == WGPUSubgroupMatrixComponentType_I32)
        return VK_COMPONENT_TYPE_SINT32_KHR;
    if (type == WGPUSubgroupMatrixComponentType_U32)
        return VK_COMPONENT_TYPE_UINT32_KHR;

    return std::nullopt;
}

WGPUFeatureName cooperative_matrix_feature_name()
{
    return WGPUFeatureName_ChromiumExperimentalSubgroupMatrix;
}
#endif

bool cooperative_matrix_translation_probe();

static std::shared_ptr<PhysicalDevice> request_physical_device(Instance* instance,
                                                               WGPUPowerPreference power_preference,
                                                               bool force_fallback,
                                                               std::string& error_message)
{
    std::shared_ptr<AdapterResult> result = std::make_shared<AdapterResult>();
    WGPURequestAdapterOptions options = WGPU_REQUEST_ADAPTER_OPTIONS_INIT;
    options.powerPreference = power_preference;
    options.forceFallbackAdapter = force_fallback ? WGPU_TRUE : WGPU_FALSE;
    WGPURequestAdapterCallbackInfo callback_info = WGPU_REQUEST_ADAPTER_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = request_adapter_callback;
    std::shared_ptr<AdapterResult>* callback_context = new std::shared_ptr<AdapterResult>(result);
    callback_info.userdata1 = callback_context;

    WGPUFuture future = wgpuInstanceRequestAdapter(instance->instance, &options, callback_info);
    if (future.id == 0)
    {
        delete callback_context;
        error_message = "request adapter returned an invalid future";
        return std::shared_ptr<PhysicalDevice>();
    }
    if (wait_future(instance, future, UINT64_MAX, "request-adapter") != 0
        || result->status != WGPURequestAdapterStatus_Success
        || !result->adapter)
    {
        error_message = result->message;
        return std::shared_ptr<PhysicalDevice>();
    }

    std::shared_ptr<PhysicalDevice> physical_device = std::make_shared<PhysicalDevice>();
    physical_device->owner = instance;
    physical_device->adapter = result->adapter;
    result->adapter = 0;

    if (wgpuAdapterGetLimits(physical_device->adapter, &physical_device->limits) != WGPUStatus_Success)
    {
        error_message = "get adapter limits failed";
        return std::shared_ptr<PhysicalDevice>();
    }

    bool adapter_info_available = false;
#if defined(__EMSCRIPTEN__)
    // Read GPUAdapter.info once and sanitize every optional field here.
    // Emdawnwebgpu's wgpuAdapterGetInfo() directly dereferences adapter.info
    // and therefore throws in browsers that only expose requestAdapterInfo().
    {
        char vendor[256] = {0};
        char architecture[256] = {0};
        char device_name[256] = {0};
        char description[256] = {0};
        uint32_t subgroup_min_size = 0;
        uint32_t subgroup_max_size = 0;
        const int info_result = nikki_read_adapter_info(
            (uintptr_t)physical_device->adapter,
            vendor, sizeof(vendor),
            architecture, sizeof(architecture),
            device_name, sizeof(device_name),
            description, sizeof(description),
            &subgroup_min_size, &subgroup_max_size);
        adapter_info_available = info_result != 0;
        if (adapter_info_available)
        {
            physical_device->backend_type = WGPUBackendType_WebGPU;
            physical_device->adapter_type = info_result == 2 ? WGPUAdapterType_CPU : WGPUAdapterType_Unknown;
            physical_device->device_name = device_name;
            physical_device->driver_name = description;
            physical_device->vendor_name = vendor;
            physical_device->architecture_name = architecture;
            physical_device->profile.shader.subgroup.min_size = subgroup_min_size;
            physical_device->profile.shader.subgroup.max_size = subgroup_max_size;
        }
    }

    const int matrix_config_count = nikki_read_subgroup_matrix_configs(
        (uintptr_t)physical_device->adapter, 0, 0);
    if (matrix_config_count > 0)
    {
        std::vector<uint32_t> matrix_configs(matrix_config_count * 5);
        const int read_count = nikki_read_subgroup_matrix_configs(
            (uintptr_t)physical_device->adapter, matrix_configs.data(), matrix_config_count);
        if (read_count == matrix_config_count)
        {
            for (int i = 0; i < matrix_config_count; i++)
            {
                const std::optional<VkComponentTypeKHR> component_type = cooperative_matrix_component_type(matrix_configs[i * 5]);
                const std::optional<VkComponentTypeKHR> result_component_type = cooperative_matrix_component_type(matrix_configs[i * 5 + 1]);
                if (!component_type.has_value() || !result_component_type.has_value())
                    continue;

                CooperativeMatrixConfig config;
                config.component_type = component_type.value();
                config.result_component_type = result_component_type.value();
                config.M = matrix_configs[i * 5 + 2];
                config.N = matrix_configs[i * 5 + 3];
                config.K = matrix_configs[i * 5 + 4];
                physical_device->profile.shader.cooperative_matrix.configs.push_back(config);
            }
        }
    }
#else
    WGPUAdapterInfo adapter_info = WGPU_ADAPTER_INFO_INIT;
    if (wgpuAdapterGetInfo(physical_device->adapter, &adapter_info) == WGPUStatus_Success)
    {
        adapter_info_available = true;
        physical_device->vendor_id = adapter_info.vendorID;
        physical_device->device_id = adapter_info.deviceID;
        physical_device->backend_type = adapter_info.backendType;
        physical_device->adapter_type = adapter_info.adapterType;
        physical_device->device_name = string_from_webgpu(adapter_info.device);
        physical_device->driver_name = string_from_webgpu(adapter_info.description);
        physical_device->vendor_name = string_from_webgpu(adapter_info.vendor);
        physical_device->architecture_name = string_from_webgpu(adapter_info.architecture);
        physical_device->profile.shader.subgroup.min_size = adapter_info.subgroupMinSize;
        physical_device->profile.shader.subgroup.max_size = adapter_info.subgroupMaxSize;

        if (physical_device->device_name.empty())
            physical_device->device_name = physical_device->architecture_name;
        if (physical_device->driver_name.empty())
            physical_device->driver_name = physical_device->vendor_name;

        wgpuAdapterInfoFreeMembers(adapter_info);
    }

    if (wgpuAdapterHasFeature(physical_device->adapter, cooperative_matrix_feature_name()) == WGPU_TRUE)
    {
        WGPUAdapterPropertiesSubgroupMatrixConfigs matrix_properties = WGPU_ADAPTER_PROPERTIES_SUBGROUP_MATRIX_CONFIGS_INIT;
        WGPUAdapterInfo matrix_info = WGPU_ADAPTER_INFO_INIT;
        matrix_info.nextInChain = &matrix_properties.chain;
        if (wgpuAdapterGetInfo(physical_device->adapter, &matrix_info) == WGPUStatus_Success)
        {
            for (size_t i = 0; i < matrix_properties.configCount; i++)
            {
                const WGPUSubgroupMatrixConfig& webgpu_config = matrix_properties.configs[i];
                const std::optional<VkComponentTypeKHR> component_type = cooperative_matrix_component_type(webgpu_config.componentType);
                const std::optional<VkComponentTypeKHR> result_component_type = cooperative_matrix_component_type(webgpu_config.resultComponentType);
                if (!component_type.has_value() || !result_component_type.has_value())
                    continue;

                CooperativeMatrixConfig config;
                config.component_type = component_type.value();
                config.result_component_type = result_component_type.value();
                config.M = webgpu_config.M;
                config.N = webgpu_config.N;
                config.K = webgpu_config.K;
                physical_device->profile.shader.cooperative_matrix.configs.push_back(config);
            }
        }
        wgpuAdapterPropertiesSubgroupMatrixConfigsFreeMembers(matrix_properties);
        wgpuAdapterInfoFreeMembers(matrix_info);
    }
#endif

    if (physical_device->device_name.empty())
        physical_device->device_name = physical_device->architecture_name;
    if (physical_device->driver_name.empty())
        physical_device->driver_name = physical_device->vendor_name;

    const bool subgroup_translation_supported = instance->translator->wgsl_allowed_features.features.count(
                                                    tint::wgsl::LanguageFeature::kSubgroupId)
                                                != 0;
    finalize_device_profile(physical_device.get(), adapter_info_available,
                            instance->packed_4x8_integer_dot_product_supported,
                            subgroup_translation_supported,
                            instance->cooperative_matrix_translation_supported);

    if (physical_device->device_name.empty())
        physical_device->device_name = "WebGPU";
    if (physical_device->driver_name.empty())
        physical_device->driver_name = "nikki";

    return physical_device;
}

VkResult ensure_physical_devices(Instance* instance)
{
    if (!instance || !instance->instance)
        return VK_ERROR_INITIALIZATION_FAILED;
    if (instance->physical_devices_enumerated)
        return instance->physical_devices.empty() ? VK_ERROR_INITIALIZATION_FAILED : VK_SUCCESS;

    const WGPUPowerPreference power_preferences[] = {
        WGPUPowerPreference_HighPerformance,
        WGPUPowerPreference_LowPower,
        WGPUPowerPreference_Undefined};
    const bool force_fallback[] = {false, false, true};
    std::string last_error;
    for (size_t i = 0; i < sizeof(power_preferences) / sizeof(power_preferences[0]); i++)
    {
        std::shared_ptr<PhysicalDevice> physical_device = request_physical_device(instance, power_preferences[i], force_fallback[i], last_error);
        if (!physical_device)
            continue;

        bool duplicate = false;
        for (size_t j = 0; j < instance->physical_devices.size(); j++)
        {
            if (same_physical_device(*instance->physical_devices[j]->impl, *physical_device))
            {
                duplicate = true;
                break;
            }
        }
        if (duplicate)
            continue;

        VkPhysicalDevice_T* handle = new VkPhysicalDevice_T;
        handle->impl = physical_device;
        instance->physical_devices.push_back(handle);
    }

    instance->physical_devices_enumerated = true;
    if (instance->physical_devices.empty())
    {
        log_error("request adapters failed: %s", last_error.empty() ? "no WebGPU adapter" : last_error.c_str());
        return VK_ERROR_INITIALIZATION_FAILED;
    }

    return VK_SUCCESS;
}

VkResult request_backend_device(const std::shared_ptr<Device>& device,
                                PhysicalDevice* physical_device,
                                WGPUDeviceDescriptor descriptor)
{
    if (!device || !physical_device)
        return VK_ERROR_INITIALIZATION_FAILED;

    descriptor.deviceLostCallbackInfo.mode = WGPUCallbackMode_AllowSpontaneous;
    descriptor.deviceLostCallbackInfo.callback = device_lost_callback;
    descriptor.deviceLostCallbackInfo.userdata1 = device.get();
    descriptor.uncapturedErrorCallbackInfo.callback = uncaptured_error_callback;
    descriptor.uncapturedErrorCallbackInfo.userdata1 = device.get();

    std::shared_ptr<DeviceResult> result = std::make_shared<DeviceResult>();
    result->owner = device;
    WGPURequestDeviceCallbackInfo callback_info = WGPU_REQUEST_DEVICE_CALLBACK_INFO_INIT;
    callback_info.mode = WGPUCallbackMode_WaitAnyOnly;
    callback_info.callback = request_device_callback;
    std::shared_ptr<DeviceResult>* callback_context = new std::shared_ptr<DeviceResult>(result);
    callback_info.userdata1 = callback_context;

    WGPUFuture future = wgpuAdapterRequestDevice(physical_device->adapter, &descriptor, callback_info);
    if (future.id == 0)
    {
        delete callback_context;
        return VK_ERROR_INITIALIZATION_FAILED;
    }
    if (wait_future(physical_device->owner, future, UINT64_MAX, "request-device") != 0
        || result->status != WGPURequestDeviceStatus_Success
        || !result->device)
        return VK_ERROR_INITIALIZATION_FAILED;

    device->device = result->device;
    result->device = 0;
    result->owner.reset();
    device->queue = wgpuDeviceGetQueue(device->device);
    if (!device->queue || wgpuDeviceGetLimits(device->device, &device->limits) != WGPUStatus_Success)
        return VK_ERROR_INITIALIZATION_FAILED;

    return VK_SUCCESS;
}

// instance, physical device and device

} // namespace nikki_detail

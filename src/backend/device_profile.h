// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_BACKEND_DEVICE_PROFILE_H
#define NIKKI_BACKEND_DEVICE_PROFILE_H

#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace nikki_detail {

struct PhysicalDevice;

struct CooperativeMatrixConfig
{
    VkComponentTypeKHR component_type;
    VkComponentTypeKHR result_component_type;
    uint32_t M;
    uint32_t N;
    uint32_t K;
};

struct SubgroupProfile
{
    SubgroupProfile()
        : supported(false), operations_supported(false), size_control_supported(false), min_size(0), max_size(0), default_size(4)
    {
    }

    bool supported;
    bool operations_supported;
    bool size_control_supported;
    uint32_t min_size;
    uint32_t max_size;
    uint32_t default_size;
};

struct CooperativeMatrixProfile
{
    CooperativeMatrixProfile()
        : supported(false)
    {
    }

    bool supported;
    std::vector<CooperativeMatrixConfig> configs;
};

struct ShaderDeviceProfile
{
    ShaderDeviceProfile()
        : shader_f16(false), packed_integer_dot(false)
    {
    }

    bool shader_f16;
    bool packed_integer_dot;
    SubgroupProfile subgroup;
    CooperativeMatrixProfile cooperative_matrix;
};

struct DeviceProfile
{
    DeviceProfile()
        : timestamp_query(false)
    {
    }

    ShaderDeviceProfile shader;
    bool timestamp_query;
};

struct EnabledShaderProfile
{
    EnabledShaderProfile()
        : shader_f16(false), packed_integer_dot(false), subgroup(false), subgroup_size_control(false), subgroup_extended_types(false), cooperative_matrix(false)
    {
    }

    bool shader_f16;
    bool packed_integer_dot;
    bool subgroup;
    bool subgroup_size_control;
    bool subgroup_extended_types;
    bool cooperative_matrix;
};

struct EnabledDeviceProfile
{
    EnabledDeviceProfile()
        : timestamp_query(false)
    {
    }

    EnabledShaderProfile shader;
    bool timestamp_query;
};

bool device_extension_supported(const PhysicalDevice* physical_device, const char* name);
void finalize_device_profile(PhysicalDevice* physical_device,
                             bool adapter_info_available,
                             bool packed_integer_dot_supported,
                             bool subgroup_translation_supported,
                             bool cooperative_matrix_translation_supported);

} // namespace nikki_detail

#endif // NIKKI_BACKEND_DEVICE_PROFILE_H

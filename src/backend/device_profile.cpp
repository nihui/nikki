// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#include "objects.h"

namespace nikki_detail {

void finalize_device_profile(PhysicalDevice* physical_device,
                             bool adapter_info_available,
                             bool packed_integer_dot_supported,
                             bool subgroup_translation_supported,
                             bool cooperative_matrix_translation_supported)
{
    if (!physical_device)
        return;

    SubgroupProfile& subgroup = physical_device->profile.shader.subgroup;
    subgroup.supported = adapter_info_available
                         && wgpuAdapterHasFeature(physical_device->adapter, WGPUFeatureName_Subgroups) == WGPU_TRUE
                         && subgroup.min_size >= 4
                         && subgroup.max_size <= 128
                         && subgroup.max_size >= subgroup.min_size
                         && is_power_of_two(subgroup.min_size)
                         && is_power_of_two(subgroup.max_size);
    physical_device->profile.shader.shader_f16 = wgpuAdapterHasFeature(physical_device->adapter, WGPUFeatureName_ShaderF16) == WGPU_TRUE;
    physical_device->profile.shader.packed_integer_dot = packed_integer_dot_supported;
    physical_device->profile.timestamp_query = wgpuAdapterHasFeature(physical_device->adapter, WGPUFeatureName_TimestampQuery) == WGPU_TRUE;
    subgroup.size_control_supported = subgroup.supported
                                      && wgpuAdapterHasFeature(physical_device->adapter, WGPUFeatureName_SubgroupSizeControl) == WGPU_TRUE
                                      && subgroup_translation_supported;
    subgroup.operations_supported = subgroup.supported
                                    && subgroup_translation_supported
                                    && (subgroup.min_size == subgroup.max_size
                                        || subgroup.size_control_supported);
    if (subgroup.operations_supported)
        subgroup.default_size = subgroup.max_size;
    if (!subgroup.supported)
    {
        subgroup.operations_supported = false;
        subgroup.size_control_supported = false;
        subgroup.min_size = 0;
        subgroup.max_size = 0;
    }

    CooperativeMatrixProfile& cooperative_matrix = physical_device->profile.shader.cooperative_matrix;
    cooperative_matrix.supported = subgroup.operations_supported
                                   && cooperative_matrix_translation_supported
                                   && !cooperative_matrix.configs.empty();
    if (!cooperative_matrix.supported)
        cooperative_matrix.configs.clear();
}

} // namespace nikki_detail

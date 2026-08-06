// Copyright 2026 nihui
// SPDX-License-Identifier: Apache-2.0

#ifndef NIKKI_BACKEND_SHADER_MODULE_H
#define NIKKI_BACKEND_SHADER_MODULE_H

#include "device.h"

namespace nikki_detail {

WGPUShaderModule create_shader_module(Device* device, const std::string& wgsl);
int pop_error_scope(Device* device, const char* stage);

} // namespace nikki_detail

#endif // NIKKI_BACKEND_SHADER_MODULE_H

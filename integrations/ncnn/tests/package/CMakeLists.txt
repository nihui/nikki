# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(ncnn_nikki_package_integration LANGUAGES CXX)

enable_testing()
find_package(ncnn CONFIG REQUIRED)
find_package(nikki CONFIG REQUIRED)
set(NCNN_VULKAN ON)
include("${CMAKE_CURRENT_LIST_DIR}/../../ncnn_nikki.cmake")
ncnn_nikki_register_ncnn_tests()

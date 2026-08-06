# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(ncnn_nikki_integration LANGUAGES C CXX)

enable_testing()
set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NIKKI_BUILD_STATIC OFF CACHE BOOL "" FORCE)
set(NIKKI_BUILD_DRIVER ON CACHE BOOL "" FORCE)
set(NIKKI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NIKKI_BUILD_AMALGAMATION OFF CACHE BOOL "" FORCE)
set(NIKKI_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../../../../ncnn" ncnn)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../../.." nikki)
include("${CMAKE_CURRENT_LIST_DIR}/../../ncnn_nikki.cmake")
ncnn_nikki_register_ncnn_tests()

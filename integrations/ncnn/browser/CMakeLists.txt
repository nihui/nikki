# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(ncnn_nikki_browser LANGUAGES C CXX)

set(NIKKI_NCNN_SOURCE_DIR "" CACHE PATH "path to the ncnn source tree")
if(NOT EXISTS "${NIKKI_NCNN_SOURCE_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "NIKKI_NCNN_SOURCE_DIR must point to the ncnn source tree")
endif()

set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NCNN_SIMPLEVK ON CACHE BOOL "" FORCE)
set(NIKKI_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(NIKKI_BUILD_DRIVER OFF CACHE BOOL "" FORCE)
set(NIKKI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NIKKI_INSTALL OFF CACHE BOOL "" FORCE)

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../.." nikki)
add_subdirectory("${NIKKI_NCNN_SOURCE_DIR}" ncnn)
target_link_libraries(ncnn PUBLIC "$<BUILD_INTERFACE:nikki::static>")

if(NCNN_BUILD_TESTS)
    include("${CMAKE_CURRENT_LIST_DIR}/webgpu_tests.cmake")
    nikki_configure_ncnn_browser_tests()
endif()

# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(nikki_superbuild_check LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(NIKKI_BUILD_STATIC ON CACHE BOOL "" FORCE)
set(NIKKI_BUILD_DRIVER OFF CACHE BOOL "" FORCE)
set(NIKKI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NIKKI_BUILD_AMALGAMATION OFF CACHE BOOL "" FORCE)
set(NIKKI_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../../.." nikki)

if(NOT CMAKE_CXX_STANDARD EQUAL 17)
    message(FATAL_ERROR "nikki changed its parent C++ standard")
endif()
if(NOT TARGET nikki::static)
    message(FATAL_ERROR "nikki superbuild target is missing")
endif()

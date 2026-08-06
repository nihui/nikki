# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)
project(nikki_package_consumer LANGUAGES CXX)

find_package(nikki CONFIG REQUIRED)

if(NOT TARGET nikki::driver)
    message(FATAL_ERROR "nikki package does not provide its documented driver target")
endif()

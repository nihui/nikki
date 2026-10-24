# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.22)
project(nikki_installed_consumer LANGUAGES C CXX)
find_package(nikki CONFIG REQUIRED)
add_executable(test_nikki_compute runtime/compute_test.cpp)
target_link_libraries(test_nikki_compute PRIVATE nikki::static)
target_link_options(test_nikki_compute PRIVATE
    "--no-entry" "-sMODULARIZE=1" "-sEXPORT_ES6=1"
    "-sENVIRONMENT=worker" "-sEXPORTED_FUNCTIONS=['_main']"
    "-sEXPORTED_RUNTIME_METHODS=['ccall']")
set_target_properties(test_nikki_compute PROPERTIES SUFFIX ".js")
foreach(asset webgpu_test.html webgpu_test_worker.mjs)
    configure_file("browser/${asset}" "${asset}" COPYONLY)
endforeach()

# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

set(NIKKI_NCNN_BROWSER_DIR ${CMAKE_CURRENT_LIST_DIR})

function(nikki_configure_ncnn_browser_tests)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        message(FATAL_ERROR "nikki ncnn browser tests require Emscripten")
    endif()
    if(NOT TARGET ncnn)
        message(FATAL_ERROR "nikki ncnn browser tests require the ncnn target")
    endif()

    get_target_property(ncnn_source_dir ncnn SOURCE_DIR)
    get_filename_component(ncnn_source_dir "${ncnn_source_dir}/.." ABSOLUTE)
    if(NOT EXISTS "${ncnn_source_dir}/tests/CMakeLists.txt")
        message(FATAL_ERROR "Could not locate the ncnn test source directory")
    endif()

    get_property(ncnn_test_names
        DIRECTORY "${ncnn_source_dir}/tests"
        PROPERTY TESTS)
    if(NOT ncnn_test_names)
        message(FATAL_ERROR
            "ncnn tests have not been registered; enable NCNN_BUILD_TESTS and "
            "call nikki_configure_ncnn_browser_tests() after adding ncnn")
    endif()

    set(ncnn_webgpu_test_output_dir "${CMAKE_BINARY_DIR}/tests/webgpu")
    file(MAKE_DIRECTORY "${ncnn_webgpu_test_output_dir}")
    configure_file(
        "${NIKKI_NCNN_BROWSER_DIR}/webgpu_test.html"
        "${ncnn_webgpu_test_output_dir}/webgpu_test.html"
        COPYONLY)
    configure_file(
        "${NIKKI_NCNN_BROWSER_DIR}/webgpu_test_worker.mjs"
        "${ncnn_webgpu_test_output_dir}/webgpu_test_worker.mjs"
        COPYONLY)

    if(NOT TARGET webgpu_tests)
        add_custom_target(webgpu_tests)
        set_property(TARGET webgpu_tests PROPERTY FOLDER "tests/webgpu")
    endif()

    foreach(ncnn_test_name IN LISTS ncnn_test_names)
        if(NOT TARGET ${ncnn_test_name})
            continue()
        endif()

        target_link_options(${ncnn_test_name} PRIVATE
            "--no-entry"
            "-sMODULARIZE=1"
            "-sEXPORT_ES6=1"
            "-sENVIRONMENT=worker"
            "-sEXPORTED_FUNCTIONS=['_main']"
            "-sEXPORTED_RUNTIME_METHODS=['ccall']")
        set_target_properties(${ncnn_test_name} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${ncnn_webgpu_test_output_dir}"
            SUFFIX ".js")
        add_dependencies(webgpu_tests ${ncnn_test_name})
    endforeach()

endfunction()

# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

set(NIKKI_NCNN_INTEGRATION_DIR "${CMAKE_CURRENT_LIST_DIR}")

function(ncnn_nikki_register_ncnn_tests)
    if(NOT NCNN_VULKAN OR NOT TARGET nikki::driver OR NOT TARGET ncnn)
        return()
    endif()

    if(NOT TARGET nikki_ncnn_model_test)
        add_executable(nikki_ncnn_model_test
            "${NIKKI_NCNN_INTEGRATION_DIR}/tests/model_test.cpp")
        target_compile_features(nikki_ncnn_model_test PRIVATE cxx_std_11)
        target_link_libraries(nikki_ncnn_model_test PRIVATE ncnn)
        add_test(NAME nikki_ncnn_model COMMAND "${CMAKE_COMMAND}" -E env
            "NCNN_VULKAN_DRIVER=$<TARGET_FILE:nikki::driver>"
            $<TARGET_FILE:nikki_ncnn_model_test>)
        set_tests_properties(nikki_ncnn_model PROPERTIES
            LABELS "nikki;ncnn;integration"
            SKIP_RETURN_CODE 77
            TIMEOUT 1800)
    endif()
endfunction()

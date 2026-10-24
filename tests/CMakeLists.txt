# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

add_executable(test_nikki_compute runtime/compute_test.cpp)
add_executable(test_nikki_profile api/profile_test.cpp)
add_executable(test_nikki_abi_guard api/abi_guard_test.cpp)
add_executable(test_nikki_non_conformant_warning
    api/non_conformant_warning_test.cpp)
set(nikki_test_targets
    test_nikki_compute
    test_nikki_profile
    test_nikki_abi_guard
    test_nikki_non_conformant_warning)

if(TARGET nikki)
    add_executable(test_nikki_cooperative_matrix_translation
        translator/cooperative_matrix_test.cpp)
    nikki_configure_target(test_nikki_cooperative_matrix_translation)
    target_link_libraries(test_nikki_cooperative_matrix_translation PRIVATE nikki::static)
    set_property(TARGET test_nikki_cooperative_matrix_translation
        PROPERTY FOLDER "nikki/tests")
endif()

add_library(nikki_32bit_handle_compile OBJECT
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/instance.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/device.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/memory.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/pipeline.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/pipeline_cache.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/command.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/queue.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/sync.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/../src/entrypoints.cpp")
target_compile_definitions(nikki_32bit_handle_compile PRIVATE
    VK_USE_64_BIT_PTR_DEFINES=0)
nikki_configure_target(nikki_32bit_handle_compile)

add_library(nikki_translator_header_compile OBJECT
    translator/header_compile.cpp)
nikki_configure_target(nikki_translator_header_compile)

foreach(nikki_test_target IN LISTS nikki_test_targets)
    target_include_directories(${nikki_test_target} PRIVATE
        "${NIKKI_DAWN_SOURCE_DIR}/third_party/vulkan-headers/src/include"
        "${NIKKI_GENERATED_INCLUDE_DIR}")
endforeach()

if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    # Keep the compute test browser-loadable, rather than merely proving that
    # the Emscripten linker can produce a wasm file.
    target_link_options(test_nikki_compute PRIVATE
        "--no-entry"
        "-sMODULARIZE=1"
        "-sEXPORT_ES6=1"
        "-sENVIRONMENT=worker"
        "-sEXPORTED_FUNCTIONS=['_main']"
        "-sEXPORTED_RUNTIME_METHODS=['ccall']")
    set_target_properties(test_nikki_compute PROPERTIES SUFFIX ".js")
    foreach(nikki_browser_asset webgpu_test.html webgpu_test_worker.mjs)
        configure_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/browser/${nikki_browser_asset}"
            "${CMAKE_CURRENT_BINARY_DIR}/${nikki_browser_asset}"
            COPYONLY)
    endforeach()

    if(NOT TARGET nikki)
        message(FATAL_ERROR "nikki tests require NIKKI_BUILD_STATIC")
    endif()
    foreach(nikki_test_target IN LISTS nikki_test_targets)
        target_link_libraries(${nikki_test_target} PRIVATE nikki::static)
    endforeach()
else()
    if(TARGET nikki_native)
        foreach(nikki_test_target IN LISTS nikki_test_targets)
            target_link_libraries(${nikki_test_target} PRIVATE nikki::driver)
        endforeach()
    elseif(TARGET nikki)
        foreach(nikki_test_target IN LISTS nikki_test_targets)
            target_link_libraries(${nikki_test_target} PRIVATE nikki::static)
        endforeach()
    else()
        message(FATAL_ERROR "nikki tests require a nikki library target")
    endif()
    add_test(NAME test_nikki_compute COMMAND test_nikki_compute)
    add_test(NAME test_nikki_profile COMMAND test_nikki_profile)
    add_test(NAME test_nikki_abi_guard COMMAND test_nikki_abi_guard)
    add_test(NAME test_nikki_non_conformant_warning
        COMMAND "${CMAKE_COMMAND}"
            -DTEST_EXECUTABLE=$<TARGET_FILE:test_nikki_non_conformant_warning>
            -DPROJECT_NAME=nikki
            -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/non_conformant_warning_test.cmake")
    if(TARGET test_nikki_cooperative_matrix_translation)
        add_test(NAME test_nikki_cooperative_matrix_translation
            COMMAND test_nikki_cooperative_matrix_translation)
        set_tests_properties(test_nikki_cooperative_matrix_translation PROPERTIES
            LABELS "nikki;translator;cooperative-matrix")
    endif()
    set_tests_properties(test_nikki_compute PROPERTIES
        LABELS "nikki;runtime;shader;queue")
    set_tests_properties(test_nikki_profile PROPERTIES
        LABELS "nikki;api;profile")
    set_tests_properties(test_nikki_abi_guard PROPERTIES
        LABELS "nikki;api;abi")
    set_tests_properties(test_nikki_non_conformant_warning PROPERTIES
        LABELS "nikki;api;conformance")

    if(Python3_Interpreter_FOUND)
        add_test(NAME test_nikki_source_manifest
            COMMAND "${Python3_EXECUTABLE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/cmake/manifest_test.py")
        set_tests_properties(test_nikki_source_manifest PROPERTIES
            LABELS "nikki;structure;amalgamation")

        if(NIKKI_BUILD_AMALGAMATION OR NIKKI_INSTALL)
            add_test(NAME test_nikki_amalgamation_fresh
                COMMAND "${Python3_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/../tools/amalgamate.py"
                    --check "${NIKKI_AMALGAMATION_SOURCE}"
                    --version-header "${NIKKI_GENERATED_INCLUDE_DIR}/nikki/version.h")
            set_tests_properties(test_nikki_amalgamation_fresh PROPERTIES
                LABELS "nikki;structure;amalgamation")
        endif()
    endif()

    if(TARGET nikki_native AND UNIX AND NOT APPLE AND CMAKE_NM)
        add_test(NAME test_nikki_driver_exports
            COMMAND "${CMAKE_COMMAND}"
                "-DNIKKI_NM=${CMAKE_NM}"
                "-DNIKKI_DRIVER=$<TARGET_FILE:nikki_native>"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/check_exports.cmake")
        set_tests_properties(test_nikki_driver_exports PROPERTIES
            LABELS "nikki;api;abi")
    endif()
endif()

set_property(TARGET ${nikki_test_targets}
    nikki_32bit_handle_compile
    nikki_translator_header_compile
    PROPERTY FOLDER "nikki/tests")

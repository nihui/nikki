# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.22)

set(NIKKI_VERSION 0.1.0)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/nikki_version.cmake")
set(NIKKI_GENERATED_INCLUDE_DIR "${CMAKE_CURRENT_BINARY_DIR}/include")
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/include/nikki/version.h.in"
    "${NIKKI_GENERATED_INCLUDE_DIR}/nikki/version.h" @ONLY)
set(NIKKI_STANDALONE OFF)
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_SOURCE_DIR)
    project(nikki
        VERSION ${NIKKI_VERSION}
        DESCRIPTION "Experimental restricted Vulkan API subset over WebGPU"
        HOMEPAGE_URL "https://github.com/nihui/nikki"
        LANGUAGES C CXX)
    set(NIKKI_STANDALONE ON)
endif()

# Select the ABI through the toolchain's configure-time compiler flags, so
# nikki and every dependency use the same architecture (for example -m64).
if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8)
        set(NIKKI_WASM_ARCH "wasm64")
    elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
        set(NIKKI_WASM_ARCH "wasm32")
    else()
        message(FATAL_ERROR "Unsupported Emscripten pointer size: ${CMAKE_SIZEOF_VOID_P}")
    endif()
    set(NIKKI_WASM_POINTER_SIZE "${CMAKE_SIZEOF_VOID_P}")
    include(CheckCXXSourceCompiles)
    unset(NIKKI_WASM_CXX_POINTERS_MATCH_${NIKKI_WASM_POINTER_SIZE} CACHE)
    check_cxx_source_compiles("typedef char nikki_pointer_size_check[sizeof(void*) == ${NIKKI_WASM_POINTER_SIZE} ? 1 : -1]; int main() {}"
        NIKKI_WASM_CXX_POINTERS_MATCH_${NIKKI_WASM_POINTER_SIZE})
    if(NOT NIKKI_WASM_CXX_POINTERS_MATCH_${NIKKI_WASM_POINTER_SIZE})
        message(FATAL_ERROR "Configure both C and CXX flags for ${NIKKI_WASM_ARCH}; their pointer sizes must match")
    endif()
endif()

function(nikki_import_legacy_bool new_name legacy_name default_value description)
    if(DEFINED ${legacy_name} AND NOT DEFINED ${new_name})
        set(${new_name} "${${legacy_name}}" CACHE BOOL "${description}")
        message(DEPRECATION "${legacy_name} is deprecated; use ${new_name}")
    else()
        option(${new_name} "${description}" ${default_value})
    endif()
    set(${legacy_name} "${${new_name}}" CACHE BOOL "Deprecated alias for ${new_name}" FORCE)
    mark_as_advanced(${legacy_name})
endfunction()

nikki_import_legacy_bool(NIKKI_BUILD_STATIC NIKKI_BUILD_STATIC_LIBRARY ON
    "Build the nikki static library")
nikki_import_legacy_bool(NIKKI_BUILD_DRIVER NIKKI_BUILD_NATIVE_DRIVER OFF
    "Build the native nikki shared restricted-API implementation")
nikki_import_legacy_bool(NIKKI_BUILD_AMALGAMATION NIKKI_USE_AMALGAMATION OFF
    "Compile targets from the generated single-source distribution")
option(NIKKI_BUILD_TESTS "Build nikki API tests" ${NIKKI_STANDALONE})
option(NIKKI_INSTALL "Install nikki implementation and package files" ${NIKKI_STANDALONE})

set(NIKKI_DEFAULT_DAWN_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/dawn")
set(NIKKI_DAWN_SOURCE_DIR "${NIKKI_DEFAULT_DAWN_SOURCE_DIR}"
    CACHE PATH "path to the Dawn source tree")
set(NIKKI_DAWN_LIBRARY_TYPE STATIC CACHE STRING
    "Dawn monolithic library type for native builds (STATIC or SHARED)")
set_property(CACHE NIKKI_DAWN_LIBRARY_TYPE PROPERTY STRINGS STATIC SHARED)

set(NIKKI_DAWN_COMMIT "1716e88cc9e58398fab5d4473f7f2e858667b677")
set(NIKKI_DAWN_DEPS_SHA256 "c0f0bce0882ea814a72469c5f72accfab3bde918c5531a9743bce1aa6f69c07c")
set(NIKKI_EMDAWNWEBGPU_PORT_FILENAME "emdawnwebgpu-v20261023.194543.remoteport.py")
set(NIKKI_EMDAWNWEBGPU_PORT_SHA256 "e235ec5616ed5bc3fc4d1b87fd2f9dc16cd8cf91ec1e835b37eea923be9d1219")
set(NIKKI_EMDAWNWEBGPU_PACKAGE_URL "https://github.com/google/dawn/releases/download/v20261023.194543/emdawnwebgpu_pkg-v20261023.194543.zip")
set(NIKKI_EMDAWNWEBGPU_PACKAGE_SHA512 "fc8ce33f1253032b9ba8e6998ca07760d87ef3f943dff40a7dffe5809d8bde4a1d32f0d982c3092122bcaf15e466437cbccb1a3b4ee1d5b3ed6ae2eff3140f12")
set(NIKKI_EMSDK_VERSION "6.0.12")
set(NIKKI_ASYNCIFY_STACK_SIZE "65536")
set(NIKKI_WASM_STACK_SIZE "1048576")

set(NIKKI_EMDAWNWEBGPU_PORT
    "${CMAKE_CURRENT_SOURCE_DIR}/ports/${NIKKI_EMDAWNWEBGPU_PORT_FILENAME}")
set(NIKKI_CLOSURE_EXTERNS
    "${CMAKE_CURRENT_SOURCE_DIR}/ports/webgpu_immediates.externs.js")

function(nikki_validate_environment)
    if(NOT EXISTS "${NIKKI_DAWN_SOURCE_DIR}/CMakeLists.txt")
        message(FATAL_ERROR
            "The Dawn source tree is missing at ${NIKKI_DAWN_SOURCE_DIR}. "
            "Run git submodule update --init third_party/dawn and python3 tools/bootstrap_dawn.py from the nikki project root, "
            "or set NIKKI_DAWN_SOURCE_DIR explicitly.")
    endif()
    if(NOT EXISTS "${NIKKI_DAWN_SOURCE_DIR}/out/.nikki-deps-stamp")
        message(FATAL_ERROR
            "Dawn dependencies are missing. Run tools/bootstrap_dawn.py from the nikki project root.")
    endif()

    find_package(Python3 REQUIRED COMPONENTS Interpreter)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/tools/bootstrap_dawn.py"
            --check --dawn-source-dir "${NIKKI_DAWN_SOURCE_DIR}"
        RESULT_VARIABLE NIKKI_DAWN_CHECK_RESULT
        OUTPUT_VARIABLE NIKKI_DAWN_CHECK_OUTPUT
        ERROR_VARIABLE NIKKI_DAWN_CHECK_ERROR)
    if(NOT NIKKI_DAWN_CHECK_RESULT EQUAL 0)
        message(FATAL_ERROR "Dawn dependency validation failed:\n${NIKKI_DAWN_CHECK_OUTPUT}${NIKKI_DAWN_CHECK_ERROR}")
    endif()

    message(STATUS "nikki Dawn/Tint commit: ${NIKKI_DAWN_COMMIT}")

    if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
            RESULT_VARIABLE NIKKI_EMCC_VERSION_RESULT
            OUTPUT_VARIABLE NIKKI_EMCC_VERSION_OUTPUT
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET)
        if(NOT NIKKI_EMCC_VERSION_RESULT EQUAL 0 OR
           NOT NIKKI_EMCC_VERSION_OUTPUT STREQUAL NIKKI_EMSDK_VERSION)
            message(FATAL_ERROR
                "nikki requires emsdk ${NIKKI_EMSDK_VERSION}, but ${CMAKE_CXX_COMPILER} reports:\n${NIKKI_EMCC_VERSION_OUTPUT}")
        endif()

        if(NOT EXISTS "${NIKKI_EMDAWNWEBGPU_PORT}")
            message(FATAL_ERROR "Missing frozen Emdawnwebgpu port: ${NIKKI_EMDAWNWEBGPU_PORT}")
        endif()
        if(NOT EXISTS "${NIKKI_CLOSURE_EXTERNS}")
            message(FATAL_ERROR "Missing WebGPU immediates Closure externs: ${NIKKI_CLOSURE_EXTERNS}")
        endif()
        file(SHA256 "${NIKKI_EMDAWNWEBGPU_PORT}" NIKKI_ACTUAL_PORT_SHA256)
        if(NOT NIKKI_ACTUAL_PORT_SHA256 STREQUAL NIKKI_EMDAWNWEBGPU_PORT_SHA256)
            message(FATAL_ERROR
                "Emdawnwebgpu port SHA256 mismatch: expected ${NIKKI_EMDAWNWEBGPU_PORT_SHA256}, got ${NIKKI_ACTUAL_PORT_SHA256}")
        endif()

        message(STATUS
            "nikki Emdawnwebgpu port: ${NIKKI_EMDAWNWEBGPU_PORT_FILENAME} (${NIKKI_EMDAWNWEBGPU_PORT_SHA256})")
        message(STATUS "nikki emsdk: ${NIKKI_EMSDK_VERSION}")
    endif()
endfunction()

function(nikki_add_dependencies)
    if(TARGET tint_api)
        return()
    endif()

    set(CMAKE_CXX_STANDARD 20)

    set(DAWN_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        set(DAWN_BUILD_MONOLITHIC_LIBRARY OFF CACHE STRING "" FORCE)
    else()
        set(DAWN_BUILD_MONOLITHIC_LIBRARY ${NIKKI_DAWN_LIBRARY_TYPE} CACHE STRING "" FORCE)
    endif()
    set(DAWN_BUILD_NODE_BINDINGS OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_PROTOBUF OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_SAMPLES OFF CACHE BOOL "" FORCE)
    set(DAWN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(DAWN_FETCH_DEPENDENCIES OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_D3D11 OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_D3D12 OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_DESKTOP_GL OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_METAL OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_NULL OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_OPENGLES OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_SPIRV_VALIDATION OFF CACHE BOOL "" FORCE)
    set(DAWN_ENABLE_VULKAN OFF CACHE BOOL "" FORCE)
    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        if(APPLE)
            set(DAWN_ENABLE_METAL ON CACHE BOOL "" FORCE)
        elseif(WIN32)
            set(DAWN_ENABLE_D3D12 ON CACHE BOOL "" FORCE)
        else()
            set(DAWN_ENABLE_VULKAN ON CACHE BOOL "" FORCE)
        endif()
    endif()
    set(DAWN_USE_GLFW OFF CACHE BOOL "" FORCE)
    set(DAWN_USE_WAYLAND OFF CACHE BOOL "" FORCE)
    set(DAWN_USE_X11 OFF CACHE BOOL "" FORCE)

    set(TINT_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_CMD_TOOLS OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_FUZZERS OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_GLSL_VALIDATOR OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_GLSL_WRITER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_HLSL_WRITER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_IR_BINARY OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_MSL_WRITER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_NULL_WRITER OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_SPV_READER ON CACHE BOOL "" FORCE)
    set(TINT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_TINTD OFF CACHE BOOL "" FORCE)
    set(TINT_BUILD_WGSL_WRITER ON CACHE BOOL "" FORCE)
    set(TINT_ENABLE_IR_DUMPING OFF CACHE BOOL "" FORCE)
    if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        set(TINT_BUILD_SPV_WRITER OFF CACHE BOOL "" FORCE)
        # Compiler regressions parse final WGSL; production translation uses IR.
        set(TINT_BUILD_WGSL_READER ${NIKKI_BUILD_TESTS} CACHE BOOL "" FORCE)
    else()
        set(TINT_BUILD_SPV_WRITER ON CACHE BOOL "" FORCE)
        set(TINT_BUILD_WGSL_READER ON CACHE BOOL "" FORCE)
    endif()

    set(SPIRV-Headers_SOURCE_DIR ${NIKKI_DAWN_SOURCE_DIR}/third_party/spirv-headers/src)
    set(SPIRV_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(SPIRV_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        ${SPIRV-Headers_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/spirv-headers
        EXCLUDE_FROM_ALL)

    set(SKIP_SPIRV_TOOLS_INSTALL ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_EXECUTABLES ON CACHE BOOL "" FORCE)
    set(SPIRV_SKIP_TESTS ON CACHE BOOL "" FORCE)
    set(SPIRV_WERROR OFF CACHE BOOL "" FORCE)
    add_subdirectory(
        ${NIKKI_DAWN_SOURCE_DIR}/third_party/spirv-tools/src
        ${CMAKE_CURRENT_BINARY_DIR}/spirv-tools
        EXCLUDE_FROM_ALL)

    add_subdirectory(
        ${NIKKI_DAWN_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}/dawn
        EXCLUDE_FROM_ALL)
endfunction()

function(nikki_configure_target target)
    target_compile_features(${target} PRIVATE cxx_std_20)
    target_include_directories(${target} PRIVATE "${NIKKI_GENERATED_INCLUDE_DIR}")

    if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        target_link_libraries(${target} PRIVATE emdawnwebgpu_c_include)
        target_compile_options(${target} PRIVATE
            "$<BUILD_INTERFACE:--use-port=${NIKKI_EMDAWNWEBGPU_PORT}:cpp_bindings=false>")
        target_link_options(${target} PUBLIC
            "$<BUILD_INTERFACE:--use-port=${NIKKI_EMDAWNWEBGPU_PORT}:cpp_bindings=false>"
            "-sASYNCIFY=1"
            "-sASYNCIFY_STACK_SIZE=${NIKKI_ASYNCIFY_STACK_SIZE}"
            "-sSTACK_SIZE=${NIKKI_WASM_STACK_SIZE}"
            "$<$<CONFIG:Release>:--closure=1>"
            "$<$<CONFIG:Release>:--closure-args=--externs=${NIKKI_CLOSURE_EXTERNS}>")
    else()
        target_link_libraries(${target} PRIVATE dawn::webgpu_dawn)
    endif()

    target_include_directories(${target} PRIVATE
        "$<BUILD_INTERFACE:${NIKKI_DAWN_SOURCE_DIR}/third_party/vulkan-headers/src/include>"
        "$<BUILD_INTERFACE:${NIKKI_DAWN_SOURCE_DIR}/third_party/spirv-headers/src/include>"
        "$<BUILD_INTERFACE:${NIKKI_DAWN_SOURCE_DIR}/third_party/spirv-tools/src/include>")
    target_link_libraries(${target} PRIVATE
        dawn_utils_config
        SPIRV-Tools-opt
        tint_api
        tint_lang_core_ir_transform)
endfunction()

if(NIKKI_BUILD_STATIC OR NIKKI_BUILD_DRIVER OR NIKKI_BUILD_TESTS)
    nikki_validate_environment()
    nikki_add_dependencies()
endif()

set(NIKKI_SOURCE_MANIFEST
    "${CMAKE_CURRENT_SOURCE_DIR}/src/source_manifest.txt")
file(STRINGS "${NIKKI_SOURCE_MANIFEST}" nikki_source_manifest_lines)
set(NIKKI_MODULE_PARTS)
set(nikki_source_manifest_paths)
foreach(nikki_manifest_line IN LISTS nikki_source_manifest_lines)
    string(STRIP "${nikki_manifest_line}" nikki_manifest_line)
    if(nikki_manifest_line STREQUAL "" OR
       nikki_manifest_line MATCHES "^#")
        continue()
    endif()
    if(NOT nikki_manifest_line MATCHES "^(translator|driver)[ \t]+(.+)$")
        message(FATAL_ERROR
            "Invalid nikki source manifest entry: ${nikki_manifest_line}")
    endif()
    set(nikki_manifest_path "${CMAKE_MATCH_2}")
    if(IS_ABSOLUTE "${nikki_manifest_path}" OR
       NOT nikki_manifest_path MATCHES "^src/" OR
       nikki_manifest_path MATCHES "(^|/)\\.\\.(/|$)")
        message(FATAL_ERROR
            "Invalid nikki source manifest path: ${nikki_manifest_path}")
    endif()
    if(nikki_manifest_path IN_LIST nikki_source_manifest_paths)
        message(FATAL_ERROR
            "Duplicate nikki source manifest path: ${nikki_manifest_path}")
    endif()
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${nikki_manifest_path}")
        message(FATAL_ERROR
            "Missing nikki source manifest file: ${nikki_manifest_path}")
    endif()
    list(APPEND nikki_source_manifest_paths "${nikki_manifest_path}")
    list(APPEND NIKKI_MODULE_PARTS "${nikki_manifest_path}")
endforeach()
if(NOT NIKKI_MODULE_PARTS)
    message(FATAL_ERROR "nikki source manifest is empty")
endif()

set(NIKKI_MODULE_SOURCES ${NIKKI_MODULE_PARTS})
list(FILTER NIKKI_MODULE_SOURCES INCLUDE REGEX "\\.cpp$")

set(NIKKI_AMALGAMATION_SOURCE
    "${CMAKE_CURRENT_BINARY_DIR}/dist/nikki.cpp")

find_package(Python3 QUIET COMPONENTS Interpreter)
if(Python3_Interpreter_FOUND)
    set(nikki_amalgamation_inputs)
    foreach(nikki_module_source IN LISTS NIKKI_MODULE_PARTS)
        list(APPEND nikki_amalgamation_inputs
            "${CMAKE_CURRENT_SOURCE_DIR}/${nikki_module_source}")
    endforeach()

    add_custom_command(
        OUTPUT "${NIKKI_AMALGAMATION_SOURCE}"
        COMMAND ${Python3_EXECUTABLE}
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/amalgamate.py"
            --output "${NIKKI_AMALGAMATION_SOURCE}"
            --version-header "${NIKKI_GENERATED_INCLUDE_DIR}/nikki/version.h"
        DEPENDS
            "${CMAKE_CURRENT_SOURCE_DIR}/tools/amalgamate.py"
            "${NIKKI_SOURCE_MANIFEST}"
            "${NIKKI_GENERATED_INCLUDE_DIR}/nikki/version.h"
            ${nikki_amalgamation_inputs}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Generating ${NIKKI_AMALGAMATION_SOURCE}"
        VERBATIM)
    set_source_files_properties("${NIKKI_AMALGAMATION_SOURCE}"
        PROPERTIES GENERATED ON)

    set(nikki_amalgamate_all)
    if(NIKKI_BUILD_AMALGAMATION OR NIKKI_INSTALL)
        set(nikki_amalgamate_all ALL)
    endif()
    add_custom_target(nikki_amalgamate ${nikki_amalgamate_all}
        DEPENDS "${NIKKI_AMALGAMATION_SOURCE}")
    set_property(TARGET nikki_amalgamate PROPERTY FOLDER "nikki")
elseif(NIKKI_BUILD_AMALGAMATION OR NIKKI_INSTALL)
    message(FATAL_ERROR "Python3 is required for amalgamation and install")
endif()

if(NIKKI_BUILD_AMALGAMATION)
    set(NIKKI_SOURCES "${NIKKI_AMALGAMATION_SOURCE}")
else()
    set(NIKKI_SOURCES ${NIKKI_MODULE_SOURCES})
endif()

if(NIKKI_BUILD_STATIC)
    add_library(nikki STATIC ${NIKKI_SOURCES})
    add_library(nikki::nikki ALIAS nikki)
    add_library(nikki::static ALIAS nikki)
    nikki_configure_target(nikki)
    set_target_properties(nikki PROPERTIES EXPORT_NAME static)
    set_property(TARGET nikki PROPERTY FOLDER "nikki")
endif()

if(NIKKI_BUILD_DRIVER)
    if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        message(FATAL_ERROR "NIKKI_BUILD_NATIVE_DRIVER is not available with Emscripten")
    endif()

    add_library(nikki_native SHARED ${NIKKI_SOURCES})
    add_library(nikki::driver ALIAS nikki_native)
    nikki_configure_target(nikki_native)
    set_target_properties(nikki_native PROPERTIES
        OUTPUT_NAME nikki
        VERSION ${NIKKI_VERSION}
        SOVERSION 0
        EXPORT_NAME driver
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN ON)
    set_property(TARGET nikki_native PROPERTY FOLDER "nikki")
    if(UNIX AND NOT APPLE)
        target_link_options(nikki_native PRIVATE
            "LINKER:--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/nikki.exports.map")
        set_property(TARGET nikki_native APPEND PROPERTY
            LINK_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/src/nikki.exports.map")
    endif()
endif()

if(NIKKI_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

if(NIKKI_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    if(CMAKE_SYSTEM_NAME STREQUAL "Emscripten" AND TARGET nikki)
        include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/nikki_wasm_package.cmake")
    endif()

    if(TARGET nikki_native)
        install(TARGETS nikki_native
            EXPORT nikkiTargets
            RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
            LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
            ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}")
        install(EXPORT nikkiTargets
            FILE nikkiTargets.cmake
            NAMESPACE nikki::
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nikki")
    endif()

    if(NOT CMAKE_SYSTEM_NAME STREQUAL "Emscripten")
        install(FILES "${NIKKI_AMALGAMATION_SOURCE}"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/nikki")
    endif()
    if(TARGET nikki_native)
        configure_package_config_file(
            "${CMAKE_CURRENT_SOURCE_DIR}/nikkiConfig.cmake.in"
            "${CMAKE_CURRENT_BINARY_DIR}/nikkiConfig.cmake"
            INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nikki")
        write_basic_package_version_file(
            "${CMAKE_CURRENT_BINARY_DIR}/nikkiConfigVersion.cmake"
            VERSION ${NIKKI_VERSION}
            COMPATIBILITY SameMajorVersion)
        install(FILES
            "${CMAKE_CURRENT_BINARY_DIR}/nikkiConfig.cmake"
            "${CMAKE_CURRENT_BINARY_DIR}/nikkiConfigVersion.cmake"
            DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nikki")
    endif()
endif()

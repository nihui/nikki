# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

# Install only the static link closure, not all of Dawn's many build targets.
# Flattening it into filenames keeps downstream consumers independent of Dawn's
# internal CMake targets and of this source/build tree.
function(nikki_collect_wasm_libraries target)
    if(NOT TARGET "${target}")
        message(FATAL_ERROR "Unexpected WASM link dependency: ${target}")
    endif()
    get_target_property(alias "${target}" ALIASED_TARGET)
    if(alias)
        set(target "${alias}")
    endif()
    get_property(visited GLOBAL PROPERTY NIKKI_WASM_VISITED)
    if(target IN_LIST visited)
        return()
    endif()
    set_property(GLOBAL APPEND PROPERTY NIKKI_WASM_VISITED "${target}")
    get_target_property(type "${target}" TYPE)
    if(type STREQUAL "STATIC_LIBRARY")
        set_property(GLOBAL APPEND PROPERTY NIKKI_WASM_LIBRARIES "${target}")
    elseif(NOT type STREQUAL "INTERFACE_LIBRARY")
        message(FATAL_ERROR "Unsupported WASM dependency type ${type}: ${target}")
    endif()
    foreach(property LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
        get_target_property(dependencies "${target}" "${property}")
        if(NOT dependencies)
            continue()
        endif()
        foreach(dependency IN LISTS dependencies)
            # CMake inserts directory scope markers for cross-directory links.
            if(dependency MATCHES "^::@")
                continue()
            endif()
            while(dependency MATCHES "^\\$<(LINK_ONLY|BUILD_INTERFACE):(.+)>$")
                set(dependency "${CMAKE_MATCH_2}")
            endwhile()
            if(dependency MATCHES "^\\$<INSTALL_INTERFACE:")
                continue()
            endif()
            if(dependency MATCHES "^\\$<\\$<BOOL:([^>]*)>:(.*)>$")
                set(condition "${CMAKE_MATCH_1}")
                set(dependency "${CMAKE_MATCH_2}")
                if(NOT condition)
                    continue()
                endif()
            endif()
            # Fail closed on new expressions/dependencies instead of silently
            # publishing a package with missing link inputs after a Dawn update.
            nikki_collect_wasm_libraries("${dependency}")
        endforeach()
    endforeach()
endfunction()

# Make the imported Threads target visible at this scope as well as Dawn's.
find_package(Threads REQUIRED)
set_property(GLOBAL PROPERTY NIKKI_WASM_VISITED "")
set_property(GLOBAL PROPERTY NIKKI_WASM_LIBRARIES "")
nikki_collect_wasm_libraries(nikki)
get_property(nikki_wasm_libraries GLOBAL PROPERTY NIKKI_WASM_LIBRARIES)
list(REMOVE_ITEM nikki_wasm_libraries nikki)
set(NIKKI_WASM_DEPENDENCY_FILES "")
foreach(dependency IN LISTS nikki_wasm_libraries)
    # Target names provide collision-free filenames even if OUTPUT_NAME differs.
    set(filename "lib${dependency}.a")
    install(FILES "$<TARGET_FILE:${dependency}>"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/nikki" RENAME "${filename}")
    list(APPEND NIKKI_WASM_DEPENDENCY_FILES "${filename}")
endforeach()
install(FILES "$<TARGET_FILE:nikki>" DESTINATION "${CMAKE_INSTALL_LIBDIR}")
install(DIRECTORY
    "${NIKKI_DAWN_SOURCE_DIR}/third_party/vulkan-headers/src/include/vulkan"
    "${NIKKI_DAWN_SOURCE_DIR}/third_party/vulkan-headers/src/include/vk_video"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}")
# Bundle the same runtime as the source build's frozen remote port. Consumers
# must not depend on the producer's Emscripten cache or download it at first link.
set(nikki_emdawnwebgpu_dir "${CMAKE_CURRENT_BINARY_DIR}/ports")
file(MAKE_DIRECTORY "${nikki_emdawnwebgpu_dir}")
get_filename_component(nikki_emdawnwebgpu_archive "${NIKKI_EMDAWNWEBGPU_PACKAGE_URL}" NAME)
set(nikki_emdawnwebgpu_archive "${nikki_emdawnwebgpu_dir}/${nikki_emdawnwebgpu_archive}")
file(DOWNLOAD "${NIKKI_EMDAWNWEBGPU_PACKAGE_URL}" "${nikki_emdawnwebgpu_archive}"
    EXPECTED_HASH "SHA512=${NIKKI_EMDAWNWEBGPU_PACKAGE_SHA512}" TLS_VERIFY ON)
file(REMOVE_RECURSE "${nikki_emdawnwebgpu_dir}/emdawnwebgpu_pkg")
file(ARCHIVE_EXTRACT INPUT "${nikki_emdawnwebgpu_archive}" DESTINATION "${nikki_emdawnwebgpu_dir}")
file(SHA256 "${nikki_emdawnwebgpu_dir}/emdawnwebgpu_pkg/emdawnwebgpu.port.py"
    NIKKI_EMDAWNWEBGPU_LOCAL_PORT_SHA256)
# Preserve the complete upstream package, including its headers and notices.
install(DIRECTORY "${nikki_emdawnwebgpu_dir}/emdawnwebgpu_pkg"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/nikki/ports")
install(FILES "${NIKKI_CLOSURE_EXTERNS}"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/nikki/ports")
install(FILES "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE.txt"
    "${CMAKE_CURRENT_SOURCE_DIR}/THIRD_PARTY_NOTICES.md" DESTINATION ".")

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/nikkiWasmConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/nikkiConfig.cmake"
    INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nikki"
    PATH_VARS CMAKE_INSTALL_LIBDIR CMAKE_INSTALL_INCLUDEDIR CMAKE_INSTALL_DATADIR)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/nikkiConfigVersion.cmake"
    VERSION "${NIKKI_RELEASE_VERSION}" COMPATIBILITY ExactVersion)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/nikkiConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/nikkiConfigVersion.cmake"
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/nikki")
message(STATUS "nikki WASM package dependencies: ${nikki_wasm_libraries}")
# Include notices for the libraries and headers redistributed in this package.
foreach(component dawn spirv-tools spirv-headers vulkan-headers)
    if(component STREQUAL "dawn")
        set(license_dir "${NIKKI_DAWN_SOURCE_DIR}")
        set(license_file LICENSE)
    else()
        set(license_dir "${NIKKI_DAWN_SOURCE_DIR}/third_party/${component}/src")
        set(license_file LICENSE)
        if(component STREQUAL "vulkan-headers")
            set(license_file LICENSE.md)
        endif()
    endif()
    install(FILES "${license_dir}/${license_file}"
        DESTINATION "${CMAKE_INSTALL_DATADIR}/nikki/licenses/${component}")
    if(IS_DIRECTORY "${license_dir}/LICENSES")
        install(DIRECTORY "${license_dir}/LICENSES/"
            DESTINATION "${CMAKE_INSTALL_DATADIR}/nikki/licenses/${component}")
    endif()
endforeach()

install(FILES "${NIKKI_DAWN_SOURCE_DIR}/third_party/abseil-cpp/LICENSE"
    DESTINATION "${CMAKE_INSTALL_DATADIR}/nikki/licenses/abseil-cpp")

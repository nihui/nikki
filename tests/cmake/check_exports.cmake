# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT DEFINED NIKKI_NM OR NOT DEFINED NIKKI_DRIVER)
    message(FATAL_ERROR "NIKKI_NM and NIKKI_DRIVER are required")
endif()

execute_process(
    COMMAND "${NIKKI_NM}" -D --defined-only "${NIKKI_DRIVER}"
    RESULT_VARIABLE nikki_nm_result
    OUTPUT_VARIABLE nikki_nm_output
    ERROR_VARIABLE nikki_nm_error)
if(NOT nikki_nm_result EQUAL 0)
    message(FATAL_ERROR "symbol inspection failed: ${nikki_nm_error}")
endif()

string(REGEX MATCHALL "[ \t]vk[A-Za-z0-9_]+([\r\n]|$)" nikki_symbol_matches
    "${nikki_nm_output}")
set(nikki_symbols)
foreach(nikki_symbol_match IN LISTS nikki_symbol_matches)
    string(STRIP "${nikki_symbol_match}" nikki_symbol)
    list(APPEND nikki_symbols "${nikki_symbol}")
endforeach()
list(REMOVE_DUPLICATES nikki_symbols)
list(SORT nikki_symbols)

set(nikki_expected_symbols
    vkCreateInstance
    vkEnumerateInstanceExtensionProperties
    vkEnumerateInstanceLayerProperties
    vkGetInstanceProcAddr)
list(SORT nikki_expected_symbols)
if(NOT nikki_symbols STREQUAL nikki_expected_symbols)
    message(FATAL_ERROR
        "unexpected exported Vulkan symbols: ${nikki_symbols}; expected: ${nikki_expected_symbols}")
endif()

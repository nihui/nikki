# Copyright 2026 nihui
# SPDX-License-Identifier: Apache-2.0

if(NOT TEST_EXECUTABLE OR NOT PROJECT_NAME)
    message(FATAL_ERROR "TEST_EXECUTABLE and PROJECT_NAME are required")
endif()

execute_process(
    COMMAND "${TEST_EXECUTABLE}"
    RESULT_VARIABLE test_result
    OUTPUT_VARIABLE test_stdout
    ERROR_VARIABLE test_stderr)
if(NOT test_result EQUAL 0)
    message(FATAL_ERROR
        "non-conformant warning test failed (${test_result})\n${test_stdout}\n${test_stderr}")
endif()

set(expected_warning
    "WARNING: ${PROJECT_NAME} is not a conformant Vulkan implementation, testing use only\\.")
string(REGEX MATCHALL "${expected_warning}" warning_matches "${test_stderr}")
list(LENGTH warning_matches warning_count)
if(NOT warning_count EQUAL 1)
    message(FATAL_ERROR
        "expected exactly one non-conformant warning, found ${warning_count}\n${test_stderr}")
endif()

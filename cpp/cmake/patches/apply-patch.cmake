# apply-patch.cmake
# Idempotent wrapper around `git apply`. Skips if the patch is already applied.
#
# Usage: cmake -DPATCH_FILE=<path> -P apply-patch.cmake
#        (run from the source tree to be patched)

if(NOT DEFINED PATCH_FILE)
    message(FATAL_ERROR "PATCH_FILE must be defined (-DPATCH_FILE=<path>)")
endif()

find_program(GIT_EXECUTABLE git REQUIRED)

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --reverse --check --ignore-whitespace "${PATCH_FILE}"
    RESULT_VARIABLE _already_applied
    OUTPUT_QUIET ERROR_QUIET
)
if(_already_applied EQUAL 0)
    message(STATUS "Patch already applied, skipping: ${PATCH_FILE}")
    return()
endif()

execute_process(
    COMMAND "${GIT_EXECUTABLE}" apply --ignore-whitespace "${PATCH_FILE}"
    RESULT_VARIABLE _result
    ERROR_VARIABLE _error
)
if(NOT _result EQUAL 0)
    message(FATAL_ERROR "Failed to apply patch ${PATCH_FILE}:\n${_error}")
endif()
message(STATUS "Applied patch: ${PATCH_FILE}")

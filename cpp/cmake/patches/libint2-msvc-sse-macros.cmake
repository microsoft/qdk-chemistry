# libint2-msvc-sse-macros.cmake
# Patch script to fix missing __SSE__ / __SSE2__ macro guards in libint2's
# vector_x86.h when compiling with native MSVC cl.exe.
#
# MSVC on x64 always supports SSE2, but does NOT define __SSE__ or __SSE2__.
# It does define __AVX__ / __AVX2__ when /arch:AVX2 is used.  This means the
# AVX section (which references VectorSSEDouble) compiles, but the SSE2 section
# that *defines* VectorSSEDouble is skipped.
#
# Fix: after the `#include <intrin.h>` block, add MSVC-specific macro
# definitions so that the SSE/SSE2 guards work correctly.
#
# Applied via PATCH_COMMAND in FetchContent.
# Usage: cmake -P libint2-msvc-sse-macros.cmake  (run from libint2 source root)

set(_file "libint-2.9.0/include/libint2/util/vector_x86.h")
if(NOT EXISTS "${_file}")
    message(WARNING "libint2-msvc-sse-macros: ${_file} not found, skipping patch")
    return()
endif()

file(READ "${_file}" _content)
set(_original "${_content}")

# Insert MSVC SSE/SSE2 macro definitions after the #include <intrin.h> block.
# MSVC x64 always has SSE2 but doesn't define these macros.
string(REPLACE
    "#if defined(_MSC_VER)
#include <intrin.h>
#elif defined(__SSE2__) || defined(__SSE__) || defined(__AVX__)
#include <x86intrin.h>
#endif"
    "#if defined(_MSC_VER)
#include <intrin.h>
// MSVC x64 always supports SSE/SSE2 but does not define these macros.
// Define them so that the SSE/SSE2 code sections are compiled.
#if defined(_M_X64) || defined(_M_AMD64)
#  if !defined(__SSE__)
#    define __SSE__ 1
#  endif
#  if !defined(__SSE2__)
#    define __SSE2__ 1
#  endif
#endif
#elif defined(__SSE2__) || defined(__SSE__) || defined(__AVX__)
#include <x86intrin.h>
#endif"
    _content "${_content}")

if(NOT "${_content}" STREQUAL "${_original}")
    file(WRITE "${_file}" "${_content}")
    message(STATUS "  Patched: ${_file} (added MSVC SSE/SSE2 macro definitions)")
else()
    message(STATUS "  Already patched or pattern not found: ${_file}")
endif()

message(STATUS "libint2 MSVC SSE macro patching complete.")

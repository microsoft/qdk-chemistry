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

# Patch array_adaptor.h: ext_stack_allocator is missing the template rebind
# constructor required by MSVC Debug CRT (_ITERATOR_DEBUG_LEVEL=2).
# The vector destructor rebinds allocator<T,N> to allocator<_ContainerProxy,N>,
# which requires a converting constructor that the class lacks (C2440 error).
set(_alloc_file "libint-2.9.0/include/libint2/util/array_adaptor.h")
if(NOT EXISTS "${_alloc_file}")
    message(WARNING "libint2-msvc-sse-macros: ${_alloc_file} not found, skipping")
    return()
endif()

file(READ "${_alloc_file}" _alloc_content)
set(_alloc_original "${_alloc_content}")

string(REPLACE
    "  template <class _Up>
  struct rebind {"
    "  // MSVC Debug CRT rebinds allocator<T> to allocator<_ContainerProxy> for iterator tracking.
  template <typename U>
  ext_stack_allocator(const ext_stack_allocator<U, N>&) noexcept
      : stack_(nullptr), free_(nullptr) {}

  template <class _Up>
  struct rebind {"
    _alloc_content "${_alloc_content}")

# Also fix allocate() and pointer_on_stack() to gracefully handle stack_==nullptr
# (which happens when the allocator was constructed via the rebind constructor above).
string(REPLACE
    "  T* allocate(std::size_t n) {
    assert(stack_ != nullptr && \"array_view_allocator not initialized\");
    if (stack_ + N - free_ >="
    "  T* allocate(std::size_t n) {
    if (stack_ != nullptr && stack_ + N - free_ >="
    _alloc_content "${_alloc_content}")

string(REPLACE
    "  bool pointer_on_stack(T* ptr) const {
    return stack_ <= ptr && ptr < stack_ + N;"
    "  bool pointer_on_stack(T* ptr) const {
    return stack_ != nullptr && stack_ <= ptr && ptr < stack_ + N;"
    _alloc_content "${_alloc_content}")

if(NOT "${_alloc_content}" STREQUAL "${_alloc_original}")
    file(WRITE "${_alloc_file}" "${_alloc_content}")
    message(STATUS "  Patched: ${_alloc_file} (added MSVC rebind constructor to ext_stack_allocator)")
else()
    message(STATUS "  Already patched or pattern not found: ${_alloc_file}")
endif()

message(STATUS "libint2 MSVC allocator patching complete.")

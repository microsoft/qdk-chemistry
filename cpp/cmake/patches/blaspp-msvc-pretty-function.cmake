# blaspp-msvc-pretty-function.cmake
# Patch script to replace __PRETTY_FUNCTION__ (GCC/Clang) with __FUNCSIG__
# (MSVC) in BLAS++ sources.
# Applied via PATCH_COMMAND in FetchContent.
#
# Usage: cmake -P blaspp-msvc-pretty-function.cmake  (run from blaspp source root)

message(STATUS "Patching blaspp for MSVC __PRETTY_FUNCTION__ compatibility...")

# --- include/blas/symv.hh ---
file(READ "include/blas/symv.hh" _content)
string(REPLACE "__PRETTY_FUNCTION__" "__FUNCSIG__" _content "${_content}")
file(WRITE "include/blas/symv.hh" "${_content}")
message(STATUS "  Patched: include/blas/symv.hh")

# --- examples/util.hh ---
# This file defines a macro using __PRETTY_FUNCTION__
file(READ "examples/util.hh" _content)
set(_old "#define print_func() print_func_( __PRETTY_FUNCTION__ )")
set(_new [=[
#ifdef _MSC_VER
    #define print_func() print_func_( __FUNCSIG__ )
#else
    #define print_func() print_func_( __PRETTY_FUNCTION__ )
#endif
]=])
string(REPLACE "${_old}" "${_new}" _content "${_content}")
file(WRITE "examples/util.hh" "${_content}")
message(STATUS "  Patched: examples/util.hh")

message(STATUS "blaspp patching complete.")

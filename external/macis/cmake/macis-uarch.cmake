# MACIS_UARCH names the target microarchitecture. MACIS_UARCH_USED records the
# validated name (NONE when disabled or unsupported); the installed package
# exports it. MACIS_UARCH_FLAGS holds the derived option for MACIS's own
# targets as a directory-scope variable, never as a cache entry.
unset(MACIS_UARCH_FLAGS CACHE)
set(MACIS_UARCH_FLAGS "")
set(MACIS_UARCH_USED "NONE" CACHE STRING "Validated target microarchitecture" FORCE)

if(NOT DEFINED MACIS_UARCH)
  message(WARNING "MACIS_UARCH not defined. This may degrade performance")
  return()
endif()
if(MACIS_UARCH STREQUAL "" OR MACIS_UARCH STREQUAL "NONE")
  message(STATUS "MACIS microarchitecture optimization disabled")
  return()
endif()
message(STATUS "Using user-defined uarch: ${MACIS_UARCH}")

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(_macis_uarch_flag "-march=${MACIS_UARCH}")
elseif(MSVC)
  # Users should set MACIS_UARCH to a valid MSVC /arch: argument (e.g. AVX2, AVX512)
  set(_macis_uarch_flag "/arch:${MACIS_UARCH}")
else()
  message(WARNING "Compiler ${CMAKE_CXX_COMPILER_ID}: unknown option syntax for MACIS_UARCH. Microarchitecture optimization disabled.")
  return()
endif()

message(STATUS "Testing MACIS_UARCH: ${MACIS_UARCH}")
include(CheckCXXCompilerFlag)
unset(COMPILER_SUPPORTS_MACIS_UARCH CACHE)
check_cxx_compiler_flag("${_macis_uarch_flag}" COMPILER_SUPPORTS_MACIS_UARCH)
if(COMPILER_SUPPORTS_MACIS_UARCH)
  set(MACIS_UARCH_USED "${MACIS_UARCH}" CACHE STRING "Validated target microarchitecture" FORCE)
  set(MACIS_UARCH_FLAGS "${_macis_uarch_flag}")
else()
  message(WARNING "Compiler ${CMAKE_CXX_COMPILER_ID} does not support MACIS_UARCH=${MACIS_UARCH}. Microarchitecture optimization disabled.")
endif()
unset(_macis_uarch_flag)

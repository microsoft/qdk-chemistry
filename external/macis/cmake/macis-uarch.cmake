# --- Step 1: Resolve MACIS_UARCH ---
# Check environment variable first, then CMake variable
if(NOT DEFINED MACIS_UARCH AND DEFINED ENV{MACIS_UARCH})
  set(MACIS_UARCH $ENV{MACIS_UARCH})
endif()

if(DEFINED MACIS_UARCH)
  message(STATUS "Using user-defined uarch: ${MACIS_UARCH}")
else()
  # CMake regexes are case sensitive, and the spelling of the processor name is
  # platform specific: Windows reports AMD64 and ARM64, Linux x86_64 and
  # aarch64. Normalise before matching so every spelling is recognised.
  string(TOLOWER "${CMAKE_SYSTEM_PROCESSOR}" _macis_system_processor)

  # Auto-detect based on the target platform
  if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    # Native MSVC cl has no generic baseline /arch: flag; leave MACIS_UARCH unset.
    message(STATUS "Native MSVC: use compiler default ISA.")
    return()
  elseif(_macis_system_processor MATCHES "x86_64|amd64")
    set(MACIS_UARCH "x86-64" CACHE STRING "Target microarchitecture")
    message(STATUS "Auto-detected x86_64 architecture, using: ${MACIS_UARCH}")
  elseif(_macis_system_processor MATCHES "aarch64|arm64")
    set(MACIS_UARCH "armv8-a" CACHE STRING "Target microarchitecture")
    message(STATUS "Auto-detected ARM64 architecture, using: ${MACIS_UARCH}")
  else()
    message(WARNING "Unknown architecture ${CMAKE_SYSTEM_PROCESSOR}. MACIS_UARCH not set. This may degrade performance")
    return()
  endif()
endif()

# --- Step 2: Set MACIS_UARCH_FLAGS based on compiler and uarch ---
set(MACIS_UARCH_USED ${MACIS_UARCH} CACHE STRING "Target microarchitecture for optimization")
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
  set(MACIS_UARCH_FLAGS "-march=${MACIS_UARCH}" CACHE STRING "Compiler flags for target microarchitecture")
elseif(MSVC)
  # Users should set MACIS_UARCH to a valid MSVC /arch: argument (e.g. AVX2, AVX512)
  set(MACIS_UARCH_FLAGS "/arch:${MACIS_UARCH}" CACHE STRING "Compiler flags for target microarchitecture")
else()
  message(WARNING "Compiler ${CMAKE_CXX_COMPILER_ID}: unknown flag syntax for ISA selection. MACIS_UARCH_FLAGS will not be set.")
  set(MACIS_UARCH_USED "NONE" CACHE STRING "Target microarchitecture for optimization")
endif()

# --- Step 3: Validate the flags ---
if(MACIS_UARCH_FLAGS)
  message(STATUS "Testing MACIS_UARCH_FLAGS: ${MACIS_UARCH_FLAGS}")
  include(CheckCXXCompilerFlag)
  check_cxx_compiler_flag("${MACIS_UARCH_FLAGS}" COMPILER_SUPPORTS_MACIS_UARCH_FLAGS)
  if(NOT COMPILER_SUPPORTS_MACIS_UARCH_FLAGS)
    message(WARNING "The compiler does not support the specified MACIS_UARCH_FLAGS: ${MACIS_UARCH_FLAGS}. Unsetting these flags.")
    unset(MACIS_UARCH_FLAGS CACHE)
    set(MACIS_UARCH_USED "NONE" CACHE STRING "Target microarchitecture for optimization")
  endif()
endif()

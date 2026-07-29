# TARGET: The target architecture
#
# Originally, OpenBLAS tries to optimize for the host CPU unless
# - being given an explicit TARGET, and
# - CMAKE_CROSSCOMPILING, and
# - not building for uwp (aka WINDOWSSTORE)
# For this optimization, it runs 'getarch' and 'getarch_2nd' which it builds
# from source. The getarch executables are not built when not optimizing.
#
# Consequences:
# - The port must ensure that TARGET is set when cross compiling for a different CPU or OS.
# - The port must install getarch executables when possible.
#
# DYNAMIC_ARCH enables support "for multiple targets with runtime detection".
# (But not for MSVC, https://github.com/OpenMathLib/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio#cmake-and-visual-studio.)
# The OpenBLAS README.md suggests that this shall be used with TARGET being
# set "to the oldest model you expect to encounter". This affects "all the
# common code in the library".

set(need_target 0)
if(NOT "${TARGET}" STREQUAL "")
    message(STATUS "TARGET: ${TARGET} (user-defined)")
elseif(DYNAMIC_ARCH)
    message(STATUS "DYNAMIC_ARCH: ${DYNAMIC_ARCH}")
    set(need_target 1) # for C
elseif(CMAKE_CROSSCOMPILING AND NOT GETARCH_BINARY_DIR)
    set(need_target 1) # for C and for optimized kernel
elseif(CMAKE_C_COMPILER_ID STREQUAL "MSVC" AND
       (VCPKG_TARGET_ARCHITECTURE MATCHES "^arm" OR
        CMAKE_SYSTEM_PROCESSOR MATCHES "^(ARM64|arm64|aarch64|ARM|arm)$"))
    # QDK overlay change: on a native ARM64 build getarch selects ARMV8, whose
    # kernels are GNU-syntax .S files that MSVC's armasm64 cannot assemble.
    # Force target selection so the MSVC branch below picks GENERIC (portable C
    # kernels). clang-cl understands the GNU dialect and keeps the fast path.
    message(STATUS "MSVC cannot assemble the ARM64 kernels; forcing a portable target")
    set(need_target 1)
else()
    message(STATUS "TARGET: <native> (OpenBLAS getarch/getarch_2nd)")
endif()

if(need_target)
    set(target_default "GENERIC")
    if(MSVC)
        # "does not support the dialect of assembly used in the cpu-specific optimized files"
        # https://github.com/OpenMathLib/OpenBLAS/wiki/How-to-use-OpenBLAS-in-Microsoft-Visual-Studio#cmake-and-visual-studio
    elseif(VCPKG_TARGET_ARCHITECTURE MATCHES "^x64|^x86")
        set(target_default "ATOM")
    elseif(VCPKG_TARGET_ARCHITECTURE MATCHES "^arm64")
        set(target_default "ARMV8")
    elseif(VCPKG_TARGET_ARCHITECTURE MATCHES "^arm")
        set(target_default "ARMV7")
    endif()
    set(TARGET "${target_default}" CACHE STRING "")
    message(STATUS "TARGET: ${TARGET}")
endif()

# NUM_THREADS: The number of threads expected to be used.
#
# This setting affects both the configuration with USE_THREAD enabled
# (multithreaded OpenBLAS) and disabled (multithreaded access to OpenBLAS).
# This shouldn't be set too low for generic packages. But it comes with a
# memory footprint.

if(DEFINED NUM_THREADS)
    message(STATUS "NUM_THREADS: ${NUM_THREADS} (user-defined)")
elseif(EMSCRIPTEN)
    message(STATUS "NUM_THREADS: <default> (for EMSCRIPTEN)")
elseif(need_target)
    set(num_threads_default 24)
    if(ANDROID OR IOS)
        set(num_threads_default 8)
    endif()
    set(NUM_THREADS "${num_threads_default}" CACHE STRING "")
    message(STATUS "NUM_THREADS: ${NUM_THREADS}")
endif()

cmake_minimum_required(VERSION 3.21)

# Standalone project for building and installing the external C++ dependencies
# (libint2, ecpint, gauxc, nlohmann_json) to a path-independent install prefix
# that can be safely cached between CI runs. macis is intentionally excluded —
# it is always built from source in the main qdk build.

project(qdk-chemistry-deps LANGUAGES C CXX)

# C++20 globally ensures MSVC passes /std:c++20 which implies /permissive-
# (required for C++ alternative tokens used by GauXC).
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ─── Options (mirror the relevant subset from the main project) ───────────────

set(QDK_UARCH "" CACHE STRING "Target microarchitecture (e.g. x86-64-v3)")
option(QDK_ALLOW_DEPENDENCY_FETCH  "Allow FetchContent fetching of dependencies" ON)
option(QDK_CHEMISTRY_ENABLE_GPU    "Enable GPU support"                          OFF)
option(QDK_CHEMISTRY_ENABLE_MPI    "Enable MPI support"                          OFF)

# OpenMP is not supported with MSVC/clang-cl on Windows; matches main project logic.
include(CMakeDependentOption)
cmake_dependent_option(QDK_ENABLE_OPENMP
    "Enable OpenMP support" ON
    "NOT (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL \"AppleClang\");NOT WIN32" OFF)

# ─── Module path — point to the main project's cmake/modules ─────────────────

set(_QDK_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../..")
list(APPEND CMAKE_MODULE_PATH "${_QDK_ROOT}/cpp/cmake/modules")

set(QDK_CMAKE_PATCHES_DIR "${_QDK_ROOT}/cpp/cmake/patches" CACHE INTERNAL "")

# ─── linalg-cmake-modules ─────────────────────────────────────────────────────
# Provides FindBLAS/FindLAPACK modules used transitively by gauxc/GauXC.

include(FetchContent)
FetchContent_Declare(linalg-cmake-modules
    GIT_REPOSITORY https://github.com/wavefunction91/linalg-cmake-modules.git
    GIT_TAG        main
)
FetchContent_MakeAvailable(linalg-cmake-modules)
list(APPEND CMAKE_MODULE_PATH ${linalg-cmake-modules_SOURCE_DIR})

# ─── Microarchitecture flags ───────────────────────────────────────────────────
# Sets QDK_UARCH_FLAGS, which third_party.cmake passes as BUILD_ARGS to each dep.

include("${_QDK_ROOT}/cpp/cmake/qdk-uarch.cmake")

# ─── External FetchContent dependencies ───────────────────────────────────────
# Handles: nlohmann_json, libint2, ecpint, gauxc.
# macis is intentionally NOT declared here.

include("${_QDK_ROOT}/cpp/cmake/third_party.cmake")

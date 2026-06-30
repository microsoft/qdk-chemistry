cmake_minimum_required(VERSION 3.21)

# Standalone project for building and installing the external C++ dependencies
# of qdk-chemistry (libint2, ecpint, gauxc, nlohmann_json) to a path-independent
# prefix that can be safely cached between CI runs.
#
# Why a standalone project?
#   - Avoids any modifications to the main cpp/CMakeLists.txt.
#   - macis is intentionally excluded: it is internal code and is always built
#     from the local external/macis source directory in the main qdk build.
#   - Installed CMake package config files use ${_IMPORT_PREFIX} (relative to the
#     config file's location), so the install prefix is fully path-independent and
#     works correctly when restored on a runner with a different drive or workspace.
#
# Typical invocation (mirroring the main project's settings):
#
#   cmake -S <repo>/.pipelines/cmake/deps-install -B <buildDir> -GNinja \
#     -DQDK_UARCH=x86-64-v3                                               \
#     -DCMAKE_BUILD_TYPE=Release                                           \
#     -DCMAKE_INSTALL_PREFIX=<repo>/deps-install-msvc                     \
#     -DCMAKE_TOOLCHAIN_FILE=<vcpkg>/scripts/buildsystems/vcpkg.cmake     \
#     -DVCPKG_TARGET_TRIPLET=x64-windows-static-md                        \
#     ...
#   cmake --build <buildDir> --target libint2_obj ecpint gauxc
#   cmake --install <buildDir>
#
# After this, the main qdk build finds all deps via:
#   -DCMAKE_PREFIX_PATH="<repo>/deps-install-msvc;<vcpkg_installed/triplet>"
#   -DQDK_ALLOW_DEPENDENCY_FETCH=OFF

project(qdk-chemistry-deps LANGUAGES C CXX)

# Mirror the main project: C++20 globally ensures MSVC passes /std:c++20 which
# implies /permissive- (required for C++ alternative tokens used by GauXC).
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

# Patch scripts in cpp/cmake/patches/ are referenced via CMAKE_CURRENT_LIST_DIR
# inside third_party.cmake, which resolves correctly because CMAKE_CURRENT_LIST_DIR
# is set to the directory of the included file (cpp/cmake/), not this file.
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

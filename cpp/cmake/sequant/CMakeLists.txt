# The dependency installers' SeQuant build; BLAS++/LAPACK++ must already be installed.
cmake_minimum_required(VERSION 3.28)
project(qdk_sequant LANGUAGES C CXX)

set(QDK_ALLOW_DEPENDENCY_FETCH ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../modules")
include(DependencyManager)

# As in ../../CMakeLists.txt; kit-cmake's older pin misses vcpkg's OpenBLAS headers.
include(FetchContent)
FetchContent_Declare(linalg-cmake-modules
    GIT_REPOSITORY https://github.com/wavefunction91/linalg-cmake-modules.git
    GIT_TAG        main
)

include("${CMAKE_CURRENT_LIST_DIR}/../sequant.cmake")

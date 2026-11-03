# The dependency installers' SeQuant build; BLAS++/LAPACK++ must already be installed.
cmake_minimum_required(VERSION 3.28)
project(qdk_sequant LANGUAGES C CXX)

set(QDK_ALLOW_DEPENDENCY_FETCH ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/../modules")
include(DependencyManager)
include("${CMAKE_CURRENT_LIST_DIR}/../sequant.cmake")

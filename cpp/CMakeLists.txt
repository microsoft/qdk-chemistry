cmake_minimum_required(VERSION 3.15)
project(qdk VERSION 1.0.0.1 LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "CMAKE_BUILD_TYPE not specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add compiler flags
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-g -O0 -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -DNDEBUG")

# Options
option(QDK_CHEMISTRY_ENABLE_COVERAGE    "Enable coverage build" OFF)
option(QDK_CHEMISTRY_ENABLE_GPU         "Enable GPU acceleration in QDK" OFF)
option(QDK_ALLOW_DEPENDENCY_FETCH "Allow fetching of dependencies"     ON)
option(QDK_CHEMISTRY_ENABLE_MPI "Enable MPI Bindings for QDK Chemistry"  OFF)
option(QDK_EMBED_RESOURCE_LOCATION "Point to embedded (permanent) resource location" ON)
option(QDK_ENABLE_OPENMP "Enable OpenMP support" ON)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Custom CMake Find* modules for linear algebra dependencies
include( FetchContent )
FetchContent_Declare( linalg-cmake-modules
  GIT_REPOSITORY https://github.com/wavefunction91/linalg-cmake-modules.git
  GIT_TAG        main
)
FetchContent_GetProperties( linalg-cmake-modules )
if( NOT linalg-cmake-modules_POPULATED )
  FetchContent_Populate( linalg-cmake-modules )
endif()

list( APPEND CMAKE_MODULE_PATH ${linalg-cmake-modules_SOURCE_DIR} )

# Microarchitecture handling
include(cmake/qdk-uarch.cmake)

# Handle TPL dependencies
include(cmake/third_party.cmake)

# Handle GPU dependencies
if(QDK_CHEMISTRY_ENABLE_GPU)
  include(${CMAKE_CURRENT_LIST_DIR}/cmake/cuda.cmake)
endif()

find_package(macis QUIET)
if(NOT macis_FOUND)
    include(FetchContent)
    if(NOT DEFINED FETCHCONTENT_SOURCE_DIR_MACIS)
        set(FETCHCONTENT_SOURCE_DIR_MACIS "${PROJECT_SOURCE_DIR}/../external/macis" CACHE PATH "Path to MACIS" FORCE)
    endif()
    set(MACIS_ENABLE_MPI    OFF CACHE BOOL "MACIS enable MPI" FORCE)
    set(MACIS_ENABLE_PYTHON OFF CACHE BOOL "MACIS Enable Python" FORCE)
    set(MACIS_ENABLE_EXAMPLES OFF CACHE BOOL "MACIS Build Examples" FORCE)
    set(MACIS_ENABLE_OPENMP ${QDK_ENABLE_OPENMP} CACHE BOOL "MACIS Enable OpenMP" FORCE)
    if(DEFINED QDK_UARCH_USED)
      set(MACIS_UARCH  ${QDK_UARCH_USED} CACHE STRING "MACIS Microarchitecture" FORCE)
    endif()
    FetchContent_Declare(macis GIT_REPOSITORY https://github.com/wavefunction91/MACIS)
    FetchContent_MakeAvailable(macis)
else()
    # Check that MACIS UARCH is consistent with QDK_UARCH
    if(
        (DEFINED QDK_UARCH_USED AND NOT DEFINED MACIS_UARCH) OR
        (NOT DEFINED QDK_UARCH_USED AND DEFINED MACIS_UARCH) OR
        (NOT "${QDK_UARCH_USED}" STREQUAL "${MACIS_UARCH}")
    )
        message(STATUS "MACIS_UARCH = ${MACIS_UARCH}")
        message(STATUS "QDK_UARCH  = ${QDK_UARCH_USED}")
        message(FATAL_ERROR "Inconsistent microarchitecture settings between QDK and MACIS")
    endif()
endif()

# Enable CUDA for GPU bindings in QDK Chemistry
if(QDK_CHEMISTRY_ENABLE_GPU)
    enable_language(CUDA)
endif()

# Create library
add_library(chemistry)
add_subdirectory(src/qdk/chemistry/data)
add_subdirectory(src/qdk/chemistry/algorithms)
add_subdirectory(src/qdk/chemistry/utils)

target_include_directories(chemistry PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_link_libraries(chemistry PUBLIC
    Eigen3::Eigen
    ${HDF5_CXX_LIBRARIES}
    nlohmann_json::nlohmann_json
    macis::macis
)

# Conditionally link OpenMP
if(QDK_ENABLE_OPENMP)
    if(TARGET OpenMP::OpenMP_CXX)
        target_link_libraries(chemistry PUBLIC OpenMP::OpenMP_CXX)
        message(STATUS "OpenMP support enabled")
    else()
        message(WARNING "QDK_ENABLE_OPENMP is ON but OpenMP was not found")
    endif()
else()
    message(STATUS "OpenMP support disabled")
endif()

target_include_directories(chemistry PUBLIC ${HDF5_INCLUDE_DIRS})
target_compile_definitions(chemistry PUBLIC ${HDF5_DEFINITIONS})
target_compile_features(chemistry PUBLIC cxx_std_20)
set_property(TARGET chemistry PROPERTY POSITION_INDEPENDENT_CODE ON)
if(DEFINED QDK_UARCH_FLAGS)
    target_compile_options(chemistry PUBLIC ${QDK_UARCH_FLAGS})
endif()

# Alias target for subprojected use case
add_library(qdk::chemistry ALIAS chemistry)

# Set output name
set_target_properties(chemistry PROPERTIES OUTPUT_NAME "qdk_chemistry")

# Handle Coverage build
if(QDK_CHEMISTRY_ENABLE_COVERAGE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        message(STATUS "Enabling coverage build")
        target_compile_options(chemistry PRIVATE --coverage -fprofile-arcs -ftest-coverage)
        target_link_libraries(chemistry PRIVATE --coverage)
    else()
        message(FATAL_ERROR "Coverage build is only supported in CMAKE_BUILD_TYPE=Debug or RelWithDebInfo mode")
    endif()
endif()

# Ensure the library is compiled with position-independent code
set_target_properties(chemistry PROPERTIES POSITION_INDEPENDENT_CODE ON)

# Determine export variables
set(QDK_DEP_EXPORTS "" CACHE STRING "QDK Chemistry Exported Dependencies" FORCE)

# Include Addons
if(QDK_ADDONS_DIR)
  add_subdirectory(${QDK_ADDONS_DIR} addons)
endif()

get_target_property(_imported nlohmann_json::nlohmann_json IMPORTED)
get_target_property(_aliased nlohmann_json::nlohmann_json ALIASED_TARGET)
if(_aliased AND NOT _imported)
  list(APPEND QDK_DEP_EXPORTS ${_aliased})
endif()

get_target_property(_imported Libint2::cxx IMPORTED)
get_target_property(_aliased Libint2::cxx ALIASED_TARGET)
if(_aliased AND NOT _imported)
  list(APPEND QDK_DEP_EXPORTS ${_aliased})
endif()

get_target_property(_imported Libint2::int2 IMPORTED)
get_target_property(_aliased Libint2::int2 ALIASED_TARGET)
if(_aliased AND NOT _imported)
  list(APPEND QDK_DEP_EXPORTS ${_aliased})
endif()

if(TARGET libint2_Eigen)
  get_target_property(_imported libint2_Eigen IMPORTED)
  if(NOT _imported)
    list(APPEND QDK_DEP_EXPORTS libint2_Eigen)
  endif()
endif()


# Install targets
install(TARGETS chemistry ${QDK_DEP_EXPORTS}
    EXPORT chemistryTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Generate config file
configure_file(
  ${PROJECT_SOURCE_DIR}/include/qdk/chemistry/config.hpp.in
  ${PROJECT_BINARY_DIR}/include/qdk/chemistry/config.hpp
)

# Install static headers
install(
  DIRECTORY   ${PROJECT_SOURCE_DIR}/include
  DESTINATION .
  FILES_MATCHING PATTERN "*.hpp"
)

# Install generated headers
install(
  FILES       ${PROJECT_BINARY_DIR}/include/qdk/chemistry/config.hpp
  DESTINATION include/qdk/chemistry
)

# Install export
install(EXPORT chemistryTargets
    FILE chemistryTargets.cmake
    NAMESPACE qdk::
    DESTINATION lib/cmake/qdk
)

# Create config files
include(CMakePackageConfigHelpers)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/qdkConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/qdkConfig.cmake"
    INSTALL_DESTINATION lib/cmake/qdk
)

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/qdkConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/qdkConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/qdkConfigVersion.cmake"
    DESTINATION lib/cmake/qdk
)

# Export targets for build tree
export(EXPORT chemistryTargets
    FILE "${CMAKE_CURRENT_BINARY_DIR}/chemistryTargets.cmake"
    NAMESPACE qdk::
)

# Testing
include(CTest) # BUILD_TESTING variable
if(BUILD_TESTING)
    add_subdirectory(tests)
endif()

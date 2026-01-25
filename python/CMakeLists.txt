cmake_minimum_required(VERSION 3.15)
project(qdk_chemistry_python VERSION 1.0.0 LANGUAGES CXX)

# Set default build type to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(WARNING "CMAKE_BUILD_TYPE not specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Enable ccache if available
find_program(CCACHE_EXECUTABLE ccache)
if(CCACHE_EXECUTABLE AND NOT CMAKE_CXX_COMPILER_LAUNCHER)
   set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_EXECUTABLE})
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable PIC for all targets" FORCE)
set(CMAKE_MACOSX_RPATH ON CACHE BOOL "Enable RPATH on macOS" FORCE)

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

# list( APPEND CMAKE_MODULE_PATH ${linalg-cmake-modules_SOURCE_DIR} )
# First try to find an installed qdk library
find_package(qdk QUIET COMPONENTS chemistry)

if(NOT qdk_FOUND)
    message(STATUS "qdk not found in system, building from source...")

    # Build the C++ library from source
    set(QDK_CPP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../cpp")

    if(NOT EXISTS "${QDK_CPP_DIR}/CMakeLists.txt")
        message(FATAL_ERROR "Could not find qdk source at ${QDK_CPP_DIR}")
    endif()

    # Setup C++ subbuild - pass through important flags
    set(QDK_EMBED_RESOURCE_LOCATION OFF CACHE BOOL "Disable embedded resource location for Python builds" FORCE)
    set(BUILD_TESTING OFF CACHE BOOL "Disable building tests for Python builds" FORCE)
    set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build dependencies as static libraries for Python builds" FORCE)

    # Add the cpp directory - this will create the qdk target
    add_subdirectory("${QDK_CPP_DIR}" qdk_build)
else()
    message(STATUS "Using installed qdk library")
endif()

if(QDK_CHEMISTRY_ENABLE_GPU)
    message(STATUS "Building Python bindings with GPU support")
    enable_language(CUDA)
else()
    message(STATUS "Building Python bindings without GPU support")
endif()

# Find Python first (required for pybind11)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)

# Find pybind11 - tell it to use the FindPython we already called
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Create the Python extension module
pybind11_add_module(_core
    src/pybind11/data/data_class.cpp
    src/pybind11/data/element_data.cpp
    src/pybind11/data/orbitals.cpp
    src/pybind11/data/wavefunction.cpp
    src/pybind11/data/hamiltonian.cpp
    src/pybind11/data/settings.cpp
    src/pybind11/data/structure.cpp
    src/pybind11/data/basis_set.cpp
    src/pybind11/data/configuration.cpp
    src/pybind11/data/configuration_set.cpp
    src/pybind11/data/ansatz.cpp
    src/pybind11/data/stability_result.cpp
    src/pybind11/data/pauli_operator.cpp
    src/pybind11/data/serialization.cpp
    src/pybind11/algorithms/localizer.cpp
    src/pybind11/algorithms/stability.cpp
    src/pybind11/algorithms/mc.cpp
    src/pybind11/algorithms/pmc.cpp
    src/pybind11/algorithms/mcscf.cpp
    src/pybind11/algorithms/hamiltonian.cpp
    src/pybind11/algorithms/scf.cpp
    src/pybind11/algorithms/active_space.cpp
    src/pybind11/algorithms/dynamical_correlation_calculator.cpp
    src/pybind11/algorithms/davidson_solver.cpp
    src/pybind11/algorithms/syev_solver.cpp
    src/pybind11/utils/logger.cpp
    src/pybind11/utils/valence_space.cpp
    src/pybind11/utils/orbital_rotation.cpp
    src/pybind11/constants.cpp
    src/pybind11/qdk_scf_config.cpp
    src/pybind11/module.cpp
)

# Link against the qdk::chemistry library
target_link_libraries(_core PRIVATE qdk::chemistry)

# Add cpp/src to include path for accessing private implementation headers
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src")

# C++20
target_compile_features(_core PUBLIC cxx_std_20)

# Handle Coverage build
if(QDK_CHEMISTRY_ENABLE_COVERAGE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        message(STATUS "Enabling coverage build")
        target_compile_options(_core PRIVATE --coverage -fprofile-arcs -ftest-coverage)
        target_link_libraries(_core PRIVATE --coverage)
    else()
        message(FATAL_ERROR "Coverage build is only supported in CMAKE_BUILD_TYPE=Debug or RelWithDebInfo mode")
    endif()
endif()

include(FindPackageHandleStandardArgs)
function(find_python_module module)
  string(TOUPPER ${module} module_upper)
  if(NOT PY_${module_upper})
    if(ARGC GREATER 1 AND ARGV1 STREQUAL "REQUIRED")
      set(${module}_FIND_REQUIRED TRUE)
    endif()
    # A module's location is usually a directory, but for binary modules
    # it's a .so file.
    execute_process(COMMAND "${Python_EXECUTABLE}" "-c"
      "import re, ${module}; print(re.compile('/__init__.py.*').sub('',${module}.__file__))"
      RESULT_VARIABLE _${module}_status
      OUTPUT_VARIABLE _${module}_location
      ERROR_QUIET
      OUTPUT_STRIP_TRAILING_WHITESPACE)
    if(NOT _${module}_status)
      set(PY_${module_upper} ${_${module}_location} CACHE STRING
        "Location of Python module ${module}")
    endif()
  endif()
  find_package_handle_standard_args(PY_${module} DEFAULT_MSG PY_${module_upper})
endfunction()

# Set module properties
target_compile_definitions(_core PRIVATE VERSION_INFO=${PROJECT_VERSION})
set_target_properties(_core PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN YES
    BUILD_RPATH "$ORIGIN:$ORIGIN/.."
    INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
    BUILD_RPATH_USE_ORIGIN ON
)

# Install the Python extension module
install(TARGETS _core
  LIBRARY DESTINATION .
  COMPONENT _core
)

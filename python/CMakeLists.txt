cmake_minimum_required(VERSION 3.26)

# Read version from VERSION file at repo root
set(_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION")
file(READ "${_VERSION_FILE}" QDK_VERSION)
string(STRIP "${QDK_VERSION}" QDK_VERSION)

# Strip prerelease/dev suffixes for development PyPI release (e.g. .dev0, .rc1)
# Keeps only the leading numeric core: X, X.Y, or X.Y.Z
set(QDK_VERSION_RAW "${QDK_VERSION}")
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)?(\\.[0-9]+)?" QDK_VERSION "${QDK_VERSION_RAW}")

if(NOT QDK_VERSION)
    message(FATAL_ERROR "Invalid VERSION for CMake project: '${QDK_VERSION_RAW}'. ")
else()
    message(STATUS "Project version: ${QDK_VERSION}")
endif()

# Compute full version: append +local for from-source builds not on a release tag.
set(QDK_FULL_VERSION "${QDK_VERSION_RAW}")
if(QDK_VERSION_RAW MATCHES "^[0-9]+\\.[0-9]+\\.[0-9]+$")
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/../.git")
        find_package(Git QUIET)
        if(GIT_FOUND)
            execute_process(
                COMMAND ${GIT_EXECUTABLE} describe --tags --exact-match HEAD
                WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/.."
                OUTPUT_VARIABLE _GIT_TAG
                OUTPUT_STRIP_TRAILING_WHITESPACE
                ERROR_QUIET
                RESULT_VARIABLE _GIT_TAG_RESULT)
            if(_GIT_TAG_RESULT EQUAL 0
               AND (_GIT_TAG STREQUAL "${QDK_VERSION_RAW}" OR _GIT_TAG STREQUAL "v${QDK_VERSION_RAW}"))
                # On a matching release tag — keep version as-is
            else()
                string(APPEND QDK_FULL_VERSION "+local")
            endif()
        else()
            # .git exists but git not found — cannot verify tag, treat as local
            string(APPEND QDK_FULL_VERSION "+local")
        endif()
    endif()
endif()
message(STATUS "Full version: ${QDK_FULL_VERSION}")

# Re-run CMake configure when the VERSION file changes
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_VERSION_FILE}")

project(qdk_chemistry_python VERSION ${QDK_VERSION} LANGUAGES CXX)

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
    # When building against a pre-installed qdk the C++ CMakeLists.txt is not
    # included, so the qdk_chemistry_resources install rule never fires. Derive
    # the install root from qdk_DIR (which is <prefix>/lib/cmake/qdk) and
    # re-register the install rule here so scikit-build-core picks it up.
    get_filename_component(_qdk_root "${qdk_DIR}" PATH)  # .../lib/cmake
    get_filename_component(_qdk_root "${_qdk_root}" PATH) # .../lib
    get_filename_component(_qdk_root "${_qdk_root}" PATH) # install root
    set(_qdk_resources "${_qdk_root}/share/qdk/chemistry/scf/resources")
    if(EXISTS "${_qdk_resources}")
        install(
            DIRECTORY "${_qdk_resources}"
            DESTINATION "share/qdk/chemistry/scf"
            COMPONENT qdk_chemistry_resources
        )
    else()
        message(FATAL_ERROR "QDK resources not found at '${_qdk_resources}' — wheel will be missing share/qdk/chemistry/scf/resources")
    endif()
endif()

if(QDK_CHEMISTRY_ENABLE_GPU)
    message(STATUS "Building Python bindings with GPU support")
    enable_language(CUDA)
else()
    message(STATUS "Building Python bindings without GPU support")
endif()

# Find Python first (required for nanobind)
find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module
             OPTIONAL_COMPONENTS Development.SABIModule)

# vcpkg's find_package wrapper may miss pip-installed nanobind.
if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE _nanobind_cmakedir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
    RESULT_VARIABLE _nanobind_result)
  if(_nanobind_result EQUAL 0 AND _nanobind_cmakedir)
    set(nanobind_DIR "${_nanobind_cmakedir}" CACHE PATH "" FORCE)
  endif()
endif()
find_package(nanobind 3.1 CONFIG REQUIRED)

# Create the Python extension module
nanobind_add_module(_core
    STABLE_ABI
    src/nanobind/data/data_class.cpp
    src/nanobind/data/symmetry/symmetry.cpp
    src/nanobind/data/symmetry/symmetry_blocked_scalar.cpp
    src/nanobind/data/symmetry/symmetry_blocked_tensor.cpp
    src/nanobind/data/symmetry/symmetry_blocked_index_set.cpp
    src/nanobind/data/symmetry/symmetry_blocked_sparse_map.cpp
    src/nanobind/data/element_data.cpp
    src/nanobind/data/orbitals.cpp
    src/nanobind/data/wavefunction.cpp
    src/nanobind/data/hamiltonian.cpp
    src/nanobind/data/settings.cpp
    src/nanobind/data/structure.cpp
    src/nanobind/data/basis_set.cpp
    src/nanobind/data/configuration.cpp
    src/nanobind/data/configuration_set.cpp
    src/nanobind/data/ansatz.cpp
    src/nanobind/data/stability_result.cpp
    src/nanobind/data/nuclear_gradients.cpp
    src/nanobind/data/nuclear_hessian.cpp
    src/nanobind/data/pauli_operator.cpp
    src/nanobind/data/majorana_mapping.cpp
    src/nanobind/data/serialization.cpp
    src/nanobind/data/lattice_geometry.cpp
    src/nanobind/data/lattice_graph.cpp
    src/nanobind/algorithms/localizer.cpp
    src/nanobind/algorithms/stability.cpp
    src/nanobind/algorithms/mc.cpp
    src/nanobind/algorithms/pmc.cpp
    src/nanobind/algorithms/mcscf.cpp
    src/nanobind/algorithms/hamiltonian.cpp
    src/nanobind/algorithms/scf.cpp
    src/nanobind/algorithms/nuclear_derivative.cpp
    src/nanobind/algorithms/population_analysis.cpp
    src/nanobind/algorithms/geometry_optimization.cpp
    src/nanobind/algorithms/active_space.cpp
    src/nanobind/algorithms/dynamical_correlation_calculator.cpp
    src/nanobind/algorithms/effective_hamiltonian.cpp
    src/nanobind/algorithms/hamiltonian_factorization.cpp
    src/nanobind/algorithms/double_factorization.cpp
    src/nanobind/algorithms/davidson_solver.cpp
    src/nanobind/algorithms/syev_solver.cpp
    src/nanobind/utils/logger.cpp
    src/nanobind/utils/valence_space.cpp
    src/nanobind/utils/orbital_rotation.cpp
    src/nanobind/utils/model_hamiltonians.cpp
    src/nanobind/utils/cube_generator.cpp
    src/nanobind/constants.cpp
    src/nanobind/qdk_scf_config.cpp
    src/nanobind/module.cpp
)

# Link against the qdk::chemistry library
target_link_libraries(_core PRIVATE qdk::chemistry)

# Add cpp/src to include path for accessing private implementation headers
target_include_directories(_core PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/../cpp/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/nanobind"
)

# C++20
target_compile_features(_core PUBLIC cxx_std_20)

# Handle Coverage build
if(QDK_CHEMISTRY_ENABLE_COVERAGE)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
        message(STATUS "Enabling coverage build")
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang" AND NOT MSVC)
            target_compile_options(_core PRIVATE --coverage -fprofile-arcs -ftest-coverage)
            target_link_libraries(_core PRIVATE --coverage)
        else()
            message(WARNING "Coverage build is only supported with GCC or Clang compilers")
        endif()
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
target_compile_definitions(_core PRIVATE VERSION_INFO="${QDK_FULL_VERSION}")
set_target_properties(_core PROPERTIES
    CXX_VISIBILITY_PRESET "hidden"
    VISIBILITY_INLINES_HIDDEN YES
)
if(NOT WIN32)
    set_target_properties(_core PROPERTIES
        BUILD_RPATH "$ORIGIN:$ORIGIN/.."
        INSTALL_RPATH "$ORIGIN:$ORIGIN/.."
        BUILD_RPATH_USE_ORIGIN ON
    )
endif()

# Install the Python extension module
install(TARGETS _core
  LIBRARY DESTINATION .
  COMPONENT _core
)

# Windows (Python 3.8+): DLL dependencies of .pyd files are only found if they
# reside in the same directory as the .pyd or are registered via os.add_dll_directory().
# PATH is NOT searched. When vcpkg provides dynamic libraries (x64-windows triplet),
# copy the runtime DLLs next to _core.pyd so the package works out of the box.
# With static triplets (x64-windows-static-md) all dependency code is linked directly
# into _core.pyd, so no bundling is needed.
if(VCPKG_LIBRARY_LINKAGE STREQUAL "static")
    set(_bundle_dlls_default OFF)
else()
    set(_bundle_dlls_default ON)
endif()
option(QDK_BUNDLE_RUNTIME_DLLS "Bundle vcpkg runtime DLLs into the install tree (Windows only)" ${_bundle_dlls_default})

if(WIN32 AND QDK_BUNDLE_RUNTIME_DLLS)
    message(STATUS "Bundling runtime DLLs into install tree")
    # -- vcpkg dependencies (openblas, hdf5, fmt, spdlog and their transitive deps) --
    if(VCPKG_INSTALLED_DIR AND VCPKG_TARGET_TRIPLET)
        set(_vcpkg_bin "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/bin")
        set(_vcpkg_runtime_dlls)
        foreach(_dll fmt hdf5 hdf5_cpp openblas spdlog szip zlib1)
            if(EXISTS "${_vcpkg_bin}/${_dll}.dll")
                list(APPEND _vcpkg_runtime_dlls "${_vcpkg_bin}/${_dll}.dll")
            endif()
        endforeach()
        if(_vcpkg_runtime_dlls)
            install(FILES ${_vcpkg_runtime_dlls}
                DESTINATION .
                COMPONENT _core
            )
        endif()
    endif()
endif()

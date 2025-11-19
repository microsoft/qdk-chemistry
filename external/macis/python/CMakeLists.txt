# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

find_package(pybind11)

if(NOT pybind11_FOUND)
    # Include FetchContent module for pybind11
    include(FetchContent)

    # Fetch pybind11
    FetchContent_Declare(
      pybind11
      GIT_REPOSITORY https://github.com/pybind/pybind11.git
      GIT_TAG        v2.11.1  # Use the latest stable version
    )

    FetchContent_MakeAvailable(pybind11)
endif()

find_package(Python REQUIRED COMPONENTS Interpreter Development)

# Add the pymacis module
pybind11_add_module(pymacis
  pymacis/pymacis.cpp
  pymacis/common.cpp
  pymacis/settings.cpp
  pymacis/active_space.cpp
  pymacis/ci.cpp
  pymacis/wavefunction_io.cpp
  pymacis/wavefunction_util.cpp
  pymacis/fcidump.cpp
  pymacis/hamiltonian.cpp
)

# Link against the MACIS library
target_link_libraries(pymacis PRIVATE macis)

# Set Python module name
set_target_properties(pymacis PROPERTIES
  OUTPUT_NAME "pymacis"
)

# Installation target for Python module
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
  OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Installing pymacis Python module to ${PYTHON_SITE_PACKAGES}")
install(TARGETS pymacis
  DESTINATION ${PYTHON_SITE_PACKAGES})

# Add tests for the Python module if testing is enabled
# if(BUILD_TESTING)
#   add_subdirectory(tests)
# endif()

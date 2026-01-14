cmake_minimum_required(VERSION 3.15)
project(qdk_chemistry_temp_release VERSION 0.0.1 LANGUAGES CXX)

set(CMAKE_POSITION_INDEPENDENT_CODE ON CACHE BOOL "Enable PIC for all targets" FORCE)

# Find Python first (required for pybind11)
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)

# Find pybind11 - tell it to use the FindPython we already called
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 REQUIRED)

# Tiny native extension module. Exists only so the wheel is a platform wheel.
pybind11_add_module(_core
		src/qdk_chemistry/_core.cpp
)

target_compile_definitions(_core PRIVATE VERSION_INFO="${PROJECT_VERSION}")
target_compile_features(_core PUBLIC cxx_std_17)

# Install into the wheel's install root. With scikit-build-core install-dir=qdk_chemistry,
# this ends up at qdk_chemistry/_core.*.so inside the wheel.
install(TARGETS _core
	LIBRARY DESTINATION .
	COMPONENT _core
)

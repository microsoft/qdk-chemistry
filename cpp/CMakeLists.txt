cmake_minimum_required(VERSION 3.15)

# Read version from VERSION file at repo root
set(_VERSION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/../VERSION")
file(READ "${_VERSION_FILE}" QDK_VERSION)
string(STRIP "${QDK_VERSION}" QDK_VERSION)

# Strip prerelease/dev suffixes for development PyPI release (e.g. .dev0, .rc1)
# Keeps only the leading numeric core: X, X.Y, or X.Y.Z
set(QDK_VERSION_RAW "${QDK_VERSION}")
string(REGEX MATCH "^[0-9]+(\\.[0-9]+)?(\\.[0-9]+)?" QDK_VERSION "${QDK_VERSION_RAW}")

if(NOT QDK_VERSION)
    message(FATAL_ERROR "Invalid VERSION file content for CMake project VERSION: '${QDK_VERSION_RAW}'. ")
else()
    message(STATUS "Project version: ${QDK_VERSION}")
endif()

# Re-run CMake configure when the VERSION file changes
# Only set at the top-level to avoid duplicate output errors when included via add_subdirectory
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_VERSION_FILE}")
endif()

project(qdk VERSION ${QDK_VERSION} LANGUAGES CXX C)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(APPLE AND NOT DEFINED CMAKE_MACOSX_RPATH)
    set(CMAKE_MACOSX_RPATH ON CACHE BOOL "Use macOS RPATH" FORCE)
endif()

include(CMakeDependentOption)
if(POLICY CMP0127)
    cmake_policy(SET CMP0127 NEW)
endif()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
    message(WARNING "CMAKE_BUILD_TYPE not specified, defaulting to Release")
    set(CMAKE_BUILD_TYPE Release)
endif()

# Set per-config compile flags.
# We append to CMAKE_CXX_FLAGS_<CONFIG> (rather than replacing) to preserve toolchain defaults like /EHsc, /MD, etc.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG   " -g -O0 -Wall -Wextra")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3 -DNDEBUG")
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    string(APPEND CMAKE_CXX_FLAGS_DEBUG " -g -O0 -Wall -Wextra")
    if(MSVC)
        # clang-cl: CMake already sets /O2 /DNDEBUG; just suppress warnings
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -Wno-all -Wno-extra")
    else()
        string(APPEND CMAKE_CXX_FLAGS_RELEASE " -O3 -DNDEBUG -Wno-all -Wno-extra")
    endif()
    # Suppress noisy clang warnings from FetchContent dependencies.
    add_compile_options(
        -Wno-c++98-compat
        -Wno-c++98-compat-extra-semi
        -Wno-c++98-compat-pedantic
        -Wno-cast-function-type-strict
        -Wno-cast-qual
        -Wno-covered-switch-default
        -Wno-defaulted-function-deleted
        -Wno-deprecated-copy-with-dtor
        -Wno-deprecated-declarations
        -Wno-disabled-macro-expansion
        -Wno-documentation
        -Wno-documentation-pedantic
        -Wno-documentation-unknown-command
        -Wno-double-promotion
        -Wno-exit-time-destructors
        -Wno-extra-semi
        -Wno-extra-semi-stmt
        -Wno-float-conversion
        -Wno-float-equal
        -Wno-format-nonliteral
        -Wno-global-constructors
        -Wno-implicit-float-conversion
        -Wno-implicit-int-float-conversion
        -Wno-inconsistent-missing-destructor-override
        -Wno-inconsistent-missing-override
        -Wno-macro-redefined
        -Wno-missing-noreturn
        -Wno-missing-prototypes
        -Wno-newline-eof
        -Wno-old-style-cast
        -Wno-reserved-identifier
        -Wno-reserved-macro-identifier
        -Wno-shorten-64-to-32
        -Wno-sign-conversion
        -Wno-sign-compare
        -Wno-suggest-override
        -Wno-suggest-destructor-override
        -Wno-switch-default
        -Wno-switch-enum
        -Wno-undefined-func-template
        -Wno-unsafe-buffer-usage
        -Wno-unsafe-buffer-usage-in-libc-call
        -Wno-unused-macros
        -Wno-unused-parameter
        -Wno-unused-variable
    )
elseif(MSVC)
    string(APPEND CMAKE_CXX_FLAGS_DEBUG   " /Zi /Od /W3")
    string(APPEND CMAKE_CXX_FLAGS_RELEASE " /O2 /W1 /DNDEBUG")
endif()

if(MSVC)
    # MSVC CRT doesn't define M_PI etc. without this.
    add_compile_definitions(_USE_MATH_DEFINES)

    # lapackpp's lartg.cc uses std::complex without #include <complex>.
    # On libstdc++ it's transitively included; on MSVC STL it isn't.
    # Use CMAKE_CXX_FLAGS (not add_compile_options) so it only applies to C++ files,
    # avoiding STL1003 errors when C files get the C++ <complex> header.
    # Force-include <complex> for C++ only — GauXC has C files that would break.
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /FI complex")

    # gau2grid_helper.c calls exit() without #include <stdlib.h>.
    # Clang-cl treats implicit function declarations as errors in C99+.
    # Native MSVC cl.exe allows them by default, so no flag needed there.
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wno-implicit-function-declaration")
    endif()

    # blaspp's BLASFinder.cmake uses try_run() to detect Fortran name mangling,
    # but on Windows the test executable can't find openblas.dll at runtime,
    # leaving blas/defines.h empty and breaking lapackpp.
    # OpenBLAS uses lowercase + trailing underscore (see OPENBLAS_FUNDERSCORE).
    add_compile_definitions(BLAS_FORTRAN_ADD_)
endif()

# Options
option(QDK_CHEMISTRY_ENABLE_COVERAGE    "Enable coverage build" OFF)
option(QDK_CHEMISTRY_ENABLE_GPU         "Enable GPU acceleration in QDK" OFF)
option(QDK_CHEMISTRY_DISABLE_TRACE_LOG  "Disable trace logging at compile time" OFF)
set(QDK_CHEMISTRY_LOG_LEVEL "2" CACHE STRING "Default log level (0=trace, 1=debug, 2=info, 3=warn, 4=error, 5=critical, 6=off)")
set_property(CACHE QDK_CHEMISTRY_LOG_LEVEL PROPERTY STRINGS "0" "1" "2" "3" "4" "5" "6")
option(QDK_ALLOW_DEPENDENCY_FETCH "Allow fetching of dependencies"     ON)
option(QDK_CHEMISTRY_ENABLE_MPI "Enable MPI Bindings for QDK Chemistry"  OFF)
option(QDK_EMBED_RESOURCE_LOCATION "Point to embedded (permanent) resource location" ON)

# Enable OpenMP by default, except on Apple platforms with AppleClang
# On Windows with clang-cl, CMake resolves OpenMP to:
#   OpenMP_CXX_FLAGS = -Xclang -fopenmp
#   OpenMP_libomp_LIBRARY = .../VC/Tools/MSVC/.../lib/x64/libomp.lib
# This uses LLVM's libomp runtime (not MSVC's vcomp*.dll). libomp.dll must
# be present at runtime alongside the built binaries. It is found in:
#   %VS_INSTALL_DIR%\VC\Tools\Llvm\x64\bin\libomp.dll
# Note: GauXC has GAUXC_ENABLE_OPENMP=OFF on Windows (see third_party.cmake)
# to avoid NaN/divergence issues in its XC integrator under libomp.
cmake_dependent_option(QDK_ENABLE_OPENMP
    "Enable OpenMP support" ON
    "NOT (APPLE AND CMAKE_CXX_COMPILER_ID STREQUAL \"AppleClang\")" OFF)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)

# Custom CMake Find* modules for linear algebra dependencies
include( FetchContent )
FetchContent_Declare( linalg-cmake-modules
  GIT_REPOSITORY https://github.com/wavefunction91/linalg-cmake-modules.git
  GIT_TAG        main
)
FetchContent_MakeAvailable( linalg-cmake-modules )

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

# Re-enable warnings for our own code that were globally suppressed for FetchContent dependencies.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    target_compile_options(chemistry PRIVATE
        -Wcast-function-type-strict
        -Wcast-qual
        # -Wcovered-switch-default
        -Wdisabled-macro-expansion
        # -Wdefaulted-function-deleted
        -Wdeprecated-copy-with-dtor
        -Wdeprecated-declarations
        # -Wdocumentation
        # -Wdocumentation-pedantic
        # -Wdocumentation-unknown-command
        # -Wdouble-promotion
        # -Wexit-time-destructors
        # -Wextra-semi
        # -Wextra-semi-stmt
        # -Wfloat-conversion
        # -Wfloat-equal
        # -Wformat-nonliteral
        # -Wglobal-constructors
        # -Wimplicit-float-conversion
        # -Wimplicit-int-float-conversion
        # -Winconsistent-missing-destructor-override
        # -Winconsistent-missing-override
        # -Wmacro-redefined
        # -Wmissing-noreturn
        # -Wmissing-prototypes
        # -Wnewline-eof
        # -Wold-style-cast
        -Wreserved-identifier
        -Wreserved-macro-identifier
        # -Wshorten-64-to-32
        # -Wsign-compare
        # -Wsign-conversion
        # -Wsuggest-override
        # -Wsuggest-destructor-override
        # -Wswitch-default
        # -Wswitch-enum
        -Wundefined-func-template
        -Wunreachable-code
        -Wunreachable-code-return
        # -Wunsafe-buffer-usage
        # -Wunsafe-buffer-usage-in-libc-call
        # -Wunused-macros
        # -Wunused-parameter
        # -Wunused-variable
    )
endif()
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
        if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang" AND NOT MSVC)
            target_compile_options(chemistry PRIVATE --coverage -fprofile-arcs -ftest-coverage)
            target_link_libraries(chemistry PRIVATE --coverage)
        else()
            message(WARNING "Coverage build is only supported with GCC or Clang compilers")
        endif()
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

# Set the config.hpp variable based on the option
if(QDK_CHEMISTRY_DISABLE_TRACE_LOG)
  set(QDK_DISABLE_TRACE_LOG 1)
endif()

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

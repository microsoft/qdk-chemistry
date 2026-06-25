# Handle discovery / fetching of dependencies
include(DependencyManager)

# Extract QDK_UARCH FLAGS
set(DEPENDENCY_BUILD_FLAGS BUILD_ARGS "${QDK_UARCH_FLAGS}")
if(NOT MSVC)
    set(DEPENDENCY_BUILD_FLAGS "${DEPENDENCY_BUILD_FLAGS} -fPIC")
endif()

# Save current warning settings
get_property(_old_warn_deprecated CACHE CMAKE_WARN_DEPRECATED PROPERTY VALUE)
get_property(_old_suppress_dev CACHE CMAKE_SUPPRESS_DEVELOPER_WARNINGS PROPERTY VALUE)

# Suppress warnings for dependencies
set(CMAKE_WARN_DEPRECATED FALSE CACHE BOOL "" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS TRUE CACHE BOOL "" FORCE)

# Dependencies that must be installed by the system
if(QDK_ENABLE_OPENMP)
    find_package(OpenMP REQUIRED)
endif()
find_package(Threads REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(HDF5 REQUIRED COMPONENTS CXX)

if(QDK_CHEMISTRY_ENABLE_MPI)
  find_package(MPI REQUIRED)
endif()

# NLOHMANN_JSON for JSON management
set(JSON_Install ON CACHE BOOL "Enable JSON Install" FORCE)
handle_dependency(nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.12.0
  BUILD_TARGET nlohmann_json::nlohmann_json
  INSTALL_TARGET nlohmann_json::nlohmann_json
  EXPORTED_VARIABLES nlohmann_json::nlohmann_json
  ${DEPENDENCY_BUILD_FLAGS}
  REQUIRED
)

# Libint2 for CPU Integral evaluation
set(_libint2_source_subdir "SOURCE_SUBDIR;libint-2.9.0")
if(APPLE)
    set(_libint2_source_subdir "")
endif()
# MSVC native cl does not define __SSE__ / __SSE2__ macros on x64, which causes
# the AVX section of vector_x86.h to reference VectorSSEDouble before it's defined.
# Apply a patch to define these macros under MSVC x64.
set(_libint2_patch_args "")
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_libint2_patch_args FETCHCONTENT_ARGS
        PATCH_COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_LIST_DIR}/patches/libint2-msvc-sse-macros.cmake"
    )
endif()
handle_dependency(libint2
  URL https://github.com/evaleev/libint/releases/download/v2.9.0/libint-2.9.0-mpqc4.tgz
  BUILD_TARGET Libint2::cxx
  INSTALL_TARGET Libint2::cxx
  ${_libint2_source_subdir}
  ${DEPENDENCY_BUILD_FLAGS}
  ${_libint2_patch_args}
  REQUIRED
)
if(MSVC AND TARGET libint2_cxx)
  # /Zc:__cplusplus: MSVC reports __cplusplus as 199711L by default; libint2's
  # cxxstd.h checks __cplusplus >= 201103L for C++11 support.
  # /Zc:preprocessor: enables conforming preprocessor; libint2's engine.impl.h
  # uses Boost.Preprocessor macros that require correct expansion order (the
  # legacy preprocessor concatenates tokens incorrectly). Available since VS 2019 16.5.
  # Skip when libint2_cxx is an IMPORTED target from a previous install
  # (target_compile_options rejects IMPORTED targets).
  get_target_property(_libint2_cxx_imported libint2_cxx IMPORTED)
  if(NOT _libint2_cxx_imported)
    # clang-cl does not support /Zc:preprocessor (MSVC-only); passing it produces
    # -Wunused-command-line-argument on every TU that consumes libint2_cxx.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
      target_compile_options(libint2_cxx INTERFACE /Zc:__cplusplus)
    else()
      target_compile_options(libint2_cxx INTERFACE /Zc:__cplusplus /Zc:preprocessor
        # Libint2 headers fire these warnings in every TU that includes them.
        # Scope: INTERFACE so the suppression travels only to direct consumers
        # of libint2_cxx, not globally.
        # /wd4018 signed/unsigned mismatch (<, >, <=, >=)
        # /wd4068 unknown pragma (GCC pragmas inside libint2 headers)
        # /wd4389 signed/unsigned mismatch (==, !=)
        /wd4018 /wd4068 /wd4389)
    endif()
  endif()
endif()
# eritest-libint2 links only to libint2-static (C library), so it does not pick
# up the INTERFACE flags from libint2_cxx. The test source includes libint2/boys.h
# which requires C++11 detection via __cplusplus. Only present when libint2 is
# built from source.
if(MSVC AND TARGET eritest-libint2)
  get_target_property(_eritest_imported eritest-libint2 IMPORTED)
  if(NOT _eritest_imported)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
      target_compile_options(eritest-libint2 PRIVATE /Zc:__cplusplus)
    else()
      target_compile_options(eritest-libint2 PRIVATE /Zc:__cplusplus /Zc:preprocessor)
    endif()
  endif()
endif()

# ecpint for ECP-related integral evaluation
set(LIBECPINT_BUILD_TESTS OFF CACHE BOOL "Enable ECPINT Tests" FORCE)
set(LIBECPINT_USE_PUGIXML OFF CACHE BOOL "Use pugixml for ECPINT" FORCE)
# MSVC native cl does not support C99 VLAs used throughout ecpint.
# Apply a patch script that replaces them with std::vector.
set(_ecpint_patch_args "")
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(_ecpint_patch_args FETCHCONTENT_ARGS
        PATCH_COMMAND "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_LIST_DIR}/patches/ecpint-msvc-vla.cmake"
    )
endif()
handle_dependency(ecpint
  GIT_REPOSITORY https://github.com/robashaw/libecpint
  GIT_TAG v1.0.7
  BUILD_TARGET ECPINT::ecpint
  INSTALL_TARGET ECPINT::ecpint
  ${DEPENDENCY_BUILD_FLAGS}
  ${_ecpint_patch_args}
  REQUIRED
)


# gauxc for XC evaluation
set(EXCHCXX_ENABLE_LIBXC OFF CACHE BOOL "Enable LibXC Support"         FORCE)
set(GAUXC_ENABLE_HDF5    OFF CACHE BOOL "Enable gauxc HDF5 Support"    FORCE)
set(GAUXC_ENABLE_MAGMA   OFF CACHE BOOL "Enable gauxc MAGMA Support"   FORCE)
set(GAUXC_ENABLE_CUTLASS ON  CACHE BOOL "Enable gauxc CUTLASS Support" FORCE)
set(GAUXC_ENABLE_CUDA ${QDK_CHEMISTRY_ENABLE_GPU} CACHE BOOL "Enable gauxc CUDA Support" FORCE)
set(GAUXC_ENABLE_MPI  ${QDK_CHEMISTRY_ENABLE_MPI} CACHE BOOL "Enable gauxc MPI Support"  FORCE)
set(GAUXC_ENABLE_OPENMP ${QDK_ENABLE_OPENMP} CACHE BOOL "Enable gauxc OpenMP Support" FORCE)

handle_dependency(gauxc
  GIT_REPOSITORY https://github.com/lorisercole/gauxc.git
  GIT_TAG 1e9d122b5c7d98b3d8187cee2ecb804b0b511136
  BUILD_TARGET gauxc::gauxc
  INSTALL_TARGET gauxc::gauxc
  ${DEPENDENCY_BUILD_FLAGS}
  REQUIRED
)

# Restore previous settings
set(CMAKE_WARN_DEPRECATED ${_old_warn_deprecated} CACHE BOOL "" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${_old_suppress_dev} CACHE BOOL "" FORCE)

if(MSVC)
  # Suppress warnings in upstream third-party sources (ecpint, libint2).
  # Helper: apply compile options only to targets actually built by this project.
  # handle_dependency() may resolve a dependency via find_package() to a prior
  # install, producing IMPORTED targets that reject target_compile_options.
  function(_qdk_suppress_if_built target)
    if(NOT TARGET ${target})
      return()
    endif()
    get_target_property(_is_imported ${target} IMPORTED)
    if(_is_imported)
      return()
    endif()
    # clang-cl: use -Wno-* flags for the specific warnings seen in these sources.
    # cl.exe: use a fixed set of /wd flags for known warning categories:
    #   /wd4018 signed/unsigned mismatch
    #   /wd4068 unknown pragma
    #   /wd4100 unreferenced formal parameter
    #   /wd4101 unreferenced local variable
    #   /wd4127 conditional expression is constant
    #   /wd4242 conversion with possible loss of data
    #   /wd4244 conversion with possible loss of data
    #   /wd4245 conversion, signed/unsigned mismatch
    #   /wd4267 conversion with possible loss of data
    #   /wd4389 signed/unsigned mismatch
    #   /wd4701 potentially uninitialized local variable used
    #   /wd4703 potentially uninitialized local pointer variable used
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
      target_compile_options(${target} PRIVATE
        -Wno-unused-variable -Wno-vla-cxx-extension -Wno-reorder-ctor)
    else()
      target_compile_options(${target} PRIVATE
        /wd4018 /wd4068 /wd4100 /wd4101 /wd4127 /wd4242 /wd4244 /wd4245
        /wd4267 /wd4389 /wd4701 /wd4703)
    endif()
  endfunction()

  _qdk_suppress_if_built(ecpint)
  # ecpint also builds a `generate` executable at build time for code generation
  _qdk_suppress_if_built(generate)
  # libint2 is a wrapper around $<TARGET_OBJECTS:libint2_obj>; the OBJECT
  # library libint2_obj is what actually compiles the auto-generated sources.
  # libint2_cxx is a header-only INTERFACE target with no compiled sources.
  _qdk_suppress_if_built(libint2_obj)
endif()

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
# MSVC x64 doesn't define __SSE__/__SSE2__; patch vector_x86.h to define them.
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
  # libint2 needs /Zc:__cplusplus (C++11 detection) and /Zc:preprocessor
  # (Boost.Preprocessor). Apply to both FetchContent and imported targets.
  # clang-cl rejects /Zc:preprocessor; omit it there.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    target_compile_options(libint2_cxx INTERFACE /Zc:__cplusplus)
  else()
    target_compile_options(libint2_cxx INTERFACE /Zc:__cplusplus /Zc:preprocessor)
  endif()
endif()
# eritest-libint2 links only to libint2-static (C library), so it misses the
# INTERFACE flags from libint2_cxx but still needs C++11 detection.
if(MSVC AND TARGET eritest-libint2)
  # eritest-libint2 links only to libint2-static (C library), so it misses the
  # INTERFACE flags from libint2_cxx but still needs C++11 detection.
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
    target_compile_options(eritest-libint2 PRIVATE /Zc:__cplusplus)
  else()
    target_compile_options(eritest-libint2 PRIVATE /Zc:__cplusplus /Zc:preprocessor)
  endif()
endif()

# MSVC's /O2 optimizer is pathologically slow on libint2's large CMake Unity
# translation units (hours vs minutes for clang-cl). Disable Unity for libint2 on
# MSVC so the small generated TUs compile quickly and parallelize; clang-cl keeps it.
if(MSVC AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND TARGET libint2_obj)
  set_target_properties(libint2_obj PROPERTIES UNITY_BUILD OFF)
endif()

# ecpint for ECP-related integral evaluation
set(LIBECPINT_BUILD_TESTS OFF CACHE BOOL "Enable ECPINT Tests" FORCE)
set(LIBECPINT_USE_PUGIXML OFF CACHE BOOL "Use pugixml for ECPINT" FORCE)
# MSVC doesn't support the C99 VLAs ecpint uses; patch replaces them with std::vector.
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
  GIT_REPOSITORY https://github.com/wavefunction91/gauxc.git
  GIT_TAG f05cd68e1fd549cc45a318e6d039f49d044d3e1d
  BUILD_TARGET gauxc::gauxc
  INSTALL_TARGET gauxc::gauxc
  ${DEPENDENCY_BUILD_FLAGS}
  REQUIRED
)

# Restore previous settings
set(CMAKE_WARN_DEPRECATED ${_old_warn_deprecated} CACHE BOOL "" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${_old_suppress_dev} CACHE BOOL "" FORCE)

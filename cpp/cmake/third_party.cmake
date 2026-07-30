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
foreach(_libint2_cxx_target libint2_cxx Libint2::libint2_cxx)
  if(MSVC AND TARGET ${_libint2_cxx_target})
    # libint2 needs /Zc:__cplusplus (C++11 detection) and /Zc:preprocessor
    # (Boost.Preprocessor). Apply to both the FetchContent target (libint2_cxx)
    # and the installed imported target (Libint2::libint2_cxx).
    # clang-cl rejects /Zc:preprocessor; omit it there.
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" AND CMAKE_CXX_SIMULATE_ID STREQUAL "MSVC")
      target_compile_options(${_libint2_cxx_target} INTERFACE /Zc:__cplusplus)
    else()
      target_compile_options(${_libint2_cxx_target} INTERFACE /Zc:__cplusplus /Zc:preprocessor)
    endif()
  endif()
endforeach()
# eritest-libint2 links only to libint2-static (C library), so it misses the
# INTERFACE flags from libint2_cxx but still needs C++11 detection.
if(MSVC AND TARGET eritest-libint2)
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

# SeQuant provides the symbolic Baker-Campbell-Hausdorff machinery for the DUCC
# effective-Hamiltonian dressing (ducc_level > 0). Pinned to the upstream commit
# the DUCC backend targets (no local patch: the >2-body pruning uses the public
# keep_up_to_body filter).
#
# SeQuant's numeric deps BTAS and range-v3 are fetched + installed by qdk here
# (via handle_dependency, before SeQuant) rather than left to SeQuant's own
# FindOrFetch modules. SeQuant fetches them with EXCLUDE_FROM_ALL, so their
# install() rules never run: the *installed* sequant-config then points at a
# btas-config.cmake / range-v3-config.cmake that was never written -- a hard
# FATAL_ERROR for anything that consumes the installed qdk (the two-phase
# build/install then find_package(qdk) path). Owning them here makes them install
# normally; SeQuant's FindOrFetch{BTAS,RangeV3} short-circuit on the pre-existing
# BTAS::BTAS / range-v3::range-v3 targets and reuse them (SEQUANT_HAS_BTAS still
# turns on, so the eval-btas backend is built against qdk's BTAS::BTAS).
#
# Boost is not a first-class qdk-chemistry dependency (it only arrives
# transitively via libint2). BTAS and SeQuant both need a wider set of Boost
# components (locale, regex, hana, ...), so discover the modular Boost here
# first: this defines Boost_CONFIG, which routes SeQuant's FindOrFetchBoost to
# reuse the same Boost libint2 uses instead of fetching a second modular copy
# (which collides on the Boost::headers target). No Fortran compiler is required:
# BTAS and SeQuant are C++/header-only and the blaspp/lapackpp wrappers are
# reused prebuilt (they need only the libgfortran runtime, not the compiler).
find_package(Boost CONFIG REQUIRED)
set(Boost_FETCH_IF_MISSING OFF CACHE BOOL "" FORCE)

# range-v3 (header-only): pinned to SeQuant's SEQUANT_TRACKED_RANGEV3_TAG. Not a
# top-level project here, so tests/examples/perf default off; force them off
# regardless so nothing extra builds.
set(RANGE_V3_TESTS    OFF CACHE BOOL "" FORCE)
set(RANGE_V3_EXAMPLES OFF CACHE BOOL "" FORCE)
set(RANGE_V3_PERF     OFF CACHE BOOL "" FORCE)
set(RANGE_V3_DOCS     OFF CACHE BOOL "" FORCE)
handle_dependency(range-v3
  GIT_REPOSITORY https://github.com/ericniebler/range-v3.git
  GIT_TAG 0.12.0
  BUILD_TARGET range-v3::range-v3
  INSTALL_TARGET range-v3::range-v3
  ${DEPENDENCY_BUILD_FLAGS}
  REQUIRED
)

# BTAS and SeQuant both FetchContent ValeevGroup's kit-cmake toolkit under the
# name "vg_cmake_kit", pinned to different commits. FetchContent uses the first
# declaration, and qdk builds BTAS before SeQuant, so BTAS's older pin would win
# and SeQuant's newer CheckCXXFeatures module would be missing (SeQuant's
# include(CheckCXXFeatures) then FATALs). Pre-declare it at SeQuant's (newer) pin
# so both reuse a compatible toolkit. Declare only -- whichever of BTAS/SeQuant
# is fetched first populates it from this declaration.
include(FetchContent)
FetchContent_Declare(
  vg_cmake_kit
  GIT_REPOSITORY https://github.com/ValeevGroup/kit-cmake.git
  GIT_TAG 256d9462bb765787f5acb69be154b26d6efba8b6
)

# BTAS (header-only tensor library, the DUCC evaluator's numeric backend): pinned
# to SeQuant's SEQUANT_TRACKED_BTAS_TAG. Reuses the prebuilt blaspp/lapackpp and
# the Boost discovered above. Do NOT set BLA_VENDOR (it flips BTAS onto the
# standard-linalg-kit path that needs an explicit Fortran mangling convention).
handle_dependency(BTAS
  GIT_REPOSITORY https://github.com/BTAS/btas.git
  GIT_TAG 9c8c8f68fee2b82e64755270a8348e4612cf9941
  BUILD_TARGET BTAS
  INSTALL_TARGET BTAS::BTAS
  ${DEPENDENCY_BUILD_FLAGS}
  REQUIRED
)

# Library only: skip benchmarks, utilities, and the python module; enable the
# BTAS numeric backend (reuses the BTAS::BTAS target above). SEQUANT_TESTS=OFF
# drops SeQuant's test executables and CTest registration. (SeQuant still
# compiles a few test *object* libraries unconditionally -- they are not gated by
# SEQUANT_TESTS upstream -- but the heavier executable links are skipped;
# removing them entirely would require patching the pinned dependency.)
set(SEQUANT_BTAS       ON  CACHE BOOL "SeQuant BTAS eval backend" FORCE)
set(SEQUANT_TESTS      OFF CACHE BOOL "SeQuant unit tests"        FORCE)
set(SEQUANT_BENCHMARKS OFF CACHE BOOL "SeQuant benchmarks"        FORCE)
set(SEQUANT_UTILITIES  OFF CACHE BOOL "SeQuant utility programs"  FORCE)
set(SEQUANT_PYTHON     OFF CACHE BOOL "SeQuant python module"     FORCE)

handle_dependency(SeQuant
  GIT_REPOSITORY https://github.com/ValeevGroup/SeQuant.git
  GIT_TAG 1e033dfc3bf21aaa30eaef9afb02ab382f6d6a07
  BUILD_TARGET SeQuant::SeQuant
  INSTALL_TARGET SeQuant::SeQuant
  ${DEPENDENCY_BUILD_FLAGS}
  REQUIRED
)

# Restore previous settings
set(CMAKE_WARN_DEPRECATED ${_old_warn_deprecated} CACHE BOOL "" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${_old_suppress_dev} CACHE BOOL "" FORCE)

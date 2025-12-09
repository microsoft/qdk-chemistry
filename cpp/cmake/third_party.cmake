# Handle discovery / fetching of dependencies
include(DependencyManager)

# Extract QDK_UARCH FLAGS
set(DEPENDENCY_BUILD_FLAGS BUILD_ARGS "${QDK_UARCH_FLAGS} -fPIC")

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
handle_dependency(libint2
  URL https://github.com/evaleev/libint/releases/download/v2.9.0/libint-2.9.0-mpqc4.tgz
  BUILD_TARGET Libint2::cxx
  INSTALL_TARGET Libint2::cxx
  ${_libint2_source_subdir}
  ${DEPENDENCY_BUILD_FLAGS}
  REQUIRED
)

# ecpint for ECP-related integral evaluation
set(LIBECPINT_BUILD_TESTS OFF CACHE BOOL "Enable ECPINT Tests" FORCE)
set(LIBECPINT_USE_PUGIXML OFF CACHE BOOL "Use pugixml for ECPINT" FORCE)
handle_dependency(ecpint
  GIT_REPOSITORY https://github.com/robashaw/libecpint
  GIT_TAG v1.0.7
  BUILD_TARGET ECPINT::ecpint
  INSTALL_TARGET ECPINT::ecpint
  ${DEPENDENCY_BUILD_FLAGS}
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
  GIT_TAG 62fea07c9306dbd83dd18b6957358827ac9b3da0
  BUILD_TARGET gauxc::gauxc
  INSTALL_TARGET gauxc::gauxc
  ${DEPENDENCY_BUILD_FLAGS}
  REQUIRED
)

# Restore previous settings
set(CMAKE_WARN_DEPRECATED ${_old_warn_deprecated} CACHE BOOL "" FORCE)
set(CMAKE_SUPPRESS_DEVELOPER_WARNINGS ${_old_suppress_dev} CACHE BOOL "" FORCE)

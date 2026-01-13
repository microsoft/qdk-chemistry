# memory_pools.cmake - Ninja job pools for memory-intensive compilations
# =========================================================================
#
# Some source files in this project use >3GB RAM during compilation due to
# heavy template instantiation (primarily libint2). When building with Ninja,
# this module defines a "heavy_compile" job pool that limits how many of these
# files compile concurrently, preventing OOM on machines with limited RAM.
#
# Pool size is auto-calculated: (RAM - 4GB) / 4GB, clamped to [1, num_cpus]
# Override with: cmake -DQDK_HEAVY_COMPILE_POOL_SIZE=N
#
# Requires CMake 4.2+ for source-file-level JOB_POOL_COMPILE property.
# =========================================================================

set(QDK_HEAVY_FILE_PEAK_GB 4)  # Peak memory per heavy file (rounded up from 3.2GB)
set(QDK_RESERVED_GB 4)         # Memory reserved for OS/other processes

# -----------------------------------------------------------------------------
# Internal: Detect system RAM in GB
# -----------------------------------------------------------------------------
function(_qdk_get_system_memory_gb out_var)
    set(mem_gb 0)
    if(EXISTS "/proc/meminfo")
        file(READ "/proc/meminfo" meminfo)
        string(REGEX MATCH "MemTotal:[ \t]+([0-9]+)" _ "${meminfo}")
        if(CMAKE_MATCH_1)
            math(EXPR mem_gb "${CMAKE_MATCH_1} / 1024 / 1024")
        endif()
    elseif(APPLE)
        execute_process(
            COMMAND sysctl -n hw.memsize
            OUTPUT_VARIABLE m OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET
        )
        if(m)
            math(EXPR mem_gb "${m} / 1024 / 1024 / 1024")
        endif()
    endif()
    set(${out_var} ${mem_gb} PARENT_SCOPE)
endfunction()

# -----------------------------------------------------------------------------
# Internal: Calculate optimal pool size based on available RAM
# -----------------------------------------------------------------------------
function(_qdk_calc_pool_size out_var)
    _qdk_get_system_memory_gb(ram_gb)
    if(ram_gb GREATER QDK_RESERVED_GB)
        math(EXPR usable "${ram_gb} - ${QDK_RESERVED_GB}")
        math(EXPR pool "${usable} / ${QDK_HEAVY_FILE_PEAK_GB}")
        cmake_host_system_information(RESULT cpus QUERY NUMBER_OF_LOGICAL_CORES)
        if(pool LESS 1)
            set(pool 1)
        elseif(pool GREATER cpus)
            set(pool ${cpus})
        endif()
        set(${out_var} ${pool} PARENT_SCOPE)
    else()
        set(${out_var} 1 PARENT_SCOPE)
    endif()
endfunction()

# -----------------------------------------------------------------------------
# Setup: Configure pools for Ninja, warn on low-memory systems
# -----------------------------------------------------------------------------
_qdk_get_system_memory_gb(_qdk_system_ram_gb)

if(CMAKE_GENERATOR MATCHES "Ninja")
    # Calculate or use user-provided pool size
    if(NOT DEFINED QDK_HEAVY_COMPILE_POOL_SIZE)
        _qdk_calc_pool_size(QDK_HEAVY_COMPILE_POOL_SIZE)
    endif()
    set(QDK_HEAVY_COMPILE_POOL_SIZE "${QDK_HEAVY_COMPILE_POOL_SIZE}"
        CACHE STRING "Max concurrent heavy compilations (auto-detected from RAM)")

    # Define the pool globally
    set_property(GLOBAL PROPERTY JOB_POOLS "heavy_compile=${QDK_HEAVY_COMPILE_POOL_SIZE}")

    message(STATUS "Memory pools: ${_qdk_system_ram_gb}GB RAM detected, heavy_compile pool = ${QDK_HEAVY_COMPILE_POOL_SIZE}")
    set(QDK_MEMORY_POOLS_AVAILABLE TRUE CACHE INTERNAL "")
else()
    # Pools are Ninja-only. Fall back to a safe global parallel level for other generators.
    set(QDK_MEMORY_POOLS_AVAILABLE FALSE CACHE INTERNAL "")

    if(NOT DEFINED QDK_HEAVY_COMPILE_POOL_SIZE)
        _qdk_calc_pool_size(QDK_HEAVY_COMPILE_POOL_SIZE)
    endif()
    set(QDK_HEAVY_COMPILE_POOL_SIZE "${QDK_HEAVY_COMPILE_POOL_SIZE}" CACHE STRING "Max concurrent heavy compilations (auto-detected from RAM)")

    # Expose a hint for non-Ninja builds: use CMAKE_BUILD_PARALLEL_LEVEL to avoid OOMs.
    set(QDK_BUILD_PARALLEL_LEVEL_HINT ${QDK_HEAVY_COMPILE_POOL_SIZE} CACHE STRING "Suggested global parallel level for non-Ninja builds")

    # Provide a CMake-only safe target to build with capped parallelism: cmake --build . --target qdk_build_safe
    if(NOT TARGET qdk_build_safe)
        add_custom_target(qdk_build_safe
            COMMAND ${CMAKE_COMMAND} -E env CMAKE_BUILD_PARALLEL_LEVEL=${QDK_BUILD_PARALLEL_LEVEL_HINT} ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} $<$<BOOL:${CMAKE_CONFIGURATION_TYPES}>:--config> $<$<BOOL:${CMAKE_CONFIGURATION_TYPES}>:$<CONFIG>>
            USES_TERMINAL
            COMMENT "Building with CMAKE_BUILD_PARALLEL_LEVEL=${QDK_BUILD_PARALLEL_LEVEL_HINT} to avoid OOM"
        )
        message(STATUS "Non-Ninja generator: added target qdk_build_safe (CMAKE_BUILD_PARALLEL_LEVEL=${QDK_BUILD_PARALLEL_LEVEL_HINT})")
    endif()

    # Warn on low-memory systems not using Ninja
    if(_qdk_system_ram_gb GREATER 0 AND _qdk_system_ram_gb LESS 32)
        message(WARNING
            "System has ${_qdk_system_ram_gb}GB RAM. Some files require >3GB to compile.\n"
            "Use Ninja (cmake -G Ninja ..) or: cmake --build . --target qdk_build_safe"
        )
    endif()
endif()

# -----------------------------------------------------------------------------
# Centralized list of memory-intensive source files (>2GB RAM during compilation)
# Paths are relative to cpp/src/qdk/chemistry/algorithms/microsoft
# -----------------------------------------------------------------------------
set(QDK_HEAVY_SOURCES
    # microsoft/ top-level
    localization/vvhv.cpp
    localization/pipek_mezey.cpp
    hamiltonian.cpp
    scf.cpp
    utils.cpp
    # scf/src/
    scf/src/core/basis_set.cpp
    scf/src/util/int1e.cpp
    scf/src/util/libint2_util.cpp
    scf/src/scf/cpscf.cpp
    scf/src/scf/scf_impl.cpp
    scf/src/scf/ks_impl.cpp
    scf/src/scf/scf_solver.cpp
    # scf/src/eri/
    scf/src/eri/schwarz.cpp
    scf/src/eri/eri_df_base.cpp
    # scf/src/eri/INCORE/
    scf/src/eri/INCORE/incore.cpp
    scf/src/eri/INCORE/incore_impl.cpp
    scf/src/eri/INCORE/incore_impl_df.cpp
    # scf/src/eri/LIBINT2_DIRECT/
    scf/src/eri/LIBINT2_DIRECT/libint2_direct.cpp
    # scf/src/scf_algorithm/
    scf/src/scf_algorithm/scf_algorithm.cpp
    scf/src/scf_algorithm/diis.cpp
    scf/src/scf_algorithm/diis_gdm.cpp
    scf/src/scf_algorithm/gdm.cpp
)

# -----------------------------------------------------------------------------
# qdk_apply_heavy_source_pools()
#
# Apply the heavy_compile job pool to all files in QDK_HEAVY_SOURCES.
# Call this once from the top-level CMakeLists.txt after all targets are defined.
# -----------------------------------------------------------------------------
function(qdk_apply_heavy_source_pools)
    if(NOT QDK_MEMORY_POOLS_AVAILABLE)
        return()
    endif()

    set(base_dir "${PROJECT_SOURCE_DIR}/src/qdk/chemistry/algorithms/microsoft")
    foreach(src IN LISTS QDK_HEAVY_SOURCES)
        set(full_path "${base_dir}/${src}")
        if(EXISTS "${full_path}")
            set_source_files_properties("${full_path}" PROPERTIES JOB_POOL_COMPILE heavy_compile)
        else()
            message(WARNING "Heavy source file not found: ${full_path}")
        endif()
    endforeach()
    list(LENGTH QDK_HEAVY_SOURCES count)
    message(STATUS "Memory pools: marked ${count} heavy source files")
endfunction()

vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO OpenMathLib/OpenBLAS
    REF "v${VERSION}"
    SHA512 046316b4297460bffca09c890ecad17ea39d8b3db92ff445d03b547dd551663d37e40f38bce8ae11e2994374ff01e622b408da27aa8e40f4140185ee8f001a60
    HEAD_REF develop
    PATCHES
        disable-testing.diff
        getarch.diff
        system-check-msvc.diff
)

vcpkg_check_features(OUT_FEATURE_OPTIONS OPTIONS
    FEATURES
        threads        USE_THREAD
        simplethread   USE_SIMPLE_THREADED_LEVEL3
        dynamic-arch   DYNAMIC_ARCH
)

# If not explicitly configured for a cross build, OpenBLAS wants to run
# getarch executables in order to optimize for the target.
# Adapting this to vcpkg triplets:
# - install-getarch.diff introduces and uses GETARCH_BINARY_DIR,
# - architecture and system name are required to match for GETARCH_BINARY_DIR, but
# - uwp (aka WindowsStore) may run windows getarch.
string(REPLACE "WindowsStore_" "_" SYSTEM_KEY "${VCPKG_CMAKE_SYSTEM_NAME}_${VCPKG_TARGET_ARCHITECTURE}")
set(GETARCH_BINARY_DIR "${CURRENT_HOST_INSTALLED_DIR}/manual-tools/${PORT}/${SYSTEM_KEY}")
if(EXISTS "${GETARCH_BINARY_DIR}")
    message(STATUS "OpenBLAS cross build, but may use ${PORT}:${HOST_TRIPLET} getarch")
    list(APPEND OPTIONS "-DGETARCH_BINARY_DIR=${GETARCH_BINARY_DIR}")
elseif(VCPKG_CROSSCOMPILING)
    message(STATUS "OpenBLAS cross build, may not be able to use getarch")
else()
    message(STATUS "OpenBLAS native build")
endif()

if(VCPKG_TARGET_IS_EMSCRIPTEN)
    # Only the riscv64 kernel with riscv64_generic target is supported.
    # Cf. https://github.com/OpenMathLib/OpenBLAS/issues/3640#issuecomment-1144029630 et al.
    list(APPEND OPTIONS
        -DEMSCRIPTEN_SYSTEM_PROCESSOR=riscv64
        -DTARGET=RISCV64_GENERIC
    )
endif()

# QDK overlay change: build with clang-cl on Windows ARM64.
#
# MSVC cannot assemble OpenBLAS's ARM64 kernels, which are GNU-syntax .S files.
# The only MSVC-compatible fallback is TARGET=GENERIC, but that yields blocking
# parameters which make the blocked LU factorisation loop forever: getrf_single.c
# advances its loops by GEMM_P / GEMM_Q / GEMM_R, and a zero step never
# terminates. It is a silent hang, not a build or link error -- dgemm returns
# correct results and dgetrf never returns above the getf2 crossover (verified
# on CI: n = 1, 2, 4 return via getf2, n = 24 hangs).
#
# clang-cl understands the GNU assembly dialect, so getarch can select the real
# ARMV8 kernels. This is also the configuration OpenBLAS itself supports: its
# Windows ARM64 CI job builds with clang-cl and never with MSVC.
if(VCPKG_TARGET_ARCHITECTURE MATCHES "^arm64" AND VCPKG_TARGET_IS_WINDOWS AND NOT VCPKG_TARGET_IS_MINGW)
    find_program(QDK_OPENBLAS_CLANG_CL
        NAMES clang-cl clang-cl.exe
        PATHS
            "$ENV{ProgramFiles}/LLVM/bin"
            "$ENV{ProgramW6432}/LLVM/bin"
            "$ENV{VCINSTALLDIR}/Tools/Llvm/ARM64/bin"
            "$ENV{VCINSTALLDIR}/Tools/Llvm/bin"
    )
    # Fail loudly rather than silently falling back to the MSVC/GENERIC build,
    # which produces a library that hangs at runtime instead of failing to build.
    if(NOT QDK_OPENBLAS_CLANG_CL)
        message(FATAL_ERROR
            "clang-cl is required to build OpenBLAS for Windows ARM64: MSVC cannot "
            "assemble the ARM64 kernels, and the GENERIC fallback hangs in getrf.")
    endif()
    message(STATUS "OpenBLAS Windows ARM64: building with ${QDK_OPENBLAS_CLANG_CL}")
    list(APPEND OPTIONS
        "-DCMAKE_C_COMPILER=${QDK_OPENBLAS_CLANG_CL}"
        "-DCMAKE_ASM_COMPILER=${QDK_OPENBLAS_CLANG_CL}"
    )
endif()

vcpkg_cmake_configure(
    SOURCE_PATH "${SOURCE_PATH}"
    OPTIONS
        ${OPTIONS}
        "-DCMAKE_PROJECT_INCLUDE=${CURRENT_PORT_DIR}/cmake-project-include.cmake"
        -DBUILD_TESTING=OFF
        # QDK overlay change: BUILD_WITHOUT_LAPACK=OFF to include LAPACK routines.
        # C_LAPACK=ON uses OpenBLAS's embedded C-translated LAPACK (no Fortran needed).
        -DBUILD_WITHOUT_LAPACK=OFF
        -DNOFORTRAN=ON
        -DC_LAPACK=ON
    MAYBE_UNUSED_VARIABLES
        GETARCH_BINARY_DIR
)

vcpkg_cmake_install()
vcpkg_copy_pdbs()
vcpkg_cmake_config_fixup(CONFIG_PATH lib/cmake/OpenBLAS)
vcpkg_fixup_pkgconfig()

# Required from native builds, optional from cross builds.
if(NOT VCPKG_CROSSCOMPILING OR EXISTS "${CURRENT_PACKAGES_DIR}/bin/getarch${VCPKG_TARGET_EXECUTABLE_SUFFIX}")
    vcpkg_copy_tools(
        TOOL_NAMES getarch getarch_2nd
        DESTINATION "${CURRENT_PACKAGES_DIR}/manual-tools/${PORT}/${SYSTEM_KEY}"
        AUTO_CLEAN
    )
endif()

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include" "${CURRENT_PACKAGES_DIR}/debug/share")

vcpkg_install_copyright(FILE_LIST "${SOURCE_PATH}/LICENSE")

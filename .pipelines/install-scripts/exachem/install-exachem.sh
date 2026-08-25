#!/usr/bin/env bash
#
# install-exachem.sh — build and install ExaChem (+ its TAMM tensor backend) for CI, to run as an external MPI
# process. Reuses OpenBLAS/BLAS++/LAPACK++/LibInt2/GauXC/spdlog/EcpInt/nlohmann_json already built into
# CPP_DEPS_PREFIX by install-cpp-deps.sh (TAMM finds BLAS++/LAPACK++ via its default find_package on
# CMAKE_PREFIX_PATH; LibInt2/GauXC/spdlog/EcpInt/nlohmann_json via explicit -D*_ROOT below), and the system MPI
# (e.g. apt-installed openmpi-bin/libopenmpi-dev on Ubuntu). GPU is not supported here. TAMM's own CMSB
# superbuild (NWChemEx-Project/CMakeBuild) is patched (see patches/cmsb-fix-dependency-reuse.patch) so it also
# reuses the system OpenBLAS/LAPACK/spdlog/EcpInt/nlohmann_json instead of building its own redundant copies.
#
# TAMM/ExaChem's commits are hardcoded below (TAMM_COMMIT/EXACHEM_COMMIT), not read from
# cpp/manifest/qdk-chemistry/cgmanifest.json: unlike the deps install-cpp-deps.sh builds, TAMM/ExaChem are never
# shipped in the qdk-chemistry wheel or linked into its binary -- they're only used as an external MPI
# subprocess in GHA CI test runs -- so they aren't Component Governance-relevant and don't belong in cgmanifest.
#
# Usage: install-exachem.sh
# Required env vars: CPP_DEPS_PREFIX
# Optional env vars: INSTALL_PREFIX, BUILD_ROOT, MARCH, JOBS, KEEP_BUILD_DIR
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${CPP_DEPS_PREFIX:?CPP_DEPS_PREFIX must be set to the cached qdk-chemistry C++ deps prefix (provides LibInt2/GauXC/BLAS++/LAPACK++)}"

if ! command -v mpicc >/dev/null 2>&1; then
  echo "ERROR: mpicc not found on PATH. Install an MPI runtime (e.g. 'sudo apt-get install -y openmpi-bin libopenmpi-dev') before running this script." >&2
  exit 1
fi

# INSTALL_PREFIX: final TAMM + ExaChem install location (this is what CI puts on PATH/LD_LIBRARY_PATH).
INSTALL_PREFIX="${INSTALL_PREFIX:-/tmp/exachem_install}"
BUILD_ROOT="${BUILD_ROOT:-/tmp/exachem_build}"
KEEP_BUILD_DIR="${KEEP_BUILD_DIR:-0}"

MARCH="${MARCH:-x86-64-v3}"
JOBS="${JOBS:-$(nproc)}"
MODULES="${MODULES:-CC}"
# MPI_PROGRESS_RANK is the only Global Arrays runtime that makes progress on a plain shm/TCP transport (no RDMA
# NIC on these runners); it requires >= 2 MPI ranks (1 data-server rank + >= 1 compute rank).
GA_RUNTIME="${GA_RUNTIME:-MPI_PROGRESS_RANK}"

# TAMM/ExaChem are pinned to specific commits rather than tagged releases: newer upstream main requires GCC >= 14.1
# for C++20 features GCC 13 miscompiles at -O2/-O3, and GitHub's Ubuntu runners default to GCC 13.
TAMM_REPO="https://github.com/NWChemEx/TAMM.git"
TAMM_COMMIT="63c274e37c102a316e844f954bb2387988b0256c"
EXACHEM_REPO="https://github.com/ExaChem/exachem.git"
EXACHEM_COMMIT="45c192e840fd1e0417871d926e9ab87748111e53"

echo "==> ExaChem CI build: march=${MARCH} jobs=${JOBS} modules=${MODULES} ga_runtime=${GA_RUNTIME}"
echo "==> TAMM: ${TAMM_COMMIT} / ExaChem: ${EXACHEM_COMMIT}"
echo "==> Reusing LibInt2/GauXC/BLAS++/LAPACK++ from CPP_DEPS_PREFIX=${CPP_DEPS_PREFIX}"
echo "==> INSTALL_PREFIX=${INSTALL_PREFIX}"

rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}" "${INSTALL_PREFIX}"

# BLAS++/LAPACK++ are found via TAMM's default find_package search (no NO_DEFAULT_PATH), so putting
# CPP_DEPS_PREFIX on CMAKE_PREFIX_PATH is enough for them -- unlike LibInt2/GauXC below, which need explicit
# -D*_ROOT flags. INSTALL_PREFIX is added too so the seeded Eigen3 below (and anything CMSB itself installs) is
# found consistently by both the outer TAMM configure and its nested TAMM_External re-configure.
export CMAKE_PREFIX_PATH="${CPP_DEPS_PREFIX}:${INSTALL_PREFIX}:${CMAKE_PREFIX_PATH:-}"

# TAMM's CMSB superbuild always tries find_package(Eigen3 CONFIG) first and only falls back to git-cloning its own
# copy (from gitlab.com, unreachable from these CI runners) if that fails. Rather than relying on CMake flags --
# which CMSB's nested TAMM_External re-configure does not forward -- seed INSTALL_PREFIX with a copy of the
# apt-installed libeigen3-dev, mirroring the same relative layout (share/eigen3/cmake + include/eigen3) that CMSB
# would have produced had it built Eigen3 itself. INSTALL_PREFIX is on CMAKE_PREFIX_PATH for every CMSB configure
# pass (outer and nested alike), so this is found everywhere without further flags.
EIGEN3_CONFIG="$(find /usr -maxdepth 6 -name 'Eigen3Config.cmake' -print -quit 2>/dev/null || true)"
if [ -z "${EIGEN3_CONFIG}" ]; then
  echo "ERROR: Eigen3Config.cmake not found under /usr (expected from apt's libeigen3-dev)." >&2
  exit 1
fi
EIGEN3_CMAKE_DIR="$(dirname "${EIGEN3_CONFIG}")"                     # e.g. /usr/share/eigen3/cmake
EIGEN3_SHARE_DIR="$(dirname "${EIGEN3_CMAKE_DIR}")"                  # e.g. /usr/share/eigen3
EIGEN3_PREFIX="$(dirname "$(dirname "${EIGEN3_SHARE_DIR}")")"        # e.g. /usr
mkdir -p "${INSTALL_PREFIX}/share/eigen3" "${INSTALL_PREFIX}/include"
cp -r "${EIGEN3_CMAKE_DIR}" "${INSTALL_PREFIX}/share/eigen3/cmake"
cp -r "${EIGEN3_PREFIX}/include/eigen3" "${INSTALL_PREFIX}/include/eigen3"
echo "==> Seeded Eigen3 from ${EIGEN3_PREFIX} into ${INSTALL_PREFIX}"

# TAMM's CMSB Findnumactl.cmake module (cmake/find_external/Findnumactl.cmake) hardcodes NO_DEFAULT_PATH on both
# its find_path/find_library calls and only searches CMAKE_INSTALL_PREFIX -- so apt's libnuma-dev (system-wide,
# under /usr) is invisible to it too, exactly like Eigen3 above. Seed numa.h + libnuma.so into INSTALL_PREFIX.
NUMA_HEADER="$(find /usr -maxdepth 6 -name 'numa.h' -print -quit 2>/dev/null || true)"
NUMA_LIB="$(find /usr -maxdepth 6 -name 'libnuma.so' -print -quit 2>/dev/null || true)"
if [ -z "${NUMA_HEADER}" ] || [ -z "${NUMA_LIB}" ]; then
  echo "ERROR: numa.h/libnuma.so not found under /usr (expected from apt's libnuma-dev)." >&2
  exit 1
fi
NUMA_INCLUDE_DIR="$(dirname "${NUMA_HEADER}")"
mkdir -p "${INSTALL_PREFIX}/include" "${INSTALL_PREFIX}/lib"
cp "${NUMA_INCLUDE_DIR}"/numa*.h "${INSTALL_PREFIX}/include/"
cp -L "${NUMA_LIB}" "${INSTALL_PREFIX}/lib/libnuma.so"
echo "==> Seeded numactl (numa.h + libnuma.so) into ${INSTALL_PREFIX}"

# --------------------------------------------------------------------------------------------------------------------
# Serial HDF5 discovery. Ubuntu's `libhdf5-dev` (already installed by the workflow for qdk-chemistry itself) puts
# headers under a "serial" subdirectory and installs a pkg-config file on most releases; fall back to the
# well-known Debian/Ubuntu multiarch layout when pkg-config can't find it.
# --------------------------------------------------------------------------------------------------------------------
if pkg-config --exists hdf5-serial 2>/dev/null; then
  HDF5_CFLAGS="$(pkg-config --cflags hdf5-serial)"
  HDF5_LIBS="$(pkg-config --libs hdf5-serial)"
elif pkg-config --exists hdf5 2>/dev/null; then
  HDF5_CFLAGS="$(pkg-config --cflags hdf5)"
  HDF5_LIBS="$(pkg-config --libs hdf5)"
else
  HDF5_MULTIARCH="$(uname -m)-linux-gnu"
  HDF5_CFLAGS="-I/usr/include/hdf5/serial"
  HDF5_LIBS="-L/usr/lib/${HDF5_MULTIARCH}/hdf5/serial -lhdf5"
  echo "==> WARNING: no hdf5 pkg-config file found; falling back to ${HDF5_CFLAGS} / ${HDF5_LIBS}." \
       "Verify this matches the actual libhdf5-dev layout if the TAMM/ExaChem configure below fails to find hdf5.h."
fi
echo "==> HDF5: cflags='${HDF5_CFLAGS}' libs='${HDF5_LIBS}'"

# Flags shared by both the TAMM and ExaChem configure lines below. USE_HDF5=OFF disables TAMM's parallel-HDF5 layer
# (the base HDF5 here, like qdk-chemistry's own, is serial-only); USE_SERIAL_IO selects ExaChem's serial-I/O SCF
# path instead. These MUST be identical on both configure lines, or ExaChem's CMSB reconfigures/rebuilds TAMM.
#
# NOTE: TAMM's CMSB always rebuilds its own copies of several dependencies that are already available (system
# BLAS/LAPACK, spdlog, EcpInt, nlohmann_json), because its dependency-resolution loop either uses an overly
# strict find_package(... CONFIG NO_DEFAULT_PATH) search (BLAS/LAPACK) or never bakes a reliable *_ROOT hint for
# re-resolution when consumed by ExaChem's own separate configure (SPDLOG/EcpInt/NJSON). A first, broader attempt
# to fix this via the global -DCMSB_DEBUG_CMAKE=OFF option was abandoned: it also changes how every *other* TAMM
# dependency is resolved in CMSB's nested "TAMM_External"/"EXACHEM_External" build phase, which has no
# build-from-source fallback there and broke outright ("could not find TARGET NJSON_External").
#
# Fixed instead with patches/cmsb-fix-dependency-reuse.patch (candidate for upstreaming; see that file for the
# full per-dependency root-cause writeup) -- a narrowly-scoped patch that leaves CMSB_DEBUG_CMAKE at its default
# and only changes how BLAS/LAPACK/SPDLOG/EcpInt/NJSON specifically are resolved:
#   - BLAS/LAPACK: cmsb_find_dependency() gets a plain find_package(BLAS/LAPACK QUIET) fallback (mirroring the
#     ELPA/HDF5/numactl special cases already there), and BuildGlobalArrays.cmake's
#     find_or_build_dependency(BLAS) call is made unconditional (previously only reachable if a BLAS_External
#     target already happened to exist).
#   - SPDLOG/EcpInt/NJSON: CMSBTargetConfig.cmake.in gets the same baked-@ROOT@ mechanism CMSB already uses for
#     LibInt2/HDF5/HPTT (a literal, build-time-substituted set(<name>_ROOT @<name>_ROOT@), independent of the
#     consuming build's own CMAKE_PREFIX_PATH state), forwarded into CMSB's nested *_External sub-builds the
#     same way -DHDF5_ROOT/-DHPTT_ROOT/-DLibInt2_ROOT already are.
#
# NJSON (nlohmann_json) needs special care: apt's nlohmann-json3-dev is v3.11.3, while ExaChem's own source
# (exachem/common/options/parser_utils.hpp) reaches into nlohmann's *private* detail:: namespace
# (string_input_adapter_type), which only exists from v3.12.0 onward -- exactly the version both CMSB
# (dep_versions.cmake) and qdk-chemistry's own cpp/cmake/third_party.cmake pin. Reusing apt's older package broke
# the ExaChem build outright (confirmed via a live CI run). Fixed by building nlohmann_json v3.12.0 (matching
# cgmanifest.json's pin exactly) into CPP_DEPS_PREFIX ourselves (install-cpp-deps.sh, header-only, no apt
# package needed/installed anymore) and pointing NJSON_ROOT there instead -- exactly matching CMSB's own pin, so
# both qdk-chemistry's own build and ExaChem/TAMM reuse the identical, compatible nlohmann_json install.
CMSB_SRC_DIR="${BUILD_ROOT}/CMakeBuild-patched"
git clone --depth 1 https://github.com/NWChemEx-Project/CMakeBuild.git "${CMSB_SRC_DIR}"
git -C "${CMSB_SRC_DIR}" apply --verbose "${SCRIPT_DIR}/patches/cmsb-fix-dependency-reuse.patch"

COMMON_CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DMODULES="${MODULES}"
  -DLINALG_VENDOR=OpenBLAS
  -DMARCH_FLAGS="-march=${MARCH}"
  -DUSE_HDF5=OFF
  -DTAMM_CXX_FLAGS="-DUSE_SERIAL_IO ${HDF5_CFLAGS}"
  -DLibInt2_ROOT="${CPP_DEPS_PREFIX}"
  -DGauXC_ROOT="${CPP_DEPS_PREFIX}"
  -DSPDLOG_ROOT="${CPP_DEPS_PREFIX}"
  -DEcpInt_ROOT="${CPP_DEPS_PREFIX}"
  -DNJSON_ROOT="${CPP_DEPS_PREFIX}"
  -DFETCHCONTENT_SOURCE_DIR_CMAKEBUILD="${CMSB_SRC_DIR}"
)

# --------------------------------------------------------------------------------------------------------------------
# Step 1: build TAMM (CMSB superbuild: GlobalArrays, HPTT, Librett, EcpInt, Eigen3, doctest, ... + TAMM itself).
# --------------------------------------------------------------------------------------------------------------------
echo "=== Building TAMM (${TAMM_COMMIT}) ==="
git clone "${TAMM_REPO}" "${BUILD_ROOT}/TAMM"
git -C "${BUILD_ROOT}/TAMM" checkout "${TAMM_COMMIT}"
# Use the default Unix Makefiles generator, not Ninja: CMSB's CMakeBuild_External sub-build invokes
# "<generator-tool> install DESTDIR=<stage>", which ninja rejects as an unknown target.
CC=gcc CXX=g++ FC=gfortran cmake -S "${BUILD_ROOT}/TAMM" -B "${BUILD_ROOT}/TAMM/build" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  "${COMMON_CMAKE_ARGS[@]}" \
  -DBUILD_LibInt2=OFF \
  -DBUILD_GauXC=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_METHODS=OFF
cmake --build "${BUILD_ROOT}/TAMM/build" -j "${JOBS}"
cmake --install "${BUILD_ROOT}/TAMM/build"

# CMSB's generated tamm-config.cmake bakes in LibInt2_ROOT (CMSBTargetConfig.cmake.in) but not GauXC_ROOT, so
# downstream consumers (ExaChem's outer configure and its nested EXACHEM_External sub-project) can't re-import
# gauxc::gauxc from the reused install without this. Patch it in if not already present.
TAMM_CFG="${INSTALL_PREFIX}/share/cmake/tamm/tamm-config.cmake"
if [ -f "${TAMM_CFG}" ] && ! grep -q 'set(GauXC_ROOT' "${TAMM_CFG}"; then
  sed -i "/^set(LibInt2_ROOT/a set(GauXC_ROOT ${CPP_DEPS_PREFIX})" "${TAMM_CFG}"
  grep -n 'set(GauXC_ROOT\|set(LibInt2_ROOT' "${TAMM_CFG}"
fi

# --------------------------------------------------------------------------------------------------------------------
# Step 2: build ExaChem, patched to compile against the reused (MPI-off) GauXC and (2.9.0) LibInt2. Same configure
# shape + same install prefix as TAMM (ExaChem find_package()s the TAMM just installed).
# --------------------------------------------------------------------------------------------------------------------
echo "=== Building ExaChem (${EXACHEM_COMMIT}) ==="
git clone "${EXACHEM_REPO}" "${BUILD_ROOT}/exachem"
git -C "${BUILD_ROOT}/exachem" checkout "${EXACHEM_COMMIT}"
git -C "${BUILD_ROOT}/exachem" apply --verbose "${SCRIPT_DIR}/patches/exachem-serial-hdf5.patch"
git -C "${BUILD_ROOT}/exachem" apply --verbose "${SCRIPT_DIR}/patches/exachem-gauxc-mpi.patch"
git -C "${BUILD_ROOT}/exachem" apply --verbose "${SCRIPT_DIR}/patches/exachem-libint2-deprecated.patch"

CC=gcc CXX=g++ FC=gfortran cmake -S "${BUILD_ROOT}/exachem" -B "${BUILD_ROOT}/exachem/build" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  "${COMMON_CMAKE_ARGS[@]}" \
  -DTAMM_EXTRA_LIBS="${HDF5_LIBS} -ldl -lm"
cmake --build "${BUILD_ROOT}/exachem/build" -j "${JOBS}"
cmake --install "${BUILD_ROOT}/exachem/build"

# --------------------------------------------------------------------------------------------------------------------
# Smoke test: binary exists and every shared library resolves.
# --------------------------------------------------------------------------------------------------------------------
test -x "${INSTALL_PREFIX}/bin/ExaChem"
missing="$(ldd "${INSTALL_PREFIX}/bin/ExaChem" | grep -i 'not found' || true)"
if [ -n "${missing}" ]; then
  echo "ERROR: unresolved shared libraries:"
  echo "${missing}"
  exit 1
fi
echo "==> Smoke test OK: ${INSTALL_PREFIX}/bin/ExaChem installed and fully linked."

if [ "${KEEP_BUILD_DIR}" != "1" ]; then
  rm -rf "${BUILD_ROOT}"
fi

echo "==> ExaChem installed to ${INSTALL_PREFIX}"

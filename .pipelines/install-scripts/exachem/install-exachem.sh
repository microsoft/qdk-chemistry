#!/usr/bin/env bash
#
# install-exachem.sh — build and install ExaChem (+ its TAMM tensor backend) for CI, run as an external MPI
# process (not linked into the qdk_chemistry wheel). Reuses BLAS++/LAPACK++/LibInt2/GauXC/spdlog/EcpInt/
# nlohmann_json/numactl already built by install-cpp-deps.sh instead of letting TAMM's CMSB superbuild rebuild
# its own copies (see patches/cmsb-fix-dependency-reuse.patch). GlobalArrays is also built here, patched via
# patches/globalarrays-fix-linalg-preference.patch.
#
# TAMM/ExaChem/CMSB/GlobalArrays commits are pinned below, not in cgmanifest.json: they're CI-only, never
# shipped in the wheel. The four pins and the five patches under patches/ are validated together as a set --
# bumping one requires re-running this script and re-checking every patch still applies.
#
# Usage: install-exachem.sh
# Required env vars: CPP_DEPS_PREFIX
# Optional env vars: INSTALL_PREFIX, BUILD_ROOT, MARCH, JOBS, KEEP_BUILD_DIR, LINALG_VENDOR, LINALG_PREFIX
#
#   LINALG_VENDOR  - CMSB's -DLINALG_VENDOR= value (default: OpenBLAS). Pass "BLIS" for the ADO wheel pipeline.
#   LINALG_PREFIX  - CMSB's -DLINALG_PREFIX= value (default: empty). Pass the BLIS+LibFLAME prefix when
#                    LINALG_VENDOR=BLIS.
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
LINALG_VENDOR="${LINALG_VENDOR:-OpenBLAS}"
LINALG_PREFIX="${LINALG_PREFIX:-}"
# MPI_PROGRESS_RANK works without an RDMA NIC on these runners; requires >= 2 MPI ranks.
GA_RUNTIME="${GA_RUNTIME:-MPI_PROGRESS_RANK}"

# Pinned to specific commits, not tags: newer TAMM/ExaChem require GCC >= 14.1 (our runners default to GCC 13).
TAMM_REPO="https://github.com/NWChemEx/TAMM.git"
TAMM_COMMIT="63c274e37c102a316e844f954bb2387988b0256c"
EXACHEM_REPO="https://github.com/ExaChem/exachem.git"
EXACHEM_COMMIT="45c192e840fd1e0417871d926e9ab87748111e53"

# CMSB (TAMM/ExaChem's superbuild helper) is pinned too: upstream TAMM doesn't pin it, and
# cmsb-fix-dependency-reuse.patch needs a fixed target to stay valid against.
CMSB_REPO="https://github.com/NWChemEx-Project/CMakeBuild.git"
CMSB_COMMIT="f5be7e2472e8ebb9bc51163d424da7c25716ce9a"

echo "==> ExaChem CI build: march=${MARCH} jobs=${JOBS} modules=${MODULES} ga_runtime=${GA_RUNTIME} linalg_vendor=${LINALG_VENDOR}"
echo "==> TAMM: ${TAMM_COMMIT} / ExaChem: ${EXACHEM_COMMIT} / CMSB: ${CMSB_COMMIT}"
echo "==> Reusing LibInt2/GauXC/BLAS++/LAPACK++ from CPP_DEPS_PREFIX=${CPP_DEPS_PREFIX}"
echo "==> INSTALL_PREFIX=${INSTALL_PREFIX}"

rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}" "${INSTALL_PREFIX}"

# CPP_DEPS_PREFIX finds BLAS++/LAPACK++ via plain find_package; INSTALL_PREFIX is added for the seeded Eigen3
# below and CMSB's own installs.
export CMAKE_PREFIX_PATH="${CPP_DEPS_PREFIX}:${INSTALL_PREFIX}:${CMAKE_PREFIX_PATH:-}"

# CMSB falls back to cloning Eigen3 from gitlab.com (unreachable in CI) if find_package fails. Seed
# INSTALL_PREFIX with apt's libeigen3-dev instead, in the layout CMSB itself would produce, so it's found
# everywhere without extra flags.
EIGEN3_CONFIG="$(find /usr -maxdepth 6 -name 'Eigen3Config.cmake' -print -quit 2>/dev/null || true)"
if [ -z "${EIGEN3_CONFIG}" ]; then
  echo "ERROR: Eigen3Config.cmake not found under /usr (expected from apt's libeigen3-dev)." >&2
  exit 1
fi
EIGEN3_CMAKE_DIR="$(dirname "${EIGEN3_CONFIG}")"                     # e.g. /usr/share/eigen3/cmake
EIGEN3_SHARE_DIR="$(dirname "${EIGEN3_CMAKE_DIR}")"                  # e.g. /usr/share/eigen3
EIGEN3_PREFIX="$(dirname "$(dirname "${EIGEN3_SHARE_DIR}")")"        # e.g. /usr
# rm -rf first: cp -r into an already-existing directory would nest instead of refreshing it.
rm -rf "${INSTALL_PREFIX}/share/eigen3/cmake" "${INSTALL_PREFIX}/include/eigen3"
mkdir -p "${INSTALL_PREFIX}/share/eigen3" "${INSTALL_PREFIX}/include"
cp -r "${EIGEN3_CMAKE_DIR}" "${INSTALL_PREFIX}/share/eigen3/cmake"
cp -r "${EIGEN3_PREFIX}/include/eigen3" "${INSTALL_PREFIX}/include/eigen3"
echo "==> Seeded Eigen3 from ${EIGEN3_PREFIX} into ${INSTALL_PREFIX}"

# CMSB's Findnumactl.cmake only searches CMAKE_INSTALL_PREFIX, so apt's libnuma-dev is invisible to it. Seed
# numa.h + libnuma.so into INSTALL_PREFIX instead (cmsb-fix-dependency-reuse.patch adds a complementary fallback).
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
# Build + install GlobalArrays ourselves, before TAMM's configure runs: CMSB's find_package(GlobalArrays QUIET)
# then detects and reuses this build instead of fetching/building its own (BUILD_GlobalArrays=OFF below fails
# loudly if detection fails). Patched via patches/globalarrays-fix-linalg-preference.patch to resolve LAPACK to
# libFLAME instead of GA's bundled ReferenceLAPACK -- see that patch for details.
# --------------------------------------------------------------------------------------------------------------------
GA_REPO="https://github.com/GlobalArrays/ga.git"
GA_COMMIT="635d6b341faf928cb5a0cddc38b1a0cbbc2b5bc4"

echo "=== Building GlobalArrays (${GA_COMMIT}) ==="
git clone "${GA_REPO}" "${BUILD_ROOT}/ga"
git -C "${BUILD_ROOT}/ga" checkout "${GA_COMMIT}"
git -C "${BUILD_ROOT}/ga" apply --verbose "${SCRIPT_DIR}/patches/globalarrays-fix-linalg-preference.patch"

GA_CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DCMAKE_C_FLAGS="-march=${MARCH}"
  -DCMAKE_CXX_FLAGS="-march=${MARCH}"
  -DCMAKE_Fortran_FLAGS="-march=${MARCH}"
  -DBUILD_SHARED_LIBS=OFF
  -DENABLE_BLAS=ON
  -DLINALG_VENDOR="${LINALG_VENDOR}"
  -DLINALG_REQUIRED_COMPONENTS=lp64
  -DGA_RUNTIME="${GA_RUNTIME}"
  -DENABLE_SYSV=OFF
  -DENABLE_PROFILING=OFF
  # GA requires LINALG_PREFIX to point at an existing directory whenever ENABLE_BLAS=ON, even when the real
  # BLAS is found via other means (e.g. GHA's OpenBLAS, where LINALG_PREFIX is otherwise left empty). Fall back
  # to INSTALL_PREFIX purely to satisfy that check.
  -DLINALG_PREFIX="${LINALG_PREFIX:-${INSTALL_PREFIX}}"
)
if [ "${LINALG_VENDOR}" = "BLIS" ]; then
  # BLIS has no LAPACK routines; prefer libFLAME over GA's bundled ReferenceLAPACK.
  GA_CMAKE_ARGS+=(-DLAPACK_PREFERENCE_LIST="FLAME;ReferenceLAPACK")
fi

CC=gcc CXX=g++ FC=gfortran cmake -S "${BUILD_ROOT}/ga" -B "${BUILD_ROOT}/ga/build" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  "${GA_CMAKE_ARGS[@]}"
cmake --build "${BUILD_ROOT}/ga/build" -j "${JOBS}"
cmake --install "${BUILD_ROOT}/ga/build"
echo "==> GlobalArrays installed to ${INSTALL_PREFIX}"

# --------------------------------------------------------------------------------------------------------------------
# Serial HDF5 discovery: prefer pkg-config, fall back to the standard Debian/Ubuntu multiarch layout.
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

# Flags shared by TAMM and ExaChem's configure below -- must be identical on both, or ExaChem's CMSB
# reconfigures/rebuilds TAMM. NJSON_ROOT must be >= 3.12.0 (ExaChem uses a private nlohmann_json API only
# present from that version on).
CMSB_SRC_DIR="${BUILD_ROOT}/CMakeBuild-patched"
mkdir -p "${CMSB_SRC_DIR}"
git -C "${CMSB_SRC_DIR}" init -q
git -C "${CMSB_SRC_DIR}" remote add origin "${CMSB_REPO}"
git -C "${CMSB_SRC_DIR}" fetch -q --depth 1 origin "${CMSB_COMMIT}"
git -C "${CMSB_SRC_DIR}" checkout -q FETCH_HEAD
git -C "${CMSB_SRC_DIR}" apply --verbose "${SCRIPT_DIR}/patches/cmsb-fix-dependency-reuse.patch"

COMMON_CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DMODULES="${MODULES}"
  -DLINALG_VENDOR="${LINALG_VENDOR}"
  -DMARCH_FLAGS="-march=${MARCH}"
  -DUSE_HDF5=OFF
  -DTAMM_CXX_FLAGS="-DUSE_SERIAL_IO ${HDF5_CFLAGS}"
  -DLibInt2_ROOT="${CPP_DEPS_PREFIX}"
  -DGauXC_ROOT="${CPP_DEPS_PREFIX}"
  -DSPDLOG_ROOT="${CPP_DEPS_PREFIX}"
  -DEcpInt_ROOT="${CPP_DEPS_PREFIX}"
  -DNJSON_ROOT="${CPP_DEPS_PREFIX}"
  -DNUMACTL_ROOT="${INSTALL_PREFIX}"
  -DFETCHCONTENT_SOURCE_DIR_CMAKEBUILD="${CMSB_SRC_DIR}"
  # -DBUILD_<dep>=OFF makes a failed reuse fail loudly at configure time instead of silently rebuilding from
  # source.
  -DBUILD_LibInt2=OFF
  -DBUILD_GauXC=OFF
  -DBUILD_SPDLOG=OFF
  -DBUILD_EcpInt=OFF
  -DBUILD_NJSON=OFF
  -DBUILD_numactl=OFF
  -DBUILD_GlobalArrays=OFF
)

# LINALG_PREFIX/LAPACK_PREFERENCE_LIST only apply for a non-default vendor (e.g. ADO's BLIS+LibFLAME); GHA's
# OpenBLAS needs neither.
if [ -n "${LINALG_PREFIX}" ]; then
  COMMON_CMAKE_ARGS+=(-DLINALG_PREFIX="${LINALG_PREFIX}")
fi
if [ "${LINALG_VENDOR}" = "BLIS" ]; then
  # BLIS has no LAPACK routines; without this, CMSB's bundled ReferenceLAPACK wins over libFLAME and both end
  # up statically linked (symbol collisions). See cmsb-fix-dependency-reuse.patch fix #5.
  COMMON_CMAKE_ARGS+=(-DLAPACK_PREFERENCE_LIST="FLAME;ReferenceLAPACK")
fi

# --------------------------------------------------------------------------------------------------------------------
# Step 1: build TAMM (CMSB superbuild). GlobalArrays/Eigen3 are reused, not built here.
# --------------------------------------------------------------------------------------------------------------------
echo "=== Building TAMM (${TAMM_COMMIT}) ==="
git clone "${TAMM_REPO}" "${BUILD_ROOT}/TAMM"
git -C "${BUILD_ROOT}/TAMM" checkout "${TAMM_COMMIT}"
# Default Unix Makefiles generator, not Ninja: CMSB's nested sub-build needs "make install DESTDIR=...".
CC=gcc CXX=g++ FC=gfortran cmake -S "${BUILD_ROOT}/TAMM" -B "${BUILD_ROOT}/TAMM/build" \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  "${COMMON_CMAKE_ARGS[@]}" \
  -DBUILD_TESTS=OFF \
  -DBUILD_METHODS=OFF
cmake --build "${BUILD_ROOT}/TAMM/build" -j "${JOBS}"
cmake --install "${BUILD_ROOT}/TAMM/build"

# tamm-config.cmake bakes in LibInt2_ROOT but not GauXC_ROOT; patch it in so ExaChem can re-import gauxc::gauxc.
TAMM_CFG="${INSTALL_PREFIX}/share/cmake/tamm/tamm-config.cmake"
if [ -f "${TAMM_CFG}" ] && ! grep -q 'set(GauXC_ROOT' "${TAMM_CFG}"; then
  sed -i "/^set(LibInt2_ROOT/a set(GauXC_ROOT ${CPP_DEPS_PREFIX})" "${TAMM_CFG}"
  grep -n 'set(GauXC_ROOT\|set(LibInt2_ROOT' "${TAMM_CFG}"
fi

# --------------------------------------------------------------------------------------------------------------------
# Step 2: build ExaChem against the just-installed TAMM, patched for the reused (MPI-off) GauXC and LibInt2.
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
# Smoke test: binary exists, every shared library resolves, and it actually runs a minimal calculation.
# --------------------------------------------------------------------------------------------------------------------
test -x "${INSTALL_PREFIX}/bin/ExaChem"
echo "==> ldd ${INSTALL_PREFIX}/bin/ExaChem:"
ldd "${INSTALL_PREFIX}/bin/ExaChem"
missing="$(ldd "${INSTALL_PREFIX}/bin/ExaChem" | grep -i 'not found' || true)"
if [ -n "${missing}" ]; then
  echo "ERROR: unresolved shared libraries:"
  echo "${missing}"
  exit 1
fi

# Minimal end-to-end run (ExaChem's own CI input): exercises the real GA/TAMM/CCSD path, not just linking.
# OMPI_ALLOW_RUN_AS_ROOT*: this may run inside a root Docker container.
SMOKE_TEST_INPUT="${BUILD_ROOT}/exachem/inputs/ci/hub_1d_6s.json"
SMOKE_TEST_DIR="$(mktemp -d)"
trap 'rm -rf "${SMOKE_TEST_DIR}"' EXIT
echo "==> Running minimal ExaChem example: ${SMOKE_TEST_INPUT}"
( cd "${SMOKE_TEST_DIR}" && OMP_NUM_THREADS=1 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    mpirun -n 2 "${INSTALL_PREFIX}/bin/ExaChem" "${SMOKE_TEST_INPUT}" )
# (cleanup of SMOKE_TEST_DIR is handled by the trap above, on every exit path -- including a failed mpirun run)

echo "==> Smoke test OK: ${INSTALL_PREFIX}/bin/ExaChem installed, fully linked, and ran a minimal example."

if [ "${KEEP_BUILD_DIR}" != "1" ]; then
  rm -rf "${BUILD_ROOT}"
fi

echo "==> ExaChem installed to ${INSTALL_PREFIX}"

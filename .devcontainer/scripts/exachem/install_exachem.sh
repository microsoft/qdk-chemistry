#!/usr/bin/env bash
#
# install_exachem.sh — build and install ExaChem (+ its TAMM tensor backend) for CI use as an external MPI process.
#
# This is the bare-metal (no container) counterpart of qaml/libs/containers/exachem.Dockerfile, adapted for
# qdk-chemistry's own GitHub Actions Linux (Ubuntu) runners:
#
#   * Open MPI comes from apt (`libopenmpi-dev` / `openmpi-bin`, installed by the workflow) rather than being
#     built from source: Ubuntu 24.04's package (Open MPI 4.1.6) already ships working Fortran bindings
#     (`mpifort`), so there is no need to build our own like the qaml/Azure-Linux container did.
#   * BLAS++ and LAPACK++ are NOT built here either: qdk-chemistry's own cpp-deps install
#     (.pipelines/install-scripts/install_cpp_dependencies.sh, run before this script into $CPP_DEPS_PREFIX)
#     already builds both for MACIS, against the same OpenBLAS this CI installs via apt (not Intel MKL, unlike
#     the qaml container, which reuses the qdk-chemistry-runtime image's MKL). TAMM finds them via its default
#     `find_package` search (no NO_DEFAULT_PATH) simply by having $CPP_DEPS_PREFIX on CMAKE_PREFIX_PATH.
#   * LibInt2 and GauXC (the two most expensive dependencies -- see the qaml technical report) are likewise NOT
#     rebuilt here. They are reused directly from qdk-chemistry's own cached C++ dependency prefix, with GauXC
#     already built GAUXC_ENABLE_MPI=OFF (exactly the configuration ExaChem's GauXC-MPI patch below expects), via
#     `-D*_ROOT` + `-DBUILD_*=OFF`. That prefix must already be populated (i.e. this script must run after
#     install_cpp_dependencies.sh) before this script is invoked.
#   * Everything else CMSB (the TAMM superbuild) would otherwise source-build itself (GlobalArrays, HPTT, Librett,
#     EcpInt, its own Eigen3, doctest, MS-GSL) is left alone, matching the container's behavior.
#   * GPU is intentionally not supported here: these are CPU-only CI runners, so there is no GPU_ARCH/CUDA handling.
#
# The only things actually built by this script are TAMM and ExaChem themselves, at the same pinned commits +
# source patches validated in the qaml ExaChem container -- see notes/exachem-technical-report.md there for the
# full rationale (GCC 13 C++20-miscompile-driven version pins, GA_RUNTIME=MPI_PROGRESS_RANK for multi-rank CCSD
# progress, serial-HDF5, LibInt2/GauXC reuse patches).
#
# Usage: install_exachem.sh
# Required env vars: CPP_DEPS_PREFIX (qdk-chemistry's own cached C++ deps prefix; provides LibInt2/GauXC/BLAS++/
#                    LAPACK++, and must already have Open MPI's mpicc/mpifort on PATH)
# Optional env vars: INSTALL_PREFIX, MARCH, JOBS, KEEP_BUILD_DIR (see defaults below)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

: "${CPP_DEPS_PREFIX:?CPP_DEPS_PREFIX must be set to qdk-chemistry's cached C++ deps prefix (provides LibInt2/GauXC/BLAS++/LAPACK++)}"

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
# GA_RUNTIME: see qaml/libs/containers/exachem.Dockerfile's ARG comment. MPI_PROGRESS_RANK is the only Global Arrays
# runtime that drives progress on a plain shm/TCP transport (no RDMA NIC on these runners), required by CCSD's many
# small one-sided tensor ops. Requires >= 2 MPI ranks (1 data-server rank + >= 1 compute rank).
GA_RUNTIME="${GA_RUNTIME:-MPI_PROGRESS_RANK}"

# Source pins -- last commits that build+run correctly with GCC 13 (see the qaml technical report §5.1: newer
# TAMM/ExaChem main requires GNU >= 14.1 for C++20 features GCC 13 miscompiles at -O2/-O3). GitHub's
# Ubuntu24.04Nightly image ships GCC 13 by default, so we reuse the exact same pins validated in that container.
TAMM_REPO="${TAMM_REPO:-https://github.com/NWChemEx/TAMM.git}"
TAMM_COMMIT="${TAMM_COMMIT:-63c274e37c102a316e844f954bb2387988b0256c}"
EXACHEM_REPO="${EXACHEM_REPO:-https://github.com/ExaChem/exachem.git}"
EXACHEM_COMMIT="${EXACHEM_COMMIT:-45c192e840fd1e0417871d926e9ab87748111e53}"

echo "==> ExaChem CI build: march=${MARCH} jobs=${JOBS} modules=${MODULES} ga_runtime=${GA_RUNTIME}"
echo "==> Reusing LibInt2/GauXC/BLAS++/LAPACK++ from CPP_DEPS_PREFIX=${CPP_DEPS_PREFIX}"
echo "==> INSTALL_PREFIX=${INSTALL_PREFIX}"

rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}" "${INSTALL_PREFIX}"

# Make sure BLAS++/LAPACK++ (built into CPP_DEPS_PREFIX by install_cpp_dependencies.sh) are visible to the
# TAMM/ExaChem configure below. They are found via TAMM's default `find_package` search (no NO_DEFAULT_PATH), so
# putting CPP_DEPS_PREFIX on CMAKE_PREFIX_PATH is sufficient -- unlike LibInt2/GauXC below, no explicit *_ROOT
# flag is needed for them.
export CMAKE_PREFIX_PATH="${CPP_DEPS_PREFIX}:${CMAKE_PREFIX_PATH:-}"

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
COMMON_CMAKE_ARGS=(
  -DCMAKE_BUILD_TYPE=Release
  -DMODULES="${MODULES}"
  -DLINALG_VENDOR=OpenBLAS
  -DMARCH_FLAGS="-march=${MARCH}"
  -DUSE_HDF5=OFF
  -DTAMM_CXX_FLAGS="-DUSE_SERIAL_IO ${HDF5_CFLAGS}"
  -DLibInt2_ROOT="${CPP_DEPS_PREFIX}"
  -DGauXC_ROOT="${CPP_DEPS_PREFIX}"
)

# --------------------------------------------------------------------------------------------------------------------
# Step 1: build TAMM (CMSB superbuild: GlobalArrays, HPTT, Librett, EcpInt, Eigen3, doctest, ... + TAMM itself).
# --------------------------------------------------------------------------------------------------------------------
echo "=== Building TAMM (${TAMM_COMMIT}) ==="
git clone "${TAMM_REPO}" "${BUILD_ROOT}/TAMM"
git -C "${BUILD_ROOT}/TAMM" checkout "${TAMM_COMMIT}"
CC=gcc CXX=g++ FC=gfortran cmake -S "${BUILD_ROOT}/TAMM" -B "${BUILD_ROOT}/TAMM/build" -GNinja \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  "${COMMON_CMAKE_ARGS[@]}" \
  -DBUILD_LibInt2=OFF \
  -DBUILD_GauXC=OFF \
  -DBUILD_TESTS=OFF \
  -DBUILD_METHODS=OFF
cmake --build "${BUILD_ROOT}/TAMM/build" -j "${JOBS}"
cmake --install "${BUILD_ROOT}/TAMM/build"

# Bake GauXC_ROOT into the generated tamm-config.cmake so every consumer (ExaChem's outer configure AND its nested
# EXACHEM_External sub-project) can re-import gauxc::gauxc from the reused install. CMSB bakes in LibInt2_ROOT
# (CMSBTargetConfig.cmake.in) but NOT GauXC_ROOT (see qaml technical report §6).
TAMM_CFG="${INSTALL_PREFIX}/share/cmake/tamm/tamm-config.cmake"
if [ -f "${TAMM_CFG}" ] && ! grep -q 'set(GauXC_ROOT' "${TAMM_CFG}"; then
  sed -i "/^set(LibInt2_ROOT/a set(GauXC_ROOT ${CPP_DEPS_PREFIX})" "${TAMM_CFG}"
  grep -n 'set(GauXC_ROOT\|set(LibInt2_ROOT' "${TAMM_CFG}"
fi

# --------------------------------------------------------------------------------------------------------------------
# Step 2: build ExaChem, patched to compile against the reused (MPI-off) GauXC and (2.9.0) LibInt2 -- same patches
# validated in qaml/libs/containers/exachem.Dockerfile. Same configure shape + same install prefix as TAMM (ExaChem
# find_package()s the TAMM just installed).
# --------------------------------------------------------------------------------------------------------------------
echo "=== Building ExaChem (${EXACHEM_COMMIT}) ==="
git clone "${EXACHEM_REPO}" "${BUILD_ROOT}/exachem"
git -C "${BUILD_ROOT}/exachem" checkout "${EXACHEM_COMMIT}"
git -C "${BUILD_ROOT}/exachem" apply --verbose "${SCRIPT_DIR}/patches/exachem-serial-hdf5.patch"
git -C "${BUILD_ROOT}/exachem" apply --verbose "${SCRIPT_DIR}/patches/exachem-gauxc-mpi.patch"
git -C "${BUILD_ROOT}/exachem" apply --verbose "${SCRIPT_DIR}/patches/exachem-libint2-deprecated.patch"

CC=gcc CXX=g++ FC=gfortran cmake -S "${BUILD_ROOT}/exachem" -B "${BUILD_ROOT}/exachem/build" -GNinja \
  -DCMAKE_INSTALL_PREFIX="${INSTALL_PREFIX}" \
  "${COMMON_CMAKE_ARGS[@]}" \
  -DTAMM_EXTRA_LIBS="${HDF5_LIBS} -ldl -lm"
cmake --build "${BUILD_ROOT}/exachem/build" -j "${JOBS}"
cmake --install "${BUILD_ROOT}/exachem/build"

# --------------------------------------------------------------------------------------------------------------------
# Smoke test: binary exists and every shared library resolves (no host-injected GPU driver libs to ignore here --
# this is a CPU-only build).
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

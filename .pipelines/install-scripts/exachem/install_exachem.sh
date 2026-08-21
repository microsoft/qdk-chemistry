#!/usr/bin/env bash
#
# install_exachem.sh — build and install ExaChem (+ its TAMM tensor backend) for CI, to run as an external MPI
# process. Reuses OpenBLAS/BLAS++/LAPACK++/LibInt2/GauXC already built into CPP_DEPS_PREFIX by
# install_cpp_dependencies.sh (TAMM finds BLAS++/LAPACK++ via its default find_package on CMAKE_PREFIX_PATH;
# LibInt2/GauXC via explicit -D*_ROOT below), and the system MPI (e.g. apt-installed openmpi-bin/libopenmpi-dev
# on Ubuntu). GPU is not supported here.
#
# Usage: install_exachem.sh <cgmanifest_path>
#   cgmanifest_path - Full path to cpp/manifest/qdk-chemistry/cgmanifest.json (source of TAMM/ExaChem commits).
# Required env vars: CPP_DEPS_PREFIX
# Optional env vars: INSTALL_PREFIX, BUILD_ROOT, MARCH, JOBS, KEEP_BUILD_DIR
#
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <cgmanifest_path>" >&2
  exit 1
fi
CGMANIFEST="$1"
if [[ ! -f "${CGMANIFEST}" ]]; then
  echo "Error: cgmanifest.json not found at ${CGMANIFEST}" >&2
  exit 1
fi

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

get_commit_hash() {
    local repo_pattern="$1"
    python3 -c "
import json
with open('${CGMANIFEST}') as f:
    data = json.load(f)
for reg in data['registrations']:
    comp = reg['component']
    if comp['type'] == 'git' and '${repo_pattern}' in comp['git'].get('repositoryUrl', ''):
        print(comp['git']['commitHash'].strip())
        break
"
}

get_repo_url() {
    local repo_pattern="$1"
    python3 -c "
import json
with open('${CGMANIFEST}') as f:
    data = json.load(f)
for reg in data['registrations']:
    comp = reg['component']
    if comp['type'] == 'git' and '${repo_pattern}' in comp['git'].get('repositoryUrl', ''):
        print(comp['git']['repositoryUrl'].strip())
        break
"
}

# TAMM/ExaChem are pinned to specific commits rather than tagged releases: newer upstream main requires GCC >= 14.1
# for C++20 features GCC 13 miscompiles at -O2/-O3, and GitHub's Ubuntu runners default to GCC 13.
TAMM_REPO="$(get_repo_url "NWChemEx/TAMM")"
TAMM_COMMIT="$(get_commit_hash "NWChemEx/TAMM")"
EXACHEM_REPO="$(get_repo_url "ExaChem/exachem")"
EXACHEM_COMMIT="$(get_commit_hash "ExaChem/exachem")"
if [[ -z "${TAMM_REPO}" || -z "${TAMM_COMMIT}" ]]; then
  echo "Error: could not find TAMM repositoryUrl/commitHash in ${CGMANIFEST}" >&2
  exit 1
fi
if [[ -z "${EXACHEM_REPO}" || -z "${EXACHEM_COMMIT}" ]]; then
  echo "Error: could not find ExaChem repositoryUrl/commitHash in ${CGMANIFEST}" >&2
  exit 1
fi

echo "==> ExaChem CI build: march=${MARCH} jobs=${JOBS} modules=${MODULES} ga_runtime=${GA_RUNTIME}"
echo "==> TAMM: ${TAMM_COMMIT} / ExaChem: ${EXACHEM_COMMIT}"
echo "==> Reusing LibInt2/GauXC/BLAS++/LAPACK++ from CPP_DEPS_PREFIX=${CPP_DEPS_PREFIX}"
echo "==> INSTALL_PREFIX=${INSTALL_PREFIX}"

rm -rf "${BUILD_ROOT}"
mkdir -p "${BUILD_ROOT}" "${INSTALL_PREFIX}"

# BLAS++/LAPACK++ are found via TAMM's default find_package search (no NO_DEFAULT_PATH), so putting
# CPP_DEPS_PREFIX on CMAKE_PREFIX_PATH is enough for them -- unlike LibInt2/GauXC below, which need explicit
# -D*_ROOT flags.
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

CC=gcc CXX=g++ FC=gfortran cmake -S "${BUILD_ROOT}/exachem" -B "${BUILD_ROOT}/exachem/build" -GNinja \
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

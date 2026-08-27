#!/usr/bin/env bash
#
# install-exachem.sh — build and install ExaChem (+ its TAMM tensor backend) for CI, to run as an external MPI
# process. Reuses OpenBLAS/BLAS++/LAPACK++/LibInt2/GauXC/spdlog/EcpInt/nlohmann_json/numactl already built into
# CPP_DEPS_PREFIX by install-cpp-deps.sh (TAMM finds BLAS++/LAPACK++ via its default find_package on
# CMAKE_PREFIX_PATH; LibInt2/GauXC/spdlog/EcpInt/nlohmann_json via explicit -D*_ROOT below; numactl via a seeded
# flat copy in INSTALL_PREFIX -- see the seeding step below for why apt's libnuma-dev can't be pointed at
# directly), and the system MPI (e.g. apt-installed openmpi-bin/libopenmpi-dev on Ubuntu). GPU is not supported
# here. TAMM's own CMSB superbuild (NWChemEx-Project/CMakeBuild) is patched (see
# patches/cmsb-fix-dependency-reuse.patch) so it also reuses the system OpenBLAS/LAPACK/spdlog/EcpInt/
# nlohmann_json/numactl instead of building its own redundant copies.
#
# TAMM/ExaChem's commits are hardcoded below (TAMM_COMMIT/EXACHEM_COMMIT), not read from
# cpp/manifest/qdk-chemistry/cgmanifest.json: unlike the deps install-cpp-deps.sh builds, TAMM/ExaChem are never
# shipped in the qdk-chemistry wheel or linked into its binary -- they're only used as an external MPI
# subprocess in GHA CI test runs -- so they aren't Component Governance-relevant and don't belong in cgmanifest.
#
# Usage: install-exachem.sh
# Required env vars: CPP_DEPS_PREFIX
# Optional env vars: INSTALL_PREFIX, BUILD_ROOT, MARCH, JOBS, KEEP_BUILD_DIR, LINALG_VENDOR, LINALG_PREFIX
#
#   LINALG_VENDOR  - CMSB's -DLINALG_VENDOR= value (default: OpenBLAS, matching GHA's apt-installed OpenBLAS).
#                    Pass "BLIS" for the ADO wheel pipeline's BLIS+LibFLAME stack. With LINALG_VENDOR=BLIS, every
#                    CMSB/TAMM/ExaChem consumer sub-build resolves LAPACK via the shared icl-utk-edu/
#                    linalg-cmake-modules ecosystem's own bundled Netlib "ReferenceLAPACK" by default (BLIS alone
#                    provides no LAPACK routines) -- see LAPACK_PREFERENCE_LIST below for why that collides with
#                    libFLAME and how it's fixed.
#   LINALG_PREFIX  - CMSB's -DLINALG_PREFIX= value (default: empty, i.e. rely on the system default search path
#                    for OpenBLAS). Pass the BLIS+LibFLAME install prefix (e.g. /usr/local) when LINALG_VENDOR=BLIS.
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
# MPI_PROGRESS_RANK is the only Global Arrays runtime that makes progress on a plain shm/TCP transport (no RDMA
# NIC on these runners); it requires >= 2 MPI ranks (1 data-server rank + >= 1 compute rank).
GA_RUNTIME="${GA_RUNTIME:-MPI_PROGRESS_RANK}"

# TAMM/ExaChem are pinned to specific commits rather than tagged releases: newer upstream main requires GCC >= 14.1
# for C++20 features GCC 13 miscompiles at -O2/-O3, and GitHub's Ubuntu runners default to GCC 13.
TAMM_REPO="https://github.com/NWChemEx/TAMM.git"
TAMM_COMMIT="63c274e37c102a316e844f954bb2387988b0256c"
EXACHEM_REPO="https://github.com/ExaChem/exachem.git"
EXACHEM_COMMIT="45c192e840fd1e0417871d926e9ab87748111e53"

# CMSB (NWChemEx-Project/CMakeBuild, TAMM/ExaChem's superbuild helper) is pinned too, for the same reproducibility
# reason as TAMM/ExaChem above: upstream TAMM's own CMakeLists.txt does not pin it either (it defaults CMSB_TAG to
# "main" if unset), so without a pin here we'd be building against whatever CMSB main happens to be on the day CI
# runs -- and our cmsb-fix-dependency-reuse.patch (below) could silently stop applying if those exact files change
# upstream. Pinned to CMSB main's HEAD as of the CI run that validated this patch end-to-end (see
# patches/cmsb-fix-dependency-reuse.patch for the patch itself).
CMSB_REPO="https://github.com/NWChemEx-Project/CMakeBuild.git"
CMSB_COMMIT="f5be7e2472e8ebb9bc51163d424da7c25716ce9a"

echo "==> ExaChem CI build: march=${MARCH} jobs=${JOBS} modules=${MODULES} ga_runtime=${GA_RUNTIME} linalg_vendor=${LINALG_VENDOR}"
echo "==> TAMM: ${TAMM_COMMIT} / ExaChem: ${EXACHEM_COMMIT} / CMSB: ${CMSB_COMMIT}"
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
# under /usr) is invisible to it too, exactly like Eigen3 above. Seed numa.h + libnuma.so into INSTALL_PREFIX
# (NOTE: pointing -DNUMACTL_ROOT at /usr directly instead, as qaml's container pipeline does, would NOT reliably
# work here -- Ubuntu's libnuma-dev installs libnuma.so under the multiarch triplet dir /usr/lib/x86_64-linux-gnu/,
# which Findnumactl.cmake's hardcoded PATH_SUFFIXES lib/lib32/lib64 never match on the *primary*, NO_DEFAULT_PATH
# search; seeding a flat copy at a path those suffixes DO match sidesteps that. patches/cmsb-fix-dependency-reuse.patch
# also gives Findnumactl.cmake a default-path *fallback* search -- which, being a plain default find_library with
# no explicit PATHS, IS multiarch-aware and would likely find the OS copy on its own -- but the seeding here is kept
# as the primary, more surgical mechanism rather than relying on that broader fallback finding the right copy).
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
# Without patches/cmsb-fix-dependency-reuse.patch, TAMM's CMSB would rebuild its own copies of system BLAS/LAPACK/
# spdlog/EcpInt/nlohmann_json instead of reusing them -- see that file for the full per-dependency root-cause
# writeup and candidate upstream fix.
#
# NJSON_ROOT specifically must point at a nlohmann_json >= 3.12.0 (built by install-cpp-deps.sh, matching
# cgmanifest.json's pin): ExaChem's own source reaches into nlohmann's private detail:: namespace
# (string_input_adapter_type), only present from 3.12.0 onward, which apt's nlohmann-json3-dev (3.11.3) predates.
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
  # Patches GlobalArrays' own cmake/ga-linalg.cmake (fetched fresh by CMSB via git clone, so not something we can
  # override via FETCHCONTENT_SOURCE_DIR the way CMSB itself is patched above) so its LAPACK_PREFERENCE_LIST
  # default becomes overridable instead of an unconditional clobber -- see patches/cmsb-fix-dependency-reuse.patch
  # fix #6. Applied unconditionally (harmless for GHA's OpenBLAS: LAPACK_PREFERENCE_LIST is left unset there, so
  # the now-conditional default still resolves to ReferenceLAPACK exactly as before).
  -DGA_LINALG_PATCH_SCRIPT="${SCRIPT_DIR}/patches/fix-ga-linalg-preference.cmake"
  # Enforce reuse instead of leaving it best-effort: CMSB's find_or_build_dependency() only hard-errors
  # ("Could not locate <dep> and user has requested we do not build one") when BUILD_<dep> is explicitly set to
  # OFF; if left unset, a failed find silently falls through to include(Build<dep>) and rebuilds from source
  # (cmake/macros/DependencyMacros.cmake). Passing -DBUILD_<dep>=OFF for all six reused deps -- not just
  # LibInt2/GauXC as before -- makes a broken *_ROOT/reuse patch fail loudly at configure time instead of
  # silently degrading into a redundant rebuild that's only visible by eyeballing the configure log.
  -DBUILD_LibInt2=OFF
  -DBUILD_GauXC=OFF
  -DBUILD_SPDLOG=OFF
  -DBUILD_EcpInt=OFF
  -DBUILD_NJSON=OFF
  -DBUILD_numactl=OFF
)

# LINALG_PREFIX/LAPACK_PREFERENCE_LIST are appended conditionally rather than baked into the array literal above,
# since they only apply for a non-default LINALG_VENDOR (e.g. the ADO wheel pipeline's BLIS+LibFLAME stack; GHA's
# default OpenBLAS needs neither -- apt's OpenBLAS is a single combined library providing both BLAS and LAPACK,
# so CMSB's own stock find_package(BLAS/LAPACK) fallback (patches/cmsb-fix-dependency-reuse.patch fix #1) finds
# it directly, without any vendor/preference hint).
if [ -n "${LINALG_PREFIX}" ]; then
  COMMON_CMAKE_ARGS+=(-DLINALG_PREFIX="${LINALG_PREFIX}")
fi
if [ "${LINALG_VENDOR}" = "BLIS" ]; then
  # BLIS has no LAPACK routines, so every CMSB consumer sub-build falls back to CMSB's own bundled Netlib
  # ReferenceLAPACK -- which wins over libFLAME (our actual LAPACK provider alongside BLIS) by default list
  # order, and both end up statically linked into the same executable ("multiple definition of `dsytrd_'" etc).
  # Prefer FLAME explicitly instead; see patches/cmsb-fix-dependency-reuse.patch fix #5 for the full mechanism.
  # Not using -DBLA_VENDOR=FLAME here: it feeds BLAS_PREFERENCE_LIST too, which would risk swapping away our
  # already-working BLIS BLAS resolution. No *_PREFIX hint needed: libFLAME shares BLIS's prefix (LINALG_PREFIX).
  COMMON_CMAKE_ARGS+=(-DLAPACK_PREFERENCE_LIST="FLAME;ReferenceLAPACK")
fi

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

# Minimal end-to-end run: a 6-site Hubbard SCF+CCSD input (ExaChem's own CI uses the same file, from the source
# tree already cloned above at ${BUILD_ROOT}/exachem) -- small enough to run in seconds, but exercises the real
# GA/TAMM/CCSD path, not just linking. GA_RUNTIME=MPI_PROGRESS_RANK (the default above) needs >= 2 MPI ranks.
# OMPI_ALLOW_RUN_AS_ROOT*: this script may run inside a root Docker container (e.g. the ADO pipeline).
SMOKE_TEST_INPUT="${BUILD_ROOT}/exachem/inputs/ci/hub_1d_6s.json"
SMOKE_TEST_DIR="$(mktemp -d)"
echo "==> Running minimal ExaChem example: ${SMOKE_TEST_INPUT}"
( cd "${SMOKE_TEST_DIR}" && OMP_NUM_THREADS=1 OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    mpirun -n 2 "${INSTALL_PREFIX}/bin/ExaChem" "${SMOKE_TEST_INPUT}" )
rm -rf "${SMOKE_TEST_DIR}"

echo "==> Smoke test OK: ${INSTALL_PREFIX}/bin/ExaChem installed, fully linked, and ran a minimal example."

if [ "${KEEP_BUILD_DIR}" != "1" ]; then
  rm -rf "${BUILD_ROOT}"
fi

echo "==> ExaChem installed to ${INSTALL_PREFIX}"

<!--
  ~ MACIS Copyright (c) 2023, The Regents of the University of California,
  ~ through Lawrence Berkeley National Laboratory (subject to receipt of
  ~ any required approvals from the U.S. Dept. of Energy). All rights reserved.
  ~
  ~ See LICENSE.txt for details
-->

# About

Many-Body Adaptive Configuration Interaction Suite (MACIS) Copyright (c) 2023, The Regents of the University of
California, through Lawrence Berkeley National Laboratory (subject to receipt of any required approvals from the U.S.
Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software, please contact Berkeley Lab's Intellectual
Property Office at <IPO@lbl.gov>.

NOTICE. This Software was developed under funding from the U.S. Department of Energy and the U.S. Government
consequently retains certain rights. As such, the U.S. Government has been granted for itself and others acting on its
behalf a paid-up, nonexclusive, irrevocable, worldwide license in the Software to reproduce, distribute copies to the
public, prepare derivative works, and perform publicly and display publicly, and to permit others to do so.

## Synopsis

The Many-Body Adaptive Configuration Interaction Suite (MACIS) is a modern, modular C++ library for high-performance
quantum many-body methods based on configuration interaction (CI). MACIS currently provides a reuseable and extentible
interface for the development of full CI (FCI), complete active-space (CAS) and selected-CI (sCI) methods for quantum
chemistry. Efforts have primarily been focused on the development of distributed memory variants of the adaptive
sampling CI (ASCI) method on CPU architectures, and work is underway to extend the functionality set to other methods
commonly encountered in quantum chemistry and to accelerator architectures targeting exascale platforms.

MACIS is a work in progress. Its development has been funded by the Computational Chemical Sciences (CCS) program of the
U.S. Department of Energy Office of Science, Office of Basic Energy Science (BES). It was originally developed under the
Scalable Predictive Methods for Excitations and Correlated Phenomena [(SPEC)](https://spec.labworks.org/home) Center.

### Main Contributors

- David Williams-Young (LBNL): dbwy [at] lbl [dot] gov
- Carlos Mejuto Zaera (SISSA)
- Norm Tubman (NASA)

## Dependencies

- CMake (3.14+)
- BLAS / LAPACK
- [BLAS++](https://github.com/icl-utk-edu/blaspp) / [LAPACK++](https://github.com/icl-utk-edu/lapackpp)
- [`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) with [Kokkos](https://github.com/kokkos/mdspan)
  extensions
- spdlog
- MPI (Optional)
- OpenMP (Optional)
- Boost (Optional)
- Catch2 (Testing)

## Publications

Please cite the following publications if MACIS was used in your publication or software:

```latex
% Distributed Memory ASCI Implementation
@article{williams23_distributed_asci,
    title={A parallel, distributed memory implementation of the adaptive
           sampling configuration interaction method},
    author={David B. Williams-Young and Norm M. Tubman and Carlos Mejuto-Zaera
            and Wibe A. de Jong},
    journal={The Journal of Chemical Physics},
    volume={158},
    pages={214109},
    year={2023},
    doi={10.1063/5.0148650},
    preprint={https://arxiv.org/abs/2303.05688},
    url={https://pubs.aip.org/aip/jcp/article/158/21/214109/2893713/A-parallel-distributed-memory-implementation-of}
}

```

## Build Instructions

MACIS provides a CMake build system with automatic dependency management (through
[FetchContent](https://cmake.org/cmake/help/latest/module/FetchContent.html)). As such, a simple CMake invocation will
often suffice for most purposes

```bash
cmake -S /path/to/macis -B /path/to/build [MACIS configure options]
cmake --build /path/to/build
```

Tests can be run as `OMP_NUM_THREADS=<ncores> make -C /path/to/build test`

Important things:

1. Build with `CMAKE_BUILD_TYPE=Release` - this adds `-O3` + other goodies, *you want this*
1. Build with `CMAKE_CXX_FLAGS="-march=native"` - this tells MACIS to use arch-specific flags, which will enable
   vectorization, FMA, etc.

### Windows Subsystem for Linux (WS) Build Instructions

The following APT packages should be installed on the WSL instance before running CMake:

```bash
sudo apt-get install openmpi-bin openmpi-common libopenmpi-dev libeigen3-dev libboost-all-dev liblapacke-dev
```

### (Microsoft) Pilgrim Build Instructions

On the 96-core (4 GPU) Pilgrim nodes (`prm96c4g`), you can use a pre-generated apptainer container

```bash
module load developers/madft/x86_64/cuda-12.1/latest
inside_clean_shell # This will spin up a local shell inside the container with a clean environment
```

### Linking to MACIS

MACIS is linkable both as an installed library as well as a CMake subproject via `FetchContent`

```cmake
# MACIS Discovery
find_package( macis REQUIRED )
target_link_libraries( my_target PUBLIC macis::macis )
```

```cmake
# MACIS as CMake Subproject
include(FetchContent)

# Set MACIS CMake options (see below)

# Pull master branch of MACIS
FetchContent_Declare( macis
  GIT_REPOSITORY https://github/com/wavefunction91/MACIS.git
  GIT_TAG master
)
FetchContent_MakeAvailable( macis )

# Link to target
target_link_libraries( my_target PUBLIC macis::macis )
```

### Influential CMake Variables

| Variable Name | Description | Default |
|----------------------------|-----------------------------------------------------------|----------|
| `MACIS_ENABLE_OPENMP` | Enable OpenMP Bindings | `ON` |
| `MACIS_ENABLE_MPI` | Enable MPI Bindings | `ON` |
|`MACIS_ENABLE_BOOST` | Enable Boost Bindings | `ON` |
| `BLAS_LIBRARIES` | Full BLAS linker. | -- |
| `LAPACK_LIBRARIES` | Full LAPACK linker. | -- |
| `BUILD_TESTING` | Whether to build unit tests | `ON` |

## Example Usage

The main standalone MACIS driver will appear as `build/tests/standalone_driver` when the build is complete.

### MACIS Input File

Example input files for FCI and SCI runs of MACIS can be found in `examples`.

#### The `CI` Section

This section outlines the physical parameters for the calculation

| Variable Name | Description | Default |
|-------------------|--------------------------------------|------------------| | `NALPHA` | Number of Alpha electrons |
-- | | `NBETA` | Number of Beta electrons | -- | | `REF_DATA_FORMAT` | Format of the Hamiltonian data | -- | |
`REF_DATA_FILE` | Hamiltonian data file (e.g. FCIDUMP) | -- | | `JOB` | Type of job (`CI` / `MCSCF`) | `MCSCF` | |
`EXPANSION` | CI expansion (`CAS` / `ASCI`) | `CAS` | | `NINACTIVE` | Number of inactive orbitals | 0 | | `NACTIVE` |
Number of active orbitals | total - ninactive | | `RDMFILE` | Output file for RDM tensors | -- | | `FCIDUMP_OUT` |
Output file for FCIDUMP | -- | | `WFN_OUT_FILE` | Output file for wavefunction | -- |

#### The `MCSCF` Section

This section outlines solver parameters (poorly named....)

| Variable Name | Description | Default |
|-------------------|---------------------------------------------|------------------| | `MAX_MACRO_ITER` | Max macro
(CI solve) iterations for MCSCF | See code | | `MAX_ORB_STEP` | Max length of a MCSCF orb rotation step | See code | |
`MCSCF_ORB_TOL` | Tolerance on orbital gradient for MCSCF | See code | | `ENABLE_DIIS` | Whether to use DISS in orbital
optimization | True | | `DIIS_START_ITER` | Macro iteration start for DIIS | 1 | | `DIIS_NKEEP` | How many error vectors
to keep for DIIS | See code | | `CI_RES_TOL` | Davidson residual tolerance | 1e-8 | | `CI_MAX_SUB` | Maximum Davidson
subspace size | 20 | | `CI_MATEL_TOL` | Tolerance on CI matrix elements | 2e-16 |

#### The `ASCI` Section

This section outlines ASCI (sCI) parameters

| Variable Name | Description | Default |
|-------------------|------------------------------------------------|------------------| | `NTDETS_MAX` | Max size of
sCI wave function | See code | | `NTDETS_MIN` | Min size of sCI wave function | See code | | `NCDETS_MAX` | Max size of
sCI "core space" | See code | | `HAM_EL_TOL` | Tolerance on kept matrix elements in search | See code | | `RV_PRUNE_TOL`
| ASCI pair value prune tolerance | See code | | `PAIR_MAX_LIM` | Max number of ASCI pairs to keep locally | See code |
| `GROW_FACTOR` | Factor by which to grow the wfn in growth phase | 8 | | `MAX_REFINE_ITER` | Max iterations for
refinement phase | See code | | `REFINE_ETOL` | Energy tolerance for refinement phase | 1e-3 | | `GROW_WITH_ROT` |
Perform natural orb rotations during growth | FALSE | | `ROT_SIZE_START` | Size of wfn to start nat orb rotataions | See
code | | `CONSTRAINT_LVL` | Max bitstring constraint used for ASCI search | See code | | `WFN_FILE` | File to read
starting wfn from | -- | | `E0_WFN` | E0 associated with `WFN_FILE` (skips eval) | -- | | `E0_WFN_DIAG` | Explicitly
diag in basis from `WFN_FILE` | -- | | `HAM_OUT_FILE` | Output file for sCI Hamiltonian | -- |

Have a question, comment or concern? Open an [Issue](https://github.com/wavefunction91/MACIS/issues) or start a
[Discussion](https://github.com/wavefunction91/MACIS/discussions).

## License

MACIS is made freely available under the terms of the LBNL modified 3-Clause BSD license. See LICENSE.txt for details.

## Acknowledgments

The development of MACIS has ben supported by the Center for Scalable Predictive methods for Excitations and Correlated
phenomena (SPEC), which is funded by the U.S. Department of Energy (DoE), Office of Science, Office of Basic Energy
Sciences, Division of Chemical Sciences, Geosciences and Biosciences as part of the Computational Chemical Sciences
(CCS) program at Lawrence Berkeley National Laboratory under FWP 12553.

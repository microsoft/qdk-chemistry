# LOBPCGXX

LOBPCGXX is a C++ implementation of the Locally Optimal Block Preconditioned Conjugate Gradient (LOBPCG) method for
solving large symmetric eigenvalue problems. It is based on the original LOBPCG algorithm by Knyazev, with additional
features and improvements.

## Method References

- Knyazev, A.V. (2001). "Toward the optimal preconditioned eigensolver: Locally optimal block preconditioned conjugate
  gradient method." SIAM Journal on Scientific Computing, 23(2), 517-541.
- Hetmaniuk, U., & Lehoucq, R. B. (2006). "Basis selection in LOBPCG." Journal of Computational Physics, 218(1),
  324-332.
- Duersch, J.A., Shao, M., Yang, S., & Gu, M. (2018) "A Robust and Efficient Implementation of LOBPCG" SIAM Journal on
  Scientific Computing, 40(1), C1-C23.

## Install Deps

LOBPCGXX depends on two libraries: blaspp and lapackpp from the ICL.

These may be obtained:

```bash
hg clone https://bitbucket.org/icl/blaspp
hg clone https://bitbucket.org/icl/lapackpp
```

and built

```bash
mkdir -p $PWD/build_icl/{blas,lapack}pp

cmake -Hblaspp -Bbuild_icp/blaspp -DBLASPP_BUILD_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$PWD/install_icl \
      -DCMAKE_PREFIX_PATH=$PWD/install_icl

make -C build_icl/blaspp -j <cores> install

cmake -Hlapackpp -Bbuild_icp/lapackpp -DBUILD_LAPACKPP_TESTS=OFF \
      -DCMAKE_INSTALL_PREFIX=$PWD/install_icl \
      -DCMAKE_PREFIX_PATH=$PWD/install_icl

make -C build_icl/lapackpp -j <cores> install
```

## Build LOBPCGXX

```bash
cmake -H/path/to/lobpcgxx -Bbuild_lobpcgxx \
      -DCMAKE_PREFIX_PATH=/path/to/icl/install \
      -DCMAKE_INSTALL_PREFIX=/path/to/where/you/want/it/installed

make -C build_lobpcgxx -j <cores> install
```

## Link to LOBPCGXX

Currently, the easiest way to link to LOBPCGXX is through CMake

```cmake
# CMakeLists.txt

find_package( lobpcgxx )
target_link_libraries( my_target PUBLIC lobpcgxx::lobpcgxx )
```

Ensure that `CMAKE_PREFIX_PATH` contains the LOBPCGXX install directory (and the ICL install directory) on CMake
invocation

## Testing LOBPCGXX

To run a test (Laplacian) problem

```bash
/path/to/lobpcg_build/lobpcg_tester
```

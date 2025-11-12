// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/scf.h>

#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cublas_utils.h>
#include <qdk/chemistry/scf/util/gpu/cusolver_utils.h>
#include <qdk/chemistry/scf/util/gpu/cutensor_utils.h>
#endif

#include <libint2/basis.h>
namespace qdk::chemistry::scf {

/**
 * @brief Base class for density fitting (DF) electron repulsion integrals
 *
 * Handles metric generation and solution for (Q|P)^{-1} operations.
 * Supports both CPU and GPU backends.
 */
class DensityFittingBase {
 protected:
  bool unrestricted_;     ///< Whether calculation is spin unrestricted
  uint64_t n_atoms_;      ///< Number of atoms in the molecule
  BasisMode basis_mode_;  ///< Basis function convention
  ParallelConfig mpi_;    ///< MPI configuration

  libint2::BasisSet obs_;  ///< Orbital basis set in libint2 format
  libint2::BasisSet abs_;  ///< Auxiliary basis set in libint2 format

  std::vector<int> obs_sh2atom_;  ///< Orbital basis shell to atom mapping
  std::vector<int> abs_sh2atom_;  ///< Auxiliary basis shell to atom mapping

  bool gpu_;                            ///< Whether to use GPU acceleration
  std::unique_ptr<double[]> h_metric_;  ///< Host storage for DF metric (Q|P)
  std::unique_ptr<int[]>
      h_metric_ipiv_;  ///< Pivot indices for LU factorization

#ifdef QDK_CHEMISTRY_ENABLE_GPU
  double* d_metric_ = nullptr;  ///< Device pointer to DF metric
  std::unique_ptr<cublas::ManagedcuBlasHandle>
      cublasHandle_;  ///< cuBLAS handle
  std::unique_ptr<cusolver::ManagedcuSolverHandle>
      cusolverHandle_;  ///< cuSOLVER handle
#endif

  /**
   * @brief Generate the density fitting metric (Q|P)
   */
  void generate_metric();

  /**
   * @brief Solve metric system on GPU
   * @param X Right-hand side / solution matrix
   * @param LDX Leading dimension of X
   */
  void solve_metric_system_device(double* X, size_t LDX);

  /**
   * @brief Solve metric system on CPU
   * @param X Right-hand side / solution matrix
   * @param LDX Leading dimension of X
   */
  void solve_metric_system_host(double* X, size_t LDX);

 public:
  /**
   * @brief Construct density fitting base
   * @param unr Whether calculation is unrestricted
   * @param obs Orbital basis set
   * @param abs Auxiliary basis set
   * @param mpi MPI configuration
   * @param gpu Whether to use GPU acceleration
   */
  DensityFittingBase(bool unr, const BasisSet& obs, const BasisSet& abs,
                     ParallelConfig mpi, bool gpu);

  virtual ~DensityFittingBase() = default;

  /**
   * @brief Check if GPU acceleration is enabled
   * @return true if using GPU
   */
  inline auto gpu() const { return gpu_; }
};
}  // namespace qdk::chemistry::scf

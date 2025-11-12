// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <libint2/basis.h>
#include <qdk/chemistry/scf/config.h>

#include "eri/eri_df_base.h"
#include "incore.h"
#ifdef QDK_CHEMISTRY_ENABLE_GPU
#include <qdk/chemistry/scf/util/gpu/cublas_utils.h>
#include <qdk/chemistry/scf/util/gpu/cusolver_utils.h>
#include <qdk/chemistry/scf/util/gpu/cutensor_utils.h>
#endif

namespace qdk::chemistry::scf::incore {

/**
 * @brief Implementation class for conventional 4-center ERI with in-memory
 * storage.
 *
 * @see ERIINCORE for public API documentation
 */
class ERI {
  /// Whether this is an unrestricted (UHF/UKS) spin calculation
  bool unrestricted_;

  /// Basis function normalization convention (PSI4 vs. default)
  BasisMode basis_mode_;

  /// MPI parallelization configuration for distributed integral storage
  ParallelConfig mpi_;

  /// Orbital basis set in libint2 format for integral engine
  libint2::BasisSet obs_;

  /// Host memory storage for 4-center ERIs (N⁴ doubles, may be NULL on GPU-only
  /// builds)
  std::unique_ptr<double[]> h_eri_;

  /// Host memory for erf-attenuated ERIs (NULL if omega=0, N⁴ doubles
  /// otherwise)
  std::unique_ptr<double[]> h_eri_erf_;

  /// Range-separation parameter ω (0.0 for standard Coulomb)
  double omega_;

  // MPI distribution: Each rank stores integral shells in range [loc_i_st_,
  // loc_i_en_)

  /// Starting shell index for local MPI rank storage
  size_t loc_i_st_;

  /// Ending shell index (exclusive) for local MPI rank storage
  size_t loc_i_en_;

  // GPU-accelerated tensor contraction infrastructure
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  /// Device pointer to ERI tensor on GPU (N⁴ doubles)
  double* d_eri_ = nullptr;

  /// Device pointer to erf-attenuated ERI tensor (NULL if omega=0)
  double* d_eri_erf_ = nullptr;

  /// cuTENSOR library context handle for tensor operations
  std::shared_ptr<cutensor::TensorHandle> handle_;

  /// Tensor descriptor for rank-4 ERI tensor with shape (N,N,N,N)
  std::shared_ptr<cutensor::TensorDesc> descERI_;

  /// Tensor descriptor for full rank-2 matrices with shape (N,N)
  std::shared_ptr<cutensor::TensorDesc> descR_;

  /// Tensor descriptor for MPI-local rank-2 matrix tiles with shape (X,N) where
  /// X = local rows
  std::shared_ptr<cutensor::TensorDesc> descP_;

  /// Pre-planned cuTENSOR contraction for Coulomb matrix: J[μν] = Σ_λσ (μν|λσ)
  /// P[λσ]
  std::unique_ptr<cutensor::ContractionData> couContraction_;

  /// Pre-planned cuTENSOR contraction for exchange matrix: K[μν] = Σ_λσ (μλ|νσ)
  /// P[λσ]
  std::unique_ptr<cutensor::ContractionData> exxContraction_;
#endif

  /**
   * @brief Compute and store all 4-center electron repulsion integrals
   *
   * Generates the complete set of (μν|λσ) integrals using Libint2 integral
   * engine and stores them in memory (host and/or device depending on
   * configuration).
   *
   * For range-separated functionals (ω ≠ 0):
   * - Computes both standard and erf-attenuated integrals
   * - Stores in separate arrays (h_eri_ and h_eri_erf_)
   *
   * MPI distribution:
   * - Rank k stores shells in range [loc_i_st_, loc_i_en_)
   * - Load balancing based on estimated integral counts
   *
   * @note This is called automatically during construction
   * @note Extremely memory intensive: O(N⁴) storage requirement
   */
  void generate_eri_();

 public:
  /**
   * @brief Construct in-core conventional ERI calculator implementation
   * @see ERIINCORE for public API documentation
   */
  ERI(bool unrestricted, const BasisSet& basis, ParallelConfig mpi,
      double omega);

  /**
   * @brief Build Coulomb (J) and exchange (K) matrices from density matrix
   * @see ERIINCORE for public API documentation
   */
  void build_JK(const double* P, double* J, double* K, double alpha,
                double beta, double omega);

  /**
   * @brief Compute nuclear gradients of Coulomb and exchange contributions
   * @see ERIINCORE for public API documentation
   */
  void get_gradients(const double* P, double* dJ, double* dK, double alpha,
                     double beta, double omega);

  /**
   * @brief Perform first quarter transformation to molecular orbital basis
   * @see ERIINCORE for public API documentation
   */
  void quarter_trans(size_t nt, const double* C, double* out);

  /**
   * @brief Destructor
   */
  ~ERI() noexcept;

  /**
   * @brief Factory method to create in-core ERI implementation
   *
   * Static factory function that constructs and returns a fully initialized
   * ERI implementation object. Provides a convenient interface that matches
   * the public API expectations.
   *
   * @param unrestricted Whether this is an unrestricted calculation
   * @param basis Orbital basis set
   * @param mpi MPI configuration
   * @param omega Range-separation parameter in bohr⁻¹
   * @return Unique pointer to newly created ERI implementation
   *
   * @note Prefer using this factory over direct construction
   */
  static std::unique_ptr<ERI> make_incore_eri(bool unrestricted,
                                              const BasisSet& basis,
                                              ParallelConfig mpi, double omega);
};

/**
 * @brief Implementation class for density-fitted 3-center ERI with in-memory
 * storage
 * @see ERIINCORE_DF for public API documentation
 */
class ERI_DF : public DensityFittingBase {
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  /// Device pointer to 3-center integrals (Q|μν) on GPU (N²M doubles)
  double* d_eri_ = nullptr;
#endif

  /// Host memory storage for 3-center integrals (N²M doubles, may be NULL on
  /// GPU-only builds)
  std::unique_ptr<double[]> h_eri_;

  // MPI distribution: Each rank stores integral shells in range [loc_i_st_,
  // loc_i_en_)

  /// Starting shell index for local MPI rank storage
  size_t loc_i_st_;

  /// Ending shell index (exclusive) for local MPI rank storage
  size_t loc_i_en_;

  /**
   * @brief Compute and store all 3-center electron repulsion integrals
   *
   * Generates the complete set of (Q|μν) integrals using Libint2 integral
   * engine with auxiliary and orbital basis sets, storing them in memory (host
   * and/or device depending on configuration).
   *
   * Storage layout:
   * - Linear array of size N_aux × N_orb × N_orb
   * - Indexed as: h_eri_[Q + N_aux*(μ + N_orb*ν)]
   *
   * MPI distribution:
   * - Rank k stores shells in range [loc_i_st_, loc_i_en_)
   * - Load balancing based on estimated integral counts
   *
   * @note This is called automatically during construction
   * @note Memory requirement: O(N²M) where M = auxiliary basis size ≈ 3N
   * @note Much more memory efficient than 4-center storage: O(N⁴)
   * @note Computational cost: O(N²M) vs O(N⁴) for conventional
   */
  void generate_eri_();

 public:
  /**
   * @brief Construct in-core density-fitted ERI calculator implementation
   * @see ERIINCORE_DF for public API documentation
   */
  ERI_DF(bool unr, const BasisSet& obs, const BasisSet& abs,
         ParallelConfig _mpi);

  /**
   * @brief Build approximate Coulomb (J) matrix using density fitting
   * @see ERIINCORE_DF for public API documentation
   */
  void build_JK(const double* P, double* J, double* K, double alpha,
                double beta, double omega);

  /**
   * @brief Compute nuclear gradients of density-fitted Coulomb contribution
   * @see ERIINCORE_DF for public API documentation
   */
  void get_gradients(const double* P, double* dJ, double* dK, double alpha,
                     double beta, double omega);

  /**
   * @brief Perform first quarter transformation with auxiliary basis
   * @see ERIINCORE_DF for public API documentation
   */
  void quarter_trans(size_t nt, const double* C, double* out);

  /**
   * @brief Destructor
   */
  ~ERI_DF() noexcept;

  /**
   * @brief Factory method to create in-core DF-ERI implementation
   *
   * Static factory function that constructs and returns a fully initialized
   * DF-ERI implementation object. Provides a convenient interface that matches
   * the public API expectations.
   *
   * @param unrestricted Whether this is an unrestricted calculation
   * @param obs Orbital basis set (primary basis)
   * @param abs Auxiliary basis set for density fitting
   * @param mpi MPI configuration
   * @return Unique pointer to newly created ERI_DF implementation
   *
   * @note Prefer using this factory over direct construction
   */
  static std::unique_ptr<ERI_DF> make_incore_eri(bool unrestricted,
                                                 const BasisSet& obs,
                                                 const BasisSet& abs,
                                                 ParallelConfig mpi);
};

}  // namespace qdk::chemistry::scf::incore

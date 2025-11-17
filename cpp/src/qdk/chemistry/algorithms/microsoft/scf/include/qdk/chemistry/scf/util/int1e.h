// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/core/types.h>

#include <Eigen/Dense>
#include <functional>
#include <vector>
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
#include <qdk/chemistry/scf/core/qmmm.h>
#endif
#include <qdk/chemistry/scf/core/scf.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <libecpint.hpp>

namespace qdk::chemistry::scf {
/**
 * @brief Abstract base class for one-body integral engines
 *
 * Provides a uniform interface for different types of one-electron integral
 * engines (overlap, kinetic, nuclear attraction, dipole, etc.). Derived
 * classes implement specific integral types using libint2 or other libraries.
 */
class OneBodyIntegralEngine {
 public:
  /**
   * @brief Virtual destructor for proper cleanup of derived classes
   */
  virtual ~OneBodyIntegralEngine() = default;

  /**
   * @brief Compute integrals for a shell pair
   *
   * Evaluates integrals between basis functions in two shells. The returned
   * pointers point to internal buffers that are valid until the next call.
   *
   * @param shellA First shell index (bra)
   * @param shellB Second shell index (ket)
   * @return std::vector<const double*> Vector of pointers to integral buffers,
   *         one per operator component (e.g., 3 for dipole x,y,z)
   */
  virtual std::vector<const double*> compute(int shellA, int shellB) = 0;

  /**
   * @brief Get number of operator components
   *
   * Returns the number of integral operators computed by this engine.
   * For example: 1 for overlap/kinetic, 3 for dipole (x,y,z),
   * 6 for quadrupole (xx,xy,xz,yy,yz,zz).
   *
   * @return size_t Number of integral operators
   */
  virtual size_t nopers() const = 0;
};

/**
 * @brief One-electron integral calculator
 *
 * Computes various one-electron integrals and their nuclear gradients for
 * quantum chemistry calculations. Supports:
 * - Overlap integrals: ⟨μ|ν⟩
 * - Kinetic energy: ⟨μ|-½∇²|ν⟩
 * - Nuclear attraction: ⟨μ|-Σ_A Z_A/|r-R_A||ν⟩
 * - Dipole integrals: ⟨μ|r|ν⟩
 * - Effective core potentials (ECPs)
 * - Point charge interactions (QM/MM)
 * - Nuclear gradients for all integral types
 *
 * Uses libint2 for standard integrals and libecpint for ECP evaluation.
 */
class OneBodyIntegral {
 public:
  using EngineFactory = std::function<
      std::unique_ptr<OneBodyIntegralEngine>()>;  ///< Factory function type for
                                                  ///< creating integral engines
  using AtomCenterFn =
      std::function<std::vector<std::pair<int /*atom*/, int /*buffer_idx*/>>(
          int,
          int)>;  ///< Function type for mapping shell pairs to atomic centers

  /**
   * @brief Construct one-electron integral calculator
   *
   * Initializes integral calculator with basis set and molecular information.
   *
   * @param basis_set Basis set defining the atomic orbital basis
   * @param mol Molecular structure with atomic positions and charges
   * @param mpi MPI configuration for parallel integral evaluation
   */
  OneBodyIntegral(const BasisSet* basis_set, const Molecule* mol,
                  ParallelConfig mpi);

  /**
   * @brief Destructor - cleans up libint2 and libecpint resources
   */
  ~OneBodyIntegral();

  /**
   * @brief Compute significant shell pairs based on overlap threshold
   *
   * @param shells Vector of basis function shells from basis set
   * @param threshold Screening threshold for Schwarz inequality (default 1e-12)
   * @return std::vector<std::pair<int, int>> List of (shellA, shellB) pairs to
   * compute
   */
  static std::vector<std::pair<int, int>> compute_shell_pairs(
      const std::vector<Shell>& shells, const double threshold = 1e-12);

  /**
   * @brief Compute overlap integral matrix S[μν] = ⟨μ|ν⟩
   *
   * Evaluates the overlap between all basis function pairs. Required
   * for orthogonalization and solving the generalized eigenvalue problem.
   *
   * @param[out] res Output buffer for overlap matrix (size: num_basis_funcs ×
   * num_basis_funcs)
   */
  void overlap_integral(double* res);

  /**
   * @brief Compute kinetic energy integral matrix T[μν] = ⟨μ|-½∇²|ν⟩
   *
   * Evaluates the kinetic energy operator integrals. Combined with
   * nuclear attraction to form the core Hamiltonian.
   *
   * @param[out] res Output buffer for kinetic matrix (size: num_basis_funcs ×
   * num_basis_funcs)
   */
  void kinetic_integral(double* res);

  /**
   * @brief Compute nuclear attraction integral matrix
   * V[μν] = ⟨μ|-Σ_A * Z_A/|r-R_A||ν⟩
   *
   * Evaluates the electron-nuclear attraction integrals for all nuclei.
   *
   * @param[out] res Output buffer for nuclear attraction matrix (size:
   * num_basis_funcs × num_basis_funcs)
   */
  void nuclear_integral(double* res);

  /**
   * @brief Compute dipole integral matrices μ_i[μν] = ⟨μ|(r_i - cen_i)|ν⟩
   *
   * Evaluates the electric dipole integrals in x, y, z directions relative
   * to a specified center.
   *
   * @param[out] res Output buffer for 3 dipole matrices (size: 3 ×
   * num_basis_funcs × num_basis_funcs)
   * @param cen Center of dipole expansion in Cartesian coordinates (default:
   * origin)
   */
  void dipole_integral(double* res, std::array<double, 3> cen = {0, 0, 0});

  /**
   * @brief Compute quadrupole integral matrices μ_i[μν] = ⟨μ|(r_i - cen_i)^2|ν⟩
   *
   * Evaluates the electric quadrupole integrals in x, y, z directions relative
   * to a specified center.
   *
   * @param[out] res Output buffer for 6 quadrupole matrices
   *                 order: xx, xy, xz, yy, yz, zz  (size: 6 × num_basis_funcs ×
   * num_basis_funcs)
   * @param cen Center of quadrupole expansion in Cartesian coordinates
   * (default: origin)
   */
  void quadrupole_integral(double* res, std::array<double, 3> cen = {0, 0, 0});

  /**
   * @brief Compute effective core potential integral matrix
   *
   * Uses libecpint library for ECP evaluation.
   *
   * @param[out] res Output buffer for ECP matrix (size: num_basis_funcs ×
   * num_basis_funcs)
   */
  void ecp_integral(double* res);

#ifdef QDK_CHEMISTRY_ENABLE_QMMM
  /**
   * @brief Compute point charge interaction integral matrix for QM/MM
   *
   * Evaluates the interaction between quantum mechanical electrons and
   * classical point charges in QM/MM calculations:
   * V_QM/MM[μν] = ⟨μ|-Σ_i q_i/|r-r_i||ν⟩
   *
   * @param charges Point charges with positions and magnitudes
   * @param[out] res Output buffer for point charge matrix (size:
   * num_basis_funcs × num_basis_funcs)
   */
  void point_charge_integral(const PointCharges* charges, double* res);
#endif

  /**
   * @brief Compute gradient contribution from overlap integrals
   *
   * Evaluates ∂S/∂R_A contribution to nuclear gradients using the
   * energy-weighted density matrix W = C_occ * ε_occ * C_occ**T.
   *
   * @param[in] W Energy-weighted density matrix (size: num_basis_funcs ×
   * num_basis_funcs)
   * @param[out] res Output gradient contribution (size: 3 × natoms)
   */
  void overlap_integral_deriv(const double* W, double* res);

  /**
   * @brief Compute gradient contribution from kinetic integrals
   *
   * Evaluates ∂T/∂R_A contribution to nuclear gradients.
   *
   * @param[in] D Density matrix (size: num_basis_funcs × num_basis_funcs)
   * @param[out] res Output gradient contribution (size: 3 × natoms)
   */
  void kinetic_integral_deriv(const double* D, double* res);

  /**
   * @brief Compute gradient contribution from nuclear attraction integrals
   *
   * Evaluates ∂V/∂R_A contribution to nuclear gradients, including both
   * the derivative of basis functions and the derivative of 1/|r-R_A|.
   *
   * @param[in] D Density matrix (size: num_basis_funcs × num_basis_funcs)
   * @param[out] res Output gradient contribution (size: 3 × natoms)
   */
  void nuclear_integral_deriv(const double* D, double* res);

#ifdef QDK_CHEMISTRY_ENABLE_QMMM
  /**
   * @brief Compute gradient contributions from point charge integrals
   *
   * Evaluates ∂V_QM/MM/∂R for both QM atoms and point charges in QM/MM
   * calculations. Provides forces on both QM and MM subsystems.
   *
   * @param[in] D Density matrix (size: num_basis_funcs × num_basis_funcs)
   * @param[out] res Output gradient for QM atoms (size: 3 × natoms_QM)
   * @param[out] pointcharges_res Output gradient for point charges (size: 3 ×
   * ncharges)
   * @param charges Point charges with positions and magnitudes
   */
  void pointcharge_integral_deriv(const double* D, double* res,
                                  double* pointcharges_res,
                                  const PointCharges* charges);
#endif

  /**
   * @brief Compute gradient contribution from ECP integrals
   *
   * Evaluates ∂V_ECP/∂R_A contribution to nuclear gradients for atoms
   * with effective core potentials.
   *
   * @param[in] D Density matrix (size: num_basis_funcs × num_basis_funcs)
   * @param[out] res Output gradient contribution (size: 3 × natoms)
   */
  void ecp_integral_deriv(const double* D, double* res);

 private:
  /**
   * @brief Convert BasisSet to libecpint format for ECP evaluation
   * @param basis_set Basis set with ECP information
   */
  void convert_to_libecp_shells_(const BasisSet& basis_set);

  /**
   * @brief Generic integral evaluation routine
   *
   * Template method for evaluating integrals with different operators.
   * Loops over significant shell pairs and accumulates results.
   *
   * @param nopers Number of operator components
   * @param engine_fn Factory function to create integral engine
   * @param[out] res Output matrices (one per operator component)
   */
  void integral_(size_t nopers, EngineFactory engine_fn, RowMajorMatrix* res);

  /**
   * @brief Generic integral derivative evaluation routine
   *
   * Template method for evaluating integral gradients. Accumulates
   * gradient contributions weighted by the coefficient matrix (density
   * or energy-weighted density).
   *
   * @param engine_fn Factory function to create derivative engine
   * @param coeff Coefficient matrix (density or energy-weighted density)
   * @param center_fn Function mapping shell pairs to atomic centers
   * @param[out] res Output gradient array (size: 3 × natoms)
   */
  void integral_deriv_(EngineFactory engine_fn, const RowMajorMatrix& coeff,
                       AtomCenterFn center_fn, double* res);

  std::vector<std::pair<int, int>>
      shell_pairs_;  ///< Significant shell pairs from Schwarz screening of the
                     ///< overlap
  libint2::BasisSet
      obs_;    ///< Basis set in libint2 format for integral evaluation
  bool pure_;  ///< True for spherical harmonics (pure), false for Cartesian
               ///< functions
  std::vector<std::pair<double, std::array<double, 3>>>
      atoms_;                 ///< Atomic charges and positions (Z, {x,y,z})
  std::vector<int> sh2atom_;  ///< Mapping from shell index to atom index

  std::vector<libecpint::GaussianShell>
      libecp_shells_;  ///< Basis shells in libecpint format
  std::vector<libecpint::ECP>
      libecp_ecps_;  ///< ECP definitions in libecpint format
  int max_ecp_am_;   ///< Maximum angular momentum in ECP basis

  ParallelConfig mpi_;    ///< MPI configuration for parallel execution
  BasisMode basis_mode_;  ///< Basis mode (spherical vs Cartesian)
};
}  // namespace qdk::chemistry::scf

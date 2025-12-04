// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/scf.h>
#include <qdk/chemistry/scf/core/types.h>

#include <limits>
#include <memory>

namespace qdk::chemistry::scf {

// Forward declaration
class SCFImpl;

/**
 * @brief Base class for SCF iteration algorithms
 *
 * This abstract base class defines the interface for different SCF convergence
 * algorithms such as DIIS, GDM, and hybrid DIIS-GDM methods. Each algorithm
 * is responsible for updating the density matrix and determining convergence
 * based on the current Fock matrix and energy.
 */
class SCFAlgorithm {
 public:
  /**
   * @brief Construct SCF algorithm given reference to SCFContext
   *
   * @param[in] ctx Reference to SCFContext
   */
  explicit SCFAlgorithm(const SCFContext& ctx);

  /**
   * @brief Default destructor
   */
  virtual ~SCFAlgorithm() noexcept = default;

  /**
   * @brief Perform one iteration of the SCF algorithm
   *
   * This method encapsulates the core iteration logic for the specific
   * algorithm. It accesses matrices and energy from the SCFImpl instance.
   *
   * @param[in,out] scf_impl Reference to SCFImpl containing all matrices and
   * energy
   */
  virtual void iterate(SCFImpl& scf_impl) = 0;

  /**
   * @brief Check if the algorithm has converged
   *
   * Uses standard SCF convergence criteria: energy change, density RMS, and
   * orbital gradient threshold.
   *
   * @param[in] scf_impl Reference to SCFImpl containing matrices and energy
   * @return true if the algorithm has reached convergence, false otherwise
   */
  bool check_convergence(const SCFImpl& scf_impl);

  /**
   * @brief Factory method to create appropriate SCF algorithm implementation
   *
   * Creates an SCF algorithm object of the appropriate derived type based on
   * the method specified in the configuration.
   *
   * @param[in] ctx SCF context containing configuration and state
   * @return std::shared_ptr<SCFAlgorithm> Pointer to newly created algorithm
   * instance
   */
  static std::shared_ptr<SCFAlgorithm> create(const SCFContext& ctx);

  /**
   * @brief Solve the eigenvalue problem for the Fock matrix and update
   * eigenvalues, molecular coefficients, and density matrix
   *
   * Solves the generalized eigenvalue problem F*C = S*C*E using the
   * orthogonalization matrix to transform to an orthogonal basis.
   *
   * @param[in] F Fock matrix to diagonalize
   * @param[in] S Overlap matrix
   * @param[in] X Orthogonalization matrix (num_atomic_orbitals ×
   * num_molecular_orbitals)
   * @param[out] C Molecular orbital coefficients
   * @param[out] eigenvalues Orbital eigenvalues
   * @param[out] P Density matrix
   * @param[in] num_occupied_orbitals Number of occupied orbitals per spin
   * [alpha, beta]
   * @param[in] num_atomic_orbitals Number of atomic orbitals
   * @param[in] num_molecular_orbitals Number of molecular orbitals
   * @param[in] idx_spin Density matrix index (0 for alpha or restricted, 1 for
   * beta)
   * @param[in] unrestricted Whether calculation is unrestricted
   */
  virtual void solve_fock_eigenproblem(
      const RowMajorMatrix& F, const RowMajorMatrix& S, const RowMajorMatrix& X,
      RowMajorMatrix& C, RowMajorMatrix& eigenvalues, RowMajorMatrix& P,
      const int num_occupied_orbitals[2], int num_atomic_orbitals,
      int num_molecular_orbitals, int idx_spin, bool unrestricted);

  /**
   * @brief Calculate orbital gradient (OG) error for convergence checking
   *
   * This method calculates the orbital gradient error in the atomic
   * orbital basis using the formula: error = FPS - SPF, where F is the Fock
   * matrix, P is the density matrix, and S is the overlap matrix.
   *
   * @param[in] F Fock matrix (ndm × num_atomic_orbitals ×
   * num_atomic_orbitals)
   * @param[in] P Density matrix (ndm × num_atomic_orbitals ×
   * num_atomic_orbitals)
   * @param[in] S Overlap matrix (num_atomic_orbitals × num_atomic_orbitals)
   * @param[out] error_matrix Output matrix to store calculated error (will be
   * resized)
   * @param[in] unrestricted Whether the calculation is unrestricted
   * @return Infinity norm of the error matrix
   */
  static double calculate_og_error_(const RowMajorMatrix& F,
                                    const RowMajorMatrix& P,
                                    const RowMajorMatrix& S,
                                    RowMajorMatrix& error_matrix,
                                    bool unrestricted);

 protected:
  const SCFContext& ctx_;  ///< Reference to SCF context
  double og_error_ = 0.0;  ///< Current orbital gradient error

  // Step tracking and density matrix history
  int step_count_ = 0;        ///< Current iteration step count
  RowMajorMatrix P_last_;     ///< Previous density matrix for RMS calculation
  double last_energy_ = 0.0;  ///< Previous energy for delta
                              ///< calculation
  double delta_energy_ =
      std::numeric_limits<double>::infinity();  ///< Energy change
  double density_rms_ = 0.0;                    ///< Last calculated density RMS
};
}  // namespace qdk::chemistry::scf

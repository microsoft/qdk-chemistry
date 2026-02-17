// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/scf_algorithm.h>
#include <qdk/chemistry/scf/core/types.h>

#include <memory>

namespace qdk::chemistry::scf {

class SCFImpl;

namespace impl {
class DIIS;
class ROHFHelper;
}  // namespace impl

/**
 * @brief Unified Pulay DIIS accelerator for RHF, UHF, and ROHF workflows
 *
 * The `DIIS` class encapsulates the Pulay Direct Inversion in the Iterative
 * Subspace machinery and the supporting bookkeeping needed to drive SCF
 * iterations across all orbital flavors. It chooses between spin-blocked and
 * ROHF-effective matrix views automatically and exposes helper hooks for
 * algorithms such as ASAHF that need custom density reconstruction.
 */
class DIIS : public SCFAlgorithm {
 public:
  /**
   * @brief Construct the DIIS helper
   *
   * @param ctx SCF context
   * @param subspace_size Max number of stored Fock/error pairs for DIIS
   */
  DIIS(const SCFContext& ctx, size_t subspace_size = 8);

  /**
   * @brief Virtual destructor
   */
  ~DIIS() noexcept override;

  /**
   * @brief Perform one Pulay DIIS iteration
   *
   * Pulls the active Fock/density matrices from `SCFImpl` (or the cached ROHF
   * view), computes a Pulay-extrapolated Fock matrix, solves the orbital
   * eigenproblem, and rebuilds the spin-blocked density matrix.
   *
   * @param[in,out] scf_impl SCF implementation object providing matrices and
   * energies
   */
  void iterate(SCFImpl& scf_impl) override;

  /**
   * @brief Update the density matrix
   *
   * Default implementation handles restricted and unrestricted cases, while
   * subclasses such as ASAHF may override it
   *
   * @param[in,out] P Density matrix to overwrite
   * @param[in] C Molecular orbital coefficient matrix
   * @param[in] unrestricted True if two spin blocks are present
   * @param[in] nelec_alpha Number of alpha electrons
   * @param[in] nelec_beta Number of beta electrons
   */
  void update_density_matrix(RowMajorMatrix& P, const RowMajorMatrix& C,
                             bool unrestricted, int nelec_alpha,
                             int nelec_beta) override;

  /**
   * @brief Build cached ROHF Fock/density matrices for convergence checks
   *
   * Converts spin-blocked Fock/density matrices into the total-density /
   * effective-Fock representation, then saved internally for use in DIIS
   * iterations and convergence checks
   *
   * @param[in] F Spin-blocked Fock matrix from `SCFImpl`
   * @param[in] C Molecular orbital coefficients
   * @param[in] P Spin-blocked density matrix
   * @param[in] nelec_alpha Number of alpha electrons
   * @param[in] nelec_beta Number of beta electrons
   */
  void build_rohf_f_p_matrix(const RowMajorMatrix& F, const RowMajorMatrix& C,
                             const RowMajorMatrix& P, int nelec_alpha,
                             int nelec_beta);

  /**
   * @brief Access the cached ROHF-effective Fock matrix
   *
   * @return Const reference to the total-density ROHF Fock matrix
   */
  const RowMajorMatrix& get_rohf_fock_matrix() const;

  /**
   * @brief Access the cached total (alpha+beta) density matrix
   *
   * @return Const reference to the cached ROHF density
   */
  const RowMajorMatrix& get_rohf_density_matrix() const;

  /**
   * @brief Mutable access to the total density matrix (used inside iterate)
   *
   * @return Reference to the cached ROHF density
   */
  RowMajorMatrix& rohf_density_matrix();

 private:
  /**
   * @brief Return most recent Pulay error metric for damping logic
   */
  double current_diis_error() const;

  /**
   * @brief Select the density view used for the current iteration
   */
  RowMajorMatrix& select_working_density(SCFImpl& scf_impl);
  /**
   * @brief Select the Fock matrix view used for the current iteration
   */
  const RowMajorMatrix& select_working_fock(const SCFImpl& scf_impl);

  std::unique_ptr<impl::DIIS> diis_impl_;
  std::unique_ptr<impl::ROHFHelper> rohf_helper_;
};

}  // namespace qdk::chemistry::scf

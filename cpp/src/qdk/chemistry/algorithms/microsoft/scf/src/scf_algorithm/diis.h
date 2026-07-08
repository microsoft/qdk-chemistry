// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/scf_algorithm.h>
#include <qdk/chemistry/scf/core/types.h>

#include <memory>
#include <utility>

namespace qdk::chemistry::scf {

class SCFImpl;

namespace impl {
class DIIS;
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

 private:
  /**
   * @brief Return most recent Pulay error metric for damping logic
   */
  double current_diis_error() const;

  /**
   * @brief Select the density/fock views used for the current iteration
   */
  std::pair<RowMajorMatrix&, const RowMajorMatrix&> select_working_matrices(
      SCFImpl& scf_impl);

  std::unique_ptr<impl::DIIS> diis_impl_;  ///< Pulay DIIS core implementation
};

}  // namespace qdk::chemistry::scf

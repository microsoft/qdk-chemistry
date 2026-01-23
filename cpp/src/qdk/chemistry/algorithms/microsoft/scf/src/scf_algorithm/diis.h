// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include "scf_matrix_handler.h"

#include <qdk/chemistry/scf/core/scf_algorithm.h>
#include <qdk/chemistry/scf/core/types.h>

#include <memory>

namespace qdk::chemistry::scf {

// Forward declaration of implementation classes
namespace impl {
class DIIS;
}  // namespace impl

/**
 * @brief Direct Inversion in the Iterative Subspace (DIIS) SCF algorithm
 *
 * Implements the DIIS algorithm for accelerating SCF convergence by
 * extrapolating from previous Fock matrices and error vectors. DIIS builds
 * a linear combination of previous trial vectors that minimizes the norm
 * of the error vector.
 *
 * The algorithm maintains a subspace of recent Fock matrices and their
 * corresponding error vectors, then solves a linear system to find optimal
 * extrapolation coefficients.
 *
 * Reference: P. Pulay, Chem. Phys. Lett. 73, 393 (1980)
 */
class DIIS : public SCFAlgorithm {
 public:
  /**
   * @brief Construct DIIS SCF algorithm
   *
   * @param[in] ctx SCFContext reference
   * @param[in] rohf_enabled Flag indicating if ROHF is enabled
   * @param[in] subspace_size Maximum number of vectors to retain in DIIS
   * subspace
   */
  explicit DIIS(const SCFContext& ctx, bool rohf_enabled,
                size_t subspace_size = 8);

  /**
   * @brief Destructor
   */
  ~DIIS() noexcept;

  /**
   * @brief Perform one DIIS SCF iteration
   *
   * @param[in,out] scf_impl Reference to SCFImpl containing all matrices and
   * energy
   */
  void iterate(SCFImpl& scf_impl) override;

 private:
  /// PIMPL pointer to implementation
  std::unique_ptr<impl::DIIS> diis_impl_;

  /// Fock matrix and density matrix handler, for RHF/UHF and ROHF cases
  std::unique_ptr<SCFMatrixHandler> matrix_handler_;
};

}  // namespace qdk::chemistry::scf

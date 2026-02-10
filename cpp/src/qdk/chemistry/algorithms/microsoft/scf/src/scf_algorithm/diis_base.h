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
}  // namespace impl

/**
 * @brief Shared infrastructure for DIIS-style SCF accelerators
 *
 * `DIISBase` wraps the Pulay DIIS extrapolation machinery and provides a set
 * of extension hooks for derived classes to control how Fock and density
 * matrices are sourced and committed back to `SCFImpl`. It encapsulates the
 * iteration boilerplate (extrapolation, eigenproblem solve, damping) so that
 * restricted/UHF, ROHF, and ASAHF variations can specialize only the pieces
 * they care about.
 */
class DIISBase : public SCFAlgorithm {
 public:
  /**
   * @brief Construct the DIIS base helper
   *
   * @param ctx SCF context
   * @param rohf_enabled Indicates whether ROHF support is requested
   * @param subspace_size Max number of stored Fock/error pairs for DIIS
   */
  DIISBase(const SCFContext& ctx, bool rohf_enabled,
           size_t subspace_size = 8);

  /**
   * @brief Virtual destructor
   */
  ~DIISBase() noexcept override;

  /**
   * @brief Perform one Pulay DIIS iteration using subclass hooks
   */
  void iterate(SCFImpl& scf_impl) override;

 protected:
  /**
   * @brief Access the Fock matrix view to feed into DIIS
   */
  virtual const RowMajorMatrix& get_active_fock(
      const SCFImpl& scf_impl) const = 0;

  /**
   * @brief Access the working density matrix used during DIIS updates
   */
  virtual RowMajorMatrix& active_density(SCFImpl& scf_impl) = 0;

  /**
   * @brief Optional post-iteration hook (default handles damping)
   */
  virtual void after_diis_iteration(SCFImpl& scf_impl);

  /**
   * @brief Indicates whether the subclass exposes a total-density view
   */
  virtual bool uses_total_density_view() const;

  /**
   * @brief Retrieve the most recent DIIS error metric
   */
  double current_diis_error() const;

 private:
  std::unique_ptr<impl::DIIS> diis_impl_;
};

}  // namespace qdk::chemistry::scf

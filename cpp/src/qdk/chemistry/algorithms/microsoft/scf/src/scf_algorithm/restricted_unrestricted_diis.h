// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include "diis_base.h"

namespace qdk::chemistry::scf {

/**
 * @brief Pulay DIIS accelerator for RHF and UHF cases
 *
 * `RestrictedUnrestrictedDIIS` reuses the `DIISBase` infrastructure but
 * directly exposes the spin-blocked Fock and density matrices coming from
 * `SCFImpl`. It covers both restricted (single block) and unrestricted (alpha
 * and beta blocks) SCF flavors without introducing ROHF-specific averaging.
 */
class RestrictedUnrestrictedDIIS : public DIISBase {
 public:
  /**
   * @brief Construct the DIIS helper for RHF/UHF workflows
   * @param ctx SCF context propagated to the base class
   * @param subspace_size Maximum number of error/Fock pairs stored by DIIS
   */
  explicit RestrictedUnrestrictedDIIS(const SCFContext& ctx,
                                      size_t subspace_size = 8);
  ~RestrictedUnrestrictedDIIS() noexcept override = default;

 protected:
  /**
   * @brief Return the spin-blocked Fock matrix maintained by `SCFImpl`
   */
  const RowMajorMatrix& get_active_fock(const SCFImpl& scf_impl) const override;

  /**
   * @brief Return the spin-blocked density matrix to be updated by DIIS
   */
  RowMajorMatrix& active_density(SCFImpl& scf_impl) override;
};

}  // namespace qdk::chemistry::scf

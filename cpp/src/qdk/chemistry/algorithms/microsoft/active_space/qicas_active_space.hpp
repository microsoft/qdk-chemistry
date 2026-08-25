// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/active_space.hpp>

#include "../qio/jacobi_settings.hpp"

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Settings for quantum information-assisted active-space optimization.
 */
class QICASActiveSpaceSettings : public qio::JacobiSettings {
 public:
  QICASActiveSpaceSettings() {
    set_default(
        "num_active_electrons", int64_t{-1},
        "Number of electrons in the target active space (required, even).");
    set_default("num_active_orbitals", int64_t{-1},
                "Number of orbitals in the target active space (required).");
  }
};

/**
 * @brief Optimize and select an active space using out-of-CAS correlation.
 *
 * The input wavefunction's active space defines the correlated optimization
 * window. QICAS minimizes the sum of single-orbital entropies outside the
 * target active slots, then reorders the optimized window by occupation to
 * assign inactive, active, and virtual orbitals. The returned wavefunction is
 * a single-reference carrier for those orbitals and partition labels; callers
 * must perform a subsequent correlated solve in the target active space.
 *
 * The candidate window must initially be ordered as prospective closed slots,
 * target active slots, then prospective virtual slots.
 */
class QICASActiveSpaceSelector final : public ActiveSpaceSelector {
 public:
  QICASActiveSpaceSelector() {
    _settings = std::make_unique<QICASActiveSpaceSettings>();
  }

  ~QICASActiveSpaceSelector() override = default;

  std::string name() const final { return "qdk_qicas"; }

 protected:
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const final;
};

}  // namespace qdk::chemistry::algorithms::microsoft

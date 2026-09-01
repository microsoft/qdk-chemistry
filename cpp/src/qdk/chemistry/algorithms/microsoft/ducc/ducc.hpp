// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

namespace qdk::chemistry::algorithms::microsoft {

class DuccSettings : public data::Settings {
 public:
  DuccSettings() {
    set_default("ducc_level", static_cast<std::int64_t>(2),
                "DUCC approximation: 0 selects A(1), 1 selects A(4), and 2 "
                "selects A(7)",
                data::BoundConstraint<std::int64_t>{0, 2});
  }
};

/**
 * @class DuccSolver
 * @brief Builds a P-space DUCC effective Hamiltonian.
 *
 * Evaluates a truncated BCH transformation
 * @f$ \bar H = e^{-\sigma} H e^{\sigma} @f$, where
 * @f$ \sigma = T_{ext} - T_{ext}^{\dagger} @f$, and returns its P-space
 * scalar, one-body, and two-body terms. The @c ducc_level setting selects the
 * perturbatively consistent A(1), A(4), and A(7) approximations of Bauman and
 * Kowalski, J. Chem. Phys. 156, 094106 (2022), DOI: 10.1063/5.0076260.
 *
 * The input Hamiltonian must be Hermitian and span the full orbital window.
 * The reference must contain real coupled-cluster amplitudes built on one
 * determinant whose occupied orbitals are contiguous from index zero in each
 * spin channel. The target P-space must be non-empty in each spin channel.
 */
class DuccSolver
    : public qdk::chemistry::algorithms::EffectiveHamiltonianConstructor {
 public:
  DuccSolver() { _settings = std::make_unique<DuccSettings>(); }

  ~DuccSolver() = default;

  std::string name() const final { return "ducc"; }

 protected:
  /**
   * @brief Build the effective P-space Hamiltonian.
   * @param reference Full-space coupled-cluster wavefunction.
   * @param hamiltonian Full-space Hamiltonian.
   * @param p_space_indices P-space orbital indices by spin.
   * @return P-space effective Hamiltonian.
   */
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<const data::SymmetryBlockedIndexSet> p_space_indices)
      const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

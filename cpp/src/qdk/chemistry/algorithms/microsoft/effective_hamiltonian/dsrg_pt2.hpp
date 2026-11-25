// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Settings for the fixed-reference DSRG-PT2 downfold.
 */
class DsrgPT2Settings : public qdk::chemistry::data::Settings {
 public:
  DsrgPT2Settings() {
    set_default(
        "flow_parameter", 0.5,
        "Gaussian-source DSRG flow s in Eh^-2; zero leaves the bare "
        "operator over P.",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
  }
  ~DsrgPT2Settings() override = default;
};

/**
 * @brief Fixed-reference, single-determinant DSRG-PT2 downfold of the window's
 * core and virtual orbitals onto the target space P (`p_indices`).
 *
 * Emits the second-order DSRG effective Hamiltonian over `p_indices`,
 * truncated to two-body terms in reference normal order and returned in bare
 * normal order. Supports closed-shell determinants on restricted orbitals and
 * closed- or open-shell determinants on unrestricted orbitals. The
 * `flow_parameter` setting is the flow s in Eh^-2; zero leaves the bare
 * operator over `p_indices`. Restricted input gives a restricted result,
 * unrestricted input an unrestricted one. The restricted and same-spin
 * two-body blocks are 4-fold (not 8-fold) symmetric, and the unrestricted
 * mixed-spin block is 2-fold Hermitian.
 */
class DsrgPT2Constructor
    : public qdk::chemistry::algorithms::EffectiveHamiltonianConstructor {
 public:
  DsrgPT2Constructor() { _settings = std::make_unique<DsrgPT2Settings>(); }
  ~DsrgPT2Constructor() override = default;

  std::string name() const final { return "qdk_dsrg_pt2"; }

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices)
      const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

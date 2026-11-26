// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

namespace qdk::chemistry::algorithms::microsoft {

/// Settings for the singles-and-doubles DUCC effective Hamiltonian.
class DuccSettings : public data::Settings {
 public:
  DuccSettings() {
    set_default("level", int64_t{2},
                "F-split BCH level: 0 is the bare downfold; levels 1-2 retain "
                "one additional Fock commutator.",
                data::BoundConstraint<int64_t>{0, 2});
  }
};

/// T1/T2 DUCC with a final reference-normal-ordered two-body truncation.
class DuccConstructor : public EffectiveHamiltonianConstructor {
 public:
  DuccConstructor() { _settings = std::make_unique<DuccSettings>(); }
  std::string name() const final { return "qdk_ducc"; }

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference,
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices)
      const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::algorithms {

class FiniteDifferenceNuclearDerivativeSettings
    : public NuclearDerivativeSettings {
 public:
  FiniteDifferenceNuclearDerivativeSettings() : NuclearDerivativeSettings() {
    set_default(
        "reuse_seed_active_space", true,
        "For multi-reference energy paths, reuse active and inactive orbital "
        "indices from an Orbitals or Wavefunction seed when fresh reference "
        "orbitals are generated for another geometry. Orbital coefficients are "
        "not reused for displaced finite-difference geometries.");
    set_default("finite_difference_step", 1.0e-3,
                "Central finite-difference nuclear displacement step in Bohr. "
                "Used for numeric gradients and Hessians.",
                data::BoundConstraint<double>{1.0e-8, 1.0});
  }
};

class FiniteDifferenceNuclearDerivativeCalculator
    : public NuclearDerivativeCalculator {
 public:
  FiniteDifferenceNuclearDerivativeCalculator() {
    _settings = std::make_unique<FiniteDifferenceNuclearDerivativeSettings>();
  }

  std::string name() const final { return "qdk_finite_difference"; }

  std::vector<std::string> aliases() const final {
    return {"qdk_finite_difference", "numeric"};
  }

 protected:
  NuclearDerivativeResult _run_impl(
      std::shared_ptr<data::Structure> structure, int charge,
      int spin_multiplicity, NuclearDerivativeSeedType seed_or_basis,
      unsigned int n_inactive_orbitals) const override;
};

std::unique_ptr<NuclearDerivativeCalculator>
make_finite_difference_nuclear_derivative_calculator();

}  // namespace qdk::chemistry::algorithms

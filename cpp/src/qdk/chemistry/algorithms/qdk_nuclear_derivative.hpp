// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::algorithms {

class QdkNuclearDerivativeSettings : public NuclearDerivativeSettings {
 public:
  QdkNuclearDerivativeSettings() : NuclearDerivativeSettings() {
    set_default(
        "finite_difference_step", 1.0e-3,
        "Central nuclear displacement step in Bohr used to compute Hessians "
        "by finite differences of analytic gradients.",
        data::BoundConstraint<double>{1.0e-8, 1.0});
  }
};

class QdkNuclearDerivativeCalculator : public NuclearDerivativeCalculator {
 public:
  QdkNuclearDerivativeCalculator() {
    _settings = std::make_unique<QdkNuclearDerivativeSettings>();
  }

  std::string name() const final { return "qdk"; }

  std::vector<std::string> aliases() const final {
    return {"qdk", "analytical_gradient"};
  }

 protected:
  NuclearDerivativeResult _run_impl(
      std::shared_ptr<data::Structure> structure, int charge,
      int spin_multiplicity, NuclearDerivativeSeedType seed_or_basis,
      unsigned int n_inactive_orbitals) const override;
};

std::unique_ptr<NuclearDerivativeCalculator>
make_qdk_nuclear_derivative_calculator();

}  // namespace qdk::chemistry::algorithms

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::algorithms {

class QdkNuclearDerivativeCalculator : public NuclearDerivativeCalculator {
 public:
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

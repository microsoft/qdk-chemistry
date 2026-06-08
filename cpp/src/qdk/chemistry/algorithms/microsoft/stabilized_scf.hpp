// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms::microsoft {

class StabilizedScfSettings
    : public qdk::chemistry::algorithms::ElectronicStructureSettings {
 public:
  StabilizedScfSettings();
};

class StabilizedScfSolver : public qdk::chemistry::algorithms::ScfSolver {
 public:
  StabilizedScfSolver();

  ~StabilizedScfSolver() = default;

  std::string name() const final { return "qdk_stabilized"; }

  std::vector<std::string> aliases() const final {
    return {"qdk_stabilized", "stabilized", "stabilized_scf"};
  }

 protected:
  std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Structure> structure, int charge,
      int spin_multiplicity, BasisOrGuessType basis_or_guess) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft
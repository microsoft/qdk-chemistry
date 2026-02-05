// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>

namespace qdk::chemistry::algorithms::microsoft {

class DensityFittedHamiltonianSettings : public qdk::chemistry::data::Settings {
 public:
  DensityFittedHamiltonianSettings() {}
  ~DensityFittedHamiltonianSettings() override = default;
};

class DensityFittedHamiltonianConstructor
    : public qdk::chemistry::algorithms::HamiltonianConstructor {
 public:
  DensityFittedHamiltonianConstructor() {
    _settings = std::make_unique<DensityFittedHamiltonianSettings>();
  };
  ~DensityFittedHamiltonianConstructor() override = default;

  virtual std::string name() const final { return "qdk_density_fitted"; };

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Orbitals> orbitals,
      OptionalAuxBasis aux_basis) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

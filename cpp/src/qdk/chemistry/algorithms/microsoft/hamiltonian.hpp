// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>

namespace qdk::chemistry::algorithms::microsoft {

class HamiltonianSettings : public qdk::chemistry::data::Settings {
 public:
  HamiltonianSettings() {
    // TODO enable and use
    // set_default("integral_threshold", 1e-12);
    set_default("eri_method", "direct");
  }
  ~HamiltonianSettings() override = default;
};

class HamiltonianConstructor
    : public qdk::chemistry::algorithms::HamiltonianConstructor {
 public:
  HamiltonianConstructor() {
    _settings = std::make_unique<HamiltonianSettings>();
  };
  ~HamiltonianConstructor() override = default;

  virtual std::string name() const final { return "qdk"; };

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Orbitals> orbitals) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

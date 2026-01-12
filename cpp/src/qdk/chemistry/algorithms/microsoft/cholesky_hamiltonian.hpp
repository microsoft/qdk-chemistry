// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>

namespace qdk::chemistry::algorithms::microsoft {

class CholeskyHamiltonianSettings : public qdk::chemistry::data::Settings {
 public:
  CholeskyHamiltonianSettings() {
    set_default("scf_type", "auto");
    set_default("cholesky_tolerance", 1e-6);
    set_default("store_cholesky_vectors", true);
  }
  ~CholeskyHamiltonianSettings() override = default;
};

class CholeskyHamiltonianConstructor
    : public qdk::chemistry::algorithms::HamiltonianConstructor {
 public:
  CholeskyHamiltonianConstructor() {
    _settings = std::make_unique<CholeskyHamiltonianSettings>();
  };
  ~CholeskyHamiltonianConstructor() override = default;

  virtual std::string name() const final { return "qdk_cholesky"; };

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Orbitals> orbitals) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

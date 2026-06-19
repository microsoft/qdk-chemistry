// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <qdk/chemistry/algorithms/f12_scf.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Settings for the canonical transcorrelated F12 (CT-F12) Hartree-Fock
 *        solver.
 *
 * All configuration lives here and is locked at run(). See the CT-F12 design
 * for the meaning of each key.
 */
class CtF12HartreeFockSettings : public qdk::chemistry::data::Settings {
 public:
  CtF12HartreeFockSettings() {
    set_default<double>("gamma", 1.0,
                        "Slater geminal exponent gamma (atomic units)",
                        data::BoundConstraint<double>{0.0, 100.0});
    set_default("cabs_basis", std::string(""),
                "Named OptRI/CABS auxiliary basis; empty derives one from the "
                "orbital basis set");
    set_default<int64_t>("frozen_core", 0,
                         "Number of frozen core orbitals (formulation (a))",
                         data::BoundConstraint<int64_t>{0});
  }
  ~CtF12HartreeFockSettings() override = default;
};

/**
 * @brief Canonical transcorrelated F12 (CT-F12) Hartree-Fock solver.
 *
 * Builds the dressed transcorrelated Hamiltonian from the reference orbitals
 * and relaxes the closed-shell orbitals in its mean field. The returned
 * wavefunction carries the relaxed orbital coefficients and the dressed-Fock
 * orbital energies, with the frozen core marked inactive, so it is a canonical
 * reference for downstream correlated methods (its conventional MP2 yields the
 * F12-MP2 energy).
 *
 * @see algorithms::F12HartreeFockSolver
 * @see microsoft::CtF12HamiltonianConstructor
 */
class CtF12HartreeFockSolver
    : public qdk::chemistry::algorithms::F12HartreeFockSolver {
 public:
  CtF12HartreeFockSolver() {
    _settings = std::make_unique<CtF12HartreeFockSettings>();
  };
  ~CtF12HartreeFockSolver() override = default;

  std::string name() const final { return "qdk_ct_f12"; };

 protected:
  std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> reference) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

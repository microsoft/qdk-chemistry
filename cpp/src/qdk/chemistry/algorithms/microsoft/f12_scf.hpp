// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>
#include <utility>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Settings for the canonical transcorrelated F12 (CT-F12) SCF solver.
 *
 * The canonical reference orbitals are produced by a configurable SCF sub-step
 * (the @c canonical_scf @ref data::AlgorithmRef, an @ref ScfSolver), so this
 * solver does not duplicate the canonical SCF configuration. The remaining keys
 * parameterize the F12 dressing. All configuration is locked at run().
 */
class CtF12ScfSettings : public qdk::chemistry::data::Settings {
 public:
  CtF12ScfSettings() {
    set_default<double>("gamma", 1.0,
                        "Slater geminal exponent gamma (atomic units)",
                        data::BoundConstraint<double>{0.0, 100.0});
    set_default("cabs_basis", std::string(""),
                "Named OptRI/CABS auxiliary basis; empty derives one from the "
                "orbital basis set");
    set_default<int64_t>("frozen_core", 0,
                         "Number of frozen core orbitals (formulation (a))",
                         data::BoundConstraint<int64_t>{0});
    set_default("canonical_scf", data::AlgorithmRef("scf_solver", "qdk"));
  }
  ~CtF12ScfSettings() override = default;
};

/**
 * @brief Canonical transcorrelated F12 (CT-F12) SCF solver.
 *
 * Produces the relaxed F12-HF orbitals: it runs the configured @c canonical_scf
 * sub-step (any @ref ScfSolver) to obtain the canonical Hartree-Fock reference,
 * builds the dressed transcorrelated Hamiltonian from it, and relaxes the
 * closed-shell orbitals in the dressed mean field. The returned wavefunction
 * carries the relaxed orbital coefficients and the dressed-Fock orbital
 * energies, with the frozen core marked inactive, so its conventional MP2
 * yields the F12-MP2 energy. As an @ref ScfSolver it is a drop-in orbitals
 * producer; only the F12 dressing distinguishes it from a canonical SCF.
 *
 * @see algorithms::ScfSolver
 * @see microsoft::CtF12HamiltonianConstructor
 */
class CtF12ScfSolver : public qdk::chemistry::algorithms::ScfSolver {
 public:
  CtF12ScfSolver() { _settings = std::make_unique<CtF12ScfSettings>(); };
  ~CtF12ScfSolver() override = default;

  std::string name() const final { return "qdk_ct_f12"; };

 protected:
  std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Structure> structure, int charge, int multiplicity,
      BasisOrGuessType basis_or_guess) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

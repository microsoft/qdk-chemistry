// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Settings for the canonical transcorrelated F12 (CT-F12) Hamiltonian
 *        constructor.
 *
 * All configuration lives here and is locked at run(). See the CT-F12 design
 * for the meaning of each key.
 */
class CtF12HamiltonianSettings : public qdk::chemistry::data::Settings {
 public:
  CtF12HamiltonianSettings() {
    set_default<double>("gamma", 1.0,
                        "Slater geminal exponent gamma (atomic units)",
                        data::BoundConstraint<double>{0.0, 100.0});
    set_default("cabs_basis", std::string(""),
                "Named OptRI/CABS auxiliary basis; empty derives one from the "
                "orbital basis set");
    set_default<int64_t>("frozen_core", 0,
                         "Number of frozen core orbitals (formulation (a))",
                         data::BoundConstraint<int64_t>{0});
    set_default("eri_method", std::string("direct"),
                "ERI evaluation method: 'direct' computes integrals "
                "on-the-fly, 'incore' stores all integrals in memory",
                data::ListConstraint<std::string>{
                    {std::vector<std::string>{"direct", "incore"}}});
    set_default("slater_factor", std::string("stg"),
                "Slater factor representation: 'stg' genuine Slater-type "
                "geminal, 'cgtg' Gaussian-fitted geminal",
                data::ListConstraint<std::string>{
                    {std::vector<std::string>{"stg", "cgtg"}}});
    set_default("orbital_basis", std::string("relaxed"),
                "Orbital basis of the emitted Hamiltonian: 'relaxed' relaxes "
                "the closed-shell orbitals in the dressed mean field and emits "
                "the F12-HF canonical basis (canonical post-HF over it "
                "reproduces F12-MP2/F12-CCSD); 'reference' keeps the reference "
                "orbital basis (a drop-in replacement for the bare "
                "Hamiltonian)",
                data::ListConstraint<std::string>{
                    {std::vector<std::string>{"relaxed", "reference"}}});
    set_default("symmetrize_two_body", false,
                "Symmetrize the dressed two-body tensor for solvers that "
                "assume 8-fold permutational symmetry");
  }
  ~CtF12HamiltonianSettings() override = default;
};

/**
 * @brief Canonical transcorrelated F12 (CT-F12) effective-Hamiltonian
 *        constructor.
 *
 * Produces an a priori, Hermitian, two-body effective Hamiltonian by an
 * approximate canonical (unitary) similarity transformation of the molecular
 * Hamiltonian with a fixed-amplitude Slater-type geminal generator. The
 * reduced density matrices that close the cumulant reduction are read from the
 * reference wavefunction, so a single-determinant reference yields the
 * single-reference flavor while a multi-determinant reference yields the
 * multireference flavor through the same code path.
 *
 * @see algorithms::EffectiveHamiltonianConstructor
 */
class CtF12HamiltonianConstructor
    : public qdk::chemistry::algorithms::EffectiveHamiltonianConstructor {
 public:
  CtF12HamiltonianConstructor() {
    _settings = std::make_unique<CtF12HamiltonianSettings>();
  };
  ~CtF12HamiltonianConstructor() override = default;

  std::string name() const final { return "qdk_ct_f12"; };

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Wavefunction> reference) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>

namespace qdk::chemistry::algorithms::detail {

struct EnergyEvaluation {
  double energy = 0.0;
  std::optional<std::shared_ptr<data::Wavefunction>> wavefunction;
};

std::shared_ptr<data::Structure> copy_structure(
    const std::shared_ptr<data::Structure>& structure);

std::shared_ptr<data::Structure> displace_structure(
    const std::shared_ptr<data::Structure>& structure, Eigen::Index coordinate,
    double displacement);

BasisOrGuessType seed_to_scf_input(const NuclearDerivativeSeedType& seed,
                                   bool allow_orbital_guess);

EnergyEvaluation evaluate_energy(const data::Settings& settings,
                                 std::shared_ptr<data::Structure> structure,
                                 int charge, int spin_multiplicity,
                                 const NuclearDerivativeSeedType& seed,
                                 bool allow_wavefunction_seed,
                                 unsigned int n_active_alpha_electrons,
                                 unsigned int n_active_beta_electrons);

}  // namespace qdk::chemistry::algorithms::detail

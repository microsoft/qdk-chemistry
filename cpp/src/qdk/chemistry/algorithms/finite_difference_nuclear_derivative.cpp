// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "finite_difference_nuclear_derivative.hpp"

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/nuclear_gradients.hpp>
#include <qdk/chemistry/data/nuclear_hessian.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <vector>

#include "nuclear_derivative_detail.hpp"

namespace qdk::chemistry::algorithms {

NuclearDerivativeResult FiniteDifferenceNuclearDerivativeCalculator::_run_impl(
    std::shared_ptr<data::Structure> structure, int charge,
    int spin_multiplicity, NuclearDerivativeSeedType seed,
    unsigned int n_inactive_orbitals) const {
  std::optional<utils::ScopedLogLevel> scoped_log_level;
  if (_settings->get<bool>("suppress_child_algorithm_logging")) {
    scoped_log_level.emplace(utils::LogLevel::error);
  }
  if (!structure) {
    throw std::invalid_argument("Structure must not be null");
  }
  const auto [n_active_alpha_electrons, n_active_beta_electrons] =
      detail::active_electron_counts(structure, charge, spin_multiplicity,
                                     n_inactive_orbitals);
  const double step = _settings->get<double>("finite_difference_step");
  const bool compute_hessian = _settings->get<bool>("compute_hessian");
  const auto dimension =
      static_cast<Eigen::Index>(3 * structure->get_num_atoms());

  auto central = detail::evaluate_energy(
      *_settings, structure, charge, spin_multiplicity, seed, true,
      n_active_alpha_electrons, n_active_beta_electrons);
  Eigen::VectorXd gradients = Eigen::VectorXd::Zero(dimension);
  std::vector<double> plus_energies(static_cast<size_t>(dimension));
  std::vector<double> minus_energies(static_cast<size_t>(dimension));

  for (Eigen::Index coordinate = 0; coordinate < dimension; ++coordinate) {
    auto plus_structure =
        detail::displace_structure(structure, coordinate, step);
    auto minus_structure =
        detail::displace_structure(structure, coordinate, -step);
    plus_energies[coordinate] =
        detail::evaluate_energy(
            *_settings, plus_structure, charge, spin_multiplicity, seed, false,
            n_active_alpha_electrons, n_active_beta_electrons)
            .energy;
    minus_energies[coordinate] =
        detail::evaluate_energy(
            *_settings, minus_structure, charge, spin_multiplicity, seed, false,
            n_active_alpha_electrons, n_active_beta_electrons)
            .energy;
    gradients(coordinate) =
        (plus_energies[coordinate] - minus_energies[coordinate]) / (2.0 * step);
  }

  std::optional<std::shared_ptr<data::NuclearHessian>> hessian;
  if (compute_hessian) {
    Eigen::MatrixXd hessian_matrix =
        Eigen::MatrixXd::Zero(dimension, dimension);
    for (Eigen::Index coordinate = 0; coordinate < dimension; ++coordinate) {
      hessian_matrix(coordinate, coordinate) =
          (plus_energies[coordinate] - 2.0 * central.energy +
           minus_energies[coordinate]) /
          (step * step);
    }

    for (Eigen::Index i = 0; i < dimension; ++i) {
      for (Eigen::Index j = i + 1; j < dimension; ++j) {
        auto pp = detail::displace_structure(
            detail::displace_structure(structure, i, step), j, step);
        auto pm = detail::displace_structure(
            detail::displace_structure(structure, i, step), j, -step);
        auto mp = detail::displace_structure(
            detail::displace_structure(structure, i, -step), j, step);
        auto mm = detail::displace_structure(
            detail::displace_structure(structure, i, -step), j, -step);
        const double value =
            (detail::evaluate_energy(*_settings, pp, charge, spin_multiplicity,
                                     seed, false, n_active_alpha_electrons,
                                     n_active_beta_electrons)
                 .energy -
             detail::evaluate_energy(*_settings, pm, charge, spin_multiplicity,
                                     seed, false, n_active_alpha_electrons,
                                     n_active_beta_electrons)
                 .energy -
             detail::evaluate_energy(*_settings, mp, charge, spin_multiplicity,
                                     seed, false, n_active_alpha_electrons,
                                     n_active_beta_electrons)
                 .energy +
             detail::evaluate_energy(*_settings, mm, charge, spin_multiplicity,
                                     seed, false, n_active_alpha_electrons,
                                     n_active_beta_electrons)
                 .energy) /
            (4.0 * step * step);
        hessian_matrix(i, j) = value;
        hessian_matrix(j, i) = value;
      }
    }
    hessian = std::make_shared<data::NuclearHessian>(
        detail::copy_structure(structure), hessian_matrix);
  }

  return {central.energy,
          std::make_shared<data::NuclearGradients>(
              detail::copy_structure(structure), gradients),
          hessian, central.wavefunction};
}

std::unique_ptr<NuclearDerivativeCalculator>
make_finite_difference_nuclear_derivative_calculator() {
  return std::make_unique<FiniteDifferenceNuclearDerivativeCalculator>();
}

}  // namespace qdk::chemistry::algorithms

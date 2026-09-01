// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk_nuclear_derivative.hpp"

#include <Eigen/Dense>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/nuclear_gradients.hpp>
#include <qdk/chemistry/data/nuclear_hessian.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

#include "microsoft/scf.hpp"
#include "nuclear_derivative_detail.hpp"

namespace qdk::chemistry::algorithms {

NuclearDerivativeResult QdkNuclearDerivativeCalculator::_run_impl(
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

  const auto ref = _settings->get<data::AlgorithmRef>("energy_calculator");
  if (ref.get_algorithm_type() != ScfSolverFactory::algorithm_type_name() ||
      ref.get_algorithm_name() != "qdk") {
    throw std::invalid_argument(
        "The QDK analytic nuclear derivative calculator requires "
        "energy_calculator to reference scf_solver/qdk.");
  }
  if (n_inactive_orbitals != 0) {
    throw std::invalid_argument(
        "The QDK analytic nuclear derivative calculator does not use an "
        "active space; n_inactive_orbitals must be 0");
  }
  (void)detail::active_electron_counts(structure, charge, spin_multiplicity,
                                       seed, n_inactive_orbitals);

  microsoft::ScfSolver solver;
  if (ref.get_settings()) {
    solver.settings().update(*ref.get_settings());
  }

  auto evaluate = [&](const std::shared_ptr<data::Structure>& current_structure,
                      bool allow_orbital_guess) {
    auto result = solver.run_with_analytic_gradient(
        current_structure, charge, spin_multiplicity,
        detail::seed_to_scf_input(seed, allow_orbital_guess));
    if (!result.nuclear_gradient.has_value()) {
      throw std::runtime_error(
          "Internal SCF did not return the requested analytic nuclear "
          "gradient");
    }
    return result;
  };

  auto scf_result = evaluate(structure, true);
  auto gradients = std::make_shared<data::NuclearGradients>(
      detail::copy_structure(structure), *scf_result.nuclear_gradient);

  std::optional<std::shared_ptr<data::NuclearHessian>> hessian;
  if (_settings->get<bool>("compute_hessian")) {
    const auto dimension =
        static_cast<Eigen::Index>(3 * structure->get_num_atoms());
    const double step = _settings->get<double>("finite_difference_step");
    Eigen::MatrixXd hessian_matrix =
        Eigen::MatrixXd::Zero(dimension, dimension);
    for (Eigen::Index coordinate = 0; coordinate < dimension; ++coordinate) {
      auto plus_structure =
          detail::displace_structure(structure, coordinate, step);
      auto minus_structure =
          detail::displace_structure(structure, coordinate, -step);
      auto plus = evaluate(plus_structure, false);
      auto minus = evaluate(minus_structure, false);
      hessian_matrix.col(coordinate) =
          (*plus.nuclear_gradient - *minus.nuclear_gradient) / (2.0 * step);
    }
    hessian_matrix =
        (0.5 * (hessian_matrix + hessian_matrix.transpose())).eval();
    hessian = std::make_shared<data::NuclearHessian>(
        detail::copy_structure(structure), hessian_matrix);
  }

  return {scf_result.energy, gradients, hessian, scf_result.wavefunction};
}

std::unique_ptr<NuclearDerivativeCalculator>
make_qdk_nuclear_derivative_calculator() {
  return std::make_unique<QdkNuclearDerivativeCalculator>();
}

}  // namespace qdk::chemistry::algorithms

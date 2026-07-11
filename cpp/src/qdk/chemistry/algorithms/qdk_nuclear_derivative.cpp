// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk_nuclear_derivative.hpp"

#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/nuclear_gradients.hpp>
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
  (void)detail::active_electron_counts(structure, charge, spin_multiplicity,
                                       n_inactive_orbitals);
  if (_settings->get<bool>("compute_hessian")) {
    throw std::invalid_argument(
        "The QDK analytic nuclear derivative calculator does not currently "
        "provide Hessians. Use the finite_difference implementation for "
        "numeric Hessians.");
  }

  const auto ref = _settings->get<data::AlgorithmRef>("energy_calculator");
  if (ref.get_algorithm_type() != ScfSolverFactory::algorithm_type_name() ||
      ref.get_algorithm_name() != "qdk") {
    throw std::invalid_argument(
        "The QDK analytic nuclear derivative calculator requires "
        "energy_calculator to reference scf_solver/qdk.");
  }

  microsoft::ScfSolver solver;
  if (ref.get_settings()) {
    solver.settings().update(*ref.get_settings());
  }

  auto scf_result =
      solver.run_with_analytic_gradient(structure, charge, spin_multiplicity,
                                        detail::seed_to_scf_input(seed, true));
  auto nuclear_gradient = scf_result.nuclear_gradient;
  if (!nuclear_gradient.has_value()) {
    throw std::runtime_error(
        "Internal SCF did not return the requested analytic nuclear gradient");
  }
  auto gradients = std::make_shared<data::NuclearGradients>(
      detail::copy_structure(structure), *nuclear_gradient);

  return {scf_result.energy, gradients, std::nullopt, scf_result.wavefunction};
}

std::unique_ptr<NuclearDerivativeCalculator>
make_qdk_nuclear_derivative_calculator() {
  return std::make_unique<QdkNuclearDerivativeCalculator>();
}

}  // namespace qdk::chemistry::algorithms

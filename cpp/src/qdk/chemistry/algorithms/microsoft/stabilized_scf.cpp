// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "stabilized_scf.hpp"

#include <qdk/chemistry/algorithms/stability.hpp>
#include <qdk/chemistry/data/stability_result.hpp>
#include <qdk/chemistry/utils/orbital_rotation.hpp>
#include <string>

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {
void copy_common_scf_settings(const data::Settings& source,
                              data::Settings& destination) {
  for (const auto& [key, value] : source.get_all_settings()) {
    if (destination.has(key)) {
      destination.update(key, value);
    }
  }
}

bool can_check_external_stability(
    const std::shared_ptr<data::Wavefunction>& wavefunction) {
  const auto& symmetries = wavefunction->get_orbitals()->symmetries();
  if (!symmetries || !symmetries->has_axis(data::AxisName::Spin)) {
    return false;
  }
  if (!symmetries->axis(data::AxisName::Spin).equivalent()) {
    return false;
  }
  const auto counts = wavefunction->total_num_particles();
  const auto num_alpha = counts->value(data::axes::alpha());
  const auto num_beta = counts->value(data::axes::beta());
  return num_alpha == num_beta;
}

}  // namespace detail

StabilizedScfSettings::StabilizedScfSettings()
    : qdk::chemistry::algorithms::ElectronicStructureSettings() {
  set_default("scf_solver", data::AlgorithmRef("scf_solver", "qdk"),
              "Nested SCF solver used for each SCF optimization.");
  set_default("stability_checker",
              data::AlgorithmRef("stability_checker", "qdk"),
              "Nested stability checker used after each SCF optimization.");
  set_default(
      "max_stability_iterations", static_cast<int64_t>(5),
      "Maximum number of stability-check/rerun cycles.",
      data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
  set_default("check_internal", true,
              "Check internal orbital-rotation stability.");
  set_default("check_external", true,
              "Check external orbital-rotation stability when applicable.");
  set_default("fail_on_unstable", true,
              "Throw if the final wavefunction remains unstable after the "
              "configured stability cycles.");
  set_default("external_instability_action", std::string("unrestricted"),
              "Action for external instabilities: 'unrestricted' switches to "
              "unrestricted SCF, 'rotate_only' only rotates orbitals.",
              data::ListConstraint<std::string>{
                  {std::vector<std::string>{"unrestricted", "rotate_only"}}});
}

StabilizedScfSolver::StabilizedScfSolver() {
  _settings = std::make_unique<StabilizedScfSettings>();
}

std::pair<double, std::shared_ptr<data::Wavefunction>>
StabilizedScfSolver::_run_impl(std::shared_ptr<data::Structure> structure,
                               int charge, int spin_multiplicity,
                               BasisOrGuessType basis_or_guess) const {
  const int64_t max_stability_iterations =
      _settings->get<int64_t>("max_stability_iterations");
  const bool check_internal = _settings->get<bool>("check_internal");
  const bool check_external_setting = _settings->get<bool>("check_external");
  const bool fail_on_unstable = _settings->get<bool>("fail_on_unstable");
  const auto external_instability_action =
      _settings->get<std::string>("external_instability_action");

  auto create_scf_solver = [&](const std::string& scf_type_override = "") {
    auto solver = _create_nested<ScfSolverFactory>("scf_solver");
    detail::copy_common_scf_settings(*_settings, solver->settings());
    if (!scf_type_override.empty()) {
      solver->settings().set("scf_type", scf_type_override);
    }
    return solver;
  };

  auto create_stability_checker = [&](bool check_external) {
    auto checker = _create_nested<StabilityCheckerFactory>("stability_checker");
    if (checker->settings().has("internal")) {
      checker->settings().set("internal", check_internal);
    }
    if (checker->settings().has("external")) {
      checker->settings().set("external", check_external);
    }
    if (checker->settings().has("method") && _settings->has("method")) {
      checker->settings().set("method", _settings->get<std::string>("method"));
    }
    return checker;
  };

  std::string scf_type_override;
  auto scf_solver = create_scf_solver();
  auto [energy, wavefunction] =
      scf_solver->run(structure, charge, spin_multiplicity, basis_or_guess);
  bool is_stable = true;

  for (int64_t iteration = 0; iteration < max_stability_iterations;
       ++iteration) {
    const bool check_external =
        check_external_setting &&
        detail::can_check_external_stability(wavefunction);
    auto stability_checker = create_stability_checker(check_external);

    std::shared_ptr<data::StabilityResult> result;
    std::tie(is_stable, result) = stability_checker->run(wavefunction);
    if (is_stable) {
      return {energy, wavefunction};
    }

    bool do_external = false;
    Eigen::VectorXd rotation_vector;
    if (!result->is_external_stable() && result->has_external_result()) {
      rotation_vector =
          result->get_smallest_external_eigenvalue_and_vector().second;
      do_external = true;
    } else if (!result->is_internal_stable()) {
      rotation_vector =
          result->get_smallest_internal_eigenvalue_and_vector().second;
    } else {
      throw std::runtime_error(
          "Stability checker reported an unstable wavefunction without an "
          "internal or external instability result");
    }

    auto [num_alpha, num_beta] = wavefunction->get_total_num_electrons();
    auto rotated_orbitals = qdk::chemistry::utils::rotate_orbitals(
        wavefunction->get_orbitals(), rotation_vector, num_alpha, num_beta,
        do_external);

    if (do_external && external_instability_action == "unrestricted") {
      scf_type_override = "unrestricted";
    }

    scf_solver = create_scf_solver(scf_type_override);
    std::tie(energy, wavefunction) =
        scf_solver->run(structure, charge, spin_multiplicity, rotated_orbitals);
  }

  if (max_stability_iterations > 0) {
    auto stability_checker = create_stability_checker(
        check_external_setting &&
        detail::can_check_external_stability(wavefunction));
    std::tie(is_stable, std::ignore) = stability_checker->run(wavefunction);
  }

  if (!is_stable && fail_on_unstable) {
    throw std::runtime_error(
        "Stabilized SCF did not find a stable wavefunction within " +
        std::to_string(max_stability_iterations) + " stability cycles");
  }

  return {energy, wavefunction};
}

}  // namespace qdk::chemistry::algorithms::microsoft

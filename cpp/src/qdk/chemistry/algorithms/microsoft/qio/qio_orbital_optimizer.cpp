// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qio_orbital_optimizer.hpp"

#include <cmath>
#include <cstdint>
#include <optional>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <span>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "frozen_space_rdms.hpp"
#include "jacobi_optimizer.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace {

void require_finite_setting(const char* setting_name, double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string("QIOOrbitalOptimizer setting '") +
                                setting_name + "' must be finite.");
  }
}

}  // namespace

std::shared_ptr<data::OrbitalOptimizationResult> QIOOrbitalOptimizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction) const {
  if (!wavefunction) {
    throw std::invalid_argument("QIOOrbitalOptimizer requires a wavefunction");
  }
  auto orbitals = wavefunction->get_orbitals();
  if (!orbitals->is_restricted()) {
    throw std::invalid_argument(
        "QIOOrbitalOptimizer requires a single spatial orbital set "
        "(RHF/ROHF)");
  }
  if (!orbitals->has_active_space()) {
    throw std::invalid_argument("QIOOrbitalOptimizer requires an active space");
  }
  if (!orbitals->has_overlap_matrix()) {
    throw std::invalid_argument(
        "QIOOrbitalOptimizer requires an overlap matrix");
  }
  if (!wavefunction->has_one_rdm_spin_dependent() ||
      !wavefunction->has_two_rdm_spin_dependent()) {
    throw std::invalid_argument(
        "QIOOrbitalOptimizer requires spin-dependent active 1- and 2-RDMs");
  }

  const auto active_indices_a = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::alpha());
  const auto active_indices_b = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::beta());
  const auto inactive_indices_a = data::spin_channel_indices(
      orbitals->inactive_indices(), data::axes::alpha());
  const auto inactive_indices_b = data::spin_channel_indices(
      orbitals->inactive_indices(), data::axes::beta());
  if (active_indices_a != active_indices_b ||
      inactive_indices_a != inactive_indices_b) {
    throw std::invalid_argument(
        "QIOOrbitalOptimizer requires matching alpha and beta orbital "
        "partitions");
  }

  const auto* active_one_rdm =
      std::get_if<data::SymmetryBlockedTensor<2, double>>(
          &wavefunction->active_one_rdm());
  const auto* active_two_rdm =
      std::get_if<data::SymmetryBlockedTensor<4, double>>(
          &wavefunction->active_two_rdm());
  if (!active_one_rdm || !active_two_rdm) {
    throw std::invalid_argument(
        "QIOOrbitalOptimizer requires real-valued active RDMs");
  }
  const auto& active_alpha =
      active_one_rdm->block({data::axes::alpha(), data::axes::alpha()});
  const auto& active_beta =
      active_one_rdm->block({data::axes::beta(), data::axes::beta()});
  const auto& active_alpha_beta =
      active_two_rdm->block({data::axes::alpha(), data::axes::alpha(),
                             data::axes::beta(), data::axes::beta()});

  const std::size_t dimension = orbitals->get_num_molecular_orbitals();
  auto rdms = qio::detail::build_frozen_space_rdms(
      active_alpha, active_beta,
      std::span<const double>(active_alpha_beta.data(),
                              active_alpha_beta.size()),
      dimension, active_indices_a, inactive_indices_a);

  const auto max_cycles =
      static_cast<std::size_t>(_settings->get<int64_t>("max_cycles"));
  const double convergence_tolerance =
      _settings->get<double>("convergence_tolerance");
  const double coarse_angle_step = _settings->get<double>("coarse_angle_step");
  const auto fine_samples =
      static_cast<int>(_settings->get<int64_t>("fine_samples"));
  const double improvement_tolerance =
      _settings->get<double>("improvement_tolerance");
  require_finite_setting("convergence_tolerance", convergence_tolerance);
  require_finite_setting("coarse_angle_step", coarse_angle_step);
  require_finite_setting("improvement_tolerance", improvement_tolerance);

  auto optimization = qio::detail::optimize_rotation(
      std::move(rdms.alpha), std::move(rdms.beta), std::move(rdms.alpha_beta),
      dimension, max_cycles, convergence_tolerance, coarse_angle_step,
      fine_samples, improvement_tolerance);

  const auto& coefficients = orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  const Eigen::MatrixXd rotated_coefficients =
      coefficients * optimization.rotation;
  auto proposed_orbitals = std::make_shared<data::Orbitals>(
      rotated_coefficients, std::nullopt, orbitals->get_overlap_matrix(),
      orbitals->get_basis_set(), orbitals->active_indices(),
      orbitals->inactive_indices());

  return std::make_shared<data::OrbitalOptimizationResult>(
      proposed_orbitals, optimization.initial_objective,
      optimization.final_objective, optimization.cycles,
      optimization.converged);
}

}  // namespace qdk::chemistry::algorithms::microsoft

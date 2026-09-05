// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qicas_active_space.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <optional>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "../qio/jacobi_optimizer.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace {

std::shared_ptr<const data::SymmetryBlockedIndexSet> make_orbital_index_set(
    const std::shared_ptr<data::Orbitals>& orbitals,
    const std::vector<std::size_t>& indices) {
  const auto to_u32 = [](const std::vector<std::size_t>& values) {
    return std::vector<std::uint32_t>(values.begin(), values.end());
  };

  std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>> blocks;
  const auto symmetries = orbitals->symmetries();
  if (symmetries && symmetries->has_axis(data::AxisName::Spin)) {
    blocks[data::axes::alpha()] = to_u32(indices);
    blocks[data::axes::beta()] = to_u32(indices);
  } else {
    blocks[data::SymmetryLabel{}] = to_u32(indices);
  }

  return std::make_shared<const data::SymmetryBlockedIndexSet>(
      symmetries, orbitals->mo_extents(), std::move(blocks));
}

void require_finite_setting(const char* setting_name, double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument(std::string("QICAS setting '") + setting_name +
                                "' must be finite.");
  }
}

}  // namespace

std::shared_ptr<data::Wavefunction> QICASActiveSpaceSelector::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction) const {
  QDK_LOG_TRACE_ENTERING();
  if (!wavefunction) {
    throw std::invalid_argument("QICAS requires a wavefunction");
  }

  const auto orbitals = wavefunction->get_orbitals();
  if (!orbitals->is_restricted()) {
    throw std::invalid_argument(
        "QICAS currently requires restricted spatial orbitals");
  }
  if (const auto symmetries = orbitals->symmetries()) {
    for (const auto& axis : symmetries->axes()) {
      if (axis.name() != data::AxisName::Spin) {
        throw std::invalid_argument(
            "QICAS currently does not support spatial symmetry blocks");
      }
    }
  }
  if (!orbitals->has_active_space()) {
    throw std::invalid_argument(
        "QICAS requires the correlated optimization window to be marked as "
        "the input active space");
  }
  if (!wavefunction->has_one_rdm_spin_dependent() ||
      !wavefunction->has_two_rdm_spin_dependent()) {
    throw std::invalid_argument(
        "QICAS requires spin-dependent active 1- and 2-RDMs over the "
        "optimization window");
  }

  const auto candidate_indices = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::alpha());
  const auto candidate_indices_beta = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::beta());
  const auto existing_inactive = data::spin_channel_indices(
      orbitals->inactive_indices(), data::axes::alpha());
  const auto existing_inactive_beta = data::spin_channel_indices(
      orbitals->inactive_indices(), data::axes::beta());
  if (candidate_indices != candidate_indices_beta ||
      existing_inactive != existing_inactive_beta) {
    throw std::invalid_argument(
        "QICAS requires matching alpha and beta orbital partitions");
  }
  if (candidate_indices.size() < 2) {
    throw std::invalid_argument(
        "QICAS requires at least two candidate orbitals");
  }

  const auto requested_active_electrons =
      _settings->get<int64_t>("num_active_electrons");
  const auto requested_active_orbitals =
      _settings->get<int64_t>("num_active_orbitals");
  if (requested_active_electrons <= 0 || requested_active_electrons % 2 != 0) {
    throw std::invalid_argument(
        "QICAS num_active_electrons must be set to a positive even value");
  }
  if (requested_active_orbitals <= 0) {
    throw std::invalid_argument(
        "QICAS num_active_orbitals must be set to a positive value");
  }
  const auto target_orbitals =
      static_cast<std::size_t>(requested_active_orbitals);
  const auto target_electrons =
      static_cast<std::size_t>(requested_active_electrons);
  if (target_orbitals >= candidate_indices.size()) {
    throw std::invalid_argument(
        "QICAS target active space must be smaller than the candidate window");
  }
  if (target_electrons > 2 * target_orbitals) {
    throw std::invalid_argument(
        "QICAS active electron count exceeds the target orbital capacity");
  }

  const auto [window_alpha_electrons, window_beta_electrons] =
      wavefunction->get_active_num_electrons();
  if (window_alpha_electrons != window_beta_electrons) {
    throw std::invalid_argument(
        "QICAS currently requires a closed-shell correlated window");
  }
  const std::size_t target_spin_electrons = target_electrons / 2;
  if (target_spin_electrons > window_alpha_electrons) {
    throw std::invalid_argument(
        "QICAS target active electron count exceeds the window electron count");
  }
  const std::size_t newly_inactive_orbitals =
      window_alpha_electrons - target_spin_electrons;
  if (newly_inactive_orbitals + target_orbitals > candidate_indices.size()) {
    throw std::invalid_argument(
        "QICAS target electron and orbital counts do not fit the candidate "
        "window");
  }

  const auto* active_one_rdm =
      std::get_if<data::SymmetryBlockedTensor<2, double>>(
          &wavefunction->active_one_rdm());
  const auto* active_two_rdm =
      std::get_if<data::SymmetryBlockedTensor<4, double>>(
          &wavefunction->active_two_rdm());
  if (!active_one_rdm || !active_two_rdm) {
    throw std::invalid_argument("QICAS requires real-valued active RDMs");
  }

  const auto& rdm_alpha =
      active_one_rdm->block({data::axes::alpha(), data::axes::alpha()});
  const auto& rdm_beta =
      active_one_rdm->block({data::axes::beta(), data::axes::beta()});
  const auto& rdm_alpha_beta =
      active_two_rdm->block({data::axes::alpha(), data::axes::alpha(),
                             data::axes::beta(), data::axes::beta()});
  const std::size_t dimension = candidate_indices.size();
  if (rdm_alpha.rows() != static_cast<Eigen::Index>(dimension) ||
      rdm_alpha.cols() != static_cast<Eigen::Index>(dimension) ||
      rdm_beta.rows() != static_cast<Eigen::Index>(dimension) ||
      rdm_beta.cols() != static_cast<Eigen::Index>(dimension) ||
      static_cast<std::size_t>(rdm_alpha_beta.size()) !=
          dimension * dimension * dimension * dimension) {
    throw std::invalid_argument(
        "QICAS RDM dimensions must match the candidate window");
  }
  if (!rdm_alpha.allFinite() || !rdm_beta.allFinite() ||
      !std::all_of(rdm_alpha_beta.data(),
                   rdm_alpha_beta.data() + rdm_alpha_beta.size(),
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument("QICAS RDMs must contain only finite values");
  }

  std::vector<std::size_t> objective_indices;
  std::vector<bool> objective_mask(dimension, false);
  for (std::size_t i = 0; i < dimension; ++i) {
    if (i < newly_inactive_orbitals ||
        i >= newly_inactive_orbitals + target_orbitals) {
      objective_indices.push_back(i);
      objective_mask[i] = true;
    }
  }

  std::vector<qio::detail::OrbitalPair> rotation_pairs;
  for (std::size_t i = 0; i < dimension; ++i) {
    for (std::size_t j = i + 1; j < dimension; ++j) {
      // Rotations entirely inside the target subspace leave F_QI invariant and
      // only change its internal gauge, so optimize pairs touching a
      // non-target slot.
      if (objective_mask[i] || objective_mask[j]) {
        rotation_pairs.push_back({i, j});
      }
    }
  }

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

  std::vector<double> rdm_alpha_beta_flat(
      rdm_alpha_beta.data(), rdm_alpha_beta.data() + rdm_alpha_beta.size());
  auto optimization = qio::detail::optimize_rotation(
      Eigen::MatrixXd(rdm_alpha), Eigen::MatrixXd(rdm_beta),
      std::move(rdm_alpha_beta_flat), dimension, objective_indices,
      rotation_pairs, max_cycles, convergence_tolerance, coarse_angle_step,
      fine_samples, improvement_tolerance);
  if (!optimization.converged) {
    QDK_LOGGER().warn(
        "QICAS Jacobi optimization reached max_cycles without convergence");
  }

  const Eigen::VectorXd occupations =
      optimization.rdm_alpha.diagonal() + optimization.rdm_beta.diagonal();
  std::vector<std::size_t> occupation_order(dimension);
  std::iota(occupation_order.begin(), occupation_order.end(), 0);
  std::stable_sort(occupation_order.begin(), occupation_order.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     return occupations(static_cast<Eigen::Index>(lhs)) >
                            occupations(static_cast<Eigen::Index>(rhs));
                   });

  const auto& coefficients = orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  Eigen::MatrixXd window_coefficients(coefficients.rows(),
                                      static_cast<Eigen::Index>(dimension));
  for (std::size_t i = 0; i < dimension; ++i) {
    window_coefficients.col(static_cast<Eigen::Index>(i)) = coefficients.col(
        static_cast<Eigen::Index>(candidate_indices[static_cast<size_t>(i)]));
  }
  const Eigen::MatrixXd optimized_window =
      window_coefficients * optimization.rotation;
  Eigen::MatrixXd optimized_coefficients = coefficients;
  // Physically place the optimized orbitals into fixed candidate destination
  // slots in descending occupation order. The partition metadata below names
  // these destination slots, not the pre-optimization orbital identities.
  for (std::size_t slot = 0; slot < dimension; ++slot) {
    optimized_coefficients.col(
        static_cast<Eigen::Index>(candidate_indices[slot])) =
        optimized_window.col(static_cast<Eigen::Index>(occupation_order[slot]));
  }

  std::vector<std::size_t> selected_active(
      candidate_indices.begin() +
          static_cast<std::ptrdiff_t>(newly_inactive_orbitals),
      candidate_indices.begin() +
          static_cast<std::ptrdiff_t>(newly_inactive_orbitals +
                                      target_orbitals));
  std::vector<std::size_t> selected_inactive = existing_inactive;
  selected_inactive.insert(
      selected_inactive.end(), candidate_indices.begin(),
      candidate_indices.begin() +
          static_cast<std::ptrdiff_t>(newly_inactive_orbitals));
  std::sort(selected_inactive.begin(), selected_inactive.end());

  std::optional<Eigen::MatrixXd> overlap;
  if (orbitals->has_overlap_matrix()) {
    overlap = orbitals->get_overlap_matrix();
  }
  std::shared_ptr<data::BasisSet> basis_set;
  if (orbitals->has_basis_set()) {
    basis_set = orbitals->get_basis_set();
  }
  auto selected_orbitals = std::make_shared<data::Orbitals>(
      optimized_coefficients, std::nullopt, overlap, basis_set,
      make_orbital_index_set(orbitals, selected_active),
      make_orbital_index_set(orbitals, selected_inactive));

  QDK_LOGGER().info(
      "QICAS selected {} active orbitals after reducing F_QI from {:.8e} to "
      "{:.8e}",
      selected_active.size(), optimization.initial_objective,
      optimization.final_objective);
  return detail::new_wavefunction(wavefunction, selected_orbitals);
}

}  // namespace qdk::chemistry::algorithms::microsoft

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "gauge_fixing.hpp"

#include <Eigen/Eigenvalues>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <unordered_map>
#include <variant>

namespace qdk::chemistry::algorithms::microsoft {

namespace detail {

Eigen::MatrixXd ao_anchor_block(const Eigen::MatrixXd& block,
                                const Eigen::MatrixXd& overlap) {
  QDK_LOG_TRACE_ENTERING();
  const Eigen::MatrixXd projected_ao = overlap * block;
  Eigen::MatrixXd residuals = projected_ao;

  std::vector<Eigen::Index> anchors;
  anchors.reserve(static_cast<size_t>(block.cols()));
  for (Eigen::Index column = 0; column < block.cols(); ++column) {
    // Rounding keeps the argmax tie-break independent of last-bit noise, which
    // is what makes symmetry-equivalent atomic orbitals resolve consistently.
    Eigen::Index anchor = 0;
    double best = -1.0;
    for (Eigen::Index row = 0; row < residuals.rows(); ++row) {
      const double norm =
          std::round(residuals.row(row).squaredNorm() * 1e14) * 1e-14;
      if (norm > best) {
        best = norm;
        anchor = row;
      }
    }
    anchors.push_back(anchor);

    Eigen::VectorXd anchor_vector = residuals.row(anchor).transpose();
    const double anchor_norm = anchor_vector.norm();
    if (anchor_norm <= std::numeric_limits<double>::epsilon()) {
      throw std::runtime_error(
          "Unable to find independent AO anchors for a degenerate orbital "
          "block");
    }
    anchor_vector /= anchor_norm;
    residuals -= (residuals * anchor_vector) * anchor_vector.transpose();
  }

  Eigen::MatrixXd anchor_coefficients(block.cols(), block.cols());
  for (size_t i = 0; i < anchors.size(); ++i) {
    anchor_coefficients.row(static_cast<Eigen::Index>(i)) =
        projected_ao.row(anchors[i]);
  }
  anchor_coefficients.transposeInPlace();

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> gram(
      anchor_coefficients.transpose() * anchor_coefficients);
  if (gram.info() != Eigen::Success) {
    throw std::runtime_error(
        "Failed to orthogonalize the AO anchors of a degenerate orbital block");
  }
  if (gram.eigenvalues().minCoeff() <= std::numeric_limits<double>::epsilon()) {
    throw std::runtime_error(
        "AO anchors for a degenerate orbital block are linearly dependent");
  }

  const Eigen::MatrixXd orthogonalizer =
      anchor_coefficients * gram.eigenvectors() *
      gram.eigenvalues().cwiseInverse().cwiseSqrt().asDiagonal() *
      gram.eigenvectors().transpose();
  return block * orthogonalizer;
}

std::pair<double, double> golden_section_minimum(
    const std::function<double(double)>& objective, double lower_bound,
    double upper_bound, double argument_tolerance) {
  QDK_LOG_TRACE_ENTERING();
  if (upper_bound <= lower_bound) {
    throw std::invalid_argument("upper_bound must be greater than lower_bound");
  }
  if (argument_tolerance <= 0.0) {
    throw std::invalid_argument("argument_tolerance must be positive");
  }

  const double inverse_golden_ratio = (std::sqrt(5.0) - 1.0) / 2.0;
  double left = lower_bound;
  double right = upper_bound;
  double inner_left = right - inverse_golden_ratio * (right - left);
  double inner_right = left + inverse_golden_ratio * (right - left);
  double value_left = objective(inner_left);
  double value_right = objective(inner_right);

  while (right - left > argument_tolerance) {
    if (value_left <= value_right) {
      right = inner_right;
      inner_right = inner_left;
      value_right = value_left;
      inner_left = right - inverse_golden_ratio * (right - left);
      value_left = objective(inner_left);
    } else {
      left = inner_left;
      inner_left = inner_right;
      value_left = value_right;
      inner_right = left + inverse_golden_ratio * (right - left);
      value_right = objective(inner_right);
    }
  }

  const double midpoint = (left + right) / 2.0;
  std::pair<double, double> best{inner_left, value_left};
  for (const std::pair<double, double>& candidate :
       {std::pair<double, double>{inner_right, value_right},
        std::pair<double, double>{midpoint, objective(midpoint)}}) {
    if (candidate.second < best.second ||
        (candidate.second == best.second && candidate.first < best.first)) {
      best = candidate;
    }
  }
  return best;
}

namespace {

std::shared_ptr<const data::SymmetryBlockedIndexSet> make_orbital_index_set(
    const std::shared_ptr<data::Orbitals>& orbitals,
    const std::vector<size_t>& selected) {
  const std::vector<std::uint32_t> selected_u32(selected.begin(),
                                                selected.end());
  std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>> indices;
  auto symmetries = orbitals->symmetries();
  if (symmetries && symmetries->has_axis(data::AxisName::Spin)) {
    indices[data::axes::alpha()] = selected_u32;
    indices[data::axes::beta()] = selected_u32;
  } else {
    indices[data::SymmetryLabel{}] = selected_u32;
  }
  return std::make_shared<const data::SymmetryBlockedIndexSet>(
      symmetries, orbitals->mo_extents(), std::move(indices));
}

}  // namespace

}  // namespace detail

std::shared_ptr<data::Wavefunction> GaugeFixingLocalizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& loc_indices_a,
    const std::vector<size_t>& loc_indices_b) const {
  QDK_LOG_TRACE_ENTERING();
  auto orbitals = wavefunction->get_orbitals();

  if (!std::is_sorted(loc_indices_a.begin(), loc_indices_a.end())) {
    throw std::invalid_argument("loc_indices_a must be sorted");
  }
  if (!std::is_sorted(loc_indices_b.begin(), loc_indices_b.end())) {
    throw std::invalid_argument("loc_indices_b must be sorted");
  }
  if (loc_indices_a != loc_indices_b) {
    throw std::invalid_argument(
        "loc_indices_a and loc_indices_b must be identical");
  }
  if (!orbitals->is_restricted()) {
    throw std::invalid_argument(
        "GaugeFixingLocalizer requires restricted orbitals; run a natural "
        "orbital localizer first.");
  }
  if (loc_indices_a.empty()) {
    return algorithms::detail::new_aufbau_determinant_wavefunction(wavefunction,
                                                                   orbitals);
  }
  if (!orbitals->has_overlap_matrix()) {
    throw std::invalid_argument(
        "GaugeFixingLocalizer requires an overlap matrix to be available in "
        "the orbitals.");
  }
  if (!orbitals->has_active_space()) {
    throw std::invalid_argument(
        "GaugeFixingLocalizer requires an active space to be defined in the "
        "orbitals.");
  }
  if (!wavefunction->has_one_rdm_spin_traced()) {
    throw std::invalid_argument(
        "GaugeFixingLocalizer requires an active-space 1-RDM in the "
        "wavefunction.");
  }

  const auto active_indices = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::alpha());
  const auto& rdm_variant = wavefunction->get_active_one_rdm_spin_traced();
  const auto* one_rdm = std::get_if<Eigen::MatrixXd>(&rdm_variant);
  if (!one_rdm) {
    throw std::invalid_argument(
        "GaugeFixingLocalizer requires a real-valued active 1-RDM.");
  }
  if (static_cast<size_t>(one_rdm->rows()) != active_indices.size()) {
    throw std::invalid_argument(
        "1-RDM dimensions do not match the orbitals' active-space size.");
  }

  const double degeneracy_tolerance =
      _settings->get<double>("degeneracy_tolerance");
  const Eigen::VectorXd occupations = one_rdm->diagonal();
  // Off-diagonal weight above the degeneracy tolerance can reorder occupations
  // by more than the tolerance, which would make the blocks meaningless.
  if ((*one_rdm - Eigen::MatrixXd(occupations.asDiagonal()))
          .cwiseAbs()
          .maxCoeff() >= degeneracy_tolerance) {
    throw std::invalid_argument(
        "The active one-particle RDM is not diagonal in the input orbital "
        "basis, so occupation degeneracies are undefined. Run the "
        "'qdk_natural_orbitals' localizer before gauge fixing.");
  }

  std::unordered_map<size_t, double> occupation_by_orbital;
  for (size_t position = 0; position < active_indices.size(); ++position) {
    occupation_by_orbital[active_indices[position]] =
        occupations[static_cast<Eigen::Index>(position)];
  }
  for (size_t index : loc_indices_a) {
    if (!occupation_by_orbital.contains(index)) {
      throw std::invalid_argument(
          "Every orbital to gauge fix must belong to the active space of the "
          "input wavefunction.");
    }
  }

  // Blocks partition the whole active space, but only wholly selected ones may
  // be rotated: anchoring a block whose members straddle the selection
  // boundary would change the selected subspace, and so the energy.
  std::vector<size_t> occupation_order(active_indices.size());
  std::iota(occupation_order.begin(), occupation_order.end(), size_t{0});
  std::stable_sort(occupation_order.begin(), occupation_order.end(),
                   [&occupations](size_t left, size_t right) {
                     return occupations[static_cast<Eigen::Index>(left)] >
                            occupations[static_cast<Eigen::Index>(right)];
                   });

  const std::vector<size_t> selected(loc_indices_a.begin(),
                                     loc_indices_a.end());
  std::vector<std::vector<size_t>> selected_blocks;
  size_t block_start = 0;
  for (size_t block_stop = 1; block_stop <= occupation_order.size();
       ++block_stop) {
    if (block_stop < occupation_order.size() &&
        std::abs(occupations[static_cast<Eigen::Index>(
                     occupation_order[block_stop])] -
                 occupations[static_cast<Eigen::Index>(
                     occupation_order[block_stop - 1])]) <
            degeneracy_tolerance) {
      continue;
    }
    std::vector<size_t> block;
    for (size_t position = block_start; position < block_stop; ++position) {
      block.push_back(active_indices[occupation_order[position]]);
    }
    std::sort(block.begin(), block.end());
    block_start = block_stop;

    const size_t members = static_cast<size_t>(
        std::count_if(block.begin(), block.end(), [&selected](size_t index) {
          return std::binary_search(selected.begin(), selected.end(), index);
        }));
    if (members != 0 && members != block.size()) {
      throw std::runtime_error(
          "The selected orbitals split an occupation-degenerate block");
    }
    if (members != 0) {
      selected_blocks.push_back(std::move(block));
    }
  }

  const Eigen::MatrixXd& overlap = orbitals->get_overlap_matrix();
  Eigen::MatrixXd coefficients = orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});

  for (const auto& block : selected_blocks) {
    Eigen::MatrixXd block_coefficients(coefficients.rows(),
                                       static_cast<Eigen::Index>(block.size()));
    for (size_t i = 0; i < block.size(); ++i) {
      block_coefficients.col(static_cast<Eigen::Index>(i)) =
          coefficients.col(static_cast<Eigen::Index>(block[i]));
    }
    const Eigen::MatrixXd anchored =
        detail::ao_anchor_block(block_coefficients, overlap);
    for (size_t i = 0; i < block.size(); ++i) {
      coefficients.col(static_cast<Eigen::Index>(block[i])) =
          anchored.col(static_cast<Eigen::Index>(i));
    }
  }

  const auto active_index_set =
      detail::make_orbital_index_set(orbitals, loc_indices_a);
  const auto inactive_index_set = detail::make_orbital_index_set(
      orbitals, data::spin_channel_indices(orbitals->inactive_indices(),
                                           data::axes::alpha()));

  const double mapper_threshold = _settings->get<double>("mapper_threshold");
  auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
  const data::MajoranaMapping mapping =
      data::MajoranaMapping::jordan_wigner(2 * loc_indices_a.size());

  auto coefficient_norm = [&](const Eigen::MatrixXd& candidate) {
    auto candidate_orbitals = std::make_shared<data::Orbitals>(
        candidate, std::nullopt, overlap, orbitals->get_basis_set(),
        active_index_set, inactive_index_set);
    auto hamiltonian = hamiltonian_constructor->run(candidate_orbitals);
    const auto mapped = data::majorana_map_hamiltonian(
        mapping, *hamiltonian, /*spin_symmetric=*/true, mapper_threshold,
        mapper_threshold);
    double norm = 0.0;
    for (const auto& coefficient : mapped.coefficients) {
      norm += std::abs(coefficient);
    }
    return norm;
  };

  const double norm_before = coefficient_norm(coefficients);
  double current_norm = norm_before;
  const auto angle_samples =
      static_cast<size_t>(_settings->get<int64_t>("angle_samples"));
  const double improvement_tolerance =
      _settings->get<double>("improvement_tolerance");
  const double angle_step = M_PI / static_cast<double>(angle_samples);
  const auto max_sweeps =
      static_cast<size_t>(_settings->get<int64_t>("max_sweeps"));

  for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
    const double sweep_start_norm = current_norm;
    for (const auto& block : selected_blocks) {
      for (size_t i = 0; i + 1 < block.size(); ++i) {
        for (size_t j = i + 1; j < block.size(); ++j) {
          const auto left = static_cast<Eigen::Index>(block[i]);
          const auto right = static_cast<Eigen::Index>(block[j]);
          Eigen::MatrixXd candidate = coefficients;
          const Eigen::VectorXd left_column = coefficients.col(left);
          const Eigen::VectorXd right_column = coefficients.col(right);

          auto plane_norm = [&](double angle) {
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            candidate.col(left) = cosine * left_column + sine * right_column;
            candidate.col(right) = cosine * right_column - sine * left_column;
            return coefficient_norm(candidate);
          };

          double best_angle = 0.0;
          double best_norm = std::numeric_limits<double>::infinity();
          for (size_t sample = 0; sample < angle_samples; ++sample) {
            const double angle = static_cast<double>(sample) * angle_step;
            const double norm = plane_norm(angle);
            if (norm < best_norm) {
              best_norm = norm;
              best_angle = angle;
            }
          }

          const auto [refined_angle, refined_norm] =
              detail::golden_section_minimum(
                  plane_norm, best_angle - angle_step, best_angle + angle_step);
          if (refined_norm < current_norm - improvement_tolerance) {
            // Rotations by an angle and by that angle plus pi differ only in
            // orbital sign; canonicalizing onto [0, pi) keeps the accepted
            // gauge reproducible.
            double canonical_angle = std::fmod(refined_angle, M_PI);
            if (canonical_angle < 0.0) {
              canonical_angle += M_PI;
            }
            current_norm = plane_norm(canonical_angle);
            coefficients.col(left) = candidate.col(left);
            coefficients.col(right) = candidate.col(right);
          }
        }
      }
    }

    if (current_norm >= sweep_start_norm - improvement_tolerance) {
      break;
    }
  }

  QDK_LOGGER().info(
      "Gauge fixing changed the mapped coefficient norm from {} to {} Hartree",
      norm_before, current_norm);

  auto gauge_fixed_orbitals = std::make_shared<data::Orbitals>(
      coefficients,
      std::nullopt,  // rotations inside a correlated active space carry no
                     // unique one-electron energies
      overlap, orbitals->get_basis_set(), active_index_set, inactive_index_set);

  // Rotations stay inside degenerate eigenspaces, so the RDM restricted to the
  // selected orbitals is unchanged. Carrying it keeps the occupations
  // available downstream and makes a second gauge-fixing run a no-op.
  Eigen::VectorXd selected_occupations(
      static_cast<Eigen::Index>(loc_indices_a.size()));
  for (size_t i = 0; i < loc_indices_a.size(); ++i) {
    selected_occupations[static_cast<Eigen::Index>(i)] =
        occupation_by_orbital.at(loc_indices_a[i]);
  }
  Eigen::MatrixXd selected_one_rdm = selected_occupations.asDiagonal();
  return algorithms::detail::new_aufbau_determinant_wavefunction(
      wavefunction, gauge_fixed_orbitals,
      data::ContainerTypes::MatrixVariant(selected_one_rdm));
}

}  // namespace qdk::chemistry::algorithms::microsoft

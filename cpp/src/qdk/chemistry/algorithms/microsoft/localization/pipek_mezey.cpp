// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "pipek_mezey.hpp"

#include <algorithm>
#include <blas.hh>
#include <iostream>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "../utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Wavefunction> PipekMezeyLocalizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& loc_indices_a,
    const std::vector<size_t>& loc_indices_b) const {
  QDK_LOG_TRACE_ENTERING();
  auto orbitals = wavefunction->get_orbitals();

  // If both index vectors are empty, return original orbitals unchanged
  if (loc_indices_a.size() == 0 && loc_indices_b.size() == 0) {
    return wavefunction;
  }

  // Early validation: Check that indices are sorted
  if (!std::is_sorted(loc_indices_a.begin(), loc_indices_a.end())) {
    throw std::invalid_argument("loc_indices_a must be sorted");
  }
  if (!std::is_sorted(loc_indices_b.begin(), loc_indices_b.end())) {
    throw std::invalid_argument("loc_indices_b must be sorted");
  }

  // Early validation: Check orbital indices are valid
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();
  if (!loc_indices_a.empty() &&
      loc_indices_a.back() >= num_molecular_orbitals) {
    throw std::invalid_argument(
        "loc_indices_a contains invalid orbital index >= "
        "num_molecular_orbitals");
  }
  if (!loc_indices_b.empty() &&
      loc_indices_b.back() >= num_molecular_orbitals) {
    throw std::invalid_argument(
        "loc_indices_b contains invalid orbital index >= "
        "num_molecular_orbitals");
  }

  // Early validation: For restricted orbitals, ensure indices are identical
  if (orbitals->is_restricted() && !(loc_indices_a == loc_indices_b)) {
    throw std::invalid_argument(
        "For restricted orbitals, loc_indices_a and loc_indices_b must be "
        "identical");
  }

  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  IterativeOrbitalLocalizationSettings settings;

  // Generate map from atomic orbital index to atom index
  const auto& basis_set = orbitals->get_basis_set();
  const size_t num_atomic_orbitals = basis_set->get_num_atomic_orbitals();
  const size_t num_atoms = basis_set->get_structure()->get_num_atoms();
  std::vector<int> atomic_orbital_to_atom_map(num_atomic_orbitals);
  for (size_t i = 0; i < num_atomic_orbitals; ++i) {
    atomic_orbital_to_atom_map[i] =
        basis_set->get_atom_index_for_atomic_orbital(i);
  }

  // Create localizer outside do_loc for reuse
  const auto& ao_overlap = orbitals->get_overlap_matrix();
  PipekMezeyLocalization localizer(settings, ao_overlap, num_atoms,
                                   atomic_orbital_to_atom_map);

  auto do_loc = [&](const auto& coeffs, const auto& ind) {
    Eigen::MatrixXd target_coeffs = coeffs;  // Start with original coefficients
    if (!ind.empty()) {
      // Extract MOs
      Eigen::MatrixXd C_ind(coeffs.rows(), ind.size());
      for (int i = 0; i < ind.size(); ++i) {
        C_ind.col(i) = coeffs.col(ind[i]);
      }
      // Perform the localization
      auto localized = localizer.localize(C_ind);
      // Apply localized orbitals back to target
      for (size_t i = 0; i < ind.size(); ++i) {
        target_coeffs.col(ind[i]) = localized.col(i);
      }
    }
    return target_coeffs;
  };

  // Construct the localized orbitals object
  auto [coeffs_alpha, coeffs_beta] = orbitals->get_coefficients();

  if (not orbitals->is_restricted()) {
    // Alpha spin channel - localize selected orbitals
    Eigen::MatrixXd C_alpha = do_loc(coeffs_alpha, loc_indices_a);

    // Beta spin channel - localize selected orbitals
    Eigen::MatrixXd C_beta = do_loc(coeffs_beta, loc_indices_b);

    // Preserve active space indices from input orbitals if they exist
    std::optional<data::Orbitals::UnrestrictedCASIndices> unrestricted_indices;
    if (orbitals->has_active_space()) {
      auto [active_a, active_b] = orbitals->get_active_space_indices();
      auto [inactive_a, inactive_b] = orbitals->get_inactive_space_indices();
      // Order: (active_alpha, active_beta, inactive_alpha, inactive_beta)
      unrestricted_indices = std::make_tuple(
          std::vector<size_t>(active_a.begin(), active_a.end()),
          std::vector<size_t>(active_b.begin(), active_b.end()),
          std::vector<size_t>(inactive_a.begin(), inactive_a.end()),
          std::vector<size_t>(inactive_b.begin(), inactive_b.end()));
    }

    auto new_orbitals = std::make_shared<data::Orbitals>(
        C_alpha, C_beta, std::nullopt,
        std::nullopt,           // no energies for localized orbitals
        ao_overlap,             // Atomic Orbital overlap
        basis_set,              // basis set
        unrestricted_indices);  // preserve active space indices from input
    return detail::new_wavefunction(wavefunction, new_orbitals);
  } else {
    // Localize selected orbitals
    Eigen::MatrixXd C_lmo = do_loc(coeffs_alpha, loc_indices_a);

    // Preserve active space indices from input orbitals if they exist
    std::optional<data::Orbitals::RestrictedCASIndices> restricted_indices;
    if (orbitals->has_active_space()) {
      auto [active_a, active_b] = orbitals->get_active_space_indices();
      auto [inactive_a, inactive_b] = orbitals->get_inactive_space_indices();
      restricted_indices = std::make_tuple(
          std::vector<size_t>(active_a.begin(), active_a.end()),
          std::vector<size_t>(inactive_a.begin(), inactive_a.end()));
    }

    auto new_orbitals = std::make_shared<data::Orbitals>(
        C_lmo,
        std::nullopt,         // no energies for localized orbitals
        ao_overlap,           // Atomic Orbital overlap
        basis_set,            // basis set
        restricted_indices);  // preserve active space indices from input
    return detail::new_wavefunction(wavefunction, new_orbitals);
  }
}

// Compute Jacobi rotation
// [w1] = [ c s] [v1]
// [w2] = [-s c] [v2]
// where c = cos(a) s = sin(a)
template <typename VectorType>
void jacobi_rotation(VectorType&& v1, VectorType&& v2, long double angle) {
  QDK_LOG_TRACE_ENTERING();

  const double c = std::cos(angle);
  const double s = std::sin(angle);
  const size_t n = v1.size();
#pragma omp simd
  for (size_t i = 0; i < n; ++i) {
    const auto t1 = v1[i];
    const auto t2 = v2[i];
    v1[i] = c * t1 + s * t2;
    v2[i] = c * t2 - s * t1;
  }
}

// Compute the Jacobi rotation parameters according to
// Edmiston & Ruedenberg (1963), doi:10.1103/RevModPhys.35.457
// INC Eq(23)
// ROT Eq(19)
auto compute_jacobi_AB(long double A, long double B) {
  QDK_LOG_TRACE_ENTERING();

  double inc = A + std::hypot(A, B);
  double rot = 0.25 * std::atan2(B, -A);
  return std::make_pair(inc, rot);
};

PipekMezeyLocalization::PipekMezeyLocalization(
    IterativeOrbitalLocalizationSettings settings,
    const Eigen::MatrixXd& overlap_matrix, size_t num_atoms,
    std::vector<int> ao_to_atom_map)
    : IterativeOrbitalLocalizationScheme(settings),
      overlap_matrix_(overlap_matrix),
      ao_to_atom_map_(std::move(ao_to_atom_map)),
      num_atoms_(num_atoms) {
  QDK_LOG_TRACE_ENTERING();
}

Eigen::MatrixXd PipekMezeyLocalization::localize(
    const Eigen::MatrixXd& initial_orbitals) {
  QDK_LOG_TRACE_ENTERING();

  using MatrixType = Eigen::MatrixXd;

  // Work with local copy of orbitals
  Eigen::MatrixXd orbitals = initial_orbitals;

  // Reset convergence diagnostics for this localization run
  this->converged_ = false;
  this->obj_fun_ = 0.0;

  const auto num_atomic_orbitals = orbitals.rows();
  const auto num_orbitals = orbitals.cols();
  MatrixType orbital_coeffs = orbitals;
  MatrixType overlap_matrix = this->overlap_matrix_;

  MatrixType overlap_times_coeffs =
      MatrixType::Zero(num_atomic_orbitals, num_orbitals);
  MatrixType Xi = MatrixType::Zero(num_atoms_, num_orbitals);
  MatrixType gamma = MatrixType::Zero(num_orbitals, num_orbitals);
  MatrixType increase_sos = MatrixType::Zero(num_orbitals, num_orbitals);

  double old_metric = std::numeric_limits<double>::infinity();

  // Initial overlap_matrix*orbital_coeffs - updated via Jacobi rotations
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             num_atomic_orbitals, num_orbitals, num_atomic_orbitals, 1.0,
             overlap_matrix.data(), num_atomic_orbitals, orbital_coeffs.data(),
             num_atomic_orbitals, 0.0, overlap_times_coeffs.data(),
             num_atomic_orbitals);

  const auto max_sweeps = this->settings_.get<int64_t>("max_iterations");
  const auto tol = this->settings_.get<double>("tolerance");
  const auto AB_tol = this->settings_.get<double>("small_rotation_tolerance");
  size_t i_sweep;
  for (i_sweep = 0; i_sweep < max_sweeps; ++i_sweep) {
    // Compute Xi
    Xi.setZero();
    for (auto p = 0; p < num_orbitals; ++p)
      for (auto mu = 0; mu < num_atomic_orbitals; ++mu) {
        Xi(ao_to_atom_map_[mu], p) +=
            orbital_coeffs(mu, p) * overlap_times_coeffs(mu, p);
      }

    // Compute metric
    double metric = Xi.cwiseProduct(Xi).sum();
    auto delta = std::abs(old_metric - metric);

    // Check convergence
    if (delta < tol) {
      this->converged_ = true;
      break;
    }
    old_metric = metric;

    // Compute Gamma
    gamma.setZero();
    increase_sos.setZero();
#pragma omp parallel for collapse(2)
    for (auto s = 0; s < num_orbitals; ++s)
      for (auto t = 0; t < s; ++t) {
        // Compute Q
        Eigen::VectorXd Q = Eigen::VectorXd::Zero(num_atoms_);
        for (auto mu = 0; mu < num_atomic_orbitals; mu++) {
          Q[ao_to_atom_map_[mu]] +=
              orbital_coeffs(mu, s) * overlap_times_coeffs(mu, t) +
              overlap_times_coeffs(mu, s) * orbital_coeffs(mu, t);
        }
        Q *= 0.5;

        // Compute A/B
        double A_st = 0;
        double B_st = 0;
        for (auto A = 0; A < num_atoms_; ++A) {
          B_st += Q(A) * (Xi(A, s) - Xi(A, t));
          A_st += std::pow(Q(A), 2) - 0.25 * std::pow(Xi(A, s) - Xi(A, t), 2);
        }

        // Compute gamma
        if (std::abs(B_st) < AB_tol) B_st = 0.0;
        if (std::abs(A_st) < AB_tol) A_st = 0.0;
        std::tie(increase_sos(s, t), gamma(s, t)) =
            compute_jacobi_AB(A_st, B_st);
      }

    // Get max angle and change
    int s_max, t_max;
    double max_sos = increase_sos.maxCoeff(&s_max, &t_max);
    auto max_gamma = gamma(s_max, t_max);

    if (std::abs(max_gamma) < tol or std::abs(max_sos) < tol) {
      this->converged_ = true;
      break;
    }

    // Rotate cols of orbital_coeffs and overlap_times_coeffs
    jacobi_rotation(orbital_coeffs.col(s_max), orbital_coeffs.col(t_max),
                    max_gamma);
    jacobi_rotation(overlap_times_coeffs.col(s_max),
                    overlap_times_coeffs.col(t_max), max_gamma);

    std::vector<bool> was_rotated(num_orbitals, false);
    was_rotated[s_max] = true;
    was_rotated[t_max] = true;

    for (auto i = 0; i < num_orbitals; ++i) {
      if (was_rotated[i]) continue;
      increase_sos.row(i).maxCoeff(&t_max);

      auto max_gamma = gamma(i, t_max);
      jacobi_rotation(orbital_coeffs.col(i), orbital_coeffs.col(t_max),
                      max_gamma);
      jacobi_rotation(overlap_times_coeffs.col(i),
                      overlap_times_coeffs.col(t_max), max_gamma);
      was_rotated[i] = true;
      was_rotated[t_max] = true;
    }
  }

  // Compute final metric
  Xi.setZero();
  for (auto p = 0; p < num_orbitals; ++p)
    for (auto mu = 0; mu < num_atomic_orbitals; ++mu) {
      Xi(ao_to_atom_map_[mu], p) +=
          orbital_coeffs(mu, p) * overlap_times_coeffs(mu, p);
    }
  this->obj_fun_ = Xi.cwiseProduct(Xi).sum();
  if (this->converged_)
    QDK_LOGGER().info(
        "Pipek-Mezey Converged in {:6} sweeps with ObjectiveFunction = "
        "{:.6e}",
        i_sweep, this->obj_fun_);
  else
    QDK_LOGGER().info(
        "Pipek-Mezey Failed to Converge in {:6} sweeps - Last "
        "ObjectiveFunction = {:.6e}",
        i_sweep, this->obj_fun_);

  return orbital_coeffs;
}

}  // namespace qdk::chemistry::algorithms::microsoft

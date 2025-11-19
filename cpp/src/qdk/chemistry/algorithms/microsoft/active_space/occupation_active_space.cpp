// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "occupation_active_space.hpp"

#include <algorithm>
#include <numeric>
#include <qdk/chemistry/data/structure.hpp>

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Wavefunction> OccupationActiveSpaceSelector::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction) const {
  // If orbitals already have an active space, we'll downselect from it
  // If not, we'll work with all orbitals

  auto orbitals = wavefunction->get_orbitals();

  // Sanity checks
  if (not orbitals->is_restricted()) {
    throw std::runtime_error(
        "OccupationActiveSpaceSelector only supports restricted orbitals.");
  }

  // Get the occupation threshold from settings
  double occupation_threshold = _settings->get<double>("occupation_threshold");

  // Get occupations
  auto [alpha_occupations, beta_occupations] =
      wavefunction->get_total_orbital_occupations();
  Eigen::VectorXd occupations = alpha_occupations + beta_occupations;
  auto [nalpha, nbeta] = wavefunction->get_total_num_electrons();

  // orbitals start with all indices active, this only fails if orbitals have a
  // modified active space
  if (!orbitals->has_active_space()) {
    throw std::runtime_error(
        "No candidate active space available in orbitals for occupation-based "
        "selection.");
  }
  std::vector<size_t> candidate_indices =
      orbitals->get_active_space_indices().first;

  // Make sure that the occupation numbers are sorted for the candidate orbitals
  std::vector<std::pair<size_t, double>> orbital_occupations;
  for (size_t idx : candidate_indices) {
    orbital_occupations.emplace_back(idx, occupations[idx]);
  }

  // Sort by occupation in descending order
  std::sort(orbital_occupations.begin(), orbital_occupations.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });

  std::vector<size_t> active_space_indices;
  for (const auto& [orbital_idx, occ] : orbital_occupations) {
    const auto occ_minus_two = std::abs(occ - 2.0);
    const auto occ_min_diff = std::min(occ, occ_minus_two);
    if (occ_min_diff >= occupation_threshold) {
      active_space_indices.push_back(orbital_idx);
    }
  }

  // get inactive indices
  size_t nelec = wavefunction->get_total_num_electrons().first +
                 wavefunction->get_total_num_electrons().second;
  if (nelec % 2 != 0) {
    throw std::runtime_error(
        "OccupationActiveSpaceSelector only supports even number of "
        "electrons.");
  }
  std::vector<size_t> inactive_indices;
  for (size_t i = 0; i < nelec / 2; ++i) {
    inactive_indices.push_back(i);
  }

  // ensure inactive indices are not in active space
  inactive_indices.erase(
      std::remove_if(inactive_indices.begin(), inactive_indices.end(),
                     [&](size_t idx) {
                       return std::find(active_space_indices.begin(),
                                        active_space_indices.end(),
                                        idx) != active_space_indices.end();
                     }),
      inactive_indices.end());

  if (active_space_indices.empty()) {
    throw std::runtime_error(
        "No orbitals selected for active space based on occupation threshold.");
  }

  // Create new orbitals with the selected active space indices
  auto new_orbitals = detail::new_orbitals(wavefunction, active_space_indices);
  return detail::new_wavefunction(wavefunction, new_orbitals);
}

}  // namespace qdk::chemistry::algorithms::microsoft

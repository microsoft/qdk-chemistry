// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "valence_active_space.hpp"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Wavefunction> ValenceActiveSpaceSelector::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction) const {
  QDK_LOG_TRACE_ENTERING();
  QDK_LOGGER().info(
      "ValenceActiveSpaceSelector::Starting active space selection.");

  // If orbitals already have an active space, we'll downselect from it
  // If not, we'll work with all orbitals

  auto orbitals = wavefunction->get_orbitals();

  // Sanity check
  if (!orbitals->is_restricted()) {
    throw std::runtime_error(
        "ValenceActiveSpaceSelector only supports restricted orbitals.");
  }

  // Check for canonical orbitals
  if (!orbitals->has_energies()) {
    throw std::runtime_error(
        "Orbitals must be canonical (have orbital energies) for valence-based "
        "selection.");
  }

  // Get the number of electrons and active orbitals from settings
  int64_t num_active_electrons =
      _settings->get<int64_t>("num_active_electrons");
  int64_t num_active_orbitals = _settings->get<int64_t>("num_active_orbitals");
  QDK_LOGGER().debug("Settings:");
  QDK_LOGGER().debug("  num_active_electrons: {}", num_active_electrons);
  QDK_LOGGER().debug("  num_active_orbitals: {}", num_active_orbitals);

  // Validate settings
  if (num_active_electrons <= 0) {
    throw std::runtime_error("Number of electrons must be set and positive. ");
  }

  if (num_active_orbitals <= 0) {
    throw std::runtime_error(
        "Number of active orbitals must be set and positive. ");
  }

  if (num_active_orbitals >
      static_cast<int>(orbitals->get_num_molecular_orbitals())) {
    throw std::runtime_error(
        "Number of active orbitals exceeds total number of orbitals.");
  }

  // Get the orbitals to consider (existing active space or all orbitals)
  std::vector<size_t> candidate_indices;
  if (orbitals->has_active_space()) {
    candidate_indices = orbitals->get_active_space_indices().first;
  } else {
    candidate_indices.resize(orbitals->get_num_molecular_orbitals());
    std::iota(candidate_indices.begin(), candidate_indices.end(), 0);
  }
  QDK_LOGGER().debug("Number of candidate orbitals: {}",
                     candidate_indices.size());

  // Validate against candidate orbitals
  if (num_active_orbitals > static_cast<int>(candidate_indices.size())) {
    throw std::runtime_error(
        "Number of active orbitals exceeds available candidate orbitals.");
  }

  auto [nalpha, nbeta] = wavefunction->get_total_num_electrons();
  if (num_active_electrons > (nalpha + nbeta)) {
    throw std::runtime_error(
        "Number of active electrons exceeds total number of electrons.");
  }

  int n_inactive_electrons =
      int(std::round(nalpha)) + int(std::round(nbeta)) - num_active_electrons;
  if (n_inactive_electrons % 2) {
    throw std::runtime_error(
        "Number of inactive electrons must be even for a valid active space.");
  }
  int n_inactive_orbitals = n_inactive_electrons / 2;

  // fill inactive indices
  std::vector<size_t> inactive_indices;
  if (orbitals->has_active_space()) {
    // Append the newly selected inactive indices to any existing ones
    inactive_indices = orbitals->get_inactive_space_indices().first;
    for (int i = 0; i < n_inactive_orbitals; ++i) {
      inactive_indices.push_back(
          candidate_indices[i]);  // Fill from start of candidates
    }
  } else {
    for (int i = 0; i < n_inactive_orbitals; ++i) {
      inactive_indices.push_back(i);
    }
  }

  // Select from candidate indices starting from the appropriate offset
  std::vector<size_t> active_space_indices;
  for (int i = 0; i < num_active_orbitals; ++i) {
    if (n_inactive_orbitals + i < static_cast<int>(candidate_indices.size())) {
      active_space_indices.push_back(
          candidate_indices[n_inactive_orbitals + i]);
    }
  }

  if (active_space_indices.size() + inactive_indices.size() >
      static_cast<size_t>(orbitals->get_num_molecular_orbitals())) {
    throw std::runtime_error(
        "Sum of inactive and active orbitals exceeds available orbitals.");
  }
  if (active_space_indices.size() != static_cast<size_t>(num_active_orbitals)) {
    throw std::runtime_error(
        "Could not select the desired number of active orbitals.");
  }
  std::sort(active_space_indices.begin(), active_space_indices.end());
  std::ostringstream oss;
  for (size_t i = 0; i < active_space_indices.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << active_space_indices[i];
  }
  QDK_LOGGER().info(
      "ValenceActiveSpaceSelector::Selected active space of {} orbitals: {}",
      active_space_indices.size(), oss.str());

  // Create new orbitals with the selected active space indices
  auto new_orbitals = detail::new_orbitals(wavefunction, active_space_indices);
  return detail::new_wavefunction(wavefunction, new_orbitals);
}

}  // namespace qdk::chemistry::algorithms::microsoft

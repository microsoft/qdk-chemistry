// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "autocas_active_space.hpp"

#include <algorithm>
#include <numeric>
#include <set>

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Wavefunction> AutocasActiveSpaceSelector::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction) const {
  // get settings
  const size_t min_plateau_size = _settings->get<size_t>("min_plateau_size");
  const size_t num_bins = _settings->get<size_t>("num_bins");
  const double entropy_threshold = _settings->get<double>("entropy_threshold");
  const bool normalize_entropies = _settings->get<bool>("normalize_entropies");

  // get max entropy, sorted entropies and corresponding orbital indices
  auto [max_entropy, selected_active_space_indices, sorted_entropies] =
      detail::_sort_entropies_and_indices(wavefunction, normalize_entropies);

  // check for single reference
  if (max_entropy < entropy_threshold) {
    // create new orbitals with active space
    auto new_orbitals = detail::new_orbitals(wavefunction, {});
    return detail::new_wavefunction(wavefunction, new_orbitals);
  }

  // create discrete bins
  const double bin_width = 1.0 / num_bins;
  std::vector<double> bins(num_bins);
  for (size_t i = 0; i < num_bins; ++i) {
    bins[i] = i * bin_width;
  }

  // Count orbitals above each bin
  std::vector<size_t> orbitals_above_bin(num_bins);
  size_t entropy_idx = sorted_entropies.size();
  for (size_t i = 0; i < num_bins; ++i) {
    double bin_threshold = bins[i];
    // Move backwards while entropies are <= threshold
    while (entropy_idx > 0 &&
           sorted_entropies[entropy_idx - 1] <= bin_threshold) {
      --entropy_idx;
    }
    // All entropies from 0 to entropy_idx-1 are > bin_threshold
    orbitals_above_bin[i] = entropy_idx;
  }

  // find plateaus
  std::vector<std::pair<size_t, size_t>> plateaus;
  // look for unique values in orbitals_above_bin
  // check that number of the values is larger equal than min_plateau_size
  // these form plateau
  size_t current_bin = 0;
  for (size_t i = 1; i < orbitals_above_bin.size(); ++i) {
    if (orbitals_above_bin[i] != orbitals_above_bin[i - 1]) {
      // end of plateau
      const size_t plateau_size = i - current_bin;
      if (plateau_size >= min_plateau_size) {
        plateaus.emplace_back(current_bin, i - 1);
      }
      current_bin = i;
    }
  }
  // check last plateau
  const size_t plateau_size = bins.size() - current_bin;
  if (plateau_size >= min_plateau_size) {
    plateaus.emplace_back(current_bin, bins.size() - 1);
  }

  // get number of orbitals from each plateau
  std::vector<size_t> active_space_sizes;
  active_space_sizes.reserve(plateaus.size());
  for (const auto& plateau : plateaus) {
    // get number of orbitals above start of plateau
    // and get the smallest entropy
    if (orbitals_above_bin[plateau.first] > 0) {
      size_t index = orbitals_above_bin[plateau.first] - 1;
      active_space_sizes.push_back(orbitals_above_bin[plateau.first]);
    }
  }

  // Select plateau with largest number of orbitals (which is above entropy
  // threshold)
  size_t max_space = active_space_sizes.empty()
                         ? 0
                         : *std::max_element(active_space_sizes.begin(),
                                             active_space_sizes.end());

  // Reuse selected_active_space_indices, keeping only the first max_space
  // elements
  selected_active_space_indices.resize(max_space);

  // sort selected_active_orbitals
  std::sort(selected_active_space_indices.begin(),
            selected_active_space_indices.end());

  // create new orbitals with active space
  auto new_orbitals =
      detail::new_orbitals(wavefunction, selected_active_space_indices);
  return detail::new_wavefunction(wavefunction, new_orbitals);
}

}  // namespace qdk::chemistry::algorithms::microsoft

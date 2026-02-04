// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "entropy_active_space.hpp"

#include <algorithm>
#include <qdk/chemistry/utils/logger.hpp>
#include <set>
#include <sstream>

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Wavefunction> AutocasEosActiveSpaceSelector::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction) const {
  QDK_LOG_TRACE_ENTERING();
  QDK_LOGGER().info("autoCAS-EOS::Starting active space selection.");

  // get settings
  const double entropy_threshold = _settings->get<double>("entropy_threshold");
  const double plateau_threshold = _settings->get<double>("diff_threshold");
  const bool normalize_entropies = _settings->get<bool>("normalize_entropies");
  QDK_LOGGER().debug("Settings:");
  QDK_LOGGER().debug("  entropy_threshold: {}", entropy_threshold);
  QDK_LOGGER().debug("  diff_threshold: {}", plateau_threshold);
  QDK_LOGGER().debug("  normalize_entropies: {}", normalize_entropies);

  // get max entropy, sorted entropies and corresponding orbital indices
  auto [max_entropy, selected_active_space_indices, sorted_entropies] =
      detail::_sort_entropies_and_indices(wavefunction, normalize_entropies);

  // Determine cutoff based on entropy differences
  Eigen::VectorXd diff = sorted_entropies.head(sorted_entropies.size() - 1) -
                         sorted_entropies.tail(sorted_entropies.size() - 1);

  // find first index where diff is larger than threshold
  std::vector<size_t> active_space_sizes;
  for (size_t i = 0; i < diff.size(); ++i) {
    if (sorted_entropies(i) > entropy_threshold &&
        diff(i) > plateau_threshold) {
      active_space_sizes.push_back(i + 1);
    }
  }
  std::ostringstream oss1;
  for (size_t i = 0; i < active_space_sizes.size(); ++i) {
    if (i > 0) oss1 << ", ";
    oss1 << active_space_sizes[i];
  }
  QDK_LOGGER().debug("Found {} candidate active spaces of sizes: {}",
                     active_space_sizes.size(), oss1.str());

  // Select plateau with largest number of orbitals (which is above entropy
  // threshold)
  size_t max_space = active_space_sizes.empty()
                         ? 0
                         : *std::max_element(active_space_sizes.begin(),
                                             active_space_sizes.end());
  // Reuse selected_active_space_indices, keeping only the first max_space
  // elements
  selected_active_space_indices.resize(max_space);

  // sort selected orbitals
  std::sort(selected_active_space_indices.begin(),
            selected_active_space_indices.end());
  std::ostringstream oss2;
  for (size_t i = 0; i < selected_active_space_indices.size(); ++i) {
    if (i > 0) oss2 << ", ";
    oss2 << selected_active_space_indices[i];
  }
  QDK_LOGGER().info("autoCAS-EOS::Selected active space of {} orbitals: {}",
                    selected_active_space_indices.size(), oss2.str());

  // create new orbitals with active space
  auto new_orbitals =
      detail::new_orbitals(wavefunction, selected_active_space_indices);
  return detail::new_wavefunction(wavefunction, new_orbitals);
}

}  // namespace qdk::chemistry::algorithms::microsoft

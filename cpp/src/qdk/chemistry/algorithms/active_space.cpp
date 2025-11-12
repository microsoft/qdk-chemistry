// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <optional>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <variant>

#include "microsoft/active_space/autocas_active_space.hpp"
#include "microsoft/active_space/entropy_active_space.hpp"
#include "microsoft/active_space/occupation_active_space.hpp"
#include "microsoft/active_space/valence_active_space.hpp"

namespace qdk::chemistry::algorithms {

namespace detail {

std::tuple<double, std::vector<size_t>, Eigen::VectorXd>
_sort_entropies_and_indices(std::shared_ptr<data::Wavefunction> wavefunction,
                            bool normalize_entropies) {
  // get orbitals which have entropies
  auto orbitals = wavefunction->get_orbitals();
  const auto& [active_space_indices, active_space_indices_beta] =
      orbitals->get_active_space_indices();

  // sanity checks
  if (active_space_indices != active_space_indices_beta) {
    throw std::runtime_error(
        "Active indices are not the same for alpha and beta orbitals.");
  }

  if (wavefunction->has_single_orbital_entropies() == false) {
    throw std::runtime_error(
        "Wavefunction does not have single orbital entropies.");
  }
  const auto entropies = wavefunction->get_single_orbital_entropies();
  if (entropies.size() != active_space_indices.size()) {
    throw std::runtime_error(
        "Entropy size does not match number of active space orbitals.");
  }

  // Sort entropies and orbital indices
  std::vector<std::pair<double, size_t>> entropy_index_pairs(entropies.size());
  for (size_t i = 0; i < entropies.size(); ++i) {
    entropy_index_pairs[i] = {entropies(i), active_space_indices[i]};
  }
  std::sort(entropy_index_pairs.begin(), entropy_index_pairs.end(),
            [](const auto& a, const auto& b) { return a.first > b.first; });

  Eigen::VectorXd sorted_entropies(entropies.size());
  // indices sorted by corresponding entropy value (largest index comes first)
  // Therefore we just need to determine the number of orbitals to select
  std::vector<size_t> selected_active_space_indices(entropies.size());
  for (size_t i = 0; i < entropy_index_pairs.size(); ++i) {
    std::tie(sorted_entropies[i], selected_active_space_indices[i]) =
        entropy_index_pairs[i];
  }

  // Normalize entropies
  const double max_entropy = sorted_entropies.maxCoeff();
  if (max_entropy > std::numeric_limits<double>::epsilon() and
      normalize_entropies) {
    sorted_entropies /= max_entropy;
  }
  return {max_entropy, selected_active_space_indices, sorted_entropies};
}

std::shared_ptr<data::Orbitals> new_orbitals(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& active_space_indices_a,
    const std::optional<std::vector<size_t>>& active_space_indices_b) {
  auto orbitals = wavefunction->get_orbitals();

  if (orbitals->is_restricted() || !active_space_indices_b.has_value()) {
    // get inactive indices
    size_t nelec = wavefunction->get_total_num_electrons().first +
                   wavefunction->get_total_num_electrons().second;
    std::vector<size_t> inactive_indices =
        _get_inactive_space_indices(nelec, active_space_indices_a);

    // check that provided active alpha indices are a subset of wavefunction
    // active orbitals
    const auto& [wavefunction_active_indices_a, wavefunction_active_indices_b] =
        orbitals->get_active_space_indices();
    for (const auto& idx : active_space_indices_a) {
      if (std::find(wavefunction_active_indices_a.begin(),
                    wavefunction_active_indices_a.end(),
                    idx) == wavefunction_active_indices_a.end()) {
        throw std::runtime_error(
            "Selected orbital (index: " + std::to_string(idx) +
            ") is outside of initial active space.");
      }
    }

    // Restricted case
    std::optional<Eigen::VectorXd> energies;
    if (orbitals->has_energies()) {
      energies = orbitals->get_energies().first;
    }

    std::optional<Eigen::MatrixXd> ao_overlap;
    if (orbitals->has_overlap_matrix()) {
      ao_overlap = orbitals->get_overlap_matrix();
    }

    std::shared_ptr<data::BasisSet> basis_set;
    if (orbitals->has_basis_set()) {
      basis_set = orbitals->get_basis_set();
    }

    return std::make_shared<data::Orbitals>(
        orbitals->get_coefficients().first, energies, ao_overlap, basis_set,
        std::make_tuple(active_space_indices_a, inactive_indices));
  } else {
    // get inactive indices
    size_t nelec_a = wavefunction->get_total_num_electrons().first;
    size_t nelec_b = wavefunction->get_total_num_electrons().second;
    auto [inactive_indices_a, inactive_indices_b] =
        _get_inactive_space_indices(nelec_a, nelec_b, active_space_indices_a,
                                    active_space_indices_b.value());

    // check that provided active alpha & beta indices are subsets of
    // wavefunction active orbitals
    const auto& [wavefunction_active_indices_a, wavefunction_active_indices_b] =
        orbitals->get_active_space_indices();
    for (const auto& idx : active_space_indices_a) {
      if (std::find(wavefunction_active_indices_a.begin(),
                    wavefunction_active_indices_a.end(),
                    idx) == wavefunction_active_indices_a.end()) {
        throw std::runtime_error(
            "Selected alpha orbital (index: " + std::to_string(idx) +
            ") is outside of initial active space.");
      }
    }
    for (const auto& idx : active_space_indices_b.value()) {
      if (std::find(wavefunction_active_indices_b.begin(),
                    wavefunction_active_indices_b.end(),
                    idx) == wavefunction_active_indices_b.end()) {
        throw std::runtime_error(
            "Selected beta orbital (index: " + std::to_string(idx) +
            ") is outside of initial active space.");
      }
    }

    // Unrestricted case
    auto coeffs_pair = orbitals->get_coefficients();
    std::optional<Eigen::VectorXd> alpha_energies, beta_energies;

    if (orbitals->has_energies()) {
      auto energies_pair = orbitals->get_energies();
      alpha_energies = energies_pair.first;
      beta_energies = energies_pair.second;
    }

    std::optional<Eigen::MatrixXd> ao_overlap;
    if (orbitals->has_overlap_matrix()) {
      ao_overlap = orbitals->get_overlap_matrix();
    }

    std::shared_ptr<data::BasisSet> basis_set;
    if (orbitals->has_basis_set()) {
      basis_set = orbitals->get_basis_set();
    }
    return std::make_shared<data::Orbitals>(
        coeffs_pair.first, coeffs_pair.second, alpha_energies, beta_energies,
        ao_overlap, basis_set,
        std::make_tuple(active_space_indices_a, active_space_indices_b.value(),
                        inactive_indices_a, inactive_indices_b));
  }
}

std::vector<size_t> _get_inactive_space_indices(
    size_t nelec, const std::vector<size_t>& active_space_indices) {
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
  return inactive_indices;
}

std::pair<std::vector<size_t>, std::vector<size_t>> _get_inactive_space_indices(
    size_t nelec_a, size_t nelec_b,
    const std::vector<size_t>& active_space_indices_a,
    const std::vector<size_t>& active_space_indices_b) {
  std::vector<size_t> inactive_indices_a;
  std::vector<size_t> inactive_indices_b;
  for (size_t i = 0; i < nelec_a; ++i) {
    inactive_indices_a.push_back(i);
  }
  for (size_t i = 0; i < nelec_b; ++i) {
    inactive_indices_b.push_back(i);
  }

  // ensure inactive indices are not in active space
  inactive_indices_a.erase(
      std::remove_if(inactive_indices_a.begin(), inactive_indices_a.end(),
                     [&](size_t idx) {
                       return std::find(active_space_indices_a.begin(),
                                        active_space_indices_a.end(),
                                        idx) != active_space_indices_a.end();
                     }),
      inactive_indices_a.end());

  // ensure inactive indices are not in active space
  inactive_indices_b.erase(
      std::remove_if(inactive_indices_b.begin(), inactive_indices_b.end(),
                     [&](size_t idx) {
                       return std::find(active_space_indices_b.begin(),
                                        active_space_indices_b.end(),
                                        idx) != active_space_indices_b.end();
                     }),
      inactive_indices_b.end());

  return std::make_pair(inactive_indices_a, inactive_indices_b);
}

// Helper function to extract active orbital occupations from a determinant
// that is already in the old active space basis
data::Configuration _extract_active_orbitals(
    const data::Configuration& det,
    const std::vector<size_t>& old_active_indices,
    const std::vector<size_t>& new_active_indices) {
  // Create a string representation with only the new active orbitals
  std::string active_det_string(new_active_indices.size(), '0');

  for (size_t i = 0; i < new_active_indices.size(); ++i) {
    size_t new_orbital_idx = new_active_indices[i];

    // Find the position of this orbital in the old active space
    auto it = std::find(old_active_indices.begin(), old_active_indices.end(),
                        new_orbital_idx);

    if (it == old_active_indices.end()) {
      // This orbital wasn't in the old active space, so it's unoccupied
      active_det_string[i] = '0';
      continue;
    }

    // Get the index in the old determinant (which is already shortened to old
    // active space)
    size_t old_det_idx = std::distance(old_active_indices.begin(), it);

    bool has_alpha = det.has_alpha_electron(old_det_idx);
    bool has_beta = det.has_beta_electron(old_det_idx);

    if (has_alpha && has_beta) {
      active_det_string[i] = '2';  // Doubly occupied
    } else if (has_alpha) {
      active_det_string[i] = 'u';  // Alpha only
    } else if (has_beta) {
      active_det_string[i] = 'd';  // Beta only
    } else {
      active_det_string[i] = '0';  // Unoccupied
    }
  }

  return data::Configuration(active_det_string);
}

std::shared_ptr<data::Wavefunction> new_wavefunction(
    std::shared_ptr<data::Wavefunction> wavefunction,
    std::shared_ptr<data::Orbitals> new_orbitals) {
  if (!wavefunction) {
    throw std::invalid_argument("Wavefunction pointer cannot be nullptr");
  }
  if (!new_orbitals) {
    throw std::invalid_argument("New orbitals pointer cannot be nullptr");
  }

  // Get the coefficients and determinants from the original wavefunction
  const auto& coefficients = wavefunction->get_coefficients();
  const auto& determinants = wavefunction->get_active_determinants();
  auto wavefunction_type = wavefunction->get_type();

  // Get the old active space indices from the original wavefunction
  const auto& old_orbitals = wavefunction->get_orbitals();
  const auto& [old_active_indices_a, old_active_indices_b] =
      old_orbitals->get_active_space_indices();

  // Get the new active space indices
  const auto& [new_active_indices_a, new_active_indices_b] =
      new_orbitals->get_active_space_indices();

  // Get the expected active electron count
  const auto& [expected_nelec_a, expected_nelec_b] =
      wavefunction->get_active_num_electrons();

  // For single determinant case, truncate the determinant
  // For multi-determinant case, create an aufbau (ground state) determinant
  data::Configuration truncated_det;

  if (wavefunction->size() == 1) {
    // Single Slater determinant - truncate it
    truncated_det = _extract_active_orbitals(
        determinants[0], old_active_indices_a, new_active_indices_a);
  } else {
    // Multi-determinant wavefunction - create aufbau occupations in old space,
    // then truncate
    std::string aufbau_string(old_active_indices_a.size(), '0');

    size_t nalpha_filled = 0;
    size_t nbeta_filled = 0;

    for (size_t i = 0; i < old_active_indices_a.size(); ++i) {
      if (nalpha_filled < expected_nelec_a && nbeta_filled < expected_nelec_b) {
        // Doubly occupy
        aufbau_string[i] = '2';
        nalpha_filled++;
        nbeta_filled++;
      } else if (nalpha_filled < expected_nelec_a) {
        // Alpha only
        aufbau_string[i] = 'u';
        nalpha_filled++;
      } else if (nbeta_filled < expected_nelec_b) {
        // Beta only
        aufbau_string[i] = 'd';
        nbeta_filled++;
      } else {
        // Unoccupied
        aufbau_string[i] = '0';
      }
    }

    data::Configuration aufbau_det(aufbau_string);
    // Now truncate the aufbau determinant to the new active space
    truncated_det = _extract_active_orbitals(aufbau_det, old_active_indices_a,
                                             new_active_indices_a);
  }

  // Create a new container with the truncated/aufbau determinant
  std::unique_ptr<data::WavefunctionContainer> new_container;
  new_container = std::make_unique<data::SlaterDeterminantContainer>(
      truncated_det, new_orbitals, wavefunction_type);

  return std::make_shared<data::Wavefunction>(std::move(new_container));
}

}  // namespace detail

std::unique_ptr<ActiveSpaceSelector> make_valence_active_space_selector() {
  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::ValenceActiveSpaceSelector>();
}

std::unique_ptr<ActiveSpaceSelector> make_occupation_active_space_selector() {
  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::OccupationActiveSpaceSelector>();
}

std::unique_ptr<ActiveSpaceSelector> make_autocas_active_space_selector() {
  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::AutocasActiveSpaceSelector>();
}

std::unique_ptr<ActiveSpaceSelector> make_autocas_eos_active_space_selector() {
  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::AutocasEosActiveSpaceSelector>();
}

void ActiveSpaceSelectorFactory::register_default_instances() {
  ActiveSpaceSelectorFactory::register_instance(
      &make_valence_active_space_selector);
  ActiveSpaceSelectorFactory::register_instance(
      &make_occupation_active_space_selector);
  ActiveSpaceSelectorFactory::register_instance(
      &make_autocas_active_space_selector);
  ActiveSpaceSelectorFactory::register_instance(
      &make_autocas_eos_active_space_selector);
}

}  // namespace qdk::chemistry::algorithms

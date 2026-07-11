// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "nuclear_derivative_detail.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/mcscf.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qdk::chemistry::algorithms::detail {

struct ReferenceOrbitals {
  std::shared_ptr<data::Orbitals> orbitals;
  std::shared_ptr<data::Wavefunction> wavefunction;
};

std::shared_ptr<data::Structure> copy_structure(
    const std::shared_ptr<data::Structure>& structure) {
  if (!structure) {
    throw std::invalid_argument("Structure must not be null");
  }
  return std::make_shared<data::Structure>(*structure);
}

std::shared_ptr<data::Structure> displace_structure(
    const std::shared_ptr<data::Structure>& structure, Eigen::Index coordinate,
    double displacement) {
  if (!structure) {
    throw std::invalid_argument("Structure must not be null");
  }
  const auto dimension =
      static_cast<Eigen::Index>(3 * structure->get_num_atoms());
  if (coordinate < 0 || coordinate >= dimension) {
    throw std::out_of_range("Coordinate index " + std::to_string(coordinate) +
                            " is out of range for dimension " +
                            std::to_string(dimension));
  }

  Eigen::MatrixXd coordinates = structure->get_coordinates();
  const Eigen::Index atom = coordinate / 3;
  const Eigen::Index component = coordinate % 3;
  coordinates(atom, component) += displacement;
  return std::make_shared<data::Structure>(
      coordinates, structure->get_elements(), structure->get_masses(),
      structure->get_nuclear_charges());
}

BasisOrGuessType seed_to_scf_input(const NuclearDerivativeSeedType& seed,
                                   bool allow_orbital_guess) {
  if (std::holds_alternative<std::string>(seed)) {
    return std::get<std::string>(seed);
  }
  if (std::holds_alternative<std::shared_ptr<data::BasisSet>>(seed)) {
    auto basis = std::get<std::shared_ptr<data::BasisSet>>(seed);
    if (!basis) {
      throw std::invalid_argument("Basis set seed must not be null");
    }
    if (allow_orbital_guess) {
      return basis;
    }
    if (basis->get_name() == data::BasisSet::custom_name) {
      throw std::invalid_argument(
          "Custom basis set cannot be reused for displaced numeric "
          "derivatives");
    }
    return basis->get_name();
  }
  std::shared_ptr<data::Orbitals> orbitals;
  if (std::holds_alternative<std::shared_ptr<data::Orbitals>>(seed)) {
    orbitals = std::get<std::shared_ptr<data::Orbitals>>(seed);
  } else if (std::holds_alternative<std::shared_ptr<data::Wavefunction>>(
                 seed)) {
    auto wavefunction = std::get<std::shared_ptr<data::Wavefunction>>(seed);
    if (!wavefunction) {
      throw std::invalid_argument("Wavefunction seed must not be null");
    }
    orbitals = wavefunction->get_orbitals();
  }

  if (!orbitals || !orbitals->get_basis_set()) {
    throw std::invalid_argument(
        "Orbital or wavefunction seed must include orbitals with a basis set");
  }
  if (allow_orbital_guess) {
    return orbitals;
  }

  auto basis = orbitals->get_basis_set();
  if (basis->get_name() == data::BasisSet::custom_name) {
    throw std::invalid_argument(
        "Custom orbital basis cannot be reused for displaced numeric "
        "derivatives");
  }
  return basis->get_name();
}

std::pair<unsigned int, unsigned int> active_electron_counts(
    const std::shared_ptr<data::Structure>& structure, int charge,
    int spin_multiplicity, const NuclearDerivativeSeedType& seed,
    unsigned int n_inactive_orbitals) {
  if (!structure) {
    throw std::invalid_argument("Structure must not be null");
  }
  if (spin_multiplicity < 1) {
    throw std::invalid_argument("spin_multiplicity must be at least 1");
  }

  std::shared_ptr<data::BasisSet> basis;
  if (std::holds_alternative<std::string>(seed)) {
    basis =
        data::BasisSet::from_basis_name(std::get<std::string>(seed), structure);
  } else if (std::holds_alternative<std::shared_ptr<data::BasisSet>>(seed)) {
    basis = std::get<std::shared_ptr<data::BasisSet>>(seed);
  } else {
    std::shared_ptr<data::Orbitals> orbitals;
    if (std::holds_alternative<std::shared_ptr<data::Orbitals>>(seed)) {
      orbitals = std::get<std::shared_ptr<data::Orbitals>>(seed);
      if (!orbitals) {
        throw std::invalid_argument("Orbital seed must not be null");
      }
    } else {
      auto wavefunction = std::get<std::shared_ptr<data::Wavefunction>>(seed);
      if (!wavefunction) {
        throw std::invalid_argument("Wavefunction seed must not be null");
      }
      orbitals = wavefunction->get_orbitals();
    }
    if (!orbitals || !orbitals->get_basis_set()) {
      throw std::invalid_argument(
          "Orbital or wavefunction seed must include orbitals with a basis "
          "set");
    }
    basis = orbitals->get_basis_set();
  }
  if (!basis) {
    throw std::invalid_argument("Basis set seed must not be null");
  }

  const auto n_ecp_electrons =
      std::accumulate(basis->get_ecp_electrons().begin(),
                      basis->get_ecp_electrons().end(), int64_t{0});
  const auto total_electrons = static_cast<int64_t>(std::llround(
                                   structure->get_nuclear_charges().sum())) -
                               charge - n_ecp_electrons;
  const auto unpaired_electrons = spin_multiplicity - 1;
  if (total_electrons < 0) {
    throw std::invalid_argument(
        "charge and ECP electrons cannot exceed the total nuclear charge");
  }
  if (unpaired_electrons > total_electrons ||
      (total_electrons + unpaired_electrons) % 2 != 0) {
    throw std::invalid_argument(
        "charge and spin_multiplicity specify incompatible electron counts");
  }

  const auto n_alpha_electrons =
      static_cast<unsigned int>((total_electrons + unpaired_electrons) / 2);
  const auto n_beta_electrons =
      static_cast<unsigned int>(total_electrons) - n_alpha_electrons;
  if (n_inactive_orbitals > std::min(n_alpha_electrons, n_beta_electrons)) {
    throw std::invalid_argument(
        "n_inactive_orbitals cannot exceed either spin electron count");
  }
  return {n_alpha_electrons - n_inactive_orbitals,
          n_beta_electrons - n_inactive_orbitals};
}

std::shared_ptr<data::Orbitals> seed_to_orbitals(
    const NuclearDerivativeSeedType& seed) {
  std::shared_ptr<data::Orbitals> orbitals;
  if (std::holds_alternative<std::shared_ptr<data::Orbitals>>(seed)) {
    orbitals = std::get<std::shared_ptr<data::Orbitals>>(seed);
    if (!orbitals) {
      throw std::invalid_argument("Orbital seed must not be null");
    }
  } else if (std::holds_alternative<std::shared_ptr<data::Wavefunction>>(
                 seed)) {
    auto wavefunction = std::get<std::shared_ptr<data::Wavefunction>>(seed);
    if (!wavefunction) {
      throw std::invalid_argument("Wavefunction seed must not be null");
    }
    orbitals = wavefunction->get_orbitals();
  }

  if (!orbitals) {
    return nullptr;
  }
  if (!orbitals->get_basis_set()) {
    throw std::invalid_argument(
        "Orbital or wavefunction seed must include orbitals with a basis set");
  }
  return orbitals;
}

std::shared_ptr<data::Orbitals> copy_active_space_metadata(
    const std::shared_ptr<data::Orbitals>& orbitals,
    const std::shared_ptr<data::Orbitals>& metadata_source) {
  if (!orbitals || !metadata_source) {
    return orbitals;
  }

  const auto& [active_a, active_b] =
      metadata_source->get_active_space_indices();
  const auto& [inactive_a, inactive_b] =
      metadata_source->get_inactive_space_indices();
  std::optional<Eigen::MatrixXd> ao_overlap;
  if (orbitals->has_overlap_matrix()) {
    ao_overlap = orbitals->get_overlap_matrix();
  }
  std::shared_ptr<data::BasisSet> basis_set = orbitals->get_basis_set();

  if (orbitals->is_restricted()) {
    std::optional<Eigen::VectorXd> energies;
    if (orbitals->has_energies()) {
      energies = orbitals->get_energies().first;
    }
    return std::make_shared<data::Orbitals>(
        orbitals->get_coefficients().first, energies, ao_overlap, basis_set,
        std::make_tuple(
            std::vector<size_t>(active_a.begin(), active_a.end()),
            std::vector<size_t>(inactive_a.begin(), inactive_a.end())));
  }

  std::optional<Eigen::VectorXd> energies_a;
  std::optional<Eigen::VectorXd> energies_b;
  if (orbitals->has_energies()) {
    auto [source_energies_a, source_energies_b] = orbitals->get_energies();
    energies_a = source_energies_a;
    energies_b = source_energies_b;
  }
  return std::make_shared<data::Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_coefficients().second,
      energies_a, energies_b, ao_overlap, basis_set,
      std::make_tuple(
          std::vector<size_t>(active_a.begin(), active_a.end()),
          std::vector<size_t>(active_b.begin(), active_b.end()),
          std::vector<size_t>(inactive_a.begin(), inactive_a.end()),
          std::vector<size_t>(inactive_b.begin(), inactive_b.end())));
}

template <typename Factory>
typename Factory::return_type create_from_ref(const data::AlgorithmRef& ref) {
  auto instance = Factory::create(ref.get_algorithm_name());
  if (ref.get_settings()) {
    instance->settings().update(*ref.get_settings());
  }
  return instance;
}

void validate_active_electron_count(unsigned int count,
                                    const std::string& argument_name) {
  if (count == 0) {
    throw std::invalid_argument(
        argument_name +
        " must be set to a positive value for this energy calculator");
  }
}

ReferenceOrbitals localize_reference_orbitals(const data::Settings& settings,
                                              ReferenceOrbitals reference,
                                              unsigned int n_active_alpha,
                                              unsigned int n_active_beta) {
  if (!settings.get<bool>("localize_reference_orbitals")) {
    return reference;
  }

  if (!reference.wavefunction) {
    const auto& [active_a, active_b] =
        reference.orbitals->get_active_space_indices();
    if (active_a.size() != active_b.size()) {
      throw std::invalid_argument(
          "Reference orbital localization requires matching alpha and beta "
          "active spaces");
    }
    if (n_active_alpha > active_a.size() || n_active_beta > active_a.size()) {
      throw std::invalid_argument(
          "Active electron count exceeds the number of active orbitals");
    }

    auto determinant = data::Configuration::canonical_hf_configuration(
        n_active_alpha, n_active_beta, active_a.size());
    auto container = std::make_unique<data::StateVectorContainer>(
        determinant, reference.orbitals);
    auto seed_wavefunction =
        std::make_shared<data::Wavefunction>(std::move(container));
    reference.wavefunction = new_aufbau_determinant_wavefunction(
        seed_wavefunction, reference.orbitals);
  }

  auto localizer = create_from_ref<LocalizerFactory>(
      settings.get<data::AlgorithmRef>("orbital_localizer"));
  auto [loc_indices_a, loc_indices_b] =
      reference.orbitals->get_active_space_indices();
  reference.wavefunction =
      localizer->run(reference.wavefunction, loc_indices_a, loc_indices_b);
  reference.orbitals = reference.wavefunction->get_orbitals();
  return reference;
}

ReferenceOrbitals reference_orbitals_for_mr_energy(
    const data::Settings& settings, std::shared_ptr<data::Structure> structure,
    int charge, int spin_multiplicity, const NuclearDerivativeSeedType& seed,
    bool allow_orbital_seed, unsigned int n_active_alpha,
    unsigned int n_active_beta) {
  auto seed_orbitals = seed_to_orbitals(seed);
  if (allow_orbital_seed && seed_orbitals) {
    std::shared_ptr<data::Wavefunction> seed_wavefunction;
    if (std::holds_alternative<std::shared_ptr<data::Wavefunction>>(seed)) {
      seed_wavefunction = std::get<std::shared_ptr<data::Wavefunction>>(seed);
      if (!seed_wavefunction) {
        throw std::invalid_argument("Wavefunction seed must not be null");
      }
    }
    return localize_reference_orbitals(settings,
                                       {seed_orbitals, seed_wavefunction},
                                       n_active_alpha, n_active_beta);
  }

  auto orbital_solver = create_from_ref<ScfSolverFactory>(
      settings.get<data::AlgorithmRef>("orbital_solver"));
  auto [_, reference_wavefunction] =
      orbital_solver->run(structure, charge, spin_multiplicity,
                          seed_to_scf_input(seed, allow_orbital_seed));
  auto reference_orbitals = reference_wavefunction->get_orbitals();
  if (settings.get<bool>("reuse_seed_active_space")) {
    reference_orbitals =
        copy_active_space_metadata(reference_orbitals, seed_orbitals);
    if (reference_orbitals != reference_wavefunction->get_orbitals()) {
      reference_wavefunction =
          new_wavefunction(reference_wavefunction, reference_orbitals);
    }
  }

  return localize_reference_orbitals(
      settings, {reference_orbitals, reference_wavefunction}, n_active_alpha,
      n_active_beta);
}

EnergyEvaluation evaluate_energy(const data::Settings& settings,
                                 std::shared_ptr<data::Structure> structure,
                                 int charge, int spin_multiplicity,
                                 const NuclearDerivativeSeedType& seed,
                                 bool allow_wavefunction_seed,
                                 unsigned int n_active_alpha_electrons,
                                 unsigned int n_active_beta_electrons) {
  const auto ref = settings.get<data::AlgorithmRef>("energy_calculator");
  const auto& algorithm_type = ref.get_algorithm_type();

  if (algorithm_type == ScfSolverFactory::algorithm_type_name()) {
    auto solver = create_from_ref<ScfSolverFactory>(ref);
    auto [energy, wavefunction] =
        solver->run(structure, charge, spin_multiplicity,
                    seed_to_scf_input(seed, allow_wavefunction_seed));
    return {energy, wavefunction};
  }

  if (algorithm_type == MultiConfigurationScfFactory::algorithm_type_name()) {
    validate_active_electron_count(n_active_alpha_electrons,
                                   "Derived active alpha electron count");
    validate_active_electron_count(n_active_beta_electrons,
                                   "Derived active beta electron count");
    auto reference = reference_orbitals_for_mr_energy(
        settings, structure, charge, spin_multiplicity, seed,
        allow_wavefunction_seed, n_active_alpha_electrons,
        n_active_beta_electrons);
    auto calculator = create_from_ref<MultiConfigurationScfFactory>(ref);
    auto [energy, wavefunction] = calculator->run(
        reference.orbitals, n_active_alpha_electrons, n_active_beta_electrons);
    return {energy, wavefunction};
  }

  if (algorithm_type ==
      MultiConfigurationCalculatorFactory::algorithm_type_name()) {
    validate_active_electron_count(n_active_alpha_electrons,
                                   "Derived active alpha electron count");
    validate_active_electron_count(n_active_beta_electrons,
                                   "Derived active beta electron count");
    auto reference = reference_orbitals_for_mr_energy(
        settings, structure, charge, spin_multiplicity, seed,
        allow_wavefunction_seed, n_active_alpha_electrons,
        n_active_beta_electrons);
    auto hamiltonian_constructor =
        create_from_ref<HamiltonianConstructorFactory>(
            settings.get<data::AlgorithmRef>("hamiltonian_constructor"));
    auto hamiltonian = hamiltonian_constructor->run(reference.orbitals);
    auto calculator = create_from_ref<MultiConfigurationCalculatorFactory>(ref);
    auto [energy, wavefunction] = calculator->run(
        hamiltonian, n_active_alpha_electrons, n_active_beta_electrons);
    return {energy, wavefunction};
  }

  throw std::invalid_argument(
      "Unsupported energy_calculator algorithm type for numeric nuclear "
      "derivatives: " +
      algorithm_type);
}

}  // namespace qdk::chemistry::algorithms::detail

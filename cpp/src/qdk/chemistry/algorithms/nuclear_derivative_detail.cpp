// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "nuclear_derivative_detail.hpp"

#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/mcscf.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
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

namespace {
// Project a source active/inactive index set onto the symmetry and mode extents
// of the target orbitals' coefficient tensor. This keeps the index set
// structurally compatible with the target (e.g. a restricted target aliases the
// beta channel from alpha), avoiding mixed restricted/unrestricted index sets
// that would fail SymmetryBlockedIndexSet validation on serialization. A null
// source (absent space) projects to null; a non-null but empty source projects
// to an explicitly empty (0-orbital) set.
std::shared_ptr<const data::SymmetryBlockedIndexSet> project_index_set_to(
    const std::shared_ptr<const data::Orbitals>& target,
    const std::shared_ptr<const data::SymmetryBlockedIndexSet>& source) {
  if (!source) {
    return nullptr;
  }
  const auto symmetries = target->coefficients()->symmetries()[1];
  const auto& extents = target->coefficients()->extents()[1];
  const auto alpha = data::spin_channel_indices(source, data::axes::alpha());
  auto to_u32 = [](const std::vector<std::size_t>& v) {
    return std::vector<std::uint32_t>(v.begin(), v.end());
  };
  std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>> indices;
  if (symmetries && symmetries->has_axis(data::AxisName::Spin)) {
    if (!alpha.empty()) indices.emplace(data::axes::alpha(), to_u32(alpha));
    if (!target->is_restricted()) {
      const auto beta = data::spin_channel_indices(source, data::axes::beta());
      if (!beta.empty()) indices.emplace(data::axes::beta(), to_u32(beta));
    }
  } else if (!alpha.empty()) {
    indices.emplace(data::SymmetryLabel{}, to_u32(alpha));
  }
  return std::make_shared<const data::SymmetryBlockedIndexSet>(
      symmetries, extents, std::move(indices));
}
}  // namespace

std::shared_ptr<data::Orbitals> copy_active_space_metadata(
    const std::shared_ptr<data::Orbitals>& orbitals,
    const std::shared_ptr<data::Orbitals>& metadata_source) {
  if (!orbitals || !metadata_source) {
    return orbitals;
  }

  std::optional<Eigen::MatrixXd> ao_overlap;
  if (orbitals->has_overlap_matrix()) {
    ao_overlap = orbitals->get_overlap_matrix();
  }

  // Copy coefficients/energies from orbitals and the active/inactive index sets
  // from metadata_source, projecting the latter onto the target symmetry so the
  // result stays a structurally valid restricted/unrestricted Orbitals.
  return std::make_shared<data::Orbitals>(
      orbitals->coefficients(),
      orbitals->has_energies() ? orbitals->energies() : nullptr, ao_overlap,
      orbitals->get_basis_set(),
      project_index_set_to(orbitals, metadata_source->active_indices()),
      project_index_set_to(orbitals, metadata_source->inactive_indices()));
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
    const auto active_ai = reference.orbitals->active_indices();
    const auto active_a =
        data::spin_channel_indices(active_ai, data::axes::alpha());
    const auto active_b =
        data::spin_channel_indices(active_ai, data::axes::beta());
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
  const auto loc_ai = reference.orbitals->active_indices();
  auto loc_indices_a = data::spin_channel_indices(loc_ai, data::axes::alpha());
  auto loc_indices_b = data::spin_channel_indices(loc_ai, data::axes::beta());
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
                                   "n_active_alpha_electrons");
    validate_active_electron_count(n_active_beta_electrons,
                                   "n_active_beta_electrons");
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
                                   "n_active_alpha_electrons");
    validate_active_electron_count(n_active_beta_electrons,
                                   "n_active_beta_electrons");
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

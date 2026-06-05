// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/mcscf.hpp>
#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

#include "microsoft/scf.hpp"

namespace qdk::chemistry::algorithms {

namespace {

class ScopedLogLevel {
 public:
  explicit ScopedLogLevel(utils::LogLevel minimum_level)
      : previous_level_(utils::Logger::get_global_level()) {
    if (static_cast<int>(previous_level_) < static_cast<int>(minimum_level)) {
      utils::Logger::set_global_level(minimum_level);
      changed_ = true;
    }
  }

  ~ScopedLogLevel() {
    if (changed_) {
      utils::Logger::set_global_level(previous_level_);
    }
  }

  ScopedLogLevel(const ScopedLogLevel&) = delete;
  ScopedLogLevel& operator=(const ScopedLogLevel&) = delete;

 private:
  utils::LogLevel previous_level_;
  bool changed_ = false;
};

struct EnergyEvaluation {
  double energy = 0.0;
  std::optional<std::shared_ptr<data::Wavefunction>> wavefunction;
};

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

template <typename Factory>
typename Factory::return_type create_from_ref(const data::AlgorithmRef& ref);

unsigned int active_electrons(const data::Settings& settings,
                              const std::string& key);

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

std::shared_ptr<data::Wavefunction> seed_to_wavefunction(
    const NuclearDerivativeSeedType& seed) {
  if (!std::holds_alternative<std::shared_ptr<data::Wavefunction>>(seed)) {
    return nullptr;
  }
  auto wavefunction = std::get<std::shared_ptr<data::Wavefunction>>(seed);
  if (!wavefunction) {
    throw std::invalid_argument("Wavefunction seed must not be null");
  }
  return wavefunction;
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

std::string active_determinant_string(size_t n_active_orbitals,
                                      unsigned int n_alpha,
                                      unsigned int n_beta) {
  if (n_alpha > n_active_orbitals || n_beta > n_active_orbitals) {
    throw std::invalid_argument(
        "Active electron count exceeds the number of active orbitals");
  }

  std::string determinant(n_active_orbitals, '0');
  size_t alpha_remaining = n_alpha;
  size_t beta_remaining = n_beta;
  for (auto& occupation : determinant) {
    if (alpha_remaining > 0 && beta_remaining > 0) {
      occupation = '2';
      --alpha_remaining;
      --beta_remaining;
    } else if (alpha_remaining > 0) {
      occupation = 'u';
      --alpha_remaining;
    } else if (beta_remaining > 0) {
      occupation = 'd';
      --beta_remaining;
    }
  }
  return determinant;
}

std::shared_ptr<data::Wavefunction> wavefunction_from_orbitals(
    std::shared_ptr<data::Orbitals> orbitals, unsigned int n_active_alpha,
    unsigned int n_active_beta) {
  const auto& [active_a, active_b] = orbitals->get_active_space_indices();
  if (active_a.size() != active_b.size()) {
    throw std::invalid_argument(
        "Reference orbital localization requires matching alpha and beta "
        "active spaces");
  }
  auto determinant = data::Configuration(active_determinant_string(
      active_a.size(), n_active_alpha, n_active_beta));
  auto container =
      std::make_unique<data::SlaterDeterminantContainer>(determinant, orbitals);
  return std::make_shared<data::Wavefunction>(std::move(container));
}

ReferenceOrbitals localize_reference_orbitals(const data::Settings& settings,
                                              ReferenceOrbitals reference) {
  if (!settings.get<bool>("localize_reference_orbitals")) {
    return reference;
  }

  if (!reference.wavefunction) {
    reference.wavefunction = wavefunction_from_orbitals(
        reference.orbitals,
        active_electrons(settings, "n_active_alpha_electrons"),
        active_electrons(settings, "n_active_beta_electrons"));
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
    bool allow_orbital_seed) {
  auto seed_orbitals = seed_to_orbitals(seed);
  if (allow_orbital_seed && seed_orbitals) {
    return localize_reference_orbitals(
        settings, {seed_orbitals, seed_to_wavefunction(seed)});
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
          detail::new_wavefunction(reference_wavefunction, reference_orbitals);
    }
  }

  return localize_reference_orbitals(
      settings, {reference_orbitals, reference_wavefunction});
}

template <typename Factory>
typename Factory::return_type create_from_ref(const data::AlgorithmRef& ref) {
  auto instance = Factory::create(ref.get_algorithm_name());
  if (ref.get_settings()) {
    instance->settings().update(*ref.get_settings());
  }
  return instance;
}

unsigned int active_electrons(const data::Settings& settings,
                              const std::string& key) {
  auto value = settings.get<int64_t>(key);
  if (value <= 0) {
    throw std::invalid_argument(
        key + " must be set to a positive value for this energy calculator");
  }
  return static_cast<unsigned int>(value);
}

EnergyEvaluation evaluate_energy(const data::Settings& settings,
                                 std::shared_ptr<data::Structure> structure,
                                 int charge, int spin_multiplicity,
                                 const NuclearDerivativeSeedType& seed,
                                 bool allow_wavefunction_seed) {
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
    auto reference = reference_orbitals_for_mr_energy(
        settings, structure, charge, spin_multiplicity, seed,
        allow_wavefunction_seed);
    auto calculator = create_from_ref<MultiConfigurationScfFactory>(ref);
    auto [energy, wavefunction] =
        calculator->run(reference.orbitals,
                        active_electrons(settings, "n_active_alpha_electrons"),
                        active_electrons(settings, "n_active_beta_electrons"));
    return {energy, wavefunction};
  }

  if (algorithm_type ==
      MultiConfigurationCalculatorFactory::algorithm_type_name()) {
    auto reference = reference_orbitals_for_mr_energy(
        settings, structure, charge, spin_multiplicity, seed,
        allow_wavefunction_seed);
    auto hamiltonian_constructor =
        create_from_ref<HamiltonianConstructorFactory>(
            settings.get<data::AlgorithmRef>("hamiltonian_constructor"));
    auto hamiltonian = hamiltonian_constructor->run(reference.orbitals);
    auto calculator = create_from_ref<MultiConfigurationCalculatorFactory>(ref);
    auto [energy, wavefunction] = calculator->run(
        hamiltonian, active_electrons(settings, "n_active_alpha_electrons"),
        active_electrons(settings, "n_active_beta_electrons"));
    return {energy, wavefunction};
  }

  throw std::invalid_argument(
      "Unsupported energy_calculator algorithm type for numeric nuclear "
      "derivatives: " +
      algorithm_type);
}

}  // namespace

NuclearDerivativeResult FiniteDifferenceNuclearDerivativeCalculator::_run_impl(
    std::shared_ptr<data::Structure> structure, int charge,
    int spin_multiplicity, NuclearDerivativeSeedType seed) const {
  const ScopedLogLevel scoped_log_level(utils::LogLevel::error);
  if (!structure) {
    throw std::invalid_argument("Structure must not be null");
  }
  const double step = _settings->get<double>("finite_difference_step");
  const bool compute_hessian = _settings->get<bool>("compute_hessian");
  const auto dimension =
      static_cast<Eigen::Index>(3 * structure->get_num_atoms());

  auto central = evaluate_energy(*_settings, structure, charge,
                                 spin_multiplicity, seed, true);
  Eigen::VectorXd gradients = Eigen::VectorXd::Zero(dimension);
  std::vector<double> plus_energies(static_cast<size_t>(dimension));
  std::vector<double> minus_energies(static_cast<size_t>(dimension));

  for (Eigen::Index coordinate = 0; coordinate < dimension; ++coordinate) {
    auto plus_structure = displace_structure(structure, coordinate, step);
    auto minus_structure = displace_structure(structure, coordinate, -step);
    plus_energies[coordinate] =
        evaluate_energy(*_settings, plus_structure, charge, spin_multiplicity,
                        seed, false)
            .energy;
    minus_energies[coordinate] =
        evaluate_energy(*_settings, minus_structure, charge, spin_multiplicity,
                        seed, false)
            .energy;
    gradients(coordinate) =
        (plus_energies[coordinate] - minus_energies[coordinate]) / (2.0 * step);
  }

  std::optional<std::shared_ptr<data::NuclearHessian>> hessian;
  if (compute_hessian) {
    Eigen::MatrixXd hessian_matrix =
        Eigen::MatrixXd::Zero(dimension, dimension);
    for (Eigen::Index coordinate = 0; coordinate < dimension; ++coordinate) {
      hessian_matrix(coordinate, coordinate) =
          (plus_energies[coordinate] - 2.0 * central.energy +
           minus_energies[coordinate]) /
          (step * step);
    }

    for (Eigen::Index i = 0; i < dimension; ++i) {
      for (Eigen::Index j = i + 1; j < dimension; ++j) {
        auto pp =
            displace_structure(displace_structure(structure, i, step), j, step);
        auto pm = displace_structure(displace_structure(structure, i, step), j,
                                     -step);
        auto mp = displace_structure(displace_structure(structure, i, -step), j,
                                     step);
        auto mm = displace_structure(displace_structure(structure, i, -step), j,
                                     -step);
        const double value = (evaluate_energy(*_settings, pp, charge,
                                              spin_multiplicity, seed, false)
                                  .energy -
                              evaluate_energy(*_settings, pm, charge,
                                              spin_multiplicity, seed, false)
                                  .energy -
                              evaluate_energy(*_settings, mp, charge,
                                              spin_multiplicity, seed, false)
                                  .energy +
                              evaluate_energy(*_settings, mm, charge,
                                              spin_multiplicity, seed, false)
                                  .energy) /
                             (4.0 * step * step);
        hessian_matrix(i, j) = value;
        hessian_matrix(j, i) = value;
      }
    }
    if (_settings->get<bool>("symmetrize_hessian")) {
      hessian_matrix = 0.5 * (hessian_matrix + hessian_matrix.transpose());
    }
    hessian = std::make_shared<data::NuclearHessian>(copy_structure(structure),
                                                     hessian_matrix);
  }

  return {central.energy,
          std::make_shared<data::NuclearGradients>(copy_structure(structure),
                                                   gradients),
          hessian, central.wavefunction};
}

std::unique_ptr<NuclearDerivativeCalculator>
make_finite_difference_nuclear_derivative_calculator() {
  return std::make_unique<FiniteDifferenceNuclearDerivativeCalculator>();
}

std::unique_ptr<NuclearDerivativeCalculator>
make_qdk_nuclear_derivative_calculator() {
  return std::make_unique<QdkNuclearDerivativeCalculator>();
}

NuclearDerivativeResult QdkNuclearDerivativeCalculator::_run_impl(
    std::shared_ptr<data::Structure> structure, int charge,
    int spin_multiplicity, NuclearDerivativeSeedType seed) const {
  const ScopedLogLevel scoped_log_level(utils::LogLevel::error);
  if (!structure) {
    throw std::invalid_argument("Structure must not be null");
  }
  if (_settings->get<bool>("compute_hessian")) {
    throw std::invalid_argument(
        "The QDK analytic nuclear derivative calculator does not currently "
        "provide Hessians. Use the finite_difference implementation for "
        "numeric Hessians.");
  }

  const auto ref = _settings->get<data::AlgorithmRef>("energy_calculator");
  if (ref.get_algorithm_type() != ScfSolverFactory::algorithm_type_name() ||
      ref.get_algorithm_name() != "qdk") {
    throw std::invalid_argument(
        "The QDK analytic nuclear derivative calculator requires "
        "energy_calculator to reference scf_solver/qdk.");
  }

  microsoft::ScfSolver solver;
  if (ref.get_settings()) {
    solver.settings().update(*ref.get_settings());
  }

  auto scf_result = solver.run_with_analytic_gradient(
      structure, charge, spin_multiplicity, seed_to_scf_input(seed, true));
  std::shared_ptr<data::NuclearGradients> gradients;
  if (scf_result.nuclear_gradient.has_value()) {
    gradients = std::make_shared<data::NuclearGradients>(
        copy_structure(structure), *scf_result.nuclear_gradient);
  }

  return {scf_result.energy, gradients, std::nullopt, scf_result.wavefunction};
}

void NuclearDerivativeCalculatorFactory::register_default_instances() {
  NuclearDerivativeCalculatorFactory::register_instance(
      &make_finite_difference_nuclear_derivative_calculator);
  NuclearDerivativeCalculatorFactory::register_instance(
      &make_qdk_nuclear_derivative_calculator);
}

}  // namespace qdk::chemistry::algorithms

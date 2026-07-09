// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <limits>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/nuclear_gradients.hpp>
#include <qdk/chemistry/data/nuclear_hessian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace qdk::chemistry::algorithms {

/**
 * @brief Initial basis-set name, basis, orbitals, or wavefunction seed for a
 * derivative run.
 *
 * A string seed is interpreted as the basis-set name, such as "sto-3g", to use
 * for SCF calculations.
 */
using NuclearDerivativeSeedType =
    std::variant<std::shared_ptr<data::Orbitals>,
                 std::shared_ptr<data::BasisSet>,
                 std::shared_ptr<data::Wavefunction>, std::string>;

/**
 * @brief Energy, gradients, optional Hessian, and optional wavefunction.
 */
using NuclearDerivativeResult =
    std::tuple<double, std::shared_ptr<data::NuclearGradients>,
               std::optional<std::shared_ptr<data::NuclearHessian>>,
               std::optional<std::shared_ptr<data::Wavefunction>>>;

/**
 * @class NuclearDerivativeSettings
 * @brief Settings for nuclear derivative calculations.
 */
class NuclearDerivativeSettings : public data::Settings {
 public:
  /**
   * @brief Construct settings with shared nuclear derivative defaults.
   */
  NuclearDerivativeSettings() {
    set_default(
        "energy_calculator", data::AlgorithmRef("scf_solver", "qdk"),
        "Algorithm used for each energy evaluation. Use an scf_solver for "
        "direct SCF finite differences, a multi_configuration_scf solver for "
        "MCSCF energies, or a multi_configuration_calculator such as CASCI or "
        "ASCI for Hamiltonian-based multi-reference energies.");
    allow_algorithm_ref_type_change("energy_calculator");
    set_default(
        "orbital_solver", data::AlgorithmRef("scf_solver", "qdk"),
        "SCF solver used to generate reference orbitals for multi-reference "
        "energy paths. This setting is ignored for direct SCF energy paths and "
        "is skipped when the derivative input seed already provides usable "
        "orbitals for the current geometry.");
    set_default("hamiltonian_constructor",
                data::AlgorithmRef("hamiltonian_constructor", "qdk"),
                "Hamiltonian constructor used when energy_calculator is a "
                "multi_configuration_calculator. It builds the active-space "
                "Hamiltonian from the reference orbitals before the energy "
                "calculation.");
    set_default("compute_hessian", false,
                "Whether to compute a nuclear Hessian in addition to energy "
                "and gradients.");
    set_default(
        "reuse_seed_active_space", true,
        "For multi-reference energy paths, reuse active and inactive orbital "
        "indices from an Orbitals or Wavefunction seed when fresh reference "
        "orbitals are generated for another geometry. Orbital coefficients are "
        "not reused for displaced finite-difference geometries.");
    set_default(
        "localize_reference_orbitals", false,
        "Whether to localize reference orbitals before multi-reference energy "
        "evaluations. Localization is applied only to MR energy paths and uses "
        "the current active-space orbital indices as the localization subset.");
    set_default(
        "orbital_localizer",
        data::AlgorithmRef("orbital_localizer", "qdk_pipek_mezey"),
        "Orbital localizer used when localize_reference_orbitals is true. The "
        "localizer runs after reference orbitals are obtained and active-space "
        "metadata from the seed has been reapplied.");
    set_default(
        "n_active_alpha_electrons", static_cast<int64_t>(0),
        "Number of alpha electrons in the active space for "
        "multi_configuration_scf and multi_configuration_calculator energy "
        "paths. Must be positive when those energy paths are selected.",
        data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
    set_default(
        "n_active_beta_electrons", static_cast<int64_t>(0),
        "Number of beta electrons in the active space for "
        "multi_configuration_scf and multi_configuration_calculator energy "
        "paths. Must be positive when those energy paths are selected.",
        data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
  }
};

/**
 * @class FiniteDifferenceNuclearDerivativeSettings
 * @brief Settings for finite-difference nuclear derivative calculations.
 */
class FiniteDifferenceNuclearDerivativeSettings
    : public NuclearDerivativeSettings {
 public:
  /**
   * @brief Construct settings with finite-difference defaults.
   */
  FiniteDifferenceNuclearDerivativeSettings() : NuclearDerivativeSettings() {
    set_default("finite_difference_step", 1.0e-3,
                "Central finite-difference nuclear displacement step in Bohr. "
                "Used for numeric gradients and Hessians.",
                data::BoundConstraint<double>{1.0e-8, 1.0});
    set_default("symmetrize_hessian", true,
                "Whether to replace the finite-difference Hessian by the "
                "average of itself and its transpose before returning it.");
  }
};

/**
 * @class NuclearDerivativeCalculator
 * @brief Base class for nuclear derivative algorithms.
 *
 * Implementations compute a total energy and nuclear gradients for a molecular
 * structure. Hessians are returned when requested by settings, and a
 * wavefunction is returned when the selected energy path naturally produces
 * one.
 */
class NuclearDerivativeCalculator
    : public Algorithm<NuclearDerivativeCalculator, NuclearDerivativeResult,
                       std::shared_ptr<data::Structure>, int, int,
                       NuclearDerivativeSeedType> {
 public:
  /**
   * @brief Construct a calculator with nuclear derivative settings.
   */
  NuclearDerivativeCalculator() {
    _settings = std::make_unique<NuclearDerivativeSettings>();
  }
  virtual ~NuclearDerivativeCalculator() = default;

  using Algorithm::run;

  virtual std::string name() const = 0;

  /**
   * @brief Return the factory type name for nuclear derivative calculators.
   */
  std::string type_name() const final {
    return "nuclear_derivative_calculator";
  }

 protected:
  /**
   * @brief Implementation hook for derived nuclear derivative calculators.
   */
  virtual NuclearDerivativeResult _run_impl(
      std::shared_ptr<data::Structure> structure, int charge,
      int spin_multiplicity, NuclearDerivativeSeedType seed) const = 0;
};

/**
 * @brief Factory for nuclear derivative calculator implementations.
 */
struct NuclearDerivativeCalculatorFactory
    : public AlgorithmFactory<NuclearDerivativeCalculator,
                              NuclearDerivativeCalculatorFactory> {
  /**
   * @brief Return the algorithm type name managed by this factory.
   */
  static std::string algorithm_type_name() {
    return "nuclear_derivative_calculator";
  }

  /**
   * @brief Register built-in nuclear derivative calculator implementations.
   */
  static void register_default_instances();

  /**
   * @brief Return the default nuclear derivative implementation name.
   */
  static std::string default_algorithm_name() { return "qdk_finite_difference"; }
};

/**
 * @class FiniteDifferenceNuclearDerivativeCalculator
 * @brief Numeric nuclear derivative calculator using central finite
 * differences.
 */
class FiniteDifferenceNuclearDerivativeCalculator
    : public NuclearDerivativeCalculator {
 public:
  /**
   * @brief Construct a calculator with finite-difference settings.
   */
  FiniteDifferenceNuclearDerivativeCalculator() {
    _settings = std::make_unique<FiniteDifferenceNuclearDerivativeSettings>();
  }

  /**
   * @brief Return the implementation name.
   */
  std::string name() const final { return "finite_difference"; }

  /**
   * @brief Return accepted factory aliases for this implementation.
   */
  std::vector<std::string> aliases() const final {
    return {"finite_difference", "numeric"};
  }

 protected:
  /**
   * @brief Compute finite-difference nuclear derivatives.
   */
  NuclearDerivativeResult _run_impl(
      std::shared_ptr<data::Structure> structure, int charge,
      int spin_multiplicity,
      NuclearDerivativeSeedType seed_or_basis) const override;
};

/**
 * @class QdkNuclearDerivativeCalculator
 * @brief Internal QDK derivative calculator using analytic SCF gradients.
 */
class QdkNuclearDerivativeCalculator : public NuclearDerivativeCalculator {
 public:
  /**
   * @brief Return the implementation name.
   */
  std::string name() const final { return "qdk"; }

  /**
   * @brief Return accepted factory aliases for this implementation.
   */
  std::vector<std::string> aliases() const final {
    return {"qdk", "analytical_gradient"};
  }

 protected:
  /**
   * @brief Compute analytic nuclear gradients with the internal SCF backend.
   */
  NuclearDerivativeResult _run_impl(
      std::shared_ptr<data::Structure> structure, int charge,
      int spin_multiplicity,
      NuclearDerivativeSeedType seed_or_basis) const override;
};

}  // namespace qdk::chemistry::algorithms

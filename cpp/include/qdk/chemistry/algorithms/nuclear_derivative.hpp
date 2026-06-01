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
 * @brief Initial basis, orbitals, or wavefunction seed for a derivative run.
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
   * @brief Construct settings with finite-difference defaults.
   */
  NuclearDerivativeSettings() {
    set_default("energy_calculator", data::AlgorithmRef("scf_solver", "qdk"));
    set_default("orbital_solver", data::AlgorithmRef("scf_solver", "qdk"));
    set_default("hamiltonian_constructor",
                data::AlgorithmRef("hamiltonian_constructor", "qdk"));
    set_default("compute_hessian", false);
    set_default("finite_difference_step", 1.0e-3,
                "Nuclear displacement step in Bohr",
                data::BoundConstraint<double>{1.0e-8, 1.0});
    set_default("symmetrize_hessian", true);
    set_default(
        "n_active_alpha_electrons", static_cast<int64_t>(0),
        "Active alpha electrons for multi-configuration energy paths",
        data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
    set_default(
        "n_active_beta_electrons", static_cast<int64_t>(0),
        "Active beta electrons for multi-configuration energy paths",
        data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
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
  static std::string default_algorithm_name() { return "finite_difference"; }
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
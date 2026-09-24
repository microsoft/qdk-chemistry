// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>

#include "finite_difference_nuclear_derivative.hpp"
#include "qdk_nuclear_derivative.hpp"

namespace qdk::chemistry::algorithms {

NuclearDerivativeSettings::NuclearDerivativeSettings() {
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
      "suppress_child_algorithm_logging", true,
      "Whether to temporarily raise the process-global log level to error "
      "while nuclear derivative calculators run child energy evaluations. "
      "Set false to preserve the current logger verbosity during those "
      "sub-runs.");
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
}

void NuclearDerivativeCalculatorFactory::register_default_instances() {
  NuclearDerivativeCalculatorFactory::register_instance(
      &make_finite_difference_nuclear_derivative_calculator);
  NuclearDerivativeCalculatorFactory::register_instance(
      &make_qdk_nuclear_derivative_calculator);
}

}  // namespace qdk::chemistry::algorithms

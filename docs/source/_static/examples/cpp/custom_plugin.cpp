/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

// Custom plugin examples for QDK/Chemistry.
//
// This file demonstrates how to extend QDK/Chemistry with custom plugins:
// 1. Adding a new backend for an existing algorithm type
// 2. Defining an entirely new algorithm type

#include <qdk/chemistry.hpp>

// -----------------------------------------------------------------------------
// start-cell-custom-settings
class CustomScfSettings
    : public qdk::chemistry::algorithms::ElectronicStructureSettings {
 public:
  CustomScfSettings() : ElectronicStructureSettings() {
    // Define additional settings beyond the inherited defaults
    set_default("custom_option", "default_value");
  }
};
// end-cell-custom-settings
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-custom-scf-solver
#include <qdk/chemistry/algorithms/scf.hpp>

#include "external_chemistry_package.hpp"

class CustomScfSolver : public qdk::chemistry::algorithms::ScfSolver {
 public:
  CustomScfSolver() { _settings = std::make_unique<CustomScfSettings>(); }

  std::string name() const override { return "custom"; }

 protected:
  std::pair<double, std::shared_ptr<qdk::chemistry::data::Wavefunction>>
  _run_impl(std::shared_ptr<qdk::chemistry::data::Structure> structure,
            int charge, int spin_multiplicity,
            std::optional<std::shared_ptr<qdk::chemistry::data::Orbitals>>
                initial_guess) override {
    // Convert to external format
    auto external_mol = convert_to_external_format(structure);

    // Execute external calculation
    auto basis = _settings->get<std::string>("basis_set");
    auto [energy, external_orbitals] =
        external_package::run_scf(external_mol, basis);

    // Convert results to QDK format
    auto wavefunction = convert_to_qdk_wavefunction(external_orbitals);

    return {energy, wavefunction};
  }
};
// end-cell-custom-scf-solver
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-registration
#include <qdk/chemistry/algorithms/scf.hpp>

// Static registration during library initialization
static auto registration =
    qdk::chemistry::algorithms::ScfSolver::register_implementation(
        []() { return std::make_unique<CustomScfSolver>(); });
// end-cell-registration
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-descriptor-settings
class MolecularDescriptorSettings : public qdk::chemistry::data::Settings {
 public:
  MolecularDescriptorSettings() {
    set_default<bool>("normalize", false, "Normalize the descriptor");
  }
};
// end-cell-descriptor-settings
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-descriptor-base-class
class MolecularDescriptorCalculator
    : public qdk::chemistry::algorithms::Algorithm<
          MolecularDescriptorCalculator,
          double,                                            // Return type
          std::shared_ptr<qdk::chemistry::data::Structure>>  // Input type
{
 public:
  std::string type_name() const final {
    return "molecular_descriptor_calculator";
  }
};
// end-cell-descriptor-base-class
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-descriptor-factory
struct MolecularDescriptorCalculatorFactory
    : public qdk::chemistry::algorithms::AlgorithmFactory<
          MolecularDescriptorCalculator, MolecularDescriptorCalculatorFactory> {
  static std::string algorithm_type_name() {
    return "molecular_descriptor_calculator";
  }

  static std::string default_algorithm_name() { return "nuclear_charge"; }
};
// end-cell-descriptor-factory
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-descriptor-implementations
class NuclearChargeDescriptor : public MolecularDescriptorCalculator {
 public:
  NuclearChargeDescriptor() {
    _settings = std::make_unique<MolecularDescriptorSettings>();
  }

  std::string name() const override { return "nuclear_charge"; }

 protected:
  double _run_impl(std::shared_ptr<qdk::chemistry::data::Structure> structure)
      const override {
    const auto& charges = structure->get_nuclear_charges();
    double descriptor = 0.0;
    for (double charge : charges) descriptor += charge;
    if (_settings->get<bool>("normalize") && structure->get_num_atoms() > 0) {
      descriptor /= static_cast<double>(structure->get_num_atoms());
    }
    return descriptor;
  }
};
// end-cell-descriptor-implementations
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-descriptor-registration
static auto descriptor_registration = []() {
  MolecularDescriptorCalculatorFactory::register_instance(
      []() { return std::make_unique<NuclearChargeDescriptor>(); });
  return true;
}();
// end-cell-descriptor-registration
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-descriptor-usage
#include <iostream>
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;

int main() {
  // After registration, use like any built-in algorithm
  auto calculator =
      MolecularDescriptorCalculatorFactory::create("nuclear_charge");
  calculator->settings().set("normalize", true);

  // List available implementations
  auto available = MolecularDescriptorCalculatorFactory::available();
  std::cout << "Available molecular descriptor calculators: ";
  for (const auto& name : available) {
    std::cout << name << " ";
  }
  std::cout << std::endl;

  return 0;
}
// end-cell-descriptor-usage
// -----------------------------------------------------------------------------

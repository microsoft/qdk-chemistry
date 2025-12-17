// Custom plugin examples for QDK/Chemistry.
//
// This file demonstrates how to extend QDK/Chemistry with custom plugins:
// 1. Adding a new backend for an existing algorithm type
// 2. Defining an entirely new algorithm type

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

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
// start-cell-geometry-settings
class GeometryOptimizerSettings : public qdk::chemistry::data::Settings {
 public:
  GeometryOptimizerSettings() {
    set_default<int64_t>(
        "max_steps", 100, "Maximum optimization steps",
        qdk::chemistry::data::BoundConstraint<int64_t>{1, 10000});
    set_default<double>("convergence_threshold", 1e-5,
                        "Gradient convergence threshold");
    set_default<double>("step_size", 0.1, "Initial optimization step size");
  }
};
// end-cell-geometry-settings
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-geometry-base-class
class GeometryOptimizer
    : public qdk::chemistry::algorithms::Algorithm<
          GeometryOptimizer,
          std::shared_ptr<qdk::chemistry::data::Structure>,  // Return type
          std::shared_ptr<qdk::chemistry::data::Structure>>  // Input type
{
 public:
  static std::string type_name() { return "geometry_optimizer"; }
};
// end-cell-geometry-base-class
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-geometry-factory
// The Algorithm base class template provides factory functionality
// automatically.
// end-cell-geometry-factory
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-geometry-implementations
class BfgsOptimizer : public GeometryOptimizer {
 public:
  BfgsOptimizer() { _settings = std::make_unique<GeometryOptimizerSettings>(); }

  std::string name() const override { return "bfgs"; }

 protected:
  std::shared_ptr<qdk::chemistry::data::Structure> _run_impl(
      std::shared_ptr<qdk::chemistry::data::Structure> structure) override {
    auto max_steps = _settings->get<int64_t>("max_steps");
    auto threshold = _settings->get<double>("convergence_threshold");

    // BFGS optimization implementation
    return optimized_structure;
  }
};
// end-cell-geometry-implementations
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-geometry-registration
// During library initialization
static auto factory_reg = register_factory<GeometryOptimizer>();
static auto bfgs_reg = GeometryOptimizer::register_implementation(
    []() { return std::make_unique<BfgsOptimizer>(); });
// end-cell-geometry-registration
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-usage
#include <iostream>
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;

int main() {
  // After registration, use like any built-in algorithm
  auto optimizer = GeometryOptimizerFactory::create("bfgs");
  optimizer->settings().set("max_steps", 200);
  optimizer->settings().set("convergence_threshold", 1e-6);

  // List available implementations
  auto available = GeometryOptimizerFactory::available();
  std::cout << "Available geometry optimizers: ";
  for (const auto& name : available) {
    std::cout << name << " ";
  }
  std::cout << std::endl;

  return 0;
}
// end-cell-usage
// -----------------------------------------------------------------------------

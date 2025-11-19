// Interfaces usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// start-cell-1
#include <qdk/chemistry.hpp>

// Create an SCF solver that uses the QDK/Chemistry library as solver
auto scf = ScfSolverFactory::create();

// Configure it using the standard settings interface
scf->settings().set("basis_set", "cc-pvdz");
scf->settings().set("method", "hf");

// Run calculation with the same API as native implementations
auto [energy, orbitals] = scf->solve(structure);
// end-cell-1

// start-cell-2
#include <iostream>
#include <qdk/chemistry.hpp>

// Get a list of available SCF solver implementations
auto available_solvers = ScfSolverFactory::available();
for (const auto& solver : available_solvers) {
  std::cout << solver << std::endl;
}

// Get documentation for a specific implementation
std::cout << ScfSolverFactory::get_docstring("default") << std::endl;
// end-cell-2

// start-cell-3
#include <qdk/chemistry.hpp>

#include "custom_chemistry_package.hpp"

namespace qdk::chemistry {
namespace algorithms {

class CustomScfSolver : public ScfSolver {
 public:
  CustomScfSolver() = default;

  std::tuple<double, data::Orbitals> solve(
      const data::Structure& structure) override {
    // Convert QDK/Chemistry structure to custom package format
    auto custom_mol = convert_to_custom_format(structure);

    // Run calculation with custom package
    auto result = custom_chemistry::run_scf(
        custom_mol, settings().get<std::string>("basis_set"),
        settings().get<std::string>("method"));

    // Convert results back to QDK/Chemistry format
    double energy = result.energy;
    data::Orbitals orbitals = convert_from_custom_format(result.orbitals);

    return {energy, orbitals};
  }

 private:
  custom_chemistry::Molecule convert_to_custom_format(
      const data::Structure& structure);
  data::Orbitals convert_from_custom_format(
      const custom_chemistry::Orbitals& orbitals);
};

// Register in a static initializer block
namespace {
bool registered = ScfSolverFactory::register_implementation(
    "custom", []() { return std::make_unique<CustomScfSolver>(); },
    "Interface to Custom Chemistry Package");
}  // anonymous namespace

}  // namespace algorithms
}  // namespace qdk::chemistry
// end-cell-3

// start-cell-4
// Set general options that work across all backends
scf->settings().set("basis_set", "cc-pvdz");
scf->settings().set("max_iterations", 100);
// end-cell-4

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Factory Pattern usage examples.
// -----------------------------------------------------------------------------
// start-cell-scf-localizer
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Create a simple molecule
auto structure = Structure::from_xyz("2\n\nH 0.0 0.0 0.0\nH 0.0 0.0 1.4");

// Create a SCF solver using the default implementation
auto scf_solver = ScfSolverFactory::create();

// Create an orbital localizer using a specific implementation
auto localizer = LocalizerFactory::create("qdk_pipek_mezey");

// Configure the SCF solver and run
auto [E_scf, wfn] = scf_solver->run(structure, 0, 1, "cc-pvdz");
// end-cell-scf-localizer
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-list-algorithms
#include <iostream>
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;

// List available SCF solver implementations
auto scf_methods = ScfSolverFactory::available();
std::cout << "Available SCF solvers:" << std::endl;
for (const auto& name : scf_methods) {
  std::cout << "  - " << name << std::endl;
}

// List available localizer implementations
auto localizer_methods = LocalizerFactory::available();
std::cout << "Available localizers:" << std::endl;
for (const auto& name : localizer_methods) {
  std::cout << "  - " << name << std::endl;
}

// List available Hamiltonian constructor implementations
auto ham_methods = HamiltonianConstructorFactory::available();
std::cout << "Available Hamiltonian constructors:" << std::endl;
for (const auto& name : ham_methods) {
  std::cout << "  - " << name << std::endl;
}

// List available multi-configuration calculator implementations
auto mc_methods = MultiConfigurationCalculatorFactory::available();
std::cout << "Available MC calculators:" << std::endl;
for (const auto& name : mc_methods) {
  std::cout << "  - " << name << std::endl;
}

// Show default implementation for each factory type
std::cout << "Default SCF solver: " << ScfSolverFactory::default_name()
          << std::endl;
std::cout << "Default localizer: " << LocalizerFactory::default_name()
          << std::endl;
std::cout << "Default Hamiltonian constructor: "
          << HamiltonianConstructorFactory::default_name() << std::endl;
std::cout << "Default MC calculator: "
          << MultiConfigurationCalculatorFactory::default_name() << std::endl;
// end-cell-list-algorithms
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-inspect-settings
#include <iostream>
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;

// Create a SCF solver and inspect its settings
auto scf = ScfSolverFactory::create("qdk");

// Print settings as a formatted table
std::cout << scf->settings().as_table() << std::endl;

// Or iterate over individual settings
for (const auto& key : scf->settings().keys()) {
  std::cout << key << ": " << scf->settings().get_as_string(key) << std::endl;
}
// end-cell-inspect-settings
// -----------------------------------------------------------------------------

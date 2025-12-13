// Interfaces usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-scf
#include <qdk/chemistry.hpp>

// Create a SCF solver that uses the QDK/Chemistry library as solver
auto scf = ScfSolverFactory::create();

// Configure it using the standard settings interface
scf->settings().set("basis_set", "cc-pvdz");
scf->settings().set("method", "hf");

// Run calculation with the same API as native implementations
auto [energy, orbitals] = scf->solve(structure);
// end-cell-scf
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-list-methods
#include <iostream>
#include <qdk/chemistry.hpp>

// Get a list of available SCF solver implementations
auto available_solvers = ScfSolverFactory::available();
for (const auto& solver : available_solvers) {
  std::cout << solver << std::endl;
}

// Get documentation for a specific implementation
std::cout << ScfSolverFactory::get_docstring("default") << std::endl;
// end-cell-list-methods
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-settings
// Set general options that work across all backends
scf->settings().set("basis_set", "cc-pvdz");
scf->settings().set("max_iterations", 100);
// end-cell-settings
// -----------------------------------------------------------------------------

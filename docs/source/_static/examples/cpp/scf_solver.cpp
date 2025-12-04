// Scf Solver usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-create
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;

// Create the default ScfSolver instance
auto scf_solver = ScfSolverFactory::create();

// Or specify a particular solver implementation
auto pyscf_solver = ScfSolverFactory::create("pyscf");
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-configure
// Standard settings that work with all solvers
// Set the method
scf_solver.settings()
    .set("method", "dft")
    // Set the basis set
    scf_solver->settings()
    .set("basis_set", "def2-tzvpp");

// For DFT calculations, set the exchange-correlation functional
scf_solver->settings().set("functional", "B3LYP");
// end-cell-configure
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-run
// Create a structure (or load from a file)
Structure structure;
// configuring structure ...

// Run the SCF calculation
// Return types are: std::tuple<double, Orbitals>
auto [E_scf, scf_orbitals] = scf_solver->solve(structure);
std::cout << "SCF Energy: " << E_scf << " Hartree" << std::endl;
// end-cell-run
// --------------------------------------------------------------------------------------------

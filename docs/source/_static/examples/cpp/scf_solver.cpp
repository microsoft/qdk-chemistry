// Scf Solver usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-create
#include <iostream>
#include <qdk/chemistry.hpp>
#include <string>
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Create default ScfSolver instance
auto scf_solver = ScfSolverFactory::create();
// end-cell-create
// --------------------------------------------------------------------------------------------

int main() {
  // --------------------------------------------------------------------------------------------
  // start-cell-configure
  // Standard settings that work with all solvers
  // Set the method
  // Note the following line is optional, since hf is the default method
  scf_solver->settings().set("method", "hf");
  // Set the basis set
  scf_solver->settings().set("basis_set", "def2-tzvpp");

  // end-cell-configure
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-run
  // Specify a structure
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure structure(coords, symbols);

  // Run the SCF calculation
  auto [E_scf, wfn] = scf_solver->run(structure, 0, 1);
  auto scf_orbitals = wfn->get_orbitals();
  std::cout << "SCF Energy: " << E_scf << " Hartree" << std::endl;
  // end-cell-run
  // --------------------------------------------------------------------------------------------
  return 0;
}

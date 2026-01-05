// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Scf Solver usage examples.
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
  // Set the method
  // Note the following line is optional, since hf is the default method
  scf_solver->settings().set("method", "hf");

  // end-cell-configure
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-run
  // Load structure from XYZ file
  auto structure = Structure::from_xyz_file("../data/h2.structure.xyz");

  // Run the SCF calculation
  auto [E_scf, wfn] = scf_solver->run(structure, 0, 1, "def2-tzvpp");
  auto scf_orbitals = wfn->get_orbitals();
  std::cout << "SCF Energy: " << E_scf << " Hartree" << std::endl;
  // end-cell-run
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-list-implementations
  auto names = ScfSolverFactory::available();
  for (const auto& name : names) {
    std::cout << name << std::endl;
  }
  // end-cell-list-implementations
  // --------------------------------------------------------------------------------------------
  return 0;
}

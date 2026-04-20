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
  // start-cell-alternative-run
  // Run scf with an initial guess from previous orbitals
  auto [E_scf2, wfn2] = scf_solver->run(structure, 0, 1, scf_orbitals);

  // Run scf with a custom basis set
  auto basis_set = BasisSet::from_basis_name("def2-tzvpp", structure);
  auto [E_scf3, wfn3] = scf_solver->run(structure, 0, 1, basis_set);
  // end-cell-alternative-run
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-list-implementations
  auto names = ScfSolverFactory::available();
  for (const auto& name : names) {
    std::cout << name << std::endl;
  }
  // end-cell-list-implementations
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-dfj
  // Run SCF with density-fitted Coulomb integrals (DF-J)
  // Create a basis set with an auxiliary basis for density fitting
  auto dfj_basis = BasisSet::from_basis_name("def2-svp", "def2-universal-jfit",
                                             structure);

  // Configure the solver to use incore ERIs (required for DF-J)
  auto dfj_solver = ScfSolverFactory::create();
  dfj_solver->settings().set("eri_method", "incore");

  // Run - DF-J is automatically enabled when auxiliary basis is detected
  auto [E_dfj, wfn_dfj] = dfj_solver->run(structure, 0, 1, dfj_basis);
  std::cout << "DF-J SCF Energy: " << E_dfj << " Hartree" << std::endl;
  // end-cell-dfj
  // --------------------------------------------------------------------------------------------

  return 0;
}

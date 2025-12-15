// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Design Principles usage examples.
// -----------------------------------------------------------------------------
// start-cell-scf-create
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

auto scf = ScfSolverFactory::create();
// end-cell-scf-create
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-scf-settings
scf_solver->settings().set("max_iterations", 100);
// end-cell-scf-settings
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-data-flow
int main() {
  // Create molecular structure from an XYZ file
  Structure molecule;
  molecule.from_xyz_file("molecule.xyz");

  // Configure and run SCF calculation
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("basis_set", "cc-pvdz");
  auto [scf_energy, orbitals] = scf_solver->solve(molecule);

  // Select active space orbitals
  auto active_selector = ActiveSpaceSelectorFactory::create();
  active_selector->settings().set("num_active_orbitals", 6);
  active_selector->settings().set("num_active_electrons", 6);
  auto active_indices = active_selector->select(orbitals);

  // Create Hamiltonian with active space
  auto ham_constructor = HamiltonianConstructorFactory::create();
  ham_constructor->settings().set("active_orbitals", active_indices);
  auto hamiltonian = ham_constructor->run(orbitals);

  // Run multi-configuration calculation
  auto mc_solver = MCCalculatorFactory::create();
  auto [mc_energy, wave_function] = mc_solver->solve(hamiltonian);

  return 0;
}
// end-cell-data-flow
// -----------------------------------------------------------------------------

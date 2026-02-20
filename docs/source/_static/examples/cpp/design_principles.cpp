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
std::cout << "Available settings: " << scf_solver.settings().get_summary()
          << std::endl;
scf_solver->settings().set("max_iterations", 100);
// end-cell-scf-settings
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-data-flow
int main() {
  // Create molecular structure from an XYZ file
  auto molecule = Structure::from_xyz_file("molecule.xyz");

  // Configure and run SCF calculation
  auto scf_solver = ScfSolverFactory::create();
  auto [scf_energy, wfn_hf] = scf_solver->run(molecule, 0, 1, "cc-pvdz");

  // Select active space orbitals
  auto active_selector = ActiveSpaceSelectorFactory::create("qdk_valence");
  active_selector->settings().set("num_active_orbitals", 6);
  active_selector->settings().set("num_active_electrons", 6);
  auto active_wfn = active_selector->run(wfn_hf);
  auto active_orbitals = active_wfn->get_orbitals();

  // Create Hamiltonian with active space
  auto ham_constructor = HamiltonianConstructorFactory::create();
  auto hamiltonian = ham_constructor->run(active_orbitals);

  // Run multi-configuration calculation
  auto mc_solver = MultiConfigurationCalculatorFactory::create();
  auto [mc_energy, wave_function] = mc_solver->run(hamiltonian, 3, 3);

  return 0;
}
// end-cell-data-flow
// -----------------------------------------------------------------------------

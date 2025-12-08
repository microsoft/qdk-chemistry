// Ansatz usage examples.

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
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

int main() {
  // Create H2 structure
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
  std::vector<std::string> symbols = {"H", "H"};

  Structure structure(coords, symbols);

  // SCF
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("basis_set", "sto-3g");
  auto [E_scf, wfn_scf] = scf_solver->run(structure, 0, 1);

  // Create Hamiltonian from SCF Orbitals
  auto ham_constructor = HamiltonianConstructorFactory::create();
  auto hamiltonian = ham_constructor->run(wfn_scf->get_orbitals());

  // Create Ansatz object from Hamiltonian and Wavefunction
  Ansatz ansatz(hamiltonian, wfn_scf);
  // end-cell-create
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-access
  // Access Ansatz components
  auto hamiltonian_ref = ansatz.get_hamiltonian();
  auto wavefunction_ref = ansatz.get_wavefunction();
  auto orbitals_ref = ansatz.get_orbitals();

  // Check component availability
  bool has_hamiltonian = ansatz.has_hamiltonian();
  bool has_wavefunction = ansatz.has_wavefunction();
  bool has_orbitals = ansatz.has_orbitals();

  // Calculate energy expectation value
  double energy = ansatz.calculate_energy();
  std::cout << "Energy expectation value: " << energy << " Ha" << std::endl;

  // Get summary
  std::string summary = ansatz.get_summary();
  std::cout << "Ansatz summary: " << summary << std::endl;
  // end-cell-access
  // --------------------------------------------------------------------------------------------
  return 0;
}

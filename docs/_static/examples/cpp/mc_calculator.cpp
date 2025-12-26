// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Mc Calculator usage examples.
// --------------------------------------------------------------------------------------------
// start-cell-create
#include <iostream>
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Create a CAS MultiConfigurationCalculator instance
auto mc_calculator = MultiConfigurationCalculatorFactory::create("macis_cas");
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-configure
// Set the convergence threshold for the CI iterations
mc_calculator->settings().set("ci_residual_tolerance", 1.0e-6);
// end-cell-configure
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-run
// Create a structure (H2 molecule)
std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
std::vector<std::string> symbols = {"H", "H"};
Structure structure(coords, symbols);

// Run SCF to get orbitals
auto scf_solver = ScfSolverFactory::create();
auto [E_scf, wfn] = scf_solver->run(structure, 0, 1, "sto-3g");

// Build Hamiltonian from orbitals
auto ham_constructor = HamiltonianConstructorFactory::create();
auto hamiltonian = ham_constructor->run(wfn->get_orbitals());

// Run the CI calculation
// For H2, we have 2 electrons (1 alpha, 1 beta)
int n_alpha = 1;
int n_beta = 1;
auto [E_ci, ci_wavefunction] = mc_calculator->run(hamiltonian, n_alpha, n_beta);

std::cout << "SCF Energy: " << E_scf << " Hartree" << std::endl;
std::cout << "CI Energy:  " << E_ci << " Hartree" << std::endl;
// end-cell-run
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-list-implementations
auto names = MultiConfigurationCalculatorFactory::available();
for (const auto& name : names) {
  std::cout << name << std::endl;
}
// end-cell-list-implementations
// --------------------------------------------------------------------------------------------

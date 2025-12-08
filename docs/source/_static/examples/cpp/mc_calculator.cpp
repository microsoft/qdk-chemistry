// Mc Calculator usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-create
#include <iostream>
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// List available multi-configuration calculator implementations
auto available_mc = MultiConfigurationCalculatorFactory::available();
std::cout << "Available MC calculators:" << std::endl;
for (const auto& name : available_mc) {
  std::cout << "  - " << name << std::endl;
}

// Create the default MultiConfigurationCalculator instance (MACIS
// implementation)
auto mc_calculator = MultiConfigurationCalculatorFactory::create();

// Create a specific type of CI calculator
auto selected_ci = MultiConfigurationCalculatorFactory::create("macis_cas");
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-configure
// Set the convergence threshold for the CI iterations
mc_calculator->settings().set("ci_residual_threshold", 1.0e-6);

// Set the maximum number of Davidson iterations
mc_calculator->settings().set("davidson_iterations", 200);

// Calculate one-electron reduced density matrix
mc_calculator->settings().set("calculate_one_rdm", true);
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
scf_solver->settings().set("basis_set", "sto-3g");
auto [E_scf, wfn] = scf_solver->run(structure, 0, 1);

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
std::cout << "Correlation energy: " << E_ci - E_scf << " Hartree" << std::endl;
std::cout << ci_wavefunction->get_summary() << std::endl;
// end-cell-run
// --------------------------------------------------------------------------------------------

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Dynamical correlation examples.
// -----------------------------------------------------------------------------
// start-cell-mp2-example
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Create a simple structure
std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {1.4, 0.0, 0.0}};
std::vector<std::string> symbols = {"H", "H"};
Structure structure(coords, symbols);

// Run initial SCF
auto scf_solver = ScfSolverFactory::create();
auto [E_HF, wfn_HF] = scf_solver->run(structure, 0, 1);

// Create a Hamiltonian constructor
auto hamiltonian_constructor = HamiltonianConstructorFactory::create();

// Construct the Hamiltonian from orbitals
auto hamiltonian = hamiltonian_constructor->run(wfn_HF->get_orbitals());

// Create ansatz for MP2 calculation
auto ansatz = std::make_shared<Ansatz>(*hamiltonian, *wfn_HF);

// Run MP2
auto mp2_calculator =
    DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");

// Get energies
auto [mp2_total_energy, final_wavefunction] = mp2_calculator->run(ansatz);

// If desired, we can extract only the correlation energy
double mp2_corr_energy = mp2_total_energy - E_HF;
// end-cell-mp2-example

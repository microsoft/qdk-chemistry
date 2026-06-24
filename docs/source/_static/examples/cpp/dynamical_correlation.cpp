// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Dynamical correlation examples.
// -----------------------------------------------------------------------------
// start-cell-create
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Create a DynamicalCorrelationCalculator instance
auto mp2_calculator =
    DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");
// end-cell-create
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-configure
// Configure settings (for implementations that support them)
// mp2_calculator->settings().set("conv_tol", 1e-8);
// end-cell-configure
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-run
// Load H2 structure from XYZ file
auto structure = Structure::from_xyz_file("../data/h2.structure.xyz");

// Run initial SCF to get reference wavefunction
auto scf_solver = ScfSolverFactory::create();
auto [E_HF, wfn_HF] = scf_solver->run(structure, 0, 1);

// Create Hamiltonian from orbitals
auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
auto hamiltonian = hamiltonian_constructor->run(wfn_HF->get_orbitals());

// Create ansatz combining wavefunction and Hamiltonian
auto ansatz = std::make_shared<Ansatz>(*hamiltonian, *wfn_HF);

// Run the correlation calculation
auto [mp2_total_energy, final_wavefunction] = mp2_calculator->run(ansatz);

// Extract correlation energy
double mp2_corr_energy = mp2_total_energy - E_HF;
std::cout << "MP2 Correlation Energy: " << mp2_corr_energy << " Hartree\n";
std::cout << "MP2 Total Energy: " << mp2_total_energy << " Hartree\n";
// end-cell-run
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// start-cell-list-implementations
auto names = DynamicalCorrelationCalculatorFactory::available();
for (const auto& name : names) {
  std::cout << name << std::endl;
}
// end-cell-list-implementations
// -----------------------------------------------------------------------------

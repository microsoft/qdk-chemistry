// Mc Calculator usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-create
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;

// Create the default MultiConfigurationCalculator instance (MACIS
// implementation)
auto mc_calculator = MultiConfigurationCalculatorFactory::create();

// Create a specific type of CI calculator
auto selected_ci = MultiConfigurationCalculatorFactory::create("macis_cas");
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-configure
// Set the number of states to solve for (ground state + two excited states)
mc_calculator->settings().set("num_roots", 3);

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
// Obtain a valid Hamiltonian
Hamiltonian hamiltonian;
/* hamiltonian = ... */

// Run the CI calculation
auto [E_ci, wavefunction] = mc_calculator->calculate(hamiltonian);

// For multiple states, access the energies and wavefunctions
auto energies = mc_calculator->get_energies();
auto wavefunctions = mc_calculator->get_wavefunctions();
// end-cell-run
// --------------------------------------------------------------------------------------------

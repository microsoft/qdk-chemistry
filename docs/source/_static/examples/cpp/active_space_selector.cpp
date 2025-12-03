// Active space selection example.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

#include <iostream>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/scf_solver.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <vector>

using namespace qdk::chemistry;

// --------------------------------------------------------------------------------------------
// start-cell-create
// Create a molecular structure (water molecule)
std::vector<std::vector<double>> coords = {
    {0.0, 0.0, 0.0}, {0.0, 0.0, 1.8897}, {1.7802, 0.0, -0.4738}};
std::vector<std::string> symbols = {"O", "H", "H"};
auto structure = std::make_shared<data::Structure>(coords, symbols);

// First, run SCF to get molecular orbitals
auto scf_solver = algorithms::ScfSolverFactory::create();
scf_solver->settings().set("basis_set", "6-31g");
auto [scf_energy, scf_wavefunction] = scf_solver->run(*structure, 0, 1);

// Create an active space selector using the default implementation
auto active_space_selector = algorithms::ActiveSpaceSelectorFactory::create();
std::cout << "Default active space selector: " << active_space_selector->name()
          << std::endl;
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-configure
// Configure the selector for valence-based selection
// Select 4 electrons in 4 orbitals (oxygen 2p and bonding/antibonding
// combinations)
auto valence_selector =
    algorithms::ActiveSpaceSelectorFactory::create("qdk_valence");
valence_selector->settings().set("num_active_electrons", 4);
valence_selector->settings().set("num_active_orbitals", 4);

// Alternative: occupation-based automatic selection
auto occupation_selector =
    algorithms::ActiveSpaceSelectorFactory::create("qdk_occupation");
occupation_selector->settings().set("occupation_threshold", 0.1);

// Alternative: entropy-based automatic selection
auto autocas_selector =
    algorithms::ActiveSpaceSelectorFactory::create("qdk_autocas");
autocas_selector->settings().set("min_plateau_size", 2);
// end-cell-configure
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-run
// Run active space selection with valence selector
auto active_wavefunction = valence_selector->run(scf_wavefunction);
auto active_orbitals = active_wavefunction->get_orbitals();

std::cout << "Active orbitals summary:\n"
          << active_orbitals->get_summary() << std::endl;
// The active space can now be used for multireference calculations

// end-cell-run
// --------------------------------------------------------------------------------------------

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Active space selection example.
// --------------------------------------------------------------------------------------------
// start-cell-create
#include <iostream>
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Create the default ActiveSpaceSelector instance
auto active_space_selector = ActiveSpaceSelectorFactory::create();
// end-cell-create
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-configure
// Configure the selector using the settings interface
// Set the number of electrons and orbitals for the active space
active_space_selector->settings().set("num_active_electrons", 4);
active_space_selector->settings().set("num_active_orbitals", 4);

// end-cell-configure
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-run
// Create a molecular structure (water molecule)
std::vector<Eigen::Vector3d> coords = {
    {0.0, 0.0, 0.0}, {0.0, 0.0, 1.8897}, {1.7802, 0.0, -0.4738}};
std::vector<std::string> symbols = {"O", "H", "H"};
Structure structure(coords, symbols);

// First, run SCF to get molecular orbitals
auto scf_solver = ScfSolverFactory::create();
auto [scf_energy, scf_wavefunction] = scf_solver->run(structure, 0, 1, "6-31g");

// Run active space selection
auto active_wavefunction = active_space_selector->run(scf_wavefunction);
auto active_orbitals = active_wavefunction->get_orbitals();

std::cout << "SCF Energy: " << scf_energy << " Hartree" << std::endl;
std::cout << "Active orbitals summary:\n"
          << active_orbitals->get_summary() << std::endl;
// end-cell-run
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-list-implementations
auto names = ActiveSpaceSelectorFactory::available();
for (const auto& name : names) {
  std::cout << name << std::endl;
}
// end-cell-list-implementations
// --------------------------------------------------------------------------------------------

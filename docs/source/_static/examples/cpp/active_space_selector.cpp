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

int main() {
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
  // Load a molecular structure (water molecule) from XYZ file
  auto structure = Structure::from_xyz_file("../data/water.structure.xyz");
  int charge = 0;

  // First, run SCF to get molecular orbitals
  auto scf_solver = ScfSolverFactory::create();
  auto [scf_energy, scf_wavefunction] =
      scf_solver->run(structure, charge, 1, "6-31g");

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

  // --------------------------------------------------------------------------------------------
  // start-cell-autocas
  // Create a valence space active space selector
  auto valence_selector = ActiveSpaceSelectorFactory::create("qdk_valence");
  // Automatically select valence parameters based on the input structure
  auto [num_electrons, num_orbitals] =
      qdk::chemistry::utils::compute_valence_space_parameters(scf_wavefunction,
                                                              charge);
  valence_selector->settings().set("num_active_electrons", num_electrons);
  valence_selector->settings().set("num_active_orbitals", num_orbitals);
  auto active_valence_wfn = valence_selector->run(scf_wavefunction);

  // Create active Hamiltonian
  auto active_hamiltonian_generator = HamiltonianConstructorFactory::create();
  auto active_hamiltonian =
      active_hamiltonian_generator->run(active_valence_wfn->get_orbitals());

  // Run Active Space Calculation with Selected CI
  auto mc_calculator =
      MultiConfigurationCalculatorFactory::create("macis_asci");
  mc_calculator->settings().set("ntdets_max", 50000);
  mc_calculator->settings().set("calculate_one_rdm", true);
  mc_calculator->settings().set("calculate_two_rdm", true);
  auto [mc_energy, mc_wavefunction] = mc_calculator->run(
      active_hamiltonian, num_electrons / 2, num_electrons / 2);

  // Print single orbital entropies which are used by autoCAS to determine the
  // active space
  auto entropies = mc_wavefunction->get_single_orbital_entropies();
  std::cout << "Single orbital entropies:" << std::endl;
  for (size_t idx = 0; idx < entropies.size(); ++idx) {
    std::cout << "Orbital " << (idx + 1) << ": " << entropies[idx] << std::endl;
  }

  // Select active space using autoCAS
  auto autocas_selector = ActiveSpaceSelectorFactory::create("qdk_autocas_eos");
  auto active_autocas_wfn = autocas_selector->run(mc_wavefunction);
  std::cout << "autoCAS selected active orbitals summary:\n"
            << active_autocas_wfn->get_orbitals()->get_summary() << std::endl;
  // end-cell-autocas
  // --------------------------------------------------------------------------------------------

  return 0;
}

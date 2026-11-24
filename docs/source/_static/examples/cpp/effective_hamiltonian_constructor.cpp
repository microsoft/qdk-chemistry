// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Effective Hamiltonian construction example.
// --------------------------------------------------------------------------------------------
// start-cell-create
#include <iostream>
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

int main() {
  // Create the DUCC effective Hamiltonian constructor
  auto ducc = EffectiveHamiltonianConstructorFactory::create("qdk_ducc");
  // end-cell-create
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-configure
  // Select the commutator level (0 to 2)
  ducc->settings().set("level", 2);
  // end-cell-configure
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // docs:xyz ../data/water.structure.xyz
  // start-cell-run
  auto structure = Structure::from_xyz(R"(3
Water molecule
O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
)");

  // Reference determinant and the Hamiltonian over the full orbital window W
  auto scf_solver = ScfSolverFactory::create();
  auto [scf_energy, scf_wavefunction] =
      scf_solver->run(structure, 0, 1, "sto-3g");
  auto hamiltonian = HamiltonianConstructorFactory::create()->run(
      scf_wavefunction->get_orbitals());

  // T1/T2 amplitudes over W; a coupled-cluster calculator can be used instead
  // of MP2
  auto mp2_calculator =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");
  auto [mp2_energy, amplitudes, bra] = mp2_calculator->run(
      std::make_shared<Ansatz>(*hamiltonian, *scf_wavefunction));

  // Target space P: a four-electron, four-orbital valence space
  auto active_space_selector =
      ActiveSpaceSelectorFactory::create("qdk_valence");
  active_space_selector->settings().set("num_active_electrons", 4);
  active_space_selector->settings().set("num_active_orbitals", 4);
  auto p_indices = active_space_selector->run(scf_wavefunction)
                       ->get_orbitals()
                       ->active_indices();

  // Downfold W into P, then diagonalize the effective Hamiltonian in P
  auto effective_hamiltonian = ducc->run(amplitudes, hamiltonian, p_indices);
  auto mc_calculator = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [ducc_energy, ci_wavefunction] =
      mc_calculator->run(effective_hamiltonian, 2, 2);
  std::cout << "SCF Energy: " << scf_energy << " Hartree" << std::endl;
  std::cout << "DUCC CASCI Energy: " << ducc_energy << " Hartree" << std::endl;
  // end-cell-run
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-list-implementations
  auto names = EffectiveHamiltonianConstructorFactory::available();
  for (const auto& name : names) {
    std::cout << name << std::endl;
  }
  // end-cell-list-implementations
  // --------------------------------------------------------------------------------------------
  return 0;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Hamiltonian Constructor usage examples.
#include <iomanip>
#include <iostream>
#include <memory>
#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

int main() {
  // --------------------------------------------------------------------------------------------
  // start-cell-create
  // Create the default HamiltonianConstructor instance
  auto hamiltonian_constructor = HamiltonianConstructorFactory::create();
  // end-cell-create
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-configure
  // Configure settings (check available options)
  // Note: Available settings can be inspected at runtime

  // Set ERI method if needed
  hamiltonian_constructor->settings().set("eri_method", "direct");
  // end-cell-configure
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // docs:xyz ../data/h2.structure.xyz
  // start-cell-construct
  // Load structure from inline XYZ file
  auto structure = Structure::from_xyz(R"(2
H2 molecule
H    0.000000    0.000000    0.000000
H    0.000000    0.000000    0.740848
)");

  // Run a SCF to get orbitals
  auto scf_solver = ScfSolverFactory::create();
  auto [E_scf, wfn] = scf_solver->run(structure, 0, 1, "sto-3g");
  auto orbitals = wfn->get_orbitals();

  // Construct the Hamiltonian from orbitals
  auto hamiltonian = hamiltonian_constructor->run(orbitals);

  // Access the resulting integrals
  auto [h1_a, h1_b] = hamiltonian->get_one_body_integrals();
  auto [h2_aaaa, h2_aabb, h2_bbbb] = hamiltonian->get_two_body_integrals();
  auto core_energy = hamiltonian->get_core_energy();

  std::cout << "One-body integrals shape: " << h1_a.rows() << "x" << h1_a.cols()
            << std::endl;
  std::cout << "Two-body integrals size: " << h2_aaaa.size() << std::endl;
  std::cout << "Core energy: " << std::fixed << std::setprecision(10)
            << core_energy << " Hartree" << std::endl;
  std::cout << hamiltonian->get_summary() << std::endl;
  // end-cell-construct
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-list-implementations
  auto names = HamiltonianConstructorFactory::available();
  for (const auto& name : names) {
    std::cout << name << std::endl;
  }
  // end-cell-list-implementations
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-cholesky
  // Reuse the SCF orbitals; Cholesky needs no auxiliary basis.
  auto cholesky_constructor =
      HamiltonianConstructorFactory::create("qdk_cholesky");
  cholesky_constructor->settings().set("cholesky_tolerance", 1e-8);
  auto cholesky_hamiltonian = cholesky_constructor->run(orbitals);
  std::cout << "Cholesky container: "
            << cholesky_hamiltonian->get_container_type() << std::endl;
  // end-cell-cholesky
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-density-fitted
  // SCF supplies MOs in the primary basis; RIFit is used for the Hamiltonian.
  auto df_basis = BasisSet::from_basis_name("cc-pvdz", structure);
  auto df_scf_solver = ScfSolverFactory::create("qdk");
  auto [E_df_scf, df_wavefunction] =
      df_scf_solver->run(structure, 0, 1, df_basis);
  auto df_orbitals = df_wavefunction->get_orbitals();

  auto rifit = AuxiliaryBasis::from_basis_name("cc-pvdz-rifit", structure);
  auto auxiliary_bases = std::make_shared<AuxiliaryBasisCollection>(
      AuxiliaryBasisCollection::Map{{AuxiliaryBasisRole::RIFit, rifit}});
  auto df_constructor =
      HamiltonianConstructorFactory::create("qdk_density_fitted_hamiltonian");
  auto df_hamiltonian = df_constructor->run(df_orbitals, auxiliary_bases);
  std::cout << "Density-fitted container: "
            << df_hamiltonian->get_container_type() << std::endl;
  // end-cell-density-fitted
  // --------------------------------------------------------------------------------------------

  return 0;
}

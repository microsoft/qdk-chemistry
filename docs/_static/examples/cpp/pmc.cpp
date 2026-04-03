// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Projected Multi-Configuration Calculator usage examples.
#include <iostream>
#include <qdk/chemistry.hpp>
#include <string>
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

int main() {
  // --------------------------------------------------------------------------------------------
  // start-cell-create
  // Create a MACIS PMC calculator instance
  auto pmc_calculator =
      ProjectedMultiConfigurationCalculatorFactory::create("macis_pmc");
  // end-cell-create
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-configure
  // Set the convergence threshold for the CI iterations
  pmc_calculator->settings().set("ci_residual_tolerance", 1.0e-6);
  pmc_calculator->settings().set("davidson_res_tol", 1.0e-8);
  // end-cell-configure
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-run
  // Create a structure (H2 molecule)
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
  std::vector<std::string> symbols = {"H", "H"};
  std::shared_ptr<Structure> structure =
      std::make_shared<Structure>(coords, symbols);

  // Run SCF to get orbitals
  auto scf_solver = ScfSolverFactory::create();
  auto [E_scf, wfn] = scf_solver->run(structure, 0, 1, "sto-3g");

  // Build Hamiltonian from orbitals
  auto ham_constructor = HamiltonianConstructorFactory::create();
  auto hamiltonian = ham_constructor->run(wfn->get_orbitals());

  // Define configurations
  std::vector<Configuration> configurations = {
      Configuration("20"),  // Ground state (both electrons in lowest orbital)
      Configuration("02"),  // Excited state (both electrons in higher orbital)
  };

  // Run the PMC calculation
  auto [E_pmc, pmc_wavefunction] =
      pmc_calculator->run(hamiltonian, configurations);

  std::cout << "SCF Energy: " << E_scf << " Hartree" << std::endl;
  std::cout << "PMC Energy: " << E_pmc << " Hartree" << std::endl;
  // end-cell-run
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-list-implementations
  auto names = ProjectedMultiConfigurationCalculatorFactory::available();
  for (const auto& name : names) {
    std::cout << name << std::endl;
  }
  // end-cell-list-implementations
  // --------------------------------------------------------------------------------------------
  return 0;
}

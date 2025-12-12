// Localizer usage examples.

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-create
#include <iostream>
#include <qdk/chemistry.hpp>
#include <string>
using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// Create a Pipek-Mezey localizer using the factory
auto localizer = LocalizerFactory::create("qdk_pipek_mezey");
// end-cell-create
// --------------------------------------------------------------------------------------------

int main() {
  // --------------------------------------------------------------------------------------------
  // start-cell-configure
  // Configure settings for localizer
  localizer->settings().set("tolerance", 1.0e-6);
  localizer->settings().set("max_iterations", 100);
  std::cout << localizer->settings().keys() << std::endl;

  // end-cell-configure
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-localize
  // Create H2O molecule
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0},
                                         {0.0, 1.43052268, 1.10926924},
                                         {0.0, -1.43052268, 1.10926924}};
  std::vector<std::string> symbols = {"O", "H", "H"};
  Structure structure(coords, symbols);

  // Obtain orbitals from SCF calculation
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("basis_set", "sto-3g");
  auto [E_scf, wavefunction] = scf_solver->run(structure, 0, 1);

  // Specify which orbitals to localize
  // For restricted calculations, alpha and beta orbitals are identical
  std::vector<size_t> loc_indices = {0, 1, 2, 3};

  // Localize the specified orbitals
  auto localized_wfn = localizer->run(wavefunction, loc_indices, loc_indices);
  auto localized_orbitals = localized_wfn->get_orbitals();

  // Print summary
  std::cout << localizer->get_summary() << std::endl;
  // end-cell-localize
  // --------------------------------------------------------------------------------------------
  return 0;
}

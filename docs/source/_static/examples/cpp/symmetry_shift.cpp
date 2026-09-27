// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Symmetry shift (fermionic low-rank BLISS) usage examples.
// --------------------------------------------------------------------------------------------
// start-cell-create
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <qdk/chemistry.hpp>

using namespace qdk::chemistry::algorithms;
using namespace qdk::chemistry::data;

// List available symmetry shifter implementations
auto shifters = SymmetryShifterFactory::available();

// Create the fermionic low-rank BLISS shifter
auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
// end-cell-create
// --------------------------------------------------------------------------------------------

int main() {
  for (const auto& name : shifters) {
    std::cout << name << std::endl;
  }

  // The shifter itself has no tunable settings; truncation is a property of
  // the double factorization it consumes.
  for (const auto& key : shifter->settings().keys()) {
    std::cout << key << std::endl;
  }

  // --------------------------------------------------------------------------------------------
  // docs:xyz ../data/water.structure.xyz
  // start-cell-factorize
  // Load H2O molecule from inline XYZ file
  auto structure = Structure::from_xyz(R"(3
Water molecule
O    0.000000    0.000000    0.000000
H    0.758602    0.000000    0.504284
H   -0.758602    0.000000    0.504284
)");

  // Obtain orbitals from SCF and build the molecular Hamiltonian
  auto scf_solver = ScfSolverFactory::create();
  auto [E_scf, wavefunction] = scf_solver->run(structure, 0, 1, "sto-3g");
  auto hamiltonian = HamiltonianConstructorFactory::create()->run(
      wavefunction->get_orbitals());

  // The shifter consumes a double-factorized Hamiltonian
  auto factorizer =
      HamiltonianFactorizationFactory::create("double_factorization");
  factorizer->settings().set("truncation_threshold", 1.0e-8);
  auto factorized = factorizer->run(hamiltonian);

  const auto& container =
      factorized->get_container<FactorizedHamiltonianContainer>();
  std::cout << "Orbitals: " << container.get_num_orbitals()
            << ", ranks: " << container.get_num_ranks() << std::endl;
  // end-cell-factorize
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-shift
  // Target electron-number sector: neutral water has 10 electrons
  const unsigned int n_alpha = 5;
  const unsigned int n_beta = 5;

  const double lambda_before = container.get_lambda();
  auto shifted = shifter->run(factorized, n_alpha, n_beta);
  const double lambda_after =
      shifted->get_container<FactorizedHamiltonianContainer>().get_lambda();

  std::cout << std::fixed << std::setprecision(8)
            << "lambda before shift: " << lambda_before << "\n"
            << "lambda after  shift: " << lambda_after << "\n"
            << "reduction          : "
            << 100.0 * (1.0 - lambda_after / lambda_before) << "%" << std::endl;
  // end-cell-shift
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-inspect-shift
  // Inspect the (mu1, mu2, xi) parameters that the last run applied
  if (auto shift = shifter->last_shift()) {
    std::cout << "mu1: " << shift->mu1 << "\n"
              << "mu2: " << shift->mu2 << "\n"
              << "xi: " << shift->xi.rows() << "x" << shift->xi.cols()
              << std::endl;
  }
  // end-cell-inspect-shift
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-persist
  // The shifted Hamiltonian stays factorized, so it round-trips without
  // re-factorization and can be block-encoded directly.
  shifted->to_hdf5_file("water_shifted.hamiltonian.h5");

  auto reloaded = Hamiltonian::from_hdf5_file("water_shifted.hamiltonian.h5");
  std::cout
      << "lambda on reload: "
      << reloaded->get_container<FactorizedHamiltonianContainer>().get_lambda()
      << std::endl;
  std::remove("water_shifted.hamiltonian.h5");
  // end-cell-persist
  // --------------------------------------------------------------------------------------------
  return 0;
}

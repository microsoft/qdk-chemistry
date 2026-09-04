// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Orbitals usage examples.
// --------------------------------------------------------------------------------------------
// docs:xyz ../data/h2.structure.xyz
// start-cell-create
#include <iostream>
#include <qdk/chemistry.hpp>
#include <string>
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

int main() {
  // Obtain orbitals from a SCF calculation
  // Load H2 molecule from inline XYZ file
  auto structure = Structure::from_xyz(R"(2
H2 molecule
H    0.000000    0.000000    0.000000
H    0.000000    0.000000    0.740848
)");

  // Obtain orbitals from a SCF calculation
  auto scf_solver = ScfSolverFactory::create();
  auto [E_scf, wfn] = scf_solver->run(structure, 0, 1, "sto-3g");
  std::shared_ptr<Orbitals> orbitals = wfn.get_orbitals();

  // end-cell-create
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-model-orbitals-create
  // Set basis set size
  size_t basis_size = 6;

  // Set active orbitals
  std::vector<size_t> alpha_active = {1, 2};
  std::vector<size_t> beta_active = {2, 3, 4};
  std::vector<size_t> alpha_inactive = {0, 3, 4, 5};
  std::vector<size_t> beta_inactive = {0, 1, 5};

  ModelOrbitals model_orbitals(
      basis_size, std::make_tuple(alpha_active, beta_active, alpha_inactive,
                                  beta_inactive));

  // We can then pass this object to a custom Hamiltonian constructor
  // end-cell-model-orbitals-create
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-access
  // Access orbital coefficients as a symmetry-blocked tensor, then per spin
  // block
  auto coefficients = orbitals->coefficients();
  const auto& coeffs_alpha =
      coefficients->block({axes::alpha(), axes::alpha()});
  const auto& coeffs_beta = coefficients->block({axes::beta(), axes::beta()});

  // Access orbital energies as a symmetry-blocked tensor, then per spin block
  auto orbital_energies = orbitals->energies();
  const auto& energies_alpha = orbital_energies->block({axes::alpha()});
  const auto& energies_beta = orbital_energies->block({axes::beta()});

  // Get active space indices per spin channel
  auto active_indices_alpha =
      orbitals->active_indices()->indices(SymmetryLabel({axes::alpha()}));
  auto active_indices_beta =
      orbitals->active_indices()->indices(SymmetryLabel({axes::beta()}));

  // Access atomic orbital overlap matrix (returns const Eigen::MatrixXd&)
  const auto& ao_overlap = orbitals->get_overlap_matrix();

  // Access basis set information (returns const BasisSet&)
  const auto& basis_set = orbitals->get_basis_set();

  // Check calculation type
  bool is_restricted = orbitals->is_restricted();

  // Get size information
  size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();
  size_t num_atomic_orbitals = orbitals->get_num_atomic_orbitals();

  std::string summary = orbitals->get_summary();
  std::cout << summary << std::endl;

  // end-cell-access
  // --------------------------------------------------------------------------------------------
  return 0;
}

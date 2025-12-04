// Orbitals usage examples.

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
using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

int main() {
  // Obtain orbitals from an SCF calculation
  // Create H2 molecule
  std::vector<Eigen::Vector3d> coords = {{0.0, 0.0, 0.0}, {0.0, 0.0, 1.4}};
  std::vector<std::string> symbols = {"H", "H"};
  Structure structure(coords, symbols);

  // Obtain orbitals from an SCF calculation
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("basis_set", "sto-3g");
  auto [E_scf, wfn] = scf_solver->run(structure, 0, 1);
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
  // Access orbital coefficients (returns std::pair<const Eigen::MatrixXd&,
  // const Eigen::MatrixXd&>)
  auto [coeffs_alpha, coeffs_beta] = orbitals->get_coefficients();

  // Access orbital energies (returns std::pair<const Eigen::VectorXd&, const
  // Eigen::VectorXd&>)
  auto [energies_alpha, energies_beta] = orbitals->get_energies();

  // Get active space indices
  auto [active_indices_alpha, active_indices_beta] =
      orbitals->get_active_space_indices();

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

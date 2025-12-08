// Wavefunction container examples

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::data;

// Helper function to create orbitals - will be used in the different
// constructors for wavefunction containers
std::shared_ptr<Orbitals> make_minimal_orbitals() {
  // Create a minimal STO-1G basis
  std::vector<Shell> shells;

  // Gaussian exponents and coefficients
  Eigen::VectorXd exps(1);
  exps << 1.0;
  Eigen::VectorXd coefs(1);
  coefs << 1.0;

  // Atom 0
  Shell h1_s_shell(0, OrbitalType::S, exps, coefs);
  shells.push_back(h1_s_shell);

  // Atom 1
  Shell h2_s_shell(1, OrbitalType::S, exps, coefs);
  shells.push_back(h2_s_shell);

  auto basis_set = std::make_shared<BasisSet>("STO-1G_H2", shells);

  // Create bonding and antibonding MOs from AOs
  Eigen::MatrixXd coefficients(2, 2);
  coefficients << 0.7071, 0.7071, 0.7071, -0.7071;

  Eigen::VectorXd energies(2);
  energies << -1.0, 0.5;  // HOMO, LUMO

  // Orbital constructor requires coefficients, energies, optionally AO overlap,
  // and basis set
  std::shared_ptr<Orbitals> orbitals = std::make_shared<Orbitals>(
      coefficients, energies, std::nullopt, basis_set);

  return orbitals;
}

int main() {
  // --------------------------------------------------------------------------------------------
  // start-cell-create-slater
  // Use helper function to get orbitals
  std::shared_ptr<Orbitals> orbitals = make_minimal_orbitals();
  // Create a simple Slater determinant wavefunction for H2 ground state
  // 2 electrons in bonding sigma orbital
  Configuration det("20");

  // Constructor takes single determinant and orbitals as input
  auto sd_container =
      std::make_unique<SlaterDeterminantContainer>(det, orbitals);
  Wavefunction sd_wavefunction(std::move(sd_container));
  // end-cell-create-slater
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-create-cas
  // Create a CAS wavefunction for H2
  // CAS(2,2) = 2 electrons in 2 MOs (bonding and antibonding)
  // All possible configurations:
  std::vector<Configuration> cas_dets = {
      Configuration("20"),  // Both electrons in bonding (ground state)
      Configuration("ud"),  // Alpha in bonding, beta in antibonding
      Configuration("du"),  // Beta in bonding, alpha in antibonding
      Configuration("02")   // Both electrons in antibonding
  };

  // Coefficients
  Eigen::VectorXd cas_coeffs(4);
  cas_coeffs << 0.95, 0.15, 0.15, 0.05;  // Normalized later by the container

  // Create a CAS wavefunction : requires all coefficients and determinants, as
  // well as orbitals, in constructor
  auto cas_container = std::make_unique<CasWavefunctionContainer>(
      cas_coeffs, cas_dets, orbitals);
  Wavefunction cas_wavefunction(std::move(cas_container));
  // end-cell-create-cas
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-create-sci
  // Create an SCI wavefunction for H2
  // SCI selects only the most important configurations/determinants from the
  // full space
  std::vector<Configuration> sci_dets = {
      Configuration("20"),  // Ground state
      Configuration("ud"),  // Mixed state
      Configuration("du")   // Mixed state
  };

  // Coefficients for selected determinants
  Eigen::VectorXd sci_coeffs(3);
  sci_coeffs << 0.96, 0.15, 0.15;

  // Create a SCI wavefunction: requires selected coefficients and determinants,
  // as well as orbitals, in constructor
  auto sci_container = std::make_unique<SciWavefunctionContainer>(
      sci_coeffs, sci_dets, orbitals);
  Wavefunction sci_wavefunction(std::move(sci_container));
  // end-cell-create-sci
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-access-data
  // Access coefficient(s) and determinant(s) - SD has only one
  auto coeffs = sd_wavefunction.get_coefficients();
  auto dets = sd_wavefunction.get_active_determinants();

  // Get orbital information
  auto orbitals_ref = sd_wavefunction.get_orbitals();

  // Get electron counts
  auto [n_alpha, n_beta] = sd_wavefunction.get_total_num_electrons();

  // Get RDMs
  auto [rdm1_aa, rdm1_bb] = sd_wavefunction.get_active_one_rdm_spin_dependent();
  auto rdm1_total = sd_wavefunction.get_active_one_rdm_spin_traced();
  auto [rdm2_aa, rdm2_aabb, rdm2_bbbb] =
      sd_wavefunction.get_active_two_rdm_spin_dependent();
  auto rdm2_total = sd_wavefunction.get_active_two_rdm_spin_traced();

  // Get single orbital entropies
  auto entropies = sd_wavefunction.get_single_orbital_entropies();
  // end-cell-access-data
  // --------------------------------------------------------------------------------------------
  return 0;
}

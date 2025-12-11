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

// Helper function to create a minimal Hamiltonian for H2
// This is needed for MP2 container examples
std::shared_ptr<Hamiltonian> make_minimal_hamiltonian(
    std::shared_ptr<Orbitals> orbitals) {
  // Create minimal one- and two-electron integrals for H2
  Eigen::MatrixXd h_core(2, 2);
  h_core << -1.5, -0.8, -0.8, 0.5;  // Core Hamiltonian

  // Two-electron integrals in MO basis, stored as flattened vector
  // These are stored like i*norb*norb*norb + j*norb*norb + k*norb + l
  // In other words, if we want to access an integral element in the vector,
  // (ij|kl), we can access using this indexing.

  // For H2: norb=2, so we need 2^4=16 elements
  Eigen::VectorXd eri = Eigen::VectorXd::Zero(16);

  // Set some representative values for H2 two-electron integrals
  // Format: (ij|kl) in physicist notation
  eri[0] = 1.0;   // (00|00) - index 0*8 + 0*4 + 0*2 + 0 = 0
  eri[5] = 0.6;   // (01|01) - index 0*8 + 1*4 + 0*2 + 1 = 5
  eri[10] = 0.6;  // (10|10) - index 1*8 + 0*4 + 1*2 + 0 = 10
  eri[15] = 0.8;  // (11|11) - index 1*8 + 1*4 + 1*2 + 1 = 15
  eri[3] = 0.4;   // (00|11) - index 0*8 + 0*4 + 1*2 + 1 = 3
  eri[12] = 0.4;  // (11|00) - index 1*8 + 1*4 + 0*2 + 0 = 12

  // Core energy (nuclear repulsion + core electron contributions)
  double core_energy = 0.0;

  // Inactive Fock matrix (empty for minimal example)
  Eigen::MatrixXd inactive_fock = Eigen::MatrixXd::Zero(2, 2);

  // Create Hamiltonian
  return std::make_shared<Hamiltonian>(h_core, eri, orbitals, core_energy,
                                       inactive_fock);
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
  // start-cell-create-mp2
  // Create an MP2 wavefunction for H2
  // MP2 uses a reference wavefunction and Hamiltonian to compute amplitudes on
  // demand

  // Use the Slater determinant as reference
  auto orbitals_mp2 = make_minimal_orbitals();
  auto hamiltonian = make_minimal_hamiltonian(orbitals_mp2);
  Configuration ref_det("20");
  auto sd_container_mp2 =
      std::make_unique<SlaterDeterminantContainer>(ref_det, orbitals_mp2);
  auto ref_wavefunction =
      std::make_shared<Wavefunction>(std::move(sd_container_mp2));

  // Create MP2 container: requires Hamiltonian and reference wavefunction
  // Amplitudes are computed lazily when first requested
  auto mp2_container =
      std::make_unique<MP2Container>(hamiltonian, ref_wavefunction, "mp");
  Wavefunction mp2_wavefunction(std::move(mp2_container));
  // end-cell-create-mp2
  // --------------------------------------------------------------------------------------------

  // --------------------------------------------------------------------------------------------
  // start-cell-create-cc
  // Create a coupled cluster wavefunction for H2
  // CC uses a reference wavefunction and pre-computed amplitudes

  // Use the Slater determinant as reference
  auto orbitals_cc = make_minimal_orbitals();
  Configuration ref_det_cc("20");
  auto sd_container_cc =
      std::make_unique<SlaterDeterminantContainer>(ref_det_cc, orbitals_cc);
  auto ref_wavefunction_cc =
      std::make_shared<Wavefunction>(std::move(sd_container_cc));

  // Create example T1 and T2 amplitudes
  // T1: occupied-virtual excitations (1 occ × 1 virt = 1 element for H2)
  Eigen::VectorXd t1_amplitudes(1);
  t1_amplitudes << 0.05;

  // T2: occupied-occupied to virtual-virtual excitations
  // (1 occ pair × 1 virt pair = 1 element for H2)
  Eigen::VectorXd t2_amplitudes(1);
  t2_amplitudes << 0.15;

  // Create CC container: requires reference wavefunction, orbitals, and
  // amplitudes
  auto cc_container = std::make_unique<CoupledClusterContainer>(
      orbitals_cc, ref_wavefunction_cc, t1_amplitudes, t2_amplitudes);
  Wavefunction cc_wavefunction(std::move(cc_container));
  // end-cell-create-cc
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

  // --------------------------------------------------------------------------------------------
  // start-cell-access-amplitudes
  // Access T1 and T2 amplitudes from MP2 and CC containers

  // MP2
  // Get the container back from wfn
  const auto& mp2_container_ref =
      mp2_wavefunction.get_container<MP2Container>();
  // Amplitudes are lazily evaluated on first call then cached
  auto [t2_abab_mp2, t2_aaaa_mp2, t2_bbbb_mp2] =
      mp2_container_ref->get_t2_amplitudes();

  // CC
  const auto& cc_container_ref =
      cc_wavefunction.get_container<CoupledClusterContainer>();
  // Amplitudes are stored already from construction
  auto [t1_aa, t1_bb] = cc_container_ref->get_t1_amplitudes();
  auto [t2_abab_cc, t2_aaaa_cc, t2_bbbb_cc] =
      cc_container_ref->get_t2_amplitudes();
  // end-cell-access-amplitudes
  // --------------------------------------------------------------------------------------------

  return 0;
}

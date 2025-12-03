// Wavefunction container examples

// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
// --------------------------------------------------------------------------------------------

#include <qdk/chemistry.hpp>
using namespace qdk::chemistry::data;

// --------------------------------------------------------------------------------------------
// start-cell-create-slater
// Create basis set
std::vector<Shell> shells;
int atom_index = 0;
int functions_created = 0;
int num_atomic_orbitals = 3;

// Create shells to reach 3 AOs
while (functions_created < num_atomic_orbitals) {
  int remaining = num_atomic_orbitals - functions_created;

  if (remaining >= 3) {
    // Add a P shell (3 functions: Px, Py, Pz)
    Eigen::VectorXd exps(2);
    exps << 1.0, 0.5;
    Eigen::VectorXd coefs(2);
    coefs << 0.6, 0.4;
    Shell shell(atom_index, OrbitalType::P, exps, coefs);
    shells.push_back(shell);
    functions_created += 3;
  } else {
    // Add S shells for remaining functions (1 function each)
    for (int i = 0; i < remaining; ++i) {
      Eigen::VectorXd exps(1);
      exps << 1.0;
      Eigen::VectorXd coefs(1);
      coefs << 1.0;
      Shell shell(atom_index, OrbitalType::S, exps, coefs);
      shells.push_back(shell);
      functions_created += 1;
    }
  }
}

auto basis_set = std::shared_ptr<BasisSet>("dummy", shells);

// Create orbitals
Eigen::MatrixXd coefficients = Eigen::MatrixXd::Identity(3, 3);
Eigen::VectorXd energies(3);
energies << -1.0, -0.5, 0.2;
auto orbitals =
    std::make_shared<Orbitals>(coefficients, energies, std::nullopt, basis_set);

// Create a simple Slater determinant wavefunction
Configuration det("20");
auto sd_container = std::make_unique<SlaterDeterminantContainer>(det, orbitals);
Wavefunction sd_wavefunction(std::move(sd_container));
// end-cell-create-slater
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-create-cas
// Create a CAS wavefunction with multiple determinants
std::vector<Configuration> cas_dets = {Configuration("20"),
                                       Configuration("ud")};
Eigen::VectorXd cas_coeffs(2);
cas_coeffs << 0.9, 0.436;
auto cas_container =
    std::make_unique<CasWavefunctionContainer>(cas_coeffs, cas_dets, orbitals);
Wavefunction cas_wavefunction(std::move(cas_container));
// end-cell-create-cas
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-create-sci
// Create an SCI wavefunction with selected determinants
std::vector<Configuration> sci_dets = {Configuration("20"), Configuration("ud"),
                                       Configuration("02")};
Eigen::VectorXd sci_coeffs(3);
sci_coeffs << 0.85, 0.4, 0.3;
auto sci_container =
    std::make_unique<SciWavefunctionContainer>(sci_coeffs, sci_dets, orbitals);
Wavefunction sci_wavefunction(std::move(sci_container));
// end-cell-create-sci
// --------------------------------------------------------------------------------------------

// --------------------------------------------------------------------------------------------
// start-cell-access-data
// Access basic wavefunction data
auto coeffs = sd_wavefunction.get_coefficients();
auto dets = sd_wavefunction.get_active_determinants();

// Get orbital information
auto orbitals_ref = sd_wavefunction.get_orbitals();

// Get electron counts
auto [n_alpha, n_beta] = sd_wavefunction.get_total_num_electrons();

// Check availability of RDMs
bool has_1rdm_spin_dep = sd_wavefunction.has_one_rdm_spin_dependent();
bool has_1rdm_spin_traced = sd_wavefunction.has_one_rdm_spin_traced();
bool has_2rdm_spin_dep = sd_wavefunction.has_two_rdm_spin_dependent();
bool has_2rdm_spin_traced = sd_wavefunction.has_two_rdm_spin_traced();

// Access if available
if (has_1rdm_spin_dep) {
  auto [rdm1_aa, rdm1_bb] = sd_wavefunction.get_active_one_rdm_spin_dependent();
}
if (has_1rdm_spin_traced) {
  auto rdm1_total = sd_wavefunction.get_active_one_rdm_spin_traced();
}
if (has_2rdm_spin_dep) {
  auto [rdm2_aa, rdm2_aabb, rdm2_bbbb] =
      sd_wavefunction.get_active_two_rdm_spin_dependent();
}
if (has_2rdm_spin_traced) {
  auto rdm2_total = sd_wavefunction.get_active_two_rdm_spin_traced();
}

// Get single orbital entropies
auto entropies = sd_wavefunction.get_single_orbital_entropies();
// end-cell-access-data
// --------------------------------------------------------------------------------------------

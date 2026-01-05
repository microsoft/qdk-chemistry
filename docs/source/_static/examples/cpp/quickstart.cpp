// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/**
 * @file quickstart.cpp
 * @brief Minimal end-to-end example demonstrating quantum chemistry workflow
 *
 * This example mirrors the Python quickstart.py, demonstrating:
 * 1. Creating molecular structures
 * 2. Performing SCF calculations
 * 3. Active space selection
 * 4. Hamiltonian construction
 * 5. CASCI calculation
 * 6. Projected multi-configuration (PMC) calculation with top determinants
 */

#include <algorithm>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <qdk/chemistry.hpp>
#include <vector>

using namespace qdk::chemistry;

// -----------------------------------------------------------------------------
// start-cell-wfn-fn-select-configs
/**
 * @brief Get top N determinants from a wavefunction sorted by coefficient
 * magnitude
 * @param wfn The wavefunction to extract determinants from
 * @param max_determinants Maximum number of determinants to return
 * @return Vector of top configurations
 */
std::vector<data::Configuration> get_top_determinants(
    std::shared_ptr<data::Wavefunction> wfn, size_t max_determinants) {
  const auto& determinants = wfn->get_active_determinants();
  const auto& coeffs = wfn->get_coefficients();

  // Extract coefficients as doubles (handle real case)
  std::vector<double> coeff_values;
  const auto& real_coeffs = std::get<Eigen::VectorXd>(coeffs);
  coeff_values.resize(real_coeffs.size());
  for (int i = 0; i < real_coeffs.size(); ++i) {
    coeff_values[i] = std::abs(real_coeffs[i]);
  }

  // Create indices and sort by coefficient magnitude
  std::vector<size_t> indices(determinants.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(),
            [&coeff_values](size_t i1, size_t i2) {
              return coeff_values[i1] > coeff_values[i2];
            });

  // Extract top determinants
  std::vector<data::Configuration> top_dets;
  size_t n = std::min(max_determinants, determinants.size());
  top_dets.reserve(n);
  for (size_t i = 0; i < n; ++i) {
    top_dets.push_back(determinants[indices[i]]);
  }

  return top_dets;
}
// end-cell-wfn-fn-select-configs
// -----------------------------------------------------------------------------

int main() {
  // ---------------------------------------------------------------------------
  // start-cell-structure
  // Load para-benzyne structure from XYZ file
  auto structure = std::make_shared<data::Structure>(
      data::Structure::from_xyz_file("../data/para_benzyne.structure.xyz"));

  std::cout << "Created structure with " << structure->get_num_atoms()
            << " atoms" << std::endl;
  std::cout << "Elements: ";
  for (const auto& elem : structure->get_elements()) {
    std::cout << data::Structure::element_to_symbol(elem) << " ";
  }
  std::cout << std::endl;
  // end-cell-structure
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // start-cell-scf
  // Perform a SCF calculation
  auto scf_solver = algorithms::ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf_solver->run(structure, 0, 1, "cc-pvdz");
  std::cout << "SCF energy is " << E_hf << " Hartree" << std::endl;

  // Display a summary of the molecular orbitals
  std::cout << "SCF Orbitals:\n"
            << wfn_hf->get_orbitals()->get_summary() << std::endl;
  // end-cell-scf
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // start-cell-active-space
  // Select active space (6 electrons in 6 orbitals)
  auto active_space_selector =
      algorithms::ActiveSpaceSelectorFactory::create("qdk_valence");
  active_space_selector->settings().set("num_active_electrons", 6);
  active_space_selector->settings().set("num_active_orbitals", 6);

  auto active_wfn = active_space_selector->run(wfn_hf);
  auto active_orbitals = active_wfn->get_orbitals();

  // Print a summary of the active space orbitals
  std::cout << "Active Space Orbitals:\n"
            << active_orbitals->get_summary() << std::endl;
  // end-cell-active-space
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // start-cell-hamiltonian-constructor
  // Construct Hamiltonian in the active space
  auto hamiltonian_constructor =
      algorithms::HamiltonianConstructorFactory::create();
  auto hamiltonian = hamiltonian_constructor->run(active_orbitals);
  std::cout << "Active Space Hamiltonian:\n"
            << hamiltonian->get_summary() << std::endl;
  // end-cell-hamiltonian-constructor
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // start-cell-mc-compute
  // Perform CASCI calculation
  auto mc = algorithms::MultiConfigurationCalculatorFactory::create();
  auto [E_cas, wfn_cas] = mc->run(hamiltonian, 3, 3);
  std::cout << "CASCI energy is " << E_cas
            << " Hartree, and the electron correlation energy is "
            << (E_cas - E_hf) << " Hartree" << std::endl;
  // end-cell-mc-compute
  // ---------------------------------------------------------------------------

  // ---------------------------------------------------------------------------
  // start-cell-wfn-select-configs
  // Get top 2 determinants from the CASCI wavefunction
  auto top_configurations = get_top_determinants(wfn_cas, 2);

  // Perform PMC calculation with selected configurations
  auto pmc = algorithms::ProjectedMultiConfigurationCalculatorFactory::create();
  auto [E_pmc, wfn_pmc] = pmc->run(hamiltonian, top_configurations);
  std::cout << "Reference energy for top 2 determinants is " << E_pmc
            << " Hartree" << std::endl;
  // end-cell-wfn-select-configs
  // ---------------------------------------------------------------------------

  return 0;
}

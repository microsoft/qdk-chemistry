// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

/**
 * @file simple_e2e.cpp
 * @brief End-to-end example demonstrating basic quantum chemistry calculations
 * using QDK
 *
 * This example demonstrates a complete workflow for molecular electronic
 * structure calculations:
 * 1. Creating or loading molecular structures
 * 2. Performing Hartree-Fock (HF) self-consistent field (SCF) calculations
 * 3. Orbital localization
 * 4. Hamiltonian construction in the molecular orbital basis
 * 5. Post-HF calculations: MP2 (Møller-Plesset perturbation theory) and CASCI
 * (Complete Active Space Configuration Interaction)
 *
 * Usage:
 *   ./simple_e2e                     # Uses default water / STO-3G
 *   ./simple_e2e molecule.xyz        # Loads XYZ file (STO-3G)
 *   ./simple_e2e molecule.xyz 6-31g  # Loads XYZ file with 6-31G basis set
 *
 * The program outputs:
 * - Input structure summary
 * - Hartree-Fock energy and orbital information
 * - MP2 correlation energy
 * - CASCI energy and wavefunction information
 *
 * This example serves as a starting point for understanding QDK's API and
 * typical workflow for quantum chemistry calculations.
 */

#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/structure.hpp>

// TODO (NAB):  this is one of our only end-to-end examples in C++.  We should
// make it more pedagogical, explaining the role of each step, writing out
// acronyms, making the variable names clearer, etc. Workitem 41302

int main(int argc, char** argv) {
  std::shared_ptr<qdk::chemistry::data::Structure> structure;
  // Default basis set to STO-3G
  std::string basis_set = "sto-3g";

  if (argc > 1) {
    // Read structure from file
    structure = qdk::chemistry::data::Structure::from_xyz_file(argv[1]);
    if (argc > 2) {
      basis_set = argv[2];
    }
  } else {
    throw std::runtime_error(
        "Missing required input! The correct syntax for this example is:\n  "
        "simple_e2e xyz_file [basis_name]");
  }

  std::cout << "Input:\n" << structure->get_summary();

  // Create an SCF solver instance
  auto scf_solver = qdk::chemistry::algorithms::ScfSolverFactory::create();
  {
    auto& settings = scf_solver->settings();
    settings.set("basis_set", basis_set);
  }

  // Perform SCF Optimization
  auto [E_hf, wfn_hf] = scf_solver->run(structure, 0, 1);
  auto orbitals_hf = wfn_hf->get_orbitals();
  auto [n_alpha, n_beta] = wfn_hf->get_total_num_electrons();

  // Print summary
  std::cout << "E(HF) = " << std::scientific << std::setprecision(8) << E_hf
            << " Eh" << std::endl;

  // Localize orbitals
  {
    auto localizer = qdk::chemistry::algorithms::LocalizerFactory::create();
    // TODO (NAB):  describe localizer: 41291/41302
    auto localized_hf_wfn = localizer->run(wfn_hf, n_alpha, n_beta);
  }

  // TODO (NAB):  we need an end-to-end C++ example for active space selection:
  // 41302

  // Compute the Hamiltonian
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(orbitals_hf);

  // Compute MP2 to show how to access the MO integrals
  {
    auto [eps_a, eps_b] = orbitals_hf->get_energies();
    const auto num_molecular_orbitals =
        orbitals_hf->get_num_molecular_orbitals();
    const size_t num_occupied_orbitals = n_alpha;
    const size_t num_virtual_orbitals =
        num_molecular_orbitals - num_occupied_orbitals;

    const auto& moeri = ham->get_two_body_integrals();
    double EMP2 = 0.0;
    for (size_t i = 0; i < num_occupied_orbitals; ++i)
      for (size_t j = 0; j < num_occupied_orbitals; ++j)
        for (size_t a = 0; a < num_virtual_orbitals; ++a)
          for (size_t b = 0; b < num_virtual_orbitals; ++b) {
            const auto eri_1 =
                moeri[i * num_molecular_orbitals * num_molecular_orbitals *
                          num_molecular_orbitals +
                      (a + num_occupied_orbitals) * num_molecular_orbitals *
                          num_molecular_orbitals +
                      j * num_molecular_orbitals + (b + num_occupied_orbitals)];
            const auto eri_2 =
                moeri[i * num_molecular_orbitals * num_molecular_orbitals *
                          num_molecular_orbitals +
                      (b + num_occupied_orbitals) * num_molecular_orbitals *
                          num_molecular_orbitals +
                      j * num_molecular_orbitals + (a + num_occupied_orbitals)];
            EMP2 += eri_1 * (2 * eri_1 - eri_2) /
                    (eps_a[i] + eps_a[j] - eps_a[a + num_occupied_orbitals] -
                     eps_a[b + num_occupied_orbitals]);
          }

    std::cout << "E(MP2) = " << E_hf + EMP2 << std::endl;
  }

  // Run CASCI
  auto mc =
      qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory::create();
  auto [E_cas, wfn_cas] = mc->run(ham, n_alpha, n_beta);
  std::cout << "E(CAS) = " << E_cas << " WFN_SZ = " << wfn_cas->size()
            << " WFN_NORM = " << wfn_cas->norm() << std::endl;

  return 0;
}

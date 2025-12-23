/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mp2.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class MP2Test : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(MP2Test, UMP2Energies_CCPVDZ) {
  // Test the UMP2 energies against reference for cc-pvdz
  float pyscf_mp2_corr_cc_pvdz = -0.3509470131940627;

  // o2 structure with 2.3 Bohr bond length
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] =
      scf_factory->run(o2_structure_ptr, 0, 3, "cc-pvdz");
  auto hf_orbitals = hf_wavefunction->get_orbitals();
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto hf_hamiltonian = ham_factory->run(hf_orbitals);
  auto [n_alpha_active, n_beta_active] =
      hf_wavefunction->get_active_num_electrons();

  // Create ansatz from Hamiltonian and wavefunction
  auto ansatz = std::make_shared<Ansatz>(*hf_hamiltonian, *hf_wavefunction);

  auto mp2_calculator =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");

  auto [mp2_total_energy, final_wavefunction] = mp2_calculator->run(ansatz);

  double reference = ansatz->calculate_energy();
  double mp2_corr_energy = mp2_total_energy - reference;

  EXPECT_LT(std::abs(mp2_corr_energy - pyscf_mp2_corr_cc_pvdz),
            testing::mp2_tolerance)
      << "UMP2 correlation energy mismatch (cc-pvdz). Calculated: "
      << mp2_corr_energy << ", Reference: " << pyscf_mp2_corr_cc_pvdz
      << ", Difference: " << (mp2_corr_energy - pyscf_mp2_corr_cc_pvdz);
}

TEST_F(MP2Test, RMP2Energies_CCPVDZ) {
  // Test the RMP2 energies against PySCF reference for singlet O2 with
  // cc-pvdz
  float pyscf_rmp2_corr_cc_pvdz = -0.38428662586339435;

  // O2 structure with 2.3 Bohr bond length
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  // Restricted HF calculation (singlet O2, multiplicity = 1)
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] =
      scf_factory->run(o2_structure_ptr, 0, 1, "cc-pvdz");
  auto hf_orbitals = hf_wavefunction->get_orbitals();

  // Create Hamiltonian
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto hf_hamiltonian = ham_factory->run(hf_orbitals);
  auto [n_alpha_active, n_beta_active] =
      hf_wavefunction->get_active_num_electrons();

  // Verify closed shell
  EXPECT_EQ(n_alpha_active, n_beta_active)
      << "Alpha and beta electrons should be equal for restricted "
         "calculation";

  // Create ansatz from Hamiltonian and wavefunction
  auto ansatz = std::make_shared<Ansatz>(*hf_hamiltonian, *hf_wavefunction);

  // Use MP2 calculator
  auto mp2_calculator =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator");

  // MP2 returns total energy, subtract reference to get correlation energy
  auto [mp2_total_energy, final_wavefunction] = mp2_calculator->run(ansatz);
  double hf_reference_energy = ansatz->calculate_energy();
  double mp2_corr_energy = mp2_total_energy - hf_reference_energy;

  // Verify correlation energy matches PySCF reference
  EXPECT_LT(std::abs(mp2_corr_energy - pyscf_rmp2_corr_cc_pvdz),
            testing::mp2_tolerance)
      << "RMP2 correlation energy mismatch (cc-pvdz). Calculated: "
      << mp2_corr_energy << ", Reference: " << pyscf_rmp2_corr_cc_pvdz
      << ", Difference: " << (mp2_corr_energy - pyscf_rmp2_corr_cc_pvdz);
}

TEST_F(MP2Test, MP2Container) {
  // Test that MP2Container properly computes amplitudes
  // O2 structure with 2.3 Bohr bond length
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  // Restricted HF calculation (singlet O2, multiplicity = 1)
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] =
      scf_factory->run(o2_structure_ptr, 0, 1, "cc-pvdz");
  auto hf_orbitals = hf_wavefunction->get_orbitals();

  // Create Hamiltonian
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto hf_hamiltonian = ham_factory->run(hf_orbitals);

  // Create MP2Container
  auto mp2_container_with_amplitudes =
      std::make_unique<MP2Container>(hf_hamiltonian, hf_wavefunction);

  // Verify Hamiltonian is stored
  EXPECT_NE(mp2_container_with_amplitudes->get_hamiltonian(), nullptr)
      << "MP2Container should store Hamiltonian reference";

  // Lazy evaluation: Amplitudes should not be available initially
  EXPECT_FALSE(mp2_container_with_amplitudes->has_t1_amplitudes())
      << "T1 amplitudes should NOT be computed until requested (lazy "
         "evaluation)";
  EXPECT_FALSE(mp2_container_with_amplitudes->has_t2_amplitudes())
      << "T2 amplitudes should NOT be computed until requested (lazy "
         "evaluation)";

  // Verify we can retrieve the amplitudes (this triggers lazy computation)
  auto [t1_aa, t1_bb] = mp2_container_with_amplitudes->get_t1_amplitudes();
  auto [t2_abab, t2_aaaa, t2_bbbb] =
      mp2_container_with_amplitudes->get_t2_amplitudes();

  // After calling getters, amplitudes should now be available
  EXPECT_TRUE(mp2_container_with_amplitudes->has_t1_amplitudes())
      << "T1 amplitudes should be cached after first access";
  EXPECT_TRUE(mp2_container_with_amplitudes->has_t2_amplitudes())
      << "T2 amplitudes should be cached after first access";

  // Verify T1 amplitudes are zero for MP2
  auto check_t1_zero = [](const MP2Container::VectorVariant& t1) {
    return std::visit([](auto&& vec) { return vec.isZero(1e-10); }, t1);
  };

  EXPECT_TRUE(check_t1_zero(t1_aa))
      << "T1 alpha amplitudes should be zero for MP2";
  EXPECT_TRUE(check_t1_zero(t1_bb))
      << "T1 beta amplitudes should be zero for MP2";

  // Verify T2 amplitudes are non-zero
  auto check_t2_nonzero = [](const MP2Container::VectorVariant& t2) {
    return std::visit([](auto&& vec) { return vec.norm() > 1e-10; }, t2);
  };

  EXPECT_TRUE(check_t2_nonzero(t2_abab))
      << "T2 alpha-beta amplitudes should be non-zero for MP2";
  EXPECT_TRUE(check_t2_nonzero(t2_aaaa))
      << "T2 alpha-alpha amplitudes should be non-zero for MP2";
  EXPECT_TRUE(check_t2_nonzero(t2_bbbb))
      << "T2 beta-beta amplitudes should be non-zero for MP2";
}

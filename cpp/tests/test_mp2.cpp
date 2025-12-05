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
  scf_factory->settings().set("basis_set", "cc-pvdz");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] = scf_factory->run(o2_structure_ptr, 0, 3);
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
  scf_factory->settings().set("basis_set", "cc-pvdz");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] = scf_factory->run(o2_structure_ptr, 0, 1);
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
  scf_factory->settings().set("basis_set", "cc-pvdz");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] = scf_factory->run(o2_structure_ptr, 0, 1);
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

// Test CI coefficients generation from MP2 amplitudes
TEST_F(MP2Test, CICoefficientsGeneration) {
  // O2 structure with 2.3 Bohr bond length
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  // Restricted HF calculation (singlet O2, multiplicity = 1)
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("basis_set", "cc-pvdz");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] = scf_factory->run(o2_structure_ptr, 0, 1);
  auto hf_orbitals = hf_wavefunction->get_orbitals();

  // Create Hamiltonian
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto hf_hamiltonian = ham_factory->run(hf_orbitals);

  // Create MP2Container
  auto mp2_container =
      std::make_unique<MP2Container>(hf_hamiltonian, hf_wavefunction);

  // Test that CI coefficients can be retrieved (lazy evaluation)
  const auto& coefficients = mp2_container->get_coefficients();

  // Verify coefficients are non-empty
  std::visit(
      [](const auto& vec) {
        EXPECT_GT(vec.size(), 0) << "CI coefficients should not be empty";
      },
      coefficients);

  // Test that determinants can be retrieved
  const auto& determinants = mp2_container->get_active_determinants();
  EXPECT_GT(determinants.size(), 0)
      << "Active determinants should not be empty";

  // The number of coefficients should match the number of determinants
  std::visit(
      [&determinants](const auto& vec) {
        EXPECT_EQ(static_cast<size_t>(vec.size()), determinants.size())
            << "Number of coefficients should match number of determinants";
      },
      coefficients);

  // Test size() returns the number of determinants
  EXPECT_EQ(mp2_container->size(), determinants.size())
      << "size() should return the number of determinants";
}

// Test CI expansion consistency for MP2
TEST_F(MP2Test, CIExpansionConsistency) {
  // O2 structure with 2.3 Bohr bond length
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  // Restricted HF calculation (singlet O2, multiplicity = 1)
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("basis_set", "cc-pvdz");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] = scf_factory->run(o2_structure_ptr, 0, 1);
  auto hf_orbitals = hf_wavefunction->get_orbitals();

  // Create Hamiltonian
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto hf_hamiltonian = ham_factory->run(hf_orbitals);

  // Create MP2Container
  auto mp2_container =
      std::make_unique<MP2Container>(hf_hamiltonian, hf_wavefunction);

  const auto& coefficients = mp2_container->get_coefficients();
  const auto& determinants = mp2_container->get_active_determinants();

  // Verify each determinant can be looked up individually
  for (size_t i = 0; i < std::min(determinants.size(), static_cast<size_t>(10));
       ++i) {
    auto coeff = mp2_container->get_coefficient(determinants[i]);
    std::visit(
        [i, &coefficients](const auto& individual_coeff) {
          using T = std::decay_t<decltype(individual_coeff)>;
          std::visit(
              [i, &individual_coeff](const auto& all_coeffs) {
                using U = std::decay_t<decltype(all_coeffs[0])>;
                if constexpr (std::is_same_v<T, U>) {
                  EXPECT_NEAR(std::abs(individual_coeff),
                              std::abs(all_coeffs[i]), testing::wf_tolerance)
                      << "Individual coefficient lookup should match vector at "
                         "index "
                      << i;
                }
              },
              coefficients);
        },
        coeff);
  }
}

// Test that reference determinant is in MP2 expansion with coefficient 1.0
TEST_F(MP2Test, ReferenceInExpansion) {
  // O2 structure with 2.3 Bohr bond length
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  // Restricted HF calculation (singlet O2, multiplicity = 1)
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("basis_set", "cc-pvdz");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] = scf_factory->run(o2_structure_ptr, 0, 1);
  auto hf_orbitals = hf_wavefunction->get_orbitals();

  // Create Hamiltonian
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto hf_hamiltonian = ham_factory->run(hf_orbitals);

  // Create MP2Container
  auto mp2_container =
      std::make_unique<MP2Container>(hf_hamiltonian, hf_wavefunction);

  const auto& determinants = mp2_container->get_active_determinants();

  // The reference determinant should be the first one in the expansion
  const auto& ref_dets = hf_wavefunction->get_total_determinants();
  ASSERT_FALSE(ref_dets.empty())
      << "Reference wavefunction should have determinants";

  // Check that the reference is in the MP2 expansion
  bool found_reference = false;
  for (const auto& det : determinants) {
    for (const auto& ref : ref_dets) {
      if (det.to_string() == ref.to_string()) {
        found_reference = true;
        // The reference should have coefficient 1.0 in MP2
        auto ref_coeff = mp2_container->get_coefficient(det);
        std::visit(
            [](const auto& coeff) {
              EXPECT_NEAR(std::abs(coeff), 0.9586079110259903,
                          testing::wf_tolerance);
            },
            ref_coeff);
        break;
      }
    }
    if (found_reference) break;
  }
  EXPECT_TRUE(found_reference)
      << "Reference determinant should be in the MP2 expansion";
}

// Test RDM availability for MP2 container
// RDMs are lazily computed from CI coefficients in MP2Container using MACIS RDM
// utilities. The CI coefficients are generated from amplitudes, and RDMs are
// computed when requested.
TEST_F(MP2Test, LazyRDMComputationFromAmplitudes) {
  // O2 structure with 2.3 Bohr bond length
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  // Restricted HF calculation (singlet O2, multiplicity = 1)
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("basis_set", "cc-pvdz");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] = scf_factory->run(o2_structure_ptr, 0, 1);
  auto hf_orbitals = hf_wavefunction->get_orbitals();

  // Create Hamiltonian
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto hf_hamiltonian = ham_factory->run(hf_orbitals);

  // Create MP2Container
  auto mp2_container =
      std::make_unique<MP2Container>(hf_hamiltonian, hf_wavefunction);

  // CI coefficients should be available (lazy evaluation from amplitudes)
  const auto& coefficients = mp2_container->get_coefficients();
  std::visit(
      [](const auto& vec) {
        EXPECT_GT(vec.size(), 0) << "CI coefficients should be available";
      },
      coefficients);

  // RDMs SHOULD be available via lazy computation from CI coefficients
  EXPECT_TRUE(mp2_container->has_one_rdm_spin_dependent())
      << "MP2 container should have spin-dependent 1-RDM via lazy computation";
  EXPECT_TRUE(mp2_container->has_one_rdm_spin_traced())
      << "MP2 container should have spin-traced 1-RDM via lazy computation";
  EXPECT_TRUE(mp2_container->has_two_rdm_spin_dependent())
      << "MP2 container should have spin-dependent 2-RDM via lazy computation";
  EXPECT_TRUE(mp2_container->has_two_rdm_spin_traced())
      << "MP2 container should have spin-traced 2-RDM via lazy computation";

  // Get RDMs and verify they have reasonable values
  auto [one_rdm_aa, one_rdm_bb] =
      mp2_container->get_active_one_rdm_spin_dependent();

  // Verify 1-RDM has non-zero values (visit each variant separately)
  std::visit(
      [](const auto& mat) {
        EXPECT_GT(mat.norm(), 0.0) << "1-RDM aa should not be zero";
      },
      one_rdm_aa);
  std::visit(
      [](const auto& mat) {
        EXPECT_GT(mat.norm(), 0.0) << "1-RDM bb should not be zero";
      },
      one_rdm_bb);

  // Verify 2-RDMs are available and have non-zero values
  auto [two_rdm_aabb, two_rdm_aaaa, two_rdm_bbbb] =
      mp2_container->get_active_two_rdm_spin_dependent();
  std::visit(
      [](const auto& rdm) {
        bool has_nonzero = false;
        for (size_t i = 0; i < rdm.size() && !has_nonzero; ++i) {
          if (std::abs(rdm.data()[i]) > 1e-12) {
            has_nonzero = true;
          }
        }
        EXPECT_TRUE(has_nonzero) << "2-RDM aaaa should have non-zero values";
      },
      two_rdm_aaaa);
}

// Test that both amplitudes and CI coefficients are available on MP2Container
TEST_F(MP2Test, AmplitudesAndCICoefficientsAvailable) {
  // O2 structure with 2.3 Bohr bond length
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  // Restricted HF calculation
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("basis_set", "cc-pvdz");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] = scf_factory->run(o2_structure_ptr, 0, 1);
  auto hf_orbitals = hf_wavefunction->get_orbitals();

  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto hf_hamiltonian = ham_factory->run(hf_orbitals);

  auto mp2_container =
      std::make_unique<MP2Container>(hf_hamiltonian, hf_wavefunction);

  // Test that both T amplitudes and CI coefficients are available
  // First, get CI coefficients (this triggers lazy evaluation)
  const auto& coefficients = mp2_container->get_coefficients();
  const auto& determinants = mp2_container->get_active_determinants();

  // Then, get amplitudes
  auto [t1_aa, t1_bb] = mp2_container->get_t1_amplitudes();
  auto [t2_abab, t2_aaaa, t2_bbbb] = mp2_container->get_t2_amplitudes();

  // Verify all are available
  EXPECT_TRUE(mp2_container->has_t1_amplitudes())
      << "T1 amplitudes should be available";
  EXPECT_TRUE(mp2_container->has_t2_amplitudes())
      << "T2 amplitudes should be available";

  std::visit(
      [](const auto& vec) {
        EXPECT_GT(vec.size(), 0) << "CI coefficients should be non-empty";
      },
      coefficients);

  EXPECT_GT(determinants.size(), 0) << "Determinants should be non-empty";

  // Verify T1 is zero for MP2
  std::visit(
      [](const auto& vec) {
        EXPECT_TRUE(vec.isZero(1e-10))
            << "T1 amplitudes should be zero for MP2";
      },
      t1_aa);
}

// Test that RDM traces are correct for MP2 wavefunction
TEST_F(MP2Test, RDMTracesAreCorrect) {
  // O2 structure with 2.3 Bohr bond length
  std::vector<Eigen::Vector3d> coordinates = {Eigen::Vector3d(0.0, 0.0, 0.0),
                                              Eigen::Vector3d(2.3, 0.0, 0.0)};
  std::vector<std::string> symbols = {"O", "O"};
  Structure o2_structure(coordinates, symbols);

  // Restricted HF calculation (singlet O2, multiplicity = 1)
  auto scf_factory = ScfSolverFactory::create("qdk");
  scf_factory->settings().set("basis_set", "cc-pvdz");
  scf_factory->settings().set("method", "hf");
  auto o2_structure_ptr = std::make_shared<Structure>(o2_structure);
  auto [hf_energy, hf_wavefunction] = scf_factory->run(o2_structure_ptr, 0, 1);
  auto hf_orbitals = hf_wavefunction->get_orbitals();

  // Create Hamiltonian
  auto ham_factory = HamiltonianConstructorFactory::create("qdk");
  auto hf_hamiltonian = ham_factory->run(hf_orbitals);

  // Create MP2Container with Hamiltonian and wavefunction
  auto mp2_container =
      std::make_unique<MP2Container>(hf_hamiltonian, hf_wavefunction);

  // Get RDMs from MP2 container (triggers lazy computation)
  const auto& rdm1 = std::get<Eigen::MatrixXd>(
      mp2_container->get_active_one_rdm_spin_traced());

  // Get number of electrons
  auto [n_alpha, n_beta] = hf_wavefunction->get_active_num_electrons();
  size_t n_electrons = n_alpha + n_beta;

  // Verify 1-RDM trace equals number of electrons
  // Tr(γ) = N
  double rdm1_trace = rdm1.trace();
  EXPECT_NEAR(rdm1_trace, static_cast<double>(n_electrons), 1e-6)
      << "1-RDM trace should equal number of electrons. "
      << "Trace: " << rdm1_trace << ", N_electrons: " << n_electrons;

  // Print diagnostic information
  std::cout << "  Number of electrons: " << n_electrons << std::endl;
  std::cout << "  1-RDM trace: " << rdm1_trace << " (expected: " << n_electrons
            << ")" << std::endl;
}

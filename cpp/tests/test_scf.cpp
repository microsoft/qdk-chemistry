// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class ScfTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Clean up any test files from previous runs
    std::filesystem::remove("temp.orbitals.xyz");
  }

  void TearDown() override {
    // Clean up test files
    std::filesystem::remove("temp.orbitals.xyz");
  }
};

class TestSCF : public ScfSolver {
 public:
  std::string name() const override { return "test_scf"; }

 protected:
  std::pair<double, std::shared_ptr<Wavefunction>> _run_impl(
      std::shared_ptr<Structure> /*structure*/, int charge, int multiplicity,
      std::optional<std::shared_ptr<Orbitals>> initial_guess) const override {
    // Dummy implementation for testing
    Eigen::MatrixXd coefficients = Eigen::MatrixXd::Zero(3, 3);
    Eigen::VectorXd energies = Eigen::VectorXd::Zero(3);

    auto orbitals = std::make_shared<Orbitals>(coefficients, energies,
                                               std::nullopt, nullptr);
    auto wfn = std::make_shared<Wavefunction>(
        std::make_unique<SlaterDeterminantContainer>(Configuration("000"),
                                                     orbitals));
    return {0.0, wfn};
  }
};

TEST_F(ScfTest, Factory) {
  auto available_solvers = ScfSolverFactory::available();
  EXPECT_EQ(available_solvers.size(), 1);
  EXPECT_EQ(available_solvers[0], "qdk");
  EXPECT_THROW(ScfSolverFactory::create("nonexistent_solver"),
               std::runtime_error);
  EXPECT_NO_THROW(ScfSolverFactory::register_instance(
      []() -> ScfSolverFactory::return_type {
        return std::make_unique<TestSCF>();
      }));
  EXPECT_THROW(ScfSolverFactory::register_instance(
                   []() -> ScfSolverFactory::return_type {
                     return std::make_unique<TestSCF>();
                   }),
               std::runtime_error);
  auto test_scf = ScfSolverFactory::create("test_scf");
}

TEST_F(ScfTest, Water) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  // Default settings
  auto [E_default, wfn_default] = scf_solver->run(water, 0, 1);
  auto orbitals_default = wfn_default->get_orbitals();
  EXPECT_NEAR(E_default - water->calculate_nuclear_repulsion_energy(),
              -83.9252697201, testing::scf_energy_tolerance);
  EXPECT_TRUE(orbitals_default->is_restricted());

  // Change basis set to def2-tzvp
  scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("basis_set", "def2-tzvp");
  std::cout << scf_solver->settings().to_json().dump(2) << std::endl;
  auto [E_def2tzvp, wfn_def2tzvp] = scf_solver->run(water, 0, 1);
  EXPECT_NEAR(E_def2tzvp - water->calculate_nuclear_repulsion_energy(),
              -84.0229441374, testing::scf_energy_tolerance);
}

TEST_F(ScfTest, Lithium) {
  auto li = testing::create_li_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Default settings
  auto [E_default, wfn_default] = scf_solver->run(li, 0, 2);
  EXPECT_NEAR(E_default, -7.4250663561e+00, testing::scf_energy_tolerance);
  EXPECT_FALSE(wfn_default->get_orbitals()->is_restricted());

  // Li +1 should be a singlet
  auto [E_li_plus_1, wfn_li_plus_1] = scf_solver->run(li, 1, 1);
  EXPECT_NEAR(E_li_plus_1, -7.232895386855468e+00,
              testing::scf_energy_tolerance);
  EXPECT_TRUE(wfn_li_plus_1->get_orbitals()->is_restricted());
}

TEST_F(ScfTest, Oxygen) {
  auto o2 = testing::create_o2_structure();
  auto scf_solver = ScfSolverFactory::create();

  // STO-3G
  scf_solver->settings().set("basis_set", "sto-3g");

  // Default should be a singlet
  auto [E_singlet, wfn_singlet] = scf_solver->run(o2, 0, 1);

  // Run as a triplet
  auto [E_triplet, wfn_triplet] = scf_solver->run(o2, 0, 3);

  EXPECT_NEAR(E_singlet - o2->calculate_nuclear_repulsion_energy(),
              -1.7558700613e+02, testing::scf_energy_tolerance);
  EXPECT_NEAR(E_triplet - o2->calculate_nuclear_repulsion_energy(),
              -1.7566984837e+02, testing::scf_energy_tolerance);

  // Check singlet orbitals
  EXPECT_TRUE(wfn_singlet->get_orbitals()->is_restricted());

  // Check triplet orbitals
  EXPECT_FALSE(wfn_triplet->get_orbitals()->is_restricted());
}

TEST_F(ScfTest, WaterDftB3lyp) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Set B3LYP DFT method
  scf_solver->settings().set("method", "b3lyp");
  scf_solver->settings().set("basis_set", "def2-svp");

  auto [E_b3lyp, wfn_b3lyp] = scf_solver->run(water, 0, 1);

  EXPECT_NEAR(E_b3lyp - water->calculate_nuclear_repulsion_energy(),
              -84.335786559482, testing::scf_energy_tolerance);
  EXPECT_TRUE(wfn_b3lyp->get_orbitals()->is_restricted());
}

TEST_F(ScfTest, WaterDftPbe) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Set PBE DFT method
  scf_solver->settings().set("method", "pbe");
  scf_solver->settings().set("basis_set", "def2-svp");

  auto [E_pbe, wfn_pbe] = scf_solver->run(water, 0, 1);

  // PBE should give a reasonable energy (different from B3LYP)
  EXPECT_TRUE(wfn_pbe->get_orbitals()->is_restricted());

  // Energy should be reasonable (negative and close to other DFT results)
  EXPECT_LT(E_pbe - water->calculate_nuclear_repulsion_energy(),
            -80.0);  // Should be reasonable for water
  EXPECT_GT(E_pbe - water->calculate_nuclear_repulsion_energy(), -90.0);
}

TEST_F(ScfTest, LithiumDftB3lypUks) {
  auto lithium = testing::create_li_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Set B3LYP DFT method with UKS for lithium
  scf_solver->settings().set("method", "b3lyp");
  scf_solver->settings().set("basis_set", "def2-svp");
  // Lithium should naturally be UKS due to its doublet ground state

  auto [energy_b3lyp, wfn_b3lyp] = scf_solver->run(lithium, 0, 2);
  auto orbitals_b3lyp = wfn_b3lyp->get_orbitals();

  // Check that we get reasonable DFT results
  EXPECT_NEAR(energy_b3lyp, -7.484980651804635, testing::scf_energy_tolerance);
  EXPECT_FALSE(
      orbitals_b3lyp->is_restricted());  // Should be UKS (unrestricted)

  // Check that basis set is populated
  EXPECT_TRUE(orbitals_b3lyp->has_basis_set());
  EXPECT_TRUE(orbitals_b3lyp->has_overlap_matrix());

  // Check occupations - lithium should have 2 alpha electrons and 1 beta
  // electron
  auto [occupations_alpha, occupations_beta] =
      wfn_b3lyp->get_total_orbital_occupations();
  double total_alpha_electrons = occupations_alpha.sum();
  double total_beta_electrons = occupations_beta.sum();
  EXPECT_NEAR(total_alpha_electrons, 2.0, 1e-10);
  EXPECT_NEAR(total_beta_electrons, 1.0, 1e-10);
}

TEST_F(ScfTest, LithiumDftPbeUks) {
  auto lithium = testing::create_li_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Set PBE DFT method with UKS for lithium
  scf_solver->settings().set("method", "pbe");
  scf_solver->settings().set("basis_set", "def2-svp");

  auto [energy_pbe, wfn_pbe] = scf_solver->run(lithium, 0, 2);
  auto orbitals_pbe = wfn_pbe->get_orbitals();

  // Check that we get reasonable DFT results (don't check specific energy for
  // PBE)
  EXPECT_FALSE(orbitals_pbe->is_restricted());  // Should be UKS (unrestricted)

  // Check that basis set is populated
  EXPECT_TRUE(orbitals_pbe->has_basis_set());
  EXPECT_TRUE(orbitals_pbe->has_overlap_matrix());

  // Check occupations
  auto [occupations_alpha, occupations_beta] =
      wfn_pbe->get_total_orbital_occupations();
  double total_alpha_electrons = occupations_alpha.sum();
  double total_beta_electrons = occupations_beta.sum();
  EXPECT_NEAR(total_alpha_electrons, 2.0, 1e-10);
  EXPECT_NEAR(total_beta_electrons, 1.0, 1e-10);

  // Energy should be reasonable for lithium
  EXPECT_LT(energy_pbe, -7.0);  // Should be reasonable for lithium
  EXPECT_GT(energy_pbe, -8.0);
}

TEST_F(ScfTest, OxygenTripletDftB3lypUks) {
  auto oxygen_molecule = testing::create_o2_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Set B3LYP DFT method for O2 triplet state
  scf_solver->settings().set("method", "b3lyp");
  scf_solver->settings().set("basis_set", "def2-svp");

  auto [energy_b3lyp, wfn_b3lyp] = scf_solver->run(oxygen_molecule, 0, 3);
  auto orbitals_b3lyp = wfn_b3lyp->get_orbitals();

  // Check that we get reasonable DFT results
  EXPECT_FALSE(
      orbitals_b3lyp->is_restricted());  // Should be UKS (unrestricted)

  // Energy should be reasonable for O2
  EXPECT_LT(
      energy_b3lyp - oxygen_molecule->calculate_nuclear_repulsion_energy(),
      -175.0);
  EXPECT_GT(
      energy_b3lyp - oxygen_molecule->calculate_nuclear_repulsion_energy(),
      -185.0);

  // Check that basis set is populated
  EXPECT_TRUE(orbitals_b3lyp->has_basis_set());
  EXPECT_TRUE(orbitals_b3lyp->has_overlap_matrix());

  // Check occupations - O2 triplet should have 9 alpha and 7 beta electrons
  auto [occupations_alpha, occupations_beta] =
      wfn_b3lyp->get_total_orbital_occupations();
  double total_alpha_electrons = occupations_alpha.sum();
  double total_beta_electrons = occupations_beta.sum();
  EXPECT_NEAR(total_alpha_electrons, 9.0, 1e-10);
  EXPECT_NEAR(total_beta_electrons, 7.0, 1e-10);
}

TEST_F(ScfTest, OxygenTripletDftPbeUks) {
  auto oxygen_molecule = testing::create_o2_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Set PBE DFT method for O2 triplet state
  scf_solver->settings().set("method", "pbe");
  scf_solver->settings().set("basis_set", "def2-svp");

  auto [energy_pbe, wfn_pbe] = scf_solver->run(oxygen_molecule, 0, 3);
  auto orbitals_pbe = wfn_pbe->get_orbitals();

  // Check that we get reasonable DFT results (don't check specific energy for
  // PBE)
  EXPECT_FALSE(orbitals_pbe->is_restricted());  // Should be UKS (unrestricted)

  // Check that basis set is populated
  EXPECT_TRUE(orbitals_pbe->has_basis_set());
  EXPECT_TRUE(orbitals_pbe->has_overlap_matrix());

  // Check occupations - O2 triplet should have 9 alpha and 7 beta electrons
  auto [occupations_alpha, occupations_beta] =
      wfn_pbe->get_total_orbital_occupations();
  double total_alpha_electrons = occupations_alpha.sum();
  double total_beta_electrons = occupations_beta.sum();
  EXPECT_NEAR(total_alpha_electrons, 9.0, 1e-10);
  EXPECT_NEAR(total_beta_electrons, 7.0, 1e-10);

  // Energy should be reasonable for O2
  EXPECT_LT(energy_pbe - oxygen_molecule->calculate_nuclear_repulsion_energy(),
            -170.0);  // Should be reasonable for O2
  EXPECT_GT(energy_pbe - oxygen_molecule->calculate_nuclear_repulsion_energy(),
            -190.0);
}

TEST_F(ScfTest, DftMethodCaseInsensitive) {
  auto water = testing::create_water_structure();

  // Test uppercase
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("basis_set", "sto-3g");
  scf_solver->settings().set("method", "B3LYP");
  auto [energy_upper, wfn_upper] = scf_solver->run(water, 0, 1);
  auto orbitals_upper = wfn_upper->get_orbitals();

  // Test lowercase
  scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("method", "b3lyp");
  scf_solver->settings().set("basis_set", "sto-3g");
  auto [energy_lower, wfn_lower] = scf_solver->run(water, 0, 1);
  auto orbitals_lower = wfn_lower->get_orbitals();

  // Should give the same result
  EXPECT_NEAR(energy_upper, energy_lower, testing::scf_energy_tolerance);
}

TEST_F(ScfTest, Settings_EdgeCases) {
  auto water = testing::create_water_structure();

  // Test invalid method - should throw during solve
  EXPECT_THROW(
      {
        auto scf_solver = ScfSolverFactory::create();
        scf_solver->settings().set("method", "not_a_method");
        scf_solver->run(water, 0, 1);
      },
      std::runtime_error);

  // Test invalid basis set - should throw during solve
  EXPECT_THROW(
      {
        auto scf_solver = ScfSolverFactory::create();
        scf_solver->settings().set("basis_set", "not_a_basis");
        scf_solver->run(water, 0, 1);
      },
      std::invalid_argument);

  // Test setting non-existent setting - should throw
  EXPECT_THROW(
      {
        auto scf_solver = ScfSolverFactory::create();
        scf_solver->settings().set("not_a_real_setting", 123);
      },
      std::runtime_error);

  // Test invalid max_iterations - should throw during solve
  EXPECT_THROW(
      {
        auto scf_solver = ScfSolverFactory::create();
        scf_solver->settings().set("max_iterations", "not_a_number");
        scf_solver->run(water, 0, 1);
      },
      qdk::chemistry::data::SettingTypeMismatch);
}

TEST_F(ScfTest, InitialGuessRestart) {
  // ===== Water as restricted test =====
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("basis_set", "def2-tzvp");
  scf_solver->settings().set("method", "hf");

  // First calculation - let it converge normally
  auto [energy_first, wfn_first] = scf_solver->run(water, 0, 1);
  auto orbitals_first = wfn_first->get_orbitals();

  // Verify we get the expected energy for HF/def2-tzvp
  EXPECT_NEAR(energy_first - water->calculate_nuclear_repulsion_energy(),
              -84.0229441374, testing::scf_energy_tolerance);

  // Now restart with the converged orbitals as initial guess
  // Create a new solver instance since settings are locked after run
  auto scf_solver2 = ScfSolverFactory::create();
  scf_solver2->settings().set("basis_set", "def2-tzvp");
  scf_solver2->settings().set("method", "hf");
  scf_solver2->settings().set(
      "max_iterations", 2);  // 2 is minimum as need to check energy difference

  // Second calculation with initial guess
  auto [energy_second, wfn_second] =
      scf_solver2->run(water, 0, 1, orbitals_first);

  // Should get the same energy (within tight tolerance)
  EXPECT_NEAR(energy_first, energy_second, testing::scf_energy_tolerance);
}

TEST_F(ScfTest, OxygenTripletInitialGuessRestart) {
  auto o2 = testing::create_o2_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("basis_set", "sto-3g");
  scf_solver->settings().set("method", "hf");

  // First calculation - let triplet converge normally
  auto [energy_o2_first, wfn_o2_first] = scf_solver->run(o2, 0, 3);
  auto orbitals_o2_first = wfn_o2_first->get_orbitals();

  // Verify we get the expected energy for HF/STO-3G triplet
  EXPECT_NEAR(energy_o2_first - o2->calculate_nuclear_repulsion_energy(),
              -1.7566984837e+02, testing::scf_energy_tolerance);

  // Now restart with the converged orbitals as initial guess
  // Create a new solver instance since settings are locked after run
  auto scf_solver_restart = ScfSolverFactory::create();
  scf_solver_restart->settings().set("basis_set", "sto-3g");
  scf_solver_restart->settings().set("method", "hf");
  scf_solver_restart->settings().set(
      "max_iterations", 2);  // 2 is minimum as need to check energy difference

  // Second calculation with initial guess
  auto [energy_o2_second, wfn_o2_second] =
      scf_solver_restart->run(o2, 0, 3, orbitals_o2_first);

  // Should get the same energy (within tight tolerance)
  EXPECT_NEAR(energy_o2_first, energy_o2_second, testing::scf_energy_tolerance);
}

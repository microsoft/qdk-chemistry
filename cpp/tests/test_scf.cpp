// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>

#include "../src/qdk/chemistry/algorithms/microsoft/utils.hpp"
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
  EXPECT_NEAR(E_default, -75.9229032345009, testing::scf_energy_tolerance);
  EXPECT_TRUE(orbitals_default->is_restricted());

  // Change basis set to def2-tzvp
  scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("basis_set", "def2-tzvp");
  std::cout << scf_solver->settings().to_json().dump(2) << std::endl;
  auto [E_def2tzvp, wfn_def2tzvp] = scf_solver->run(water, 0, 1);
  EXPECT_NEAR(E_def2tzvp, -76.0205776517675, testing::scf_energy_tolerance);
}

TEST_F(ScfTest, Lithium) {
  auto li = testing::create_li_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Default settings
  auto [E_default, wfn_default] = scf_solver->run(li, 0, 2);
  EXPECT_NEAR(E_default, -7.4250663561, testing::scf_energy_tolerance);
  EXPECT_FALSE(wfn_default->get_orbitals()->is_restricted());

  // Li +1 should be a singlet
  auto [E_li_plus_1, wfn_li_plus_1] = scf_solver->run(li, 1, 1);
  EXPECT_NEAR(E_li_plus_1, -7.2328981138900552, testing::scf_energy_tolerance);
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

  EXPECT_NEAR(E_singlet, -147.551127403083, testing::scf_energy_tolerance);
  EXPECT_NEAR(E_triplet, -147.633969643351, testing::scf_energy_tolerance);

  // Check singlet orbitals
  EXPECT_TRUE(wfn_singlet->get_orbitals()->is_restricted());

  // Check triplet orbitals
  EXPECT_FALSE(wfn_triplet->get_orbitals()->is_restricted());
}

TEST_F(ScfTest, Oxygen_atom_gdm) {
  auto oxygen = testing::create_oxygen_structure();
  auto scf_solver = ScfSolverFactory::create();
  // cc-pvdz
  scf_solver->settings().set("method", "pbe");
  scf_solver->settings().set("basis_set", "cc-pvdz");
  scf_solver->settings().set("enable_gdm", true);
  // Default should be a singlet
  auto [E_singlet, wfn_singlet] = scf_solver->run(oxygen, 0, 1);
  EXPECT_NEAR(E_singlet, -74.873106298, testing::scf_energy_tolerance);
  // Check singlet orbitals
  EXPECT_TRUE(wfn_singlet->get_orbitals()->is_restricted());
}

TEST_F(ScfTest, Oxygen_atom_history_size_limit_gdm) {
  auto oxygen = testing::create_oxygen_structure();
  auto scf_solver = ScfSolverFactory::create();
  // cc-pvdz
  scf_solver->settings().set("method", "pbe");
  scf_solver->settings().set("basis_set", "cc-pvdz");
  scf_solver->settings().set("enable_gdm", true);
  scf_solver->settings().set("gdm_bfgs_history_size_limit", 20);
  // Default should be a singlet
  auto [E_singlet, wfn_singlet] = scf_solver->run(oxygen, 0, 1);
  EXPECT_NEAR(E_singlet, -74.873106298, testing::scf_energy_tolerance);
  // Check singlet orbitals
  EXPECT_TRUE(wfn_singlet->get_orbitals()->is_restricted());
}

TEST_F(ScfTest, Oxygen_atom_one_diis_step_gdm) {
  auto oxygen = testing::create_oxygen_structure();
  auto scf_solver = ScfSolverFactory::create();
  // cc-pvdz
  scf_solver->settings().set("method", "pbe");
  scf_solver->settings().set("basis_set", "cc-pvdz");
  scf_solver->settings().set("enable_gdm", true);
  scf_solver->settings().set("gdm_max_diis_iteration", 1);

  auto [E_singlet, wfn_singlet] = scf_solver->run(oxygen, 0, 1);
  EXPECT_NEAR(E_singlet, -74.873106298, testing::scf_energy_tolerance);
  // Check singlet orbitals
  EXPECT_TRUE(wfn_singlet->get_orbitals()->is_restricted());
}

TEST_F(ScfTest, Water_triplet_gdm) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  // Default settings
  scf_solver->settings().set("method", "pbe");
  scf_solver->settings().set("enable_gdm", true);
  auto [E_default, wfn_default] = scf_solver->run(water, 0, 3);
  auto orbitals_default = wfn_default->get_orbitals();
  EXPECT_NEAR(E_default, -76.0343083322644, testing::scf_energy_tolerance);
  EXPECT_FALSE(orbitals_default->is_restricted());
}

TEST_F(ScfTest, Oxygen_atom_charged_doublet_gdm) {
  auto oxygen = testing::create_oxygen_structure();
  auto scf_solver = ScfSolverFactory::create();
  // cc-pvdz
  scf_solver->settings().set("method", "pbe");
  scf_solver->settings().set("basis_set", "cc-pvdz");
  scf_solver->settings().set("enable_gdm", true);
  scf_solver->settings().set("max_iterations", 100);

  auto [E_doublet, wfn_doublet] = scf_solver->run(oxygen, 1, 2);
  EXPECT_NEAR(E_doublet, -74.416994299, testing::scf_energy_tolerance);
  // Check singlet orbitals
  EXPECT_FALSE(wfn_doublet->get_orbitals()->is_restricted());
}

TEST_F(ScfTest, Oxygen_atom_invalid_energy_thresh_diis_switch_gdm) {
  auto oxygen = testing::create_oxygen_structure();
  auto scf_solver = ScfSolverFactory::create();
  // cc-pvdz
  scf_solver->settings().set("method", "pbe");
  scf_solver->settings().set("basis_set", "cc-pvdz");
  scf_solver->settings().set("enable_gdm", true);
  scf_solver->settings().set("energy_thresh_diis_switch", -2e-4);
  // Default should be a singlet
  EXPECT_THROW(scf_solver->run(oxygen, 0, 1),
               std::invalid_argument);  // open-shell dublet
}

TEST_F(ScfTest, Oxygen_atom_invalid_bfgs_history_size_limit_gdm) {
  auto oxygen = testing::create_oxygen_structure();
  auto scf_solver = ScfSolverFactory::create();
  // cc-pvdz
  scf_solver->settings().set("method", "pbe");
  scf_solver->settings().set("basis_set", "cc-pvdz");
  scf_solver->settings().set("enable_gdm", true);
  scf_solver->settings().set("gdm_bfgs_history_size_limit", 0);
  // Default should be a singlet
  EXPECT_THROW(scf_solver->run(oxygen, 0, 1), std::invalid_argument);
}

TEST_F(ScfTest, WaterDftB3lyp) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Set B3LYP DFT method
  scf_solver->settings().set("method", "b3lyp");
  scf_solver->settings().set("basis_set", "def2-svp");

  auto [E_b3lyp, wfn_b3lyp] = scf_solver->run(water, 0, 1);

  EXPECT_NEAR(E_b3lyp, -76.3334200741567, testing::scf_energy_tolerance);
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
  EXPECT_LT(E_pbe, -75.0);  // Should be reasonable for water
  EXPECT_GT(E_pbe, -77.0);
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
  EXPECT_NEAR(total_alpha_electrons, 2.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(total_beta_electrons, 1.0, testing::numerical_zero_tolerance);
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
  EXPECT_NEAR(total_alpha_electrons, 2.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(total_beta_electrons, 1.0, testing::numerical_zero_tolerance);

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
  EXPECT_LT(energy_b3lyp, -149.0);
  EXPECT_GT(energy_b3lyp, -151.0);

  // Check that basis set is populated
  EXPECT_TRUE(orbitals_b3lyp->has_basis_set());
  EXPECT_TRUE(orbitals_b3lyp->has_overlap_matrix());

  // Check occupations - O2 triplet should have 9 alpha and 7 beta electrons
  auto [occupations_alpha, occupations_beta] =
      wfn_b3lyp->get_total_orbital_occupations();
  double total_alpha_electrons = occupations_alpha.sum();
  double total_beta_electrons = occupations_beta.sum();
  EXPECT_NEAR(total_alpha_electrons, 9.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(total_beta_electrons, 7.0, testing::numerical_zero_tolerance);
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
  EXPECT_NEAR(total_alpha_electrons, 9.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(total_beta_electrons, 7.0, testing::numerical_zero_tolerance);

  // Energy should be reasonable for O2
  EXPECT_LT(energy_pbe, -149.0);  // Should be reasonable for O2
  EXPECT_GT(energy_pbe, -151.0);
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
  EXPECT_NEAR(energy_first, -76.0205776517675, testing::scf_energy_tolerance);

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
  EXPECT_NEAR(energy_o2_first, -147.633969643351,
              testing::scf_energy_tolerance);

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

TEST_F(ScfTest, AgHDef2SvpWithEcp) {
  // Test AgH with def2-svp basis set (includes ECP for Ag)
  auto agh = testing::create_agh_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Set def2-svp basis which includes ECP for Ag
  scf_solver->settings().set("basis_set", "def2-svp");
  scf_solver->settings().set("method", "hf");

  // Run SCF calculation - AgH has 48 electrons (Ag: 47, H: 1)
  // but with ECP, Ag has 19 valence electrons, so total is 20 electrons
  auto [energy, wfn] = scf_solver->run(agh, 0, 1);
  auto orbitals = wfn->get_orbitals();

  // Verify the calculation completed successfully
  EXPECT_TRUE(orbitals->is_restricted());
  EXPECT_TRUE(orbitals->has_basis_set());

  // Verify basis set includes ECP information
  auto basis_set = orbitals->get_basis_set();
  EXPECT_TRUE(basis_set->has_ecp_shells());

  // Check that ECP is properly configured for Ag
  // def2-svp has 4 ECP shells for Ag
  EXPECT_EQ(basis_set->get_num_ecp_shells(), 4);

  // Verify ECP name matches the basis set
  EXPECT_EQ(basis_set->get_ecp_name(), "def2-svp");

  // Check ECP electrons configuration
  EXPECT_TRUE(basis_set->has_ecp_electrons());
  auto ecp_electrons = basis_set->get_ecp_electrons();
  EXPECT_EQ(ecp_electrons.size(), 2);  // Two atoms: Ag and H
  EXPECT_EQ(ecp_electrons[0], 28);  // Ag has 28 core electrons replaced by ECP
  EXPECT_EQ(ecp_electrons[1], 0);   // H has no ECP (0 core electrons)

  // Verify the electronic energy matches expected value
  double nuclear_repulsion = agh->calculate_nuclear_repulsion_energy();
  double electronic_energy = energy - nuclear_repulsion;
  EXPECT_NEAR(electronic_energy, -162.0054639416,
              testing::scf_energy_tolerance);

  // Check electron count - with ECP, should have 20 valence electrons
  auto [occupations_alpha, occupations_beta] =
      wfn->get_total_orbital_occupations();
  double total_electrons = occupations_alpha.sum() + occupations_beta.sum();
  EXPECT_NEAR(total_electrons, 20.0, testing::numerical_zero_tolerance);

  // Verify ECP angular momentum types are present
  // def2-svp ECP for Ag includes different angular momentum shells
  auto ecp_shells = basis_set->get_ecp_shells();
  EXPECT_EQ(ecp_shells.size(), 4);

  // Verify ECP shells belong to the correct atom (Ag is atom 0)
  for (const auto& shell : ecp_shells) {
    EXPECT_EQ(shell.atom_index, 0);  // All ECP shells should be for Ag
  }

  // Verify that H (atom 1) has no ECP shells
  auto h_ecp_shells = basis_set->get_ecp_shells_for_atom(1);
  EXPECT_EQ(h_ecp_shells.size(), 0);

  // Verify Ag (atom 0) has all the ECP shells
  auto ag_ecp_shells = basis_set->get_ecp_shells_for_atom(0);
  EXPECT_EQ(ag_ecp_shells.size(), 4);

  // Verify ECP shells have various angular momentum types
  // def2-svp ECP for Ag should have different angular momentum values
  std::set<int> angular_momenta;
  for (const auto& shell : ag_ecp_shells) {
    angular_momenta.insert(shell.get_angular_momentum());
  }
  EXPECT_GT(angular_momenta.size(),
            1);  // Should have multiple angular momentum types
}

TEST_F(ScfTest, AgHBasisSetRoundTripSerialization) {
  // Test that basis set with ECP can be serialized and deserialized
  // This tests both to_json() and from_serialized_json() methods,
  // specifically covering ECP data preservation
  auto agh = testing::create_agh_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Set def2-svp basis which includes ECP for Ag
  scf_solver->settings().set("basis_set", "def2-svp");
  scf_solver->settings().set("method", "hf");

  // Run SCF calculation to get original basis set with ECP
  auto [energy1, wfn1] = scf_solver->run(agh, 0, 1);
  auto orbitals1 = wfn1->get_orbitals();
  auto basis_set1 = orbitals1->get_basis_set();

  // Save basis set to JSON file
  std::string temp_json_file = "temp_test.basis_set.json";
  basis_set1->to_json_file(temp_json_file);

  // Load basis set from JSON file
  auto basis_set2 = BasisSet::from_json_file(temp_json_file);

  // Clean up temp file
  std::filesystem::remove(temp_json_file);

  // Verify the loaded basis set has the same properties as the original
  EXPECT_STREQ(basis_set2->get_name().c_str(), basis_set1->get_name().c_str());
  EXPECT_EQ(basis_set2->get_num_shells(), basis_set1->get_num_shells());
  EXPECT_EQ(basis_set2->get_num_atomic_orbitals(),
            basis_set1->get_num_atomic_orbitals());
  EXPECT_EQ(basis_set2->get_num_atoms(), basis_set1->get_num_atoms());

  // Verify ECP data is preserved
  EXPECT_TRUE(basis_set2->has_ecp_shells());
  EXPECT_EQ(basis_set2->get_num_ecp_shells(), basis_set1->get_num_ecp_shells());
  EXPECT_STREQ(basis_set2->get_ecp_name().c_str(),
               basis_set1->get_ecp_name().c_str());

  // Verify ECP electrons are correctly deserialized
  EXPECT_TRUE(basis_set2->has_ecp_electrons());
  auto ecp_electrons1 = basis_set1->get_ecp_electrons();
  auto ecp_electrons2 = basis_set2->get_ecp_electrons();
  EXPECT_EQ(ecp_electrons2.size(), ecp_electrons1.size());
  EXPECT_EQ(ecp_electrons2[0], 28);  // Ag has 28 core electrons
  EXPECT_EQ(ecp_electrons2[1], 0);   // H has no ECP

  // Verify ECP shells are correctly deserialized
  auto ecp_shells1 = basis_set1->get_ecp_shells();
  auto ecp_shells2 = basis_set2->get_ecp_shells();
  EXPECT_EQ(ecp_shells2.size(), ecp_shells1.size());
  EXPECT_EQ(ecp_shells2.size(), 4);  // 4 ECP shells for Ag

  // Verify each ECP shell is correctly deserialized
  for (size_t i = 0; i < ecp_shells2.size(); ++i) {
    EXPECT_EQ(ecp_shells2[i].atom_index, ecp_shells1[i].atom_index);
    EXPECT_EQ(ecp_shells2[i].get_angular_momentum(),
              ecp_shells1[i].get_angular_momentum());
    EXPECT_EQ(ecp_shells2[i].get_num_atomic_orbitals(),
              ecp_shells1[i].get_num_atomic_orbitals());
  }

  // Verify ECP shells for individual atoms are correctly preserved
  auto ag_ecp_shells2 = basis_set2->get_ecp_shells_for_atom(0);
  EXPECT_EQ(ag_ecp_shells2.size(), 4);  // Ag should have 4 ECP shells

  auto h_ecp_shells2 = basis_set2->get_ecp_shells_for_atom(1);
  EXPECT_EQ(h_ecp_shells2.size(), 0);  // H should have no ECP shells

  // Verify regular shells are also preserved
  auto shells1 = basis_set1->get_shells();
  auto shells2 = basis_set2->get_shells();
  EXPECT_EQ(shells2.size(), shells1.size());

  // Verify the basis set structure can be used to create valid orbitals
  auto [coeff_alpha, coeff_beta] = orbitals1->get_coefficients();
  auto [energies_alpha, energies_beta] = orbitals1->get_energies();
  auto overlap = orbitals1->get_overlap_matrix();

  // Create orbitals with the deserialized basis set - this validates
  // that the basis set is fully functional
  auto orbitals2 = std::make_shared<Orbitals>(
      coeff_alpha, energies_alpha, overlap, basis_set2, std::nullopt);

  EXPECT_TRUE(orbitals2->has_basis_set());
  EXPECT_EQ(orbitals2->get_basis_set()->get_name(), "def2-svp");
}

TEST_F(ScfTest, AgHBasisSetEcpConversion) {
  // Test ECP conversion
  auto agh = testing::create_agh_structure();
  auto scf_solver = ScfSolverFactory::create();

  scf_solver->settings().set("basis_set", "def2-svp");
  scf_solver->settings().set("method", "hf");

  auto [energy, wfn] = scf_solver->run(agh, 0, 1);
  auto orbitals = wfn->get_orbitals();
  auto basis_set = orbitals->get_basis_set();

  // Test ECP shell conversion
  EXPECT_TRUE(basis_set->has_ecp_shells());
  EXPECT_EQ(basis_set->get_num_ecp_shells(), 4);

  auto ecp_shells = basis_set->get_ecp_shells();
  EXPECT_EQ(ecp_shells.size(), 4);

  // Verify ECP shells have correct structure
  for (const auto& shell : ecp_shells) {
    // Each ECP shell should belong to atom 0 (Ag)
    EXPECT_EQ(shell.atom_index, 0);

    // ECP shells should have exponents, coefficients, and rpowers
    const auto& exponents = shell.exponents;
    const auto& coefficients = shell.coefficients;
    const auto& rpowers = shell.rpowers;

    EXPECT_GT(exponents.size(), 0);
    EXPECT_EQ(exponents.size(), coefficients.size());
    EXPECT_EQ(exponents.size(), rpowers.size());

    // Verify all exponents are positive (physical constraint)
    for (int i = 0; i < exponents.size(); ++i) {
      EXPECT_GT(exponents[i], 0.0);
    }

    // Verify rpowers are non-negative integers (r^n where n >= 0)
    for (int i = 0; i < rpowers.size(); ++i) {
      EXPECT_GE(rpowers[i], 0);
    }
  }

  // Test ECP metadata conversion
  // Verify ECP name was set correctly
  EXPECT_EQ(basis_set->get_ecp_name(), "def2-svp");

  // Verify per-atom ECP electrons vector was built correctly
  auto ecp_electrons = basis_set->get_ecp_electrons();
  EXPECT_EQ(ecp_electrons.size(), 2);  // 2 atoms: Ag and H

  // Ag (atom 0) should have 28 core electrons replaced by ECP
  EXPECT_EQ(ecp_electrons[0], 28);

  // H (atom 1) should have 0 core electrons (no ECP)
  EXPECT_EQ(ecp_electrons[1], 0);

  // Verify the element_ecp_electrons mapping was correctly applied
  // Total electrons: Ag=47, H=1 -> 48 total
  // With ECP: Ag has 28 core electrons removed -> 19 valence
  // So total valence electrons = 19 (Ag) + 1 (H) = 20
  size_t total_ecp_electrons = 0;
  for (size_t ecp : ecp_electrons) {
    total_ecp_electrons += ecp;
  }
  EXPECT_EQ(total_ecp_electrons, 28);

  // Verify the orbital count is consistent with valence electrons
  // With 20 valence electrons and restricted calculation, we expect 10 occupied
  // orbitals
  auto [coeff_alpha, coeff_beta] = orbitals->get_coefficients();
  EXPECT_EQ(coeff_alpha.rows(), basis_set->get_num_atomic_orbitals());
}

TEST_F(ScfTest, AgHBasisSetEcpJsonMapping) {
  // Test element_ecp_electrons mapping in convert_to_json
  // This validates that the element_ecp_electrons map is correctly built
  // from per-atom ECP electrons and serialized as a flat list in JSON
  auto agh = testing::create_agh_structure();
  auto scf_solver = ScfSolverFactory::create();

  scf_solver->settings().set("basis_set", "def2-svp");
  scf_solver->settings().set("method", "hf");

  auto [energy, wfn] = scf_solver->run(agh, 0, 1);
  auto orbitals = wfn->get_orbitals();
  auto basis_set = orbitals->get_basis_set();

  // Serialize to JSON using convert_to_json
  auto json = qdk::chemistry::utils::microsoft::convert_to_json(*basis_set);

  // Verify ECP shells are present in JSON
  EXPECT_TRUE(json.contains("ecp_shells"));
  auto ecp_shells_json = json["ecp_shells"];
  EXPECT_EQ(ecp_shells_json.size(), 4);

  // Each ECP shell should have rpowers field
  for (const auto& shell_json : ecp_shells_json) {
    EXPECT_TRUE(shell_json.contains("rpowers"));
    EXPECT_TRUE(shell_json.contains("atom"));
    EXPECT_TRUE(shell_json.contains("am"));
    EXPECT_TRUE(shell_json.contains("exp"));
    EXPECT_TRUE(shell_json.contains("coeff"));
  }

  // Verify element_ecp_electrons mapping
  EXPECT_TRUE(json.contains("element_ecp_electrons"));
  auto element_ecp_electrons_json = json["element_ecp_electrons"];

  // element_ecp_electrons should be a flat list: [atomic_num1, ecp_elec1,
  // atomic_num2, ecp_elec2, ...] For AgH: Ag (Z=47) has 28 ECP electrons, H
  // (Z=1) has 0 (not in map) So the flat list should be: [47, 28]
  EXPECT_TRUE(element_ecp_electrons_json.is_array());
  EXPECT_EQ(element_ecp_electrons_json.size(),
            2);  // One element (Ag) with non-zero ECP

  // Parse the flat list
  std::map<int, int> element_ecp_map;
  for (size_t i = 0; i + 1 < element_ecp_electrons_json.size(); i += 2) {
    int atomic_num = element_ecp_electrons_json[i];
    int ecp_elec = element_ecp_electrons_json[i + 1];
    element_ecp_map[atomic_num] = ecp_elec;
  }

  // Verify Ag (Z=47) has 28 ECP electrons
  EXPECT_EQ(element_ecp_map.size(), 1);
  EXPECT_TRUE(element_ecp_map.find(47) != element_ecp_map.end());
  EXPECT_EQ(element_ecp_map[47], 28);

  // Verify H (Z=1) is NOT in the map (has 0 ECP electrons)
  EXPECT_TRUE(element_ecp_map.find(1) == element_ecp_map.end());

  // Verify the logic that builds element_ecp_electrons from per-atom vector
  // Iterating through atoms and filtering non-zero ECP
  auto ecp_electrons = basis_set->get_ecp_electrons();
  auto structure = basis_set->get_structure();
  auto nuclear_charges = structure->get_nuclear_charges();

  // Build the map manually to verify the algorithm
  std::map<int, int> expected_map;
  for (size_t i = 0; i < ecp_electrons.size(); ++i) {
    if (ecp_electrons[i] > 0) {
      int atomic_num = static_cast<int>(nuclear_charges[i]);
      expected_map[atomic_num] = static_cast<int>(ecp_electrons[i]);
    }
  }

  EXPECT_EQ(expected_map.size(), element_ecp_map.size());
  EXPECT_EQ(expected_map, element_ecp_map);

  // Verify the flat list serialization
  std::vector<int> expected_flat_list;
  for (const auto& [k, v] : expected_map) {
    expected_flat_list.push_back(k);
    expected_flat_list.push_back(v);
  }

  std::vector<int> actual_flat_list = element_ecp_electrons_json;
  EXPECT_EQ(actual_flat_list, expected_flat_list);

  // Verify nuclear_charges transformation
  EXPECT_TRUE(json.contains("atoms"));
  auto atoms_json = json["atoms"];
  EXPECT_EQ(atoms_json.size(), 2);  // Ag and H

  std::vector<unsigned> expected_atoms = {47, 1};  // Ag, H
  std::vector<unsigned> actual_atoms = atoms_json;
  EXPECT_EQ(actual_atoms, expected_atoms);

  // Verify the complete JSON structure matches the template
  EXPECT_TRUE(json.contains("name"));
  EXPECT_EQ(json["name"], "def2-svp");

  EXPECT_TRUE(json.contains("pure"));
  EXPECT_EQ(json["pure"], true);

  EXPECT_TRUE(json.contains("mode"));
  EXPECT_EQ(json["mode"], "RAW");

  EXPECT_TRUE(json.contains("num_atomic_orbitals"));
  EXPECT_EQ(json["num_atomic_orbitals"], basis_set->get_num_atomic_orbitals());

  EXPECT_TRUE(json.contains("electron_shells"));
  auto electron_shells_json = json["electron_shells"];
  EXPECT_GT(electron_shells_json.size(), 0);
}

TEST_F(ScfTest, AgHEcpShellIndices) {
  // Test ECP shell index retrieval methods
  auto agh = testing::create_agh_structure();
  auto scf_solver = ScfSolverFactory::create();

  scf_solver->settings().set("basis_set", "def2-svp");
  scf_solver->settings().set("method", "hf");

  auto [energy, wfn] = scf_solver->run(agh, 0, 1);
  auto orbitals = wfn->get_orbitals();
  auto basis_set = orbitals->get_basis_set();

  // Test get_ecp_shell_indices_for_atom
  // Ag (atom 0) should have ECP shells
  auto ag_ecp_indices = basis_set->get_ecp_shell_indices_for_atom(0);
  EXPECT_EQ(ag_ecp_indices.size(), 4);  // def2-svp has 4 ECP shells for Ag

  // All indices should be valid (< total number of ECP shells)
  for (size_t idx : ag_ecp_indices) {
    EXPECT_LT(idx, basis_set->get_num_ecp_shells());
  }

  // H (atom 1) should have no ECP shells
  auto h_ecp_indices = basis_set->get_ecp_shell_indices_for_atom(1);
  EXPECT_EQ(h_ecp_indices.size(), 0);

  // Test get_ecp_shell_indices_for_orbital_type
  // Get all ECP shells to determine which orbital types are present
  auto ecp_shells = basis_set->get_ecp_shells();
  std::map<OrbitalType, size_t> orbital_type_counts;

  for (const auto& shell : ecp_shells) {
    int l = shell.get_angular_momentum();
    OrbitalType type;
    if (l == -1) {
      type = OrbitalType::UL;  // Local potential
    } else if (l == 0) {
      type = OrbitalType::S;
    } else if (l == 1) {
      type = OrbitalType::P;
    } else if (l == 2) {
      type = OrbitalType::D;
    } else if (l == 3) {
      type = OrbitalType::F;
    } else if (l == 4) {
      type = OrbitalType::G;
    } else if (l == 5) {
      type = OrbitalType::H;
    } else if (l == 6) {
      type = OrbitalType::I;
    } else {
      continue;  // Skip unknown types
    }
    orbital_type_counts[type]++;
  }

  // Test that we can retrieve shells by orbital type
  EXPECT_GT(orbital_type_counts.size(), 0);

  for (const auto& [orbital_type, expected_count] : orbital_type_counts) {
    auto indices =
        basis_set->get_ecp_shell_indices_for_orbital_type(orbital_type);
    EXPECT_EQ(indices.size(), expected_count);

    // Verify all indices are valid
    for (size_t idx : indices) {
      EXPECT_LT(idx, basis_set->get_num_ecp_shells());
      // Verify the shell at this index has the correct orbital type
      auto shell = ecp_shells[idx];
      int l = shell.get_angular_momentum();
      if (orbital_type == OrbitalType::UL) {
        EXPECT_EQ(l, -1);
      } else {
        EXPECT_EQ(l, static_cast<int>(orbital_type));
      }
    }
  }

  // Test get_ecp_shell_indices_for_atom_and_orbital_type
  for (const auto& [orbital_type, _] : orbital_type_counts) {
    // Get indices for Ag (atom 0) with this orbital type
    auto ag_type_indices =
        basis_set->get_ecp_shell_indices_for_atom_and_orbital_type(
            0, orbital_type);

    // Should be a subset of both atom 0 shells and orbital type shells
    auto ag_all_indices = basis_set->get_ecp_shell_indices_for_atom(0);
    auto type_all_indices =
        basis_set->get_ecp_shell_indices_for_orbital_type(orbital_type);

    EXPECT_LE(ag_type_indices.size(), ag_all_indices.size());
    EXPECT_LE(ag_type_indices.size(), type_all_indices.size());

    // Verify each index is in both sets
    for (size_t idx : ag_type_indices) {
      EXPECT_TRUE(std::find(ag_all_indices.begin(), ag_all_indices.end(),
                            idx) != ag_all_indices.end());
      EXPECT_TRUE(std::find(type_all_indices.begin(), type_all_indices.end(),
                            idx) != type_all_indices.end());

      // Verify the shell matches both criteria
      auto shell = ecp_shells[idx];
      EXPECT_EQ(shell.atom_index, 0);
      int l = shell.get_angular_momentum();
      if (orbital_type == OrbitalType::UL) {
        EXPECT_EQ(l, -1);
      } else {
        EXPECT_EQ(l, static_cast<int>(orbital_type));
      }
    }

    // H (atom 1) should have no ECP shells for any orbital type
    auto h_type_indices =
        basis_set->get_ecp_shell_indices_for_atom_and_orbital_type(
            1, orbital_type);
    EXPECT_EQ(h_type_indices.size(), 0);
  }
}

TEST_F(ScfTest, H2ScanDIISNumericalStability) {
  // Test that SCF handles numerical edge cases in H2 bond scans
  // This reproduces issues found with exact floating-point values from linspace
  // where b_max can become zero in DIIS extrapolation

  // Helper lambda to generate linspace values like numpy
  auto linspace = [](double start, double stop, size_t num) {
    std::vector<double> result(num);
    if (num == 1) {
      result[0] = start;
      return result;
    }
    double step = (stop - start) / (num - 1);
    for (size_t i = 0; i < num; ++i) {
      result[i] = start + i * step;
    }
    return result;
  };

  auto full_linspace = linspace(0.5, 5.0, 100);
  std::vector<double> test_lengths = {
      full_linspace[3],                            // b_max = 0 in DIIS
      std::round(full_linspace[3] * 1e15) / 1e15,  // b_max approx 0 in DIIS
      full_linspace[0]                             // b_max != 0 in DIIS
  };
  std::vector<double> expected_energies = {
      -0.7383108980408086,  // full_linspace[3]
      -0.7383108980408086,  // rounded full_linspace[3]
      -0.4033264392907958   // full_linspace[0]
  };

  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("basis_set", "sto-3g");

  std::vector<std::string> symbols = {"H", "H"};
  for (size_t i = 0; i < test_lengths.size(); ++i) {
    std::vector<Eigen::Vector3d> coordinates = {
        Eigen::Vector3d(0.0, 0.0, 0.0),
        Eigen::Vector3d(0.0, 0.0, test_lengths[i])};

    auto structure = std::make_shared<Structure>(coordinates, symbols);
    auto [energy, wavefunction] = scf_solver->run(structure, 0, 1);

    EXPECT_NEAR(energy, expected_energies[i], testing::scf_energy_tolerance);
  }
}

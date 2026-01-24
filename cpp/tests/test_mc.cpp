// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class MCTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

class TestMultiConfigurationCalculator : public MultiConfigurationCalculator {
 public:
  std::string name() const override { return "_test_mc_dummy"; }
  std::pair<double, std::shared_ptr<Wavefunction>> _run_impl(
      std::shared_ptr<Hamiltonian> hamiltonian, unsigned int /*nalpha*/,
      unsigned int /*nbeta*/) const override {
    // Dummy implementation for testing
    Eigen::VectorXcd coeffs(1);
    coeffs(0) = std::complex<double>(1.0, 0.0);
    Wavefunction::DeterminantVector dets{Configuration("2000000")};
    auto container = std::make_unique<CasWavefunctionContainer>(
        coeffs, dets, hamiltonian->get_orbitals());
    Wavefunction wfn(std::move(container));
    return {0.0, std::make_shared<Wavefunction>(std::move(wfn))};
  }
};

TEST_F(MCTest, MCSettings) {
  // Test MCSettings constructor and default values
  MultiConfigurationSettings mc_settings;

  // Test that default values are set correctly
  EXPECT_FALSE(mc_settings.get<bool>("calculate_one_rdm"));
  EXPECT_FALSE(mc_settings.get<bool>("calculate_two_rdm"));
  EXPECT_DOUBLE_EQ(mc_settings.get<double>("ci_residual_tolerance"), 1.0e-6);
  EXPECT_EQ(mc_settings.get<int64_t>("max_solver_iterations"), 200);

  // Test destructor by creating and destroying in scope
  {
    MultiConfigurationSettings temp_settings;
  }  // Destructor called here
}

TEST_F(MCTest, MultiConfigurationCalculatorBase) {
  // Test direct instantiation of TestMultiConfigurationCalculator to cover base
  // class methods
  TestMultiConfigurationCalculator test_calc;

  // Test destructor by creating and destroying in scope
  {
    TestMultiConfigurationCalculator temp_calc;
  }  // Destructor called here
}

TEST_F(MCTest, MACIS_MetaData) {
  auto mc = MultiConfigurationCalculatorFactory::create();
  EXPECT_NO_THROW({ auto settings = mc->settings(); });
}

TEST_F(MCTest, Factory) {
  auto available_solvers = MultiConfigurationCalculatorFactory::available();
  EXPECT_EQ(available_solvers.size(), 2);
  EXPECT_TRUE(std::find(available_solvers.begin(), available_solvers.end(),
                        "macis_asci") != available_solvers.end());
  EXPECT_TRUE(std::find(available_solvers.begin(), available_solvers.end(),
                        "macis_cas") != available_solvers.end());
  EXPECT_THROW(MultiConfigurationCalculatorFactory::create("nonexistent_mc"),
               std::runtime_error);
  EXPECT_NO_THROW(MultiConfigurationCalculatorFactory::register_instance(
      []() -> MultiConfigurationCalculatorFactory::return_type {
        return std::make_unique<TestMultiConfigurationCalculator>();
      }));
  EXPECT_THROW(
      MultiConfigurationCalculatorFactory::register_instance(
          []() -> MultiConfigurationCalculatorFactory::return_type {
            return std::make_unique<TestMultiConfigurationCalculator>();
          }),
      std::runtime_error);
  auto test_mc = MultiConfigurationCalculatorFactory::create("_test_mc_dummy");
}

TEST_F(MCTest, Water_STO3G_ASCI) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Run SCF
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  // Construct the Hamiltonian
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  // Run FCI
  auto mc =
      qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory::create(
          "macis_asci");
  mc->settings().set("ntdets_max", 128);
  mc->settings().set("ntdets_min", 1);
  mc->settings().set("core_selection_strategy", "fixed");  // Deterministic size

  auto [E_fci, wfn_fci] = mc->run(ham, 5, 5);
  // FCI electronic energy is -8.301534669468e+01
  // Need to subtract the core energy to get this result
  EXPECT_NEAR(E_fci - ham->get_core_energy(), -8.301534669468e+01,
              testing::ci_energy_tolerance);
  EXPECT_EQ(wfn_fci->size(), 128);
}

TEST_F(MCTest, Water_STO3G_FCI) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Run SCF
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

  // Construct the Hamiltonian
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

  // Run FCI
  auto mc =
      qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory::create();
  auto [E_fci, wfn_fci] = mc->run(ham, 5, 5);
  EXPECT_NEAR(ham->get_core_energy(),
              water->calculate_nuclear_repulsion_energy(),
              testing::numerical_zero_tolerance);

  // FCI electronic energy is -8.301534669468e+01
  // Subtract core energy
  EXPECT_NEAR(E_fci - ham->get_core_energy(), -8.301534669468e+01,
              testing::ci_energy_tolerance);
  EXPECT_EQ(wfn_fci->size(), 441);
}

TEST_F(MCTest, Water_DEF2SVP_CASCI) {
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Run SCF
  auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "def2-svp");

  // Construct the Hamiltonian
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();

  // Create a new orbitals object with active space information
  auto orbitals_with_active_space = testing::with_active_space(
      wfn_HF->get_orbitals(), std::vector<size_t>{2, 3, 4, 5, 6, 7},
      std::vector<size_t>{0, 1});
  // Print number of electrons before and after selection
  auto ham = hamiltonian_constructor->run(orbitals_with_active_space);
  EXPECT_NEAR(ham->get_core_energy(), -63.499129387385118,
              testing::numerical_zero_tolerance * 10);

  // Run CASCI
  auto mc =
      qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory::create();
  auto [E_cas, wfn_cas] = mc->run(ham, 3, 3);
  EXPECT_NEAR(E_cas, -75.945290197648347, testing::ci_energy_tolerance);
  EXPECT_EQ(wfn_cas->size(), 400);
}

TEST_F(MCTest, StretchedN2_CCPVDZ_CASCI) {
  auto n2 = testing::create_stretched_n2_structure();
  auto scf_solver = ScfSolverFactory::create();

  // Run SCF
  auto [E_HF, wfn_HF] = scf_solver->run(n2, 0, 1, "cc-pvdz");

  // Construct the Hamiltonian
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();

  // Create a new orbitals object with active space information
  auto orbitals_with_active_space = testing::with_active_space(
      wfn_HF->get_orbitals(), std::vector<size_t>{2, 3, 8, 9},
      std::vector<size_t>{0, 1, 4, 5, 6});

  auto ham = hamiltonian_constructor->run(orbitals_with_active_space);
  EXPECT_NEAR(ham->get_core_energy(), -101.76185007585768,
              testing::numerical_zero_tolerance * 10);

  // Run CASCI
  auto mc =
      qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory::create();
  auto [E_cas, wfn_cas] = mc->run(ham, 2, 2);
  EXPECT_NEAR(E_cas, -108.343414308242, testing::ci_energy_tolerance);
}

TEST_F(MCTest, MCSettings_DefaultValues) {
  // Test MCSettings constructor and default values (covers lines 30-39, 41)
  MultiConfigurationSettings mc_settings;

  // Test all default values set in the constructor
  EXPECT_FALSE(mc_settings.get<bool>("calculate_one_rdm"));
  EXPECT_FALSE(mc_settings.get<bool>("calculate_two_rdm"));
  EXPECT_DOUBLE_EQ(mc_settings.get<double>("ci_residual_tolerance"), 1.0e-6);
  EXPECT_EQ(mc_settings.get<int64_t>("max_solver_iterations"), 200);

  // Test that we can modify settings
  mc_settings.set("calculate_one_rdm", true);
  EXPECT_TRUE(mc_settings.get<bool>("calculate_one_rdm"));

  mc_settings.set("ci_residual_tolerance", 1.0e-8);
  EXPECT_DOUBLE_EQ(mc_settings.get<double>("ci_residual_tolerance"), 1.0e-8);

  // Test destructor is called implicitly when object goes out of scope
}

TEST_F(MCTest, TestMultiConfigurationCalculator_ConstructorDestructor) {
  // Test MultiConfigurationCalculator constructor and destructor
  {
    TestMultiConfigurationCalculator test_calc;

    // Verify the object is constructed properly
    EXPECT_NO_THROW({ auto& settings = test_calc.settings(); });

    // Test the calculate method works
    auto water = testing::create_water_structure();
    auto scf_solver = ScfSolverFactory::create();
    auto [E_HF, wfn_HF] = scf_solver->run(water, 0, 1, "sto-3g");

    auto hamiltonian_constructor =
        qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
    auto ham = hamiltonian_constructor->run(wfn_HF->get_orbitals());

    auto [energy, wfn] = test_calc.run(ham, 5, 5);
    EXPECT_DOUBLE_EQ(energy, 0.0);  // Our test implementation returns 0.0

    // Destructor will be called automatically when test_calc goes out of scope
  }
}

TEST_F(MCTest, Factory_UnregisterInstance) {
  // Test the unregister functionality
  // First register a test instance
  auto available_before = MultiConfigurationCalculatorFactory::available();
  if (std::find(available_before.begin(), available_before.end(),
                "_test_mc_dummy") == available_before.end()) {
    MultiConfigurationCalculatorFactory::register_instance(
        []() -> MultiConfigurationCalculatorFactory::return_type {
          return std::make_unique<TestMultiConfigurationCalculator>();
        });
  }

  // Verify it was registered
  available_before = MultiConfigurationCalculatorFactory::available();
  EXPECT_TRUE(std::find(available_before.begin(), available_before.end(),
                        "_test_mc_dummy") != available_before.end());

  // Test unregistering an existing key
  EXPECT_TRUE(MultiConfigurationCalculatorFactory::unregister_instance(
      "_test_mc_dummy"));

  // Verify it was removed
  auto available_after = MultiConfigurationCalculatorFactory::available();
  EXPECT_TRUE(std::find(available_after.begin(), available_after.end(),
                        "_test_mc_dummy") == available_after.end());

  // Test unregistering a non-existent key
  EXPECT_FALSE(MultiConfigurationCalculatorFactory::unregister_instance(
      "_nonexistent_key"));
}

/**
 * @brief Test selected CI calculation on hydrogen atom (single electron edge
 * case)
 *
 */
TEST_F(MCTest, HydrogenAtom_CCPVDZ_SCI) {
  // Create structure for H atom
  auto h_atom = testing::create_hydrogen_structure();

  // Run SCF for H atom (charge=0, spin_multiplicity=2)
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("auto"));
  scf_solver->settings().set("enable_gdm", true);
  auto [E_HF, wfn_HF] = scf_solver->run(h_atom, 0, 2, "cc-pvdz");
  EXPECT_NEAR(E_HF, -0.49927840341958307, testing::scf_energy_tolerance);

  // Construct the Hamiltonian
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto orbitals_with_active_space = testing::with_active_space(
      wfn_HF->get_orbitals(), std::vector<size_t>{0, 1}, std::vector<size_t>{});
  auto ham = hamiltonian_constructor->run(orbitals_with_active_space);

  // Run selected CI calculation (1 alpha electron, 0 beta electrons)
  auto mc =
      qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory::create(
          "macis_asci");
  mc->settings().set("core_selection_strategy", "fixed");
  auto [E_sci, wfn_sci] = mc->run(ham, 1, 0);
  EXPECT_NEAR(E_sci, -0.49927840341958307, testing::ci_energy_tolerance);
}

/**
 * @brief Test selected CI calculation on nitrogen atom (edge case for Davidson)
 *
 */
TEST_F(MCTest, NitrogenAtom_CCPVDZ_SCI) {
  // Create structure for N atom
  auto n_atom = testing::create_nitrogen_structure();

  // Run SCF for N atom (charge=0, spin_multiplicity=4)
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("auto"));
  scf_solver->settings().set("enable_gdm", true);
  auto [E_HF, wfn_HF] = scf_solver->run(n_atom, 0, 4, "cc-pvdz");
  EXPECT_NEAR(E_HF, -54.391114562199718, testing::scf_energy_tolerance);

  // Construct the Hamiltonian (frozen 1s core)
  auto hamiltonian_constructor =
      qdk::chemistry::algorithms::HamiltonianConstructorFactory::create();
  auto orbitals_with_active_space = testing::with_active_space(
      wfn_HF->get_orbitals(), std::vector<size_t>{1, 2, 3, 4},
      std::vector<size_t>{0});
  auto ham = hamiltonian_constructor->run(orbitals_with_active_space);

  // Run selected CI calculation (4 alpha electrons, 1 beta electron)
  auto mc =
      qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory::create(
          "macis_asci");
  mc->settings().set("core_selection_strategy", "fixed");
  auto [E_sci, wfn_sci] = mc->run(ham, 4, 1);
  EXPECT_NEAR(E_sci, -54.385841001513917, testing::ci_energy_tolerance);
}

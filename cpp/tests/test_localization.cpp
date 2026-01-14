// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <filesystem>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>

#include "testing_utilities.hpp"
#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class LocalizationTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

class TestLocalization : public Localizer {
 private:
  Settings settings_;

 public:
  TestLocalization() = default;
  ~TestLocalization() override = default;

  std::string name() const override { return "_dummy_localizer"; }
  std::shared_ptr<qdk::chemistry::data::Wavefunction> _run_impl(
      std::shared_ptr<qdk::chemistry::data::Wavefunction> wavefunction,
      const std::vector<size_t>& loc_indices_a,
      const std::vector<size_t>& loc_indices_b) const override {
    std::cout << "TestLocalization: Localizing orbitals..." << std::endl;
    // For testing, just return the input orbitals
    return wavefunction;
  }
};

TEST_F(LocalizationTest, LocalizationSelector_MetaData) {
  auto selector = LocalizerFactory::create();
  EXPECT_NO_THROW({ auto settings = selector->settings(); });
}

TEST_F(LocalizationTest, Factory) {
  auto available_localizers = LocalizerFactory::available();
  EXPECT_EQ(available_localizers.size(), 3);
  EXPECT_TRUE(std::find(available_localizers.begin(),
                        available_localizers.end(),
                        "qdk_pipek_mezey") != available_localizers.end());
  EXPECT_TRUE(
      std::find(available_localizers.begin(), available_localizers.end(),
                "qdk_mp2_natural_orbitals") != available_localizers.end());
  EXPECT_TRUE(std::find(available_localizers.begin(),
                        available_localizers.end(),
                        "qdk_vvhv") != available_localizers.end());
  EXPECT_THROW(LocalizerFactory::create("nonexistent_localizer"),
               std::runtime_error);
  EXPECT_NO_THROW(LocalizerFactory::register_instance(
      []() -> LocalizerFactory::return_type {
        return std::make_unique<TestLocalization>();
      }));
  EXPECT_THROW(LocalizerFactory::register_instance(
                   []() -> LocalizerFactory::return_type {
                     return std::make_unique<TestLocalization>();
                   }),
               std::runtime_error);

  // Test unregister_instance
  // First test unregistering a non-existent key (should return false)
  EXPECT_FALSE(LocalizerFactory::unregister_instance("nonexistent_key"));

  // Test unregistering an existing key (should return true)
  EXPECT_TRUE(LocalizerFactory::unregister_instance("_dummy_localizer"));

  // Test unregistering the same key again (should return false since it's
  // already removed)
  EXPECT_FALSE(LocalizerFactory::unregister_instance("_dummy_localizer"));
}

TEST_F(LocalizationTest, WaterPipekMezey) {
  auto localizer = LocalizerFactory::create("qdk_pipek_mezey");
  EXPECT_NO_THROW({ auto settings = localizer->settings(); });

  // Get a canonical set of water orbitals
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(water, 0, 1, "def2-svp");
  auto orbitals = wfn->get_orbitals();
  const auto& S = orbitals->get_overlap_matrix();
  const auto& [Ca_can, Cb_can] = orbitals->get_coefficients();

  const size_t num_occupied_orbitals = wfn->get_total_num_electrons().first;

  // Test selected indices

  // Localize occupied and virtual orbitals separately
  std::vector<size_t> occ_indices, virt_indices;
  for (unsigned i = 0; i < num_occupied_orbitals; ++i) {
    occ_indices.push_back(i);
  }
  for (unsigned i = num_occupied_orbitals;
       i < orbitals->get_num_molecular_orbitals(); ++i) {
    virt_indices.push_back(i);
  }

  // Localize occupied orbitals only
  auto wfn_occ_ptr = localizer->run(wfn, occ_indices, occ_indices);
  // Localize virtual orbitals only
  auto wfn_virt_ptr = localizer->run(wfn_occ_ptr, virt_indices, virt_indices);
  const auto& [Ca_loc_virt, Cb_loc_virt] =
      wfn_virt_ptr->get_orbitals()->get_coefficients();

  // Check pipek_mezey_metric (WaterPipekMezey)
  EXPECT_NEAR(
      2.537885187700e+01,
      testing::pipek_mezey_metric(*wfn_virt_ptr->get_orbitals(), Ca_loc_virt),
      testing::localization_tolerance * 10);

  // Randomly choose indices to localize, then the transformation
  // should be unitary and localized orbitals should be orthonormal
  std::vector<size_t> random_indices = {1, 3, 4};  // Random subset of orbitals
  auto localized_random_ptr =
      localizer->run(wfn, random_indices, random_indices);
  const auto& [Ca_loc_rand, Cb_loc_rand] =
      localized_random_ptr->get_orbitals()->get_coefficients();

  auto pm_metric_random = testing::pipek_mezey_metric(
      *localized_random_ptr->get_orbitals(), Ca_loc_rand);
  EXPECT_NEAR(1.562735085e+01, pm_metric_random,
              testing::localization_tolerance * 10);

  // Extract the submatrix for the localized indices
  Eigen::MatrixXd Ca_selected(Ca_can.rows(), random_indices.size());
  Eigen::MatrixXd Ca_loc_selected(Ca_loc_rand.rows(), random_indices.size());
  for (size_t i = 0; i < random_indices.size(); ++i) {
    Ca_selected.col(i) = Ca_can.col(random_indices[i]);
    Ca_loc_selected.col(i) = Ca_loc_rand.col(random_indices[i]);
  }

  // Check that the transformation for selected indices is unitary
  Eigen::MatrixXd U_selected = Ca_selected.transpose() * S * Ca_loc_selected;
  EXPECT_NEAR(0.0, testing::norm_diff_from_unitary(U_selected),
              testing::numerical_zero_tolerance * 10);

  // Check that localized orbitals are orthonormal
  Eigen::MatrixXd overlap_check =
      Ca_loc_selected.transpose() * S * Ca_loc_selected;
  EXPECT_NEAR(0.0,
              (overlap_check - Eigen::MatrixXd::Identity(random_indices.size(),
                                                         random_indices.size()))
                  .norm(),
              testing::numerical_zero_tolerance * 10);
}

TEST_F(LocalizationTest, O2TripletPipekMezey) {
  auto localizer = LocalizerFactory::create("qdk_pipek_mezey");
  EXPECT_NO_THROW({ auto settings = localizer->settings(); });

  // Get a canonical set of o2 orbitals
  auto o2 = testing::create_o2_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(o2, 0, 3, "def2-svp");
  auto orbitals = wfn->get_orbitals();
  const auto& [Ca_can, Cb_can] = orbitals->get_coefficients();
  const auto& S = orbitals->get_overlap_matrix();

  // Get dimensions
  const auto [_na, _nb] = wfn->get_total_num_electrons();
  const size_t num_alpha = std::round(_na);
  const size_t num_beta = std::round(_nb);

  // Test selected indices

  // Localize occupied and virtual orbitals separately for each spin
  std::vector<size_t> occ_indices_alpha, virt_indices_alpha;
  std::vector<size_t> occ_indices_beta, virt_indices_beta;

  for (unsigned i = 0; i < num_alpha; ++i) {
    occ_indices_alpha.push_back(i);
  }
  for (unsigned i = num_alpha; i < orbitals->get_num_molecular_orbitals();
       ++i) {
    virt_indices_alpha.push_back(i);
  }
  for (unsigned i = 0; i < num_beta; ++i) {
    occ_indices_beta.push_back(i);
  }
  for (unsigned i = num_beta; i < orbitals->get_num_molecular_orbitals(); ++i) {
    virt_indices_beta.push_back(i);
  }

  // Localize occupied orbitals only
  auto localized_occ_ptr =
      localizer->run(wfn, occ_indices_alpha, occ_indices_beta);
  const auto& [Ca_loc_occ, Cb_loc_occ] =
      localized_occ_ptr->get_orbitals()->get_coefficients();

  // Localize virtual orbitals only
  auto localized_virt_ptr =
      localizer->run(localized_occ_ptr, virt_indices_alpha, virt_indices_beta);
  const auto& [Ca_loc_virt, Cb_loc_virt] =
      localized_virt_ptr->get_orbitals()->get_coefficients();

  // Check pipek_mezey_metric (O2TripletPipekMezey)
  EXPECT_NEAR(32.865411837,
              testing::pipek_mezey_metric(*localized_virt_ptr->get_orbitals(),
                                          Ca_loc_virt),
              testing::localization_tolerance * 10);
  EXPECT_NEAR(30.177455521,
              testing::pipek_mezey_metric(*localized_virt_ptr->get_orbitals(),
                                          Cb_loc_virt),
              testing::localization_tolerance * 10);

  // Randomly choose indices to localize, then the transformation
  // should be unitary and localized orbitals should be orthonormal

  // Random subset (with consideration of degeneracy)
  std::vector<size_t> random_indices_alpha = {1, 3, 6, 7, 8};
  std::vector<size_t> random_indices_beta = {0, 2, 4};
  auto localized_random_ptr =
      localizer->run(wfn, random_indices_alpha, random_indices_beta);
  const auto& [Ca_loc_rand, Cb_loc_rand] =
      localized_random_ptr->get_orbitals()->get_coefficients();

  auto pm_metric_random_alpha = testing::pipek_mezey_metric(
      *localized_random_ptr->get_orbitals(), Ca_loc_rand);
  auto pm_metric_random_beta = testing::pipek_mezey_metric(
      *localized_random_ptr->get_orbitals(), Cb_loc_rand);
  EXPECT_NEAR(14.480272, pm_metric_random_alpha,
              testing::localization_tolerance * 10);
  EXPECT_NEAR(14.000000, pm_metric_random_beta,
              testing::localization_tolerance * 10);

  // Extract the submatrix for the localized indices - alpha
  Eigen::MatrixXd Ca_selected(Ca_can.rows(), random_indices_alpha.size());
  Eigen::MatrixXd Ca_loc_selected(Ca_loc_rand.rows(),
                                  random_indices_alpha.size());
  for (size_t i = 0; i < random_indices_alpha.size(); ++i) {
    Ca_selected.col(i) = Ca_can.col(random_indices_alpha[i]);
    Ca_loc_selected.col(i) = Ca_loc_rand.col(random_indices_alpha[i]);
  }

  // Check that the transformation for selected indices is unitary - alpha
  Eigen::MatrixXd U_selected_alpha =
      Ca_selected.transpose() * S * Ca_loc_selected;
  EXPECT_NEAR(0.0, testing::norm_diff_from_unitary(U_selected_alpha),
              testing::numerical_zero_tolerance * 10);

  // Check that localized orbitals are orthonormal - alpha
  Eigen::MatrixXd overlap_check_alpha =
      Ca_loc_selected.transpose() * S * Ca_loc_selected;
  EXPECT_NEAR(0.0,
              (overlap_check_alpha -
               Eigen::MatrixXd::Identity(random_indices_alpha.size(),
                                         random_indices_alpha.size()))
                  .norm(),
              testing::numerical_zero_tolerance * 10);

  // Extract the submatrix for the localized indices - beta
  Eigen::MatrixXd Cb_selected(Cb_can.rows(), random_indices_beta.size());
  Eigen::MatrixXd Cb_loc_selected(Cb_loc_rand.rows(),
                                  random_indices_beta.size());
  for (size_t i = 0; i < random_indices_beta.size(); ++i) {
    Cb_selected.col(i) = Cb_can.col(random_indices_beta[i]);
    Cb_loc_selected.col(i) = Cb_loc_rand.col(random_indices_beta[i]);
  }

  // Check that the transformation for selected indices is unitary - beta
  Eigen::MatrixXd U_selected_beta =
      Cb_selected.transpose() * S * Cb_loc_selected;
  EXPECT_NEAR(0.0, testing::norm_diff_from_unitary(U_selected_beta),
              testing::numerical_zero_tolerance * 10);

  // Check that localized orbitals are orthonormal - beta
  Eigen::MatrixXd overlap_check_beta =
      Cb_loc_selected.transpose() * S * Cb_loc_selected;
  EXPECT_NEAR(0.0,
              (overlap_check_beta -
               Eigen::MatrixXd::Identity(random_indices_beta.size(),
                                         random_indices_beta.size()))
                  .norm(),
              testing::numerical_zero_tolerance * 10);
}

TEST_F(LocalizationTest, Iterative_EdgeCase) {
  // Test that Pipek-Mezey throws std::invalid_argument when loc_indices_a !=
  // loc_indices_b for restricted orbitals
  std::vector<size_t> indices_a({0, 1, 2});
  std::vector<size_t> indices_b({0, 1, 3});
  EXPECT_THROW(
      {
        auto localizer = LocalizerFactory::create("qdk_pipek_mezey");
        Eigen::MatrixXd coeffs(4, 4);
        coeffs.setIdentity();
        // Create a fake basis set with 4 atomic orbitals for testing
        auto fake_basis_set = testing::create_random_basis_set(4, "test");
        // Create a fake AO overlap matrix (4x4 identity matrix for simplicity)
        Eigen::MatrixXd fake_ao_overlap = Eigen::MatrixXd::Identity(4, 4);
        // Create restricted orbitals (only alpha coefficients provided)
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        // Provide different indices for alpha and beta to trigger the exception
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals));
        localizer->run(wfn, indices_a, indices_b);
      },
      std::invalid_argument);

  // Test that Pipek-Mezey throws std::invalid_argument when indices are not
  // sorted
  std::vector<size_t> unsorted_indices({2, 0, 1});  // Not sorted
  std::vector<size_t> sorted_indices({0, 1, 2});
  EXPECT_THROW(
      {
        auto localizer = LocalizerFactory::create("qdk_pipek_mezey");
        Eigen::MatrixXd coeffs(4, 4);
        coeffs.setIdentity();
        auto fake_basis_set = testing::create_random_basis_set(4, "test");
        Eigen::MatrixXd fake_ao_overlap = Eigen::MatrixXd::Identity(4, 4);
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals));
        localizer->run(wfn, unsorted_indices, unsorted_indices);
      },
      std::invalid_argument);
}

TEST_F(LocalizationTest, MP2) {
  auto localizer = LocalizerFactory::create("qdk_mp2_natural_orbitals");
  EXPECT_NO_THROW({ auto settings = localizer->settings(); });

  // Get a canonical set of water orbitals
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(water, 0, 1, "sto-3g");
  auto orbitals = wfn->get_orbitals();

  // // Use the orbitals directly for MP2 calculation
  std::shared_ptr<Wavefunction> mp2_orbitals_wfn_ptr;
  auto all_indices = orbitals->get_all_mo_indices();
  EXPECT_NO_THROW({
    mp2_orbitals_wfn_ptr = localizer->run(wfn, all_indices, all_indices);
  });
  auto& mp2_orbitals = *mp2_orbitals_wfn_ptr->get_orbitals();
  // Dimension checks
  EXPECT_EQ(mp2_orbitals.get_coefficients().first.rows(),
            orbitals->get_coefficients().first.rows());
  EXPECT_EQ(mp2_orbitals.get_coefficients().second.rows(),
            orbitals->get_coefficients().second.rows());

  // Test selected indices
  const auto& S = orbitals->get_overlap_matrix();
  const auto& [Ca_can, Cb_can] = orbitals->get_coefficients();
  const auto& [Ca_mp2, Cb_mp2] = mp2_orbitals.get_coefficients();
  const size_t num_occupied_orbitals = wfn->get_total_num_electrons().first;

  // Randomly choose indices to localize, then the transformation
  // should be unitary and localized orbitals should be orthonormal
  std::vector<size_t> random_indices = {0, 2, 4, 6};  // Random subset
  auto mp2_random_ptr = localizer->run(wfn, random_indices, random_indices);
  const auto& [Ca_mp2_rand, Cb_mp2_rand] =
      mp2_random_ptr->get_orbitals()->get_coefficients();

  // Extract the submatrix for the localized indices
  Eigen::MatrixXd Ca_selected(Ca_can.rows(), random_indices.size());
  Eigen::MatrixXd Ca_mp2_selected(Ca_mp2_rand.rows(), random_indices.size());
  for (size_t i = 0; i < random_indices.size(); ++i) {
    Ca_selected.col(i) = Ca_can.col(random_indices[i]);
    Ca_mp2_selected.col(i) = Ca_mp2_rand.col(random_indices[i]);
  }

  // Check that the transformation for selected indices is unitary
  Eigen::MatrixXd U_selected = Ca_selected.transpose() * S * Ca_mp2_selected;
  EXPECT_NEAR(0.0, testing::norm_diff_from_unitary(U_selected),
              testing::numerical_zero_tolerance * 10);

  // Check that MP2 natural orbitals are orthonormal
  Eigen::MatrixXd overlap_check =
      Ca_mp2_selected.transpose() * S * Ca_mp2_selected;
  EXPECT_NEAR(0.0,
              (overlap_check - Eigen::MatrixXd::Identity(random_indices.size(),
                                                         random_indices.size()))
                  .norm(),
              testing::numerical_zero_tolerance * 10);
}

TEST_F(LocalizationTest, MP2_EdgeCase) {
  // Common variables used across tests
  Eigen::MatrixXd coeffs(4, 4);
  coeffs.setIdentity();
  Eigen::MatrixXd coeffs_beta(4, 4);
  coeffs_beta.setIdentity();
  coeffs_beta(0, 0) = 0.5;  // Make beta coefficients different from alpha
  Eigen::VectorXd fake_energies = Eigen::VectorXd::Zero(4);
  auto fake_basis_set = testing::create_random_basis_set(4, "test");
  Eigen::MatrixXd fake_ao_overlap = Eigen::MatrixXd::Identity(4, 4);

  // Throw on missing orbital energies (non-canonical orbitals)
  EXPECT_THROW(
      {
        auto localizer = LocalizerFactory::create("qdk_mp2_natural_orbitals");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals));
        auto all_indices = orbitals->get_all_mo_indices();
        localizer->run(wfn, all_indices, all_indices);
      },
      std::invalid_argument);

  // Throw on unrestricted orbitals
  EXPECT_THROW(
      {
        auto localizer = LocalizerFactory::create("qdk_mp2_natural_orbitals");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, coeffs_beta, fake_energies, fake_energies,
            std::make_optional(fake_ao_overlap), fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2000"),
                                                         orbitals));
        auto all_indices = orbitals->get_all_mo_indices();
        localizer->run(wfn, all_indices, all_indices);
      },
      std::invalid_argument);

  // Throw on no occupations
  EXPECT_THROW(
      {
        auto localizer = LocalizerFactory::create("qdk_mp2_natural_orbitals");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, fake_energies, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("0000"),
                                                         orbitals));
        auto all_indices = orbitals->get_all_mo_indices();
        localizer->run(wfn, all_indices, all_indices);
      },
      std::invalid_argument);

  // Throw on open shell
  EXPECT_THROW(
      {
        auto localizer = LocalizerFactory::create("qdk_mp2_natural_orbitals");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, fake_energies, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2u00"),
                                                         orbitals));
        auto all_indices = orbitals->get_all_mo_indices();
        localizer->run(wfn, all_indices, all_indices);
      },
      std::invalid_argument);

  // Test that MP2 Natural Orbital localizer throws std::invalid_argument when
  // loc_indices_a != loc_indices_b for restricted orbitals
  std::vector<size_t> indices_a({0, 1});
  std::vector<size_t> indices_b({1, 2});  // Different from indices_a
  EXPECT_THROW(
      {
        auto localizer = LocalizerFactory::create("qdk_mp2_natural_orbitals");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, fake_energies, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals));
        localizer->run(wfn, indices_a, indices_b);
      },
      std::invalid_argument);

  // Test that MP2 Natural Orbital localizer throws std::invalid_argument when
  // indices are not sorted
  std::vector<size_t> unsorted_indices({2, 0, 1});  // Not sorted
  EXPECT_THROW(
      {
        auto localizer = LocalizerFactory::create("qdk_mp2_natural_orbitals");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, fake_energies, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals));
        localizer->run(wfn, unsorted_indices, unsorted_indices);
      },
      std::invalid_argument);
}

TEST_F(LocalizationTest, VVHV_EdgeCase) {
  // Common variables used across tests
  Eigen::MatrixXd coeffs(4, 4);
  coeffs.setIdentity();
  auto fake_basis_set = testing::create_random_basis_set(4, "test");
  Eigen::MatrixXd fake_ao_overlap = Eigen::MatrixXd::Identity(4, 4);

  // VVHV throws std::invalid_argument when not all alpha orbital
  // indices are covered
  std::vector<size_t> incomplete_indices_a({0, 1, 2});  // Missing orbital 3
  std::vector<size_t> all_indices_b({0, 1, 2, 3});
  EXPECT_THROW(
      {
        auto vvhv_localizer = LocalizerFactory::create("qdk_vvhv");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals));
        vvhv_localizer->run(wfn, incomplete_indices_a, all_indices_b);
      },
      std::invalid_argument);

  // VVHV throws std::invalid_argument when not all beta orbital
  // indices are covered for unrestricted case
  std::vector<size_t> all_indices_a({0, 1, 2, 3});
  std::vector<size_t> incomplete_indices_b({0, 1, 2});  // Missing orbital 3
  EXPECT_THROW(
      {
        auto vvhv_localizer = LocalizerFactory::create("qdk_vvhv");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, coeffs, std::nullopt, std::nullopt,
            std::make_optional(fake_ao_overlap), fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals));
        vvhv_localizer->run(wfn, all_indices_a, incomplete_indices_b);
      },
      std::invalid_argument);

  // VVHV throws std::invalid_argument when loc_indices_a !=
  // loc_indices_b for restricted orbitals
  std::vector<size_t> indices_a({0, 1, 2, 3});
  std::vector<size_t> indices_b(
      {0, 1, 3, 2});  // Different order from indices_a
  EXPECT_THROW(
      {
        auto vvhv_localizer = LocalizerFactory::create("qdk_vvhv");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals));
        vvhv_localizer->run(wfn, indices_a, indices_b);
      },
      std::invalid_argument);

  // VVHV throws std::invalid_argument when indices are not sorted
  std::vector<size_t> unsorted_indices({3, 1, 0, 2});  // Not sorted
  std::vector<size_t> sorted_indices({0, 1, 2, 3});
  EXPECT_THROW(
      {
        auto vvhv_localizer = LocalizerFactory::create("qdk_vvhv");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals));
        vvhv_localizer->run(wfn, unsorted_indices, unsorted_indices);
      },
      std::invalid_argument);

  // VVHV throws std::invalid_argument when indices contain
  // out-of-bounds values
  std::vector<size_t> out_of_bounds_indices(
      {0, 1, 2, 4});  // Index 4 is out of bounds for 4x4 matrix
  std::vector<size_t> valid_indices({0, 1, 2, 3});
  EXPECT_THROW(
      {
        auto vvhv_localizer = LocalizerFactory::create("qdk_vvhv");
        auto orbitals = std::make_shared<Orbitals>(
            coeffs, std::nullopt, std::make_optional(fake_ao_overlap),
            fake_basis_set, std::nullopt);
        auto wfn = std::make_shared<Wavefunction>(
            std::make_unique<SlaterDeterminantContainer>(Configuration("2200"),
                                                         orbitals));
        vvhv_localizer->run(wfn, out_of_bounds_indices, valid_indices);
      },
      std::invalid_argument);
}

TEST_F(LocalizationTest, WaterVVHV) {
  auto vvhv_localizer = LocalizerFactory::create("qdk_vvhv");
  auto pm_localizer = LocalizerFactory::create("qdk_pipek_mezey");
  EXPECT_NO_THROW({ auto settings = vvhv_localizer->settings(); });

  // Set the minimal basis to lowercase as required by VVHV
  vvhv_localizer->settings().set("minimal_basis", "sto-3g");
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("method", "hf");
  auto [E, wfn] = scf_solver->run(water, 0, 1, "def2-svp");
  auto orbitals = wfn->get_orbitals();
  const auto& [Ca_can, Cb_can] = orbitals->get_coefficients();

  const size_t num_occupied_orbitals = wfn->get_total_num_electrons().first;

  // First localize occupied orbitals with Pipek-Mezey
  std::vector<size_t> occ_indices, virt_indices;
  for (unsigned i = 0; i < num_occupied_orbitals; ++i) {
    occ_indices.push_back(i);
  }
  for (unsigned i = num_occupied_orbitals;
       i < orbitals->get_num_molecular_orbitals(); ++i) {
    virt_indices.push_back(i);
  }

  auto localized_occ_ptr = pm_localizer->run(wfn, occ_indices, occ_indices);

  // Then pass all orbitals (localized occupied + canonical virtual) to VVHV
  auto localized_wfn_ptr =
      vvhv_localizer->run(localized_occ_ptr, virt_indices, virt_indices);
  auto& localized_orbitals = *localized_wfn_ptr->get_orbitals();

  // Simple checks
  const auto& [Ca_loc, Cb_loc] = localized_orbitals.get_coefficients();
  EXPECT_EQ(Ca_loc.rows(), Ca_can.rows());
  EXPECT_EQ(Cb_loc.rows(), Cb_can.rows());

  auto pm_metric = testing::pipek_mezey_metric(localized_orbitals, Ca_loc);
  EXPECT_NEAR(2.361763653e+01, pm_metric, testing::localization_tolerance);
}

TEST_F(LocalizationTest, O2TripletVVHV) {
  auto vvhv_localizer = LocalizerFactory::create("qdk_vvhv");
  auto pm_localizer = LocalizerFactory::create("qdk_pipek_mezey");
  EXPECT_NO_THROW({ auto settings = vvhv_localizer->settings(); });

  // Set the minimal basis to lowercase as required by VVHV
  vvhv_localizer->settings().set("minimal_basis", "sto-3g");

  // Get a canonical set of o2 orbitals
  auto o2 = testing::create_o2_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(o2, 0, 3, "def2-svp");
  auto orbitals = wfn->get_orbitals();
  const auto& [Ca_can, Cb_can] = orbitals->get_coefficients();

  // Get dimensions
  const auto [_na, _nb] = wfn->get_total_num_electrons();
  const size_t num_alpha = std::round(_na);
  const size_t num_beta = std::round(_nb);

  // First localize occupied orbitals with Pipek-Mezey
  std::vector<size_t> occ_indices_alpha, virt_indices_alpha;
  std::vector<size_t> occ_indices_beta, virt_indices_beta;

  for (unsigned i = 0; i < num_alpha; ++i) {
    occ_indices_alpha.push_back(i);
  }
  for (unsigned i = num_alpha; i < orbitals->get_num_molecular_orbitals();
       ++i) {
    virt_indices_alpha.push_back(i);
  }
  for (unsigned i = 0; i < num_beta; ++i) {
    occ_indices_beta.push_back(i);
  }
  for (unsigned i = num_beta; i < orbitals->get_num_molecular_orbitals(); ++i) {
    virt_indices_beta.push_back(i);
  }

  auto localized_occ_ptr =
      pm_localizer->run(wfn, occ_indices_alpha, occ_indices_beta);

  // Then pass all orbitals (localized occupied + canonical virtual) to VVHV
  auto localized_orbitals_ptr = vvhv_localizer->run(
      localized_occ_ptr, virt_indices_alpha, virt_indices_beta);
  auto& localized_orbitals = *localized_orbitals_ptr->get_orbitals();

  // Simple checks
  const auto& [Ca_loc, Cb_loc] = localized_orbitals.get_coefficients();
  EXPECT_EQ(Ca_loc.rows(), Ca_can.rows());
  EXPECT_EQ(Cb_loc.rows(), Cb_can.rows());

  auto pm_metric_alpha =
      testing::pipek_mezey_metric(localized_orbitals, Ca_loc);
  auto pm_metric_beta = testing::pipek_mezey_metric(localized_orbitals, Cb_loc);

  // Check metric values to reference (commented out temporarily)
  EXPECT_NEAR(3.148110465e+01, pm_metric_alpha,
              testing::localization_tolerance);
  EXPECT_NEAR(2.862067715e+01, pm_metric_beta, testing::localization_tolerance);
}

// =============================================================================
// Tests for active space preservation after localization
// Regression tests for bug: active space indices lost after orbital
// localization
// =============================================================================

// Helper function to verify active space preservation after localization
void verify_active_space_preserved(
    std::shared_ptr<qdk::chemistry::data::Wavefunction> wfn_before,
    std::shared_ptr<qdk::chemistry::data::Wavefunction> wfn_after,
    const std::string& localizer_name) {
  auto orbitals_before = wfn_before->get_orbitals();
  auto orbitals_after = wfn_after->get_orbitals();

  ASSERT_TRUE(orbitals_before->has_active_space());
  ASSERT_TRUE(orbitals_after->has_active_space())
      << "Active space lost after " << localizer_name << " localization";

  auto [active_alpha_before, active_beta_before] =
      orbitals_before->get_active_space_indices();
  auto [active_alpha_after, active_beta_after] =
      orbitals_after->get_active_space_indices();

  EXPECT_EQ(active_alpha_before, active_alpha_after)
      << localizer_name << ": alpha indices changed";
  EXPECT_EQ(active_beta_before, active_beta_after)
      << localizer_name << ": beta indices changed";
}

// Parameterized test for active space preservation
class ActiveSpacePreservationTest
    : public ::testing::TestWithParam<std::string> {};

TEST_P(ActiveSpacePreservationTest, PreservesActiveSpaceRestricted) {
  const std::string localizer_name = GetParam();

  // Setup: Water molecule with active space
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(water, 0, 1, "sto-3g");

  // Select an active space
  auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
  selector->settings().set("num_active_electrons", 6);
  selector->settings().set("num_active_orbitals", 5);
  auto active_wfn = selector->run(wfn);

  auto [active_alpha, active_beta] =
      active_wfn->get_orbitals()->get_active_space_indices();

  // Localize
  auto localizer = LocalizerFactory::create(localizer_name);
  auto localized_wfn = localizer->run(active_wfn, active_alpha, active_beta);

  verify_active_space_preserved(active_wfn, localized_wfn, localizer_name);
}

INSTANTIATE_TEST_SUITE_P(Localizers, ActiveSpacePreservationTest,
                         ::testing::Values("qdk_pipek_mezey",
                                           "qdk_mp2_natural_orbitals"),
                         [](const ::testing::TestParamInfo<std::string>& info) {
                           // Convert localizer name to valid test name (replace
                           // non-alphanumeric)
                           std::string name = info.param;
                           std::replace(name.begin(), name.end(), '_', ' ');
                           name.erase(0, name.find_first_not_of(' '));
                           std::replace(name.begin(), name.end(), ' ', '_');
                           return name;
                         });

TEST_F(LocalizationTest, PipekMezeyPreservesActiveSpaceUnrestricted) {
  // Setup: Water with unrestricted orbitals and active space
  // (Closed-shell but exercises the unrestricted code path)
  auto h2o = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", "unrestricted");
  auto [E, wfn] = scf_solver->run(h2o, 0, 1, "sto-3g");

  // Manually set active space indices (ValenceActiveSpaceSelector doesn't
  // support UHF)
  auto orbitals = wfn->get_orbitals();
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();

  // Define active space: frozen core (first 2 are inactive), rest are active
  // Must include all occupied orbitals in active space for
  // SlaterDeterminantContainer
  std::vector<size_t> active_alpha, active_beta, inactive_alpha, inactive_beta;

  // First 2 are inactive (core)
  inactive_alpha = {0, 1};
  inactive_beta = {0, 1};
  // Rest of orbitals are active
  for (size_t i = 2; i < num_molecular_orbitals; ++i) {
    active_alpha.push_back(i);
    active_beta.push_back(i);
  }

  // Create orbitals with active space
  auto active_orbitals = std::make_shared<Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_coefficients().second,
      std::nullopt, std::nullopt, orbitals->get_overlap_matrix(),
      orbitals->get_basis_set(),
      std::make_tuple(active_alpha, active_beta, inactive_alpha,
                      inactive_beta));

  auto active_wfn = std::make_shared<Wavefunction>(
      std::make_unique<SlaterDeterminantContainer>(
          wfn->get_active_determinants()[0], active_orbitals));

  // Localize only the active orbitals
  auto localizer = LocalizerFactory::create("qdk_pipek_mezey");
  auto localized_wfn = localizer->run(active_wfn, active_alpha, active_beta);

  verify_active_space_preserved(active_wfn, localized_wfn,
                                "qdk_pipek_mezey_unrestricted");
}

TEST_F(LocalizationTest, VVHVPreservesActiveSpace) {
  // Setup: Water molecule with active space (needs def2-svp for VVHV)
  auto water = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(water, 0, 1, "def2-svp");

  // Select an active space
  auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
  selector->settings().set("num_active_electrons", 6);
  selector->settings().set("num_active_orbitals", 10);
  auto active_wfn = selector->run(wfn);

  // Get virtual indices for VVHV
  const auto [na, nb] = wfn->get_total_num_electrons();
  const size_t num_occupied = std::round(na);
  std::vector<size_t> virt_indices;
  for (size_t i = num_occupied;
       i < active_wfn->get_orbitals()->get_num_molecular_orbitals(); ++i) {
    virt_indices.push_back(i);
  }

  // Localize virtual orbitals with VVHV
  auto localizer = LocalizerFactory::create("qdk_vvhv");
  localizer->settings().set("minimal_basis", "sto-3g");
  auto localized_wfn = localizer->run(active_wfn, virt_indices, virt_indices);

  verify_active_space_preserved(active_wfn, localized_wfn, "qdk_vvhv");
}

TEST_F(LocalizationTest, VVHVPreservesActiveSpaceUnrestricted) {
  // Setup: Water with unrestricted orbitals and active space
  // (Boring since water is closed-shell, but tests the unrestricted code path)
  auto h2o = testing::create_water_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", "unrestricted");
  auto [E, wfn] = scf_solver->run(h2o, 0, 1, "def2-svp");

  // Manually set active space indices (ValenceActiveSpaceSelector doesn't
  // support UHF)
  auto orbitals = wfn->get_orbitals();
  const size_t num_molecular_orbitals = orbitals->get_num_molecular_orbitals();

  // Define active space: frozen core (first 2 occupied are inactive),
  // rest of occupied + all virtuals are active.
  // VVHV requires all virtual indices, so we can only have inactive occupied.
  std::vector<size_t> active_alpha, active_beta, inactive_alpha, inactive_beta;

  // First 2 occupied are inactive (core)
  for (size_t i = 0; i < 2; ++i) {
    inactive_alpha.push_back(i);
    inactive_beta.push_back(i);
  }
  // Rest of orbitals are active
  for (size_t i = 2; i < num_molecular_orbitals; ++i) {
    active_alpha.push_back(i);
    active_beta.push_back(i);
  }

  // Create orbitals with active space
  auto active_orbitals = std::make_shared<Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_coefficients().second,
      std::nullopt, std::nullopt, orbitals->get_overlap_matrix(),
      orbitals->get_basis_set(),
      std::make_tuple(active_alpha, active_beta, inactive_alpha,
                      inactive_beta));

  auto active_wfn = std::make_shared<Wavefunction>(
      std::make_unique<SlaterDeterminantContainer>(
          wfn->get_active_determinants()[0], active_orbitals));

  // Get virtual indices for VVHV
  const auto [na, nb] = active_wfn->get_total_num_electrons();
  const size_t num_alpha = std::round(na);
  const size_t num_beta = std::round(nb);

  std::vector<size_t> virt_alpha, virt_beta;
  for (size_t i = num_alpha; i < num_molecular_orbitals; ++i) {
    virt_alpha.push_back(i);
  }
  for (size_t i = num_beta; i < num_molecular_orbitals; ++i) {
    virt_beta.push_back(i);
  }

  // Localize virtual orbitals with VVHV
  auto localizer = LocalizerFactory::create("qdk_vvhv");
  localizer->settings().set("minimal_basis", "sto-3g");
  auto localized_wfn = localizer->run(active_wfn, virt_alpha, virt_beta);

  verify_active_space_preserved(active_wfn, localized_wfn,
                                "qdk_vvhv_unrestricted");
}

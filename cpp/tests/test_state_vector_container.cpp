// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class StateVectorContainerTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(StateVectorContainerTest, BasicProperties) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration::from_spin_half_string("2200"),
      Configuration::from_spin_half_string("2020"),
      Configuration::from_spin_half_string("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  StateVectorContainer sv(coeffs, dets, orbitals, one_rdm, std::nullopt);

  EXPECT_EQ(sv.size(), 3);
  EXPECT_DOUBLE_EQ(std::get<double>(sv.get_coefficient(
                       Configuration::from_spin_half_string("2200"))),
                   0.5);
  EXPECT_DOUBLE_EQ(std::get<double>(sv.get_coefficient(
                       Configuration::from_spin_half_string("2020"))),
                   0.5);
  EXPECT_DOUBLE_EQ(std::get<double>(sv.get_coefficient(
                       Configuration::from_spin_half_string("2002"))),
                   1.0 / sqrt(2));
  EXPECT_DOUBLE_EQ(std::get<double>(sv.get_coefficient(
                       Configuration::from_spin_half_string("2000"))),
                   0.0);
  EXPECT_EQ(sv.get_active_determinants().size(), 3);
  EXPECT_EQ(sv.get_active_determinants()[0].to_string(), "2200");
  EXPECT_EQ(sv.get_active_determinants()[1].to_string(), "2020");
  EXPECT_EQ(sv.get_active_determinants()[2].to_string(), "2002");
  EXPECT_EQ(sv.get_total_num_electrons().first, 2);
  EXPECT_EQ(sv.get_total_num_electrons().second, 2);

  auto [total_alpha_elec, total_beta_elec] = sv.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sv.get_active_num_electrons();
  EXPECT_EQ(total_alpha_elec, 2);
  EXPECT_EQ(total_beta_elec, 2);
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);
  EXPECT_EQ(total_alpha_elec, active_alpha_elec);
  EXPECT_EQ(total_beta_elec, active_beta_elec);

  auto [alpha_occ, beta_occ] = sv.get_total_orbital_occupations();
  EXPECT_EQ(alpha_occ.size(), 4);
  EXPECT_EQ(beta_occ.size(), 4);

  Eigen::VectorXd expected_alpha(4);
  expected_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_beta(4);
  expected_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(alpha_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_occ.isApprox(expected_beta, testing::wf_tolerance));

  auto [alpha_total_occ, beta_total_occ] = sv.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sv.get_active_orbital_occupations();
  EXPECT_EQ(alpha_total_occ.size(), 4);
  EXPECT_EQ(beta_total_occ.size(), 4);
  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);
  EXPECT_TRUE(alpha_total_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_total_occ.isApprox(expected_beta, testing::wf_tolerance));
  EXPECT_TRUE(alpha_active_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_active_occ.isApprox(expected_beta, testing::wf_tolerance));
  EXPECT_TRUE(
      alpha_total_occ.isApprox(alpha_active_occ, testing::wf_tolerance));
  EXPECT_TRUE(beta_total_occ.isApprox(beta_active_occ, testing::wf_tolerance));
}

TEST_F(StateVectorContainerTest, EmptyDeterminantsThrows) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> empty_dets;
  Eigen::VectorXd empty_coeffs(0);

  StateVectorContainer sv(empty_coeffs, empty_dets, orbitals);

  EXPECT_THROW(
      sv.get_coefficient(Configuration::from_spin_half_string("2200")),
      std::runtime_error);
  EXPECT_THROW(sv.get_total_num_electrons(), std::runtime_error);
  EXPECT_THROW(sv.get_active_num_electrons(), std::runtime_error);
  EXPECT_THROW(sv.get_total_orbital_occupations(), std::runtime_error);
  EXPECT_THROW(sv.get_active_orbital_occupations(), std::runtime_error);

  EXPECT_EQ(sv.size(), 0);
}

TEST_F(StateVectorContainerTest, ErrorMessagesAreDescriptive) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> empty_dets;
  Eigen::VectorXd empty_coeffs(0);

  StateVectorContainer sv(empty_coeffs, empty_dets, orbitals);

  try {
    sv.get_total_orbital_occupations();
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "No determinants available");
  }
}

TEST_F(StateVectorContainerTest, EntropyWithMissingRDMs) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration::from_spin_half_string("2200"),
      Configuration::from_spin_half_string("2020")};
  Eigen::VectorXd coeffs(2);
  coeffs << 1.0 / sqrt(2), 1.0 / sqrt(2);

  StateVectorContainer sv(coeffs, dets, orbitals);

  EXPECT_THROW(sv.get_single_orbital_entropies(), std::runtime_error);
}

TEST_F(StateVectorContainerTest, WithInactiveOrbitals) {
  auto base_orbitals = testing::create_test_orbitals(6, 6, true);
  std::vector<size_t> active_indices = {2, 3, 4, 5};
  std::vector<size_t> inactive_indices = {0, 1};
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  std::vector<Configuration> dets = {
      Configuration::from_spin_half_string("2200"),
      Configuration::from_spin_half_string("2020"),
      Configuration::from_spin_half_string("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  StateVectorContainer sv(coeffs, dets, orbitals, one_rdm, std::nullopt);

  auto [total_alpha_elec, total_beta_elec] = sv.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sv.get_active_num_electrons();

  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  EXPECT_EQ(total_alpha_elec, 4);
  EXPECT_EQ(total_beta_elec, 4);

  auto [alpha_total_occ, beta_total_occ] = sv.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sv.get_active_orbital_occupations();

  EXPECT_EQ(alpha_total_occ.size(), 6);
  EXPECT_EQ(beta_total_occ.size(), 6);

  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);

  Eigen::VectorXd expected_active_alpha(4);
  expected_active_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_active_beta(4);
  expected_active_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_active_occ.isApprox(expected_active_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_active_occ.isApprox(expected_active_beta, testing::wf_tolerance));

  Eigen::VectorXd expected_total_alpha(6);
  expected_total_alpha << 1.0, 1.0, 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_total_beta(6);
  expected_total_beta << 1.0, 1.0, 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_total_occ.isApprox(expected_total_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_total_occ.isApprox(expected_total_beta, testing::wf_tolerance));
}

TEST_F(StateVectorContainerTest, WithNonContinuousActiveSpace) {
  auto base_orbitals = testing::create_test_orbitals(8, 8, true);
  std::vector<size_t> active_indices = {2, 4, 6, 7};
  std::vector<size_t> inactive_indices = {0, 1};
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  std::vector<Configuration> dets = {
      Configuration::from_spin_half_string("2200"),
      Configuration::from_spin_half_string("2020"),
      Configuration::from_spin_half_string("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  StateVectorContainer sv(coeffs, dets, orbitals, one_rdm, std::nullopt);

  auto [total_alpha_elec, total_beta_elec] = sv.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sv.get_active_num_electrons();

  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  EXPECT_EQ(total_alpha_elec, 4);
  EXPECT_EQ(total_beta_elec, 4);

  auto [alpha_total_occ, beta_total_occ] = sv.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sv.get_active_orbital_occupations();

  EXPECT_EQ(alpha_total_occ.size(), 8);
  EXPECT_EQ(beta_total_occ.size(), 8);

  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);

  Eigen::VectorXd expected_active_alpha(4);
  expected_active_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_active_beta(4);
  expected_active_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_active_occ.isApprox(expected_active_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_active_occ.isApprox(expected_active_beta, testing::wf_tolerance));

  Eigen::VectorXd expected_total_alpha(8);
  expected_total_alpha << 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  Eigen::VectorXd expected_total_beta(8);
  expected_total_beta << 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_total_occ.isApprox(expected_total_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_total_occ.isApprox(expected_total_beta, testing::wf_tolerance));

  EXPECT_DOUBLE_EQ(alpha_total_occ(3), 0.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(5), 0.0);
  EXPECT_DOUBLE_EQ(beta_total_occ(3), 0.0);
  EXPECT_DOUBLE_EQ(beta_total_occ(5), 0.0);

  EXPECT_DOUBLE_EQ(alpha_total_occ(0), 1.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(1), 1.0);
  EXPECT_DOUBLE_EQ(beta_total_occ(0), 1.0);
  EXPECT_DOUBLE_EQ(beta_total_occ(1), 1.0);

  EXPECT_DOUBLE_EQ(alpha_total_occ(2), 1.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(4), 1.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(6), 0.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(7), 0.0);
}

TEST_F(StateVectorContainerTest, JsonSerialization) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration::from_spin_half_string("2200"),
      Configuration::from_spin_half_string("2020"),
      Configuration::from_spin_half_string("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1.0 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;

  StateVectorContainer original(coeffs, dets, orbitals, one_rdm, std::nullopt);

  nlohmann::json j = original.to_json();

  auto restored =
      std::unique_ptr<StateVectorContainer>(dynamic_cast<StateVectorContainer*>(
          WavefunctionContainer::from_json(j).release()));

  auto original_wf =
      std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
          coeffs, dets, orbitals, one_rdm, std::nullopt));
  nlohmann::json wf_j = original_wf->to_json();
  auto wf_restored = Wavefunction::from_json(wf_j);
  EXPECT_EQ(wf_restored->get_container_type(), "state_vector");
  auto& wf_restored_container =
      wf_restored->get_container<StateVectorContainer>();

  EXPECT_EQ(original.size(), restored->size());
  EXPECT_EQ(original.get_active_determinants().size(),
            restored->get_active_determinants().size());

  const auto& orig_coeffs =
      std::get<Eigen::VectorXd>(original.get_coefficients());
  const auto& rest_coeffs =
      std::get<Eigen::VectorXd>(restored->get_coefficients());
  EXPECT_TRUE(orig_coeffs.isApprox(rest_coeffs, testing::wf_tolerance));

  for (size_t i = 0; i < original.get_active_determinants().size(); ++i) {
    EXPECT_EQ(original.get_active_determinants()[i],
              restored->get_active_determinants()[i]);
  }

  EXPECT_EQ(restored->size(), wf_restored_container.size());
  const auto& wf_rest_coeffs =
      std::get<Eigen::VectorXd>(wf_restored_container.get_coefficients());
  EXPECT_TRUE(rest_coeffs.isApprox(wf_rest_coeffs, testing::wf_tolerance));
  for (size_t i = 0; i < restored->get_active_determinants().size(); ++i) {
    EXPECT_EQ(restored->get_active_determinants()[i],
              wf_restored_container.get_active_determinants()[i]);
  }
}

TEST_F(StateVectorContainerTest, Hdf5Serialization) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration::from_spin_half_string("2200"),
      Configuration::from_spin_half_string("2020"),
      Configuration::from_spin_half_string("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1.0 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;

  StateVectorContainer original(coeffs, dets, orbitals, one_rdm, std::nullopt);

  std::string filename = "test_state_vector_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    original.to_hdf5(root);

    auto restored = StateVectorContainer::from_hdf5(root);

    EXPECT_EQ(original.size(), restored->size());
    EXPECT_EQ(original.get_active_determinants().size(),
              restored->get_active_determinants().size());

    const auto& orig_coeffs =
        std::get<Eigen::VectorXd>(original.get_coefficients());
    const auto& rest_coeffs =
        std::get<Eigen::VectorXd>(restored->get_coefficients());
    EXPECT_TRUE(orig_coeffs.isApprox(rest_coeffs, testing::wf_tolerance));

    for (size_t i = 0; i < original.get_active_determinants().size(); ++i) {
      EXPECT_EQ(original.get_active_determinants()[i],
                restored->get_active_determinants()[i]);
    }

    file.close();
  }

  std::string wf_filename = "test_state_vector_wavefunction_serialization.h5";
  {
    auto original_wf =
        std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
            coeffs, dets, orbitals, one_rdm, std::nullopt));
    H5::H5File file(wf_filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original_wf->to_hdf5(root);
    file.close();
  }
  {
    H5::H5File file(wf_filename, H5F_ACC_RDONLY);
    H5::Group root = file.openGroup("/");
    auto wf_restored = Wavefunction::from_hdf5(root);
    EXPECT_EQ(wf_restored->get_container_type(), "state_vector");
    auto& wf_restored_container =
        wf_restored->get_container<StateVectorContainer>();

    H5::H5File file2(filename, H5F_ACC_RDONLY);
    H5::Group root2 = file2.openGroup("/");
    auto restored = StateVectorContainer::from_hdf5(root2);

    EXPECT_EQ(restored->size(), wf_restored_container.size());
    const auto& rest_coeffs =
        std::get<Eigen::VectorXd>(restored->get_coefficients());
    const auto& wf_rest_coeffs =
        std::get<Eigen::VectorXd>(wf_restored_container.get_coefficients());
    EXPECT_TRUE(rest_coeffs.isApprox(wf_rest_coeffs, testing::wf_tolerance));
    for (size_t i = 0; i < restored->get_active_determinants().size(); ++i) {
      EXPECT_EQ(restored->get_active_determinants()[i],
                wf_restored_container.get_active_determinants()[i]);
    }

    file.close();
    file2.close();
  }

  std::remove(filename.c_str());
  std::remove(wf_filename.c_str());
}

TEST_F(StateVectorContainerTest, Hdf5SerializationComplex) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration::from_spin_half_string("2200"),
      Configuration::from_spin_half_string("2020")};
  Eigen::VectorXcd coeffs(2);
  coeffs << std::complex<double>(0.5, 0.3), std::complex<double>(0.6, -0.2);

  StateVectorContainer original(coeffs, dets, orbitals);

  nlohmann::json j = original.to_json();
  auto restored_json =
      std::unique_ptr<StateVectorContainer>(dynamic_cast<StateVectorContainer*>(
          WavefunctionContainer::from_json(j).release()));

  const auto& orig_coeffs =
      std::get<Eigen::VectorXcd>(original.get_coefficients());
  const auto& rest_coeffs =
      std::get<Eigen::VectorXcd>(restored_json->get_coefficients());
  EXPECT_TRUE(orig_coeffs.isApprox(rest_coeffs, testing::wf_tolerance));

  std::string filename = "test_state_vector_complex_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original.to_hdf5(root);
    auto restored_hdf5 = StateVectorContainer::from_hdf5(root);

    const auto& rest_coeffs_h5 =
        std::get<Eigen::VectorXcd>(restored_hdf5->get_coefficients());
    EXPECT_TRUE(orig_coeffs.isApprox(rest_coeffs_h5, testing::wf_tolerance));

    file.close();
  }
  std::remove(filename.c_str());
}

TEST_F(StateVectorContainerTest, JsonSerializationRDMs) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration::from_spin_half_string("2200"),
      Configuration::from_spin_half_string("2020"),
      Configuration::from_spin_half_string("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1.0 / sqrt(2);

  Eigen::MatrixXd one_rdm_aa(4, 4);
  one_rdm_aa.setIdentity();
  one_rdm_aa *= 2.0;
  one_rdm_aa(2, 2) = 0.0;
  one_rdm_aa(3, 3) = 0.0;

  size_t two_rdm_size = 4 * 4 * 4 * 4;
  Eigen::VectorXd two_rdm_aabb(two_rdm_size);
  two_rdm_aabb.setOnes();
  two_rdm_aabb *= 0.5;

  Eigen::VectorXd two_rdm_aaaa(two_rdm_size);
  two_rdm_aaaa.setOnes();
  two_rdm_aaaa *= 0.25;

  StateVectorContainer original(coeffs, dets, orbitals, std::nullopt,
                                one_rdm_aa, one_rdm_aa, std::nullopt,
                                two_rdm_aaaa, two_rdm_aabb, two_rdm_aaaa);

  nlohmann::json j = original.to_json();

  EXPECT_TRUE(j.contains("rdms"));
  EXPECT_TRUE(j["rdms"].contains("active_one_rdm"));
  EXPECT_TRUE(j["rdms"].contains("active_two_rdm"));

  auto restored =
      std::unique_ptr<StateVectorContainer>(dynamic_cast<StateVectorContainer*>(
          WavefunctionContainer::from_json(j).release()));

  EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
  EXPECT_TRUE(restored->has_two_rdm_spin_dependent());

  auto [orig_one_aa, orig_one_bb] =
      original.get_active_one_rdm_spin_dependent();
  auto [rest_one_aa, rest_one_bb] =
      restored->get_active_one_rdm_spin_dependent();

  EXPECT_TRUE(std::get<Eigen::MatrixXd>(orig_one_aa)
                  .isApprox(std::get<Eigen::MatrixXd>(rest_one_aa),
                            testing::wf_tolerance));

  auto [orig_two_aaaa, orig_two_aabb, orig_two_bbbb] =
      original.get_active_two_rdm_spin_dependent();
  auto [rest_two_aaaa, rest_two_aabb, rest_two_bbbb] =
      restored->get_active_two_rdm_spin_dependent();

  EXPECT_TRUE(std::get<Eigen::VectorXd>(orig_two_aabb)
                  .isApprox(std::get<Eigen::VectorXd>(rest_two_aabb),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(orig_two_aaaa)
                  .isApprox(std::get<Eigen::VectorXd>(rest_two_aaaa),
                            testing::wf_tolerance));
}

TEST_F(StateVectorContainerTest, JsonSerializationRDMsOpenShell) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {1.4, 0., 0.}};
  std::vector<std::string> symbols = {"H", "H"};
  auto structure = std::make_shared<Structure>(coords, symbols);
  const char* basis_set = "def2-svp";

  auto scf_solver = ScfSolverFactory::create();
  auto [E_default, wfn_default] = scf_solver->run(structure, -1, 2, basis_set);

  auto orbitals = wfn_default->get_orbitals();
  auto restricted_orbitals = std::make_shared<Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_energies().first,
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      testing::restricted_index_set(orbitals->get_num_molecular_orbitals(),
                                    orbitals->get_active_space_indices().first),
      testing::restricted_index_set(
          orbitals->get_num_molecular_orbitals(),
          orbitals->get_inactive_space_indices().first));

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(restricted_orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create();
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  auto [E_mc, wfn_mc] = mc->run(H, 2, 1);

  const auto& original = wfn_mc->get_container<StateVectorContainer>();

  EXPECT_TRUE(original.has_one_rdm_spin_dependent());
  EXPECT_TRUE(original.has_one_rdm_spin_traced());
  EXPECT_TRUE(original.has_two_rdm_spin_dependent());
  EXPECT_TRUE(original.has_two_rdm_spin_traced());

  EXPECT_TRUE(original.get_orbitals()->is_restricted());

  nlohmann::json j = original.to_json();

  EXPECT_TRUE(j.contains("rdms"));
  EXPECT_TRUE(j["rdms"].contains("active_one_rdm"));
  EXPECT_TRUE(j["rdms"].contains("active_two_rdm"));

  auto restored =
      std::unique_ptr<StateVectorContainer>(dynamic_cast<StateVectorContainer*>(
          WavefunctionContainer::from_json(j).release()));

  EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
  EXPECT_TRUE(restored->has_one_rdm_spin_traced());
  EXPECT_TRUE(restored->has_two_rdm_spin_dependent());
  EXPECT_TRUE(restored->has_two_rdm_spin_traced());

  EXPECT_TRUE(restored->get_orbitals()->is_restricted());

  auto [restored_aa_rdm, restored_bb_rdm] =
      restored->get_active_one_rdm_spin_dependent();
  auto [original_aa_rdm, original_bb_rdm] =
      original.get_active_one_rdm_spin_dependent();

  const auto& restored_aa_rdm_r = std::get<Eigen::MatrixXd>(restored_aa_rdm);
  const auto& restored_bb_rdm_r = std::get<Eigen::MatrixXd>(restored_bb_rdm);
  const auto& original_aa_rdm_r = std::get<Eigen::MatrixXd>(original_aa_rdm);
  const auto& original_bb_rdm_r = std::get<Eigen::MatrixXd>(original_bb_rdm);

  EXPECT_TRUE(
      restored_aa_rdm_r.isApprox(original_aa_rdm_r, testing::rdm_tolerance));
  EXPECT_TRUE(
      restored_bb_rdm_r.isApprox(original_bb_rdm_r, testing::rdm_tolerance));

  EXPECT_FALSE(
      restored_aa_rdm_r.isApprox(restored_bb_rdm_r, testing::rdm_tolerance));

  auto restored_one_rdm = restored->get_active_one_rdm_spin_traced();
  auto original_one_rdm = original.get_active_one_rdm_spin_traced();

  const auto& restored_one_rdm_r = std::get<Eigen::MatrixXd>(restored_one_rdm);
  const auto& original_one_rdm_r = std::get<Eigen::MatrixXd>(original_one_rdm);

  EXPECT_TRUE(
      restored_one_rdm_r.isApprox(original_one_rdm_r, testing::rdm_tolerance));

  auto [restored_aaaa_rdm, restored_aabb_rdm, restored_bbbb_rdm] =
      restored->get_active_two_rdm_spin_dependent();
  auto [original_aaaa_rdm, original_aabb_rdm, original_bbbb_rdm] =
      original.get_active_two_rdm_spin_dependent();

  const auto& restored_aabb_rdm_r =
      std::get<Eigen::VectorXd>(restored_aabb_rdm);
  const auto& restored_aaaa_rdm_r =
      std::get<Eigen::VectorXd>(restored_aaaa_rdm);
  const auto& restored_bbbb_rdm_r =
      std::get<Eigen::VectorXd>(restored_bbbb_rdm);
  const auto& original_aabb_rdm_r =
      std::get<Eigen::VectorXd>(original_aabb_rdm);
  const auto& original_aaaa_rdm_r =
      std::get<Eigen::VectorXd>(original_aaaa_rdm);
  const auto& original_bbbb_rdm_r =
      std::get<Eigen::VectorXd>(original_bbbb_rdm);

  EXPECT_TRUE(restored_aabb_rdm_r.isApprox(original_aabb_rdm_r,
                                           testing::rdm_tolerance));
  EXPECT_TRUE(restored_aaaa_rdm_r.isApprox(original_aaaa_rdm_r,
                                           testing::rdm_tolerance));
  EXPECT_TRUE(restored_bbbb_rdm_r.isApprox(original_bbbb_rdm_r,
                                           testing::rdm_tolerance));

  EXPECT_FALSE(restored_aaaa_rdm_r.isApprox(restored_bbbb_rdm_r,
                                            testing::rdm_tolerance));

  auto restored_two_rdm = restored->get_active_two_rdm_spin_traced();
  auto original_two_rdm = original.get_active_two_rdm_spin_traced();

  const auto& restored_two_rdm_r = std::get<Eigen::VectorXd>(restored_two_rdm);
  const auto& original_two_rdm_r = std::get<Eigen::VectorXd>(original_two_rdm);
  EXPECT_TRUE(
      restored_two_rdm_r.isApprox(original_two_rdm_r, testing::rdm_tolerance));
}

TEST_F(StateVectorContainerTest, Hdf5SerializationRDMs) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {1.4, 0., 0.}};
  std::vector<std::string> symbols = {"H", "H"};
  auto structure = std::make_shared<Structure>(coords, symbols);
  const char* basis_set = "def2-svp";

  auto scf_solver = ScfSolverFactory::create();
  auto [E_default, wfn_default] = scf_solver->run(structure, 0, 1, basis_set);

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(wfn_default->get_orbitals());

  auto mc = MultiConfigurationCalculatorFactory::create();
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  auto [E_mc, wfn_mc] = mc->run(H, 2, 2);

  const auto& original = wfn_mc->get_container<StateVectorContainer>();

  EXPECT_TRUE(original.has_one_rdm_spin_dependent());
  EXPECT_TRUE(original.has_one_rdm_spin_traced());
  EXPECT_TRUE(original.has_two_rdm_spin_dependent());
  EXPECT_TRUE(original.has_two_rdm_spin_traced());

  std::string filename = "test_state_vector_rdm_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    original.to_hdf5(root);

    auto restored = StateVectorContainer::from_hdf5(root);

    EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_one_rdm_spin_traced());
    EXPECT_TRUE(restored->has_two_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_two_rdm_spin_traced());

    auto [restored_aa_rdm, restored_bb_rdm] =
        restored->get_active_one_rdm_spin_dependent();
    auto [original_aa_rdm, original_bb_rdm] =
        original.get_active_one_rdm_spin_dependent();
    const auto& restored_aa_rdm_r = std::get<Eigen::MatrixXd>(restored_aa_rdm);
    const auto& restored_bb_rdm_r = std::get<Eigen::MatrixXd>(restored_bb_rdm);
    const auto& original_aa_rdm_r = std::get<Eigen::MatrixXd>(original_aa_rdm);
    const auto& original_bb_rdm_r = std::get<Eigen::MatrixXd>(original_bb_rdm);

    EXPECT_TRUE(
        restored_aa_rdm_r.isApprox(original_aa_rdm_r, testing::rdm_tolerance));
    EXPECT_TRUE(
        restored_bb_rdm_r.isApprox(original_bb_rdm_r, testing::rdm_tolerance));

    auto restored_one_rdm = restored->get_active_one_rdm_spin_traced();
    auto original_one_rdm = original.get_active_one_rdm_spin_traced();
    const auto& restored_one_rdm_r =
        std::get<Eigen::MatrixXd>(restored_one_rdm);
    const auto& original_one_rdm_r =
        std::get<Eigen::MatrixXd>(original_one_rdm);

    EXPECT_TRUE(restored_one_rdm_r.isApprox(original_one_rdm_r,
                                            testing::rdm_tolerance));

    auto [restored_aaaa_rdm, restored_aabb_rdm, restored_bbbb_rdm] =
        restored->get_active_two_rdm_spin_dependent();
    auto [original_aaaa_rdm, original_aabb_rdm, original_bbbb_rdm] =
        original.get_active_two_rdm_spin_dependent();
    const auto& restored_aabb_rdm_r =
        std::get<Eigen::VectorXd>(restored_aabb_rdm);
    const auto& restored_aaaa_rdm_r =
        std::get<Eigen::VectorXd>(restored_aaaa_rdm);
    const auto& restored_bbbb_rdm_r =
        std::get<Eigen::VectorXd>(restored_bbbb_rdm);
    const auto& original_aabb_rdm_r =
        std::get<Eigen::VectorXd>(original_aabb_rdm);
    const auto& original_aaaa_rdm_r =
        std::get<Eigen::VectorXd>(original_aaaa_rdm);
    const auto& original_bbbb_rdm_r =
        std::get<Eigen::VectorXd>(original_bbbb_rdm);

    EXPECT_TRUE(restored_aabb_rdm_r.isApprox(original_aabb_rdm_r,
                                             testing::rdm_tolerance));
    EXPECT_TRUE(restored_aaaa_rdm_r.isApprox(original_aaaa_rdm_r,
                                             testing::rdm_tolerance));
    EXPECT_TRUE(restored_bbbb_rdm_r.isApprox(original_bbbb_rdm_r,
                                             testing::rdm_tolerance));

    auto restored_two_rdm = restored->get_active_two_rdm_spin_traced();
    auto original_two_rdm = original.get_active_two_rdm_spin_traced();
    const auto& restored_two_rdm_r =
        std::get<Eigen::VectorXd>(restored_two_rdm);
    const auto& original_two_rdm_r =
        std::get<Eigen::VectorXd>(original_two_rdm);
    EXPECT_TRUE(restored_two_rdm_r.isApprox(original_two_rdm_r,
                                            testing::rdm_tolerance));

    file.close();
  }
}

TEST_F(StateVectorContainerTest, Hdf5SerializationRDMsOpenShell) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {1.4, 0., 0.}};
  std::vector<std::string> symbols = {"H", "H"};
  auto structure = std::make_shared<Structure>(coords, symbols);
  const char* basis_set = "def2-svp";

  auto scf_solver = ScfSolverFactory::create();
  auto [E_default, wfn_default] = scf_solver->run(structure, -1, 2, basis_set);

  auto orbitals = wfn_default->get_orbitals();
  auto restricted_orbitals = std::make_shared<Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_energies().first,
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      testing::restricted_index_set(orbitals->get_num_molecular_orbitals(),
                                    orbitals->get_active_space_indices().first),
      testing::restricted_index_set(
          orbitals->get_num_molecular_orbitals(),
          orbitals->get_inactive_space_indices().first));

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(restricted_orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create();
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  auto [E_mc, wfn_mc] = mc->run(H, 2, 1);

  const auto& original = wfn_mc->get_container<StateVectorContainer>();

  EXPECT_TRUE(original.has_one_rdm_spin_dependent());
  EXPECT_TRUE(original.has_one_rdm_spin_traced());
  EXPECT_TRUE(original.has_two_rdm_spin_dependent());
  EXPECT_TRUE(original.has_two_rdm_spin_traced());

  EXPECT_TRUE(original.get_orbitals()->is_restricted());

  std::string filename = "test_state_vector_rdm_openshell_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    original.to_hdf5(root);

    auto restored = StateVectorContainer::from_hdf5(root);

    EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_one_rdm_spin_traced());
    EXPECT_TRUE(restored->has_two_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_two_rdm_spin_traced());

    EXPECT_TRUE(restored->get_orbitals()->is_restricted());

    auto [restored_aa_rdm, restored_bb_rdm] =
        restored->get_active_one_rdm_spin_dependent();
    auto [original_aa_rdm, original_bb_rdm] =
        original.get_active_one_rdm_spin_dependent();

    const auto& restored_aa_rdm_r = std::get<Eigen::MatrixXd>(restored_aa_rdm);
    const auto& restored_bb_rdm_r = std::get<Eigen::MatrixXd>(restored_bb_rdm);
    const auto& original_aa_rdm_r = std::get<Eigen::MatrixXd>(original_aa_rdm);
    const auto& original_bb_rdm_r = std::get<Eigen::MatrixXd>(original_bb_rdm);

    EXPECT_TRUE(
        restored_aa_rdm_r.isApprox(original_aa_rdm_r, testing::rdm_tolerance));
    EXPECT_TRUE(
        restored_bb_rdm_r.isApprox(original_bb_rdm_r, testing::rdm_tolerance));

    EXPECT_FALSE(
        restored_aa_rdm_r.isApprox(restored_bb_rdm_r, testing::rdm_tolerance));

    auto restored_one_rdm = restored->get_active_one_rdm_spin_traced();
    auto original_one_rdm = original.get_active_one_rdm_spin_traced();

    const auto& restored_one_rdm_r =
        std::get<Eigen::MatrixXd>(restored_one_rdm);
    const auto& original_one_rdm_r =
        std::get<Eigen::MatrixXd>(original_one_rdm);

    EXPECT_TRUE(restored_one_rdm_r.isApprox(original_one_rdm_r,
                                            testing::rdm_tolerance));

    auto [restored_aaaa_rdm, restored_aabb_rdm, restored_bbbb_rdm] =
        restored->get_active_two_rdm_spin_dependent();
    auto [original_aaaa_rdm, original_aabb_rdm, original_bbbb_rdm] =
        original.get_active_two_rdm_spin_dependent();

    const auto& restored_aabb_rdm_r =
        std::get<Eigen::VectorXd>(restored_aabb_rdm);
    const auto& restored_aaaa_rdm_r =
        std::get<Eigen::VectorXd>(restored_aaaa_rdm);
    const auto& restored_bbbb_rdm_r =
        std::get<Eigen::VectorXd>(restored_bbbb_rdm);
    const auto& original_aabb_rdm_r =
        std::get<Eigen::VectorXd>(original_aabb_rdm);
    const auto& original_aaaa_rdm_r =
        std::get<Eigen::VectorXd>(original_aaaa_rdm);
    const auto& original_bbbb_rdm_r =
        std::get<Eigen::VectorXd>(original_bbbb_rdm);

    EXPECT_TRUE(restored_aabb_rdm_r.isApprox(original_aabb_rdm_r,
                                             testing::rdm_tolerance));
    EXPECT_TRUE(restored_aaaa_rdm_r.isApprox(original_aaaa_rdm_r,
                                             testing::rdm_tolerance));
    EXPECT_TRUE(restored_bbbb_rdm_r.isApprox(original_bbbb_rdm_r,
                                             testing::rdm_tolerance));

    EXPECT_FALSE(restored_aaaa_rdm_r.isApprox(restored_bbbb_rdm_r,
                                              testing::rdm_tolerance));

    auto restored_two_rdm = restored->get_active_two_rdm_spin_traced();
    auto original_two_rdm = original.get_active_two_rdm_spin_traced();

    const auto& restored_two_rdm_r =
        std::get<Eigen::VectorXd>(restored_two_rdm);
    const auto& original_two_rdm_r =
        std::get<Eigen::VectorXd>(original_two_rdm);
    EXPECT_TRUE(restored_two_rdm_r.isApprox(original_two_rdm_r,
                                            testing::rdm_tolerance));

    file.close();
  }

  std::remove(filename.c_str());
}

TEST_F(StateVectorContainerTest, Truncate) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration::from_spin_half_string("2200"),
      Configuration::from_spin_half_string("2020"),
      Configuration::from_spin_half_string("2002"),
      Configuration::from_spin_half_string("0220"),
  };
  Eigen::VectorXd coeffs(4);
  coeffs << 0.8, 0.4, 0.3, 0.1;

  Wavefunction wfn(
      std::make_unique<StateVectorContainer>(coeffs, dets, orbitals));

  auto truncated = wfn.truncate(2);

  EXPECT_EQ(truncated->size(), 2);

  EXPECT_NEAR(truncated->norm(), 1.0, testing::wf_tolerance);

  const auto& truncated_dets = truncated->get_active_determinants();
  EXPECT_EQ(truncated_dets.size(), 2);
  EXPECT_EQ(truncated_dets[0].to_string(), "2200");
  EXPECT_EQ(truncated_dets[1].to_string(), "2020");

  double expected_norm = std::sqrt(0.8 * 0.8 + 0.4 * 0.4);
  const auto& truncated_coeffs =
      std::get<Eigen::VectorXd>(truncated->get_coefficients());
  EXPECT_NEAR(truncated_coeffs[0], 0.8 / expected_norm, testing::wf_tolerance);
  EXPECT_NEAR(truncated_coeffs[1], 0.4 / expected_norm, testing::wf_tolerance);

  auto full_copy = wfn.truncate(std::nullopt);
  EXPECT_EQ(full_copy->size(), 4);
  EXPECT_NEAR(full_copy->norm(), 1.0, testing::wf_tolerance);

  auto over_truncate = wfn.truncate(10);
  EXPECT_EQ(over_truncate->size(), 4);
  EXPECT_NEAR(over_truncate->norm(), 1.0, testing::wf_tolerance);
}

class SingleDeterminantTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(SingleDeterminantTest, BasicProperties) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  auto det = Configuration::from_spin_half_string("2200");
  StateVectorContainer sd(det, orbitals);

  EXPECT_EQ(sd.size(), 1);
  EXPECT_DOUBLE_EQ(std::get<double>(sd.get_coefficient(det)), 1.0);
  EXPECT_DOUBLE_EQ(std::get<double>(sd.get_coefficient(
                       Configuration::from_spin_half_string("2000"))),
                   0.0);
  EXPECT_TRUE(sd.contains_determinant(det));
  EXPECT_FALSE(
      sd.contains_determinant(Configuration::from_spin_half_string("2000")));
  EXPECT_EQ(sd.get_active_determinants().size(), 1);
  EXPECT_EQ(sd.get_active_determinants()[0].to_string(), "2200");
  EXPECT_DOUBLE_EQ(sd.norm(), 1.0);
  EXPECT_DOUBLE_EQ(sd.get_total_num_electrons().first, 2);
  EXPECT_DOUBLE_EQ(sd.get_total_num_electrons().second, 2);

  auto [total_alpha_elec, total_beta_elec] = sd.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sd.get_active_num_electrons();
  EXPECT_EQ(total_alpha_elec, 2);
  EXPECT_EQ(total_beta_elec, 2);
  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);
  EXPECT_EQ(total_alpha_elec, active_alpha_elec);
  EXPECT_EQ(total_beta_elec, active_beta_elec);

  auto [alpha_occ, beta_occ] = sd.get_total_orbital_occupations();
  EXPECT_EQ(alpha_occ.size(), 4);
  EXPECT_EQ(beta_occ.size(), 4);

  Eigen::VectorXd expected_alpha(4);
  expected_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_beta(4);
  expected_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(alpha_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_occ.isApprox(expected_beta, testing::wf_tolerance));

  auto [alpha_total_occ, beta_total_occ] = sd.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sd.get_active_orbital_occupations();
  EXPECT_EQ(alpha_total_occ.size(), 4);
  EXPECT_EQ(beta_total_occ.size(), 4);
  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);
  EXPECT_TRUE(alpha_total_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_total_occ.isApprox(expected_beta, testing::wf_tolerance));
  EXPECT_TRUE(alpha_active_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_active_occ.isApprox(expected_beta, testing::wf_tolerance));
  EXPECT_TRUE(
      alpha_total_occ.isApprox(alpha_active_occ, testing::wf_tolerance));
  EXPECT_TRUE(beta_total_occ.isApprox(beta_active_occ, testing::wf_tolerance));
}

TEST_F(SingleDeterminantTest, WithInactiveOrbitals) {
  auto base_orbitals = testing::create_test_orbitals(6, 6, true);
  std::vector<size_t> active_indices = {2, 3, 4, 5};
  std::vector<size_t> inactive_indices = {0, 1};
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  auto det = Configuration::from_spin_half_string("2200");
  StateVectorContainer sd(det, orbitals);

  auto [total_alpha_elec, total_beta_elec] = sd.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sd.get_active_num_electrons();

  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  EXPECT_EQ(total_alpha_elec, 4);
  EXPECT_EQ(total_beta_elec, 4);

  auto [alpha_total_occ, beta_total_occ] = sd.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sd.get_active_orbital_occupations();

  EXPECT_EQ(alpha_total_occ.size(), 6);
  EXPECT_EQ(beta_total_occ.size(), 6);

  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);

  Eigen::VectorXd expected_active_alpha(4);
  expected_active_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_active_beta(4);
  expected_active_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_active_occ.isApprox(expected_active_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_active_occ.isApprox(expected_active_beta, testing::wf_tolerance));

  Eigen::VectorXd expected_total_alpha(6);
  expected_total_alpha << 1.0, 1.0, 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_total_beta(6);
  expected_total_beta << 1.0, 1.0, 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_total_occ.isApprox(expected_total_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_total_occ.isApprox(expected_total_beta, testing::wf_tolerance));
}

TEST_F(SingleDeterminantTest, WithNonContinuousActiveSpace) {
  auto base_orbitals = testing::create_test_orbitals(8, 8, true);
  std::vector<size_t> active_indices = {2, 4, 6, 7};
  std::vector<size_t> inactive_indices = {0, 1};
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  auto det = Configuration::from_spin_half_string("2200");
  StateVectorContainer sd(det, orbitals);

  auto [total_alpha_elec, total_beta_elec] = sd.get_total_num_electrons();
  auto [active_alpha_elec, active_beta_elec] = sd.get_active_num_electrons();

  EXPECT_EQ(active_alpha_elec, 2);
  EXPECT_EQ(active_beta_elec, 2);

  EXPECT_EQ(total_alpha_elec, 4);
  EXPECT_EQ(total_beta_elec, 4);

  auto [alpha_total_occ, beta_total_occ] = sd.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      sd.get_active_orbital_occupations();

  EXPECT_EQ(alpha_total_occ.size(), 8);
  EXPECT_EQ(beta_total_occ.size(), 8);

  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);

  Eigen::VectorXd expected_active_alpha(4);
  expected_active_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_active_beta(4);
  expected_active_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_active_occ.isApprox(expected_active_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_active_occ.isApprox(expected_active_beta, testing::wf_tolerance));

  Eigen::VectorXd expected_total_alpha(8);
  expected_total_alpha << 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0;
  Eigen::VectorXd expected_total_beta(8);
  expected_total_beta << 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0;

  EXPECT_TRUE(
      alpha_total_occ.isApprox(expected_total_alpha, testing::wf_tolerance));
  EXPECT_TRUE(
      beta_total_occ.isApprox(expected_total_beta, testing::wf_tolerance));

  EXPECT_DOUBLE_EQ(alpha_total_occ(3), 0.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(5), 0.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(0), 1.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(1), 1.0);
}

TEST_F(SingleDeterminantTest, JsonSerialization) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  auto det = Configuration::from_spin_half_string("2200");

  StateVectorContainer original(det, orbitals);

  nlohmann::json j = original.to_json();

  auto restored = StateVectorContainer::from_json(j);

  auto original_wf = std::make_shared<Wavefunction>(
      std::make_unique<StateVectorContainer>(det, orbitals));
  nlohmann::json wf_j = original_wf->to_json();
  auto wf_restored = Wavefunction::from_json(wf_j);
  EXPECT_EQ(wf_restored->get_container_type(), "state_vector");
  auto& wf_restored_container =
      wf_restored->get_container<StateVectorContainer>();

  EXPECT_EQ(original.size(), restored->size());
  EXPECT_EQ(original.get_active_determinants().size(),
            restored->get_active_determinants().size());
  EXPECT_EQ(original.get_active_determinants()[0],
            restored->get_active_determinants()[0]);

  EXPECT_EQ(restored->size(), wf_restored_container.size());
  EXPECT_EQ(restored->get_active_determinants().size(),
            wf_restored_container.get_active_determinants().size());
  EXPECT_EQ(restored->get_active_determinants()[0],
            wf_restored_container.get_active_determinants()[0]);
}

TEST_F(SingleDeterminantTest, Hdf5Serialization) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  auto det = Configuration::from_spin_half_string("2200");

  StateVectorContainer original(det, orbitals);

  std::string filename = "test_single_det_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");

    original.to_hdf5(root);

    auto restored = StateVectorContainer::from_hdf5(root);

    EXPECT_EQ(original.size(), restored->size());
    EXPECT_EQ(original.get_active_determinants().size(),
              restored->get_active_determinants().size());
    EXPECT_EQ(original.get_active_determinants()[0],
              restored->get_active_determinants()[0]);

    file.close();
  }

  std::string wf_filename = "test_single_det_wavefunction_serialization.h5";
  {
    auto original_wf = std::make_shared<Wavefunction>(
        std::make_unique<StateVectorContainer>(det, orbitals));
    H5::H5File file(wf_filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original_wf->to_hdf5(root);
    file.close();
  }
  {
    H5::H5File file(wf_filename, H5F_ACC_RDONLY);
    H5::Group root = file.openGroup("/");
    auto wf_restored = Wavefunction::from_hdf5(root);
    EXPECT_EQ(wf_restored->get_container_type(), "state_vector");
    auto& wf_restored_container =
        wf_restored->get_container<StateVectorContainer>();

    H5::H5File file2(filename, H5F_ACC_RDONLY);
    H5::Group root2 = file2.openGroup("/");
    auto restored = StateVectorContainer::from_hdf5(root2);

    EXPECT_EQ(restored->size(), wf_restored_container.size());
    EXPECT_EQ(restored->get_active_determinants().size(),
              wf_restored_container.get_active_determinants().size());
    EXPECT_EQ(restored->get_active_determinants()[0],
              wf_restored_container.get_active_determinants()[0]);

    file.close();
    file2.close();
  }

  std::remove(filename.c_str());
  std::remove(wf_filename.c_str());
}

TEST_F(SingleDeterminantTest, ClosedShellReducedDensityMatrices) {
  size_t norb = 4;
  auto orbitals = testing::create_test_orbitals(norb, norb, true);
  auto det = Configuration::from_spin_half_string("2200");
  StateVectorContainer sd(det, orbitals);

  auto one_rdm = std::get<Eigen::MatrixXd>(sd.get_active_one_rdm_spin_traced());
  auto two_rdm = std::get<Eigen::VectorXd>(sd.get_active_two_rdm_spin_traced());
  auto [one_rdm_aa, one_rdm_bb] = sd.get_active_one_rdm_spin_dependent();
  auto [two_rdm_aaaa, two_rdm_aabb, two_rdm_bbbb] =
      sd.get_active_two_rdm_spin_dependent();

  auto two_rdm_bbaa =
      detail::transpose_ijkl_klij_vector_variant(two_rdm_aabb, norb);

  Eigen::MatrixXd expected_one_rdm = Eigen::MatrixXd::Zero(norb, norb);
  expected_one_rdm(0, 0) = 2.0;
  expected_one_rdm(1, 1) = 2.0;
  EXPECT_TRUE(
      one_rdm.isApprox(expected_one_rdm, testing::numerical_zero_tolerance));

  Eigen::MatrixXd one_rdm_sum = std::get<Eigen::MatrixXd>(one_rdm_aa) +
                                std::get<Eigen::MatrixXd>(one_rdm_bb);
  EXPECT_TRUE(one_rdm_sum.isApprox(expected_one_rdm,
                                   testing::numerical_zero_tolerance));

  auto sum_two_rdm = std::get<Eigen::VectorXd>(two_rdm_aabb) +
                     std::get<Eigen::VectorXd>(*two_rdm_bbaa) +
                     std::get<Eigen::VectorXd>(two_rdm_aaaa) +
                     std::get<Eigen::VectorXd>(two_rdm_bbbb);

  for (size_t i = 0; i < two_rdm.size(); ++i) {
    EXPECT_NEAR(two_rdm(i), sum_two_rdm(i), testing::numerical_zero_tolerance);
  }
}

TEST_F(SingleDeterminantTest, OpenShellReducedDensityMatrices) {
  size_t norb = 4;
  auto orbitals = testing::create_test_orbitals(norb, norb, true);
  auto det = Configuration::from_spin_half_string("2uu0");
  StateVectorContainer sd(det, orbitals);

  auto one_rdm = std::get<Eigen::MatrixXd>(sd.get_active_one_rdm_spin_traced());
  auto two_rdm = std::get<Eigen::VectorXd>(sd.get_active_two_rdm_spin_traced());
  auto [one_rdm_aa, one_rdm_bb] = sd.get_active_one_rdm_spin_dependent();
  auto [two_rdm_aaaa, two_rdm_aabb, two_rdm_bbbb] =
      sd.get_active_two_rdm_spin_dependent();

  auto two_rdm_bbaa =
      detail::transpose_ijkl_klij_vector_variant(two_rdm_aabb, norb);

  Eigen::MatrixXd expected_one_rdm = Eigen::MatrixXd::Zero(norb, norb);
  expected_one_rdm(0, 0) = 2.0;
  expected_one_rdm(1, 1) = 1.0;
  expected_one_rdm(2, 2) = 1.0;
  EXPECT_TRUE(
      one_rdm.isApprox(expected_one_rdm, testing::numerical_zero_tolerance));

  Eigen::MatrixXd one_rdm_sum = std::get<Eigen::MatrixXd>(one_rdm_aa) +
                                std::get<Eigen::MatrixXd>(one_rdm_bb);
  EXPECT_TRUE(one_rdm_sum.isApprox(expected_one_rdm,
                                   testing::numerical_zero_tolerance));

  auto sum_two_rdm = std::get<Eigen::VectorXd>(two_rdm_aabb) +
                     std::get<Eigen::VectorXd>(*two_rdm_bbaa) +
                     std::get<Eigen::VectorXd>(two_rdm_aaaa) +
                     std::get<Eigen::VectorXd>(two_rdm_bbbb);

  for (size_t i = 0; i < two_rdm.size(); ++i) {
    EXPECT_NEAR(two_rdm(i), sum_two_rdm(i), testing::numerical_zero_tolerance);
  }
}

TEST_F(SingleDeterminantTest, NonContinuousDeterminantReducedDensityMatrices) {
  size_t norb = 12;
  auto orbitals = testing::create_test_orbitals(norb, norb, true);
  auto det = Configuration::from_spin_half_string("2ud0200u0u2d");
  StateVectorContainer sd(det, orbitals);

  auto one_rdm = std::get<Eigen::MatrixXd>(sd.get_active_one_rdm_spin_traced());
  auto two_rdm = std::get<Eigen::VectorXd>(sd.get_active_two_rdm_spin_traced());
  auto [one_rdm_aa, one_rdm_bb] = sd.get_active_one_rdm_spin_dependent();
  auto [two_rdm_aaaa, two_rdm_aabb, two_rdm_bbbb] =
      sd.get_active_two_rdm_spin_dependent();

  auto two_rdm_bbaa =
      detail::transpose_ijkl_klij_vector_variant(two_rdm_aabb, norb);

  Eigen::MatrixXd expected_one_rdm_aa = Eigen::MatrixXd::Zero(norb, norb);
  Eigen::MatrixXd expected_one_rdm_bb = Eigen::MatrixXd::Zero(norb, norb);
  expected_one_rdm_aa(0, 0) = 1.0;
  expected_one_rdm_aa(1, 1) = 1.0;
  expected_one_rdm_aa(4, 4) = 1.0;
  expected_one_rdm_aa(7, 7) = 1.0;
  expected_one_rdm_aa(9, 9) = 1.0;
  expected_one_rdm_aa(10, 10) = 1.0;
  expected_one_rdm_bb(0, 0) = 1.0;
  expected_one_rdm_bb(2, 2) = 1.0;
  expected_one_rdm_bb(4, 4) = 1.0;
  expected_one_rdm_bb(10, 10) = 1.0;
  expected_one_rdm_bb(11, 11) = 1.0;

  EXPECT_TRUE(
      std::get<Eigen::MatrixXd>(one_rdm_aa)
          .isApprox(expected_one_rdm_aa, testing::numerical_zero_tolerance));
  EXPECT_TRUE(
      std::get<Eigen::MatrixXd>(one_rdm_bb)
          .isApprox(expected_one_rdm_bb, testing::numerical_zero_tolerance));
  Eigen::MatrixXd one_rdm_sum = std::get<Eigen::MatrixXd>(one_rdm_aa) +
                                std::get<Eigen::MatrixXd>(one_rdm_bb);
  EXPECT_TRUE(one_rdm_sum.isApprox(one_rdm, testing::numerical_zero_tolerance));

  auto sum_two_rdm = std::get<Eigen::VectorXd>(two_rdm_aabb) +
                     std::get<Eigen::VectorXd>(*two_rdm_bbaa) +
                     std::get<Eigen::VectorXd>(two_rdm_aaaa) +
                     std::get<Eigen::VectorXd>(two_rdm_bbbb);

  for (size_t i = 0; i < two_rdm.size(); ++i) {
    EXPECT_NEAR(two_rdm(i), sum_two_rdm(i), testing::numerical_zero_tolerance);
  }
}

TEST_F(SingleDeterminantTest, EntropiesTest) {
  size_t norb = 12;
  auto orbitals = testing::create_test_orbitals(norb, norb, true);
  auto det = Configuration::from_spin_half_string("2ud0200u0u2d");
  StateVectorContainer sd(det, orbitals);
  sd.get_active_one_rdm_spin_dependent();
  auto [aaaa, aabb, bbbb] = sd.get_active_two_rdm_spin_dependent();

  auto s1 = sd.get_single_orbital_entropies();

  Eigen::VectorXd expected_s1 = Eigen::VectorXd::Zero(norb);
  EXPECT_TRUE(s1.isApprox(expected_s1, testing::numerical_zero_tolerance));
}

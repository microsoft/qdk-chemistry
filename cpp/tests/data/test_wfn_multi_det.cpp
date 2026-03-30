// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.
//
// Consolidated typed tests for multi-determinant wavefunction containers
// (CAS and SCI). These two container types share an identical interface;
// this file eliminates the copy-paste that previously existed across
// test_wfn_cas.cpp and test_wfn_sci.cpp.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <filesystem>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

// ---------------------------------------------------------------------------
// Traits for parameterized tests — maps container type to its string name
// and file prefix for temp files.
// ---------------------------------------------------------------------------
template <typename T>
struct WfnContainerTraits;

template <>
struct WfnContainerTraits<CasWavefunctionContainer> {
  static constexpr const char* type_name = "cas";
  static constexpr const char* file_prefix = "test_cas";
};

template <>
struct WfnContainerTraits<SciWavefunctionContainer> {
  static constexpr const char* type_name = "sci";
  static constexpr const char* file_prefix = "test_sci";
};

// ---------------------------------------------------------------------------
// Typed test fixture for CAS + SCI shared behavior
// ---------------------------------------------------------------------------
template <typename T>
class MultiDetWfnTest : public ::testing::Test {};

using MultiDetTypes =
    ::testing::Types<CasWavefunctionContainer, SciWavefunctionContainer>;
TYPED_TEST_SUITE(MultiDetWfnTest, MultiDetTypes);

// ---------------------------------------------------------------------------
// Shared tests — run once per container type
// ---------------------------------------------------------------------------

TYPED_TEST(MultiDetWfnTest, BasicProperties) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  TypeParam container(coeffs, dets, orbitals, one_rdm, std::nullopt);

  EXPECT_EQ(container.size(), 3);
  EXPECT_DOUBLE_EQ(
      std::get<double>(container.get_coefficient(Configuration("2200"))), 0.5);
  EXPECT_DOUBLE_EQ(
      std::get<double>(container.get_coefficient(Configuration("2020"))), 0.5);
  EXPECT_DOUBLE_EQ(
      std::get<double>(container.get_coefficient(Configuration("2002"))),
      1.0 / sqrt(2));
  EXPECT_THROW(
      std::get<double>(container.get_coefficient(Configuration("2000"))),
      std::runtime_error);
  EXPECT_EQ(container.get_active_determinants().size(), 3);
  EXPECT_EQ(container.get_active_determinants()[0].to_string(), "2200");
  EXPECT_EQ(container.get_active_determinants()[1].to_string(), "2020");
  EXPECT_EQ(container.get_active_determinants()[2].to_string(), "2002");
  EXPECT_EQ(container.get_total_num_electrons().first, 2);
  EXPECT_EQ(container.get_total_num_electrons().second, 2);

  auto [total_alpha, total_beta] = container.get_total_num_electrons();
  auto [active_alpha, active_beta] = container.get_active_num_electrons();
  EXPECT_EQ(total_alpha, 2);
  EXPECT_EQ(total_beta, 2);
  EXPECT_EQ(active_alpha, 2);
  EXPECT_EQ(active_beta, 2);
  EXPECT_EQ(total_alpha, active_alpha);
  EXPECT_EQ(total_beta, active_beta);

  auto [alpha_occ, beta_occ] = container.get_total_orbital_occupations();
  EXPECT_EQ(alpha_occ.size(), 4);
  EXPECT_EQ(beta_occ.size(), 4);

  Eigen::VectorXd expected_alpha(4);
  expected_alpha << 1.0, 1.0, 0.0, 0.0;
  Eigen::VectorXd expected_beta(4);
  expected_beta << 1.0, 1.0, 0.0, 0.0;

  EXPECT_TRUE(alpha_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_occ.isApprox(expected_beta, testing::wf_tolerance));

  auto [alpha_total_occ, beta_total_occ] =
      container.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      container.get_active_orbital_occupations();
  EXPECT_EQ(alpha_total_occ.size(), 4);
  EXPECT_EQ(beta_total_occ.size(), 4);
  EXPECT_EQ(alpha_active_occ.size(), 4);
  EXPECT_EQ(beta_active_occ.size(), 4);
  EXPECT_TRUE(alpha_total_occ.isApprox(expected_alpha, testing::wf_tolerance));
  EXPECT_TRUE(beta_total_occ.isApprox(expected_beta, testing::wf_tolerance));
  EXPECT_TRUE(
      alpha_total_occ.isApprox(alpha_active_occ, testing::wf_tolerance));
  EXPECT_TRUE(beta_total_occ.isApprox(beta_active_occ, testing::wf_tolerance));
}

TYPED_TEST(MultiDetWfnTest, EmptyDeterminantsThrows) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> empty_dets;
  Eigen::VectorXd empty_coeffs(0);

  TypeParam container(empty_coeffs, empty_dets, orbitals);

  EXPECT_THROW(container.get_coefficient(Configuration("2200")),
               std::runtime_error);
  EXPECT_THROW(container.get_total_num_electrons(), std::runtime_error);
  EXPECT_THROW(container.get_active_num_electrons(), std::runtime_error);
  EXPECT_THROW(container.get_total_orbital_occupations(), std::runtime_error);
  EXPECT_THROW(container.get_active_orbital_occupations(), std::runtime_error);
  EXPECT_EQ(container.size(), 0);
}

TYPED_TEST(MultiDetWfnTest, ErrorMessagesAreDescriptive) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> empty_dets;
  Eigen::VectorXd empty_coeffs(0);

  TypeParam container(empty_coeffs, empty_dets, orbitals);

  try {
    container.get_total_orbital_occupations();
    FAIL() << "Expected std::runtime_error";
  } catch (const std::runtime_error& e) {
    EXPECT_STREQ(e.what(), "No determinants available");
  }
}

TYPED_TEST(MultiDetWfnTest, EntropyWithMissingRDMs) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {Configuration("2200")};
  Eigen::VectorXd coeffs(1);
  coeffs << 1.0;

  TypeParam container(coeffs, dets, orbitals);
  EXPECT_THROW(container.get_single_orbital_entropies(), std::runtime_error);
}

TYPED_TEST(MultiDetWfnTest, WithInactiveOrbitals) {
  auto base_orbitals = testing::create_test_orbitals(6, 6, true);
  std::vector<size_t> active_indices = {2, 3, 4, 5};
  std::vector<size_t> inactive_indices = {0, 1};
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  TypeParam container(coeffs, dets, orbitals, one_rdm, std::nullopt);

  auto [total_alpha, total_beta] = container.get_total_num_electrons();
  auto [active_alpha, active_beta] = container.get_active_num_electrons();

  EXPECT_EQ(active_alpha, 2);
  EXPECT_EQ(active_beta, 2);
  EXPECT_EQ(total_alpha, 4);
  EXPECT_EQ(total_beta, 4);

  auto [alpha_total_occ, beta_total_occ] =
      container.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      container.get_active_orbital_occupations();

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

TYPED_TEST(MultiDetWfnTest, WithNonContinuousActiveSpace) {
  auto base_orbitals = testing::create_test_orbitals(8, 8, true);
  std::vector<size_t> active_indices = {2, 4, 6, 7};
  std::vector<size_t> inactive_indices = {0, 1};
  auto orbitals = testing::with_active_space(base_orbitals, active_indices,
                                             inactive_indices);

  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;
  one_rdm(2, 2) = 0.0;
  one_rdm(3, 3) = 0.0;

  TypeParam container(coeffs, dets, orbitals, one_rdm, std::nullopt);

  auto [total_alpha, total_beta] = container.get_total_num_electrons();
  auto [active_alpha, active_beta] = container.get_active_num_electrons();
  EXPECT_EQ(active_alpha, 2);
  EXPECT_EQ(active_beta, 2);
  EXPECT_EQ(total_alpha, 4);
  EXPECT_EQ(total_beta, 4);

  auto [alpha_total_occ, beta_total_occ] =
      container.get_total_orbital_occupations();
  auto [alpha_active_occ, beta_active_occ] =
      container.get_active_orbital_occupations();

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

  // Orbital indices: 0,   1,   2,   3,   4,   5,   6,   7
  // Types:          ina, ina, act, vir, act, vir, act, act
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
  EXPECT_DOUBLE_EQ(alpha_total_occ(2), 1.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(4), 1.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(6), 0.0);
  EXPECT_DOUBLE_EQ(alpha_total_occ(7), 0.0);
}

TYPED_TEST(MultiDetWfnTest, JsonSerialization) {
  using Traits = WfnContainerTraits<TypeParam>;

  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1.0 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;

  TypeParam original(coeffs, dets, orbitals, one_rdm, std::nullopt);

  nlohmann::json j = original.to_json();

  auto restored = std::unique_ptr<TypeParam>(
      dynamic_cast<TypeParam*>(WavefunctionContainer::from_json(j).release()));

  auto original_wf = std::make_shared<Wavefunction>(
      std::make_unique<TypeParam>(coeffs, dets, orbitals, one_rdm,
                                  std::nullopt));
  nlohmann::json wf_j = original_wf->to_json();
  auto wf_restored = Wavefunction::from_json(wf_j);
  EXPECT_EQ(wf_restored->get_container_type(), Traits::type_name);
  auto& wf_restored_container =
      wf_restored->template get_container<TypeParam>();

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
}

TYPED_TEST(MultiDetWfnTest, Hdf5Serialization) {
  using Traits = WfnContainerTraits<TypeParam>;

  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
  Eigen::VectorXd coeffs(3);
  coeffs << 0.5, 0.5, 1.0 / sqrt(2);

  Eigen::MatrixXd one_rdm(4, 4);
  one_rdm.setIdentity();
  one_rdm *= 2.0;

  TypeParam original(coeffs, dets, orbitals, one_rdm, std::nullopt);

  std::string filename =
      std::string(Traits::file_prefix) + "_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original.to_hdf5(root);
    auto restored = TypeParam::from_hdf5(root);

    EXPECT_EQ(original.size(), restored->size());
    const auto& orig_c =
        std::get<Eigen::VectorXd>(original.get_coefficients());
    const auto& rest_c =
        std::get<Eigen::VectorXd>(restored->get_coefficients());
    EXPECT_TRUE(orig_c.isApprox(rest_c, testing::wf_tolerance));

    for (size_t i = 0; i < original.get_active_determinants().size(); ++i) {
      EXPECT_EQ(original.get_active_determinants()[i],
                restored->get_active_determinants()[i]);
    }
    file.close();
  }

  // Also test via Wavefunction::from_hdf5
  std::string wf_filename =
      std::string(Traits::file_prefix) + "_wfn_serialization.h5";
  {
    auto original_wf = std::make_shared<Wavefunction>(
        std::make_unique<TypeParam>(coeffs, dets, orbitals, one_rdm,
                                    std::nullopt));
    H5::H5File file(wf_filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original_wf->to_hdf5(root);
    file.close();
  }
  {
    H5::H5File file(wf_filename, H5F_ACC_RDONLY);
    H5::Group root = file.openGroup("/");
    auto wf_restored = Wavefunction::from_hdf5(root);
    EXPECT_EQ(wf_restored->get_container_type(), Traits::type_name);

    auto& container = wf_restored->template get_container<TypeParam>();
    EXPECT_EQ(container.size(), original.size());
    file.close();
  }

  std::remove(filename.c_str());
  std::remove(wf_filename.c_str());
}

TYPED_TEST(MultiDetWfnTest, JsonSerializationRDMs) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002")};
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

  TypeParam original(coeffs, dets, orbitals, std::nullopt, one_rdm_aa,
                     one_rdm_aa, std::nullopt, two_rdm_aabb, two_rdm_aaaa,
                     two_rdm_aaaa);

  nlohmann::json j = original.to_json();

  EXPECT_TRUE(j.contains("rdms"));
  EXPECT_TRUE(j["rdms"].contains("one_rdm_aa"));
  EXPECT_TRUE(j["rdms"].contains("two_rdm_aabb"));
  EXPECT_TRUE(j["rdms"].contains("two_rdm_aaaa"));

  auto restored = std::unique_ptr<TypeParam>(
      dynamic_cast<TypeParam*>(WavefunctionContainer::from_json(j).release()));

  EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
  EXPECT_TRUE(restored->has_two_rdm_spin_dependent());

  auto [orig_one_aa, orig_one_bb] =
      original.get_active_one_rdm_spin_dependent();
  auto [rest_one_aa, rest_one_bb] =
      restored->get_active_one_rdm_spin_dependent();

  EXPECT_TRUE(std::get<Eigen::MatrixXd>(orig_one_aa)
                  .isApprox(std::get<Eigen::MatrixXd>(rest_one_aa),
                            testing::wf_tolerance));

  auto [orig_two_aabb, orig_two_aaaa, orig_two_bbbb] =
      original.get_active_two_rdm_spin_dependent();
  auto [rest_two_aabb, rest_two_aaaa, rest_two_bbbb] =
      restored->get_active_two_rdm_spin_dependent();

  EXPECT_TRUE(std::get<Eigen::VectorXd>(orig_two_aabb)
                  .isApprox(std::get<Eigen::VectorXd>(rest_two_aabb),
                            testing::wf_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(orig_two_aaaa)
                  .isApprox(std::get<Eigen::VectorXd>(rest_two_aaaa),
                            testing::wf_tolerance));
}

// ---------------------------------------------------------------------------
// Helper: validate RDM round-trip for serialization tests that run real
// SCF/MC pipelines. Extracted to avoid duplicating assertion logic.
// ---------------------------------------------------------------------------
namespace {

void verify_rdm_roundtrip(const WavefunctionContainer& original,
                          const WavefunctionContainer& restored) {
  EXPECT_TRUE(restored.has_one_rdm_spin_dependent());
  EXPECT_TRUE(restored.has_one_rdm_spin_traced());
  EXPECT_TRUE(restored.has_two_rdm_spin_dependent());
  EXPECT_TRUE(restored.has_two_rdm_spin_traced());
  EXPECT_TRUE(restored.get_orbitals()->is_restricted());

  auto [restored_aa, restored_bb] =
      restored.get_active_one_rdm_spin_dependent();
  auto [original_aa, original_bb] =
      original.get_active_one_rdm_spin_dependent();

  const auto& r_aa = std::get<Eigen::MatrixXd>(restored_aa);
  const auto& r_bb = std::get<Eigen::MatrixXd>(restored_bb);
  const auto& o_aa = std::get<Eigen::MatrixXd>(original_aa);
  const auto& o_bb = std::get<Eigen::MatrixXd>(original_bb);

  EXPECT_TRUE(r_aa.isApprox(o_aa, testing::rdm_tolerance));
  EXPECT_TRUE(r_bb.isApprox(o_bb, testing::rdm_tolerance));

  // alpha and beta 1-RDMs should differ for open-shell
  EXPECT_FALSE(r_aa.isApprox(r_bb, testing::rdm_tolerance));

  auto restored_1rdm = restored.get_active_one_rdm_spin_traced();
  auto original_1rdm = original.get_active_one_rdm_spin_traced();
  EXPECT_TRUE(std::get<Eigen::MatrixXd>(restored_1rdm)
                  .isApprox(std::get<Eigen::MatrixXd>(original_1rdm),
                            testing::rdm_tolerance));

  auto [r_aabb, r_aaaa, r_bbbb] =
      restored.get_active_two_rdm_spin_dependent();
  auto [o_aabb, o_aaaa, o_bbbb] =
      original.get_active_two_rdm_spin_dependent();

  EXPECT_TRUE(std::get<Eigen::VectorXd>(r_aabb).isApprox(
      std::get<Eigen::VectorXd>(o_aabb), testing::rdm_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(r_aaaa).isApprox(
      std::get<Eigen::VectorXd>(o_aaaa), testing::rdm_tolerance));
  EXPECT_TRUE(std::get<Eigen::VectorXd>(r_bbbb).isApprox(
      std::get<Eigen::VectorXd>(o_bbbb), testing::rdm_tolerance));
  EXPECT_FALSE(std::get<Eigen::VectorXd>(r_aaaa).isApprox(
      std::get<Eigen::VectorXd>(r_bbbb), testing::rdm_tolerance));

  auto r_2rdm = restored.get_active_two_rdm_spin_traced();
  auto o_2rdm = original.get_active_two_rdm_spin_traced();
  EXPECT_TRUE(std::get<Eigen::VectorXd>(r_2rdm).isApprox(
      std::get<Eigen::VectorXd>(o_2rdm), testing::rdm_tolerance));
}

}  // namespace

// ---------------------------------------------------------------------------
// CAS-specific tests (cannot be trivially typed because the MC calculator
// produces CAS containers)
// ---------------------------------------------------------------------------
class CasWavefunctionTest : public ::testing::Test {};

TEST_F(CasWavefunctionTest, Hdf5SerializationComplex) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {Configuration("2200"),
                                     Configuration("2020")};
  Eigen::VectorXcd coeffs(2);
  coeffs << std::complex<double>(0.5, 0.3), std::complex<double>(0.6, -0.2);

  CasWavefunctionContainer original(coeffs, dets, orbitals);

  nlohmann::json j = original.to_json();
  auto restored_json = std::unique_ptr<CasWavefunctionContainer>(
      dynamic_cast<CasWavefunctionContainer*>(
          WavefunctionContainer::from_json(j).release()));

  const auto& orig_c = std::get<Eigen::VectorXcd>(original.get_coefficients());
  const auto& rest_c =
      std::get<Eigen::VectorXcd>(restored_json->get_coefficients());
  EXPECT_TRUE(orig_c.isApprox(rest_c, testing::wf_tolerance));

  std::string filename = "test_cas_complex_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original.to_hdf5(root);
    auto restored_hdf5 = CasWavefunctionContainer::from_hdf5(root);
    const auto& rest_h5 =
        std::get<Eigen::VectorXcd>(restored_hdf5->get_coefficients());
    EXPECT_TRUE(orig_c.isApprox(rest_h5, testing::wf_tolerance));
    file.close();
  }
  std::remove(filename.c_str());
}

TEST_F(CasWavefunctionTest, JsonSerializationRDMsOpenShell) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {1.4, 0., 0.}};
  std::vector<std::string> symbols = {"H", "H"};
  auto structure = std::make_shared<Structure>(coords, symbols);

  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(structure, -1, 2, "def2-svp");

  auto orbitals = wfn->get_orbitals();
  auto restricted_orbitals = std::make_shared<Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_energies().first,
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      std::make_tuple(orbitals->get_active_space_indices().first,
                      orbitals->get_inactive_space_indices().first));

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(restricted_orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create();
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  auto [E_cas, wfn_cas] = mc->run(H, 2, 1);

  const auto& original = wfn_cas->get_container<CasWavefunctionContainer>();

  nlohmann::json j = original.to_json();
  EXPECT_TRUE(j.contains("rdms"));

  auto restored = std::unique_ptr<CasWavefunctionContainer>(
      dynamic_cast<CasWavefunctionContainer*>(
          WavefunctionContainer::from_json(j).release()));

  verify_rdm_roundtrip(original, *restored);
}

TEST_F(CasWavefunctionTest, Hdf5SerializationRDMs) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {1.4, 0., 0.}};
  std::vector<std::string> symbols = {"H", "H"};
  auto structure = std::make_shared<Structure>(coords, symbols);

  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(structure, 0, 1, "def2-svp");

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(wfn->get_orbitals());

  auto mc = MultiConfigurationCalculatorFactory::create();
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  auto [E_cas, wfn_cas] = mc->run(H, 2, 2);

  const auto& original = wfn_cas->get_container<CasWavefunctionContainer>();

  std::string filename = "test_cas_rdm_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original.to_hdf5(root);
    auto restored = CasWavefunctionContainer::from_hdf5(root);

    EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_one_rdm_spin_traced());
    EXPECT_TRUE(restored->has_two_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_two_rdm_spin_traced());

    auto [r_aa, r_bb] = restored->get_active_one_rdm_spin_dependent();
    auto [o_aa, o_bb] = original.get_active_one_rdm_spin_dependent();
    EXPECT_TRUE(std::get<Eigen::MatrixXd>(r_aa).isApprox(
        std::get<Eigen::MatrixXd>(o_aa), testing::rdm_tolerance));
    EXPECT_TRUE(std::get<Eigen::MatrixXd>(r_bb).isApprox(
        std::get<Eigen::MatrixXd>(o_bb), testing::rdm_tolerance));

    auto [r_aabb, r_aaaa, r_bbbb] =
        restored->get_active_two_rdm_spin_dependent();
    auto [o_aabb, o_aaaa, o_bbbb] =
        original.get_active_two_rdm_spin_dependent();
    EXPECT_TRUE(std::get<Eigen::VectorXd>(r_aabb).isApprox(
        std::get<Eigen::VectorXd>(o_aabb), testing::rdm_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(r_aaaa).isApprox(
        std::get<Eigen::VectorXd>(o_aaaa), testing::rdm_tolerance));
    EXPECT_TRUE(std::get<Eigen::VectorXd>(r_bbbb).isApprox(
        std::get<Eigen::VectorXd>(o_bbbb), testing::rdm_tolerance));

    file.close();
  }
  std::remove(filename.c_str());
}

TEST_F(CasWavefunctionTest, Hdf5SerializationRDMsOpenShell) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {1.4, 0., 0.}};
  std::vector<std::string> symbols = {"H", "H"};
  auto structure = std::make_shared<Structure>(coords, symbols);

  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(structure, -1, 2, "def2-svp");

  auto orbitals = wfn->get_orbitals();
  auto restricted_orbitals = std::make_shared<Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_energies().first,
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      std::make_tuple(orbitals->get_active_space_indices().first,
                      orbitals->get_inactive_space_indices().first));

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(restricted_orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create();
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  auto [E_cas, wfn_cas] = mc->run(H, 2, 1);

  const auto& original = wfn_cas->get_container<CasWavefunctionContainer>();

  std::string filename = "test_cas_rdm_openshell_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original.to_hdf5(root);
    auto restored = CasWavefunctionContainer::from_hdf5(root);
    verify_rdm_roundtrip(original, *restored);
    file.close();
  }
  std::remove(filename.c_str());
}

// ---------------------------------------------------------------------------
// SCI-specific tests
// ---------------------------------------------------------------------------
class SciWavefunctionTest : public ::testing::Test {};

TEST_F(SciWavefunctionTest, Truncate) {
  auto orbitals = testing::create_test_orbitals(4, 4, true);
  std::vector<Configuration> dets = {
      Configuration("2200"), Configuration("2020"), Configuration("2002"),
      Configuration("0220")};
  Eigen::VectorXd coeffs(4);
  coeffs << 0.8, 0.4, 0.3, 0.1;

  Wavefunction wfn(
      std::make_unique<SciWavefunctionContainer>(coeffs, dets, orbitals));

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

TEST_F(SciWavefunctionTest, JsonSerializationRDMsOpenShell) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}};
  std::vector<std::string> symbols = {"Li"};
  auto structure = std::make_shared<Structure>(coords, symbols);

  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(structure, 0, 2, "sto-3g");

  auto orbitals = wfn->get_orbitals();
  auto restricted_orbitals = std::make_shared<Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_energies().first,
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      std::make_tuple(orbitals->get_active_space_indices().first,
                      orbitals->get_inactive_space_indices().first));

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(restricted_orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create("macis_asci");
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  mc->settings().set("ntdets_max", 1);
  mc->settings().set("max_refine_iter", 0);
  mc->settings().set("grow_factor", 2);
  auto [E_sci, wfn_sci] = mc->run(H, 2, 1);

  const auto& original = wfn_sci->get_container<SciWavefunctionContainer>();

  nlohmann::json j = original.to_json();
  EXPECT_TRUE(j.contains("rdms"));

  auto restored = std::unique_ptr<SciWavefunctionContainer>(
      dynamic_cast<SciWavefunctionContainer*>(
          WavefunctionContainer::from_json(j).release()));

  verify_rdm_roundtrip(original, *restored);
}

TEST_F(SciWavefunctionTest, Hdf5SerializationRDMs) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}};
  std::vector<std::string> symbols = {"Li"};
  auto structure = std::make_shared<Structure>(coords, symbols);

  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(structure, 0, 2, "sto-3g");

  auto orbitals = wfn->get_orbitals();
  auto restricted_orbitals = std::make_shared<Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_energies().first,
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      std::make_tuple(orbitals->get_active_space_indices().first,
                      orbitals->get_inactive_space_indices().first));

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(restricted_orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create("macis_asci");
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  mc->settings().set("ntdets_max", 1);
  mc->settings().set("max_refine_iter", 0);
  mc->settings().set("grow_factor", 2);
  auto [E_sci, wfn_sci] = mc->run(H, 2, 1);

  const auto& original = wfn_sci->get_container<SciWavefunctionContainer>();

  std::string filename = "test_sci_rdm_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original.to_hdf5(root);
    auto restored = SciWavefunctionContainer::from_hdf5(root);

    EXPECT_TRUE(restored->has_one_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_one_rdm_spin_traced());
    EXPECT_TRUE(restored->has_two_rdm_spin_dependent());
    EXPECT_TRUE(restored->has_two_rdm_spin_traced());

    auto [r_aa, r_bb] = restored->get_active_one_rdm_spin_dependent();
    auto [o_aa, o_bb] = original.get_active_one_rdm_spin_dependent();
    EXPECT_TRUE(std::get<Eigen::MatrixXd>(r_aa).isApprox(
        std::get<Eigen::MatrixXd>(o_aa), testing::rdm_tolerance));
    EXPECT_TRUE(std::get<Eigen::MatrixXd>(r_bb).isApprox(
        std::get<Eigen::MatrixXd>(o_bb), testing::rdm_tolerance));

    file.close();
  }
  std::remove(filename.c_str());
}

TEST_F(SciWavefunctionTest, Hdf5SerializationRDMsOpenShell) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}};
  std::vector<std::string> symbols = {"Li"};
  auto structure = std::make_shared<Structure>(coords, symbols);

  auto scf_solver = ScfSolverFactory::create();
  auto [E, wfn] = scf_solver->run(structure, 0, 2, "sto-3g");

  auto orbitals = wfn->get_orbitals();
  auto restricted_orbitals = std::make_shared<Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_energies().first,
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      std::make_tuple(orbitals->get_active_space_indices().first,
                      orbitals->get_inactive_space_indices().first));

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(restricted_orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create("macis_asci");
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  mc->settings().set("ntdets_max", 1);
  mc->settings().set("max_refine_iter", 0);
  mc->settings().set("grow_factor", 2);
  auto [E_sci, wfn_sci] = mc->run(H, 2, 1);

  const auto& original = wfn_sci->get_container<SciWavefunctionContainer>();

  std::string filename = "test_sci_rdm_openshell_serialization.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root = file.openGroup("/");
    original.to_hdf5(root);
    auto restored = SciWavefunctionContainer::from_hdf5(root);
    verify_rdm_roundtrip(original, *restored);
    file.close();
  }
  std::remove(filename.c_str());
}

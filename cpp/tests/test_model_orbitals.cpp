// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <memory>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

namespace qdk::chemistry::tests::test_support {

// Spin (S_z) symmetry; equivalent=true is restricted, false is unrestricted.
std::shared_ptr<const SymmetryProduct> spin_sym(bool equivalent) {
  return std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, equivalent)}));
}

// No-symmetry (trivial) index set over `num_modes` carrying `idx` in a single
// statistics-generic block.
std::shared_ptr<const SymmetryBlockedIndexSet> trivial_iset(
    size_t num_modes, const std::vector<size_t>& idx) {
  auto sym =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  std::unordered_map<SymmetryLabel, std::size_t> ext{
      {SymmetryLabel{}, num_modes}};
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices{
      {SymmetryLabel{}, std::vector<std::uint32_t>(idx.begin(), idx.end())}};
  return std::make_shared<const SymmetryBlockedIndexSet>(sym, ext,
                                                         std::move(indices));
}

// Spin-resolved index set over `num_modes` with per-spin alpha/beta index
// lists.
std::shared_ptr<const SymmetryBlockedIndexSet> spin_iset(
    size_t num_modes, const std::vector<size_t>& alpha,
    const std::vector<size_t>& beta, bool equivalent) {
  std::unordered_map<SymmetryLabel, std::size_t> ext{{axes::alpha(), num_modes},
                                                     {axes::beta(), num_modes}};
  std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>> indices{
      {axes::alpha(), std::vector<std::uint32_t>(alpha.begin(), alpha.end())},
      {axes::beta(), std::vector<std::uint32_t>(beta.begin(), beta.end())}};
  return std::make_shared<const SymmetryBlockedIndexSet>(
      spin_sym(equivalent), ext, std::move(indices));
}

}  // namespace qdk::chemistry::tests::test_support

namespace test_support = qdk::chemistry::tests::test_support;
using test_support::spin_iset;
using test_support::spin_sym;
using test_support::trivial_iset;

class ModelOrbitalsTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(ModelOrbitalsTest, BasicConstructor) {
  // Full active space; restricted-ness inferred from the declared symmetries.
  const size_t basis_size = 4;

  // No symmetry: a single statistics-generic channel, reported as restricted.
  ModelOrbitals model_restricted(basis_size);
  EXPECT_TRUE(model_restricted.is_restricted());
  EXPECT_FALSE(model_restricted.is_unrestricted());
  EXPECT_EQ(basis_size, model_restricted.get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, model_restricted.get_num_molecular_orbitals());

  // Spin axis with distinct alpha/beta labels: unrestricted.
  ModelOrbitals model_unrestricted(basis_size, spin_sym(/*equivalent=*/false));
  EXPECT_FALSE(model_unrestricted.is_restricted());
  EXPECT_TRUE(model_unrestricted.is_unrestricted());
  EXPECT_EQ(basis_size, model_unrestricted.get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, model_unrestricted.get_num_molecular_orbitals());
}

TEST_F(ModelOrbitalsTest, RestrictedActiveSpaceConstructor) {
  // Active/inactive spaces from no-symmetry index sets (restricted).
  const size_t basis_size = 6;
  std::vector<size_t> active_indices = {1, 2, 3};
  std::vector<size_t> inactive_indices = {0, 4, 5};

  ModelOrbitals model(trivial_iset(basis_size, active_indices),
                      trivial_iset(basis_size, inactive_indices));

  EXPECT_TRUE(model.is_restricted());
  EXPECT_EQ(basis_size, model.get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, model.get_num_molecular_orbitals());

  // Check active space (alpha == beta for a single channel).
  EXPECT_TRUE(model.has_active_space());
  auto alpha_active =
      spin_channel_indices(model.active_indices(), axes::alpha());
  auto beta_active = spin_channel_indices(model.active_indices(), axes::beta());
  EXPECT_EQ(active_indices, alpha_active);
  EXPECT_EQ(active_indices, beta_active);

  // Check inactive space.
  EXPECT_TRUE(model.has_inactive_space());
  auto alpha_inactive =
      spin_channel_indices(model.inactive_indices(), axes::alpha());
  auto beta_inactive =
      spin_channel_indices(model.inactive_indices(), axes::beta());
  EXPECT_EQ(inactive_indices, alpha_inactive);
  EXPECT_EQ(inactive_indices, beta_inactive);
}

TEST_F(ModelOrbitalsTest, UnrestrictedActiveSpaceConstructor) {
  // Different alpha/beta active and inactive spaces (spin-resolved).
  const size_t basis_size = 6;
  std::vector<size_t> alpha_active = {1, 2};
  std::vector<size_t> beta_active = {2, 3, 4};
  std::vector<size_t> alpha_inactive = {0, 3, 4, 5};
  std::vector<size_t> beta_inactive = {0, 1, 5};

  ModelOrbitals model(
      spin_iset(basis_size, alpha_active, beta_active, /*equivalent=*/false),
      spin_iset(basis_size, alpha_inactive, beta_inactive,
                /*equivalent=*/false));

  EXPECT_FALSE(model.is_restricted());
  EXPECT_TRUE(model.is_unrestricted());
  EXPECT_EQ(basis_size, model.get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, model.get_num_molecular_orbitals());

  // Check active space.
  EXPECT_TRUE(model.has_active_space());
  auto retrieved_alpha_active =
      spin_channel_indices(model.active_indices(), axes::alpha());
  auto retrieved_beta_active =
      spin_channel_indices(model.active_indices(), axes::beta());
  EXPECT_EQ(alpha_active, retrieved_alpha_active);
  EXPECT_EQ(beta_active, retrieved_beta_active);

  // Check inactive space.
  EXPECT_TRUE(model.has_inactive_space());
  auto retrieved_alpha_inactive =
      spin_channel_indices(model.inactive_indices(), axes::alpha());
  auto retrieved_beta_inactive =
      spin_channel_indices(model.inactive_indices(), axes::beta());
  EXPECT_EQ(alpha_inactive, retrieved_alpha_inactive);
  EXPECT_EQ(beta_inactive, retrieved_beta_inactive);
}

TEST_F(ModelOrbitalsTest, ConstructorValidation) {
  const size_t basis_size = 4;

  // Out-of-bounds indices are rejected when building the index set.
  EXPECT_THROW(trivial_iset(basis_size, {0, 1, 4}),  // 4 >= basis_size
               std::out_of_range);

  // Active and inactive spaces must be disjoint.
  EXPECT_THROW(ModelOrbitals(trivial_iset(basis_size, {0, 1, 2}),
                             trivial_iset(basis_size, {2, 3})),  // 2 in both
               std::invalid_argument);
}

TEST_F(ModelOrbitalsTest, UnrestrictedConstructorValidation) {
  const size_t basis_size = 4;

  // Out-of-bounds alpha index rejected when building the index set.
  EXPECT_THROW(spin_iset(basis_size, {0, 1, 4}, {0, 1}, /*equivalent=*/false),
               std::out_of_range);

  // Overlap between active and inactive (alpha channel) is rejected.
  EXPECT_THROW(
      ModelOrbitals(
          spin_iset(basis_size, {0, 1, 2}, {0, 1}, /*equivalent=*/false),
          spin_iset(basis_size, {2, 3}, {2, 3}, /*equivalent=*/false)),
      std::invalid_argument);
}

TEST_F(ModelOrbitalsTest, DefaultActiveSpace) {
  // Full active space over all modes; inactive empty.
  const size_t basis_size = 5;

  ModelOrbitals model(basis_size);
  EXPECT_TRUE(model.has_active_space());

  auto alpha_active =
      spin_channel_indices(model.active_indices(), axes::alpha());
  auto beta_active = spin_channel_indices(model.active_indices(), axes::beta());
  EXPECT_EQ(basis_size, alpha_active.size());
  EXPECT_EQ(basis_size, beta_active.size());

  for (size_t i = 0; i < basis_size; ++i) {
    EXPECT_EQ(i, alpha_active[i]);
    EXPECT_EQ(i, beta_active[i]);
  }

  EXPECT_FALSE(model.has_inactive_space());
}

TEST_F(ModelOrbitalsTest, ThrowingMethods) {
  // Methods requiring real basis data throw for model systems.
  const size_t basis_size = 3;
  ModelOrbitals model(basis_size);

  EXPECT_THROW(model.coefficients(), std::runtime_error);
  EXPECT_THROW(model.energies(), std::runtime_error);
  EXPECT_THROW(model.get_overlap_matrix(), std::runtime_error);
  EXPECT_THROW(model.get_basis_set(), std::runtime_error);
  EXPECT_THROW(model.coefficients()->block({axes::alpha(), axes::alpha()}),
               std::runtime_error);
  EXPECT_THROW(model.coefficients()->block({axes::beta(), axes::beta()}),
               std::runtime_error);
  EXPECT_THROW(model.energies()->block({axes::alpha()}), std::runtime_error);
  EXPECT_THROW(model.energies()->block({axes::beta()}), std::runtime_error);
  EXPECT_THROW(model.get_overlap_matrix(), std::runtime_error);

  Eigen::VectorXd occupations = Eigen::VectorXd::Ones(basis_size);
  EXPECT_THROW(model.calculate_ao_density_matrix(occupations),
               std::runtime_error);
  EXPECT_THROW(model.calculate_ao_density_matrix(occupations, occupations),
               std::runtime_error);

  Eigen::MatrixXd rdm = Eigen::MatrixXd::Identity(basis_size, basis_size);
  EXPECT_THROW(model.calculate_ao_density_matrix_from_rdm(rdm),
               std::runtime_error);
  EXPECT_THROW(model.calculate_ao_density_matrix_from_rdm(rdm, rdm),
               std::runtime_error);
}

TEST_F(ModelOrbitalsTest, MOOverlapMethods) {
  // MO overlap methods return identity matrices.
  const size_t basis_size = 4;
  ModelOrbitals model(basis_size);

  auto [alpha_alpha, alpha_beta, beta_beta] = model.get_mo_overlap();
  Eigen::MatrixXd expected_identity =
      Eigen::MatrixXd::Identity(basis_size, basis_size);

  EXPECT_TRUE(alpha_alpha.isApprox(expected_identity,
                                   testing::numerical_zero_tolerance));
  EXPECT_TRUE(alpha_beta.isApprox(expected_identity,
                                  testing::numerical_zero_tolerance));
  EXPECT_TRUE(
      beta_beta.isApprox(expected_identity, testing::numerical_zero_tolerance));

  EXPECT_TRUE(model.get_mo_overlap_alpha_alpha().isApprox(
      expected_identity, testing::numerical_zero_tolerance));
  EXPECT_TRUE(model.get_mo_overlap_alpha_beta().isApprox(
      expected_identity, testing::numerical_zero_tolerance));
  EXPECT_TRUE(model.get_mo_overlap_beta_beta().isApprox(
      expected_identity, testing::numerical_zero_tolerance));
}

TEST_F(ModelOrbitalsTest, VirtualSpaceIndices) {
  // Virtual space = modes not in active or inactive.
  const size_t basis_size = 6;
  std::vector<size_t> active_indices = {1, 2, 3};
  std::vector<size_t> inactive_indices = {0, 4};

  ModelOrbitals model(trivial_iset(basis_size, active_indices),
                      trivial_iset(basis_size, inactive_indices));

  auto [alpha_virtual, beta_virtual] = model.get_virtual_space_indices();

  std::vector<size_t> expected_virtual = {5};
  EXPECT_EQ(expected_virtual, alpha_virtual);
  EXPECT_EQ(expected_virtual, beta_virtual);
}

TEST_F(ModelOrbitalsTest, JSONSerialization) {
  // JSON serialization for a (restricted) active-space model.
  const size_t basis_size = 4;
  std::vector<size_t> active_indices = {1, 2};
  std::vector<size_t> inactive_indices = {0, 3};

  ModelOrbitals model(trivial_iset(basis_size, active_indices),
                      trivial_iset(basis_size, inactive_indices));

  auto json_data = model.to_json();
  EXPECT_FALSE(json_data.empty());

  EXPECT_TRUE(json_data.contains("num_orbitals"));
  EXPECT_EQ(basis_size, json_data["num_orbitals"].get<size_t>());

  EXPECT_TRUE(json_data.contains("is_restricted"));
  EXPECT_TRUE(json_data["is_restricted"].get<bool>());

  EXPECT_TRUE(json_data.contains("active_space_indices"));
  EXPECT_TRUE(json_data["active_space_indices"].contains("alpha"));
  EXPECT_TRUE(json_data["active_space_indices"].contains("beta"));

  auto json_alpha_active =
      json_data["active_space_indices"]["alpha"].get<std::vector<size_t>>();
  auto json_beta_active =
      json_data["active_space_indices"]["beta"].get<std::vector<size_t>>();

  EXPECT_EQ(active_indices, json_alpha_active);
  EXPECT_EQ(active_indices, json_beta_active);
}

TEST_F(ModelOrbitalsTest, JSONRoundTrip) {
  // JSON round-trip for an unrestricted model with distinct alpha/beta spaces.
  const size_t basis_size = 5;
  std::vector<size_t> alpha_active = {1, 2};
  std::vector<size_t> beta_active = {0, 3, 4};
  std::vector<size_t> alpha_inactive = {0, 3, 4};
  std::vector<size_t> beta_inactive = {1, 2};

  ModelOrbitals original(
      spin_iset(basis_size, alpha_active, beta_active, /*equivalent=*/false),
      spin_iset(basis_size, alpha_inactive, beta_inactive,
                /*equivalent=*/false));

  auto json_data = original.to_json();
  auto reconstructed = ModelOrbitals::from_json(json_data);

  EXPECT_EQ(original.get_num_atomic_orbitals(),
            reconstructed->get_num_atomic_orbitals());
  EXPECT_EQ(original.get_num_molecular_orbitals(),
            reconstructed->get_num_molecular_orbitals());
  EXPECT_EQ(original.is_restricted(), reconstructed->is_restricted());

  auto orig_alpha_active =
      spin_channel_indices(original.active_indices(), axes::alpha());
  auto orig_beta_active =
      spin_channel_indices(original.active_indices(), axes::beta());
  auto recon_alpha_active =
      spin_channel_indices(reconstructed->active_indices(), axes::alpha());
  auto recon_beta_active =
      spin_channel_indices(reconstructed->active_indices(), axes::beta());

  EXPECT_EQ(orig_alpha_active, recon_alpha_active);
  EXPECT_EQ(orig_beta_active, recon_beta_active);

  auto orig_alpha_inactive =
      spin_channel_indices(original.inactive_indices(), axes::alpha());
  auto orig_beta_inactive =
      spin_channel_indices(original.inactive_indices(), axes::beta());
  auto recon_alpha_inactive =
      spin_channel_indices(reconstructed->inactive_indices(), axes::alpha());
  auto recon_beta_inactive =
      spin_channel_indices(reconstructed->inactive_indices(), axes::beta());

  EXPECT_EQ(orig_alpha_inactive, recon_alpha_inactive);
  EXPECT_EQ(orig_beta_inactive, recon_beta_inactive);
}

TEST_F(ModelOrbitalsTest, SimpleRestrictedJSONRoundTrip) {
  // JSON round-trip for the simple full-active case.
  const size_t basis_size = 3;
  ModelOrbitals original(basis_size);

  auto json_data = original.to_json();
  auto reconstructed = ModelOrbitals::from_json(json_data);

  EXPECT_EQ(original.get_num_atomic_orbitals(),
            reconstructed->get_num_atomic_orbitals());
  EXPECT_EQ(original.get_num_molecular_orbitals(),
            reconstructed->get_num_molecular_orbitals());
  EXPECT_EQ(original.is_restricted(), reconstructed->is_restricted());
}

TEST_F(ModelOrbitalsTest, EdgeCaseEmptySpaces) {
  // Empty active and inactive spaces are permitted.
  const size_t basis_size = 3;

  EXPECT_NO_THROW(ModelOrbitals(trivial_iset(basis_size, {}),
                                trivial_iset(basis_size, {})));

  ModelOrbitals model(trivial_iset(basis_size, {}),
                      trivial_iset(basis_size, {}));

  // With empty active/inactive spaces, all modes are virtual.
  auto [alpha_virtual, beta_virtual] = model.get_virtual_space_indices();
  std::vector<size_t> expected_all_indices = {0, 1, 2};
  EXPECT_EQ(expected_all_indices, alpha_virtual);
  EXPECT_EQ(expected_all_indices, beta_virtual);
}

TEST_F(ModelOrbitalsTest, LargerSystem) {
  // Larger system to exercise scalability.
  const size_t basis_size = 20;
  std::vector<size_t> active_indices = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  std::vector<size_t> inactive_indices = {0, 1, 2, 3, 4};

  ModelOrbitals model(trivial_iset(basis_size, active_indices),
                      trivial_iset(basis_size, inactive_indices));

  EXPECT_EQ(basis_size, model.get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, model.get_num_molecular_orbitals());

  auto alpha_active =
      spin_channel_indices(model.active_indices(), axes::alpha());
  auto beta_active = spin_channel_indices(model.active_indices(), axes::beta());
  EXPECT_EQ(active_indices, alpha_active);
  EXPECT_EQ(active_indices, beta_active);

  auto alpha_inactive =
      spin_channel_indices(model.inactive_indices(), axes::alpha());
  auto beta_inactive =
      spin_channel_indices(model.inactive_indices(), axes::beta());
  EXPECT_EQ(inactive_indices, alpha_inactive);
  EXPECT_EQ(inactive_indices, beta_inactive);

  auto [alpha_virtual, beta_virtual] = model.get_virtual_space_indices();
  std::vector<size_t> expected_virtual = {15, 16, 17, 18, 19};
  EXPECT_EQ(expected_virtual, alpha_virtual);
  EXPECT_EQ(expected_virtual, beta_virtual);
}

TEST_F(ModelOrbitalsTest, InheritanceBehavior) {
  // ModelOrbitals is usable through an Orbitals pointer.
  const size_t basis_size = 4;
  ModelOrbitals model(basis_size);

  std::shared_ptr<Orbitals> orbitals_ptr =
      std::make_shared<ModelOrbitals>(model);

  EXPECT_EQ(basis_size, orbitals_ptr->get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, orbitals_ptr->get_num_molecular_orbitals());
  EXPECT_TRUE(orbitals_ptr->is_restricted());

  EXPECT_THROW(orbitals_ptr->coefficients(), std::runtime_error);
  EXPECT_THROW(orbitals_ptr->energies(), std::runtime_error);
}

TEST_F(ModelOrbitalsTest, CopyAndMoveSemantics) {
  // Copy and assignment preserve the model's spaces and symmetry.
  const size_t basis_size = 3;
  std::vector<size_t> active_indices = {0, 1};
  std::vector<size_t> inactive_indices = {2};

  ModelOrbitals original(trivial_iset(basis_size, active_indices),
                         trivial_iset(basis_size, inactive_indices));

  ModelOrbitals copy(original);
  EXPECT_EQ(original.get_num_atomic_orbitals(), copy.get_num_atomic_orbitals());
  EXPECT_EQ(original.get_num_molecular_orbitals(),
            copy.get_num_molecular_orbitals());
  EXPECT_EQ(original.is_restricted(), copy.is_restricted());

  auto orig_alpha_active =
      spin_channel_indices(original.active_indices(), axes::alpha());
  auto orig_beta_active =
      spin_channel_indices(original.active_indices(), axes::beta());
  auto copy_alpha_active =
      spin_channel_indices(copy.active_indices(), axes::alpha());
  auto copy_beta_active =
      spin_channel_indices(copy.active_indices(), axes::beta());
  EXPECT_EQ(orig_alpha_active, copy_alpha_active);
  EXPECT_EQ(orig_beta_active, copy_beta_active);

  // Assignment from a different initial state.
  ModelOrbitals assigned(2, spin_sym(/*equivalent=*/false));
  assigned = original;
  EXPECT_EQ(original.get_num_atomic_orbitals(),
            assigned.get_num_atomic_orbitals());
  EXPECT_EQ(original.is_restricted(), assigned.is_restricted());
}

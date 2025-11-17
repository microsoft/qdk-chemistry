// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <qdk/chemistry/data/orbitals.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

class ModelOrbitalsTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

TEST_F(ModelOrbitalsTest, BasicConstructor) {
  // Test basic constructor with restricted and unrestricted cases
  const size_t basis_size = 4;

  // Restricted case
  ModelOrbitals model_restricted(basis_size, true);
  EXPECT_TRUE(model_restricted.is_restricted());
  EXPECT_FALSE(model_restricted.is_unrestricted());
  EXPECT_EQ(basis_size, model_restricted.get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, model_restricted.get_num_molecular_orbitals());

  // Unrestricted case
  ModelOrbitals model_unrestricted(basis_size, false);
  EXPECT_FALSE(model_unrestricted.is_restricted());
  EXPECT_TRUE(model_unrestricted.is_unrestricted());
  EXPECT_EQ(basis_size, model_unrestricted.get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, model_unrestricted.get_num_molecular_orbitals());
}

TEST_F(ModelOrbitalsTest, RestrictedActiveSpaceConstructor) {
  // Test constructor with active and inactive space indices (restricted)
  const size_t basis_size = 6;
  std::vector<size_t> active_indices = {1, 2, 3};
  std::vector<size_t> inactive_indices = {0, 4, 5};

  ModelOrbitals model(basis_size,
                      std::make_tuple(active_indices, inactive_indices));

  EXPECT_TRUE(model.is_restricted());
  EXPECT_EQ(basis_size, model.get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, model.get_num_molecular_orbitals());

  // Check active space
  EXPECT_TRUE(model.has_active_space());
  auto [alpha_active, beta_active] = model.get_active_space_indices();
  EXPECT_EQ(active_indices, alpha_active);
  EXPECT_EQ(active_indices, beta_active);  // Should be same for restricted

  // Check inactive space
  EXPECT_TRUE(model.has_inactive_space());
  auto [alpha_inactive, beta_inactive] = model.get_inactive_space_indices();
  EXPECT_EQ(inactive_indices, alpha_inactive);
  EXPECT_EQ(inactive_indices, beta_inactive);  // Should be same for restricted
}

TEST_F(ModelOrbitalsTest, UnrestrictedActiveSpaceConstructor) {
  // Test constructor with different alpha/beta active and inactive spaces
  const size_t basis_size = 6;
  std::vector<size_t> alpha_active = {1, 2};
  std::vector<size_t> beta_active = {2, 3, 4};
  std::vector<size_t> alpha_inactive = {0, 3, 4, 5};
  std::vector<size_t> beta_inactive = {0, 1, 5};

  ModelOrbitals model(basis_size,
                      std::make_tuple(alpha_active, beta_active, alpha_inactive,
                                      beta_inactive));

  EXPECT_FALSE(model.is_restricted());
  EXPECT_TRUE(model.is_unrestricted());
  EXPECT_EQ(basis_size, model.get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, model.get_num_molecular_orbitals());

  // Check active space
  EXPECT_TRUE(model.has_active_space());
  auto [retrieved_alpha_active, retrieved_beta_active] =
      model.get_active_space_indices();
  EXPECT_EQ(alpha_active, retrieved_alpha_active);
  EXPECT_EQ(beta_active, retrieved_beta_active);

  // Check inactive space
  EXPECT_TRUE(model.has_inactive_space());
  auto [retrieved_alpha_inactive, retrieved_beta_inactive] =
      model.get_inactive_space_indices();
  EXPECT_EQ(alpha_inactive, retrieved_alpha_inactive);
  EXPECT_EQ(beta_inactive, retrieved_beta_inactive);
}

TEST_F(ModelOrbitalsTest, ConstructorValidation) {
  const size_t basis_size = 4;

  // Test that indices >= basis_size throw exceptions
  std::vector<size_t> invalid_active = {0, 1, 4};  // 4 >= basis_size
  std::vector<size_t> valid_inactive = {2, 3};

  EXPECT_THROW(
      ModelOrbitals(basis_size, std::make_tuple(std::move(invalid_active),
                                                std::move(valid_inactive))),
      std::invalid_argument);

  // Test invalid inactive indices
  std::vector<size_t> valid_active = {0, 1};
  std::vector<size_t> invalid_inactive = {2, 3, 5};  // 5 >= basis_size

  EXPECT_THROW(
      ModelOrbitals(basis_size, std::make_tuple(std::move(valid_active),
                                                std::move(invalid_inactive))),
      std::invalid_argument);

  // Test overlap between active and inactive spaces
  std::vector<size_t> overlapping_active = {0, 1, 2};
  std::vector<size_t> overlapping_inactive = {2, 3};  // 2 appears in both

  EXPECT_THROW(ModelOrbitals(basis_size,
                             std::make_tuple(std::move(overlapping_active),
                                             std::move(overlapping_inactive))),
               std::invalid_argument);
}

TEST_F(ModelOrbitalsTest, UnrestrictedConstructorValidation) {
  const size_t basis_size = 4;
  std::vector<size_t> valid_alpha_active = {0, 1};
  std::vector<size_t> valid_beta_active = {0, 1};
  std::vector<size_t> valid_alpha_inactive = {2, 3};
  std::vector<size_t> valid_beta_inactive = {2, 3};

  // Test invalid alpha active indices
  std::vector<size_t> invalid_alpha_active = {0, 1, 4};  // 4 >= basis_size
  EXPECT_THROW(ModelOrbitals(basis_size,
                             std::make_tuple(std::move(invalid_alpha_active),
                                             std::move(valid_beta_active),
                                             std::move(valid_alpha_inactive),
                                             std::move(valid_beta_inactive))),
               std::invalid_argument);

  // Test invalid beta active indices
  std::vector<size_t> invalid_beta_active = {0, 1, 5};  // 5 >= basis_size
  EXPECT_THROW(ModelOrbitals(basis_size,
                             std::make_tuple(std::move(valid_alpha_active),
                                             std::move(invalid_beta_active),
                                             std::move(valid_alpha_inactive),
                                             std::move(valid_beta_inactive))),
               std::invalid_argument);

  // Test overlap in alpha space
  std::vector<size_t> overlapping_alpha_active = {0, 1, 2};
  std::vector<size_t> overlapping_alpha_inactive = {2, 3};  // 2 appears in both
  EXPECT_THROW(
      ModelOrbitals(basis_size,
                    std::make_tuple(std::move(overlapping_alpha_active),
                                    std::move(valid_beta_active),
                                    std::move(overlapping_alpha_inactive),
                                    std::move(valid_beta_inactive))),
      std::invalid_argument);

  // Test overlap in beta space
  std::vector<size_t> overlapping_beta_active = {0, 1, 2};
  std::vector<size_t> overlapping_beta_inactive = {2, 3};  // 2 appears in both
  EXPECT_THROW(
      ModelOrbitals(basis_size,
                    std::make_tuple(std::move(valid_alpha_active),
                                    std::move(overlapping_beta_active),
                                    std::move(valid_alpha_inactive),
                                    std::move(overlapping_beta_inactive))),
      std::invalid_argument);
}

TEST_F(ModelOrbitalsTest, DefaultActiveSpace) {
  // Test that default constructor sets all orbitals as active
  const size_t basis_size = 5;

  ModelOrbitals model_restricted(basis_size, true);
  EXPECT_TRUE(model_restricted.has_active_space());

  auto [alpha_active, beta_active] =
      model_restricted.get_active_space_indices();
  EXPECT_EQ(basis_size, alpha_active.size());
  EXPECT_EQ(basis_size, beta_active.size());

  // Check that indices are 0, 1, 2, ..., basis_size-1
  for (size_t i = 0; i < basis_size; ++i) {
    EXPECT_EQ(i, alpha_active[i]);
    EXPECT_EQ(i, beta_active[i]);
  }

  // Inactive space should be empty
  EXPECT_FALSE(model_restricted.has_inactive_space());
}

TEST_F(ModelOrbitalsTest, ThrowingMethods) {
  // Test that methods that should throw for model systems actually do
  const size_t basis_size = 3;
  ModelOrbitals model(basis_size, true);

  // These methods should throw runtime_error for model systems
  EXPECT_THROW(model.get_coefficients(), std::runtime_error);
  EXPECT_THROW(model.get_energies(), std::runtime_error);
  EXPECT_THROW(model.get_overlap_matrix(), std::runtime_error);
  EXPECT_THROW(model.get_basis_set(), std::runtime_error);
  EXPECT_THROW(model.get_coefficients_alpha(), std::runtime_error);
  EXPECT_THROW(model.get_coefficients_beta(), std::runtime_error);
  EXPECT_THROW(model.get_energies_alpha(), std::runtime_error);
  EXPECT_THROW(model.get_energies_beta(), std::runtime_error);
  EXPECT_THROW(model.get_overlap_matrix(), std::runtime_error);

  // Density matrix methods should also throw
  Eigen::VectorXd occupations = Eigen::VectorXd::Ones(basis_size);
  EXPECT_THROW(model.calculate_ao_density_matrix(occupations),
               std::runtime_error);
  EXPECT_THROW(model.calculate_ao_density_matrix(occupations, occupations),
               std::runtime_error);

  // RDM-based density matrix methods should also throw
  Eigen::MatrixXd rdm = Eigen::MatrixXd::Identity(basis_size, basis_size);
  EXPECT_THROW(model.calculate_ao_density_matrix_from_rdm(rdm),
               std::runtime_error);
  EXPECT_THROW(model.calculate_ao_density_matrix_from_rdm(rdm, rdm),
               std::runtime_error);
}

TEST_F(ModelOrbitalsTest, MOOverlapMethods) {
  // Test that MO overlap methods return identity matrices
  const size_t basis_size = 4;
  ModelOrbitals model(basis_size, true);

  // Test get_mo_overlap
  auto [alpha_alpha, alpha_beta, beta_beta] = model.get_mo_overlap();
  Eigen::MatrixXd expected_identity =
      Eigen::MatrixXd::Identity(basis_size, basis_size);

  EXPECT_TRUE(alpha_alpha.isApprox(expected_identity,
                                   testing::numerical_zero_tolerance));
  EXPECT_TRUE(alpha_beta.isApprox(expected_identity,
                                  testing::numerical_zero_tolerance));
  EXPECT_TRUE(
      beta_beta.isApprox(expected_identity, testing::numerical_zero_tolerance));

  // Test individual overlap methods
  EXPECT_TRUE(model.get_mo_overlap_alpha_alpha().isApprox(
      expected_identity, testing::numerical_zero_tolerance));
  EXPECT_TRUE(model.get_mo_overlap_alpha_beta().isApprox(
      expected_identity, testing::numerical_zero_tolerance));
  EXPECT_TRUE(model.get_mo_overlap_beta_beta().isApprox(
      expected_identity, testing::numerical_zero_tolerance));
}

TEST_F(ModelOrbitalsTest, VirtualSpaceIndices) {
  // Test virtual space indices calculation
  const size_t basis_size = 6;
  std::vector<size_t> active_indices = {1, 2, 3};
  std::vector<size_t> inactive_indices = {0, 4};
  // Virtual space should be {5} (remaining index)

  ModelOrbitals model(basis_size, std::make_tuple(std::move(active_indices),
                                                  std::move(inactive_indices)));

  auto [alpha_virtual, beta_virtual] = model.get_virtual_space_indices();

  // Expected virtual space: all indices not in active or inactive
  std::vector<size_t> expected_virtual = {5};
  EXPECT_EQ(expected_virtual, alpha_virtual);
  EXPECT_EQ(expected_virtual, beta_virtual);  // Same for restricted
}

TEST_F(ModelOrbitalsTest, JSONSerialization) {
  // Test JSON serialization for ModelOrbitals
  const size_t basis_size = 4;
  std::vector<size_t> active_indices = {1, 2};
  std::vector<size_t> inactive_indices = {0, 3};

  ModelOrbitals model(basis_size,
                      std::make_tuple(active_indices, inactive_indices));

  // Test JSON conversion
  auto json_data = model.to_json();
  EXPECT_FALSE(json_data.empty());

  // Check that essential fields are present
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
  // Test JSON round-trip serialization
  const size_t basis_size = 5;
  std::vector<size_t> alpha_active = {1, 2};
  std::vector<size_t> beta_active = {0, 3, 4};
  std::vector<size_t> alpha_inactive = {0, 3, 4};
  std::vector<size_t> beta_inactive = {1, 2};

  // Keep copies for comparison
  std::vector<size_t> expected_alpha_active = alpha_active;
  std::vector<size_t> expected_beta_active = beta_active;
  std::vector<size_t> expected_alpha_inactive = alpha_inactive;
  std::vector<size_t> expected_beta_inactive = beta_inactive;

  ModelOrbitals original(
      basis_size,
      std::make_tuple(std::move(alpha_active), std::move(beta_active),
                      std::move(alpha_inactive), std::move(beta_inactive)));

  // Convert to JSON and back
  auto json_data = original.to_json();
  auto reconstructed = ModelOrbitals::from_json(json_data);

  // Check that properties are preserved
  EXPECT_EQ(original.get_num_atomic_orbitals(),
            reconstructed->get_num_atomic_orbitals());
  EXPECT_EQ(original.get_num_molecular_orbitals(),
            reconstructed->get_num_molecular_orbitals());
  EXPECT_EQ(original.is_restricted(), reconstructed->is_restricted());

  // Check active space indices
  auto [orig_alpha_active, orig_beta_active] =
      original.get_active_space_indices();
  auto [recon_alpha_active, recon_beta_active] =
      reconstructed->get_active_space_indices();

  EXPECT_EQ(orig_alpha_active, recon_alpha_active);
  EXPECT_EQ(orig_beta_active, recon_beta_active);

  // Check inactive space indices
  auto [orig_alpha_inactive, orig_beta_inactive] =
      original.get_inactive_space_indices();
  auto [recon_alpha_inactive, recon_beta_inactive] =
      reconstructed->get_inactive_space_indices();

  EXPECT_EQ(orig_alpha_inactive, recon_alpha_inactive);
  EXPECT_EQ(orig_beta_inactive, recon_beta_inactive);
}

TEST_F(ModelOrbitalsTest, SimpleRestrictedJSONRoundTrip) {
  // Test JSON round-trip for simple restricted case
  const size_t basis_size = 3;
  ModelOrbitals original(basis_size, true);

  auto json_data = original.to_json();
  auto reconstructed = ModelOrbitals::from_json(json_data);

  EXPECT_EQ(original.get_num_atomic_orbitals(),
            reconstructed->get_num_atomic_orbitals());
  EXPECT_EQ(original.get_num_molecular_orbitals(),
            reconstructed->get_num_molecular_orbitals());
  EXPECT_EQ(original.is_restricted(), reconstructed->is_restricted());
}

TEST_F(ModelOrbitalsTest, EdgeCaseEmptySpaces) {
  // Test edge case with empty active/inactive spaces
  const size_t basis_size = 3;
  std::vector<size_t> empty_indices;

  // Should not throw for empty spaces
  EXPECT_NO_THROW(ModelOrbitals(
      basis_size,
      std::make_tuple(std::move(empty_indices), std::vector<size_t>{})));

  ModelOrbitals model(basis_size, std::make_tuple(std::vector<size_t>{},
                                                  std::vector<size_t>{}));

  // With empty active/inactive spaces, virtual space should contain all indices
  auto [alpha_virtual, beta_virtual] = model.get_virtual_space_indices();
  std::vector<size_t> expected_all_indices = {0, 1, 2};
  EXPECT_EQ(expected_all_indices, alpha_virtual);
  EXPECT_EQ(expected_all_indices, beta_virtual);
}

TEST_F(ModelOrbitalsTest, LargerSystem) {
  // Test with a larger system to ensure scalability
  const size_t basis_size = 20;
  std::vector<size_t> active_indices = {5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
  std::vector<size_t> inactive_indices = {0, 1, 2, 3, 4};
  // Virtual space should be {15, 16, 17, 18, 19}

  ModelOrbitals model(basis_size,
                      std::make_tuple(active_indices, inactive_indices));

  EXPECT_EQ(basis_size, model.get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, model.get_num_molecular_orbitals());

  auto [alpha_active, beta_active] = model.get_active_space_indices();
  EXPECT_EQ(active_indices, alpha_active);
  EXPECT_EQ(active_indices, beta_active);

  auto [alpha_inactive, beta_inactive] = model.get_inactive_space_indices();
  EXPECT_EQ(inactive_indices, alpha_inactive);
  EXPECT_EQ(inactive_indices, beta_inactive);

  auto [alpha_virtual, beta_virtual] = model.get_virtual_space_indices();
  std::vector<size_t> expected_virtual = {15, 16, 17, 18, 19};
  EXPECT_EQ(expected_virtual, alpha_virtual);
  EXPECT_EQ(expected_virtual, beta_virtual);
}

TEST_F(ModelOrbitalsTest, InheritanceBehavior) {
  // Test that ModelOrbitals properly inherits from Orbitals
  const size_t basis_size = 4;
  ModelOrbitals model(basis_size, true);

  // Test that we can use it as an Orbitals pointer
  std::shared_ptr<Orbitals> orbitals_ptr =
      std::make_shared<ModelOrbitals>(model);

  EXPECT_EQ(basis_size, orbitals_ptr->get_num_atomic_orbitals());
  EXPECT_EQ(basis_size, orbitals_ptr->get_num_molecular_orbitals());
  EXPECT_TRUE(orbitals_ptr->is_restricted());

  // Test that methods that should throw still throw through base pointer
  EXPECT_THROW(orbitals_ptr->get_coefficients(), std::runtime_error);
  EXPECT_THROW(orbitals_ptr->get_energies(), std::runtime_error);
}

TEST_F(ModelOrbitalsTest, CopyAndMoveSemantics) {
  // Test copy and move behavior
  const size_t basis_size = 3;
  std::vector<size_t> active_indices = {0, 1};
  std::vector<size_t> inactive_indices = {2};

  ModelOrbitals original(
      basis_size,
      std::make_tuple(std::move(active_indices), std::move(inactive_indices)));

  // Test copy constructor
  ModelOrbitals copy(original);
  EXPECT_EQ(original.get_num_atomic_orbitals(), copy.get_num_atomic_orbitals());
  EXPECT_EQ(original.get_num_molecular_orbitals(),
            copy.get_num_molecular_orbitals());
  EXPECT_EQ(original.is_restricted(), copy.is_restricted());

  // Test that active/inactive spaces are preserved
  auto [orig_alpha_active, orig_beta_active] =
      original.get_active_space_indices();
  auto [copy_alpha_active, copy_beta_active] = copy.get_active_space_indices();
  EXPECT_EQ(orig_alpha_active, copy_alpha_active);
  EXPECT_EQ(orig_beta_active, copy_beta_active);

  // Test assignment
  ModelOrbitals assigned(2, false);  // Different initial state
  assigned = original;
  EXPECT_EQ(original.get_num_atomic_orbitals(),
            assigned.get_num_atomic_orbitals());
  EXPECT_EQ(original.is_restricted(), assigned.is_restricted());
}

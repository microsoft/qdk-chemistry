// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <cstdio>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/algorithms/stability.hpp>
#include <qdk/chemistry/data/stability_result.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class StabilityCheckerTest : public ::testing::Test {
 protected:
  void SetUp() override {}

  void TearDown() override {}
};

/**
 * @brief Test implementation of StabilityChecker for testing purposes
 */
class TestStabilityChecker : public StabilityChecker {
 public:
  TestStabilityChecker() : StabilityChecker() {}

  std::string name() const override { return "TestStabilityChecker"; }

 protected:
  std::pair<bool, std::shared_ptr<StabilityResult>> _run_impl(
      std::shared_ptr<qdk::chemistry::data::Wavefunction> wavefunction)
      const override {
    // Create dummy results for testing
    Eigen::VectorXd internal_eigenvalues(3);
    internal_eigenvalues << 0.5, 1.0, 1.5;  // All positive = stable

    Eigen::VectorXd external_eigenvalues(2);
    external_eigenvalues << 0.3, 0.8;  // All positive = stable

    Eigen::MatrixXd internal_eigenvectors = Eigen::MatrixXd::Identity(3, 3);
    Eigen::MatrixXd external_eigenvectors = Eigen::MatrixXd::Identity(2, 2);

    bool internal_stable = (internal_eigenvalues.array() > 0.0).all();
    bool external_stable = (external_eigenvalues.array() > 0.0).all();

    auto result = std::make_shared<StabilityResult>(
        internal_stable, external_stable, internal_eigenvalues,
        internal_eigenvectors, external_eigenvalues, external_eigenvectors);

    return std::make_pair(result->is_stable(), result);
  }
};

TEST_F(StabilityCheckerTest, StabilityChecker_MetaData) {
  // Register a test implementation first
  StabilityCheckerFactory::register_instance(
      []() -> StabilityCheckerFactory::return_type {
        return std::make_unique<TestStabilityChecker>();
      });

  auto selector = StabilityCheckerFactory::create("TestStabilityChecker");
  EXPECT_NO_THROW({ auto settings = selector->settings(); });

  // Clean up
  StabilityCheckerFactory::unregister_instance("TestStabilityChecker");
}

TEST_F(StabilityCheckerTest, StabilityResult) {
  // Test StabilityResult functionality
  TestStabilityChecker checker;

  // Create a dummy wavefunction for testing using SlaterDeterminantContainer
  auto wavefunction = std::make_shared<qdk::chemistry::data::Wavefunction>(
      std::make_unique<SlaterDeterminantContainer>(
          Configuration("200000"), testing::create_test_orbitals()));

  // Get the result
  auto [is_stable, result] = checker.run(wavefunction);

  // Test the result - should be stable since both internal and external are
  // stable
  EXPECT_TRUE(is_stable);
  EXPECT_TRUE(result->is_stable());
  EXPECT_TRUE(result->is_internal_stable());
  EXPECT_TRUE(result->is_external_stable());

  // Test eigenvalue access
  EXPECT_EQ(result->get_internal_eigenvalues().size(), 3);
  EXPECT_EQ(result->get_external_eigenvalues().size(), 2);
  EXPECT_EQ(result->get_internal_eigenvectors().rows(), 3);
  EXPECT_EQ(result->get_internal_eigenvectors().cols(), 3);
  EXPECT_EQ(result->get_external_eigenvectors().rows(), 2);
  EXPECT_EQ(result->get_external_eigenvectors().cols(), 2);

  // Test size methods
  EXPECT_EQ(result->internal_size(), 3);
  EXPECT_EQ(result->external_size(), 2);

  // Test smallest eigenvalue methods
  EXPECT_DOUBLE_EQ(result->get_smallest_internal_eigenvalue(), 0.5);
  EXPECT_DOUBLE_EQ(result->get_smallest_external_eigenvalue(), 0.3);
  EXPECT_DOUBLE_EQ(result->get_smallest_eigenvalue(), 0.3);  // Overall smallest

  // Test eigenvalue-eigenvector pair methods
  auto internal_pair = result->get_smallest_internal_eigenvalue_and_vector();
  EXPECT_DOUBLE_EQ(internal_pair.first, 0.5);
  EXPECT_EQ(internal_pair.second.size(), 3);

  auto external_pair = result->get_smallest_external_eigenvalue_and_vector();
  EXPECT_DOUBLE_EQ(external_pair.first, 0.3);
  EXPECT_EQ(external_pair.second.size(), 2);

  auto overall_pair = result->get_smallest_eigenvalue_and_vector();
  EXPECT_DOUBLE_EQ(overall_pair.first, 0.3);
  EXPECT_EQ(overall_pair.second.size(), 2);  // Should be from external

  // Test setting values
  result->set_internal_stable(false);
  EXPECT_FALSE(result->is_internal_stable());
  EXPECT_FALSE(result->is_stable());  // Overall should be false now

  result->set_external_stable(false);
  EXPECT_FALSE(result->is_external_stable());
  EXPECT_FALSE(result->is_stable());

  result->set_internal_stable(true);
  EXPECT_TRUE(result->is_internal_stable());
  EXPECT_FALSE(result->is_stable());  // Still false because external is false

  result->set_external_stable(true);
  EXPECT_TRUE(result->is_external_stable());
  EXPECT_TRUE(result->is_stable());  // Now both are true

  // Test setting new eigenvalues
  Eigen::VectorXd new_internal_eigenvalues(2);
  new_internal_eigenvalues << -0.1, 0.2;
  result->set_internal_eigenvalues(new_internal_eigenvalues);
  EXPECT_EQ(result->get_internal_eigenvalues().size(), 2);
  EXPECT_DOUBLE_EQ(result->get_smallest_internal_eigenvalue(), -0.1);

  Eigen::VectorXd new_external_eigenvalues(3);
  new_external_eigenvalues << 0.1, -0.5, 0.3;
  result->set_external_eigenvalues(new_external_eigenvalues);
  EXPECT_EQ(result->get_external_eigenvalues().size(), 3);
  EXPECT_DOUBLE_EQ(result->get_smallest_external_eigenvalue(), -0.5);
  EXPECT_DOUBLE_EQ(result->get_smallest_eigenvalue(),
                   -0.5);  // Overall smallest

  // Test setting new eigenvectors
  Eigen::MatrixXd new_internal_eigenvectors = Eigen::MatrixXd::Identity(2, 2);
  result->set_internal_eigenvectors(new_internal_eigenvectors);
  EXPECT_EQ(result->get_internal_eigenvectors().rows(), 2);
  EXPECT_EQ(result->get_internal_eigenvectors().cols(), 2);

  Eigen::MatrixXd new_external_eigenvectors = Eigen::MatrixXd::Identity(3, 3);
  result->set_external_eigenvectors(new_external_eigenvectors);
  EXPECT_EQ(result->get_external_eigenvectors().rows(), 3);
  EXPECT_EQ(result->get_external_eigenvectors().cols(), 3);
}

TEST_F(StabilityCheckerTest, Factory) {
  auto available_checkers = StabilityCheckerFactory::available();
  EXPECT_EQ(available_checkers.size(), 1);
  EXPECT_EQ(available_checkers[0], "qdk");
  EXPECT_THROW(StabilityCheckerFactory::create("nonexistent_checker"),
               std::runtime_error);

  // First register test_checker successfully
  EXPECT_NO_THROW(StabilityCheckerFactory::register_instance(
      []() -> StabilityCheckerFactory::return_type {
        return std::make_unique<TestStabilityChecker>();
      }));

  // Now registering test_checker again should throw an error
  EXPECT_THROW(StabilityCheckerFactory::register_instance(
                   []() -> StabilityCheckerFactory::return_type {
                     return std::make_unique<TestStabilityChecker>();
                   }),
               std::runtime_error);

  // Test unregister_instance
  // First test unregistering a non-existent key (should return false)
  EXPECT_FALSE(StabilityCheckerFactory::unregister_instance("nonexistent_key"));

  // Test unregistering the same key again (should return false since it's
  // already removed)
  EXPECT_FALSE(StabilityCheckerFactory::unregister_instance("_dummy_checker"));

  // Clean up
  EXPECT_TRUE(
      StabilityCheckerFactory::unregister_instance("TestStabilityChecker"));
}

TEST_F(StabilityCheckerTest, StabilityResult_EdgeCases) {
  // Test StabilityResult with empty eigenvalues
  StabilityResult empty_result;

  // Test default constructor values
  EXPECT_TRUE(empty_result.empty());
  EXPECT_TRUE(empty_result.is_internal_stable());  // Default should be true
  EXPECT_TRUE(empty_result.is_external_stable());  // Default should be true
  EXPECT_TRUE(empty_result.is_stable());           // Both true = overall true

  EXPECT_EQ(empty_result.internal_size(), 0);
  EXPECT_EQ(empty_result.external_size(), 0);

  // Test error handling for empty eigenvalues
  EXPECT_THROW(empty_result.get_smallest_internal_eigenvalue(),
               std::runtime_error);
  EXPECT_THROW(empty_result.get_smallest_external_eigenvalue(),
               std::runtime_error);
  EXPECT_THROW(empty_result.get_smallest_eigenvalue(), std::runtime_error);
  EXPECT_THROW(empty_result.get_smallest_internal_eigenvalue_and_vector(),
               std::runtime_error);
  EXPECT_THROW(empty_result.get_smallest_external_eigenvalue_and_vector(),
               std::runtime_error);
  EXPECT_THROW(empty_result.get_smallest_eigenvalue_and_vector(),
               std::runtime_error);

  // Test with only internal eigenvalues
  Eigen::VectorXd internal_only(2);
  internal_only << -0.2, 0.4;
  Eigen::MatrixXd internal_vecs = Eigen::MatrixXd::Identity(2, 2);

  empty_result.set_internal_eigenvalues(internal_only);
  empty_result.set_internal_eigenvectors(internal_vecs);

  EXPECT_FALSE(empty_result.empty());
  EXPECT_EQ(empty_result.internal_size(), 2);
  EXPECT_EQ(empty_result.external_size(), 0);
  EXPECT_DOUBLE_EQ(empty_result.get_smallest_internal_eigenvalue(), -0.2);
  EXPECT_DOUBLE_EQ(empty_result.get_smallest_eigenvalue(),
                   -0.2);  // Should work with only internal

  // External methods should still throw
  EXPECT_THROW(empty_result.get_smallest_external_eigenvalue(),
               std::runtime_error);
  EXPECT_THROW(empty_result.get_smallest_external_eigenvalue_and_vector(),
               std::runtime_error);

  // Internal methods should work
  EXPECT_NO_THROW(empty_result.get_smallest_internal_eigenvalue_and_vector());
  EXPECT_NO_THROW(empty_result.get_smallest_eigenvalue_and_vector());

  // Test with only external eigenvalues
  StabilityResult external_only_result;
  Eigen::VectorXd external_only(1);
  external_only << 0.7;
  Eigen::MatrixXd external_vecs = Eigen::MatrixXd::Identity(1, 1);

  external_only_result.set_external_eigenvalues(external_only);
  external_only_result.set_external_eigenvectors(external_vecs);

  EXPECT_FALSE(external_only_result.empty());
  EXPECT_EQ(external_only_result.internal_size(), 0);
  EXPECT_EQ(external_only_result.external_size(), 1);
  EXPECT_DOUBLE_EQ(external_only_result.get_smallest_external_eigenvalue(),
                   0.7);
  EXPECT_DOUBLE_EQ(external_only_result.get_smallest_eigenvalue(),
                   0.7);  // Should work with only external

  // Internal methods should throw
  EXPECT_THROW(external_only_result.get_smallest_internal_eigenvalue(),
               std::runtime_error);
  EXPECT_THROW(
      external_only_result.get_smallest_internal_eigenvalue_and_vector(),
      std::runtime_error);

  // External methods should work
  EXPECT_NO_THROW(
      external_only_result.get_smallest_external_eigenvalue_and_vector());
  EXPECT_NO_THROW(external_only_result.get_smallest_eigenvalue_and_vector());
}

TEST_F(StabilityCheckerTest, StabilityResult_DataClass_API) {
  // Test the DataClass interface methods
  Eigen::VectorXd internal_eigenvalues(3);
  internal_eigenvalues << -0.1, 0.3, 0.8;

  Eigen::VectorXd external_eigenvalues(2);
  external_eigenvalues << 0.2, 0.6;

  Eigen::MatrixXd internal_eigenvectors = Eigen::MatrixXd::Identity(3, 3);
  Eigen::MatrixXd external_eigenvectors = Eigen::MatrixXd::Identity(2, 2);

  StabilityResult result(false, true, internal_eigenvalues,
                         internal_eigenvectors, external_eigenvalues,
                         external_eigenvectors);

  // Test convenience methods
  EXPECT_FALSE(result.empty());
  EXPECT_TRUE(result.has_internal_result());
  EXPECT_TRUE(result.has_external_result());

  // Test summary contains validity info
  std::string summary = result.get_summary();
  EXPECT_FALSE(summary.empty());
  EXPECT_TRUE(summary.find("unstable") !=
              std::string::npos);  // Should be unstable due to internal=false
}

TEST_F(StabilityCheckerTest, StabilityResult_JSON_IO) {
  // Create test data
  Eigen::VectorXd internal_eigenvalues(2);
  internal_eigenvalues << -0.2, 0.5;

  Eigen::VectorXd external_eigenvalues(3);
  external_eigenvalues << 0.1, 0.3, 0.7;

  Eigen::MatrixXd internal_eigenvectors(2, 2);
  internal_eigenvectors << 0.8, 0.6, -0.6, 0.8;

  Eigen::MatrixXd external_eigenvectors(3, 3);
  external_eigenvectors.setIdentity();

  StabilityResult original(true, false, internal_eigenvalues,
                           internal_eigenvectors, external_eigenvalues,
                           external_eigenvectors);

  // Test JSON serialization
  auto json_data = original.to_json();
  EXPECT_FALSE(json_data.empty());
  EXPECT_TRUE(json_data.contains("type"));
  EXPECT_EQ("StabilityResult", json_data["type"].get<std::string>());

  // Test JSON round-trip via string
  auto reconstructed = StabilityResult::from_json(json_data);

  // Verify basic properties
  EXPECT_EQ(original.is_internal_stable(), reconstructed->is_internal_stable());
  EXPECT_EQ(original.is_external_stable(), reconstructed->is_external_stable());
  EXPECT_EQ(original.internal_size(), reconstructed->internal_size());
  EXPECT_EQ(original.external_size(), reconstructed->external_size());

  // Verify eigenvalues are preserved
  EXPECT_TRUE(original.get_internal_eigenvalues() ==
              reconstructed->get_internal_eigenvalues());
  EXPECT_TRUE(original.get_external_eigenvalues() ==
              reconstructed->get_external_eigenvalues());

  // Test file I/O
  original.to_json_file("test.stability_result.json");
  auto from_file =
      StabilityResult::from_json_file("test.stability_result.json");

  EXPECT_EQ(original.internal_size(), from_file->internal_size());
  EXPECT_EQ(original.external_size(), from_file->external_size());
  EXPECT_TRUE(original.get_internal_eigenvalues() ==
              from_file->get_internal_eigenvalues());
  EXPECT_TRUE(original.get_external_eigenvalues() ==
              from_file->get_external_eigenvalues());

  // Cleanup temporary file
  std::remove("test.stability_result.json");
}

TEST_F(StabilityCheckerTest, StabilityResult_HDF5_IO) {
  // Create test data
  Eigen::VectorXd internal_eigenvalues(3);
  internal_eigenvalues << 0.1, 0.4, 0.9;

  Eigen::VectorXd external_eigenvalues(1);
  external_eigenvalues << -0.3;

  Eigen::MatrixXd internal_eigenvectors(3, 3);
  internal_eigenvectors.setRandom();

  Eigen::MatrixXd external_eigenvectors(1, 1);
  external_eigenvectors << 1.0;

  StabilityResult original(true, false, internal_eigenvalues,
                           internal_eigenvectors, external_eigenvalues,
                           external_eigenvectors);

  // Test HDF5 file I/O
  original.to_hdf5_file("test.stability_result.h5");
  auto from_hdf5 = StabilityResult::from_hdf5_file("test.stability_result.h5");

  // Verify data preservation
  EXPECT_EQ(original.is_internal_stable(), from_hdf5->is_internal_stable());
  EXPECT_EQ(original.is_external_stable(), from_hdf5->is_external_stable());
  EXPECT_TRUE(original.get_internal_eigenvalues() ==
              from_hdf5->get_internal_eigenvalues());
  EXPECT_TRUE(original.get_external_eigenvalues() ==
              from_hdf5->get_external_eigenvalues());
  EXPECT_TRUE(original.get_internal_eigenvectors() ==
              from_hdf5->get_internal_eigenvectors());
  EXPECT_TRUE(original.get_external_eigenvectors() ==
              from_hdf5->get_external_eigenvectors());

  // Cleanup temporary file
  std::remove("test.stability_result.h5");
}

TEST_F(StabilityCheckerTest, StabilityResult_Generic_File_IO) {
  // Test the generic to_file/from_file methods
  Eigen::VectorXd eigenvalues(2);
  eigenvalues << 0.2, 0.8;

  Eigen::MatrixXd eigenvectors = Eigen::MatrixXd::Identity(2, 2);

  StabilityResult original(true, true, eigenvalues, eigenvectors, eigenvalues,
                           eigenvectors);

  // Test JSON via generic interface
  original.to_file("test.stability_result.json", "json");
  auto from_json =
      StabilityResult::from_file("test.stability_result.json", "json");
  EXPECT_EQ(original.internal_size(), from_json->internal_size());
  EXPECT_TRUE(original.get_external_eigenvectors() ==
              from_json->get_external_eigenvectors());

  // Test HDF5 via generic interface
  original.to_file("test.stability_result.h5", "hdf5");
  auto from_hdf5 =
      StabilityResult::from_file("test.stability_result.h5", "hdf5");
  EXPECT_EQ(original.external_size(), from_hdf5->external_size());
  EXPECT_TRUE(original.get_external_eigenvectors() ==
              from_hdf5->get_external_eigenvectors());

  // Test unsupported format
  EXPECT_THROW(original.to_file("test.stability_result.xyz", "xyz"),
               std::invalid_argument);
  EXPECT_THROW(StabilityResult::from_file("test.stability_result.xyz", "xyz"),
               std::invalid_argument);

  // Cleanup temporary files
  std::remove("test.stability_result.json");
  std::remove("test.stability_result.h5");
}

TEST_F(StabilityCheckerTest, StabilityResult_File_Validation) {
  StabilityResult result;

  // Test filename validation - wrong suffix should throw
  EXPECT_THROW(result.to_json_file("test.json"), std::invalid_argument);
  EXPECT_THROW(result.to_hdf5_file("test.h5"), std::invalid_argument);
  EXPECT_THROW(StabilityResult::from_json_file("test.json"),
               std::invalid_argument);
  EXPECT_THROW(StabilityResult::from_hdf5_file("test.h5"),
               std::invalid_argument);

  // Test non-existent files
  EXPECT_THROW(
      StabilityResult::from_json_file("nonexistent.stability_result.json"),
      std::runtime_error);
  EXPECT_THROW(
      StabilityResult::from_hdf5_file("nonexistent.stability_result.h5"),
      std::runtime_error);
}

TEST_F(StabilityCheckerTest, StabilityResult_Empty_Data_IO) {
  // Test I/O with empty StabilityResult (no eigenvalues/eigenvectors)
  StabilityResult empty_result;
  EXPECT_TRUE(empty_result.is_stable());  // Default should be stable
  EXPECT_TRUE(empty_result.empty());

  // Test JSON I/O with empty data
  auto json_data = empty_result.to_json();
  auto from_json = StabilityResult::from_json(json_data);
  EXPECT_EQ(empty_result.is_stable(), from_json->is_stable());
  EXPECT_EQ(empty_result.internal_size(), from_json->internal_size());
  EXPECT_EQ(empty_result.external_size(), from_json->external_size());

  // Test file I/O with empty data
  empty_result.to_json_file("empty.stability_result.json");
  auto empty_from_file =
      StabilityResult::from_json_file("empty.stability_result.json");
  EXPECT_TRUE(empty_from_file->is_stable());
  EXPECT_TRUE(empty_from_file->empty());

  // Cleanup temporary file
  std::remove("empty.stability_result.json");
}

TEST_F(StabilityCheckerTest, StabilityResult_Validation) {
  // Test validation logic - just test that our empty() methods work
  StabilityResult empty_result;
  EXPECT_TRUE(empty_result.empty());
  EXPECT_FALSE(empty_result.has_internal_result());
  EXPECT_FALSE(empty_result.has_external_result());

  // Add some data and check again
  Eigen::VectorXd eigenvalues(2);
  eigenvalues << 0.1, 0.5;
  Eigen::MatrixXd eigenvectors = Eigen::MatrixXd::Identity(2, 2);

  empty_result.set_internal_eigenvalues(eigenvalues);
  empty_result.set_internal_eigenvectors(eigenvectors);

  EXPECT_FALSE(empty_result.empty());
  EXPECT_TRUE(empty_result.has_internal_result());
  EXPECT_FALSE(empty_result.has_external_result());
}

// Tests for QDK (first-party) stability checker implementation
TEST_F(StabilityCheckerTest, QDK_RHF_Water_HF_Stable) {
  // Test internal and external stability analysis on stable RHF water molecule
  auto water = testing::create_water_structure();

  // Run SCF calculation
  auto scf_solver = ScfSolverFactory::create();
  auto [energy, wavefunction] = scf_solver->run(water, 0, 1, "def2-svp");

  // Create stability checker for full analysis (internal + external)
  auto stability_checker = StabilityCheckerFactory::create("qdk");
  stability_checker->settings().set("internal", true);
  stability_checker->settings().set("external", true);

  // Run stability analysis
  auto [is_stable, result] = stability_checker->run(wavefunction);

  // Verify result properties
  EXPECT_TRUE(result != nullptr);
  EXPECT_TRUE(result->has_internal_result());
  EXPECT_TRUE(result->has_external_result());
  EXPECT_GT(result->internal_size(), 0);
  EXPECT_GT(result->external_size(), 0);

  // Water RHF should be both internally and externally stable
  EXPECT_TRUE(result->is_internal_stable());
  EXPECT_TRUE(result->is_external_stable());
  EXPECT_TRUE(is_stable);

  // Check the smallest internal eigenvalue against reference
  double smallest_internal_eigenvalue =
      result->get_smallest_internal_eigenvalue();
  double davidson_tol =
      stability_checker->settings().get<double>("davidson_tolerance") * 1e2;
  EXPECT_NEAR(smallest_internal_eigenvalue, 0.2978821635793, davidson_tol);

  // Check the smallest external eigenvalue against reference
  double smallest_external_eigenvalue =
      result->get_smallest_external_eigenvalue();
  EXPECT_NEAR(smallest_external_eigenvalue, 0.1798656249185, davidson_tol);
}

TEST_F(StabilityCheckerTest, QDK_RHF_Water_M06_2X_Stable) {
  // Test internal and external stability analysis on stable RHF water molecule
  // using the M06-2X density functional
  auto water = testing::create_water_structure();

  // Run SCF calculation with M06-2X
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("method", "m06-2x");
  auto [energy, wavefunction] = scf_solver->run(water, 0, 1, "def2-svp");

  // Create stability checker for full analysis (internal + external)
  auto stability_checker = StabilityCheckerFactory::create("qdk");
  stability_checker->settings().set("internal", true);
  stability_checker->settings().set("external", true);
  stability_checker->settings().set("method", "m06-2x");

  // Run stability analysis
  auto [is_stable, result] = stability_checker->run(wavefunction);

  // Verify result properties
  EXPECT_TRUE(result != nullptr);
  EXPECT_TRUE(result->has_internal_result());
  EXPECT_TRUE(result->has_external_result());
  EXPECT_GT(result->internal_size(), 0);
  EXPECT_GT(result->external_size(), 0);

  // Water RHF with M06-2X should be both internally and externally stable
  EXPECT_TRUE(result->is_internal_stable());
  EXPECT_TRUE(result->is_external_stable());
  EXPECT_TRUE(is_stable);

  // Check the smallest eigenvalues
  double smallest_internal_eigenvalue =
      result->get_smallest_internal_eigenvalue();
  double davidson_tol =
      stability_checker->settings().get<double>("davidson_tolerance") * 1e2;
  EXPECT_NEAR(smallest_internal_eigenvalue, 0.2628966726304, davidson_tol);

  double smallest_external_eigenvalue =
      result->get_smallest_external_eigenvalue();
  EXPECT_NEAR(smallest_external_eigenvalue, 0.2101707964732, davidson_tol);
}

TEST_F(StabilityCheckerTest, QDK_UHF_O2_HF_Stable) {
  // Test internal stability analysis on UHF oxygen molecule
  auto o2 = testing::create_o2_structure();

  // Run UHF SCF calculation (triplet state)
  auto scf_solver = ScfSolverFactory::create();
  auto [energy, wavefunction] = scf_solver->run(o2, 0, 3, "def2-svp");

  // Verify we have UHF orbitals
  auto orbitals = wavefunction->get_orbitals();
  EXPECT_FALSE(orbitals->is_restricted());

  // Create stability checker for internal-only analysis
  auto stability_checker = StabilityCheckerFactory::create("qdk");
  stability_checker->settings().set("internal", true);
  stability_checker->settings().set("external", false);

  // Run stability analysis
  auto [is_stable, result] = stability_checker->run(wavefunction);

  // Verify result properties
  EXPECT_TRUE(result != nullptr);
  EXPECT_TRUE(result->has_internal_result());
  EXPECT_FALSE(result->has_external_result());
  EXPECT_GT(result->internal_size(), 0);
  EXPECT_EQ(result->external_size(), 0);

  // UHF O2 triplet should be internally stable
  EXPECT_TRUE(result->is_internal_stable());
  EXPECT_TRUE(is_stable);

  // Check the smallest internal eigenvalue against reference
  double smallest_eigenvalue = result->get_smallest_internal_eigenvalue();
  double davidson_tol =
      stability_checker->settings().get<double>("davidson_tolerance") * 1e2;
  EXPECT_NEAR(smallest_eigenvalue, 0.02098145148042, davidson_tol);
}

TEST_F(StabilityCheckerTest, QDK_UHF_O2_M06_2X_Stable) {
  // Test internal stability analysis on UHF oxygen molecule using M06-2X
  auto o2 = testing::create_o2_structure();

  // Run UHF SCF calculation (triplet state) with M06-2X
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("method", "m06-2x");
  auto [energy, wavefunction] = scf_solver->run(o2, 0, 3, "def2-svp");

  // Check SCF energy
  EXPECT_NEAR(energy, -150.14208965738047, testing::scf_energy_tolerance);

  // Verify we have UHF orbitals
  auto orbitals = wavefunction->get_orbitals();
  EXPECT_FALSE(orbitals->is_restricted());

  // Create stability checker for internal-only analysis
  auto stability_checker = StabilityCheckerFactory::create("qdk");
  stability_checker->settings().set("internal", true);
  stability_checker->settings().set("external", false);
  stability_checker->settings().set("method", "m06-2x");

  // Run stability analysis
  auto [is_stable, result] = stability_checker->run(wavefunction);

  // Verify result properties
  EXPECT_TRUE(result != nullptr);
  EXPECT_TRUE(result->has_internal_result());
  EXPECT_FALSE(result->has_external_result());
  EXPECT_GT(result->internal_size(), 0);
  EXPECT_EQ(result->external_size(), 0);

  // UHF O2 triplet should be internally stable
  EXPECT_TRUE(result->is_internal_stable());
  EXPECT_TRUE(is_stable);

  // Check the smallest internal eigenvalue against Python reference
  double smallest_eigenvalue = result->get_smallest_internal_eigenvalue();
  double davidson_tol =
      stability_checker->settings().get<double>("davidson_tolerance") * 1e2;
  EXPECT_NEAR(smallest_eigenvalue, 0.1972664301656, davidson_tol);
}

TEST_F(StabilityCheckerTest, QDK_UHF_O2_PBE_Stable) {
  // Test internal stability analysis on UHF oxygen molecule using PBE
  auto o2 = testing::create_o2_structure();

  // Run UHF SCF calculation (triplet state) with PBE
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("method", "pbe");
  auto [energy, wavefunction] = scf_solver->run(o2, 0, 3, "def2-svp");

  // Check SCF energy
  EXPECT_NEAR(energy, -150.06573508243756, testing::scf_energy_tolerance);

  // Verify we have UHF orbitals
  auto orbitals = wavefunction->get_orbitals();
  EXPECT_FALSE(orbitals->is_restricted());

  // Create stability checker for internal-only analysis
  auto stability_checker = StabilityCheckerFactory::create("qdk");
  stability_checker->settings().set("internal", true);
  stability_checker->settings().set("external", false);
  stability_checker->settings().set("method", "pbe");

  // Run stability analysis
  auto [is_stable, result] = stability_checker->run(wavefunction);

  // Verify result properties
  EXPECT_TRUE(result != nullptr);
  EXPECT_TRUE(result->has_internal_result());
  EXPECT_FALSE(result->has_external_result());
  EXPECT_GT(result->internal_size(), 0);
  EXPECT_EQ(result->external_size(), 0);

  // UHF O2 triplet should be internally stable
  EXPECT_TRUE(result->is_internal_stable());
  EXPECT_TRUE(is_stable);

  // Check the smallest internal eigenvalue against Python reference
  double smallest_eigenvalue = result->get_smallest_internal_eigenvalue();
  double davidson_tol =
      stability_checker->settings().get<double>("davidson_tolerance") * 1e2;
  EXPECT_NEAR(smallest_eigenvalue, 0.2284794502473, davidson_tol);
}

TEST_F(StabilityCheckerTest, QDK_RHF_N2_Stretched_External_Instability) {
  // Test external stability analysis on stretched N2 molecule (RHF)
  // At 1.2 Angstrom, N2 RHF should be internally stable but externally unstable
  auto n2 = testing::create_stretched_n2_structure(1.2);

  // Run RHF SCF calculation
  auto scf_solver = ScfSolverFactory::create();
  auto [energy, wavefunction] = scf_solver->run(n2, 0, 1, "def2-svp");

  // Create stability checker for full analysis (internal + external)
  auto stability_checker = StabilityCheckerFactory::create("qdk");
  stability_checker->settings().set("internal", true);
  stability_checker->settings().set("external", true);

  // Run stability analysis
  auto [is_stable, result] = stability_checker->run(wavefunction);

  // Verify result properties
  EXPECT_TRUE(result != nullptr);
  EXPECT_TRUE(result->has_internal_result());
  EXPECT_TRUE(result->has_external_result());
  EXPECT_GT(result->internal_size(), 0);
  EXPECT_GT(result->external_size(), 0);

  // N2 RHF at 1.2 Angstrom should be internally stable but externally unstable
  EXPECT_TRUE(result->is_internal_stable());
  EXPECT_FALSE(result->is_external_stable());
  EXPECT_FALSE(is_stable);  // Overall unstable due to external instability

  // Check the smallest internal eigenvalue against reference
  double smallest_internal_eigenvalue =
      result->get_smallest_internal_eigenvalue();
  double davidson_tol =
      stability_checker->settings().get<double>("davidson_tolerance") * 1e2;
  EXPECT_NEAR(smallest_internal_eigenvalue, 0.1670995165125, davidson_tol);

  // Check the smallest external eigenvalue against reference
  double smallest_external_eigenvalue =
      result->get_smallest_external_eigenvalue();
  EXPECT_NEAR(smallest_external_eigenvalue, -0.04997551673696, davidson_tol);
}

TEST_F(StabilityCheckerTest, QDK_RHF_N2_Stretched_Internal_Instability) {
  // Test internal stability analysis on stretched N2 molecule (RHF)
  // At 1.6 Angstrom, N2 RHF should be internally unstable
  auto n2 = testing::create_stretched_n2_structure(1.6);

  // Run RHF SCF calculation
  auto scf_solver = ScfSolverFactory::create();
  auto [energy, wavefunction] = scf_solver->run(n2, 0, 1, "def2-svp");

  // Create stability checker for full analysis (internal + external)
  auto stability_checker = StabilityCheckerFactory::create("qdk");
  stability_checker->settings().set("internal", true);
  stability_checker->settings().set("external", true);

  // Run stability analysis
  auto [is_stable, result] = stability_checker->run(wavefunction);

  // Verify result properties
  EXPECT_TRUE(result != nullptr);
  EXPECT_TRUE(result->has_internal_result());
  EXPECT_TRUE(result->has_external_result());
  EXPECT_GT(result->internal_size(), 0);
  EXPECT_GT(result->external_size(), 0);

  // N2 RHF at 1.6 Angstrom should be both internally and externally unstable
  EXPECT_FALSE(result->is_internal_stable());
  EXPECT_FALSE(result->is_external_stable());
  EXPECT_FALSE(is_stable);

  // Check the smallest internal eigenvalue against reference
  double smallest_internal_eigenvalue =
      result->get_smallest_internal_eigenvalue();
  double davidson_tol =
      stability_checker->settings().get<double>("davidson_tolerance") * 1e2;
  EXPECT_NEAR(smallest_internal_eigenvalue, -0.06540850491442, davidson_tol);

  // Check the smallest external eigenvalue against reference
  double smallest_external_eigenvalue =
      result->get_smallest_external_eigenvalue();
  EXPECT_NEAR(smallest_external_eigenvalue, -0.2824536604197, davidson_tol);
}

TEST_F(StabilityCheckerTest, QDK_UHF_BN_Plus_Internal_Instability) {
  // Test internal stability analysis on BN+ cation (UHF)
  // BN+ should be internally unstable
  auto bn_plus = testing::create_bn_plus_structure();

  // Run UHF SCF calculation (doublet state)
  auto scf_solver = ScfSolverFactory::create();
  auto [energy, wavefunction] = scf_solver->run(bn_plus, 1, 2, "def2-svp");

  // Verify we have UHF orbitals
  auto orbitals = wavefunction->get_orbitals();
  EXPECT_FALSE(orbitals->is_restricted());

  // Create stability checker for internal-only analysis
  auto stability_checker = StabilityCheckerFactory::create("qdk");
  stability_checker->settings().set("internal", true);
  stability_checker->settings().set("external", false);

  // Run stability analysis
  auto [is_stable, result] = stability_checker->run(wavefunction);

  // Verify result properties
  EXPECT_TRUE(result != nullptr);
  EXPECT_TRUE(result->has_internal_result());
  EXPECT_FALSE(result->has_external_result());
  EXPECT_GT(result->internal_size(), 0);
  EXPECT_EQ(result->external_size(), 0);

  // BN+ UHF should be internally unstable
  EXPECT_FALSE(result->is_internal_stable());
  EXPECT_FALSE(is_stable);

  // Check the smallest internal eigenvalue against reference
  double smallest_eigenvalue = result->get_smallest_internal_eigenvalue();
  double davidson_tol =
      stability_checker->settings().get<double>("davidson_tolerance") * 1e2;
  EXPECT_NEAR(smallest_eigenvalue, -0.07936094808975, davidson_tol);
}

TEST_F(StabilityCheckerTest, QDK_RHF_N2_Stretched_PBE_Instability) {
  // Test stability analysis on stretched N2 molecule with PBE
  // At 1.6 Angstrom, N2 RHF/PBE is internally stable but externally unstable
  auto n2 = testing::create_stretched_n2_structure(1.6);

  // Run RHF SCF calculation with PBE
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("method", "pbe");
  auto [energy, wavefunction] = scf_solver->run(n2, 0, 1, "def2-svp");

  EXPECT_NEAR(energy, -109.09003668989645, testing::scf_energy_tolerance);

  // Create stability checker for full analysis (internal + external)
  auto stability_checker = StabilityCheckerFactory::create("qdk");
  stability_checker->settings().set("internal", true);
  stability_checker->settings().set("external", true);
  stability_checker->settings().set("method", "pbe");

  // Run stability analysis
  auto [is_stable, result] = stability_checker->run(wavefunction);

  // Verify result properties
  EXPECT_TRUE(result != nullptr);
  EXPECT_TRUE(result->has_internal_result());
  EXPECT_TRUE(result->has_external_result());
  EXPECT_GT(result->internal_size(), 0);
  EXPECT_GT(result->external_size(), 0);

  // N2 RHF/PBE at 1.6 Angstrom is internally stable but externally unstable
  EXPECT_TRUE(result->is_internal_stable());   // Internal eigenvalue ~0.0
  EXPECT_FALSE(result->is_external_stable());  // External eigenvalue ~-0.02
  EXPECT_FALSE(is_stable);  // Overall unstable due to external instability

  // Check eigenvalues
  double smallest_internal_eigenvalue =
      result->get_smallest_internal_eigenvalue();
  double smallest_external_eigenvalue =
      result->get_smallest_external_eigenvalue();
  double davidson_tol =
      stability_checker->settings().get<double>("davidson_tolerance") * 1e2;
  // Internal eigenvalue should be very small but positive (stable)
  EXPECT_NEAR(smallest_internal_eigenvalue, 0.1659455646937, davidson_tol);
  EXPECT_NEAR(smallest_external_eigenvalue, -0.01967330067025, davidson_tol);
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <memory>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/data/orbitals.hpp>

using namespace qdk::chemistry::data;

// ---------------------------------------------------------------------------
// Fixture: synthetic factorized Hamiltonian (N=2, R=1, B=2, C=1)
// ---------------------------------------------------------------------------
class FactorizedHamiltonianTest : public ::testing::Test {
 protected:
  void SetUp() override {
    N = 2;
    R = 1;
    B = 2;
    C = 1;
    core_energy = 1.5;
    bliss_core_shift = 0.1;
    energy_gap = 0.0;

    // One-body integrals [N x N]
    one_body = Eigen::MatrixXd::Identity(N, N);
    one_body(0, 1) = 0.3;
    one_body(1, 0) = 0.3;

    // U matrices flat [R*B*N] = 4 elements
    // Two orthonormal-ish basis vectors in R^2
    u = Eigen::VectorXd(R * B * N);
    u << 0.8, 0.6,  // U[0,0,:] = (0.8, 0.6)
        -0.6, 0.8;  // U[0,1,:] = (-0.6, 0.8)

    // W matrices flat [R*B*C] = 2 elements
    w = Eigen::VectorXd(R * B * C);
    w << 0.5, -0.3;

    // WB matrix [R x C] = [1 x 1]
    wb = Eigen::MatrixXd(R, C);
    wb(0, 0) = 0.2;

    // Inactive Fock (empty for restricted)
    inactive_fock = Eigen::MatrixXd::Zero(0, 0);

    // Model orbitals with N modes
    orbitals = std::make_shared<ModelOrbitals>(N);
  }

  void TearDown() override {
    std::filesystem::remove("test_factorized.hamiltonian.json");
    std::filesystem::remove("test_factorized.hamiltonian.h5");
  }

  /// Helper: build a FactorizedHamiltonianContainer from fixture data.
  std::unique_ptr<FactorizedHamiltonianContainer> make_container() const {
    return std::make_unique<FactorizedHamiltonianContainer>(
        R, B, C, core_energy, u, w, one_body, wb, inactive_fock, orbitals,
        bliss_core_shift, energy_gap);
  }

  size_t N, R, B, C;
  double core_energy, bliss_core_shift, energy_gap;
  Eigen::MatrixXd one_body;
  Eigen::VectorXd u, w;
  Eigen::MatrixXd wb;
  Eigen::MatrixXd inactive_fock;
  std::shared_ptr<Orbitals> orbitals;
};

// ---------------------------------------------------------------------------
// Basic construction & accessors
// ---------------------------------------------------------------------------

TEST_F(FactorizedHamiltonianTest, Construction) {
  auto container = make_container();

  EXPECT_EQ(container->get_container_type(), "factorized");
  EXPECT_TRUE(container->is_restricted());
  EXPECT_TRUE(container->is_valid());
  EXPECT_TRUE(container->has_one_body_integrals());
  EXPECT_TRUE(container->has_two_body_integrals());

  EXPECT_EQ(container->get_num_orbitals(), N);
  EXPECT_EQ(container->get_num_ranks(), R);
  EXPECT_EQ(container->get_num_bases(), B);
  EXPECT_EQ(container->get_num_copies(), C);
  EXPECT_DOUBLE_EQ(container->get_core_energy(), core_energy);
  EXPECT_DOUBLE_EQ(container->get_bliss_core_shift(), bliss_core_shift);
  EXPECT_DOUBLE_EQ(container->get_energy_gap(), energy_gap);
}

TEST_F(FactorizedHamiltonianTest, StoredTensorsMatchInput) {
  auto container = make_container();

  EXPECT_TRUE(container->get_u_matrices().isApprox(u));
  EXPECT_TRUE(container->get_w_matrices().isApprox(w));
  EXPECT_TRUE(container->get_wb_matrix().isApprox(wb));
}

// ---------------------------------------------------------------------------
// JSON round-trip (container level)
// ---------------------------------------------------------------------------

TEST_F(FactorizedHamiltonianTest, JSONRoundTrip) {
  auto original = make_container();
  nlohmann::json j = original->to_json();

  EXPECT_EQ(j["container_type"], "factorized");
  EXPECT_EQ(j["num_ranks"], R);
  EXPECT_EQ(j["num_bases"], B);
  EXPECT_EQ(j["num_copies"], C);
  EXPECT_DOUBLE_EQ(j["core_energy"].get<double>(), core_energy);
  EXPECT_DOUBLE_EQ(j["bliss_core_shift"].get<double>(), bliss_core_shift);

  auto loaded = FactorizedHamiltonianContainer::from_json(j);

  // Dimensions
  EXPECT_EQ(loaded->get_num_orbitals(), N);
  EXPECT_EQ(loaded->get_num_ranks(), R);
  EXPECT_EQ(loaded->get_num_bases(), B);
  EXPECT_EQ(loaded->get_num_copies(), C);

  // Scalars
  EXPECT_DOUBLE_EQ(loaded->get_core_energy(), core_energy);
  EXPECT_DOUBLE_EQ(loaded->get_bliss_core_shift(), bliss_core_shift);
  EXPECT_DOUBLE_EQ(loaded->get_energy_gap(), energy_gap);

  // Tensors
  EXPECT_TRUE(loaded->get_u_matrices().isApprox(u));
  EXPECT_TRUE(loaded->get_w_matrices().isApprox(w));
  EXPECT_TRUE(loaded->get_wb_matrix().isApprox(wb));

  // One-body integrals
  auto [orig_h1a, orig_h1b] = original->get_one_body_integrals();
  auto [load_h1a, load_h1b] = loaded->get_one_body_integrals();
  EXPECT_TRUE(orig_h1a.isApprox(load_h1a));

  // Reconstructed two-body integrals
  auto [orig_aa, orig_ab, orig_bb] = original->get_two_body_integrals();
  auto [load_aa, load_ab, load_bb] = loaded->get_two_body_integrals();
  EXPECT_TRUE(orig_aa.isApprox(load_aa));

  EXPECT_TRUE(loaded->is_restricted());
  EXPECT_TRUE(loaded->is_valid());
}

// ---------------------------------------------------------------------------
// JSON round-trip via Hamiltonian wrapper (tests container_type dispatch)
// ---------------------------------------------------------------------------

TEST_F(FactorizedHamiltonianTest, JSONRoundTripViaHamiltonian) {
  Hamiltonian h(make_container());

  nlohmann::json j = h.to_json();
  auto h2 = Hamiltonian::from_json(j);

  EXPECT_EQ(h2->get_container_type(), "factorized");
  EXPECT_TRUE(h2->has_container_type<FactorizedHamiltonianContainer>());
  EXPECT_EQ(h2->get_core_energy(), core_energy);

  auto [h1a, h1b] = h.get_one_body_integrals();
  auto [h2_h1a, h2_h1b] = h2->get_one_body_integrals();
  EXPECT_TRUE(h1a.isApprox(h2_h1a));
}

// ---------------------------------------------------------------------------
// HDF5 round-trip (container level)
// ---------------------------------------------------------------------------

TEST_F(FactorizedHamiltonianTest, HDF5RoundTrip) {
  auto original = make_container();

  // Write
  std::string filename = "test_factorized.hamiltonian.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group group = file.createGroup("container");
    original->to_hdf5(group);
  }

  // Read
  {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group group = file.openGroup("container");
    auto loaded = FactorizedHamiltonianContainer::from_hdf5(group);

    EXPECT_EQ(loaded->get_num_orbitals(), N);
    EXPECT_EQ(loaded->get_num_ranks(), R);
    EXPECT_EQ(loaded->get_num_bases(), B);
    EXPECT_EQ(loaded->get_num_copies(), C);

    EXPECT_DOUBLE_EQ(loaded->get_core_energy(), core_energy);
    EXPECT_DOUBLE_EQ(loaded->get_bliss_core_shift(), bliss_core_shift);
    EXPECT_DOUBLE_EQ(loaded->get_energy_gap(), energy_gap);

    EXPECT_TRUE(loaded->get_u_matrices().isApprox(u));
    EXPECT_TRUE(loaded->get_w_matrices().isApprox(w));
    EXPECT_TRUE(loaded->get_wb_matrix().isApprox(wb));

    auto [orig_h1a, orig_h1b] = original->get_one_body_integrals();
    auto [load_h1a, load_h1b] = loaded->get_one_body_integrals();
    EXPECT_TRUE(orig_h1a.isApprox(load_h1a));

    EXPECT_TRUE(loaded->is_restricted());
    EXPECT_TRUE(loaded->is_valid());
  }
}

// ---------------------------------------------------------------------------
// HDF5 file I/O via Hamiltonian wrapper
// ---------------------------------------------------------------------------

TEST_F(FactorizedHamiltonianTest, HDF5FileRoundTripViaHamiltonian) {
  Hamiltonian h(make_container());

  std::string filename = "test_factorized.hamiltonian.h5";
  h.to_hdf5_file(filename);
  EXPECT_TRUE(std::filesystem::exists(filename));

  auto h2 = Hamiltonian::from_hdf5_file(filename);

  EXPECT_EQ(h2->get_container_type(), "factorized");
  EXPECT_TRUE(h2->has_container_type<FactorizedHamiltonianContainer>());
  EXPECT_DOUBLE_EQ(h2->get_core_energy(), core_energy);

  auto& fc = h2->get_container<FactorizedHamiltonianContainer>();
  EXPECT_EQ(fc.get_num_ranks(), R);
  EXPECT_EQ(fc.get_num_bases(), B);
  EXPECT_EQ(fc.get_num_copies(), C);
  EXPECT_TRUE(fc.get_u_matrices().isApprox(u));
  EXPECT_TRUE(fc.get_w_matrices().isApprox(w));
  EXPECT_TRUE(fc.get_wb_matrix().isApprox(wb));
}

// ---------------------------------------------------------------------------
// Clone
// ---------------------------------------------------------------------------

TEST_F(FactorizedHamiltonianTest, Clone) {
  auto original = make_container();
  auto cloned = original->clone();

  EXPECT_EQ(cloned->get_container_type(), "factorized");
  EXPECT_EQ(cloned->get_core_energy(), core_energy);

  // Downcast and verify factorized-specific data
  auto* fc = dynamic_cast<FactorizedHamiltonianContainer*>(cloned.get());
  ASSERT_NE(fc, nullptr);
  EXPECT_EQ(fc->get_num_ranks(), R);
  EXPECT_EQ(fc->get_num_bases(), B);
  EXPECT_EQ(fc->get_num_copies(), C);
  EXPECT_TRUE(fc->get_u_matrices().isApprox(u));
  EXPECT_TRUE(fc->get_w_matrices().isApprox(w));
  EXPECT_TRUE(fc->get_wb_matrix().isApprox(wb));
}

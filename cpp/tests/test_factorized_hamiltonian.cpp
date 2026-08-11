// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <memory>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/factorized.hpp>
#include <qdk/chemistry/data/orbitals.hpp>

using namespace qdk::chemistry::data;

// synthetic factorized Hamiltonian (N=2, R=1, B=2, C=1)
class FactorizedHamiltonianTest : public ::testing::Test {
 protected:
  void SetUp() override {
    N = 2;
    R = 1;
    B = 2;
    C = 1;
    core_energy = 1.5;
    bliss_shift = 0.1;
    energy_gap = 0.0;

    one_body = Eigen::MatrixXd::Identity(N, N);
    one_body(0, 1) = 0.3;
    one_body(1, 0) = 0.3;

    u = Eigen::VectorXd(R * B * N);
    u << 0.8, 0.6, -0.6, 0.8;
    w = Eigen::VectorXd(R * B * C);
    w << 0.5, -0.3;
    wb = Eigen::MatrixXd(R, C);
    wb(0, 0) = 0.2;

    inactive_fock = Eigen::MatrixXd::Zero(0, 0);
    orbitals = std::make_shared<ModelOrbitals>(N);
  }

  void TearDown() override {
    std::filesystem::remove("test_factorized.hamiltonian.json");
    std::filesystem::remove("test_factorized.hamiltonian.h5");
  }

  std::unique_ptr<FactorizedHamiltonianContainer> make_container() const {
    return std::make_unique<FactorizedHamiltonianContainer>(
        core_energy, u, w, wb, one_body, inactive_fock, orbitals, bliss_shift,
        energy_gap);
  }

  size_t N, R, B, C;
  double core_energy, bliss_shift, energy_gap;
  Eigen::MatrixXd one_body;
  Eigen::VectorXd u, w;
  Eigen::MatrixXd wb;
  Eigen::MatrixXd inactive_fock;
  std::shared_ptr<Orbitals> orbitals;
};

TEST_F(FactorizedHamiltonianTest, Properties) {
  auto container = make_container();

  // Factorization dimensions.
  EXPECT_EQ(container->get_num_orbitals(), N);
  EXPECT_EQ(container->get_num_ranks(), R);
  EXPECT_EQ(container->get_num_bases(), B);
  EXPECT_EQ(container->get_num_copies(), C);

  // Majorana one-body integrals (row-major 2x2).
  const double expected_h1[4] = {0.90379999999999994, 0.26160000000000005,
                                 0.26160000000000005, 0.92620000000000002};
  Eigen::MatrixXd h1m = container->get_h1_majorana();
  ASSERT_EQ(h1m.rows(), static_cast<Eigen::Index>(N));
  ASSERT_EQ(h1m.cols(), static_cast<Eigen::Index>(N));
  for (size_t p = 0; p < N; ++p) {
    for (size_t q = 0; q < N; ++q) {
      EXPECT_NEAR(h1m(p, q), expected_h1[p * N + q], 1e-12);
    }
  }

  // Reconstructed two-body integrals (row-major, N^4 = 16).
  const double expected_h2[16] = {0.044944000000000033,
                                  0.081408000000000036,
                                  0.081408000000000036,
                                  -0.0025440000000000033,
                                  0.081408000000000036,
                                  0.147456,
                                  0.147456,
                                  -0.0046080000000000045,
                                  0.081408000000000036,
                                  0.147456,
                                  0.147456,
                                  -0.0046080000000000045,
                                  -0.0025440000000000033,
                                  -0.0046080000000000045,
                                  -0.0046080000000000045,
                                  0.00014400000000000025};
  Eigen::VectorXd h2 = container->reconstruct_two_body_integrals();
  ASSERT_EQ(h2.size(), static_cast<Eigen::Index>(N * N * N * N));
  for (Eigen::Index i = 0; i < h2.size(); ++i) {
    EXPECT_NEAR(h2(i), expected_h2[i], 1e-12);
  }

  // Block-encoding 1-norm Lambda.
  EXPECT_NEAR(container->get_lambda(), 2.0800000000000001, 1e-12);

  // Effective SOS 1-norm needs a positive energy gap (the fixture uses 0.0).
  FactorizedHamiltonianContainer gapped(core_energy, u, w, wb, one_body,
                                        inactive_fock, orbitals, bliss_shift,
                                        0.5);
  EXPECT_NEAR(gapped.get_lambda_eff(), 1.3527749258468684, 1e-12);
}

// The identity weight wB is a *gauge* parameter: paper Eq. 26 builds the two-body
// tensor purely from (u, w), so moving wB must leave it bit-identical -- while Eq. 38
// and Lambda must both respond, which is what makes wB a knob rather than dead data.
//
// This pins the two halves against each other. Folding wB into the reconstruction
// (a tempting "fix" when reading Eq. 26 next to Eq. 38) breaks the first half;
// dropping it from get_h1_majorana breaks the second.
TEST_F(FactorizedHamiltonianTest, IdentityWeightIsGaugeForTwoBodyOnly) {
  auto reference = make_container();
  const Eigen::VectorXd h2_ref = reference->reconstruct_two_body_integrals();
  const Eigen::MatrixXd h1_ref = reference->get_h1_majorana();
  const double lambda_ref = reference->get_lambda();

  // Deliberately excludes the fixture's own 0.2 so every case is a real change.
  const double wb_values[] = {0.0, -3.5, 7.25};
  for (double wb_value : wb_values) {
    Eigen::MatrixXd wb_alt(R, C);
    wb_alt(0, 0) = wb_value;
    FactorizedHamiltonianContainer shifted(core_energy, u, w, wb_alt, one_body,
                                           inactive_fock, orbitals, bliss_shift,
                                           energy_gap);

    // (a) the reconstructed two-body tensor is untouched, exactly.
    const Eigen::VectorXd h2_alt = shifted.reconstruct_two_body_integrals();
    ASSERT_EQ(h2_alt.size(), h2_ref.size());
    for (Eigen::Index i = 0; i < h2_ref.size(); ++i) {
      EXPECT_NEAR(h2_alt(i), h2_ref(i), 1e-12)
          << "wB=" << wb_value << " moved h2 element " << i;
    }

    // (b) ... but the Majorana one-body shift and Lambda both move.
    EXPECT_FALSE(shifted.get_h1_majorana().isApprox(h1_ref, 1e-9))
        << "wB=" << wb_value << " left h1_majorana unchanged";
    EXPECT_GT(std::abs(shifted.get_lambda() - lambda_ref), 1e-9)
        << "wB=" << wb_value << " left Lambda unchanged";
  }
}

// Eq. 38 as printed lists two corrections; get_h1_majorana applies three. The extra
// -1/2 (M M)_pq converts the stored normal-ordered h2 = (pq|rs) to the paper's
// plain-product convention, so it is required rather than optional.
//
// Note the base fixture has tr(M) == wB == 0.2, which makes terms (b) and (c) cancel
// and leaves the golden-value test above unable to tell them apart. Perturbing wB
// here breaks that degeneracy so all three terms are exercised independently.
TEST_F(FactorizedHamiltonianTest, MajoranaOneBodyCarriesNormalOrderingTerm) {
  // M_{pq} = sum_b w_b u_{b,p} u_{b,q}  (the fixture has a single rank and copy).
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(N, N);
  for (size_t b = 0; b < B; ++b) {
    Eigen::VectorXd ub(N);
    for (size_t p = 0; p < N; ++p) {
      ub(p) = u(static_cast<Eigen::Index>(b * N + p));
    }
    m += w(static_cast<Eigen::Index>(b)) * ub * ub.transpose();
  }

  const double wb_values[] = {0.2, 0.0, -3.5};
  for (double wb_value : wb_values) {
    Eigen::MatrixXd wb_alt(R, C);
    wb_alt(0, 0) = wb_value;
    FactorizedHamiltonianContainer container(core_energy, u, w, wb_alt, one_body,
                                             inactive_fock, orbitals, bliss_shift,
                                             energy_gap);

    Eigen::MatrixXd expected = one_body;
    expected -= 0.5 * (m * m);   // (a) normal-ordering remainder
    expected += m.trace() * m;   // (b)
    expected -= wb_value * m;    // (c)
    EXPECT_TRUE(container.get_h1_majorana().isApprox(expected, 1e-12))
        << "three-term Eq. 38 model failed at wB=" << wb_value;

    // Dropping term (a) is not a rounding-level difference.
    const Eigen::MatrixXd without_normal_ordering = expected + 0.5 * (m * m);
    EXPECT_FALSE(
        container.get_h1_majorana().isApprox(without_normal_ordering, 1e-9))
        << "normal-ordering term looks absent at wB=" << wb_value;
  }
}

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

TEST_F(FactorizedHamiltonianTest, HDF5RoundTrip) {
  auto original = make_container();

  std::string filename = "test_factorized.hamiltonian.h5";
  {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group group = file.createGroup("container");
    original->to_hdf5(group);
  }

  {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group group = file.openGroup("container");
    auto loaded = FactorizedHamiltonianContainer::from_hdf5(group);

    EXPECT_EQ(loaded->get_num_orbitals(), N);
    EXPECT_EQ(loaded->get_num_ranks(), R);
    EXPECT_EQ(loaded->get_num_bases(), B);
    EXPECT_EQ(loaded->get_num_copies(), C);

    EXPECT_DOUBLE_EQ(loaded->get_core_energy(), core_energy);
    EXPECT_DOUBLE_EQ(loaded->get_bliss_shift(), bliss_shift);
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

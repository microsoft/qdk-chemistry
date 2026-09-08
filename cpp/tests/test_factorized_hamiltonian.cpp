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
#include <stdexcept>

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
        core_energy, u, w, wb, one_body, inactive_fock, orbitals, energy_gap);
  }

  size_t N, R, B, C;
  double core_energy, energy_gap;
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

  // The literals below are hand-derived from the fixture in closed form, so
  // they check the implementation rather than restate it. With one rank, one
  // copy and s = +1, the fixture's bases give the single mode matrix
  //   M_{pq} = Sum_b W_b U_{bp} U_{bq}
  //          = 0.5 * outer([0.8, 0.6]) - 0.3 * outer([-0.6, 0.8])
  //          = [[0.212, 0.384], [0.384, -0.012]],
  // whose trace is 0.2 and therefore cancels wB = 0.2 exactly. Hence
  //   h1' = h1 - M^2 / 2 + (tr(M) - wB) M = h1 - M^2 / 2,
  //   h2_{pqrs} = M_{pq} M_{rs},
  //   Lambda = Sum_i |eig_i(h1')| + (|wB| + Sum_b |W_b|)^2 / 4
  //          = 1.83 + 1.0 / 4 = 2.08.
  const double expected_h1[4] = {0.90379999999999994, 0.26160000000000005,
                                 0.26160000000000005, 0.92620000000000002};
  Eigen::MatrixXd h1p = container->get_h1_prime();
  ASSERT_EQ(h1p.rows(), static_cast<Eigen::Index>(N));
  ASSERT_EQ(h1p.cols(), static_cast<Eigen::Index>(N));
  for (size_t p = 0; p < N; ++p) {
    for (size_t q = 0; q < N; ++q) {
      EXPECT_NEAR(h1p(p, q), expected_h1[p * N + q], 1e-12);
    }
  }

  // Reconstructed two-body integrals (row-major, N^4 = 16): the outer product
  // M_{pq} M_{rs} of the mode matrix derived above.
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

  EXPECT_NEAR(container->get_lambda(), 2.0800000000000001, 1e-12);
  EXPECT_DOUBLE_EQ(container->get_lambda_eff(), 0.0);

  FactorizedHamiltonianContainer gapped(core_energy, u, w, wb, one_body,
                                        inactive_fock, orbitals, 0.5);
  EXPECT_NEAR(gapped.get_lambda_eff(), 1.3527749258468684, 1e-12);

  FactorizedHamiltonianContainer gap_at_upper_bound(
      core_energy, u, w, wb, one_body, inactive_fock, orbitals,
      2.0 * container->get_lambda());
  EXPECT_DOUBLE_EQ(gap_at_upper_bound.get_lambda_eff(), 0.0);
}

TEST_F(FactorizedHamiltonianTest, IdentityWeightDoesNotChangeTwoBodyTensor) {
  auto reference = make_container();
  const Eigen::VectorXd h2_ref = reference->reconstruct_two_body_integrals();
  const Eigen::MatrixXd h1_ref = reference->get_h1_prime();
  const double lambda_ref = reference->get_lambda();

  const double wb_values[] = {0.0, -3.5, 7.25};
  for (double wb_value : wb_values) {
    Eigen::MatrixXd wb_alt(R, C);
    wb_alt(0, 0) = wb_value;
    FactorizedHamiltonianContainer shifted(core_energy, u, w, wb_alt, one_body,
                                           inactive_fock, orbitals, energy_gap);

    const Eigen::VectorXd h2_alt = shifted.reconstruct_two_body_integrals();
    ASSERT_EQ(h2_alt.size(), h2_ref.size());
    for (Eigen::Index i = 0; i < h2_ref.size(); ++i) {
      EXPECT_NEAR(h2_alt(i), h2_ref(i), 1e-12)
          << "wB=" << wb_value << " moved h2 element " << i;
    }

    EXPECT_FALSE(shifted.get_h1_prime().isApprox(h1_ref, 1e-9))
        << "wB=" << wb_value << " left h1_prime unchanged";
    EXPECT_GT(std::abs(shifted.get_lambda() - lambda_ref), 1e-9)
        << "wB=" << wb_value << " left Lambda unchanged";
  }
}

TEST_F(FactorizedHamiltonianTest, H1PrimeMatchesClosedForm) {
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(N, N);
  for (size_t b = 0; b < B; ++b) {
    Eigen::VectorXd ub(N);
    for (size_t p = 0; p < N; ++p) {
      ub(p) = u(static_cast<Eigen::Index>(b * N + p));
    }
    m += w(static_cast<Eigen::Index>(b)) * ub * ub.transpose();
  }

  // wB only enters through the -wB * M term, so varying it is what separates
  // that term from the trace correction it sits next to.
  const double wb_values[] = {0.2, -3.5};
  for (double wb_value : wb_values) {
    Eigen::MatrixXd wb_alt(R, C);
    wb_alt(0, 0) = wb_value;

    FactorizedHamiltonianContainer container(core_energy, u, w, wb_alt,
                                             one_body, inactive_fock, orbitals,
                                             energy_gap);

    Eigen::MatrixXd expected = one_body;
    expected -= 0.5 * (m * m);
    expected += m.trace() * m;
    expected -= wb_value * m;

    EXPECT_TRUE(container.get_h1_prime().isApprox(expected, 1e-12))
        << "wB=" << wb_value;
  }
}

TEST_F(FactorizedHamiltonianTest, RejectsNonSymmetricH1PrimeInLambda) {
  Eigen::MatrixXd asymmetric = one_body;
  asymmetric(0, 1) = 0.3;
  asymmetric(1, 0) = -1.5;

  Eigen::MatrixXd from_lower = asymmetric;
  from_lower(0, 1) = asymmetric(1, 0);
  Eigen::MatrixXd from_upper = asymmetric;
  from_upper(1, 0) = asymmetric(0, 1);

  const double positive_gap = 0.1;
  auto lambda_of = [&](const Eigen::MatrixXd& h1) {
    return FactorizedHamiltonianContainer(core_energy, u, w, wb, h1,
                                          inactive_fock, orbitals, positive_gap)
        .get_lambda();
  };

  EXPECT_GT(std::abs(lambda_of(from_lower) - lambda_of(from_upper)), 1e-6)
      << "the two triangles must disagree, otherwise this input cannot "
         "demonstrate the ambiguity the guard exists to reject";

  FactorizedHamiltonianContainer container(
      core_energy, u, w, wb, asymmetric, inactive_fock, orbitals, positive_gap);
  EXPECT_THROW(container.get_lambda(), std::runtime_error);
  EXPECT_THROW(container.get_lambda_eff(), std::runtime_error);
}

TEST_F(FactorizedHamiltonianTest, JSONRoundTripViaHamiltonian) {
  Hamiltonian h(make_container());
  nlohmann::json j = h.to_json();
  auto h2 = Hamiltonian::from_json(j);

  EXPECT_EQ(h2->get_container_type(), "factorized");
  EXPECT_TRUE(h2->has_container_type<FactorizedHamiltonianContainer>());
  EXPECT_EQ(h2->get_core_energy(), core_energy);
  EXPECT_TRUE(
      h2->get_container<FactorizedHamiltonianContainer>()
          .reconstruct_two_body_integrals()
          .isApprox(make_container()->reconstruct_two_body_integrals()));

  auto [h1a, h1b] = h.get_one_body_integrals();
  auto [h2_h1a, h2_h1b] = h2->get_one_body_integrals();
  EXPECT_TRUE(h1a.isApprox(h2_h1a));
}

TEST_F(FactorizedHamiltonianTest, RejectsInconsistentSerializedShape) {
  const nlohmann::json serialized = make_container()->to_json();

  auto wrong_ranks = serialized;
  wrong_ranks["num_ranks"] = R + 1;
  EXPECT_THROW(FactorizedHamiltonianContainer::from_json(wrong_ranks),
               std::invalid_argument);

  auto wrong_bases = serialized;
  wrong_bases["num_bases"] = B + 1;
  EXPECT_THROW(FactorizedHamiltonianContainer::from_json(wrong_bases),
               std::invalid_argument);

  auto wrong_copies = serialized;
  wrong_copies["num_copies"] = C + 1;
  EXPECT_THROW(FactorizedHamiltonianContainer::from_json(wrong_copies),
               std::invalid_argument);
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

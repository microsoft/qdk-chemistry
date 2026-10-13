// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <filesystem>
#include <limits>
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
        one_body, u, w, wb, orbitals, core_energy, inactive_fock);
  }

  size_t N, R, B, C;
  double core_energy;
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
}

TEST_F(FactorizedHamiltonianTest, MultipleRanksAndCopiesReconstructAndIndex) {
  // The fixture is R=1, C=1, where every stride into the flattened [R,B,N] and
  // [R,B,C] buffers collapses onto the same offset, so an indexing mistake
  // cannot show up. Use two ranks and two copies, and check against loops
  // written straight from the definition rather than from the implementation.
  constexpr size_t ranks = 2;
  constexpr size_t bases = 2;
  constexpr size_t copies = 2;

  Eigen::VectorXd u_multi(ranks * bases * N);
  u_multi << 0.8, 0.6, -0.6, 0.8,  // rank 0: an orthonormal pair
      1.0, 0.0, 0.0, 1.0;          // rank 1: the canonical basis
  Eigen::VectorXd w_multi(ranks * bases * copies);
  w_multi << 0.5, -0.2, 0.3, 0.4, -0.7, 0.1, 0.25, 0.6;
  Eigen::MatrixXd wb_multi(ranks, copies);
  wb_multi << 0.2, -0.1, 0.05, 0.3;

  FactorizedHamiltonianContainer container(one_body, u_multi, w_multi, wb_multi,
                                           orbitals, core_energy,
                                           inactive_fock);

  ASSERT_EQ(container.get_num_ranks(), ranks);
  ASSERT_EQ(container.get_num_bases(), bases);
  ASSERT_EQ(container.get_num_copies(), copies);

  auto U = [&](size_t t, size_t b, size_t p) {
    return u_multi(static_cast<Eigen::Index>((t * bases + b) * N + p));
  };
  auto W = [&](size_t t, size_t b, size_t c) {
    return w_multi(
        static_cast<Eigen::Index>(t * bases * copies + b * copies + c));
  };
  // h2_{pqrs} = Sum_{t,c} M^{tc}_{pq} M^{tc}_{rs},
  // M^{tc}_{pq}  = Sum_b W^{tc}_b U^t_{bp} U^t_{bq}.
  auto reference = [&](size_t p, size_t q, size_t r, size_t s) {
    double total = 0.0;
    for (size_t t = 0; t < ranks; ++t) {
      for (size_t c = 0; c < copies; ++c) {
        double m_pq = 0.0;
        double m_rs = 0.0;
        for (size_t b = 0; b < bases; ++b) {
          m_pq += W(t, b, c) * U(t, b, p) * U(t, b, q);
          m_rs += W(t, b, c) * U(t, b, r) * U(t, b, s);
        }
        total += m_pq * m_rs;
      }
    }
    return total;
  };

  // Read elements first, while the dense cache is still cold, so this covers
  // the direct contraction rather than a lookup into a materialized tensor.
  for (size_t p = 0; p < N; ++p) {
    for (size_t q = 0; q < N; ++q) {
      for (size_t r = 0; r < N; ++r) {
        for (size_t s = 0; s < N; ++s) {
          EXPECT_NEAR(container.get_two_body_element(
                          static_cast<unsigned>(p), static_cast<unsigned>(q),
                          static_cast<unsigned>(r), static_cast<unsigned>(s)),
                      reference(p, q, r, s), 1e-12)
              << "element (" << p << q << "|" << r << s << ")";
        }
      }
    }
  }

  // The dense path has to agree with both the definition and the element path.
  const Eigen::VectorXd h2 = container.reconstruct_two_body_integrals();
  ASSERT_EQ(static_cast<size_t>(h2.size()), N * N * N * N);
  for (size_t p = 0; p < N; ++p) {
    for (size_t q = 0; q < N; ++q) {
      for (size_t r = 0; r < N; ++r) {
        for (size_t s = 0; s < N; ++s) {
          const size_t idx = ((p * N + q) * N + r) * N + s;
          EXPECT_NEAR(h2(static_cast<Eigen::Index>(idx)), reference(p, q, r, s),
                      1e-12)
              << "dense element (" << p << q << "|" << r << s << ")";
        }
      }
    }
  }
}

TEST_F(FactorizedHamiltonianTest, ElementAgreesWithTheCachedTensor) {
  // get_two_body_element() has two branches: a direct contraction taken while
  // the dense cache is cold, and a lookup into the cache once it is warm. They
  // are separate code paths with different summation orders, so read every
  // element on both sides of the cache being built. Use two ranks and two
  // copies, since at R=1, C=1 the strides collapse and an indexing mistake in
  // either branch cannot show up.
  constexpr size_t ranks = 2;
  constexpr size_t bases = 2;
  constexpr size_t copies = 2;

  Eigen::VectorXd u_multi(ranks * bases * N);
  u_multi << 0.8, 0.6, -0.6, 0.8, 1.0, 0.0, 0.0, 1.0;
  Eigen::VectorXd w_multi(ranks * bases * copies);
  w_multi << 0.5, -0.2, 0.3, 0.4, -0.7, 0.1, 0.25, 0.6;
  Eigen::MatrixXd wb_multi(ranks, copies);
  wb_multi << 0.2, -0.1, 0.05, 0.3;

  FactorizedHamiltonianContainer container(one_body, u_multi, w_multi, wb_multi,
                                           orbitals, core_energy,
                                           inactive_fock);

  const size_t total = N * N * N * N;
  Eigen::VectorXd cold(static_cast<Eigen::Index>(total));
  for (size_t p = 0; p < N; ++p) {
    for (size_t q = 0; q < N; ++q) {
      for (size_t r = 0; r < N; ++r) {
        for (size_t s = 0; s < N; ++s) {
          const size_t idx = ((p * N + q) * N + r) * N + s;
          cold(static_cast<Eigen::Index>(idx)) = container.get_two_body_element(
              static_cast<unsigned>(p), static_cast<unsigned>(q),
              static_cast<unsigned>(r), static_cast<unsigned>(s));
        }
      }
    }
  }

  // Building the dense tensor switches the accessor to the lookup branch.
  auto [g_aaaa, g_aabb, g_bbbb] = container.get_two_body_integrals();
  ASSERT_EQ(static_cast<size_t>(g_aaaa.size()), total);

  for (size_t p = 0; p < N; ++p) {
    for (size_t q = 0; q < N; ++q) {
      for (size_t r = 0; r < N; ++r) {
        for (size_t s = 0; s < N; ++s) {
          const size_t idx = ((p * N + q) * N + r) * N + s;
          const double warm = container.get_two_body_element(
              static_cast<unsigned>(p), static_cast<unsigned>(q),
              static_cast<unsigned>(r), static_cast<unsigned>(s));
          // The contraction and the cache have to describe the same tensor,
          // and the accessor has to index the cache the way the dense layout
          // is flattened.
          EXPECT_NEAR(warm, cold(static_cast<Eigen::Index>(idx)), 1e-12)
              << "cold and warm disagree at (" << p << q << "|" << r << s
              << ")";
          EXPECT_NEAR(warm, g_aaaa(static_cast<Eigen::Index>(idx)), 1e-12)
              << "element does not match the cached tensor at (" << p << q
              << "|" << r << s << ")";
        }
      }
    }
  }
}

TEST_F(FactorizedHamiltonianTest, RejectsAnUnnormalizedBasisRow) {
  // get_lambda() reads the W entries as eigenvalues of the fragment, which
  // only holds when each basis row is a unit vector. Nothing downstream can
  // notice the difference, so construction has to reject it.
  Eigen::VectorXd u_scaled = u;
  u_scaled *= 2.0;
  EXPECT_THROW(
      FactorizedHamiltonianContainer(one_body, u_scaled, w, wb, orbitals,
                                     core_energy, inactive_fock),
      std::invalid_argument);
}

TEST_F(FactorizedHamiltonianTest, RejectsNonFiniteFactorEntries) {
  // NaN compares false against the normalization tolerance, so it would pass
  // that check unless the factors are rejected as non-finite first.
  const double nan = std::numeric_limits<double>::quiet_NaN();

  Eigen::VectorXd u_nan = u;
  u_nan(0) = nan;
  EXPECT_THROW(FactorizedHamiltonianContainer(one_body, u_nan, w, wb, orbitals,
                                              core_energy, inactive_fock),
               std::invalid_argument);

  // A NaN in W or wB never reaches the normalization check at all.
  Eigen::VectorXd w_nan = w;
  w_nan(0) = nan;
  EXPECT_THROW(FactorizedHamiltonianContainer(one_body, u, w_nan, wb, orbitals,
                                              core_energy, inactive_fock),
               std::invalid_argument);

  Eigen::MatrixXd wb_nan = wb;
  wb_nan(0, 0) = nan;
  EXPECT_THROW(FactorizedHamiltonianContainer(one_body, u, w, wb_nan, orbitals,
                                              core_energy, inactive_fock),
               std::invalid_argument);

  // The fixture itself is finite, so the new guard cannot be what makes the
  // cases above throw.
  EXPECT_NO_THROW(FactorizedHamiltonianContainer(one_body, u, w, wb, orbitals,
                                                 core_energy, inactive_fock));
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
    FactorizedHamiltonianContainer shifted(one_body, u, w, wb_alt, orbitals,
                                           core_energy, inactive_fock);

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

    FactorizedHamiltonianContainer container(one_body, u, w, wb_alt, orbitals,
                                             core_energy, inactive_fock);

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

  auto lambda_of = [&](const Eigen::MatrixXd& h1) {
    return FactorizedHamiltonianContainer(h1, u, w, wb, orbitals, core_energy,
                                          inactive_fock)
        .get_lambda();
  };

  EXPECT_GT(std::abs(lambda_of(from_lower) - lambda_of(from_upper)), 1e-6)
      << "the two triangles must disagree, otherwise this input cannot "
         "demonstrate the ambiguity the guard exists to reject";

  FactorizedHamiltonianContainer container(asymmetric, u, w, wb, orbitals,
                                           core_energy, inactive_fock);
  EXPECT_THROW(container.get_lambda(), std::runtime_error);
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

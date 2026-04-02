// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/utils/model_hamiltonians.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
namespace model_hamiltonians = qdk::chemistry::utils::model_hamiltonians;

namespace testing_detail {
/** @brief Materialise a TwoBodyMap into a flat dense vector for test
 * comparison. */
Eigen::VectorXd to_dense_h2(const SparseHamiltonianContainer::TwoBodyMap& h2,
                            int n) {
  int n2 = n * n, n3 = n2 * n;
  Eigen::VectorXd v = Eigen::VectorXd::Zero(n * n * n * n);
  for (const auto& [idx, val] : h2) {
    const auto& [p, q, r, s] = idx;
    v(p * n3 + q * n2 + r * n + s) = val;
  }
  return v;
}
}  // namespace testing_detail

class ModelHamiltonianTest : public ::testing::Test {};
TEST_F(ModelHamiltonianTest, HuckelChain) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);
  Eigen::MatrixXd h1_expected = Eigen::MatrixXd::Zero(n, n);
  h1_expected(0, 1) = h1_expected(1, 0) = -1.0;
  h1_expected(1, 2) = h1_expected(2, 1) = -1.0;
  h1_expected(2, 3) = h1_expected(3, 2) = -1.0;
  h1_expected(0, 0) = h1_expected(1, 1) = h1_expected(2, 2) =
      h1_expected(3, 3) = -0.5;

  // test scalar huckel — returns SparseMatrix<double>
  auto h1 =
      model_hamiltonians::detail::_build_huckel_integrals(lattice, -0.5, 1.0);
  EXPECT_TRUE(Eigen::MatrixXd(h1).isApprox(h1_expected,
                                           testing::numerical_zero_tolerance));

  // test vector/matrix huckel
  Eigen::MatrixXd t_mat = Eigen::MatrixXd::Constant(n, n, 1.0);
  Eigen::VectorXd eps_vec = Eigen::VectorXd::Constant(n, -0.5);
  auto h1_explicit = model_hamiltonians::detail::_build_huckel_integrals(
      lattice, eps_vec, t_mat);
  EXPECT_TRUE(Eigen::MatrixXd(h1_explicit)
                  .isApprox(h1_expected, testing::numerical_zero_tolerance));

  // modify t and eps and verify different integrals
  t_mat(1, 2) = t_mat(2, 1) = 0.5;
  eps_vec(2) = 2.0;
  auto h1_modified = model_hamiltonians::detail::_build_huckel_integrals(
      lattice, eps_vec, t_mat);
  Eigen::MatrixXd h1_mod_dense = Eigen::MatrixXd(h1_modified);
  EXPECT_FALSE(
      h1_mod_dense.isApprox(h1_expected, testing::numerical_zero_tolerance));
  EXPECT_NEAR(h1_mod_dense(1, 2), -0.5, testing::numerical_zero_tolerance);
  EXPECT_NEAR(h1_mod_dense(2, 1), -0.5, testing::numerical_zero_tolerance);
  EXPECT_NEAR(h1_mod_dense(2, 2), 2.0, testing::numerical_zero_tolerance);
  h1_expected(1, 2) = h1_expected(2, 1) = -0.5;
  h1_expected(2, 2) = 2.0;
  EXPECT_TRUE(
      h1_mod_dense.isApprox(h1_expected, testing::numerical_zero_tolerance));
}

TEST_F(ModelHamiltonianTest, HubbardChain) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);
  Eigen::MatrixXd h1_expected = Eigen::MatrixXd::Zero(n, n);
  h1_expected(0, 1) = h1_expected(1, 0) = -1.0;
  h1_expected(1, 2) = h1_expected(2, 1) = -1.0;
  h1_expected(2, 3) = h1_expected(3, 2) = -1.0;
  h1_expected(0, 0) = h1_expected(1, 1) = h1_expected(2, 2) =
      h1_expected(3, 3) = -0.5;

  const size_t offset = n * n * n + n * n + n + 1;
  Eigen::VectorXd h2_expected = Eigen::VectorXd::Constant(n * n * n * n, 0.0);
  for (int i = 0; i < n; i++) {
    h2_expected[i * offset] = 0.3;
  }

  // test scalar hubbard — returns tuple of (SparseMatrix, TwoBodyMap)
  auto [h1, h2] = model_hamiltonians::detail::_build_hubbard_integrals(
      lattice, -0.5, 1.0, 0.3);
  EXPECT_TRUE(Eigen::MatrixXd(h1).isApprox(h1_expected,
                                           testing::numerical_zero_tolerance));
  EXPECT_TRUE(testing_detail::to_dense_h2(h2, n).isApprox(
      h2_expected, testing::numerical_zero_tolerance));

  // test vector/matrix hubbard
  Eigen::MatrixXd t_mat = Eigen::MatrixXd::Constant(n, n, 1.0);
  Eigen::VectorXd eps_vec = Eigen::VectorXd::Constant(n, -0.5);
  Eigen::VectorXd U_vec = Eigen::VectorXd::Constant(n, 0.3);
  auto [h1_explicit, h2_explicit] =
      model_hamiltonians::detail::_build_hubbard_integrals(lattice, eps_vec,
                                                           t_mat, U_vec);
  EXPECT_TRUE(Eigen::MatrixXd(h1_explicit)
                  .isApprox(h1_expected, testing::numerical_zero_tolerance));
  EXPECT_TRUE(testing_detail::to_dense_h2(h2_explicit, n)
                  .isApprox(h2_expected, testing::numerical_zero_tolerance));

  // modify U, t and eps and verify different integrals
  t_mat(1, 2) = t_mat(2, 1) = 0.5;
  eps_vec(2) = 2.0;
  U_vec(0) = 0.7;
  auto [h1_mod, h2_mod] = model_hamiltonians::detail::_build_hubbard_integrals(
      lattice, eps_vec, t_mat, U_vec);
  Eigen::MatrixXd h1_mod_dense = Eigen::MatrixXd(h1_mod);
  Eigen::VectorXd h2_mod_dense = testing_detail::to_dense_h2(h2_mod, n);
  EXPECT_FALSE(
      h1_mod_dense.isApprox(h1_expected, testing::numerical_zero_tolerance));
  EXPECT_NEAR(h1_mod_dense(1, 2), -0.5, testing::numerical_zero_tolerance);
  EXPECT_NEAR(h1_mod_dense(2, 1), -0.5, testing::numerical_zero_tolerance);
  EXPECT_NEAR(h1_mod_dense(2, 2), 2.0, testing::numerical_zero_tolerance);
  EXPECT_FALSE(
      h2_mod_dense.isApprox(h2_expected, testing::numerical_zero_tolerance));
  EXPECT_NEAR(h2_mod_dense[0], 0.7, testing::numerical_zero_tolerance);
  h1_expected(1, 2) = h1_expected(2, 1) = -0.5;
  h1_expected(2, 2) = 2.0;
  h2_expected[0] = 0.7;
  EXPECT_TRUE(
      h1_mod_dense.isApprox(h1_expected, testing::numerical_zero_tolerance));
  EXPECT_TRUE(
      h2_mod_dense.isApprox(h2_expected, testing::numerical_zero_tolerance));
}

TEST_F(ModelHamiltonianTest, PPPChainPotentialAndChargeOnly) {
  const int n = 4;
  double V = 1.0;
  double z = 1.0;
  auto lattice = LatticeGraph::chain(n);
  Eigen::MatrixXd h1_expected = Eigen::MatrixXd::Zero(n, n);
  h1_expected(0, 0) = h1_expected(1, 1) = h1_expected(2, 2) =
      h1_expected(3, 3) = -6.0;
  Eigen::VectorXd h2_expected = Eigen::VectorXd::Zero(n * n * n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      h2_expected[i * n * n * n + i * n * n + j * n + j] = V;
    }
  }

  double energy_offset_expected = 12.0;

  // test scalar form — returns tuple of (SparseMatrix, TwoBodyMap, double)
  auto [h1, h2, offset] = model_hamiltonians::detail::_build_ppp_integrals(
      lattice, 0.0, 0.0, 0.0, V, z);
  EXPECT_TRUE(Eigen::MatrixXd(h1).isApprox(h1_expected,
                                           testing::numerical_zero_tolerance));
  EXPECT_TRUE(testing_detail::to_dense_h2(h2, n).isApprox(
      h2_expected, testing::numerical_zero_tolerance));
  EXPECT_NEAR(offset, energy_offset_expected,
              testing::numerical_zero_tolerance);

  // test vector/matrix form
  Eigen::MatrixXd V_mat = Eigen::MatrixXd::Constant(n, n, V);
  Eigen::VectorXd z_vec = Eigen::VectorXd::Constant(n, z);
  auto [h1_explicit, h2_explicit, offset_explicit] =
      model_hamiltonians::detail::_build_ppp_integrals(lattice, 0.0, 0.0, 0.0,
                                                       V_mat, z_vec);
  EXPECT_TRUE(Eigen::MatrixXd(h1_explicit)
                  .isApprox(h1_expected, testing::numerical_zero_tolerance));
  EXPECT_TRUE(testing_detail::to_dense_h2(h2_explicit, n)
                  .isApprox(h2_expected, testing::numerical_zero_tolerance));
  EXPECT_NEAR(offset_explicit, energy_offset_expected,
              testing::numerical_zero_tolerance);

  // change V and z and verify different integrals
  V_mat(0, 1) = V_mat(1, 0) = 2.0;
  z_vec(0) = 2.0;
  auto [h1_mod, h2_mod, offset_mod] =
      model_hamiltonians::detail::_build_ppp_integrals(lattice, 0.0, 0.0, 0.0,
                                                       V_mat, z_vec);
  Eigen::MatrixXd h1_mod_dense = Eigen::MatrixXd(h1_mod);
  Eigen::VectorXd h2_mod_dense = testing_detail::to_dense_h2(h2_mod, n);
  EXPECT_FALSE(
      h1_mod_dense.isApprox(h1_expected, testing::numerical_zero_tolerance));
  EXPECT_FALSE(
      h2_mod_dense.isApprox(h2_expected, testing::numerical_zero_tolerance));
  h1_expected(0, 0) = -8.0;
  h1_expected(1, 1) = -12.0;
  h1_expected(2, 2) = -8.0;
  h1_expected(3, 3) = -8.0;
  h2_expected[0 * n * n * n + 0 * n * n + 1 * n + 1] = 2.0;
  h2_expected[1 * n * n * n + 1 * n * n + 0 * n + 0] = 2.0;
  energy_offset_expected = 22.0;
  EXPECT_TRUE(
      h1_mod_dense.isApprox(h1_expected, testing::numerical_zero_tolerance));
  EXPECT_TRUE(
      h2_mod_dense.isApprox(h2_expected, testing::numerical_zero_tolerance));
  EXPECT_NEAR(offset_mod, energy_offset_expected,
              testing::numerical_zero_tolerance);
}

TEST_F(ModelHamiltonianTest, OhnoPotentialTest) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);
  double epsilon_r = 0.9;
  double U = 0.4;
  double R = 1.2;

  auto potential = model_hamiltonians::ohno_potential(lattice, U, R, epsilon_r);
  Eigen::MatrixXd V_expected = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i != j)
        V_expected(i, j) = U / std::sqrt(1.0 + std::pow(R * U * epsilon_r, 2));
  EXPECT_TRUE(
      potential.isApprox(V_expected, testing::numerical_zero_tolerance));
}

TEST_F(ModelHamiltonianTest, MatagaNishimotoPotentialTest) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);
  double epsilon_r = 0.9;
  double U = 0.4;
  double R = 1.2;

  auto potential =
      model_hamiltonians::mataga_nishimoto_potential(lattice, U, R, epsilon_r);
  Eigen::MatrixXd V_expected = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i != j) V_expected(i, j) = U / (1.0 + R * U * epsilon_r);
  EXPECT_TRUE(
      potential.isApprox(V_expected, testing::numerical_zero_tolerance));
}

TEST_F(ModelHamiltonianTest, CreateHuckelHamiltonian) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);

  auto ham = model_hamiltonians::create_huckel_hamiltonian(lattice, -0.5, 1.0);
  EXPECT_TRUE(ham.has_container_type<SparseHamiltonianContainer>());
  EXPECT_TRUE(ham.is_restricted());
  EXPECT_FALSE(ham.has_two_body_integrals());

  // One-body integrals should match the detail builder
  auto h1_expected = Eigen::MatrixXd(
      model_hamiltonians::detail::_build_huckel_integrals(lattice, -0.5, 1.0));
  auto [h1_alpha, h1_beta] = ham.get_one_body_integrals();
  EXPECT_TRUE(
      h1_alpha.isApprox(h1_expected, testing::numerical_zero_tolerance));
  EXPECT_TRUE(h1_beta.isApprox(h1_expected, testing::numerical_zero_tolerance));
}

TEST_F(ModelHamiltonianTest, CreateHubbardHamiltonian) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);

  auto ham =
      model_hamiltonians::create_hubbard_hamiltonian(lattice, -0.5, 1.0, 0.3);
  EXPECT_TRUE(ham.has_container_type<SparseHamiltonianContainer>());
  EXPECT_TRUE(ham.is_restricted());
  EXPECT_TRUE(ham.has_two_body_integrals());
  EXPECT_NEAR(ham.get_core_energy(), 0.0, testing::numerical_zero_tolerance);

  // One-body
  auto [h1_ref, h2_ref] = model_hamiltonians::detail::_build_hubbard_integrals(
      lattice, -0.5, 1.0, 0.3);
  auto [h1_alpha, h1_beta] = ham.get_one_body_integrals();
  EXPECT_TRUE(h1_alpha.isApprox(Eigen::MatrixXd(h1_ref),
                                testing::numerical_zero_tolerance));

  // Two-body element check
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(ham.get_two_body_element(i, i, i, i), 0.3,
                testing::numerical_zero_tolerance);
  }
}

TEST_F(ModelHamiltonianTest, CreatePPPHamiltonian) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);
  double V = 1.0, z = 1.0;

  auto ham =
      model_hamiltonians::create_ppp_hamiltonian(lattice, 0.0, 0.0, 0.0, V, z);
  EXPECT_TRUE(ham.has_container_type<SparseHamiltonianContainer>());
  EXPECT_TRUE(ham.is_restricted());
  EXPECT_TRUE(ham.has_two_body_integrals());
  EXPECT_NEAR(ham.get_core_energy(), 12.0, testing::numerical_zero_tolerance);
}

TEST_F(ModelHamiltonianTest, PairwisePotentialCustomFunction) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);
  double U = 1.0;
  double R = 1.0;

  // Simple function: V_ij = U_ij * R_ij
  auto V = model_hamiltonians::pairwise_potential(
      lattice, U, R,
      [](int, int, double Uij, double Rij) { return Uij * Rij; });

  // Potential should be symmetric
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      EXPECT_NEAR(V(i, j), V(j, i), testing::numerical_zero_tolerance);
    }

  // Diagonal should be zero (pairwise only i < j)
  for (int i = 0; i < n; ++i) {
    EXPECT_NEAR(V(i, i), 0.0, testing::numerical_zero_tolerance);
  }

  // All off-diagonal should be U * R = 1.0
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) {
      if (i != j) {
        EXPECT_NEAR(V(i, j), 1.0, testing::numerical_zero_tolerance);
      }
    }
}

TEST_F(ModelHamiltonianTest, PairwisePotentialNearestNeighborOnly) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);
  double U = 1.0;
  double R = 1.0;

  auto V = model_hamiltonians::pairwise_potential(
      lattice, U, R, [](int, int, double Uij, double Rij) { return Uij * Rij; },
      true);

  // Only nearest-neighbor entries should be nonzero
  EXPECT_NEAR(V(0, 1), 1.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(V(1, 0), 1.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(V(1, 2), 1.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(V(2, 3), 1.0, testing::numerical_zero_tolerance);
  // Non-neighbor pair should be zero
  EXPECT_NEAR(V(0, 2), 0.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(V(0, 3), 0.0, testing::numerical_zero_tolerance);
  EXPECT_NEAR(V(1, 3), 0.0, testing::numerical_zero_tolerance);
}

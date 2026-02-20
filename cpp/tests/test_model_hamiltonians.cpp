// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/utils/model_hamiltonians/ppp.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
namespace model_hamiltonians = qdk::chemistry::utils::model_hamiltonians;

namespace testing_detail {
/** @brief Materialise a TwoBodyMap into a flat dense vector for test
 * comparison. */
Eigen::VectorXd to_dense_h2(const ModelHamiltonianContainer::TwoBodyMap& h2,
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

class LatticeGraphTest : public ::testing::Test {};

TEST_F(LatticeGraphTest, ChainConstructor) {
  // create chain lattic
  Eigen::MatrixXd adj(4, 4);
  adj << 0, 1, 0, 0,  // 1
      1, 0, 1, 0,     // 2
      0, 1, 0, 1,     // 3
      0, 0, 1, 0;     // 4
  LatticeGraph lattice(adj, false);

  EXPECT_EQ(lattice.num_sites(), 4);
  EXPECT_EQ(lattice.num_edges(), 3);
  EXPECT_TRUE(lattice.symmetry());
  EXPECT_DOUBLE_EQ(lattice.weight(0, 1), 1.0);
  EXPECT_DOUBLE_EQ(lattice.weight(1, 0), 1.0);
  EXPECT_FALSE(lattice.are_connected(0, 3));
  EXPECT_EQ(lattice.num_nonzeros(), 6);

  // use chain function
  auto chain_lattice = LatticeGraph::chain(4);
  EXPECT_EQ(chain_lattice.num_sites(), lattice.num_sites());
  EXPECT_EQ(chain_lattice.num_edges(), lattice.num_edges());
  EXPECT_TRUE(chain_lattice.symmetry());
  EXPECT_DOUBLE_EQ(chain_lattice.weight(0, 1), lattice.weight(0, 1));
  EXPECT_DOUBLE_EQ(chain_lattice.weight(1, 0), lattice.weight(1, 0));
  EXPECT_FALSE(chain_lattice.are_connected(0, 3));
  EXPECT_EQ(chain_lattice.num_nonzeros(), lattice.num_nonzeros());
  EXPECT_TRUE(chain_lattice.adjacency_matrix().isApprox(adj));

  // use periodic boundary condition
  auto ring_lattice = LatticeGraph::chain(4, true);
  Eigen::MatrixXd ring_adj(4, 4);
  ring_adj << 0, 1, 0, 1,  // 1
      1, 0, 1, 0,          // 2
      0, 1, 0, 1,          // 3
      1, 0, 1, 0;          // 4
  EXPECT_EQ(ring_lattice.num_sites(), 4);
  EXPECT_EQ(ring_lattice.num_edges(), 4);
  EXPECT_TRUE(ring_lattice.symmetry());
  EXPECT_DOUBLE_EQ(ring_lattice.weight(0, 1), 1.0);
  EXPECT_DOUBLE_EQ(ring_lattice.weight(1, 0), 1.0);
  EXPECT_TRUE(ring_lattice.are_connected(0, 3));
  EXPECT_EQ(ring_lattice.num_nonzeros(), 8);
  EXPECT_TRUE(ring_lattice.adjacency_matrix().isApprox(ring_adj));
}

TEST_F(LatticeGraphTest, SquareConstructor) {
  // create 4x4 square lattic
  Eigen::MatrixXd adj(16, 16);
  adj << 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 1
      1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,     // 2
      0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,     // 3
      0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,     // 4
      1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0,     // 5
      0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,     // 6
      0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,     // 7
      0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0,     // 8
      0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0,     // 9
      0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,     // 10
      0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,     // 11
      0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1,     // 12
      0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,     // 13
      0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,     // 14
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,     // 15
      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0;     // 16

  LatticeGraph lattice(adj, false);
  EXPECT_EQ(lattice.num_sites(), 16);
  EXPECT_EQ(lattice.num_edges(), 24);
  EXPECT_TRUE(lattice.symmetry());
  EXPECT_DOUBLE_EQ(lattice.weight(0, 1), 1.0);
  EXPECT_DOUBLE_EQ(lattice.weight(1, 0), 1.0);
  EXPECT_FALSE(lattice.are_connected(0, 2));

  // use square function
  auto square_lattice = LatticeGraph::square(4, 4);
  EXPECT_EQ(square_lattice.num_sites(), lattice.num_sites());
  EXPECT_EQ(square_lattice.num_edges(), lattice.num_edges());
  EXPECT_TRUE(square_lattice.symmetry());
  EXPECT_DOUBLE_EQ(square_lattice.weight(0, 1), lattice.weight(0, 1));
  EXPECT_DOUBLE_EQ(square_lattice.weight(1, 0), lattice.weight(1, 0));
  EXPECT_FALSE(square_lattice.are_connected(0, 2));
  EXPECT_EQ(square_lattice.num_nonzeros(), lattice.num_nonzeros());
  EXPECT_TRUE(square_lattice.adjacency_matrix().isApprox(adj));

  // use periodic boundary condition
  Eigen::MatrixXd adj_periodic(16, 16);
  adj << 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0,  // 1
      1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,     // 2
      0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0,     // 3
      1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1,     // 4
      1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0,     // 5
      0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0,     // 6
      0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0,     // 7
      0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0,     // 8
      0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 0,     // 9
      0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0,     // 10
      0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0,     // 11
      0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1,     // 12
      1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1,     // 13
      0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0,     // 14
      0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1,     // 15
      0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0;     // 16
  auto square_lattice_periodic = LatticeGraph::square(4, 4, true);
  EXPECT_EQ(square_lattice_periodic.num_sites(), 16);
  EXPECT_EQ(square_lattice_periodic.num_edges(), 32);
  EXPECT_TRUE(square_lattice_periodic.symmetry());
  EXPECT_DOUBLE_EQ(square_lattice_periodic.weight(0, 1), lattice.weight(0, 1));
  EXPECT_DOUBLE_EQ(square_lattice_periodic.weight(1, 0), lattice.weight(1, 0));
  EXPECT_TRUE(square_lattice_periodic.are_connected(0, 3));
}

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
  auto h1 = model_hamiltonians::detail::from_huckel(lattice, -0.5, 1.0);
  EXPECT_TRUE(Eigen::MatrixXd(h1).isApprox(h1_expected, 1e-12));

  // test vector/matrix huckel
  Eigen::MatrixXd t_mat = Eigen::MatrixXd::Constant(n, n, 1.0);
  Eigen::VectorXd eps_vec = Eigen::VectorXd::Constant(n, -0.5);
  auto h1_explicit =
      model_hamiltonians::detail::from_huckel(lattice, eps_vec, t_mat);
  EXPECT_TRUE(Eigen::MatrixXd(h1_explicit).isApprox(h1_expected, 1e-12));

  // modify t and eps and verify different integrals
  t_mat(1, 2) = t_mat(2, 1) = 0.5;
  eps_vec(2) = 2.0;
  auto h1_modified =
      model_hamiltonians::detail::from_huckel(lattice, eps_vec, t_mat);
  Eigen::MatrixXd h1_mod_dense = Eigen::MatrixXd(h1_modified);
  EXPECT_FALSE(h1_mod_dense.isApprox(h1_expected, 1e-12));
  EXPECT_NEAR(h1_mod_dense(1, 2), -0.5, 1e-12);
  EXPECT_NEAR(h1_mod_dense(2, 1), -0.5, 1e-12);
  EXPECT_NEAR(h1_mod_dense(2, 2), 2.0, 1e-12);
  h1_expected(1, 2) = h1_expected(2, 1) = -0.5;
  h1_expected(2, 2) = 2.0;
  EXPECT_TRUE(h1_mod_dense.isApprox(h1_expected, 1e-12));
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

  Eigen::VectorXd h2_expected = Eigen::VectorXd::Constant(n * n * n * n, 0.0);
  for (int i = 0; i < n; i++) {
    h2_expected[i * n * n * n + i * n * n + i * n + i] = 0.3;
  }

  // test scalar hubbard — returns tuple of (SparseMatrix, TwoBodyMap)
  auto [h1, h2] =
      model_hamiltonians::detail::from_hubbard(lattice, -0.5, 1.0, 0.3);
  EXPECT_TRUE(Eigen::MatrixXd(h1).isApprox(h1_expected, 1e-12));
  EXPECT_TRUE(testing_detail::to_dense_h2(h2, n).isApprox(h2_expected, 1e-12));

  // test vector/matrix hubbard
  Eigen::MatrixXd t_mat = Eigen::MatrixXd::Constant(n, n, 1.0);
  Eigen::VectorXd eps_vec = Eigen::VectorXd::Constant(n, -0.5);
  Eigen::VectorXd U_vec = Eigen::VectorXd::Constant(n, 0.3);
  auto [h1_explicit, h2_explicit] =
      model_hamiltonians::detail::from_hubbard(lattice, eps_vec, t_mat, U_vec);
  EXPECT_TRUE(Eigen::MatrixXd(h1_explicit).isApprox(h1_expected, 1e-12));
  EXPECT_TRUE(
      testing_detail::to_dense_h2(h2_explicit, n).isApprox(h2_expected, 1e-12));

  // modify U, t and eps and verify different integrals
  t_mat(1, 2) = t_mat(2, 1) = 0.5;
  eps_vec(2) = 2.0;
  U_vec(0) = 0.7;
  auto [h1_mod, h2_mod] =
      model_hamiltonians::detail::from_hubbard(lattice, eps_vec, t_mat, U_vec);
  Eigen::MatrixXd h1_mod_dense = Eigen::MatrixXd(h1_mod);
  Eigen::VectorXd h2_mod_dense = testing_detail::to_dense_h2(h2_mod, n);
  EXPECT_FALSE(h1_mod_dense.isApprox(h1_expected, 1e-12));
  EXPECT_NEAR(h1_mod_dense(1, 2), -0.5, 1e-12);
  EXPECT_NEAR(h1_mod_dense(2, 1), -0.5, 1e-12);
  EXPECT_NEAR(h1_mod_dense(2, 2), 2.0, 1e-12);
  EXPECT_FALSE(h2_mod_dense.isApprox(h2_expected, 1e-12));
  EXPECT_NEAR(h2_mod_dense[0], 0.7, 1e-12);
  h1_expected(1, 2) = h1_expected(2, 1) = -0.5;
  h1_expected(2, 2) = 2.0;
  h2_expected[0] = 0.7;
  EXPECT_TRUE(h1_mod_dense.isApprox(h1_expected, 1e-12));
  EXPECT_TRUE(h2_mod_dense.isApprox(h2_expected, 1e-12));
}

TEST_F(ModelHamiltonianTest, PPPChainPotentialAndChargeOnly) {
  const int n = 4;
  double V = 2.0;
  double z = 1.0;
  auto lattice = LatticeGraph::chain(n);
  Eigen::MatrixXd h1_expected = Eigen::MatrixXd::Zero(n, n);
  h1_expected(0, 0) = h1_expected(1, 1) = h1_expected(2, 2) =
      h1_expected(3, 3) = -6.0;
  Eigen::VectorXd h2_expected = Eigen::VectorXd::Zero(n * n * n * n);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < n; j++) {
      if (i == j) continue;
      h2_expected[i * n * n * n + i * n * n + j * n + j] = 0.5 * V;
    }
  }

  double energy_offset_expected = 12.0;

  // test scalar form — returns tuple of (SparseMatrix, TwoBodyMap, double)
  auto [h1, h2, offset] =
      model_hamiltonians::detail::from_ppp(lattice, 0.0, 0.0, 0.0, V, z);
  EXPECT_TRUE(Eigen::MatrixXd(h1).isApprox(h1_expected, 1e-12));
  EXPECT_TRUE(testing_detail::to_dense_h2(h2, n).isApprox(h2_expected, 1e-12));
  EXPECT_NEAR(offset, energy_offset_expected, 1e-12);

  // test vector/matrix form
  Eigen::MatrixXd V_mat = Eigen::MatrixXd::Constant(n, n, V);
  Eigen::VectorXd z_vec = Eigen::VectorXd::Constant(n, z);
  auto [h1_explicit, h2_explicit, offset_explicit] =
      model_hamiltonians::detail::from_ppp(lattice, 0.0, 0.0, 0.0, V_mat,
                                           z_vec);
  EXPECT_TRUE(Eigen::MatrixXd(h1_explicit).isApprox(h1_expected, 1e-12));
  EXPECT_TRUE(
      testing_detail::to_dense_h2(h2_explicit, n).isApprox(h2_expected, 1e-12));
  EXPECT_NEAR(offset_explicit, energy_offset_expected, 1e-12);

  // change V and z and verify different integrals
  V_mat(0, 1) = V_mat(1, 0) = 4.0;
  z_vec(0) = 2.0;
  auto [h1_mod, h2_mod, offset_mod] = model_hamiltonians::detail::from_ppp(
      lattice, 0.0, 0.0, 0.0, V_mat, z_vec);
  Eigen::MatrixXd h1_mod_dense = Eigen::MatrixXd(h1_mod);
  Eigen::VectorXd h2_mod_dense = testing_detail::to_dense_h2(h2_mod, n);
  EXPECT_FALSE(h1_mod_dense.isApprox(h1_expected, 1e-12));
  EXPECT_FALSE(h2_mod_dense.isApprox(h2_expected, 1e-12));
  h1_expected(0, 0) = -8.0;
  h1_expected(1, 1) = -12.0;
  h1_expected(2, 2) = -8.0;
  h1_expected(3, 3) = -8.0;
  h2_expected[0 * n * n * n + 0 * n * n + 1 * n + 1] = 2.0;
  h2_expected[1 * n * n * n + 1 * n * n + 0 * n + 0] = 2.0;
  energy_offset_expected = 22.0;
  EXPECT_TRUE(h1_mod_dense.isApprox(h1_expected, 1e-12));
  EXPECT_TRUE(h2_mod_dense.isApprox(h2_expected, 1e-12));
  EXPECT_NEAR(offset_mod, energy_offset_expected, 1e-12);
}

TEST_F(ModelHamiltonianTest, OhnoPotentialTest) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);
  double epsilon_r = 0.9;
  double U = 0.4;
  double R = 1.2;
  double constant = model_hamiltonians::detail::COULOMB_CONSTANT;

  auto potential = model_hamiltonians::ohno_potential(lattice, U, R, epsilon_r);
  Eigen::MatrixXd V_expected = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i != j)
        V_expected(i, j) =
            U / std::sqrt(1.0 + std::pow(R * U * epsilon_r / constant, 2));
  EXPECT_TRUE(potential.isApprox(V_expected, 1e-12));
}

TEST_F(ModelHamiltonianTest, MatagaNishimotoPotentialTest) {
  const int n = 4;
  auto lattice = LatticeGraph::chain(n);
  double epsilon_r = 0.9;
  double U = 0.4;
  double R = 1.2;
  double constant = model_hamiltonians::detail::COULOMB_CONSTANT;

  auto potential =
      model_hamiltonians::mataga_nishimoto_potential(lattice, U, R, epsilon_r);
  Eigen::MatrixXd V_expected = Eigen::MatrixXd::Zero(n, n);
  for (int i = 0; i < n; i++)
    for (int j = 0; j < n; j++)
      if (i != j) V_expected(i, j) = U / (1.0 + R * U * epsilon_r / constant);
  EXPECT_TRUE(potential.isApprox(V_expected, 1e-12));
}

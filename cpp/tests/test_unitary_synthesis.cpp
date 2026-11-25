// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <lapack.hh>
#include <limits>
#include <numeric>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using qdk::chemistry::utils::block_sparse_isometry_synthesis;
using qdk::chemistry::utils::dense_isometry_synthesis;

class UnitarySynthesisTest : public ::testing::Test {
 protected:
  static Eigen::MatrixXd random_orthogonal(Eigen::Index dim,
                                           std::uint32_t seed) {
    std::mt19937 generator(seed);
    std::normal_distribution<double> distribution;
    Eigen::MatrixXd q = Eigen::MatrixXd::NullaryExpr(
        dim, dim, [&] { return distribution(generator); });
    Eigen::VectorXd tau(dim);
    lapack::geqrf(dim, dim, q.data(), dim, tau.data());
    lapack::orgqr(dim, dim, dim, q.data(), dim, tau.data());
    return q;
  }

  // D L[m-1] ... L[0], checking that each layer acts on pairs of one parity.
  static Eigen::MatrixXd reconstruct(const Eigen::MatrixXd& angles,
                                     const Eigen::VectorXd& signs) {
    const Eigen::Index dim = signs.size();
    EXPECT_EQ(angles.cols(), std::max<Eigen::Index>(dim - 1, 0));
    Eigen::MatrixXd result = Eigen::MatrixXd::Identity(dim, dim);
    for (Eigen::Index layer = 0; layer < angles.rows(); ++layer) {
      int parity = -1;
      for (Eigen::Index pair = 0; pair < angles.cols(); ++pair) {
        const double angle = angles(layer, pair);
        if (angle == 0.0) continue;
        EXPECT_TRUE(parity < 0 || parity == pair % 2);
        parity = static_cast<int>(pair % 2);
        const Eigen::RowVectorXd upper = result.row(pair);
        const Eigen::RowVectorXd lower = result.row(pair + 1);
        result.row(pair) = std::cos(angle) * upper - std::sin(angle) * lower;
        result.row(pair + 1) =
            std::sin(angle) * upper + std::cos(angle) * lower;
      }
      EXPECT_GE(parity, 0) << "empty layer " << layer;
    }
    EXPECT_TRUE(signs.cwiseAbs().isOnes());
    return signs.asDiagonal() * result;
  }

  static Eigen::MatrixXd pad_isometry(const Eigen::MatrixXd& isometry,
                                      Eigen::Index blocks, Eigen::Index dim) {
    const Eigen::Index rows = isometry.rows() / blocks;
    Eigen::MatrixXd padded =
        Eigen::MatrixXd::Zero(blocks * dim, isometry.cols());
    for (Eigen::Index block = 0; block < blocks; ++block) {
      padded.middleRows(block * dim, rows) =
          isometry.middleRows(block * rows, rows);
    }
    return padded;
  }

  // Ry(angles[a]) on bit `target_bit`, gated on bit `control_bit` if >= 0.
  static Eigen::MatrixXd multiplexed_ry(const Eigen::RowVectorXd& angles,
                                        Eigen::Index added, Eigen::Index chi,
                                        int target_bit, int control_bit = -1) {
    Eigen::MatrixXd result =
        Eigen::MatrixXd::Identity(added * chi, added * chi);
    for (Eigen::Index q = 0; q < added; ++q) {
      if (((q >> target_bit) & 1) != 0 ||
          (control_bit >= 0 && ((q >> control_bit) & 1) == 0)) {
        continue;
      }
      const Eigen::Index flipped = q | (Eigen::Index{1} << target_bit);
      for (Eigen::Index a = 0; a < chi; ++a) {
        const double c = std::cos(angles(a) / 2), s = std::sin(angles(a) / 2);
        result(q * chi + a, q * chi + a) = c;
        result(q * chi + a, flipped * chi + a) = -s;
        result(flipped * chi + a, q * chi + a) = s;
        result(flipped * chi + a, flipped * chi + a) = c;
      }
    }
    return result;
  }

  // `unitary` on the input of added states with bit `control_bit` (all if -1).
  static Eigen::MatrixXd input_unitary(const Eigen::MatrixXd& unitary,
                                       Eigen::Index added,
                                       int control_bit = -1) {
    const Eigen::Index chi = unitary.rows();
    Eigen::MatrixXd result =
        Eigen::MatrixXd::Identity(added * chi, added * chi);
    for (Eigen::Index q = 0; q < added; ++q) {
      if (control_bit < 0 || ((q >> control_bit) & 1) != 0) {
        result.block(q * chi, q * chi, chi, chi) = unitary;
      }
    }
    return result;
  }

  // Checks B UCR_2 W_1 CNOT UCR_1 W_0 CNOT UCR_0 V against `target`.
  static void expect_dense_synthesis(const Eigen::MatrixXd& target,
                                     Eigen::Index chi,
                                     const Eigen::MatrixXd* input = nullptr) {
    const Eigen::Index added = target.rows() / chi;
    const Eigen::Index width = target.cols();
    for (const bool omit : {false, true}) {
      SCOPED_TRACE(omit ? "omitted right factor" : "full");
      const auto [ucr, layers, signs, right_factor] =
          dense_isometry_synthesis(input ? *input : target, chi, omit);
      ASSERT_EQ(ucr.rows(), added == 4 ? 3 : added / 2);
      ASSERT_EQ(layers.size(),
                (added == 4 ? 3u : 1u) + (added > 1 && !omit ? 1u : 0u));
      ASSERT_EQ(signs.size(), layers.size());
      ASSERT_TRUE((right_factor.transpose() * right_factor).isIdentity(1e-12));
      const auto factor = [&](std::size_t j) {
        return reconstruct(layers[j], signs[j]);
      };
      Eigen::MatrixXd circuit = factor(0);
      if (added == 2) {
        circuit *= multiplexed_ry(ucr.row(0), 2, chi, 0);
      } else if (added == 4) {
        Eigen::MatrixXd cnot = Eigen::MatrixXd::Zero(4 * chi, 4 * chi);
        for (Eigen::Index q = 0; q < 4; ++q) {
          cnot.block(((q & 2) != 0 ? (q ^ 1) : q) * chi, q * chi, chi, chi)
              .setIdentity();
        }
        circuit *= multiplexed_ry(ucr.row(2), 4, chi, 0, 1) *
                   input_unitary(factor(2), 4, 1) * cnot *
                   multiplexed_ry(ucr.row(1), 4, chi, 1, 0) *
                   input_unitary(factor(1), 4, 0) * cnot *
                   multiplexed_ry(ucr.row(0), 4, chi, 0);
      }
      if (added > 1 && !omit) {
        const Eigen::MatrixXd v = factor(layers.size() - 1);
        EXPECT_TRUE(v.topLeftCorner(width, width).isApprox(right_factor));
        circuit *= input_unitary(v, added);
      }
      const Eigen::MatrixXd expected =
          omit ? Eigen::MatrixXd(target * right_factor.transpose()) : target;
      EXPECT_TRUE(circuit.leftCols(width).isApprox(expected, 1.0e-10));
    }
  }

  // Checks that P_r B P_c has `target` as its first columns.
  static void expect_sparse_reconstruction(
      const Eigen::SparseMatrix<double>& input, const Eigen::MatrixXd& target) {
    const auto [rows, columns, angles, signs] =
        block_sparse_isometry_synthesis(input);
    std::vector<Eigen::Index> identity(target.rows());
    std::iota(identity.begin(), identity.end(), 0);
    for (auto permutation : {rows, columns}) {
      std::sort(permutation.begin(), permutation.end());
      ASSERT_EQ(permutation, identity);
    }
    const Eigen::MatrixXd b = reconstruct(angles, signs);
    Eigen::MatrixXd realized(target.rows(), target.cols());
    for (Eigen::Index row = 0; row < target.rows(); ++row) {
      for (Eigen::Index column = 0; column < target.cols(); ++column) {
        realized(rows[row], column) = b(row, columns[column]);
      }
    }
    EXPECT_TRUE(realized.isApprox(target, 1.0e-10));
  }
};

// Dense synthesis reconstructs random, structured and perturbed isometries.
TEST_F(UnitarySynthesisTest, ReconstructsDenseIsometries) {
  std::uint32_t seed = 50;
  for (const Eigen::Index added : {1, 2, 4}) {
    for (const Eigen::Index chi : {1, 2, 3, 4, 32}) {
      for (const Eigen::Index width : {Eigen::Index{1}, (chi + 1) / 2, chi}) {
        SCOPED_TRACE(::testing::Message()
                     << "random, added states " << added << " chi " << chi
                     << " width " << width);
        expect_dense_synthesis(
            random_orthogonal(added * chi, seed++).leftCols(width), chi);
      }
    }
  }
  SCOPED_TRACE("structured");
  Eigen::MatrixXd four_states = Eigen::MatrixXd::Zero(8, 2);
  four_states(6, 0) = 1.0;
  four_states(3, 1) = 1.0 / std::sqrt(2.0);
  four_states(4, 1) = -1.0 / std::sqrt(2.0);
  expect_dense_synthesis(four_states, 2);

  Eigen::MatrixXd rank_deficient = Eigen::MatrixXd::Zero(8, 4);
  rank_deficient.topLeftCorner(4, 2).setIdentity();
  rank_deficient.bottomRightCorner(4, 2).setIdentity();
  expect_dense_synthesis(rank_deficient, 4);

  Eigen::MatrixXd reflection = random_orthogonal(4, 21);
  if (reflection.determinant() > 0) {
    reflection.col(0) *= -1;
  }
  for (const Eigen::Index added : {2, 4}) {
    Eigen::MatrixXd zero_block = Eigen::MatrixXd::Zero(added * 4, 4);
    zero_block.topRows(4) = reflection;
    expect_dense_synthesis(zero_block, 4);
  }

  Eigen::MatrixXd alternating = Eigen::MatrixXd::Zero(8, 7);
  for (Eigen::Index state = 0; state < 7; ++state) {
    alternating(state, state) = state % 2 == 0 ? 1.0 : -1.0;
  }
  expect_dense_synthesis(alternating, 8);
  expect_dense_synthesis(
      pad_isometry(random_orthogonal(4 * 3, 19).leftCols(2), 4, 4), 4);
  expect_dense_synthesis(
      pad_isometry(random_orthogonal(2 * 3, 20).leftCols(3), 2, 4), 4);

  seed = 90;
  for (const Eigen::Index added : {1, 2, 4}) {
    SCOPED_TRACE(::testing::Message() << "within tolerance, added " << added);
    const Eigen::MatrixXd target =
        random_orthogonal(added * 8, seed++).leftCols(8);
    Eigen::MatrixXd symmetric = random_orthogonal(8, seed++);
    symmetric += symmetric.transpose().eval();
    symmetric *= 3.0e-8 / symmetric.norm();
    const Eigen::MatrixXd perturbed =
        target * (Eigen::MatrixXd::Identity(8, 8) + symmetric);
    expect_dense_synthesis(target, 8, &perturbed);
  }

  // Near-permutations put cosine-sine values near 0 and 1.
  for (const Eigen::Index added : {2, 4}) {
    for (const double eps : {1.0e-6, 1.0e-8}) {
      SCOPED_TRACE(::testing::Message()
                   << "near permutation, added " << added << " eps " << eps);
      const Eigen::Index dim = added * 8;
      Eigen::MatrixXd q = random_orthogonal(dim, seed++);
      q = (Eigen::MatrixXd::Identity(dim, dim) + eps * (q - q.transpose()))
              .householderQr()
              .householderQ();
      std::vector<Eigen::Index> rows(dim);
      std::iota(rows.begin(), rows.end(), 0);
      std::shuffle(rows.begin(), rows.end(), std::mt19937(seed++));
      expect_dense_synthesis(q(rows, Eigen::seqN(0, 8)), 8);
    }
  }
}

// Sparse synthesis reconstructs structured and random block-sparse isometries.
TEST_F(UnitarySynthesisTest, ReconstructsSparseIsometries) {
  std::vector<std::pair<std::string, Eigen::MatrixXd>> cases;
  Eigen::MatrixXd mixed = Eigen::MatrixXd::Zero(16, 8);
  mixed(0, 0) = -1.0;
  mixed.block(1, 1, 3, 2) = random_orthogonal(3, 20).leftCols(2);
  mixed.block(4, 3, 5, 3) = random_orthogonal(5, 21).leftCols(3);
  mixed.block(9, 6, 2, 2) = random_orthogonal(2, 22);
  cases.emplace_back("mixed block sizes", mixed);

  Eigen::MatrixXd revisiting = Eigen::MatrixXd::Zero(8, 5);
  for (Eigen::Index column = 0; column < 3; ++column) {
    revisiting(2 * column, column) = 1.0 / std::sqrt(2.0);
    revisiting(2 * column + 1, column) = 1.0 / std::sqrt(2.0);
  }
  for (Eigen::Index row = 0; row < 6; ++row) {
    revisiting(row, 3) = (row % 2 == 0 ? 1.0 : -1.0) / std::sqrt(6.0);
  }
  revisiting(6, 4) = 1.0;
  cases.emplace_back("columns revisiting earlier rows", revisiting);

  Eigen::MatrixXd signed_permutation = Eigen::MatrixXd::Zero(8, 4);
  signed_permutation(5, 0) = 1.0;
  signed_permutation(0, 1) = -1.0;
  signed_permutation(6, 2) = 1.0;
  signed_permutation(3, 3) = -1.0;
  cases.emplace_back("signed permutation", signed_permutation);
  cases.emplace_back("square orthogonal", random_orthogonal(8, 42));
  cases.emplace_back("padded row blocks",
                     pad_isometry(random_orthogonal(6, 81).leftCols(3), 2, 4));
  for (const Eigen::Index dim : {2, 3, 5, 16, 33, 64}) {
    Eigen::MatrixXd block =
        Eigen::MatrixXd::Zero(64, std::min<Eigen::Index>(dim, 32));
    block.topRows(dim) = random_orthogonal(dim, 0).leftCols(block.cols());
    cases.emplace_back("random, dim " + std::to_string(dim), block);
  }
  for (const auto& [name, target] : cases) {
    SCOPED_TRACE(name);
    expect_sparse_reconstruction(target.sparseView(), target);
  }

  // A block inside its tolerance is replaced by its polar factor.
  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 4);
  target.topLeftCorner(3, 3) = random_orthogonal(3, 41);
  target(4, 3) = 1.0;
  Eigen::MatrixXd perturbed = target;
  perturbed(4, 3) += 1.5e-8;
  SCOPED_TRACE("within tolerance");
  expect_sparse_reconstruction(perturbed.sparseView(), target);
}

// Explicitly stored zeros join no blocks.
TEST_F(UnitarySynthesisTest, IgnoresStoredZeros) {
  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(16, 6);
  target.block(2, 0, 4, 3) = random_orthogonal(4, 90).leftCols(3);
  target.block(9, 3, 3, 2) = random_orthogonal(3, 91).leftCols(2);
  target(14, 5) = -1.0;
  Eigen::SparseMatrix<double> stored = target.sparseView();
  stored.coeffRef(9, 0) = 0.0;
  stored.coeffRef(2, 5) = 0.0;
  expect_sparse_reconstruction(stored, target);
  EXPECT_EQ(block_sparse_isometry_synthesis(stored),
            block_sparse_isometry_synthesis(target.sparseView()));
}

// Invalid shapes and non-isometric, non-finite or empty inputs throw.
TEST_F(UnitarySynthesisTest, RejectsInvalidIsometries) {
  const Eigen::MatrixXd valid = random_orthogonal(8, 95).leftCols(2);
  for (const Eigen::Index chi :
       {Eigen::Index{0}, Eigen::Index{-2}, Eigen::Index{3}, Eigen::Index{1},
        std::numeric_limits<Eigen::Index>::max()}) {
    EXPECT_THROW(dense_isometry_synthesis(valid, chi), std::invalid_argument);
  }
  Eigen::MatrixXd non_isometric = Eigen::MatrixXd::Zero(8, 2);
  non_isometric(0, 0) = 1.0;
  non_isometric(1, 1) = 2.0;
  Eigen::MatrixXd not_finite = valid;
  not_finite(3, 1) = std::numeric_limits<double>::quiet_NaN();
  Eigen::MatrixXd zero_column = Eigen::MatrixXd::Zero(8, 2);
  zero_column(0, 0) = 1.0;
  for (const auto& invalid :
       {non_isometric, not_finite, zero_column, Eigen::MatrixXd(8, 0)}) {
    EXPECT_THROW(dense_isometry_synthesis(invalid, 4), std::invalid_argument);
    EXPECT_THROW(block_sparse_isometry_synthesis(invalid.sparseView()),
                 std::invalid_argument);
  }
  EXPECT_THROW(block_sparse_isometry_synthesis(
                   Eigen::MatrixXd::Identity(2, 3).sparseView()),
               std::invalid_argument);
}

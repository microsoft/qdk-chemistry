// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <limits>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <random>
#include <stdexcept>

using qdk::chemistry::utils::detail::decompose_2d;
using qdk::chemistry::utils::detail::decompose_block_diagonal_to_givens;
using qdk::chemistry::utils::detail::decompose_site_csd;
using qdk::chemistry::utils::detail::decompose_sparse_site;
using qdk::chemistry::utils::detail::decompose_unitary_to_givens;
using qdk::chemistry::utils::detail::GivensDecomposition;

namespace {

Eigen::MatrixXd reconstruct(const GivensDecomposition& decomposition) {
  const Eigen::Index dim =
      static_cast<Eigen::Index>(decomposition.phases.size());
  Eigen::MatrixXd result = Eigen::MatrixXd::Identity(dim, dim);
  for (std::size_t layer = 0; layer < decomposition.layer_angles.size();
       ++layer) {
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(dim, dim);
    const Eigen::Index offset = decomposition.layer_shifted[layer] ? 1 : 0;
    for (std::size_t slot = 0; slot < decomposition.layer_angles[layer].size();
         ++slot) {
      const Eigen::Index pair = offset + 2 * static_cast<Eigen::Index>(slot);
      const double angle = decomposition.layer_angles[layer][slot];
      const double cosine = std::cos(angle);
      const double sine = std::sin(angle);
      rotation(pair, pair) = cosine;
      rotation(pair, pair + 1) = -sine;
      rotation(pair + 1, pair) = sine;
      rotation(pair + 1, pair + 1) = cosine;
    }
    result = rotation * result;
  }
  for (Eigen::Index row = 0; row < dim; ++row) {
    if (decomposition.phases[static_cast<std::size_t>(row)]) {
      result.row(row) *= -1.0;
    }
  }
  return result;
}

Eigen::MatrixXd random_orthogonal(Eigen::Index dim, std::uint32_t seed) {
  std::mt19937 generator(seed);
  std::normal_distribution<double> distribution;
  Eigen::MatrixXd raw(dim, dim);
  for (Eigen::Index row = 0; row < dim; ++row) {
    for (Eigen::Index column = 0; column < dim; ++column) {
      raw(row, column) = distribution(generator);
    }
  }
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(raw);
  return qr.householderQ() * Eigen::MatrixXd::Identity(dim, dim);
}

std::vector<Eigen::Index> invert_permutation(
    const std::vector<Eigen::Index>& permutation) {
  std::vector<Eigen::Index> inverse(permutation.size());
  for (std::size_t index = 0; index < permutation.size(); ++index) {
    inverse[static_cast<std::size_t>(permutation[index])] =
        static_cast<Eigen::Index>(index);
  }
  return inverse;
}

void expect_reconstruction(const Eigen::MatrixXd& matrix,
                           double tolerance = 1.0e-12) {
  const auto decomposition = decompose_unitary_to_givens(matrix);
  EXPECT_TRUE(reconstruct(decomposition).isApprox(matrix, tolerance));
  EXPECT_EQ(decomposition.layer_angles.size(),
            decomposition.layer_shifted.size());
}

void expect_two_block_reconstruction(const Eigen::MatrixXd& stacked,
                                     double tolerance = 1.0e-11) {
  const Eigen::Index rows = stacked.rows() / 2;
  const Eigen::Index width = stacked.cols();
  const auto result =
      decompose_2d(stacked.topRows(rows), stacked.bottomRows(rows));
  EXPECT_TRUE((result.u_1.leftCols(width) * result.d_1.asDiagonal() * result.v)
                  .isApprox(stacked.topRows(rows), tolerance));
  EXPECT_TRUE((result.u_2.leftCols(width) * result.d_2.asDiagonal() * result.v)
                  .isApprox(stacked.bottomRows(rows), tolerance));
  EXPECT_TRUE((result.d_1.array().square() + result.d_2.array().square())
                  .matrix()
                  .isApprox(Eigen::VectorXd::Ones(width), tolerance));
}

}  // namespace

TEST(UnitarySynthesisTest, ReconstructsTwoBlockCsd) {
  for (const auto [rows, width] :
       {std::pair<Eigen::Index, Eigen::Index>{4, 2}, {4, 4}, {8, 3}}) {
    expect_two_block_reconstruction(
        random_orthogonal(2 * rows, 42).leftCols(width));
  }
}

TEST(UnitarySynthesisTest, ReconstructsRankDeficientTwoBlockCsd) {
  Eigen::MatrixXd stacked = Eigen::MatrixXd::Zero(8, 4);
  stacked.topLeftCorner(4, 2).setIdentity();
  stacked.bottomRightCorner(4, 2).setIdentity();
  expect_two_block_reconstruction(stacked);
}

TEST(UnitarySynthesisTest, ProducesOrthogonalSiteCsdFactors) {
  constexpr Eigen::Index dim = 4;
  constexpr Eigen::Index width = 3;
  const Eigen::MatrixXd matrix = random_orthogonal(4 * dim, 17).leftCols(width);
  const auto result = decompose_site_csd(matrix, dim);

  for (const auto& unitary : result.u) {
    EXPECT_TRUE((unitary.transpose() * unitary)
                    .isApprox(Eigen::MatrixXd::Identity(dim, dim), 1.0e-11));
  }
  EXPECT_TRUE((result.w_0.transpose() * result.w_0)
                  .isApprox(Eigen::MatrixXd::Identity(dim, dim), 1.0e-11));
  EXPECT_TRUE((result.w_1.transpose() * result.w_1)
                  .isApprox(Eigen::MatrixXd::Identity(dim, dim), 1.0e-11));
  EXPECT_TRUE((result.v.transpose() * result.v)
                  .isApprox(Eigen::MatrixXd::Identity(width, width), 1.0e-11));
}

TEST(UnitarySynthesisTest, ReconstructsMixedBlockDiagonalMatrix) {
  const std::vector<Eigen::Index> dimensions{1, 3, 4, 2};
  std::vector<Eigen::MatrixXd> blocks;
  Eigen::Index total_dim = 0;
  for (std::size_t index = 0; index < dimensions.size(); ++index) {
    blocks.push_back(random_orthogonal(dimensions[index],
                                       static_cast<std::uint32_t>(20 + index)));
    total_dim += dimensions[index];
  }

  Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(total_dim, total_dim);
  Eigen::Index offset = 0;
  for (const auto& block : blocks) {
    expected.block(offset, offset, block.rows(), block.cols()) = block;
    offset += block.rows();
  }

  EXPECT_TRUE(reconstruct(decompose_block_diagonal_to_givens(blocks))
                  .isApprox(expected, 1.0e-11));
}

TEST(UnitarySynthesisTest, ReconstructsSparseSiteIsometry) {
  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 4);
  const Eigen::MatrixXd first = random_orthogonal(3, 31).leftCols(2);
  const Eigen::MatrixXd second = random_orthogonal(2, 32).leftCols(1);
  target.block(0, 0, 3, 2) = first;
  target.block(4, 2, 2, 1) = second;
  target(7, 3) = 1.0;

  const auto result = decompose_sparse_site(target);
  const Eigen::MatrixXd block_diagonal = reconstruct(result.block_givens);
  const auto inverse_rows = invert_permutation(result.row_permutation);
  Eigen::MatrixXd reconstructed(target.rows(), target.cols());
  for (Eigen::Index row = 0; row < target.rows(); ++row) {
    for (Eigen::Index column = 0; column < target.cols(); ++column) {
      reconstructed(row, column) = block_diagonal(
          inverse_rows[static_cast<std::size_t>(row)],
          result.column_permutation[static_cast<std::size_t>(column)]);
    }
  }
  EXPECT_TRUE(reconstructed.isApprox(target, 1.0e-11));
}

TEST(UnitarySynthesisTest, RejectsInvalidCsdInputs) {
  EXPECT_THROW(decompose_2d(Eigen::MatrixXd::Identity(2, 2),
                            Eigen::MatrixXd::Identity(3, 2)),
               std::invalid_argument);
  EXPECT_THROW(
      decompose_2d(Eigen::MatrixXd::Zero(2, 2), Eigen::MatrixXd::Zero(2, 2)),
      std::invalid_argument);
  EXPECT_THROW(decompose_site_csd(Eigen::MatrixXd::Identity(4, 4), 2),
               std::invalid_argument);
  EXPECT_THROW(decompose_block_diagonal_to_givens({}), std::invalid_argument);
  EXPECT_THROW(decompose_sparse_site(Eigen::MatrixXd::Zero(4, 2)),
               std::invalid_argument);
}

TEST(UnitarySynthesisTest, ReconstructsScalarSigns) {
  expect_reconstruction(Eigen::MatrixXd::Identity(1, 1));
  Eigen::MatrixXd negative(1, 1);
  negative(0, 0) = -1.0;
  expect_reconstruction(negative);
}

TEST(UnitarySynthesisTest, ReconstructsPlaneRotation) {
  const double angle = 0.37;
  Eigen::MatrixXd rotation(2, 2);
  rotation << std::cos(angle), -std::sin(angle), std::sin(angle),
      std::cos(angle);
  expect_reconstruction(rotation);
}

TEST(UnitarySynthesisTest, ReconstructsSignedPermutation) {
  Eigen::MatrixXd matrix = Eigen::MatrixXd::Zero(4, 4);
  matrix(0, 2) = 1.0;
  matrix(1, 0) = -1.0;
  matrix(2, 3) = 1.0;
  matrix(3, 1) = 1.0;
  expect_reconstruction(matrix);
}

TEST(UnitarySynthesisTest, ReconstructsRandomOrthogonalMatrices) {
  for (const Eigen::Index dim : {2, 3, 4, 5, 8, 16}) {
    for (std::uint32_t seed = 0; seed < 8; ++seed) {
      expect_reconstruction(random_orthogonal(dim, seed), 1.0e-11);
    }
  }
}

TEST(UnitarySynthesisTest, RejectsInvalidMatrices) {
  EXPECT_THROW(decompose_unitary_to_givens(Eigen::MatrixXd::Zero(2, 3)),
               std::invalid_argument);
  EXPECT_THROW(decompose_unitary_to_givens(Eigen::MatrixXd::Zero(0, 0)),
               std::invalid_argument);
  EXPECT_THROW(decompose_unitary_to_givens(Eigen::MatrixXd::Zero(4, 4)),
               std::invalid_argument);
  Eigen::MatrixXd nonfinite = Eigen::MatrixXd::Identity(2, 2);
  nonfinite(0, 0) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(decompose_unitary_to_givens(nonfinite), std::invalid_argument);
}
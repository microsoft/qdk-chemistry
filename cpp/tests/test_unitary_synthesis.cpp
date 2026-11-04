// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <Eigen/SparseCore>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <random>
#include <stdexcept>
#include <vector>

using qdk::chemistry::utils::block_sparse_isometry_synthesis;
using qdk::chemistry::utils::dense_isometry_right_factor;
using qdk::chemistry::utils::dense_isometry_synthesis;
using qdk::chemistry::utils::DenseIsometrySynthesis;
using qdk::chemistry::utils::GivensDecomposition;
using qdk::chemistry::utils::SparseIsometrySynthesis;

namespace {

Eigen::MatrixXd reconstruct(const GivensDecomposition& decomposition) {
  const Eigen::Index dim =
      static_cast<Eigen::Index>(decomposition.phases.size());
  Eigen::MatrixXd result = Eigen::MatrixXd::Identity(dim, dim);
  EXPECT_EQ(decomposition.layer_shifted.size(),
            decomposition.layer_angles.size());
  for (std::size_t layer = 0; layer < decomposition.layer_angles.size();
       ++layer) {
    const Eigen::Index offset = decomposition.layer_shifted.at(layer) ? 1 : 0;
    EXPECT_EQ(
        static_cast<Eigen::Index>(decomposition.layer_angles[layer].size()),
        (dim - offset) / 2);
    for (std::size_t slot = 0; slot < decomposition.layer_angles[layer].size();
         ++slot) {
      const Eigen::Index pair = offset + 2 * static_cast<Eigen::Index>(slot);
      const double angle = decomposition.layer_angles[layer][slot];
      const Eigen::RowVectorXd upper = result.row(pair);
      const Eigen::RowVectorXd lower = result.row(pair + 1);
      result.row(pair) = std::cos(angle) * upper - std::sin(angle) * lower;
      result.row(pair + 1) = std::sin(angle) * upper + std::cos(angle) * lower;
    }
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

void expect_sparse_reconstruction(const SparseIsometrySynthesis& result,
                                  const Eigen::MatrixXd& target) {
  const Eigen::MatrixXd block_diagonal = reconstruct(result.block_givens);
  ASSERT_EQ(block_diagonal.rows(), target.rows());
  ASSERT_EQ(result.row_permutation.size(),
            static_cast<std::size_t>(target.rows()));
  ASSERT_EQ(result.column_permutation.size(), result.row_permutation.size());
  std::vector<Eigen::Index> expected_indices(result.row_permutation.size());
  std::iota(expected_indices.begin(), expected_indices.end(), 0);
  auto sorted_rows = result.row_permutation;
  auto sorted_columns = result.column_permutation;
  std::sort(sorted_rows.begin(), sorted_rows.end());
  std::sort(sorted_columns.begin(), sorted_columns.end());
  ASSERT_EQ(sorted_rows, expected_indices);
  ASSERT_EQ(sorted_columns, expected_indices);
  EXPECT_TRUE(
      (block_diagonal.transpose() * block_diagonal)
          .isApprox(Eigen::MatrixXd::Identity(target.rows(), target.rows()),
                    1.0e-11));
  std::vector<Eigen::Index> inverse_rows(result.row_permutation.size());
  for (std::size_t index = 0; index < inverse_rows.size(); ++index) {
    inverse_rows[static_cast<std::size_t>(result.row_permutation[index])] =
        static_cast<Eigen::Index>(index);
  }
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

// Pads each of the `blocks` row blocks of an isometry to `dim` rows, as when
// an isometry on fewer states is embedded in a larger register.
Eigen::MatrixXd pad_isometry(const Eigen::MatrixXd& isometry,
                             Eigen::Index blocks, Eigen::Index dim) {
  const Eigen::Index rows = isometry.rows() / blocks;
  Eigen::MatrixXd padded = Eigen::MatrixXd::Zero(blocks * dim, isometry.cols());
  for (Eigen::Index block = 0; block < blocks; ++block) {
    padded.middleRows(block * dim, rows) =
        isometry.middleRows(block * rows, rows);
  }
  return padded;
}

// Ry rotations on added qubit target_bit addressed by the input state, with
// an optional added control qubit. Basis state q * chi + a holds added-qubit
// state q, whose bit k is added qubit k, and input state a.
Eigen::MatrixXd multiplexed_ry(const std::vector<double>& angles,
                               Eigen::Index added, Eigen::Index chi,
                               int target_bit, int control_bit = -1) {
  Eigen::MatrixXd result = Eigen::MatrixXd::Identity(added * chi, added * chi);
  for (Eigen::Index q = 0; q < added; ++q) {
    if (((q >> target_bit) & 1) != 0 ||
        (control_bit >= 0 && ((q >> control_bit) & 1) == 0)) {
      continue;
    }
    const Eigen::Index flipped = q | (Eigen::Index{1} << target_bit);
    for (Eigen::Index a = 0; a < chi; ++a) {
      const double cosine = std::cos(angles[static_cast<std::size_t>(a)] / 2);
      const double sine = std::sin(angles[static_cast<std::size_t>(a)] / 2);
      const Eigen::Index zero = q * chi + a;
      const Eigen::Index one = flipped * chi + a;
      result(zero, zero) = cosine;
      result(zero, one) = -sine;
      result(one, zero) = sine;
      result(one, one) = cosine;
    }
  }
  return result;
}

// Applies `unitary` to the input register when added qubit control_bit is
// set, or unconditionally when control_bit is negative.
Eigen::MatrixXd input_unitary(const Eigen::MatrixXd& unitary,
                              Eigen::Index added, int control_bit = -1) {
  const Eigen::Index chi = unitary.rows();
  Eigen::MatrixXd result = Eigen::MatrixXd::Identity(added * chi, added * chi);
  for (Eigen::Index q = 0; q < added; ++q) {
    if (control_bit < 0 || ((q >> control_bit) & 1) != 0) {
      result.block(q * chi, q * chi, chi, chi) = unitary;
    }
  }
  return result;
}

// Unitary of the circuit described by a dense synthesis.
Eigen::MatrixXd dense_circuit(const DenseIsometrySynthesis& synthesis,
                              Eigen::Index added, Eigen::Index chi) {
  Eigen::MatrixXd circuit = reconstruct(synthesis.block_givens);
  if (added == 2) {
    circuit = circuit * multiplexed_ry(synthesis.rotation_angles[0], 2, chi, 0);
  } else if (added == 4) {
    // CNOT with added qubit 1 as control and added qubit 0 as target.
    Eigen::MatrixXd cnot = Eigen::MatrixXd::Zero(4 * chi, 4 * chi);
    for (Eigen::Index q = 0; q < 4; ++q) {
      const Eigen::Index mapped = (q & 2) != 0 ? (q ^ 1) : q;
      cnot.block(mapped * chi, q * chi, chi, chi).setIdentity();
    }
    circuit =
        circuit * multiplexed_ry(synthesis.rotation_angles[2], 4, chi, 0, 1) *
        input_unitary(reconstruct(synthesis.mixing_givens[1]), 4, 1) * cnot *
        multiplexed_ry(synthesis.rotation_angles[1], 4, chi, 1, 0) *
        input_unitary(reconstruct(synthesis.mixing_givens[0]), 4, 0) * cnot *
        multiplexed_ry(synthesis.rotation_angles[0], 4, chi, 0);
  }
  if (synthesis.right_factor_givens) {
    circuit = circuit *
              input_unitary(reconstruct(*synthesis.right_factor_givens), added);
  }
  return circuit;
}

void expect_dense_reconstruction(const DenseIsometrySynthesis& synthesis,
                                 const Eigen::MatrixXd& target,
                                 Eigen::Index chi, bool omitted) {
  const Eigen::Index added = target.rows() / chi;
  const Eigen::Index width = target.cols();
  const std::size_t num_rotations = added == 4 ? 3u : (added == 2 ? 1u : 0u);
  ASSERT_EQ(synthesis.rotation_angles.size(), num_rotations);
  ASSERT_EQ(synthesis.mixing_givens.size(), added == 4 ? 2u : 0u);
  for (const auto& angles : synthesis.rotation_angles) {
    EXPECT_EQ(angles.size(), static_cast<std::size_t>(chi));
  }
  for (const auto& givens : synthesis.mixing_givens) {
    EXPECT_EQ(givens.phases.size(), static_cast<std::size_t>(chi));
  }
  ASSERT_EQ(synthesis.right_factor.rows(), width);
  ASSERT_EQ(synthesis.right_factor.cols(), width);
  EXPECT_TRUE((synthesis.right_factor.transpose() * synthesis.right_factor)
                  .isApprox(Eigen::MatrixXd::Identity(width, width), 1.0e-11));
  ASSERT_EQ(synthesis.right_factor_givens.has_value(), !omitted && added > 1);
  if (synthesis.right_factor_givens) {
    const Eigen::MatrixXd padded = reconstruct(*synthesis.right_factor_givens);
    ASSERT_EQ(padded.rows(), chi);
    EXPECT_TRUE(padded.topLeftCorner(width, width)
                    .isApprox(synthesis.right_factor, 1.0e-11));
  }

  const Eigen::MatrixXd circuit = dense_circuit(synthesis, added, chi);
  ASSERT_EQ(circuit.rows(), target.rows());
  EXPECT_TRUE(
      (circuit.transpose() * circuit)
          .isApprox(Eigen::MatrixXd::Identity(circuit.rows(), circuit.cols()),
                    1.0e-11));
  const Eigen::MatrixXd expected =
      omitted ? Eigen::MatrixXd(target * synthesis.right_factor.transpose())
              : target;
  EXPECT_TRUE(circuit.leftCols(width).isApprox(expected, 1.0e-10));
}

// Checks both circuit variants and that the omitted factor is the helper's.
void expect_dense_synthesis(const Eigen::MatrixXd& target, Eigen::Index chi) {
  const auto full = dense_isometry_synthesis(target, chi);
  const auto omitted = dense_isometry_synthesis(target, chi, true);
  expect_dense_reconstruction(full, target, chi, false);
  expect_dense_reconstruction(omitted, target, chi, true);
  const Eigen::MatrixXd helper = dense_isometry_right_factor(target, chi);
  EXPECT_TRUE(helper == omitted.right_factor);
  EXPECT_TRUE(helper == full.right_factor);
  EXPECT_EQ(full.rotation_angles, omitted.rotation_angles);
  EXPECT_EQ(full.block_givens.layer_angles, omitted.block_givens.layer_angles);
  EXPECT_EQ(full.block_givens.phases, omitted.block_givens.phases);
}

}  // namespace

TEST(UnitarySynthesisTest, ReconstructsDenseIsometries) {
  {
    SCOPED_TRACE("random");
    std::uint32_t seed = 50;
    for (const Eigen::Index added : {1, 2, 4}) {
      for (const Eigen::Index chi : {1, 2, 4}) {
        for (const Eigen::Index width : {Eigen::Index{1}, chi}) {
          SCOPED_TRACE(::testing::Message()
                       << "added states " << added << " chi " << chi
                       << " width " << width);
          const Eigen::MatrixXd target =
              random_orthogonal(added * chi, seed++).leftCols(width);
          expect_dense_synthesis(target, chi);
        }
      }
    }
  }
  {
    SCOPED_TRACE("partial width");
    constexpr Eigen::Index chi = 4;
    expect_dense_synthesis(random_orthogonal(4 * chi, 17).leftCols(3), chi);
    expect_dense_synthesis(random_orthogonal(chi, 18).leftCols(3), chi);
  }
  {
    constexpr Eigen::Index chi = 32;
    std::uint32_t seed = 60;
    for (const Eigen::Index added : {2, 4}) {
      SCOPED_TRACE(::testing::Message() << "large, added states " << added);
      for (const Eigen::Index width : {Eigen::Index{20}, chi}) {
        expect_dense_synthesis(
            random_orthogonal(added * chi, seed++).leftCols(width), chi);
      }

      // Input sector k couples only to output sector (k + q) mod 4, so every
      // CSD block has a highly degenerate spectrum.
      constexpr Eigen::Index sectors = 4;
      constexpr Eigen::Index sector_dim = chi / sectors;
      Eigen::MatrixXd blocked = Eigen::MatrixXd::Zero(added * chi, chi);
      for (Eigen::Index k = 0; k < sectors; ++k) {
        const Eigen::MatrixXd block =
            random_orthogonal(added * sector_dim, seed++).leftCols(sector_dim);
        for (Eigen::Index q = 0; q < added; ++q) {
          blocked.block(q * chi + ((k + q) % sectors) * sector_dim,
                        k * sector_dim, sector_dim, sector_dim) =
              block.middleRows(q * sector_dim, sector_dim);
        }
      }
      expect_dense_synthesis(blocked, chi);
    }
  }
  {
    SCOPED_TRACE("structured");
    // Product states and rank-deficient blocks exercise the zero and pi
    // angles.
    Eigen::MatrixXd two_states = Eigen::MatrixXd::Zero(4, 2);
    two_states(2, 0) = 1.0;
    two_states(1, 1) = 1.0;
    expect_dense_synthesis(two_states, 2);

    Eigen::MatrixXd four_states = Eigen::MatrixXd::Zero(8, 2);
    four_states(6, 0) = 1.0;
    four_states(3, 1) = 1.0 / std::sqrt(2.0);
    four_states(4, 1) = -1.0 / std::sqrt(2.0);
    expect_dense_synthesis(four_states, 2);

    // The two added-qubit blocks each have rank two, so the CSD has angles 0
    // and pi with doubly degenerate singular values.
    Eigen::MatrixXd rank_deficient = Eigen::MatrixXd::Zero(8, 4);
    rank_deficient.topLeftCorner(4, 2).setIdentity();
    rank_deficient.bottomRightCorner(4, 2).setIdentity();
    expect_dense_synthesis(rank_deficient, 4);

    // Rows padded with zeros, as for an isometry embedded in a larger input
    // register.
    expect_dense_synthesis(
        pad_isometry(random_orthogonal(4 * 3, 19).leftCols(2), 4, 4), 4);
    expect_dense_synthesis(
        pad_isometry(random_orthogonal(2 * 3, 20).leftCols(3), 2, 4), 4);
  }
}

TEST(UnitarySynthesisTest, RightFactorHelperMatchesTheOmittedFactor) {
  constexpr Eigen::Index chi = 8;
  std::uint32_t seed = 70;
  for (const Eigen::Index added : {1, 2, 4}) {
    SCOPED_TRACE(::testing::Message() << "added states " << added);
    const Eigen::MatrixXd target =
        random_orthogonal(added * chi, seed++).leftCols(5);
    const Eigen::MatrixXd helper = dense_isometry_right_factor(target, chi);
    const auto omitted = dense_isometry_synthesis(target, chi, true);
    EXPECT_TRUE(helper == omitted.right_factor);
    if (added == 1) {
      EXPECT_TRUE(helper == Eigen::MatrixXd::Identity(5, 5));
    }

    // Applying V to the input register before the omitted circuit realizes
    // the isometry.
    Eigen::MatrixXd padded = Eigen::MatrixXd::Identity(chi, chi);
    padded.topLeftCorner(5, 5) = helper;
    const Eigen::MatrixXd circuit =
        dense_circuit(omitted, added, chi) * input_unitary(padded, added);
    EXPECT_TRUE(circuit.leftCols(5).isApprox(target, 1.0e-10));
  }

  // An isometry pre-rotated by the following factor V' on its outputs is
  // synthesized like any other isometry, as when V' is absorbed upstream.
  const Eigen::MatrixXd next = random_orthogonal(2 * chi, 80).leftCols(chi);
  const Eigen::MatrixXd following = dense_isometry_right_factor(next, chi);
  Eigen::MatrixXd rotated = random_orthogonal(4 * chi, 81).leftCols(6);
  for (Eigen::Index q = 0; q < 4; ++q) {
    rotated.middleRows(q * chi, chi) =
        following * rotated.middleRows(q * chi, chi);
  }
  expect_dense_synthesis(rotated, chi);
}

TEST(UnitarySynthesisTest, ReconstructsSparseIsometries) {
  {
    SCOPED_TRACE("mixed block sizes");
    // Column groups of sizes (rows, columns) = (1, 1), (3, 2), (5, 3), and
    // (2, 2) on disjoint rows, plus five unused rows, give diagonal blocks of
    // five different sizes.
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(16, 8);
    target(0, 0) = -1.0;
    target.block(1, 1, 3, 2) = random_orthogonal(3, 20).leftCols(2);
    target.block(4, 3, 5, 3) = random_orthogonal(5, 21).leftCols(3);
    target.block(9, 6, 2, 2) = random_orthogonal(2, 22);
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                 target);
  }
  {
    SCOPED_TRACE("three-row blocks and unused rows");
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 2);
    const double amplitude = 1.0 / std::sqrt(3.0);
    for (const Eigen::Index row : {0, 5, 6}) {
      target(row, 0) = amplitude;
    }
    for (const Eigen::Index row : {1, 2, 7}) {
      target(row, 1) = amplitude;
    }
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                 target);
  }
  {
    SCOPED_TRACE("full register");
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 4);
    target.block(0, 0, 3, 2) = random_orthogonal(3, 31).leftCols(2);
    target.block(4, 2, 2, 1) = random_orthogonal(2, 32).leftCols(1);
    target(7, 3) = 1.0;
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                 target);
  }
  {
    SCOPED_TRACE("columns revisiting earlier rows");
    // Columns 0-2 have disjoint supports; column 3 overlaps all of them, so the
    // first four columns must form a single block. Column 4 starts a new
    // block.
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 5);
    const double pair = 1.0 / std::sqrt(2.0);
    for (Eigen::Index column = 0; column < 3; ++column) {
      target(2 * column, column) = pair;
      target(2 * column + 1, column) = pair;
    }
    for (Eigen::Index row = 0; row < 6; ++row) {
      target(row, 3) = (row % 2 == 0 ? 1.0 : -1.0) / std::sqrt(6.0);
    }
    target(6, 4) = 1.0;
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                 target);
  }
  {
    SCOPED_TRACE("non-contiguous columns");
    // Columns 0 and 2 share rows {0, 1}, while column 1 lives on rows {2, 3}.
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 3);
    target(0, 0) = 0.6;
    target(1, 0) = 0.8;
    target(0, 2) = -0.8;
    target(1, 2) = 0.6;
    target(2, 1) = 1.0 / std::sqrt(2.0);
    target(3, 1) = -1.0 / std::sqrt(2.0);
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                 target);
  }
  {
    SCOPED_TRACE("padded row blocks");
    // Row blocks 0 and 2 couple to rows {0, 1}, blocks 1 and 3 to row 2, and
    // each block is padded from three to four rows.
    constexpr Eigen::Index blocks = 4;
    constexpr Eigen::Index rows = 3;
    Eigen::MatrixXd isometry = Eigen::MatrixXd::Zero(blocks * rows, 3);
    const Eigen::MatrixXd paired = random_orthogonal(4, 81).leftCols(2);
    for (Eigen::Index row = 0; row < 2; ++row) {
      isometry.row(row) = Eigen::RowVector3d(paired(row, 0), paired(row, 1), 0);
      isometry.row(2 * rows + row) =
          Eigen::RowVector3d(paired(2 + row, 0), paired(2 + row, 1), 0);
    }
    isometry(rows + 2, 2) = 0.6;
    isometry(3 * rows + 2, 2) = 0.8;
    const Eigen::MatrixXd target = pad_isometry(isometry, blocks, 4);
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                 target);
  }
  {
    SCOPED_TRACE("signed permutation");
    // Every column holds a single signed entry, so every block is 1x1.
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 4);
    target(5, 0) = 1.0;
    target(0, 1) = -1.0;
    target(6, 2) = 1.0;
    target(3, 3) = -1.0;
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                 target);
  }
  {
    SCOPED_TRACE("single rotation");
    const double angle = 0.37;
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(4, 1);
    target(1, 0) = std::cos(angle);
    target(2, 0) = std::sin(angle);
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                 target);
  }
  {
    SCOPED_TRACE("square orthogonal");
    const Eigen::MatrixXd target = random_orthogonal(8, 42);
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                 target);
  }
  {
    SCOPED_TRACE("within tolerance");
    // The one-row block of column 3 carries a residual of 3e-8: inside the
    // accepted 1e-8 * columns = 4e-8 but above the 1e-8 guard of a 1x1 block.
    Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 4);
    target.topLeftCorner(3, 3) = random_orthogonal(3, 41);
    target(4, 3) = 1.0;
    Eigen::MatrixXd perturbed = target;
    perturbed(4, 3) += 1.5e-8;
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(perturbed),
                                 target);
  }
  // One dense block of n rows, completed to an n x n orthogonal block.
  constexpr Eigen::Index chi = 32;
  for (const Eigen::Index dim : {2, 3, 4, 5, 8, 16, 33, 64}) {
    for (std::uint32_t seed = 0; seed < 2; ++seed) {
      SCOPED_TRACE(::testing::Message()
                   << "random, dim " << dim << " seed " << seed);
      const Eigen::Index width = std::min(dim, chi);
      Eigen::MatrixXd target = Eigen::MatrixXd::Zero(2 * chi, width);
      target.topRows(dim) = random_orthogonal(dim, seed).leftCols(width);
      expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                   target);
    }
  }
}

TEST(UnitarySynthesisTest, SparseAndDenseInputsAgree) {
  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(16, 6);
  target.block(2, 0, 4, 3) = random_orthogonal(4, 90).leftCols(3);
  target.block(9, 3, 3, 2) = random_orthogonal(3, 91).leftCols(2);
  target(14, 5) = -1.0;
  Eigen::SparseMatrix<double> sparse = target.sparseView();
  // Stored zeros do not join columns into one block.
  sparse.coeffRef(9, 0) = 0.0;
  sparse.coeffRef(2, 5) = 0.0;
  const auto from_dense = block_sparse_isometry_synthesis(target);
  const auto from_sparse = block_sparse_isometry_synthesis(sparse);
  expect_sparse_reconstruction(from_sparse, target);
  EXPECT_EQ(from_sparse.row_permutation, from_dense.row_permutation);
  EXPECT_EQ(from_sparse.column_permutation, from_dense.column_permutation);
  EXPECT_EQ(from_sparse.block_givens.layer_angles,
            from_dense.block_givens.layer_angles);
  EXPECT_EQ(from_sparse.block_givens.layer_shifted,
            from_dense.block_givens.layer_shifted);
  EXPECT_EQ(from_sparse.block_givens.phases, from_dense.block_givens.phases);
}

TEST(UnitarySynthesisTest, RejectsInvalidIsometries) {
  const Eigen::MatrixXd valid = random_orthogonal(8, 95).leftCols(2);
  // Input dimensions that are empty, do not divide the rows, leave a row
  // ratio other than 1, 2, or 4, or are smaller than the column count.
  for (const Eigen::Index chi :
       {Eigen::Index{0}, Eigen::Index{-2}, Eigen::Index{3}, Eigen::Index{1},
        std::numeric_limits<Eigen::Index>::max()}) {
    SCOPED_TRACE(::testing::Message() << "chi " << chi);
    EXPECT_THROW(dense_isometry_synthesis(valid, chi), std::invalid_argument);
    EXPECT_THROW(dense_isometry_right_factor(valid, chi),
                 std::invalid_argument);
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
    EXPECT_THROW(dense_isometry_right_factor(invalid, 4),
                 std::invalid_argument);
    EXPECT_THROW(block_sparse_isometry_synthesis(invalid),
                 std::invalid_argument);
    EXPECT_THROW(block_sparse_isometry_synthesis(
                     Eigen::SparseMatrix<double>(invalid.sparseView())),
                 std::invalid_argument);
  }
  EXPECT_THROW(block_sparse_isometry_synthesis(Eigen::MatrixXd::Identity(2, 3)),
               std::invalid_argument);
}

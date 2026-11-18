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
#include <string>
#include <utility>
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

// Checks that the permutations and blocks of a sparse synthesis give target.
void expect_sparse_reconstruction(const SparseIsometrySynthesis& result,
                                  const Eigen::MatrixXd& target) {
  const Eigen::MatrixXd block_diagonal = reconstruct(result.block_givens);
  ASSERT_EQ(block_diagonal.rows(), target.rows());
  std::vector<Eigen::Index> indices(static_cast<std::size_t>(target.rows()));
  std::iota(indices.begin(), indices.end(), 0);
  for (auto permutation : {result.row_permutation, result.column_permutation}) {
    std::sort(permutation.begin(), permutation.end());
    ASSERT_EQ(permutation, indices);
  }
  Eigen::MatrixXd reconstructed(target.rows(), target.cols());
  for (Eigen::Index row = 0; row < target.rows(); ++row) {
    for (Eigen::Index column = 0; column < target.cols(); ++column) {
      reconstructed(result.row_permutation[static_cast<std::size_t>(row)],
                    column) =
          block_diagonal(
              row, result.column_permutation[static_cast<std::size_t>(column)]);
    }
  }
  EXPECT_TRUE(reconstructed.isApprox(target, 1.0e-11));
}

// Pads each of the `blocks` row blocks of an isometry to `dim` rows.
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

// Ry on added qubit target_bit multiplexed by input state a of basis state
// q * chi + a, optionally controlled by added qubit control_bit.
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

// Applies `unitary` to the input register, controlled by added qubit
// control_bit unless it is negative.
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

// Checks both dense circuit variants of `input` (default `target`) against
// `target` and the right-factor helper.
void expect_dense_synthesis(const Eigen::MatrixXd& target, Eigen::Index chi,
                            bool generic = true,
                            const Eigen::MatrixXd* input = nullptr) {
  const Eigen::MatrixXd& source = input != nullptr ? *input : target;
  const Eigen::Index added = target.rows() / chi;
  const Eigen::Index width = target.cols();
  const auto full = dense_isometry_synthesis(source, chi);
  const auto omitted = dense_isometry_synthesis(source, chi, true);
  const Eigen::MatrixXd helper = dense_isometry_right_factor(source, chi);
  if (added == 1) {
    EXPECT_TRUE(helper.isIdentity());
  }
  const auto count_signs = [](const GivensDecomposition& givens) {
    return std::count_if(givens.phases.begin(), givens.phases.end(),
                         [](std::uint8_t phase) { return phase != 0; });
  };
  for (const auto* synthesis : {&full, &omitted}) {
    const bool omit = synthesis == &omitted;
    SCOPED_TRACE(omit ? "omitted right factor" : "full");
    ASSERT_EQ(synthesis->rotation_angles.size(),
              added == 4 ? 3u : (added == 2 ? 1u : 0u));
    ASSERT_EQ(synthesis->mixing_givens.size(), added == 4 ? 2u : 0u);
    for (const auto& angles : synthesis->rotation_angles) {
      ASSERT_EQ(angles.size(), static_cast<std::size_t>(chi));
    }
    ASSERT_TRUE(synthesis->right_factor.rows() == width &&
                synthesis->right_factor.cols() == width);
    EXPECT_TRUE(helper == synthesis->right_factor);
    ASSERT_EQ(synthesis->right_factor_givens.has_value(), !omit && added > 1);
    if (synthesis->right_factor_givens) {
      // The Givens data realizes V up to the row signs folded into UCR 0.
      EXPECT_TRUE(
          (reconstruct(*synthesis->right_factor_givens)
               .topLeftCorner(width, width) *
           synthesis->right_factor.transpose())
              .cwiseAbs()
              .isApprox(Eigen::MatrixXd::Identity(width, width), 1.0e-11));
    }
    if (generic) {
      // Every sign is folded, except one when B alone uses every state.
      auto num_signs = count_signs(synthesis->block_givens);
      for (const auto& givens : synthesis->mixing_givens) {
        num_signs += count_signs(givens);
      }
      if (synthesis->right_factor_givens) {
        num_signs += count_signs(*synthesis->right_factor_givens);
      }
      EXPECT_LE(num_signs, added == 1 && width == chi ? 1 : 0);
    }
    const Eigen::MatrixXd expected =
        omit ? Eigen::MatrixXd(target * synthesis->right_factor.transpose())
             : target;
    EXPECT_TRUE(dense_circuit(*synthesis, added, chi)
                    .leftCols(width)
                    .isApprox(expected, 1.0e-10));
  }
  // Only UCR 0 carries the folded signs of V.
  for (std::size_t index = 1; index < full.rotation_angles.size(); ++index) {
    EXPECT_EQ(full.rotation_angles[index], omitted.rotation_angles[index]);
  }
  if (generic) {
    EXPECT_EQ(full.block_givens.layer_angles,
              omitted.block_givens.layer_angles);
    EXPECT_EQ(full.block_givens.phases, omitted.block_givens.phases);
  }
}

}  // namespace

// Dense synthesis reconstructs random, large, structured, and perturbed
// isometries with and without the right factor.
TEST(UnitarySynthesisTest, ReconstructsDenseIsometries) {
  std::uint32_t seed = 50;
  for (const Eigen::Index added : {1, 2, 4}) {
    for (const Eigen::Index chi : {1, 2, 4}) {
      for (const Eigen::Index width : {Eigen::Index{1}, chi}) {
        SCOPED_TRACE(::testing::Message()
                     << "random, added states " << added << " chi " << chi
                     << " width " << width);
        expect_dense_synthesis(
            random_orthogonal(added * chi, seed++).leftCols(width), chi);
      }
    }
  }
  {
    SCOPED_TRACE("partial width");
    expect_dense_synthesis(random_orthogonal(16, 17).leftCols(3), 4);
    expect_dense_synthesis(random_orthogonal(4, 18).leftCols(3), 4);
  }
  {
    constexpr Eigen::Index chi = 32;
    constexpr Eigen::Index sectors = 4;
    constexpr Eigen::Index sector_dim = chi / sectors;
    seed = 60;
    for (const Eigen::Index added : {2, 4}) {
      SCOPED_TRACE(::testing::Message() << "large, added states " << added);
      for (const Eigen::Index width : {Eigen::Index{20}, chi}) {
        expect_dense_synthesis(
            random_orthogonal(added * chi, seed++).leftCols(width), chi);
      }
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
      SCOPED_TRACE(::testing::Message() << "zero block, added " << added);
      Eigen::MatrixXd zero_block = Eigen::MatrixXd::Zero(added * 4, 4);
      zero_block.topRows(4) = reflection;
      expect_dense_synthesis(zero_block, 4, false);
      for (const bool omit : {false, true}) {
        for (const auto& angles :
             dense_isometry_synthesis(zero_block, 4, omit).rotation_angles) {
          for (const double angle : angles) {
            EXPECT_LE(std::abs(angle), 1.0e-12);
          }
        }
      }
    }

    Eigen::MatrixXd alternating = Eigen::MatrixXd::Zero(8, 7);
    for (Eigen::Index state = 0; state < 7; ++state) {
      alternating(state, state) = state % 2 == 0 ? 1.0 : -1.0;
    }
    expect_dense_synthesis(alternating, 8, false);
    EXPECT_EQ(dense_isometry_synthesis(alternating, 8).block_givens.phases,
              (std::vector<std::uint8_t>{0, 1, 0, 1, 0, 1, 0, 1}));

    expect_dense_synthesis(
        pad_isometry(random_orthogonal(4 * 3, 19).leftCols(2), 4, 4), 4);
    expect_dense_synthesis(
        pad_isometry(random_orthogonal(2 * 3, 20).leftCols(3), 2, 4), 4);
  }
  seed = 90;
  for (const Eigen::Index added : {1, 2, 4}) {
    SCOPED_TRACE(::testing::Message()
                 << "within tolerance, added states " << added);
    const Eigen::MatrixXd target =
        random_orthogonal(added * 8, seed++).leftCols(8);
    Eigen::MatrixXd symmetric = random_orthogonal(8, seed++);
    symmetric += symmetric.transpose().eval();
    symmetric *= 3.0e-8 / symmetric.norm();
    const Eigen::MatrixXd perturbed =
        target * (Eigen::MatrixXd::Identity(8, 8) + symmetric);
    expect_dense_synthesis(target, 8, true, &perturbed);
  }
}

// Sparse synthesis reconstructs structured, square, perturbed, and random
// block-sparse isometries.
TEST(UnitarySynthesisTest, ReconstructsSparseIsometries) {
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

  Eigen::MatrixXd non_contiguous = Eigen::MatrixXd::Zero(8, 3);
  non_contiguous(0, 0) = 0.6;
  non_contiguous(1, 0) = 0.8;
  non_contiguous(0, 2) = -0.8;
  non_contiguous(1, 2) = 0.6;
  non_contiguous(2, 1) = 1.0 / std::sqrt(2.0);
  non_contiguous(3, 1) = -1.0 / std::sqrt(2.0);
  cases.emplace_back("non-contiguous columns", non_contiguous);

  Eigen::MatrixXd isometry = Eigen::MatrixXd::Zero(4 * 3, 3);
  const Eigen::MatrixXd paired = random_orthogonal(4, 81).leftCols(2);
  for (Eigen::Index row = 0; row < 2; ++row) {
    isometry.row(row) = Eigen::RowVector3d(paired(row, 0), paired(row, 1), 0);
    isometry.row(6 + row) =
        Eigen::RowVector3d(paired(2 + row, 0), paired(2 + row, 1), 0);
  }
  isometry(5, 2) = 0.6;
  isometry(11, 2) = 0.8;
  cases.emplace_back("padded row blocks", pad_isometry(isometry, 4, 4));

  Eigen::MatrixXd signed_permutation = Eigen::MatrixXd::Zero(8, 4);
  signed_permutation(5, 0) = 1.0;
  signed_permutation(0, 1) = -1.0;
  signed_permutation(6, 2) = 1.0;
  signed_permutation(3, 3) = -1.0;
  cases.emplace_back("signed permutation", signed_permutation);

  Eigen::MatrixXd single_rotation = Eigen::MatrixXd::Zero(4, 1);
  single_rotation(1, 0) = std::cos(0.37);
  single_rotation(2, 0) = std::sin(0.37);
  cases.emplace_back("single rotation", single_rotation);
  cases.emplace_back("square orthogonal", random_orthogonal(8, 42));

  constexpr Eigen::Index chi = 32;
  for (const Eigen::Index dim : {2, 3, 5, 16, 33, 64}) {
    Eigen::MatrixXd dense_block =
        Eigen::MatrixXd::Zero(2 * chi, std::min(dim, chi));
    dense_block.topRows(dim) =
        random_orthogonal(dim, 0).leftCols(dense_block.cols());
    cases.emplace_back("random, dim " + std::to_string(dim), dense_block);
  }
  for (const auto& [name, target] : cases) {
    SCOPED_TRACE(name);
    expect_sparse_reconstruction(block_sparse_isometry_synthesis(target),
                                 target);
  }

  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 4);
  target.topLeftCorner(3, 3) = random_orthogonal(3, 41);
  target(4, 3) = 1.0;
  Eigen::MatrixXd perturbed = target;
  perturbed(4, 3) += 1.5e-8;
  SCOPED_TRACE("within tolerance");
  expect_sparse_reconstruction(block_sparse_isometry_synthesis(perturbed),
                               target);
}

// Dense and sparse inputs, including stored zeros, give the same synthesis.
TEST(UnitarySynthesisTest, SparseAndDenseInputsAgree) {
  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(16, 6);
  target.block(2, 0, 4, 3) = random_orthogonal(4, 90).leftCols(3);
  target.block(9, 3, 3, 2) = random_orthogonal(3, 91).leftCols(2);
  target(14, 5) = -1.0;
  Eigen::SparseMatrix<double> sparse = target.sparseView();
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

// Invalid input dimensions and non-isometric, non-finite, zero, empty, or wide
// inputs throw.
TEST(UnitarySynthesisTest, RejectsInvalidIsometries) {
  const Eigen::MatrixXd valid = random_orthogonal(8, 95).leftCols(2);
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

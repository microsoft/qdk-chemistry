// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <complex>
#include <limits>
#include <memory>
#include <numeric>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

using qdk::chemistry::data::Configuration;
using qdk::chemistry::data::MPSContainer;
using qdk::chemistry::data::MPSSite;
using qdk::chemistry::data::SymmetryBlockedTensor;
using qdk::chemistry::data::SymmetryLabel;
using qdk::chemistry::data::SymmetryProduct;
using qdk::chemistry::data::Tensor;
using qdk::chemistry::utils::detail::block_sparse_unitary_synthesis;
using qdk::chemistry::utils::detail::dense_unitary_synthesis;
using qdk::chemistry::utils::detail::DenseSiteSynthesis;
using qdk::chemistry::utils::detail::GivensDecomposition;
using qdk::chemistry::utils::detail::matrix_product_state_synthesis;
using qdk::chemistry::utils::detail::SparseSiteSynthesis;

namespace detail {

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

void expect_sparse_reconstruction(const SparseSiteSynthesis& result,
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

template <typename Scalar>
MPSSite make_site(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& packed,
    Eigen::Index physical_dimension,
    std::vector<Configuration> physical_basis = {}) {
  using SBT = SymmetryBlockedTensor<3, Scalar>;
  auto trivial =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  const SymmetryLabel label;
  typename SBT::ExtentsArray extents;
  extents[0][label] =
      static_cast<std::size_t>(packed.rows() / physical_dimension);
  extents[1][label] = static_cast<std::size_t>(physical_dimension);
  extents[2][label] = static_cast<std::size_t>(packed.cols());
  typename SBT::BlockMap blocks{
      {{label, label, label},
       std::make_shared<const Tensor<3, Scalar>>(packed)}};
  auto tensor = std::make_shared<const MPSSite::TensorVariant>(
      SBT({trivial, trivial, trivial}, extents, std::move(blocks)));
  return MPSSite(tensor, MPSSite::SectorOrders{{{label}, {label}, {label}}},
                 std::move(physical_basis));
}

// Builds the site whose packed (left * physical, right) matrix holds
// isometry(p * right + b, a) at row a * physical + p and column b.
MPSSite site_from_isometry(const Eigen::MatrixXd& isometry,
                           Eigen::Index physical) {
  const Eigen::Index right = isometry.rows() / physical;
  const Eigen::Index left = isometry.cols();
  Eigen::MatrixXd packed(left * physical, right);
  for (Eigen::Index a = 0; a < left; ++a) {
    for (Eigen::Index p = 0; p < physical; ++p) {
      packed.row(a * physical + p) =
          isometry.block(p * right, a, right, 1).transpose();
    }
  }
  return make_site(packed, physical);
}

// Dense synthesis of the site built from a physical-major isometry whose rows
// fill the bond register.
DenseSiteSynthesis synthesize_dense_site(const MPSSite& site,
                                         Eigen::Index chi) {
  return dense_unitary_synthesis(site, chi);
}

SparseSiteSynthesis synthesize_sparse_site(const MPSSite& site,
                                           Eigen::Index chi) {
  return block_sparse_unitary_synthesis(site, chi);
}

MPSContainer make_container(const std::vector<MPSSite>& sites) {
  std::vector<MPSContainer::SitePtr> pointers;
  for (const auto& site : sites) {
    pointers.push_back(std::make_shared<const MPSSite>(site));
  }
  return MPSContainer(
      std::move(pointers),
      std::make_shared<qdk::chemistry::data::ModelOrbitals>(sites.size()));
}

DenseSiteSynthesis decompose_dense_target(const Eigen::MatrixXd& target,
                                          Eigen::Index chi) {
  return synthesize_dense_site(site_from_isometry(target, target.rows() / chi),
                               chi);
}

// Pads the right-bond rows of a physical-major isometry to ancilla_dim.
Eigen::MatrixXd pad_isometry(const Eigen::MatrixXd& isometry,
                             Eigen::Index physical, Eigen::Index ancilla_dim) {
  const Eigen::Index right = isometry.rows() / physical;
  Eigen::MatrixXd padded =
      Eigen::MatrixXd::Zero(physical * ancilla_dim, isometry.cols());
  for (Eigen::Index p = 0; p < physical; ++p) {
    padded.middleRows(p * ancilla_dim, right) =
        isometry.middleRows(p * right, right);
  }
  return padded;
}

// Decomposes the site built from a physical-major isometry and checks the
// result against the isometry padded to the bond register.
SparseSiteSynthesis expect_sparse_site_reconstruction(
    const Eigen::MatrixXd& isometry, Eigen::Index physical,
    Eigen::Index ancilla_dim) {
  auto result = synthesize_sparse_site(site_from_isometry(isometry, physical),
                                       ancilla_dim);
  expect_sparse_reconstruction(result,
                               pad_isometry(isometry, physical, ancilla_dim));
  return result;
}

// Ry rotations on physical qubit target_bit addressed by the bond state, with
// an optional physical control qubit. Basis state p * chi + a holds physical
// state p, whose bit k is physical qubit k, and bond state a.
Eigen::MatrixXd multiplexed_ry(const std::vector<double>& angles,
                               Eigen::Index physical, Eigen::Index chi,
                               int target_bit, int control_bit = -1) {
  Eigen::MatrixXd result =
      Eigen::MatrixXd::Identity(physical * chi, physical * chi);
  for (Eigen::Index p = 0; p < physical; ++p) {
    if (((p >> target_bit) & 1) != 0 ||
        (control_bit >= 0 && ((p >> control_bit) & 1) == 0)) {
      continue;
    }
    const Eigen::Index flipped = p | (Eigen::Index{1} << target_bit);
    for (Eigen::Index a = 0; a < chi; ++a) {
      const double cosine = std::cos(angles[static_cast<std::size_t>(a)] / 2);
      const double sine = std::sin(angles[static_cast<std::size_t>(a)] / 2);
      const Eigen::Index zero = p * chi + a;
      const Eigen::Index one = flipped * chi + a;
      result(zero, zero) = cosine;
      result(zero, one) = -sine;
      result(one, zero) = sine;
      result(one, one) = cosine;
    }
  }
  return result;
}

Eigen::MatrixXd controlled_bond_unitary(const Eigen::MatrixXd& unitary,
                                        Eigen::Index physical,
                                        int control_bit) {
  const Eigen::Index chi = unitary.rows();
  Eigen::MatrixXd result =
      Eigen::MatrixXd::Identity(physical * chi, physical * chi);
  for (Eigen::Index p = 0; p < physical; ++p) {
    if (((p >> control_bit) & 1) != 0) {
      result.block(p * chi, p * chi, chi, chi) = unitary;
    }
  }
  return result;
}

void expect_dense_reconstruction(const DenseSiteSynthesis& synthesis,
                                 const Eigen::MatrixXd& target,
                                 Eigen::Index chi) {
  const Eigen::Index physical = target.rows() / chi;
  const Eigen::Index width = target.cols();
  ASSERT_EQ(synthesis.rotation_angles.size(), physical == 4 ? 3u : 1u);
  ASSERT_EQ(synthesis.mixing_givens.size(), physical == 4 ? 2u : 0u);
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

  // Unitary of the dense site circuit described by the synthesis.
  const Eigen::MatrixXd block = reconstruct(synthesis.block_givens);
  Eigen::MatrixXd circuit;
  if (physical == 2) {
    circuit = block * multiplexed_ry(synthesis.rotation_angles[0], 2, chi, 0);
  } else {
    // CNOT with physical qubit 1 as control and physical qubit 0 as target.
    Eigen::MatrixXd cnot = Eigen::MatrixXd::Zero(4 * chi, 4 * chi);
    for (Eigen::Index p = 0; p < 4; ++p) {
      const Eigen::Index mapped = (p & 2) != 0 ? (p ^ 1) : p;
      cnot.block(mapped * chi, p * chi, chi, chi).setIdentity();
    }
    circuit =
        block * multiplexed_ry(synthesis.rotation_angles[2], 4, chi, 0, 1) *
        controlled_bond_unitary(reconstruct(synthesis.mixing_givens[1]), 4, 1) *
        cnot * multiplexed_ry(synthesis.rotation_angles[1], 4, chi, 1, 0) *
        controlled_bond_unitary(reconstruct(synthesis.mixing_givens[0]), 4, 0) *
        cnot * multiplexed_ry(synthesis.rotation_angles[0], 4, chi, 0);
  }
  ASSERT_EQ(circuit.rows(), target.rows());
  EXPECT_TRUE(
      (circuit.transpose() * circuit)
          .isApprox(Eigen::MatrixXd::Identity(circuit.rows(), circuit.cols()),
                    1.0e-11));
  EXPECT_TRUE(circuit.leftCols(width).isApprox(
      target * synthesis.right_factor.transpose(), 1.0e-10));
}

TEST(UnitarySynthesisTest, ReconstructsPartialWidthDenseSite) {
  // Three left-bond states on a four-state bond register leave one CSD column
  // of each step unconstrained.
  constexpr Eigen::Index chi = 4;
  const Eigen::MatrixXd target = random_orthogonal(4 * chi, 17).leftCols(3);
  expect_dense_reconstruction(decompose_dense_target(target, chi), target, chi);
}

TEST(UnitarySynthesisTest, ReconstructsSparseSiteWithMixedBlockSizes) {
  // Column groups of sizes (rows, columns) = (1, 1), (3, 2), (5, 3), and
  // (2, 2) on disjoint rows, plus five unused rows, give diagonal blocks of
  // five different sizes.
  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(16, 8);
  target(0, 0) = -1.0;
  target.block(1, 1, 3, 2) = random_orthogonal(3, 20).leftCols(2);
  target.block(4, 3, 5, 3) = random_orthogonal(5, 21).leftCols(3);
  target.block(9, 6, 2, 2) = random_orthogonal(2, 22);

  const auto result = expect_sparse_site_reconstruction(target, 2, 8);
  EXPECT_EQ(result.block_givens.phases.size(), 16u);
  EXPECT_EQ(result.block_givens.layer_angles.size(),
            result.block_givens.layer_shifted.size());
  const std::vector<Eigen::Index> target_columns(
      result.column_permutation.begin(), result.column_permutation.begin() + 8);
  EXPECT_EQ(target_columns,
            (std::vector<Eigen::Index>{10, 5, 6, 0, 1, 2, 8, 9}));
  EXPECT_EQ(result.row_permutation.front(), 4);
}

TEST(UnitarySynthesisTest, ReconstructsTwoThreeRowBlocksAndUnusedRows) {
  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 2);
  const double amplitude = 1.0 / std::sqrt(3.0);
  for (const Eigen::Index row : {0, 5, 6}) {
    target(row, 0) = amplitude;
  }
  for (const Eigen::Index row : {1, 2, 7}) {
    target(row, 1) = amplitude;
  }
  const auto result = expect_sparse_site_reconstruction(target, 2, 4);
  EXPECT_EQ(result.row_permutation,
            (std::vector<Eigen::Index>{0, 5, 6, 1, 2, 7, 3, 4}));
  EXPECT_EQ(result.column_permutation[0], 0);
  EXPECT_EQ(result.column_permutation[1], 3);
  const auto block_diagonal = reconstruct(result.block_givens);
  EXPECT_TRUE(block_diagonal.bottomRightCorner(2, 2).isApprox(
      Eigen::MatrixXd::Identity(2, 2), 1.0e-12));
}

TEST(UnitarySynthesisTest, ReconstructsSparseSiteIsometry) {
  // Two physical states and four right-bond states fill the register.
  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 4);
  const Eigen::MatrixXd first = random_orthogonal(3, 31).leftCols(2);
  const Eigen::MatrixXd second = random_orthogonal(2, 32).leftCols(1);
  target.block(0, 0, 3, 2) = first;
  target.block(4, 2, 2, 1) = second;
  target(7, 3) = 1.0;

  expect_sparse_site_reconstruction(target, 2, 4);
}

TEST(UnitarySynthesisTest, MergesSparseSiteColumnsThatRevisitEarlierRows) {
  // Columns 0-2 have disjoint supports; column 3 overlaps all of them, so the
  // first four columns must form a single block. Column 4 starts a new block.
  // The five left-bond states need an eight-state register.
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

  expect_sparse_site_reconstruction(target, 2, 8);
}

TEST(UnitarySynthesisTest, GroupsNonContiguousSparseSiteColumns) {
  // Columns 0 and 2 share rows {0, 1}, while column 1 lives on rows {2, 3}.
  // The columns split into two 2x2 blocks rather than one 4x4 block.
  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(8, 3);
  target(0, 0) = 0.6;
  target(1, 0) = 0.8;
  target(0, 2) = -0.8;
  target(1, 2) = 0.6;
  target(2, 1) = 1.0 / std::sqrt(2.0);
  target(3, 1) = -1.0 / std::sqrt(2.0);

  const auto result = expect_sparse_site_reconstruction(target, 2, 4);
  EXPECT_EQ(result.column_permutation[0] / 2, result.column_permutation[2] / 2);
  EXPECT_NE(result.column_permutation[0] / 2, result.column_permutation[1] / 2);
  EXPECT_LE(result.column_permutation[1], 3);
  EXPECT_LE(result.block_givens.layer_angles.size(), 1u);
}

TEST(UnitarySynthesisTest, DecomposesSymmetryBlockedSparseMpsSite) {
  // Particle-number blocks with right sector N_L + N_p, stored in non-default
  // sector orders. The empty left sector couples to five right-bond states and
  // the singly occupied sector to the other six, so the site splits into two
  // diagonal blocks.
  using SBT = SymmetryBlockedTensor<3>;
  namespace axes = qdk::chemistry::data::axes;
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(2)}));
  const SymmetryLabel zero({axes::particle_number_value(0)});
  const SymmetryLabel one({axes::particle_number_value(1)});
  const SymmetryLabel two({axes::particle_number_value(2)});
  SBT::ExtentsArray extents;
  extents[0] = {{zero, 1}, {one, 2}};
  extents[1] = {{zero, 1}, {one, 2}, {two, 1}};
  extents[2] = {{zero, 1}, {one, 2}, {two, 2}};

  const Eigen::VectorXd empty = random_orthogonal(7, 101).col(0);
  const Eigen::MatrixXd occupied = random_orthogonal(6, 102).leftCols(2);
  Eigen::MatrixXd zero_zero_zero(1, 1);
  Eigen::MatrixXd zero_one_one(2, 2);
  Eigen::MatrixXd zero_two_two(1, 2);
  Eigen::MatrixXd one_zero_one(2, 2);
  Eigen::MatrixXd one_one_two(4, 2);
  zero_zero_zero << empty(0);
  zero_one_one << empty(1), empty(2), empty(3), empty(4);
  zero_two_two << empty(5), empty(6);
  for (Eigen::Index l = 0; l < 2; ++l) {
    one_zero_one.row(l) = occupied.col(l).head(2).transpose();
    for (Eigen::Index p = 0; p < 2; ++p) {
      one_one_two.row(l * 2 + p) =
          occupied.col(l).segment(2 + 2 * p, 2).transpose();
    }
  }
  SBT::BlockMap blocks{
      {{zero, zero, zero},
       std::make_shared<const Eigen::MatrixXd>(zero_zero_zero)},
      {{zero, one, one}, std::make_shared<const Eigen::MatrixXd>(zero_one_one)},
      {{zero, two, two}, std::make_shared<const Eigen::MatrixXd>(zero_two_two)},
      {{one, zero, one}, std::make_shared<const Eigen::MatrixXd>(one_zero_one)},
      {{one, one, two}, std::make_shared<const Eigen::MatrixXd>(one_one_two)}};
  const MPSSite blocked(
      std::make_shared<const MPSSite::TensorVariant>(
          SBT({symmetry, symmetry, symmetry}, extents, std::move(blocks))),
      MPSSite::SectorOrders{{{one, zero}, {zero, one, two}, {two, zero, one}}});

  constexpr Eigen::Index physical = 4;
  constexpr Eigen::Index chi = 8;
  const Eigen::MatrixXd packed = std::get<Eigen::MatrixXd>(blocked.to_dense());
  Eigen::MatrixXd target = Eigen::MatrixXd::Zero(physical * chi, 3);
  for (Eigen::Index a = 0; a < 3; ++a) {
    for (Eigen::Index p = 0; p < physical; ++p) {
      target.block(p * chi, a, 5, 1) = packed.row(a * physical + p).transpose();
    }
  }

  const auto from_blocks = synthesize_sparse_site(blocked, chi);
  expect_sparse_reconstruction(from_blocks, target);

  // The same entries stored as one dense block give the identical synthesis.
  const auto from_dense =
      synthesize_sparse_site(make_site(packed, physical), chi);
  EXPECT_EQ(from_blocks.row_permutation, from_dense.row_permutation);
  EXPECT_EQ(from_blocks.column_permutation, from_dense.column_permutation);
  EXPECT_EQ(from_blocks.block_givens.layer_angles,
            from_dense.block_givens.layer_angles);
  EXPECT_EQ(from_blocks.block_givens.layer_shifted,
            from_dense.block_givens.layer_shifted);
  EXPECT_EQ(from_blocks.block_givens.phases, from_dense.block_givens.phases);
}

TEST(UnitarySynthesisTest, RejectsInvalidSparseInputs) {
  const MPSSite valid =
      site_from_isometry(random_orthogonal(4, 3).leftCols(2), 2);
  const MPSSite zero = site_from_isometry(Eigen::MatrixXd::Zero(4, 2), 2);
  EXPECT_THROW(synthesize_sparse_site(zero, 2), std::invalid_argument);
  Eigen::MatrixXd nonfinite = Eigen::MatrixXd::Identity(4, 2);
  nonfinite(3, 1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(synthesize_sparse_site(site_from_isometry(nonfinite, 2), 2),
               std::invalid_argument);
  EXPECT_THROW(synthesize_sparse_site(valid, 0), std::invalid_argument);
  EXPECT_THROW(
      synthesize_sparse_site(valid, std::numeric_limits<Eigen::Index>::max()),
      std::invalid_argument);
}

TEST(UnitarySynthesisTest, ReconstructsDenseSiteIsometry) {
  std::uint32_t seed = 50;
  for (const Eigen::Index physical : {2, 4}) {
    for (const Eigen::Index chi : {1, 2, 4}) {
      for (const Eigen::Index width : {Eigen::Index{1}, chi}) {
        const Eigen::MatrixXd target =
            random_orthogonal(physical * chi, seed++).leftCols(width);
        expect_dense_reconstruction(decompose_dense_target(target, chi), target,
                                    chi);
      }
    }
  }
}

TEST(UnitarySynthesisTest, ReconstructsLargeDenseSiteIsometry) {
  constexpr Eigen::Index chi = 32;
  std::uint32_t seed = 60;
  for (const Eigen::Index physical : {2, 4}) {
    for (const Eigen::Index width : {Eigen::Index{20}, chi}) {
      const Eigen::MatrixXd target =
          random_orthogonal(physical * chi, seed++).leftCols(width);
      expect_dense_reconstruction(decompose_dense_target(target, chi), target,
                                  chi);
    }

    // Left sector k couples only to right sector (k + p) mod 4, so every CSD
    // block has a highly degenerate spectrum.
    constexpr Eigen::Index sectors = 4;
    constexpr Eigen::Index sector_dim = chi / sectors;
    Eigen::MatrixXd blocked = Eigen::MatrixXd::Zero(physical * chi, chi);
    for (Eigen::Index k = 0; k < sectors; ++k) {
      const Eigen::MatrixXd block =
          random_orthogonal(physical * sector_dim, seed++).leftCols(sector_dim);
      for (Eigen::Index p = 0; p < physical; ++p) {
        blocked.block(p * chi + ((k + p) % sectors) * sector_dim,
                      k * sector_dim, sector_dim, sector_dim) =
            block.middleRows(p * sector_dim, sector_dim);
      }
    }
    expect_dense_reconstruction(decompose_dense_target(blocked, chi), blocked,
                                chi);
  }
}

TEST(UnitarySynthesisTest, ReconstructsStructuredDenseSiteIsometry) {
  // Product states and rank-deficient blocks exercise the zero and pi angles.
  Eigen::MatrixXd two_states = Eigen::MatrixXd::Zero(4, 2);
  two_states(2, 0) = 1.0;
  two_states(1, 1) = 1.0;
  expect_dense_reconstruction(decompose_dense_target(two_states, 2), two_states,
                              2);

  Eigen::MatrixXd four_states = Eigen::MatrixXd::Zero(8, 2);
  four_states(6, 0) = 1.0;
  four_states(3, 1) = 1.0 / std::sqrt(2.0);
  four_states(4, 1) = -1.0 / std::sqrt(2.0);
  expect_dense_reconstruction(decompose_dense_target(four_states, 2),
                              four_states, 2);

  // The two physical blocks each have rank two, so the CSD has angles 0 and
  // pi with doubly degenerate singular values.
  Eigen::MatrixXd rank_deficient = Eigen::MatrixXd::Zero(8, 4);
  rank_deficient.topLeftCorner(4, 2).setIdentity();
  rank_deficient.bottomRightCorner(4, 2).setIdentity();
  expect_dense_reconstruction(decompose_dense_target(rank_deficient, 4),
                              rank_deficient, 4);
}

void expect_near(const std::vector<double>& actual,
                 const std::vector<double>& expected) {
  ASSERT_EQ(actual.size(), expected.size());
  for (std::size_t index = 0; index < actual.size(); ++index) {
    EXPECT_NEAR(actual[index], expected[index], 1.0e-12);
  }
}

TEST(UnitarySynthesisTest, ChainsDenseMpsSiteRightFactors) {
  constexpr Eigen::Index chi = 4;
  std::uint32_t seed = 70;
  for (const Eigen::Index physical : {2, 4}) {
    SCOPED_TRACE(::testing::Message() << "physical " << physical);
    // Site i has left bond bonds[i] and right bond bonds[i + 1].
    std::vector<Eigen::Index> bonds{3, 2, 4, physical == 4 ? 1 : 2};
    if (physical == 2) {
      bonds.push_back(1);
    }
    std::vector<Eigen::MatrixXd> isometries;
    std::vector<MPSSite> sites;
    for (std::size_t index = 0; index + 1 < bonds.size(); ++index) {
      isometries.push_back(
          random_orthogonal(physical * bonds[index + 1], seed++)
              .leftCols(bonds[index]));
      sites.push_back(site_from_isometry(isometries.back(), physical));
    }
    std::vector<MPSSite> chain{make_site(
        Eigen::MatrixXd::Ones(physical, bonds.front()).eval(), physical)};
    chain.insert(chain.end(), sites.begin(), sites.end());
    const auto mps = make_container(chain);
    const auto results = std::get<std::vector<DenseSiteSynthesis>>(
        matrix_product_state_synthesis(mps, chi));
    ASSERT_EQ(results.size(), sites.size());

    for (std::size_t index = 0; index < sites.size(); ++index) {
      SCOPED_TRACE(::testing::Message() << "site " << index);
      // Each site absorbs the right factor of the following site.
      const Eigen::Index right = bonds[index + 1];
      Eigen::MatrixXd rotated = isometries[index];
      if (index + 1 < sites.size()) {
        for (Eigen::Index p = 0; p < physical; ++p) {
          rotated.middleRows(p * right, right) =
              results[index + 1].right_factor *
              isometries[index].middleRows(p * right, right);
        }
      }
      expect_dense_reconstruction(results[index],
                                  pad_isometry(rotated, physical, chi), chi);

      // Absorbing the following right factor changes only the terminal
      // blocks, so the rest matches the synthesis of the site on its own.
      const auto alone = synthesize_dense_site(sites[index], chi);
      ASSERT_EQ(results[index].rotation_angles.size(),
                alone.rotation_angles.size());
      for (std::size_t step = 0; step < alone.rotation_angles.size(); ++step) {
        expect_near(results[index].rotation_angles[step],
                    alone.rotation_angles[step]);
      }
      ASSERT_EQ(results[index].mixing_givens.size(),
                alone.mixing_givens.size());
      for (std::size_t step = 0; step < alone.mixing_givens.size(); ++step) {
        EXPECT_TRUE(
            reconstruct(results[index].mixing_givens[step])
                .isApprox(reconstruct(alone.mixing_givens[step]), 1.0e-12));
      }
      EXPECT_TRUE(
          results[index].right_factor.isApprox(alone.right_factor, 1.0e-12));
      if (index + 1 == sites.size()) {
        EXPECT_TRUE(reconstruct(results[index].block_givens)
                        .isApprox(reconstruct(alone.block_givens), 1.0e-12));
      }
    }
  }
}

TEST(UnitarySynthesisTest, DecomposesSparseMpsSite) {
  // Physical states 0 and 2 couple to right-bond states {0, 1}, physical
  // states 1 and 3 to right-bond state 2, and the register has four bond
  // states, so the packed isometry has zero rows to pad.
  constexpr Eigen::Index physical = 4;
  constexpr Eigen::Index right = 3;
  Eigen::MatrixXd isometry = Eigen::MatrixXd::Zero(physical * right, 3);
  const Eigen::MatrixXd paired = random_orthogonal(4, 81).leftCols(2);
  for (Eigen::Index row = 0; row < 2; ++row) {
    isometry.row(row) = Eigen::RowVector3d(paired(row, 0), paired(row, 1), 0);
    isometry.row(2 * right + row) =
        Eigen::RowVector3d(paired(2 + row, 0), paired(2 + row, 1), 0);
  }
  isometry(right + 2, 2) = 0.6;
  isometry(3 * right + 2, 2) = 0.8;

  const MPSSite site = site_from_isometry(isometry, physical);
  expect_sparse_reconstruction(synthesize_sparse_site(site, 4),
                               pad_isometry(isometry, physical, 4));
}

TEST(UnitarySynthesisTest, RejectsInvalidMpsSites) {
  const Eigen::MatrixXd isometry = random_orthogonal(8, 90).leftCols(2);
  const MPSSite site = site_from_isometry(isometry, 4);
  EXPECT_THROW(synthesize_dense_site(site, 1), std::invalid_argument);
  EXPECT_THROW(synthesize_sparse_site(site, 1), std::invalid_argument);
  // The right bond of the first site holds two states, while the following
  // site has three left-bond states.
  const MPSSite three_left =
      site_from_isometry(random_orthogonal(8, 91).leftCols(3), 4);
  EXPECT_THROW(
      make_container(
          {make_site(Eigen::MatrixXd::Ones(4, 2).eval(), 4), site, three_left,
           site_from_isometry(random_orthogonal(4, 92).leftCols(2), 4)}),
      std::invalid_argument);

  Eigen::MatrixXd non_isometric = Eigen::MatrixXd::Zero(8, 2);
  non_isometric(0, 0) = 1.0;
  non_isometric(1, 1) = 2.0;
  const MPSSite scaled = site_from_isometry(non_isometric, 4);
  EXPECT_THROW(synthesize_dense_site(scaled, 2), std::invalid_argument);
  EXPECT_THROW(synthesize_sparse_site(scaled, 2), std::invalid_argument);

  const Eigen::MatrixXcd complex_packed =
      Eigen::MatrixXcd::Identity(4, 2) * std::complex<double>(0.0, 1.0);
  const MPSSite complex_site = make_site(complex_packed, 4);
  EXPECT_THROW(synthesize_dense_site(complex_site, 2), std::invalid_argument);
  EXPECT_THROW(synthesize_sparse_site(complex_site, 2), std::invalid_argument);

  // Three physical states are neither the ('0', '1') nor the ('0', 'u', 'd',
  // '2') basis.
  std::vector<Configuration> three_states;
  for (const auto* state : {"0", "u", "d"}) {
    three_states.push_back(Configuration::from_spin_half_string(state));
  }
  const MPSSite three_state_site =
      make_site(Eigen::MatrixXd(Eigen::MatrixXd::Identity(3, 2)), 3,
                std::move(three_states));
  EXPECT_THROW(synthesize_dense_site(three_state_site, 2),
               std::invalid_argument);
  // Three left-bond states do not fit a two-state bond register.
  EXPECT_THROW(decompose_dense_target(Eigen::MatrixXd::Identity(8, 3), 2),
               std::invalid_argument);

  Eigen::MatrixXd nonfinite = isometry;
  nonfinite(5, 1) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_THROW(synthesize_dense_site(site_from_isometry(nonfinite, 4), 2),
               std::invalid_argument);

  EXPECT_THROW(dense_unitary_synthesis(site, 0), std::invalid_argument);
  EXPECT_THROW(
      dense_unitary_synthesis(site, std::numeric_limits<Eigen::Index>::max()),
      std::invalid_argument);
  EXPECT_THROW(
      dense_unitary_synthesis(site, 2, Eigen::MatrixXd::Identity(3, 3)),
      std::invalid_argument);
  EXPECT_THROW(dense_unitary_synthesis(site, 2, Eigen::MatrixXd::Zero(2, 2)),
               std::invalid_argument);
}

TEST(UnitarySynthesisTest, ReconstructsStructuredSparseSites) {
  // Every column holds a single signed entry, so every block is 1x1 and the
  // site needs sign flips only.
  Eigen::MatrixXd signed_permutation = Eigen::MatrixXd::Zero(8, 4);
  signed_permutation(5, 0) = 1.0;
  signed_permutation(0, 1) = -1.0;
  signed_permutation(6, 2) = 1.0;
  signed_permutation(3, 3) = -1.0;
  const auto permutation =
      expect_sparse_site_reconstruction(signed_permutation, 2, 4);
  EXPECT_TRUE(permutation.block_givens.layer_angles.empty());

  // A single column on two rows needs one rotation.
  const double angle = 0.37;
  Eigen::MatrixXd rotation = Eigen::MatrixXd::Zero(4, 1);
  rotation(1, 0) = std::cos(angle);
  rotation(2, 0) = std::sin(angle);
  const auto rotated = expect_sparse_site_reconstruction(rotation, 2, 2);
  EXPECT_EQ(rotated.block_givens.layer_angles.size(), 1u);
}

TEST(UnitarySynthesisTest, ReconstructsRandomSparseSites) {
  // One dense block of n rows per site, completed to an n x n orthogonal
  // block.
  constexpr Eigen::Index chi = 32;
  std::vector<Eigen::MatrixXd> targets;
  std::vector<MPSSite> sites;
  for (const Eigen::Index dim : {2, 3, 4, 5, 8, 16, 33, 64}) {
    for (std::uint32_t seed = 0; seed < 2; ++seed) {
      const Eigen::Index width = std::min(dim, chi);
      Eigen::MatrixXd target = Eigen::MatrixXd::Zero(2 * chi, width);
      target.topRows(dim) = random_orthogonal(dim, seed).leftCols(width);
      sites.push_back(site_from_isometry(target, 2));
      targets.push_back(std::move(target));
    }
  }
  for (std::size_t index = 0; index < sites.size(); ++index) {
    SCOPED_TRACE(::testing::Message() << "site " << index);
    expect_sparse_reconstruction(synthesize_sparse_site(sites[index], chi),
                                 targets[index]);
  }
}

TEST(UnitarySynthesisTest, ContainerDispatchSkipsInitialSite) {
  const auto first = make_site(Eigen::MatrixXd::Ones(4, 2).eval(), 4);
  const auto last =
      site_from_isometry(random_orthogonal(4, 122).leftCols(2), 4);
  const auto mps = make_container({first, last});
  const auto sparse = std::get<std::vector<SparseSiteSynthesis>>(
      matrix_product_state_synthesis(mps, 2, "block_sparse"));
  ASSERT_EQ(sparse.size(), 1u);
  expect_sparse_reconstruction(
      sparse.front(),
      pad_isometry(random_orthogonal(4, 122).leftCols(2), 4, 2));
  const auto dense = std::get<std::vector<DenseSiteSynthesis>>(
      matrix_product_state_synthesis(mps, 2, "dense"));
  ASSERT_EQ(dense.size(), 1u);
  expect_dense_reconstruction(
      dense.front(), pad_isometry(random_orthogonal(4, 122).leftCols(2), 4, 2),
      2);
  EXPECT_THROW(matrix_product_state_synthesis(mps, 1), std::invalid_argument);
  EXPECT_THROW(matrix_product_state_synthesis(mps, 2, "unknown"),
               std::invalid_argument);
  EXPECT_THROW(matrix_product_state_synthesis(mps, 2, "general"),
               std::invalid_argument);

  const auto single =
      make_container({make_site(Eigen::MatrixXd::Ones(4, 1).eval(), 4)});
  EXPECT_TRUE(std::get<std::vector<DenseSiteSynthesis>>(
                  matrix_product_state_synthesis(single, 2))
                  .empty());
  EXPECT_TRUE(std::get<std::vector<SparseSiteSynthesis>>(
                  matrix_product_state_synthesis(single, 2, "block_sparse"))
                  .empty());

  const auto invalid = make_container(
      {first, make_site(Eigen::MatrixXd::Zero(8, 2).eval(), 4), last});
  EXPECT_THROW(matrix_product_state_synthesis(invalid, 2, "dense"),
               std::invalid_argument);
  EXPECT_THROW(matrix_product_state_synthesis(invalid, 2, "block_sparse"),
               std::invalid_argument);
}
}  // namespace detail

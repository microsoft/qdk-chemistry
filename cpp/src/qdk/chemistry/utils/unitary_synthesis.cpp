// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <deque>
#include <numeric>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <stdexcept>
#include <utility>

namespace qdk::chemistry::utils::detail {
namespace {

constexpr double elimination_tolerance = 1.0e-15;
constexpr double orthogonality_tolerance = 1.0e-8;

// A rotation is stored by the lower index of its adjacent pair and its angle.
using Rotation = std::pair<Eigen::Index, double>;

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> complete_qr(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(matrix);
  Eigen::MatrixXd q = qr.householderQ() *
                      Eigen::MatrixXd::Identity(matrix.rows(), matrix.rows());
  Eigen::MatrixXd r = Eigen::MatrixXd::Zero(matrix.rows(), matrix.cols());
  r.topRows(matrix.cols()) = qr.matrixQR()
                                 .topRows(matrix.cols())
                                 .template triangularView<Eigen::Upper>();
  return {std::move(q), std::move(r)};
}

void validate_isometry(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                       const char* message) {
  const Eigen::MatrixXd gram = matrix.transpose() * matrix;
  const double residual =
      (gram - Eigen::MatrixXd::Identity(matrix.cols(), matrix.cols())).norm();
  if (residual > orthogonality_tolerance * static_cast<double>(matrix.cols())) {
    throw std::invalid_argument(message);
  }
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

}  // namespace

TwoBlockCsd decompose_2d(const Eigen::Ref<const Eigen::MatrixXd>& a,
                         const Eigen::Ref<const Eigen::MatrixXd>& b) {
  if (a.rows() == 0 || a.cols() == 0 || a.rows() < a.cols() ||
      a.rows() != b.rows() || a.cols() != b.cols()) {
    throw std::invalid_argument(
        "Two-block CSD requires equally sized nonempty matrices with rows >= "
        "columns.");
  }
  if (!a.allFinite() || !b.allFinite()) {
    throw std::invalid_argument(
        "Two-block CSD requires finite matrix entries.");
  }

  Eigen::MatrixXd stacked(2 * a.rows(), a.cols());
  stacked << a, b;
  validate_isometry(stacked,
                    "Two-block CSD requires an isometric vertical stack.");

  Eigen::JacobiSVD<Eigen::MatrixXd> upper_svd(
      a, Eigen::ComputeFullU | Eigen::ComputeFullV);
  TwoBlockCsd result;
  result.u_1 = upper_svd.matrixU();
  result.d_1 = upper_svd.singularValues();
  result.v = upper_svd.matrixV().transpose();

  const Eigen::MatrixXd lower_in_right_basis = b * upper_svd.matrixV();
  Eigen::JacobiSVD<Eigen::MatrixXd> lower_svd(
      lower_in_right_basis, Eigen::ComputeFullU | Eigen::ComputeFullV);
  result.u_2 = lower_svd.matrixU();
  result.u_2.leftCols(a.cols()) =
      lower_svd.matrixU().leftCols(a.cols()) * lower_svd.matrixV().transpose();
  const Eigen::MatrixXd d_2_matrix = lower_svd.matrixV() *
                                     lower_svd.singularValues().asDiagonal() *
                                     lower_svd.matrixV().transpose();
  result.d_2 = d_2_matrix.diagonal();
  return result;
}

SiteCsd decompose_site_csd(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                           Eigen::Index ancilla_dim) {
  if (ancilla_dim <= 0 || matrix.rows() != 4 * ancilla_dim ||
      matrix.cols() == 0 || matrix.cols() > ancilla_dim) {
    throw std::invalid_argument(
        "Site CSD requires a (4 * ancilla_dim) by width matrix with 0 < width "
        "<= ancilla_dim.");
  }
  if (!matrix.allFinite()) {
    throw std::invalid_argument("Site CSD requires finite matrix entries.");
  }
  validate_isometry(matrix, "Site CSD requires an isometric matrix.");

  const auto [b_full, r] = complete_qr(matrix.bottomRows(3 * ancilla_dim));
  const auto [c, s] =
      complete_qr(b_full.block(ancilla_dim, 0, 2 * ancilla_dim, ancilla_dim));

  const auto bottom =
      decompose_2d(c.topLeftCorner(ancilla_dim, ancilla_dim),
                   c.bottomLeftCorner(ancilla_dim, ancilla_dim));
  const auto middle = decompose_2d(
      b_full.topLeftCorner(ancilla_dim, ancilla_dim), s.topRows(ancilla_dim));
  const auto top =
      decompose_2d(matrix.topRows(ancilla_dim), r.topRows(ancilla_dim));

  SiteCsd result;
  result.u = {top.u_1, middle.u_1, bottom.u_1, bottom.u_2};
  result.d_prime = {top.d_2, middle.d_2, bottom.d_2};
  result.w_0 = middle.v * top.u_2;
  result.w_1 = bottom.v * middle.u_2;
  result.v = top.v;
  return result;
}

GivensDecomposition decompose_unitary_to_givens(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  if (matrix.rows() != matrix.cols() || matrix.rows() == 0) {
    throw std::invalid_argument(
        "Givens decomposition requires a nonempty square matrix.");
  }
  if (!matrix.allFinite()) {
    throw std::invalid_argument(
        "Givens decomposition requires finite matrix entries.");
  }

  const Eigen::Index dim = matrix.rows();
  const Eigen::MatrixXd gram = matrix.transpose() * matrix;
  const double residual = (gram - Eigen::MatrixXd::Identity(dim, dim)).norm();
  if (residual > orthogonality_tolerance * static_cast<double>(dim)) {
    throw std::invalid_argument(
        "Givens decomposition requires an orthogonal matrix.");
  }

  Eigen::MatrixXd work = matrix;
  if (dim == 1) {
    return {{}, {}, {static_cast<std::uint8_t>(work(0, 0) < 0.0)}};
  }

  const Eigen::Index num_layers = dim == 2 ? 1 : dim;
  std::vector<std::vector<Rotation>> upper_rotations(num_layers);
  std::vector<std::vector<Rotation>> lower_rotations(num_layers);

  // Clements elimination alternates right column rotations with left row
  // rotations so each diagonal sweep consists of disjoint adjacent pairs.
  for (Eigen::Index diagonal = 0; diagonal < dim - 1; ++diagonal) {
    if (diagonal % 2 == 0) {
      Eigen::Index slot = 0;
      for (Eigen::Index column = diagonal; column >= 0; --column, ++slot) {
        const Eigen::Index row = dim - 1 - slot;
        const double adjacent = work(row, column + 1);
        const double eliminated = work(row, column);
        if (std::abs(eliminated) < elimination_tolerance) {
          continue;
        }
        const double angle = std::atan2(eliminated, adjacent);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const Eigen::VectorXd first = work.col(column);
        const Eigen::VectorXd second = work.col(column + 1);
        work.col(column) = cosine * first - sine * second;
        work.col(column + 1) = sine * first + cosine * second;
        upper_rotations[slot].emplace_back(column, angle);
      }
    } else {
      Eigen::Index column = 0;
      for (Eigen::Index row = dim - diagonal - 1; row < dim; ++row, ++column) {
        const double adjacent = work(row - 1, column);
        const double eliminated = work(row, column);
        if (std::abs(eliminated) < elimination_tolerance) {
          continue;
        }
        const double angle = std::atan2(eliminated, adjacent);
        const double cosine = std::cos(angle);
        const double sine = std::sin(angle);
        const Eigen::RowVectorXd first = work.row(row - 1);
        const Eigen::RowVectorXd second = work.row(row);
        work.row(row - 1) = cosine * first + sine * second;
        work.row(row) = -sine * first + cosine * second;
        lower_rotations[column].emplace_back(row - 1, angle);
      }
    }
  }

  const Eigen::VectorXd diagonal = work.diagonal();
  GivensDecomposition result;
  result.phases.reserve(dim);
  for (Eigen::Index index = 0; index < dim; ++index) {
    result.phases.push_back(static_cast<std::uint8_t>(diagonal(index) < 0.0));
  }

  // Convert both elimination directions to the circuit convention in which
  // every layer multiplies from the right. Commuting a left rotation through
  // D reverses its angle exactly when the adjacent diagonal signs differ.
  const Eigen::Index even_slots = dim / 2;
  const Eigen::Index odd_slots = (dim - 1) / 2;
  for (Eigen::Index layer = 0; layer < num_layers; ++layer) {
    const bool shifted = layer % 2 == 1;
    const Eigen::Index num_slots = shifted ? odd_slots : even_slots;
    std::vector<double> angles(static_cast<std::size_t>(num_slots), 0.0);

    const auto store_rotation = [&](Eigen::Index pair, double angle) {
      if ((pair % 2 == 1) == shifted) {
        angles[static_cast<std::size_t>(pair / 2)] = angle;
      }
    };

    for (const auto& [pair, angle] : upper_rotations[layer]) {
      store_rotation(pair, angle);
    }

    const Eigen::Index lower_column = num_layers - 1 - layer;
    if (lower_column < static_cast<Eigen::Index>(lower_rotations.size())) {
      const auto& rotations = lower_rotations[lower_column];
      for (auto rotation = rotations.rbegin(); rotation != rotations.rend();
           ++rotation) {
        const auto [pair, angle] = *rotation;
        const double sign =
            diagonal(pair) * diagonal(pair + 1) > 0.0 ? 1.0 : -1.0;
        store_rotation(pair, sign * angle);
      }
    }

    if (std::any_of(angles.begin(), angles.end(), [](double angle) {
          return std::abs(angle) > elimination_tolerance;
        })) {
      result.layer_angles.push_back(std::move(angles));
      result.layer_shifted.push_back(static_cast<std::uint8_t>(shifted));
    }
  }

  return result;
}

GivensDecomposition decompose_block_diagonal_to_givens(
    const std::vector<Eigen::MatrixXd>& blocks) {
  if (blocks.empty()) {
    throw std::invalid_argument(
        "Block-diagonal Givens decomposition requires at least one block.");
  }

  struct BlockLayer {
    bool shifted;
    std::vector<Rotation> rotations;
  };

  Eigen::Index total_dim = 0;
  std::vector<Eigen::Index> starts;
  std::vector<std::deque<BlockLayer>> queues;
  std::vector<std::uint8_t> phases;
  starts.reserve(blocks.size());
  queues.reserve(blocks.size());

  std::size_t largest_block = 0;
  for (std::size_t block_index = 0; block_index < blocks.size();
       ++block_index) {
    const auto& block = blocks[block_index];
    starts.push_back(total_dim);
    if (block.rows() > blocks[largest_block].rows()) {
      largest_block = block_index;
    }

    const auto decomposition = decompose_unitary_to_givens(block);
    std::deque<BlockLayer> layers;
    for (std::size_t layer = 0; layer < decomposition.layer_angles.size();
         ++layer) {
      const bool shifted = decomposition.layer_shifted[layer] != 0;
      const Eigen::Index local_offset = shifted ? 1 : 0;
      std::vector<Rotation> rotations;
      for (std::size_t slot = 0;
           slot < decomposition.layer_angles[layer].size(); ++slot) {
        const double angle = decomposition.layer_angles[layer][slot];
        if (std::abs(angle) > elimination_tolerance) {
          rotations.emplace_back(
              total_dim + local_offset + 2 * static_cast<Eigen::Index>(slot),
              angle);
        }
      }
      if (!rotations.empty()) {
        layers.push_back({shifted, std::move(rotations)});
      }
    }
    queues.push_back(std::move(layers));
    phases.insert(phases.end(), decomposition.phases.begin(),
                  decomposition.phases.end());
    total_dim += block.rows();
  }

  bool global_shifted = false;
  if (!queues[largest_block].empty()) {
    global_shifted = queues[largest_block].front().shifted ^
                     (starts[largest_block] % 2 == 1);
  }

  GivensDecomposition result;
  result.phases = std::move(phases);
  const auto has_layers = [&]() {
    return std::any_of(queues.begin(), queues.end(),
                       [](const auto& queue) { return !queue.empty(); });
  };

  while (has_layers()) {
    const Eigen::Index num_slots =
        global_shifted ? (total_dim - 1) / 2 : total_dim / 2;
    std::vector<double> angles(static_cast<std::size_t>(num_slots), 0.0);

    for (std::size_t block_index = 0; block_index < queues.size();
         ++block_index) {
      auto& queue = queues[block_index];
      if (queue.empty()) {
        continue;
      }
      const bool aligned =
          ((starts[block_index] + (queue.front().shifted ? 1 : 0)) % 2 ==
           (global_shifted ? 1 : 0));
      if (!aligned) {
        continue;
      }
      for (const auto& [pair, angle] : queue.front().rotations) {
        angles[static_cast<std::size_t>(pair / 2)] = angle;
      }
      queue.pop_front();
    }

    if (std::any_of(angles.begin(), angles.end(), [](double angle) {
          return std::abs(angle) > elimination_tolerance;
        })) {
      result.layer_angles.push_back(std::move(angles));
      result.layer_shifted.push_back(static_cast<std::uint8_t>(global_shifted));
    }
    global_shifted = !global_shifted;
  }

  return result;
}

SparseSiteSynthesis decompose_sparse_site(
    const Eigen::Ref<const Eigen::MatrixXd>& target) {
  if (target.rows() == 0 || target.cols() == 0 ||
      target.cols() > target.rows()) {
    throw std::invalid_argument(
        "Sparse site decomposition requires a nonempty matrix with rows >= "
        "columns.");
  }
  if (!target.allFinite()) {
    throw std::invalid_argument(
        "Sparse site decomposition requires finite matrix entries.");
  }
  validate_isometry(target,
                    "Sparse site decomposition requires an isometric matrix.");

  const Eigen::Index dim = target.rows();
  std::vector<Eigen::MatrixXd> rectangles;
  std::vector<Eigen::Index> row_permutation;
  std::vector<bool> seen_rows(static_cast<std::size_t>(dim), false);
  std::vector<Eigen::Index> rectangle_rows;
  std::vector<Eigen::Index> rectangle_columns;

  const auto flush_rectangle = [&]() {
    if (rectangle_rows.empty()) {
      return;
    }
    Eigen::MatrixXd rectangle(rectangle_rows.size(), rectangle_columns.size());
    for (std::size_t column = 0; column < rectangle_columns.size(); ++column) {
      for (std::size_t row = 0; row < rectangle_rows.size(); ++row) {
        rectangle(static_cast<Eigen::Index>(row),
                  static_cast<Eigen::Index>(column)) =
            target(rectangle_rows[row], rectangle_columns[column]);
      }
    }
    rectangles.push_back(std::move(rectangle));
  };

  for (Eigen::Index column = 0; column < target.cols(); ++column) {
    std::vector<Eigen::Index> nonzero_rows;
    std::vector<Eigen::Index> new_rows;
    for (Eigen::Index row = 0; row < dim; ++row) {
      if (target(row, column) != 0.0) {
        nonzero_rows.push_back(row);
        if (!seen_rows[static_cast<std::size_t>(row)]) {
          new_rows.push_back(row);
        }
      }
    }

    if (!nonzero_rows.empty() && new_rows.size() == nonzero_rows.size()) {
      flush_rectangle();
      rectangle_rows = new_rows;
      rectangle_columns = {column};
      row_permutation.insert(row_permutation.end(), new_rows.begin(),
                             new_rows.end());
    } else {
      rectangle_columns.push_back(column);
      rectangle_rows.insert(rectangle_rows.end(), new_rows.begin(),
                            new_rows.end());
      row_permutation.insert(row_permutation.end(), new_rows.begin(),
                             new_rows.end());
    }
    for (const auto row : new_rows) {
      seen_rows[static_cast<std::size_t>(row)] = true;
    }
  }
  flush_rectangle();
  for (Eigen::Index row = 0; row < dim; ++row) {
    if (!seen_rows[static_cast<std::size_t>(row)]) {
      row_permutation.push_back(row);
    }
  }

  std::vector<Eigen::Index> column_mapping(static_cast<std::size_t>(dim));
  std::iota(column_mapping.begin(), column_mapping.end(), 0);
  Eigen::Index column_left = 0;
  Eigen::Index column_right = dim;
  Eigen::Index diagonal = 0;
  for (const auto& rectangle : rectangles) {
    const Eigen::Index width = rectangle.cols();
    const Eigen::Index difference = rectangle.rows() - width;
    if (difference > 0) {
      for (Eigen::Index index = width; index < column_right - column_left;
           ++index) {
        column_mapping[static_cast<std::size_t>(column_left + index)] +=
            difference;
      }
      for (Eigen::Index index = 0; index < difference; ++index) {
        column_mapping[static_cast<std::size_t>(
            column_right - difference + index)] = diagonal + width + index;
      }
      column_right -= difference;
    }
    column_left += width;
    diagonal += rectangle.rows();
  }
  const auto column_permutation = invert_permutation(column_mapping);

  std::vector<Eigen::MatrixXd> blocks;
  blocks.reserve(rectangles.size() + static_cast<std::size_t>(dim));
  Eigen::Index used_dim = 0;
  for (const auto& rectangle : rectangles) {
    if (rectangle.rows() == rectangle.cols()) {
      blocks.push_back(rectangle);
    } else {
      Eigen::JacobiSVD<Eigen::MatrixXd> svd(rectangle.transpose(),
                                            Eigen::ComputeFullV);
      Eigen::MatrixXd block(rectangle.rows(), rectangle.rows());
      block.leftCols(rectangle.cols()) = rectangle;
      block.rightCols(rectangle.rows() - rectangle.cols()) =
          svd.matrixV().rightCols(rectangle.rows() - rectangle.cols());
      blocks.push_back(std::move(block));
    }
    used_dim += rectangle.rows();
  }
  while (used_dim < dim) {
    blocks.push_back(Eigen::MatrixXd::Identity(1, 1));
    ++used_dim;
  }

  std::vector<std::size_t> sorted_indices(blocks.size());
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::stable_sort(sorted_indices.begin(), sorted_indices.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     return blocks[lhs].rows() > blocks[rhs].rows();
                   });
  std::vector<Eigen::Index> offsets(blocks.size());
  Eigen::Index offset = 0;
  for (std::size_t index = 0; index < blocks.size(); ++index) {
    offsets[index] = offset;
    offset += blocks[index].rows();
  }
  std::vector<Eigen::Index> ordering_mapping(static_cast<std::size_t>(dim));
  Eigen::Index new_offset = 0;
  std::vector<Eigen::MatrixXd> sorted_blocks;
  sorted_blocks.reserve(blocks.size());
  for (const auto index : sorted_indices) {
    for (Eigen::Index local = 0; local < blocks[index].rows(); ++local) {
      ordering_mapping[static_cast<std::size_t>(offsets[index] + local)] =
          new_offset + local;
    }
    new_offset += blocks[index].rows();
    sorted_blocks.push_back(std::move(blocks[index]));
  }
  const auto ordering_permutation = invert_permutation(ordering_mapping);

  std::vector<Eigen::Index> composed(static_cast<std::size_t>(dim));
  std::vector<Eigen::Index> final_rows(static_cast<std::size_t>(dim));
  for (Eigen::Index index = 0; index < dim; ++index) {
    composed[static_cast<std::size_t>(index)] =
        column_permutation[static_cast<std::size_t>(
            ordering_permutation[static_cast<std::size_t>(index)])];
    final_rows[static_cast<std::size_t>(index)] =
        row_permutation[static_cast<std::size_t>(
            ordering_permutation[static_cast<std::size_t>(index)])];
  }

  SparseSiteSynthesis result;
  result.column_permutation = invert_permutation(composed);
  result.row_permutation = std::move(final_rows);
  result.block_givens = decompose_block_diagonal_to_givens(sorted_blocks);
  return result;
}

}  // namespace qdk::chemistry::utils::detail
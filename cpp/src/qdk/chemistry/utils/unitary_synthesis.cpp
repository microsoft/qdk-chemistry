// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <iterator>
#include <lapack.hh>
#include <numeric>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qdk::chemistry::utils {
namespace {

constexpr double orthogonality_tolerance = 1.0e-8;

// Frobenius norm of M^T M - I.
double isometry_residual(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  return (matrix.transpose() * matrix -
          Eigen::MatrixXd::Identity(matrix.cols(), matrix.cols()))
      .norm();
}

// The tolerance scales with the column count. The comparison rejects NaN.
bool is_isometry(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  return isometry_residual(matrix) <=
         orthogonality_tolerance * static_cast<double>(matrix.cols());
}

struct FullSvd {
  Eigen::MatrixXd u;
  Eigen::VectorXd singular_values;
  Eigen::MatrixXd v;
};

// [A; B] = diag(U_1, U_2) [D_1; D_2] V, with D_1^2 + D_2^2 = I.
struct TwoBlockCsd {
  Eigen::MatrixXd u_1;
  Eigen::MatrixXd u_2;
  Eigen::VectorXd d_1;
  Eigen::VectorXd d_2;
  Eigen::MatrixXd v;
};

// Cosine-sine decompositions of a validated dense isometry, shared by the
// synthesis and the right-factor helper so both see the same V.
struct DenseIsometryFactors {
  Eigen::Index num_added_qubits = 0;
  // Nearest exact isometry to the input, which the factors decompose.
  Eigen::MatrixXd isometry;
  std::vector<TwoBlockCsd> csds;
};

constexpr double elimination_tolerance = 1.0e-15;

// Whether an angle is large enough to keep as a rotation.
bool is_rotation(double angle) {
  return std::abs(angle) > elimination_tolerance;
}

// singular value decomposition M = U · diag(s) · Vᵀ. with complete orthogonal U
// and V.
FullSvd singular_value_decomposition(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  const auto rows = static_cast<std::int64_t>(matrix.rows());
  const auto cols = static_cast<std::int64_t>(matrix.cols());
  Eigen::MatrixXd work = matrix;
  FullSvd result{Eigen::MatrixXd(rows, rows),
                 Eigen::VectorXd(std::min(rows, cols)),
                 Eigen::MatrixXd(cols, cols)};
  const auto info = lapack::gesvd(
      lapack::Job::AllVec, lapack::Job::AllVec, rows, cols, work.data(),
      std::max<std::int64_t>(1, rows), result.singular_values.data(),
      result.u.data(), std::max<std::int64_t>(1, rows), result.v.data(),
      std::max<std::int64_t>(1, cols));
  if (info != 0) {
    throw std::runtime_error("Unitary synthesis SVD did not converge.");
  }
  result.v.transposeInPlace();
  return result;
}

// QR decomposition M = Q · R with complete Q and upper-triangular R.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> qr_decomposition(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix, Eigen::Index num_columns) {
  Eigen::HouseholderQR<Eigen::MatrixXd> qr(matrix);
  Eigen::MatrixXd q =
      qr.householderQ() * Eigen::MatrixXd::Identity(matrix.rows(), num_columns);
  Eigen::MatrixXd r = Eigen::MatrixXd::Zero(num_columns, matrix.cols());
  r.topRows(matrix.cols()) = qr.matrixQR()
                                 .topRows(matrix.cols())
                                 .template triangularView<Eigen::Upper>();
  return {std::move(q), std::move(r)};
}

// Two-block CSD from an SVD of the top block and a polar decomposition of the
// bottom block.
TwoBlockCsd cosine_sine_decomposition(
    const Eigen::Ref<const Eigen::MatrixXd>& a,
    const Eigen::Ref<const Eigen::MatrixXd>& b) {
  const Eigen::Index width = a.cols();
  auto top = singular_value_decomposition(a);
  auto bottom = singular_value_decomposition(b * top.v);
  TwoBlockCsd csd;
  csd.u_1 = std::move(top.u);
  csd.d_1 = std::move(top.singular_values);
  csd.v = top.v.transpose();
  const Eigen::MatrixXd polar_factor =
      bottom.u.leftCols(width) * bottom.v.transpose();
  csd.d_2 =
      (bottom.v * bottom.singular_values.asDiagonal() * bottom.v.transpose())
          .diagonal();
  csd.u_2 = std::move(bottom.u);
  csd.u_2.leftCols(width) = polar_factor;
  return csd;
}

// Ry angles 2 atan2(d', d) of a cosine-sine pair, zero-padded to the
// ancilla_dim
std::vector<double> rotation_angles(const Eigen::VectorXd& d,
                                    const Eigen::VectorXd& d_prime,
                                    Eigen::Index ancilla_dim) {
  std::vector<double> angles(static_cast<std::size_t>(ancilla_dim), 0.0);
  for (Eigen::Index index = 0;
       index < std::min<Eigen::Index>(d_prime.size(), ancilla_dim); ++index) {
    angles[static_cast<std::size_t>(index)] =
        2.0 * std::atan2(d_prime(index), d(index));
  }
  return angles;
}

// Clements elimination into parallel Givens layers and a diagonal sign matrix.
GivensDecomposition decompose_orthogonal(const Eigen::MatrixXd& matrix) {
  if (!is_isometry(matrix)) {
    throw std::runtime_error(
        "Unitary synthesis produced a non-orthogonal factor.");
  }
  const Eigen::Index n = matrix.rows();
  const auto size = static_cast<std::size_t>(n);
  std::vector<std::vector<double>> angles(size);
  for (std::size_t layer = 0; layer < size; ++layer) {
    angles[layer].assign(layer % 2 == 1 ? (size - 1) / 2 : size / 2, 0.0);
  }
  // Left rotations get their sign once the final diagonal is known.
  struct LeftRotation {
    std::size_t layer;
    Eigen::Index pair;
    double angle;
  };
  std::vector<LeftRotation> left_rotations;
  Eigen::MatrixXd work = matrix;
  for (Eigen::Index diagonal = 0; diagonal + 1 < n; ++diagonal) {
    const bool from_right = diagonal % 2 == 0;
    for (Eigen::Index step = 0; step <= diagonal; ++step) {
      const Eigen::Index row =
          from_right ? n - 1 - step : n - 1 - diagonal + step;
      const Eigen::Index col = from_right ? diagonal - step : step;
      if (std::abs(work(row, col)) < elimination_tolerance) {
        continue;
      }
      const double angle = from_right
                               ? std::atan2(work(row, col), work(row, col + 1))
                               : std::atan2(work(row, col), work(row - 1, col));
      const Eigen::JacobiRotation<double> rotation(std::cos(angle),
                                                   std::sin(angle));
      if (from_right) {
        work.topRows(row + 1).applyOnTheRight(col, col + 1, rotation);
        angles[static_cast<std::size_t>(step)]
              [static_cast<std::size_t>(col / 2)] = angle;
      } else {
        work.rightCols(n - col).applyOnTheLeft(row - 1, row, rotation);
        left_rotations.push_back(
            {static_cast<std::size_t>(n - 1 - step), row - 1, angle});
      }
    }
  }

  GivensDecomposition result;
  result.phases.reserve(size);
  for (Eigen::Index index = 0; index < n; ++index) {
    result.phases.push_back(work(index, index) < 0.0 ? 1 : 0);
  }
  for (const auto& rotation : left_rotations) {
    const Eigen::Index pair = rotation.pair;
    const bool same_sign = work(pair, pair) * work(pair + 1, pair + 1) > 0.0;
    angles[rotation.layer][static_cast<std::size_t>(pair / 2)] =
        same_sign ? rotation.angle : -rotation.angle;
  }
  for (std::size_t layer = 0; layer < size; ++layer) {
    if (std::any_of(angles[layer].begin(), angles[layer].end(), is_rotation)) {
      result.layer_angles.push_back(std::move(angles[layer]));
      result.layer_shifted.push_back(layer % 2 == 1 ? 1 : 0);
    }
  }
  return result;
}

std::vector<GivensDecomposition> givens_decomposition(
    const std::vector<std::reference_wrapper<const Eigen::MatrixXd>>&
        matrices) {
  std::vector<GivensDecomposition> results(matrices.size());
  // An exception must not escape an OpenMP region, so each iteration keeps its
  // own and the first is rethrown after the loop.
  std::vector<std::exception_ptr> errors(matrices.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (std::size_t position = 0; position < matrices.size(); ++position) {
    try {
      results[position] = decompose_orthogonal(matrices[position].get());
    } catch (...) {
      errors[position] = std::current_exception();
    }
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  return results;
}

// Interleaves the layers of consecutive diagonal blocks into layers of the
// whole block-diagonal matrix.
GivensDecomposition merge_block_givens(
    const std::vector<GivensDecomposition>& decompositions) {
  struct BlockLayer {
    bool shifted;
    std::vector<std::pair<Eigen::Index, double>> rotations;
  };
  GivensDecomposition merged;
  if (decompositions.empty()) {
    return merged;
  }
  std::vector<std::deque<BlockLayer>> queues(decompositions.size());
  std::vector<Eigen::Index> offsets(decompositions.size());
  std::size_t largest = 0;
  std::size_t remaining = 0;
  Eigen::Index total = 0;
  for (std::size_t block = 0; block < decompositions.size(); ++block) {
    const auto& decomposition = decompositions[block];
    offsets[block] = total;
    if (decomposition.phases.size() > decompositions[largest].phases.size()) {
      largest = block;
    }
    merged.phases.insert(merged.phases.end(), decomposition.phases.begin(),
                         decomposition.phases.end());
    for (std::size_t layer = 0; layer < decomposition.layer_angles.size();
         ++layer) {
      BlockLayer entry{decomposition.layer_shifted[layer] != 0, {}};
      const auto& angles = decomposition.layer_angles[layer];
      for (std::size_t slot = 0; slot < angles.size(); ++slot) {
        if (is_rotation(angles[slot])) {
          entry.rotations.emplace_back(total + (entry.shifted ? 1 : 0) +
                                           2 * static_cast<Eigen::Index>(slot),
                                       angles[slot]);
        }
      }
      if (!entry.rotations.empty()) {
        queues[block].push_back(std::move(entry));
        ++remaining;
      }
    }
    total += static_cast<Eigen::Index>(decomposition.phases.size());
  }

  // Whether the front layer of a block acts on odd-starting global pairs.
  const auto front_is_odd = [&](std::size_t block) {
    return (offsets[block] + (queues[block].front().shifted ? 1 : 0)) % 2 == 1;
  };
  bool shifted = !queues[largest].empty() && front_is_odd(largest);
  while (remaining > 0) {
    std::vector<double> angles(
        static_cast<std::size_t>(shifted ? (total - 1) / 2 : total / 2), 0.0);
    for (std::size_t block = 0; block < queues.size(); ++block) {
      if (queues[block].empty() || front_is_odd(block) != shifted) {
        continue;
      }
      for (const auto& [pair, angle] : queues[block].front().rotations) {
        angles[static_cast<std::size_t>(pair / 2)] = angle;
      }
      queues[block].pop_front();
      --remaining;
    }
    if (std::any_of(angles.begin(), angles.end(), is_rotation)) {
      merged.layer_angles.push_back(std::move(angles));
      merged.layer_shifted.push_back(shifted ? 1 : 0);
    }
    shifted = !shifted;
  }
  return merged;
}

DenseIsometryFactors dense_isometry_factors(const Eigen::MatrixXd& isometry,
                                            Eigen::Index num_input_states) {
  const Eigen::Index dim = num_input_states;
  if (dim <= 0 || isometry.rows() % dim != 0) {
    throw std::invalid_argument(
        "Dense isometry synthesis requires a positive input dimension that "
        "divides the row count.");
  }
  const Eigen::Index ratio = isometry.rows() / dim;
  if (ratio != 1 && ratio != 2 && ratio != 4) {
    throw std::invalid_argument(
        "Dense isometry synthesis requires one, two, or four times as many "
        "rows as input states.");
  }
  if (isometry.cols() < 1 || isometry.cols() > dim) {
    throw std::invalid_argument(
        "Dense isometry synthesis requires between one column and one column "
        "per input state.");
  }
  if (!isometry.allFinite() || !is_isometry(isometry)) {
    throw std::invalid_argument(
        "Dense isometry synthesis requires a finite matrix with orthonormal "
        "columns.");
  }

  DenseIsometryFactors factors;
  factors.num_added_qubits = ratio == 1 ? 0 : (ratio == 2 ? 1 : 2);
  // The CSD keeps only the diagonal of D_2, which is exact only for an exact
  // isometry, so the input is replaced by its polar factor U V^T.
  const auto svd = singular_value_decomposition(isometry);
  factors.isometry = svd.u.leftCols(isometry.cols()) * svd.v.transpose();
  const auto& exact = factors.isometry;
  std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> blocks;
  if (ratio == 2) {
    blocks.emplace_back(exact.topRows(dim), exact.bottomRows(dim));
  } else if (ratio == 4) {
    auto [b, r] = qr_decomposition(exact.bottomRows(3 * dim), dim);
    auto [c, s] = qr_decomposition(b.bottomRows(2 * dim), dim);
    blocks.emplace_back(exact.topRows(dim), std::move(r));
    blocks.emplace_back(b.topRows(dim), std::move(s));
    blocks.emplace_back(c.topRows(dim), c.bottomRows(dim));
  }
  factors.csds.resize(blocks.size());
  // An exception must not escape an OpenMP region, so each iteration keeps its
  // own and the first is rethrown after the loop.
  std::vector<std::exception_ptr> errors(blocks.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (std::size_t step = 0; step < blocks.size(); ++step) {
    try {
      factors.csds[step] =
          cosine_sine_decomposition(blocks[step].first, blocks[step].second);
    } catch (...) {
      errors[step] = std::current_exception();
    }
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  return factors;
}

constexpr Eigen::Index unused_row = -1;

// Visit(column, row, value) for every stored nonzero entry.
template <typename Visit>
void for_each_nonzero_entry(const Eigen::SparseMatrix<double>& matrix,
                            const Visit& visit) {
  for (Eigen::Index column = 0; column < matrix.outerSize(); ++column) {
    for (Eigen::SparseMatrix<double>::InnerIterator entry(matrix, column);
         entry; ++entry) {
      if (entry.value() != 0.0) {
        visit(entry.col(), entry.row(), entry.value());
      }
    }
  }
}

// One column group with original column and row indices
struct ColumnGroup {
  std::vector<Eigen::Index> columns;
  std::vector<Eigen::Index> rows;
  Eigen::MatrixXd block;
  double residual = 0.0;
};

// All column groups
struct ColumnSupports {
  std::vector<ColumnGroup> groups;
  std::vector<std::size_t> column_group;
  std::vector<Eigen::Index> local_column;
  std::vector<Eigen::Index> first_column;
};

// Row permutation and starting row indices for each block
struct RowGathering {
  std::vector<Eigen::Index> permutation;
  std::vector<Eigen::Index> block_starts;
  std::size_t num_unused_rows = 0;
};

ColumnSupports discover_column_supports(
    const Eigen::SparseMatrix<double>& matrix) {
  const auto left = matrix.cols();
  const auto dim = matrix.rows();
  ColumnSupports supports;
  std::vector<Eigen::Index> parent(static_cast<std::size_t>(left));
  std::iota(parent.begin(), parent.end(), 0);
  const auto find = [&](Eigen::Index column) {
    while (parent[static_cast<std::size_t>(column)] != column) {
      auto& next = parent[static_cast<std::size_t>(column)];
      next = parent[static_cast<std::size_t>(next)];
      column = next;
    }
    return column;
  };
  supports.first_column.assign(static_cast<std::size_t>(dim), unused_row);
  for_each_nonzero_entry(matrix, [&](Eigen::Index column, Eigen::Index row,
                                     double) {
    auto& first = supports.first_column[static_cast<std::size_t>(row)];
    if (first == unused_row) {
      first = column;
      return;
    }
    const auto lhs = find(first);
    const auto rhs = find(column);
    if (lhs != rhs) {
      parent[static_cast<std::size_t>(std::max(lhs, rhs))] = std::min(lhs, rhs);
    }
    first = std::min(first, column);
  });

  // Roots are the smallest columns, so discovery order is independent of
  // storage.
  supports.column_group.resize(static_cast<std::size_t>(left));
  supports.local_column.resize(static_cast<std::size_t>(left));
  for (Eigen::Index column = 0; column < left; ++column) {
    const auto root = find(column);
    const auto group =
        root == column ? supports.groups.size()
                       : supports.column_group[static_cast<std::size_t>(root)];
    if (group == supports.groups.size()) {
      supports.groups.emplace_back();
    }
    supports.column_group[static_cast<std::size_t>(column)] = group;
    supports.local_column[static_cast<std::size_t>(column)] =
        static_cast<Eigen::Index>(supports.groups[group].columns.size());
    supports.groups[group].columns.push_back(column);
  }
  return supports;
}

RowGathering gather_rows(const Eigen::SparseMatrix<double>& matrix,
                         ColumnSupports& supports) {
  const auto left = matrix.cols();
  const auto dim = matrix.rows();
  auto& groups = supports.groups;
  std::vector<Eigen::Index> local_index(static_cast<std::size_t>(dim));
  std::vector<std::vector<Eigen::Index>> rows_by_column(
      static_cast<std::size_t>(left));
  for (Eigen::Index row = 0; row < dim; ++row) {
    const auto first = supports.first_column[static_cast<std::size_t>(row)];
    if (first != unused_row) {
      rows_by_column[static_cast<std::size_t>(first)].push_back(row);
    }
  }
  // First-reaching-column order keeps blocks close to triangular.
  for (Eigen::Index column = 0; column < left; ++column) {
    auto& group =
        groups[supports.column_group[static_cast<std::size_t>(column)]];
    for (const auto row : rows_by_column[static_cast<std::size_t>(column)]) {
      local_index[static_cast<std::size_t>(row)] =
          static_cast<Eigen::Index>(group.rows.size());
      group.rows.push_back(row);
    }
  }
  RowGathering gathering;
  gathering.block_starts.reserve(groups.size());
  gathering.permutation.reserve(static_cast<std::size_t>(dim));
  for (auto& group : groups) {
    if (group.rows.size() < group.columns.size()) {
      throw std::invalid_argument(
          "Block-sparse isometry synthesis requires an isometric matrix.");
    }
    const auto size = static_cast<Eigen::Index>(group.rows.size());
    group.block = Eigen::MatrixXd::Zero(size, size);
    gathering.block_starts.push_back(
        static_cast<Eigen::Index>(gathering.permutation.size()));
    gathering.permutation.insert(gathering.permutation.end(),
                                 group.rows.begin(), group.rows.end());
  }
  for_each_nonzero_entry(
      matrix, [&](Eigen::Index column, Eigen::Index row, double value) {
        groups[supports.column_group[static_cast<std::size_t>(column)]].block(
            local_index[static_cast<std::size_t>(row)],
            supports.local_column[static_cast<std::size_t>(column)]) += value;
      });
  for (Eigen::Index row = 0; row < dim; ++row) {
    if (supports.first_column[static_cast<std::size_t>(row)] == unused_row) {
      gathering.permutation.push_back(row);
      ++gathering.num_unused_rows;
    }
  }
  return gathering;
}

void complete_blocks(std::vector<ColumnGroup>& groups, Eigen::Index left) {
  const double tolerance = orthogonality_tolerance * static_cast<double>(left);
  // An exception must not escape an OpenMP region, so each iteration keeps its
  // own and the first is rethrown after the loop.
  std::vector<std::exception_ptr> errors(groups.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (std::size_t index = 0; index < groups.size(); ++index) {
    try {
      auto& group = groups[index];
      const Eigen::Index size = group.block.rows();
      const auto width = static_cast<Eigen::Index>(group.columns.size());
      auto rectangle = group.block.leftCols(width);
      const double block_residual = isometry_residual(rectangle);
      group.residual = block_residual * block_residual;
      if (!(block_residual <= tolerance)) {
        continue;
      }
      const bool project = !is_isometry(rectangle);
      if (size == width && !project) {
        continue;
      }
      const auto svd = singular_value_decomposition(rectangle.transpose());
      if (project) {
        rectangle = svd.v.leftCols(width) * svd.u.transpose();
      }
      group.block.rightCols(size - width) = svd.v.rightCols(size - width);
    } catch (...) {
      errors[index] = std::current_exception();
    }
  }
  for (const auto& error : errors) {
    if (error) {
      std::rethrow_exception(error);
    }
  }
  double residual = 0.0;
  for (const auto& group : groups) {
    residual += group.residual;
  }
  if (!(std::sqrt(residual) <= tolerance)) {
    throw std::invalid_argument(
        "Block-sparse isometry synthesis requires an isometric matrix.");
  }
}

// Maps input columns to block positions, inverse to a spy plot's column gather.
std::vector<Eigen::Index> gather_columns(
    const std::vector<ColumnGroup>& groups,
    const std::vector<Eigen::Index>& block_starts, Eigen::Index left,
    Eigen::Index dim) {
  std::vector<Eigen::Index> permutation(static_cast<std::size_t>(dim), 0);
  Eigen::Index free_column = left;
  Eigen::Index position = 0;
  for (std::size_t group_index = 0; group_index < groups.size();
       ++group_index) {
    const auto& group = groups[group_index];
    const auto width = static_cast<Eigen::Index>(group.columns.size());
    const auto size = static_cast<Eigen::Index>(group.rows.size());
    position = block_starts[group_index];
    for (Eigen::Index local = 0; local < width; ++local) {
      permutation[static_cast<std::size_t>(
          group.columns[static_cast<std::size_t>(local)])] = position + local;
    }
    for (Eigen::Index local = width; local < size; ++local) {
      permutation[static_cast<std::size_t>(free_column++)] = position + local;
    }
    position += size;
  }
  while (position < dim) {
    permutation[static_cast<std::size_t>(free_column++)] = position++;
  }
  return permutation;
}

// Absorb P_o into both mappings so largest-first sorting needs no extra
// circuit.
std::vector<std::size_t> sort_blocks(
    const std::vector<ColumnGroup>& groups,
    const std::vector<Eigen::Index>& block_starts,
    SparseIsometrySynthesis& result) {
  std::vector<std::size_t> order(groups.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t lhs, std::size_t rhs) {
                     return groups[lhs].rows.size() > groups[rhs].rows.size();
                   });
  std::vector<Eigen::Index> sorted_positions(result.row_permutation.size());
  std::iota(sorted_positions.begin(), sorted_positions.end(), 0);
  auto sorted_rows = result.row_permutation;
  Eigen::Index position = 0;
  for (const auto group_index : order) {
    const auto start = block_starts[group_index];
    const auto size =
        static_cast<Eigen::Index>(groups[group_index].rows.size());
    for (Eigen::Index local = 0; local < size; ++local) {
      sorted_positions[static_cast<std::size_t>(start + local)] =
          position + local;
      sorted_rows[static_cast<std::size_t>(position + local)] =
          result.row_permutation[static_cast<std::size_t>(start + local)];
    }
    position += size;
  }
  for (auto& column : result.column_permutation) {
    column = sorted_positions[static_cast<std::size_t>(column)];
  }
  result.row_permutation = std::move(sorted_rows);
  return order;
}

GivensDecomposition synthesize_completed_blocks(
    const std::vector<ColumnGroup>& groups,
    const std::vector<std::size_t>& order, std::size_t num_unused_rows) {
  std::vector<std::reference_wrapper<const Eigen::MatrixXd>> blocks;
  for (const auto group_index : order) {
    blocks.push_back(std::cref(groups[group_index].block));
  }
  auto decompositions = givens_decomposition(blocks);
  decompositions.resize(decompositions.size() + num_unused_rows,
                        GivensDecomposition{{}, {}, {0}});
  return merge_block_givens(decompositions);
}

}  // namespace

DenseIsometrySynthesis dense_isometry_synthesis(const Eigen::MatrixXd& isometry,
                                                Eigen::Index num_input_states,
                                                bool omit_right_factor) {
  auto factors = dense_isometry_factors(isometry, num_input_states);
  const Eigen::Index dim = num_input_states;
  const Eigen::Index width = isometry.cols();
  auto& csds = factors.csds;

  DenseIsometrySynthesis result;
  std::vector<Eigen::MatrixXd> unitaries;
  if (csds.empty()) {
    // Complete the columns with an orthonormal basis of their complement.
    Eigen::MatrixXd completed(dim, dim);
    completed.leftCols(width) = factors.isometry;
    if (width < dim) {
      const auto svd =
          singular_value_decomposition(factors.isometry.transpose());
      completed.rightCols(dim - width) = svd.v.rightCols(dim - width);
    }
    unitaries.push_back(std::move(completed));
    result.right_factor = Eigen::MatrixXd::Identity(width, width);
  } else {
    for (const auto& csd : csds) {
      result.rotation_angles.push_back(rotation_angles(csd.d_1, csd.d_2, dim));
    }
    if (csds.size() == 3) {
      unitaries.push_back(csds[1].v * csds[0].u_2);
      unitaries.push_back(csds[2].v * csds[1].u_2);
    }
    for (auto& csd : csds) {
      unitaries.push_back(std::move(csd.u_1));
    }
    unitaries.push_back(std::move(csds.back().u_2));
    result.right_factor = std::move(csds.front().v);
    if (!omit_right_factor) {
      Eigen::MatrixXd padded = Eigen::MatrixXd::Identity(dim, dim);
      padded.topLeftCorner(width, width) = result.right_factor;
      unitaries.push_back(std::move(padded));
    }
  }

  std::vector<std::reference_wrapper<const Eigen::MatrixXd>> matrices;
  for (const auto& unitary : unitaries) {
    matrices.push_back(std::cref(unitary));
  }
  auto givens = givens_decomposition(matrices);
  const auto num_mixing = csds.size() == 3 ? 2 : 0;
  const auto num_block =
      static_cast<std::ptrdiff_t>(csds.empty() ? 1 : csds.size() + 1);
  const auto block_begin = givens.begin() + num_mixing;
  const auto block_end = block_begin + num_block;
  result.mixing_givens.assign(std::make_move_iterator(givens.begin()),
                              std::make_move_iterator(block_begin));
  result.block_givens = merge_block_givens(
      std::vector<GivensDecomposition>(std::make_move_iterator(block_begin),
                                       std::make_move_iterator(block_end)));
  if (block_end != givens.end()) {
    result.right_factor_givens = std::move(*block_end);
  }
  return result;
}

Eigen::MatrixXd dense_isometry_right_factor(const Eigen::MatrixXd& isometry,
                                            Eigen::Index num_input_states) {
  auto factors = dense_isometry_factors(isometry, num_input_states);
  if (factors.csds.empty()) {
    return Eigen::MatrixXd::Identity(isometry.cols(), isometry.cols());
  }
  return std::move(factors.csds.front().v);
}

SparseIsometrySynthesis block_sparse_isometry_synthesis(
    const Eigen::SparseMatrix<double>& isometry) {
  const auto left = isometry.cols();
  const auto dim = isometry.rows();
  if (left < 1 || dim < left) {
    throw std::invalid_argument(
        "Block-sparse isometry synthesis requires at least one column and no "
        "more columns than rows.");
  }

  // Sum duplicate entries first so cancelling pairs couple no columns.
  std::vector<Eigen::Triplet<double>> entries;
  entries.reserve(static_cast<std::size_t>(isometry.nonZeros()));
  for_each_nonzero_entry(
      isometry, [&](Eigen::Index column, Eigen::Index row, double value) {
        entries.emplace_back(row, column, value);
      });
  Eigen::SparseMatrix<double> matrix(dim, left);
  matrix.setFromTriplets(entries.begin(), entries.end());

  // Step 1: [U' *].
  auto supports = discover_column_supports(matrix);

  // Step 2: [P_r^T U' R].
  auto rows = gather_rows(matrix, supports);
  complete_blocks(supports.groups, left);
  SparseIsometrySynthesis result;
  result.row_permutation = std::move(rows.permutation);

  // Step 3: Gather each block's target and completion columns together.
  result.column_permutation =
      gather_columns(supports.groups, rows.block_starts, left, dim);

  // Step 4: V = P_o^T B P_o.
  const auto order = sort_blocks(supports.groups, rows.block_starts, result);
  result.block_givens =
      synthesize_completed_blocks(supports.groups, order, rows.num_unused_rows);
  return result;
}

SparseIsometrySynthesis block_sparse_isometry_synthesis(
    const Eigen::MatrixXd& isometry) {
  return block_sparse_isometry_synthesis(
      Eigen::SparseMatrix<double>(isometry.sparseView()));
}

}  // namespace qdk::chemistry::utils

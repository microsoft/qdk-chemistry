// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include "unitary_synthesis_utils.hpp"

namespace qdk::chemistry::utils::detail {
namespace {

constexpr Eigen::Index unused_row = -1;

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

ColumnSupports discover_column_supports(const data::MPSSite& site,
                                        Eigen::Index ancilla_dim) {
  const auto& layout = site.tensor_layout();
  const auto left = layout.dimensions[0];
  const Eigen::Index dim = layout.dimensions[1] * ancilla_dim;
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
  for_each_nonzero_entry(
      site, ancilla_dim, [&](Eigen::Index column, Eigen::Index row, double) {
        auto& first = supports.first_column[static_cast<std::size_t>(row)];
        if (first == unused_row) {
          first = column;
          return;
        }
        const auto lhs = find(first);
        const auto rhs = find(column);
        if (lhs != rhs) {
          parent[static_cast<std::size_t>(std::max(lhs, rhs))] =
              std::min(lhs, rhs);
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

RowGathering gather_rows(const data::MPSSite& site, Eigen::Index ancilla_dim,
                         ColumnSupports& supports) {
  const auto& layout = site.tensor_layout();
  const auto left = layout.dimensions[0];
  const Eigen::Index dim = layout.dimensions[1] * ancilla_dim;
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
          "Sparse site decomposition requires an isometric matrix.");
    }
    const auto size = static_cast<Eigen::Index>(group.rows.size());
    group.block = Eigen::MatrixXd::Zero(size, size);
    gathering.block_starts.push_back(
        static_cast<Eigen::Index>(gathering.permutation.size()));
    gathering.permutation.insert(gathering.permutation.end(),
                                 group.rows.begin(), group.rows.end());
  }
  for_each_nonzero_entry(
      site, ancilla_dim,
      [&](Eigen::Index column, Eigen::Index row, double value) {
        groups[supports.column_group[static_cast<std::size_t>(column)]].block(
            local_index[static_cast<std::size_t>(row)],
            supports.local_column[static_cast<std::size_t>(column)]) = value;
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
  run_tasks(
      static_cast<std::ptrdiff_t>(groups.size()), [&](std::ptrdiff_t index) {
        auto& group = groups[static_cast<std::size_t>(index)];
        const Eigen::Index size = group.block.rows();
        const auto width = static_cast<Eigen::Index>(group.columns.size());
        const auto rectangle = group.block.leftCols(width);
        group.residual = (rectangle.transpose() * rectangle -
                          Eigen::MatrixXd::Identity(width, width))
                             .squaredNorm();
        if (size > width) {
          group.block.rightCols(size - width) =
              full_svd(rectangle.transpose()).v.rightCols(size - width);
        }
      });
  double residual = 0.0;
  for (const auto& group : groups) {
    residual += group.residual;
  }
  if (!(std::sqrt(residual) <=
        orthogonality_tolerance * static_cast<double>(left))) {
    throw std::invalid_argument(
        "Sparse site decomposition requires an isometric matrix.");
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
    SparseSiteSynthesis& result) {
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
  auto decompositions = decompose_unitaries_to_givens(blocks);
  decompositions.resize(decompositions.size() + num_unused_rows,
                        GivensDecomposition{{}, {}, {0}});
  return merge_block_givens(decompositions);
}

}  // namespace

SparseSiteSynthesis block_sparse_unitary_synthesis(const data::MPSSite& site,
                                                   Eigen::Index ancilla_dim) {
  validate_site(site, ancilla_dim);
  const auto& layout = site.tensor_layout();

  // Step 1: [U' *].
  auto supports = discover_column_supports(site, ancilla_dim);

  // Step 2: [P_r^T U' R].
  auto rows = gather_rows(site, ancilla_dim, supports);
  complete_blocks(supports.groups, layout.dimensions[0]);
  SparseSiteSynthesis result;
  result.row_permutation = std::move(rows.permutation);

  // Step 3: Gather each block's target and completion columns together.
  result.column_permutation =
      gather_columns(supports.groups, rows.block_starts, layout.dimensions[0],
                     layout.dimensions[1] * ancilla_dim);

  // Step 4: V = P_o^T B P_o.
  const auto order = sort_blocks(supports.groups, rows.block_starts, result);
  result.block_givens =
      synthesize_completed_blocks(supports.groups, order, rows.num_unused_rows);
  return result;
}

}  // namespace qdk::chemistry::utils::detail

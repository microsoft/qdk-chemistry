// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <blas.hh>
#include <cmath>
#include <lapack.hh>
#include <numeric>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <stdexcept>

namespace qdk::chemistry::utils {

namespace detail {

// Frobenius norm of M^T M - I.
double isometry_residual(const Eigen::MatrixXd& m) {
  Eigen::MatrixXd gram = Eigen::MatrixXd::Identity(m.cols(), m.cols());
  blas::syrk(blas::Layout::ColMajor, blas::Uplo::Upper, blas::Op::Trans,
             m.cols(), m.rows(), 1.0, m.data(), m.rows(), -1.0, gram.data(),
             m.cols());
  return lapack::lansy(lapack::Norm::Fro, lapack::Uplo::Upper, m.cols(),
                       gram.data(), m.cols());
}

// Orthogonal matrix whose leading columns are the polar factor of m.
Eigen::MatrixXd complete_isometry(Eigen::MatrixXd m) {
  const Eigen::Index rows = m.rows(), cols = m.cols();
  Eigen::MatrixXd u(rows, rows), vt(cols, cols);
  Eigen::VectorXd s(cols);
  if (lapack::gesvd(lapack::Job::AllVec, lapack::Job::AllVec, rows, cols,
                    m.data(), rows, s.data(), u.data(), rows, vt.data(),
                    cols) != 0) {
    throw std::runtime_error("Unitary synthesis SVD did not converge.");
  }
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, rows,
             cols, cols, 1.0, u.data(), rows, vt.data(), cols, 0.0, m.data(),
             rows);
  u.leftCols(cols) = m;
  return u;
}

// Clements layers (Optica 3, 1460) of an orthogonal matrix; angles must be 0.
void givens_layers(Eigen::MatrixXd u, Eigen::Ref<Eigen::MatrixXd> angles,
                   Eigen::Ref<Eigen::VectorXd> signs) {
  const Eigen::Index n = u.rows();
  for (Eigen::Index diagonal = 0; diagonal + 1 < n; ++diagonal) {
    for (Eigen::Index step = 0; step <= diagonal; ++step) {
      const bool right = diagonal % 2 == 0;
      const Eigen::Index row = right ? n - 1 - step : n - 1 - diagonal + step;
      const Eigen::Index col = right ? diagonal - step : step;
      if (std::abs(u(row, col)) < 1.0e-15) continue;
      if (right) {
        angles(step, col) = std::atan2(u(row, col), u(row, col + 1));
        blas::rot(row + 1, &u(0, col), 1, &u(0, col + 1), 1,
                  std::cos(angles(step, col)), -std::sin(angles(step, col)));
      } else {
        angles(n - 1 - step, row - 1) =
            std::atan2(u(row, col), u(row - 1, col));
        blas::rot(n - col, &u(row - 1, col), n, &u(row, col), n,
                  std::cos(angles(n - 1 - step, row - 1)),
                  std::sin(angles(n - 1 - step, row - 1)));
      }
    }
  }
  signs = u.diagonal().unaryExpr([](double x) { return x < 0.0 ? -1.0 : 1.0; });
  // Moving D past a left rotation multiplies its angle by its pair's signs.
  for (Eigen::Index diagonal = 1; diagonal + 1 < n; diagonal += 2) {
    for (Eigen::Index step = 0; step <= diagonal; ++step) {
      const Eigen::Index row = n - 1 - diagonal + step;
      angles(n - 1 - step, row - 1) *= signs(row - 1) * signs(row);
    }
  }
}

// Layers of a block-diagonal matrix; odd-offset blocks start a layer later.
void merge_layers(const std::vector<Eigen::MatrixXd>& blocks,
                  Eigen::MatrixXd& angles, Eigen::VectorXd& signs) {
  Eigen::Index n = 0, num_layers = 0;
  for (const auto& block : blocks) {
    num_layers = std::max(num_layers, block.rows() + n % 2);
    n += block.rows();
  }
  angles = Eigen::MatrixXd::Zero(num_layers, n - 1);
  signs.resize(n);
  for (Eigen::Index offset = 0; const auto& block : blocks) {
    givens_layers(
        block, angles.block(offset % 2, offset, block.rows(), block.rows() - 1),
        signs.segment(offset, block.rows()));
    offset += block.rows();
  }
  std::vector<Eigen::Index> kept;
  for (Eigen::Index j = 0; j < num_layers; ++j) {
    if ((angles.row(j).array() != 0.0).any()) kept.push_back(j);
  }
  angles = angles(kept, Eigen::all).eval();
}

}  // namespace detail

std::tuple<Eigen::MatrixXd, std::vector<Eigen::MatrixXd>,
           std::vector<Eigen::VectorXd>, Eigen::MatrixXd>
dense_isometry_synthesis(const Eigen::MatrixXd& isometry,
                         Eigen::Index num_input_states,
                         bool omit_right_factor) {
  const Eigen::Index rows = isometry.rows(), cols = isometry.cols();
  const Eigen::Index chi = num_input_states;
  const Eigen::Index ratio = chi > 0 && rows % chi == 0 ? rows / chi : 0;
  if ((ratio != 1 && ratio != 2 && ratio != 4) || cols < 1 || cols > chi) {
    throw std::invalid_argument(
        "Dense isometry synthesis requires one, two or four times as many rows "
        "as input states and between one and that many columns.");
  }
  if (!isometry.allFinite() ||
      !(detail::isometry_residual(isometry) <= 1.0e-8 * cols)) {
    throw std::invalid_argument(
        "Dense isometry synthesis requires a finite matrix with orthonormal "
        "columns.");
  }
  const Eigen::MatrixXd exact = detail::complete_isometry(isometry);
  // Each CSD splits a 2 chi-row input; for four blocks, QRs build the inputs.
  std::vector<Eigen::MatrixXd> inputs;
  if (ratio == 2) inputs = {exact.leftCols(cols)};
  if (ratio == 4) {
    Eigen::MatrixXd b(3 * chi, chi), r = Eigen::MatrixXd::Zero(chi, cols);
    Eigen::VectorXd tau(chi);
    b.leftCols(cols) = exact.block(chi, 0, 3 * chi, cols);
    lapack::geqrf(3 * chi, cols, b.data(), 3 * chi, tau.data());
    r.topRows(cols) =
        b.topLeftCorner(cols, cols).triangularView<Eigen::Upper>();
    lapack::orgqr(3 * chi, chi, cols, b.data(), 3 * chi, tau.data());
    Eigen::MatrixXd c = b.bottomRows(2 * chi);
    lapack::geqrf(2 * chi, chi, c.data(), 2 * chi, tau.data());
    const Eigen::MatrixXd t = c.topRows(chi).triangularView<Eigen::Upper>();
    lapack::orgqr(2 * chi, chi, chi, c.data(), 2 * chi, tau.data());
    inputs = {Eigen::MatrixXd(2 * chi, cols), Eigen::MatrixXd(2 * chi, chi), c};
    inputs[0] << exact.topLeftCorner(chi, cols), r;
    inputs[1] << b.topRows(chi), t;
  }
  const Eigen::Index num_csds = inputs.size();
  Eigen::MatrixXd ucr_angles = Eigen::MatrixXd::Zero(num_csds, chi);
  std::vector<Eigen::MatrixXd> u1(num_csds, Eigen::MatrixXd(chi, chi)),
      u2 = u1, vt(num_csds);
  for (Eigen::Index j = 0; j < num_csds; ++j) {
    const Eigen::Index width = inputs[j].cols();
    Eigen::VectorXd theta(width);
    vt[j].resize(width, width);
    if (lapack::orcsd2by1(lapack::Job::Vec, lapack::Job::Vec, lapack::Job::Vec,
                          2 * chi, chi, width, inputs[j].data(), 2 * chi,
                          inputs[j].data() + chi, 2 * chi, theta.data(),
                          u1[j].data(), chi, u2[j].data(), chi, vt[j].data(),
                          width) != 0) {
      throw std::runtime_error("Unitary synthesis CSD did not converge.");
    }
    // LAPACK pairs the sines with the last columns of U2; move them first.
    std::rotate(u2[j].data(), u2[j].data() + (chi - width) * chi,
                u2[j].data() + u2[j].size());
    ucr_angles.row(j).head(width) = 2.0 * theta;
  }
  std::vector<Eigen::MatrixXd> blocks{exact}, factors;
  Eigen::MatrixXd right_factor = Eigen::MatrixXd::Identity(cols, cols);
  if (num_csds > 0) {
    blocks = u1;
    blocks.push_back(u2.back());
    right_factor = vt[0];
  }
  for (Eigen::Index j = 0; j + 1 < num_csds; ++j) {
    Eigen::MatrixXd& w = factors.emplace_back(chi, chi);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               chi, chi, chi, 1.0, vt[j + 1].data(), chi, u2[j].data(), chi,
               0.0, w.data(), chi);
  }
  if (num_csds > 0 && !omit_right_factor) {
    factors.push_back(Eigen::MatrixXd::Identity(chi, chi));
    factors.back().topLeftCorner(cols, cols) = right_factor;
  }
  std::vector<Eigen::MatrixXd> layers;
  std::vector<Eigen::VectorXd> signs;
  detail::merge_layers(blocks, layers.emplace_back(), signs.emplace_back());
  for (const auto& factor : factors) {
    detail::merge_layers({factor}, layers.emplace_back(), signs.emplace_back());
  }
  return {ucr_angles, layers, signs, right_factor};
}

std::tuple<std::vector<Eigen::Index>, std::vector<Eigen::Index>,
           Eigen::MatrixXd, Eigen::VectorXd>
block_sparse_isometry_synthesis(const Eigen::SparseMatrix<double>& isometry) {
  const Eigen::Index rows = isometry.rows(), cols = isometry.cols();
  // Summing duplicates first keeps cancelling entries from joining blocks.
  std::vector<Eigen::Triplet<double>> entries;
  for (Eigen::Index c = 0; c < cols; ++c) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(isometry, c); it; ++it) {
      entries.emplace_back(it.row(), c, it.value());
    }
  }
  Eigen::SparseMatrix<double> matrix(rows, cols);
  matrix.setFromTriplets(entries.begin(), entries.end());
  matrix.prune([](Eigen::Index, Eigen::Index, double x) { return x != 0.0; });
  if (cols < 1 || rows < cols ||
      !Eigen::Map<const Eigen::VectorXd>(matrix.valuePtr(), matrix.nonZeros())
           .allFinite()) {
    throw std::invalid_argument(
        "Block-sparse isometry synthesis requires a finite matrix with at "
        "least one column and no more columns than rows.");
  }
  // Columns that share a nonzero row join one group, rooted at its smallest.
  std::vector<Eigen::Index> parent(cols), first(rows, -1), group(cols);
  std::iota(parent.begin(), parent.end(), 0);
  const auto find = [&](Eigen::Index c) {
    while (parent[c] != c) c = parent[c] = parent[parent[c]];
    return c;
  };
  for (Eigen::Index c = 0; c < cols; ++c) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, c); it; ++it) {
      Eigen::Index& f = first[it.row()];
      if (f < 0) f = c;
      const Eigen::Index a = find(f), b = find(c);
      parent[std::max(a, b)] = std::min(a, b);
    }
  }
  std::vector<std::vector<Eigen::Index>> group_cols, group_rows;
  for (Eigen::Index c = 0; c < cols; ++c) {
    if (find(c) == c) {
      group[c] = group_cols.size();
      group_cols.emplace_back();
      group_rows.emplace_back();
    }
    group[c] = group[find(c)];
    group_cols[group[c]].push_back(c);
  }
  // Rows of a group follow the first column that reaches them.
  std::vector<Eigen::Index> used, order(group_cols.size()), local(rows);
  for (Eigen::Index r = 0; r < rows; ++r) {
    if (first[r] >= 0) used.push_back(r);
  }
  std::stable_sort(used.begin(), used.end(),
                   [&](auto a, auto b) { return first[a] < first[b]; });
  for (const Eigen::Index r : used) group_rows[group[first[r]]].push_back(r);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](auto a, auto b) {
    return group_rows[a].size() > group_rows[b].size();
  });
  // Lay the blocks out largest first, each followed by its completion.
  std::vector<Eigen::Index> row_permutation, column_permutation(rows);
  std::vector<Eigen::MatrixXd> blocks;
  Eigen::Index free_column = cols;
  double residual = 0.0;
  for (const Eigen::Index g : order) {
    const Eigen::Index size = group_rows[g].size(),
                       width = group_cols[g].size(),
                       start = row_permutation.size();
    if (size < width) {
      throw std::invalid_argument(
          "Block-sparse isometry synthesis requires orthonormal columns.");
    }
    Eigen::MatrixXd block = Eigen::MatrixXd::Zero(size, width);
    for (Eigen::Index i = 0; i < size; ++i) local[group_rows[g][i]] = i;
    for (Eigen::Index j = 0; j < width; ++j) {
      const Eigen::Index c = group_cols[g][j];
      for (Eigen::SparseMatrix<double>::InnerIterator it(matrix, c); it; ++it) {
        block(local[it.row()], j) = it.value();
      }
      column_permutation[c] = start + j;
    }
    for (Eigen::Index j = width; j < size; ++j) {
      column_permutation[free_column++] = start + j;
    }
    residual += std::pow(detail::isometry_residual(block), 2);
    blocks.push_back(detail::complete_isometry(block));
    row_permutation.insert(row_permutation.end(), group_rows[g].begin(),
                           group_rows[g].end());
  }
  if (!(std::sqrt(residual) <= 1.0e-8 * cols)) {
    throw std::invalid_argument(
        "Block-sparse isometry synthesis requires orthonormal columns.");
  }
  for (Eigen::Index r = 0; r < rows; ++r) {
    if (first[r] < 0) {
      column_permutation[free_column++] = row_permutation.size();
      row_permutation.push_back(r);
      blocks.push_back(Eigen::MatrixXd::Identity(1, 1));
    }
  }
  Eigen::MatrixXd angles;
  Eigen::VectorXd signs;
  detail::merge_layers(blocks, angles, signs);
  return {row_permutation, column_permutation, angles, signs};
}

}  // namespace qdk::chemistry::utils

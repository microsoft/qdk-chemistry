// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <exception>
#include <functional>
#include <lapack.hh>
#include <stdexcept>
#include <utility>
#include <vector>

#include "unitary_synthesis_detail.hpp"

namespace qdk::chemistry::utils::detail {
namespace {

constexpr double elimination_tolerance = 1.0e-15;

// Whether an angle is large enough to keep as a rotation.
bool is_rotation(double angle) {
  return std::abs(angle) > elimination_tolerance;
}

}  // namespace

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

namespace {

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

}  // namespace

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

}  // namespace qdk::chemistry::utils::detail

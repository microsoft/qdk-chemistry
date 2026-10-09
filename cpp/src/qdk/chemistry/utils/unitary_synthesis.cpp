// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <functional>
#include <iterator>
#include <lapack.hh>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <variant>

#include "unitary_synthesis_detail.hpp"

namespace qdk::chemistry::utils::detail {
namespace {

// Cosine-sine factors [A; B] = diag(U_1, U_2) [D_1; D_2] V of a vertically
// stacked two-block isometry, with diagonal D_1^2 + D_2^2 = I.
struct TwoBlockCsd {
  Eigen::MatrixXd u_1;
  Eigen::MatrixXd u_2;
  Eigen::VectorXd d_1;
  Eigen::VectorXd d_2;
  Eigen::MatrixXd v;
};

void validate_isometry(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                       const char* message);

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> decompose_qr(
    const Eigen::Ref<const Eigen::MatrixXd>& matrix, Eigen::Index num_columns);

TwoBlockCsd decompose_csd(const Eigen::Ref<const Eigen::MatrixXd>& a,
                          const Eigen::Ref<const Eigen::MatrixXd>& b);

std::vector<double> rotation_angles(const Eigen::VectorXd& d,
                                    const Eigen::VectorXd& d_prime,
                                    Eigen::Index ancilla_dim);

}  // namespace

DenseSiteSynthesis dense_unitary_synthesis(
    const data::MPSSite& site, Eigen::Index ancilla_dim,
    const Eigen::MatrixXd& following_right_factor) {
  validate_site(site, ancilla_dim);
  const auto left = static_cast<Eigen::Index>(site.left_bond_dimension());
  const auto physical = static_cast<Eigen::Index>(site.physical_dimension());
  const auto right = static_cast<Eigen::Index>(site.right_bond_dimension());
  if (physical != 2 && physical != 4) {
    throw std::invalid_argument(
        "Dense site synthesis requires two or four physical states.");
  }
  if (following_right_factor.size() != 0) {
    if (following_right_factor.rows() != right ||
        following_right_factor.cols() != right ||
        !following_right_factor.allFinite()) {
      throw std::invalid_argument(
          "Dense site synthesis requires a finite successor factor matching "
          "the right bond.");
    }
    validate_isometry(
        following_right_factor,
        "Dense site synthesis requires an orthogonal successor factor.");
  } else if (following_right_factor.rows() != 0 ||
             following_right_factor.cols() != 0) {
    throw std::invalid_argument(
        "Dense site synthesis requires an empty or square successor factor.");
  }

  // Row p * ancilla_dim + b and column a hold M^p_{ab}.
  Eigen::MatrixXd isometry =
      Eigen::MatrixXd::Zero(physical * ancilla_dim, left);
  for_each_nonzero_entry(site, ancilla_dim,
                         [&](Eigen::Index column, Eigen::Index row,
                             double value) { isometry(row, column) = value; });
  validate_isometry(isometry,
                    "Dense site synthesis requires an isometric site.");

  std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> blocks;
  if (physical == 2) {
    blocks.emplace_back(isometry.topRows(ancilla_dim),
                        isometry.bottomRows(ancilla_dim));
  } else {
    // Three-step CSD peel: two QRs split the four physical blocks into three
    // independent two-block CSDs.
    auto [b, r] =
        decompose_qr(isometry.bottomRows(3 * ancilla_dim), ancilla_dim);
    auto [c, s] = decompose_qr(b.bottomRows(2 * ancilla_dim), ancilla_dim);
    blocks.emplace_back(isometry.topRows(ancilla_dim), std::move(r));
    blocks.emplace_back(b.topRows(ancilla_dim), std::move(s));
    blocks.emplace_back(c.topRows(ancilla_dim), c.bottomRows(ancilla_dim));
  }
  std::vector<TwoBlockCsd> csds(blocks.size());
  run_tasks(
      static_cast<std::ptrdiff_t>(blocks.size()), [&](std::ptrdiff_t index) {
        const auto step = static_cast<std::size_t>(index);
        csds[step] = decompose_csd(blocks[step].first, blocks[step].second);
      });

  DenseSiteSynthesis result;
  std::vector<Eigen::MatrixXd> unitaries;
  for (const auto& csd : csds) {
    result.rotation_angles.push_back(
        rotation_angles(csd.d_1, csd.d_2, ancilla_dim));
  }
  if (csds.size() == 3) {
    unitaries.push_back(csds[1].v * csds[0].u_2);
    unitaries.push_back(csds[2].v * csds[1].u_2);
  }
  for (auto& csd : csds) {
    unitaries.push_back(std::move(csd.u_1));
  }
  unitaries.push_back(std::move(csds.back().u_2));
  if (following_right_factor.size() != 0) {
    // Rotating the right bond changes only the terminal blocks, not the CSD
    // angles, mixing unitaries, or this site's right factor.
    const auto num_blocks = static_cast<std::ptrdiff_t>(csds.size() + 1);
    for (auto block = unitaries.end() - num_blocks; block != unitaries.end();
         ++block) {
      block->topRows(right) = following_right_factor * block->topRows(right);
    }
  }
  result.right_factor = std::move(csds.front().v);

  std::vector<std::reference_wrapper<const Eigen::MatrixXd>> matrices;
  for (const auto& unitary : unitaries) {
    matrices.push_back(std::cref(unitary));
  }
  auto givens = decompose_unitaries_to_givens(matrices);
  const auto terminal = givens.begin() + (physical == 4 ? 2 : 0);
  result.mixing_givens.assign(std::make_move_iterator(givens.begin()),
                              std::make_move_iterator(terminal));
  result.block_givens = merge_block_givens(
      std::vector<GivensDecomposition>(std::make_move_iterator(terminal),
                                       std::make_move_iterator(givens.end())));
  return result;
}

MPSSynthesis matrix_product_state_synthesis(
    const data::MPSContainer& mps, Eigen::Index ancilla_dim,
    std::string_view unitary_synthesis) {
  if (unitary_synthesis != "dense" && unitary_synthesis != "block_sparse") {
    throw std::invalid_argument(
        "MPS unitary synthesis must be 'dense' or 'block_sparse'.");
  }
  const auto& sites = mps.sites();
  validate_site(*sites.front(), ancilla_dim);
  const auto count = sites.size() - 1;
  if (unitary_synthesis == "dense") {
    std::vector<DenseSiteSynthesis> results(count);
    for (std::size_t index = count; index > 0; --index) {
      const auto& site = *sites[index];
      const Eigen::MatrixXd empty;
      const auto& following =
          index < count ? results[index].right_factor : empty;
      results[index - 1] =
          dense_unitary_synthesis(site, ancilla_dim, following);
    }
    return results;
  }
  std::vector<SparseSiteSynthesis> results(count);
  run_tasks(static_cast<std::ptrdiff_t>(count), [&](std::ptrdiff_t index) {
    const auto position = static_cast<std::size_t>(index);
    const auto& site = *sites[position + 1];
    results[position] = block_sparse_unitary_synthesis(site, ancilla_dim);
  });
  return results;
}

namespace {

constexpr double elimination_tolerance = 1.0e-15;

// A rotation is stored by the lower index of its adjacent pair and its angle.
using Rotation = std::pair<Eigen::Index, double>;

void validate_isometry(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                       const char* message) {
  const Eigen::MatrixXd gram = matrix.transpose() * matrix;
  const double residual =
      (gram - Eigen::MatrixXd::Identity(matrix.cols(), matrix.cols())).norm();
  if (!(residual <=
        orthogonality_tolerance * static_cast<double>(matrix.cols()))) {
    throw std::invalid_argument(message);
  }
}

}  // namespace

void validate_site(const data::MPSSite& site, Eigen::Index ancilla_dim) {
  if (site.is_complex()) {
    throw std::invalid_argument(
        "MPS site synthesis requires a real site tensor.");
  }
  const auto left = static_cast<Eigen::Index>(site.left_bond_dimension());
  const auto physical = static_cast<Eigen::Index>(site.physical_dimension());
  const auto right = static_cast<Eigen::Index>(site.right_bond_dimension());
  if (ancilla_dim <= 0 || left > ancilla_dim || right > ancilla_dim ||
      ancilla_dim > std::numeric_limits<Eigen::Index>::max() / physical) {
    throw std::invalid_argument(
        "MPS site synthesis requires both bond dimensions to be at most the "
        "ancilla dimension.");
  }
}

// Full SVD M = U diag(s) V^T with complete orthogonal U and V, computed by
// LAPACK's QR-iteration driver. Eigen 3.4's divide-and-conquer SVD can return
// non-finite factors or crash for spectra with many exact zeros, which the
// zero-padded site blocks routinely have.
FullSvd decompose_svd(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
  const auto rows = static_cast<std::int64_t>(matrix.rows());
  const auto cols = static_cast<std::int64_t>(matrix.cols());
  Eigen::MatrixXd work = matrix;
  FullSvd result{Eigen::MatrixXd(rows, rows),
                 Eigen::VectorXd(std::min(rows, cols)),
                 Eigen::MatrixXd(cols, cols)};
  Eigen::MatrixXd v_transpose(cols, cols);
  const auto info = lapack::gesvd(
      lapack::Job::AllVec, lapack::Job::AllVec, rows, cols, work.data(),
      std::max<std::int64_t>(1, rows), result.singular_values.data(),
      result.u.data(), std::max<std::int64_t>(1, rows), v_transpose.data(),
      std::max<std::int64_t>(1, cols));
  if (info != 0) {
    throw std::runtime_error("Unitary synthesis SVD did not converge.");
  }
  result.v = v_transpose.transpose();
  return result;
}

namespace {

std::pair<Eigen::MatrixXd, Eigen::MatrixXd> decompose_qr(
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

// Two-block CSD of equally sized m x k blocks, m >= k, whose vertical stack is
// an isometry. U_1 and U_2 are complete m x m orthogonal factors.
TwoBlockCsd decompose_csd(const Eigen::Ref<const Eigen::MatrixXd>& a,
                          const Eigen::Ref<const Eigen::MatrixXd>& b) {
  FullSvd upper = decompose_svd(a);
  // In the right basis of A the lower block has orthogonal columns with norms
  // sqrt(1 - d_1^2); its polar factor completes U_2 without reordering D_2.
  const FullSvd lower = decompose_svd(b * upper.v);
  TwoBlockCsd result;
  result.u_1 = std::move(upper.u);
  result.d_1 = std::move(upper.singular_values);
  result.v = upper.v.transpose();
  result.u_2 = lower.u;
  result.u_2.leftCols(a.cols()) =
      lower.u.leftCols(a.cols()) * lower.v.transpose();
  const Eigen::MatrixXd d_2_matrix =
      lower.v * lower.singular_values.asDiagonal() * lower.v.transpose();
  result.d_2 = d_2_matrix.diagonal();
  return result;
}

// Ry angles 2 atan2(d', d) of a cosine-sine pair, zero-padded to the bond
// register. atan2 keeps full precision where asin(d') is ill-conditioned.
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

}  // namespace

// Decomposes square orthogonal matrices into parallel Givens layers with the
// Clements double-sided elimination schedule. Alternating sweeps eliminate
// entries with right column rotations and left row rotations; the left
// rotations are then commuted through the final diagonal sign matrix, so
// applying the layers in increasing index order followed by the phase signs
// reconstructs each matrix as D L_{m-1} ... L_0 with m <= dim layers.
// Matrices are decomposed concurrently, largest first.
std::vector<GivensDecomposition> decompose_unitaries_to_givens(
    const std::vector<std::reference_wrapper<const Eigen::MatrixXd>>&
        matrices) {
  // Largest first so concurrent workers stay balanced.
  std::vector<std::size_t> order(matrices.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(
      order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        return matrices[lhs].get().rows() > matrices[rhs].get().rows();
      });
  std::vector<GivensDecomposition> results(matrices.size());
  run_tasks(static_cast<std::ptrdiff_t>(order.size()), [&](std::ptrdiff_t
                                                               index) {
    const auto position = order[static_cast<std::size_t>(index)];
    const Eigen::MatrixXd& matrix = matrices[position].get();
    auto& result = results[position];
    const Eigen::Index dim = matrix.rows();
    // Every matrix is a completion or product of factors of a validated
    // isometry, so it is orthogonal up to rounding unless a numerical routine
    // broke down. The negated comparison also rejects NaN.
    const double residual =
        (matrix.transpose() * matrix - Eigen::MatrixXd::Identity(dim, dim))
            .norm();
    if (!(residual <= orthogonality_tolerance * static_cast<double>(dim))) {
      throw std::runtime_error(
          "Unitary synthesis produced a non-orthogonal factor.");
    }

    Eigen::MatrixXd work = matrix;
    if (dim == 1) {
      result = {{}, {}, {static_cast<std::uint8_t>(work(0, 0) < 0.0)}};
      return;
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
          // Both columns are already zero below this row.
          work.topRows(row + 1).applyOnTheRight(
              column, column + 1,
              Eigen::JacobiRotation<double>(std::cos(angle), std::sin(angle)));
          upper_rotations[slot].emplace_back(column, angle);
        }
      } else {
        Eigen::Index column = 0;
        for (Eigen::Index row = dim - diagonal - 1; row < dim;
             ++row, ++column) {
          const double adjacent = work(row - 1, column);
          const double eliminated = work(row, column);
          if (std::abs(eliminated) < elimination_tolerance) {
            continue;
          }
          const double angle = std::atan2(eliminated, adjacent);
          // Both rows are already zero left of this column.
          work.rightCols(dim - column)
              .applyOnTheLeft(row - 1, row,
                              Eigen::JacobiRotation<double>(std::cos(angle),
                                                            std::sin(angle)));
          lower_rotations[column].emplace_back(row - 1, angle);
        }
      }
    }

    const Eigen::VectorXd diagonal = work.diagonal();
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
  });
  return results;
}

// Merges the Givens decompositions of nonempty diagonal blocks, which occupy
// consecutive diagonal ranges in input order, into global adjacent-pair layers
// of the block-diagonal matrix. Global layers alternate parity, starting
// aligned with the largest block, and each block contributes its next layer to
// the first global layer whose pair parity matches it.
GivensDecomposition merge_block_givens(
    const std::vector<GivensDecomposition>& decompositions) {
  struct BlockLayer {
    bool shifted;
    std::vector<Rotation> rotations;
  };

  Eigen::Index total_dim = 0;
  std::vector<Eigen::Index> starts;
  std::vector<std::deque<BlockLayer>> queues;
  std::vector<std::uint8_t> phases;
  starts.reserve(decompositions.size());
  queues.reserve(decompositions.size());

  std::size_t largest_block = 0;
  for (std::size_t block_index = 0; block_index < decompositions.size();
       ++block_index) {
    const auto& decomposition = decompositions[block_index];
    starts.push_back(total_dim);
    if (decomposition.phases.size() >
        decompositions[largest_block].phases.size()) {
      largest_block = block_index;
    }

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
    total_dim += static_cast<Eigen::Index>(decomposition.phases.size());
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

}  // namespace qdk::chemistry::utils::detail

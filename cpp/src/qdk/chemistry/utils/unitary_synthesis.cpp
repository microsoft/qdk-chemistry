// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <array>
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
#include <unordered_map>
#include <utility>
#include <variant>

namespace qdk::chemistry::utils::detail {
namespace {

constexpr double elimination_tolerance = 1.0e-15;
constexpr double orthogonality_tolerance = 1.0e-8;

// A rotation is stored by the lower index of its adjacent pair and its angle.
using Rotation = std::pair<Eigen::Index, double>;

// Leading num_columns columns of the complete Q factor and the matching rows of
// R, without forming the full square Q.
std::pair<Eigen::MatrixXd, Eigen::MatrixXd> leading_qr(
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

void validate_isometry(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
                       const char* message) {
  const Eigen::MatrixXd gram = matrix.transpose() * matrix;
  const double residual =
      (gram - Eigen::MatrixXd::Identity(matrix.cols(), matrix.cols())).norm();
  if (residual > orthogonality_tolerance * static_cast<double>(matrix.cols())) {
    throw std::invalid_argument(message);
  }
}

struct FullSvd {
  Eigen::MatrixXd u;
  Eigen::VectorXd singular_values;
  Eigen::MatrixXd v;
};

// Full SVD M = U diag(s) V^T with complete orthogonal U and V, computed by
// LAPACK's QR-iteration driver. Eigen 3.4's divide-and-conquer SVD can return
// non-finite factors or crash for spectra with many exact zeros, which the
// zero-padded site blocks routinely have.
FullSvd full_svd(const Eigen::Ref<const Eigen::MatrixXd>& matrix) {
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

// Runs task(0), ..., task(count - 1) concurrently when OpenMP is available and
// rethrows the first exception once every task has finished.
template <typename Task>
void run_tasks(std::ptrdiff_t count, const Task& task) {
  std::exception_ptr error;
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1)
#endif
  for (std::ptrdiff_t index = 0; index < count; ++index) {
    try {
      task(index);
    } catch (...) {
#ifdef _OPENMP
#pragma omp critical(qdk_unitary_synthesis_error)
#endif
      {
        if (!error) {
          error = std::current_exception();
        }
      }
    }
  }
  if (error) {
    std::rethrow_exception(error);
  }
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

void validate_site(const data::MPSSite& site, Eigen::Index ancilla_dim) {
  if (site.is_complex()) {
    throw std::invalid_argument(
        "MPS site synthesis requires a real site tensor.");
  }
  if (ancilla_dim <= 0 ||
      static_cast<Eigen::Index>(site.left_bond_dimension()) > ancilla_dim ||
      static_cast<Eigen::Index>(site.right_bond_dimension()) > ancilla_dim) {
    throw std::invalid_argument(
        "MPS site synthesis requires both bond dimensions to be at most the "
        "ancilla dimension.");
  }
}

// Cosine-sine factors [A; B] = diag(U_1, U_2) [D_1; D_2] V of a vertically
// stacked two-block isometry, with diagonal D_1^2 + D_2^2 = I.
struct TwoBlockCsd {
  Eigen::MatrixXd u_1;
  Eigen::MatrixXd u_2;
  Eigen::VectorXd d_1;
  Eigen::VectorXd d_2;
  Eigen::MatrixXd v;
};

// Two-block CSD of equally sized m x k blocks, m >= k, whose vertical stack is
// an isometry. U_1 and U_2 are complete m x m orthogonal factors.
TwoBlockCsd decompose_2d(const Eigen::Ref<const Eigen::MatrixXd>& a,
                         const Eigen::Ref<const Eigen::MatrixXd>& b) {
  FullSvd upper = full_svd(a);
  // In the right basis of A the lower block has orthogonal columns with norms
  // sqrt(1 - d_1^2); its polar factor completes U_2 without reordering D_2.
  const FullSvd lower = full_svd(b * upper.v);
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

}  // namespace

std::vector<DenseSiteSynthesis> decompose_dense_sites(
    const std::vector<data::MPSSite>& sites, Eigen::Index ancilla_dim) {
  for (std::size_t index = 0; index < sites.size(); ++index) {
    validate_site(sites[index], ancilla_dim);
    const auto physical = sites[index].physical_dimension();
    if (physical != 2 && physical != 4) {
      throw std::invalid_argument(
          "Dense site synthesis requires two or four physical states.");
    }
    if (index + 1 < sites.size() &&
        sites[index].right_bond_dimension() !=
            sites[index + 1].left_bond_dimension()) {
      throw std::invalid_argument(
          "Dense site synthesis requires each right bond dimension to match "
          "the left bond dimension of the following site.");
    }
  }

  // The right factor F of the following site rotates the right-bond rows of
  // every physical block. That leaves the QR R factors, the cosine-sine
  // angles, the mixing unitaries, and the site's own right factor unchanged and
  // left-multiplies each terminal diagonal block by F, so every site is
  // factored independently and F is applied to its terminal blocks afterwards.
  struct SiteFactors {
    // Blocks [A; B] of the two-block CSDs: one for two physical states, three
    // after the QR peel for four.
    std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> csd_blocks;
    std::vector<TwoBlockCsd> csds;
    // W_0 and W_1 for four physical states, then the terminal diagonal blocks.
    std::vector<Eigen::MatrixXd> unitaries;
  };
  std::vector<SiteFactors> factors(sites.size());
  const auto num_sites = static_cast<std::ptrdiff_t>(sites.size());
  run_tasks(num_sites, [&](std::ptrdiff_t index) {
    const auto& site = sites[static_cast<std::size_t>(index)];
    const auto physical = static_cast<Eigen::Index>(site.physical_dimension());
    const auto left = static_cast<Eigen::Index>(site.left_bond_dimension());
    const auto right = static_cast<Eigen::Index>(site.right_bond_dimension());

    // Row p * ancilla_dim + b and column a hold M^p_{ab}.
    const auto dense = site.to_dense();
    const auto& packed = std::get<Eigen::MatrixXd>(dense);
    Eigen::MatrixXd isometry =
        Eigen::MatrixXd::Zero(physical * ancilla_dim, left);
    for (Eigen::Index a = 0; a < left; ++a) {
      for (Eigen::Index p = 0; p < physical; ++p) {
        isometry.block(p * ancilla_dim, a, right, 1) =
            packed.row(a * physical + p).transpose();
      }
    }
    if (!isometry.allFinite()) {
      throw std::invalid_argument(
          "Dense site synthesis requires finite site entries.");
    }
    validate_isometry(isometry,
                      "Dense site synthesis requires an isometric site.");

    auto& blocks = factors[static_cast<std::size_t>(index)].csd_blocks;
    if (physical == 2) {
      blocks.emplace_back(isometry.topRows(ancilla_dim),
                          isometry.bottomRows(ancilla_dim));
      return;
    }
    // Three-step CSD peel: two QRs split the four physical blocks into three
    // independent two-block CSDs.
    auto [b, r] = leading_qr(isometry.bottomRows(3 * ancilla_dim), ancilla_dim);
    auto [c, s] = leading_qr(b.bottomRows(2 * ancilla_dim), ancilla_dim);
    blocks.emplace_back(isometry.topRows(ancilla_dim), std::move(r));
    blocks.emplace_back(b.topRows(ancilla_dim), std::move(s));
    blocks.emplace_back(c.topRows(ancilla_dim), c.bottomRows(ancilla_dim));
  });

  // The CSDs of all sites are independent.
  std::vector<std::pair<std::size_t, std::size_t>> csd_tasks;
  for (std::size_t site = 0; site < factors.size(); ++site) {
    factors[site].csds.resize(factors[site].csd_blocks.size());
    for (std::size_t step = 0; step < factors[site].csd_blocks.size(); ++step) {
      csd_tasks.emplace_back(site, step);
    }
  }
  run_tasks(
      static_cast<std::ptrdiff_t>(csd_tasks.size()), [&](std::ptrdiff_t index) {
        const auto [site, step] = csd_tasks[static_cast<std::size_t>(index)];
        auto& [upper, lower] = factors[site].csd_blocks[step];
        factors[site].csds[step] = decompose_2d(upper, lower);
        upper.resize(0, 0);
        lower.resize(0, 0);
      });

  std::vector<DenseSiteSynthesis> results(sites.size());
  run_tasks(num_sites, [&](std::ptrdiff_t index) {
    const auto site = static_cast<std::size_t>(index);
    auto& csds = factors[site].csds;
    auto& unitaries = factors[site].unitaries;
    auto& result = results[site];
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
    if (site + 1 < sites.size()) {
      // Only the right factor of the following site is read, which no task
      // modifies.
      const Eigen::MatrixXd& following = factors[site + 1].csds.front().v;
      const auto num_blocks = static_cast<std::ptrdiff_t>(csds.size() + 1);
      for (auto block = unitaries.end() - num_blocks; block != unitaries.end();
           ++block) {
        block->topRows(following.rows()) =
            following * block->topRows(following.rows());
      }
    }
    result.right_factor = csds.front().v;
  });

  // Decompose the mixing and terminal blocks of all sites together.
  std::vector<std::reference_wrapper<const Eigen::MatrixXd>> matrices;
  std::vector<std::ptrdiff_t> offsets;
  offsets.reserve(sites.size());
  for (const auto& site_factors : factors) {
    offsets.push_back(static_cast<std::ptrdiff_t>(matrices.size()));
    for (const auto& unitary : site_factors.unitaries) {
      matrices.push_back(std::cref(unitary));
    }
  }
  auto givens = decompose_unitaries_to_givens(matrices);
  run_tasks(num_sites, [&](std::ptrdiff_t index) {
    const auto site = static_cast<std::size_t>(index);
    const auto first = givens.begin() + offsets[site];
    const auto blocks = first + (factors[site].csds.size() == 3 ? 2 : 0);
    const auto last =
        first + static_cast<std::ptrdiff_t>(factors[site].unitaries.size());
    results[site].mixing_givens.assign(std::make_move_iterator(first),
                                       std::make_move_iterator(blocks));
    results[site].block_givens =
        merge_block_givens(std::vector<GivensDecomposition>(
            std::make_move_iterator(blocks), std::make_move_iterator(last)));
  });
  return results;
}

std::vector<SparseSiteSynthesis> decompose_sparse_sites(
    const std::vector<data::MPSSite>& sites, Eigen::Index ancilla_dim) {
  for (const auto& site : sites) {
    validate_site(site, ancilla_dim);
  }

  // Left-bond columns connected by shared nonzero rows, and the site isometry
  // restricted to them, completed to an orthogonal block.
  struct ColumnGroup {
    std::vector<Eigen::Index> columns;
    std::vector<Eigen::Index> rows;
    Eigen::MatrixXd block;
    double residual = 0.0;
  };
  struct SiteGroups {
    std::vector<ColumnGroup> groups;
    // Group indices by decreasing block size.
    std::vector<std::size_t> order;
    // Rows that no column reaches, each a 1x1 identity block.
    std::size_t num_unused_rows = 0;
  };
  std::vector<SiteGroups> site_groups(sites.size());
  std::vector<SparseSiteSynthesis> results(sites.size());
  const auto num_sites = static_cast<std::ptrdiff_t>(sites.size());
  run_tasks(num_sites, [&](std::ptrdiff_t index) {
    const auto& site = sites[static_cast<std::size_t>(index)];
    auto& groups = site_groups[static_cast<std::size_t>(index)].groups;
    auto& result = results[static_cast<std::size_t>(index)];
    const auto& tensor =
        std::get<data::SymmetryBlockedTensor<3, double>>(site.tensor());
    const auto left = static_cast<Eigen::Index>(site.left_bond_dimension());
    const Eigen::Index dim =
        static_cast<Eigen::Index>(site.physical_dimension()) * ancilla_dim;

    std::array<std::unordered_map<data::SymmetryLabel, Eigen::Index>, 3>
        offsets;
    for (std::size_t slot = 0; slot < offsets.size(); ++slot) {
      Eigen::Index offset = 0;
      for (const auto& label : site.sector_orders()[slot]) {
        offsets[slot].emplace(label, offset);
        offset += static_cast<Eigen::Index>(tensor.extents()[slot].at(label));
      }
    }

    // Calls visit(column, row, value) for every nonzero stored entry M^p_{ab},
    // which sits at column a and row p * ancilla_dim + b of the site isometry.
    // Blocks absent from the symmetry-blocked storage are never visited.
    const auto for_each_entry = [&](const auto& visit) {
      for (const auto& [labels, block] : tensor.blocks()) {
        const Eigen::Index left_offset = offsets[0].at(labels[0]);
        const Eigen::Index physical_offset = offsets[1].at(labels[1]);
        const Eigen::Index right_offset = offsets[2].at(labels[2]);
        const auto local_physical =
            static_cast<Eigen::Index>(tensor.extents()[1].at(labels[1]));
        for (Eigen::Index b = 0; b < block->cols(); ++b) {
          for (Eigen::Index packed = 0; packed < block->rows(); ++packed) {
            const double value = (*block)(packed, b);
            if (value != 0.0) {
              visit(left_offset + packed / local_physical,
                    (physical_offset + packed % local_physical) * ancilla_dim +
                        right_offset + b,
                    value);
            }
          }
        }
      }
    };

    // Union columns that share a nonzero row, so the resulting components have
    // disjoint row supports. Every root is the smallest column of its
    // component, and every row records the smallest column that reaches it.
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
    constexpr Eigen::Index unused_row = -1;
    std::vector<Eigen::Index> first_column(static_cast<std::size_t>(dim),
                                           unused_row);
    bool finite = true;
    for_each_entry([&](Eigen::Index column, Eigen::Index row, double value) {
      if (!std::isfinite(value)) {
        finite = false;
        return;
      }
      auto& first = first_column[static_cast<std::size_t>(row)];
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
    if (!finite) {
      throw std::invalid_argument(
          "Sparse site decomposition requires finite matrix entries.");
    }

    // Groups are ordered by their smallest column. Rows appear in the order in
    // which a scan over the columns first reaches them, which keeps each block
    // close to upper triangular and its Givens network short. Both orders
    // depend only on the nonzero pattern, not on how the entries are blocked.
    std::vector<std::size_t> column_group(static_cast<std::size_t>(left));
    std::vector<Eigen::Index> local_index(static_cast<std::size_t>(dim));
    std::vector<Eigen::Index> local_column(static_cast<std::size_t>(left));
    for (Eigen::Index column = 0; column < left; ++column) {
      const auto root = find(column);
      const auto group = root == column
                             ? groups.size()
                             : column_group[static_cast<std::size_t>(root)];
      if (group == groups.size()) {
        groups.emplace_back();
      }
      column_group[static_cast<std::size_t>(column)] = group;
      local_column[static_cast<std::size_t>(column)] =
          static_cast<Eigen::Index>(groups[group].columns.size());
      groups[group].columns.push_back(column);
    }
    std::vector<std::vector<Eigen::Index>> rows_by_column(
        static_cast<std::size_t>(left));
    for (Eigen::Index row = 0; row < dim; ++row) {
      const auto first = first_column[static_cast<std::size_t>(row)];
      if (first != unused_row) {
        rows_by_column[static_cast<std::size_t>(first)].push_back(row);
      }
    }
    for (Eigen::Index column = 0; column < left; ++column) {
      auto& group = groups[column_group[static_cast<std::size_t>(column)]];
      for (const auto row : rows_by_column[static_cast<std::size_t>(column)]) {
        local_index[static_cast<std::size_t>(row)] =
            static_cast<Eigen::Index>(group.rows.size());
        group.rows.push_back(row);
      }
    }
    for (auto& group : groups) {
      if (group.rows.size() < group.columns.size()) {
        throw std::invalid_argument(
            "Sparse site decomposition requires an isometric matrix.");
      }
      const auto size = static_cast<Eigen::Index>(group.rows.size());
      group.block = Eigen::MatrixXd::Zero(size, size);
    }
    for_each_entry([&](Eigen::Index column, Eigen::Index row, double value) {
      groups[column_group[static_cast<std::size_t>(column)]].block(
          local_index[static_cast<std::size_t>(row)],
          local_column[static_cast<std::size_t>(column)]) = value;
    });

    // Largest blocks first, then one 1x1 identity block per unused row. Target
    // column a maps to its group's slot; the columns a >= left, which no input
    // populates, fill the completion and identity slots.
    auto& order = site_groups[static_cast<std::size_t>(index)].order;
    order.resize(groups.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](std::size_t lhs, std::size_t rhs) {
                       return groups[lhs].rows.size() > groups[rhs].rows.size();
                     });
    result.row_permutation.reserve(static_cast<std::size_t>(dim));
    result.column_permutation.assign(static_cast<std::size_t>(dim), 0);
    Eigen::Index position = 0;
    Eigen::Index free_column = left;
    for (const auto group_index : order) {
      const auto& group = groups[group_index];
      const auto width = static_cast<Eigen::Index>(group.columns.size());
      const auto size = static_cast<Eigen::Index>(group.rows.size());
      for (Eigen::Index local = 0; local < width; ++local) {
        result.column_permutation[static_cast<std::size_t>(
            group.columns[static_cast<std::size_t>(local)])] = position + local;
      }
      for (Eigen::Index local = width; local < size; ++local) {
        result.column_permutation[static_cast<std::size_t>(free_column++)] =
            position + local;
      }
      result.row_permutation.insert(result.row_permutation.end(),
                                    group.rows.begin(), group.rows.end());
      position += size;
    }
    for (Eigen::Index row = 0; row < dim; ++row) {
      if (first_column[static_cast<std::size_t>(row)] == unused_row) {
        result.row_permutation.push_back(row);
        result.column_permutation[static_cast<std::size_t>(free_column++)] =
            position++;
        ++site_groups[static_cast<std::size_t>(index)].num_unused_rows;
      }
    }
  });

  // Each group restricted to its rows is an isometry. Complete it to an
  // orthogonal block with an orthonormal basis of its complement. The groups
  // of all sites are independent.
  std::vector<std::pair<std::size_t, std::size_t>> group_tasks;
  for (std::size_t site = 0; site < site_groups.size(); ++site) {
    for (std::size_t group = 0; group < site_groups[site].groups.size();
         ++group) {
      group_tasks.emplace_back(site, group);
    }
  }
  run_tasks(static_cast<std::ptrdiff_t>(group_tasks.size()),
            [&](std::ptrdiff_t index) {
              const auto [site, group_index] =
                  group_tasks[static_cast<std::size_t>(index)];
              auto& group = site_groups[site].groups[group_index];
              const Eigen::Index size = group.block.rows();
              const auto width =
                  static_cast<Eigen::Index>(group.columns.size());
              const auto rectangle = group.block.leftCols(width);
              group.residual = (rectangle.transpose() * rectangle -
                                Eigen::MatrixXd::Identity(width, width))
                                   .squaredNorm();
              if (size > width) {
                group.block.rightCols(size - width) =
                    full_svd(rectangle.transpose()).v.rightCols(size - width);
              }
            });
  for (std::size_t site = 0; site < sites.size(); ++site) {
    double residual = 0.0;
    for (const auto& group : site_groups[site].groups) {
      residual += group.residual;
    }
    if (std::sqrt(residual) >
        orthogonality_tolerance *
            static_cast<double>(sites[site].left_bond_dimension())) {
      throw std::invalid_argument(
          "Sparse site decomposition requires an isometric matrix.");
    }
  }

  // Decompose the blocks of all sites together.
  std::vector<std::reference_wrapper<const Eigen::MatrixXd>> blocks;
  std::vector<std::ptrdiff_t> offsets;
  offsets.reserve(sites.size());
  for (const auto& site : site_groups) {
    offsets.push_back(static_cast<std::ptrdiff_t>(blocks.size()));
    for (const auto group_index : site.order) {
      blocks.push_back(std::cref(site.groups[group_index].block));
    }
  }
  auto givens = decompose_unitaries_to_givens(blocks);
  run_tasks(num_sites, [&](std::ptrdiff_t index) {
    const auto site = static_cast<std::size_t>(index);
    const auto first = givens.begin() + offsets[site];
    std::vector<GivensDecomposition> decompositions(
        std::make_move_iterator(first),
        std::make_move_iterator(first + static_cast<std::ptrdiff_t>(
                                            site_groups[site].groups.size())));
    decompositions.resize(
        decompositions.size() + site_groups[site].num_unused_rows,
        GivensDecomposition{{}, {}, {0}});
    results[site].block_givens = merge_block_givens(decompositions);
  });
  return results;
}

}  // namespace qdk::chemistry::utils::detail

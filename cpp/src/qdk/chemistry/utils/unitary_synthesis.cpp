// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <exception>
#include <functional>
#include <iterator>
#include <stdexcept>
#include <utility>
#include <vector>

#include "unitary_synthesis_detail.hpp"

namespace qdk::chemistry::utils {
namespace detail {

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
  std::vector<std::pair<Eigen::MatrixXd, Eigen::MatrixXd>> blocks;
  if (ratio == 2) {
    blocks.emplace_back(isometry.topRows(dim), isometry.bottomRows(dim));
  } else if (ratio == 4) {
    auto [b, r] = qr_decomposition(isometry.bottomRows(3 * dim), dim);
    auto [c, s] = qr_decomposition(b.bottomRows(2 * dim), dim);
    blocks.emplace_back(isometry.topRows(dim), std::move(r));
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

}  // namespace detail

DenseIsometrySynthesis dense_isometry_synthesis(const Eigen::MatrixXd& isometry,
                                                Eigen::Index num_input_states,
                                                bool omit_right_factor) {
  auto factors = detail::dense_isometry_factors(isometry, num_input_states);
  const Eigen::Index dim = num_input_states;
  const Eigen::Index width = isometry.cols();
  auto& csds = factors.csds;

  DenseIsometrySynthesis result;
  std::vector<Eigen::MatrixXd> unitaries;
  if (csds.empty()) {
    // Complete the columns with an orthonormal basis of their complement.
    Eigen::MatrixXd completed(dim, dim);
    completed.leftCols(width) = isometry;
    if (width < dim) {
      const auto svd =
          detail::singular_value_decomposition(isometry.transpose());
      completed.rightCols(dim - width) = svd.v.rightCols(dim - width);
    }
    unitaries.push_back(std::move(completed));
    result.right_factor = Eigen::MatrixXd::Identity(width, width);
  } else {
    for (const auto& csd : csds) {
      result.rotation_angles.push_back(
          detail::rotation_angles(csd.d_1, csd.d_2, dim));
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
  auto givens = detail::givens_decomposition(matrices);
  const auto num_mixing = csds.size() == 3 ? 2 : 0;
  const auto num_block =
      static_cast<std::ptrdiff_t>(csds.empty() ? 1 : csds.size() + 1);
  const auto block_begin = givens.begin() + num_mixing;
  const auto block_end = block_begin + num_block;
  result.mixing_givens.assign(std::make_move_iterator(givens.begin()),
                              std::make_move_iterator(block_begin));
  result.block_givens = detail::merge_block_givens(
      std::vector<GivensDecomposition>(std::make_move_iterator(block_begin),
                                       std::make_move_iterator(block_end)));
  if (block_end != givens.end()) {
    result.right_factor_givens = std::move(*block_end);
  }
  return result;
}

Eigen::MatrixXd dense_isometry_right_factor(const Eigen::MatrixXd& isometry,
                                            Eigen::Index num_input_states) {
  auto factors = detail::dense_isometry_factors(isometry, num_input_states);
  if (factors.csds.empty()) {
    return Eigen::MatrixXd::Identity(isometry.cols(), isometry.cols());
  }
  return std::move(factors.csds.front().v);
}

}  // namespace qdk::chemistry::utils

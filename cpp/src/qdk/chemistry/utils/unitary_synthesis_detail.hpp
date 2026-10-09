// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <exception>
#include <functional>
#include <qdk/chemistry/utils/unitary_synthesis.hpp>
#include <vector>

namespace qdk::chemistry::utils::detail {

inline constexpr double orthogonality_tolerance = 1.0e-8;

struct FullSvd {
  Eigen::MatrixXd u;
  Eigen::VectorXd singular_values;
  Eigen::MatrixXd v;
};

FullSvd decompose_svd(const Eigen::Ref<const Eigen::MatrixXd>& matrix);

std::vector<GivensDecomposition> decompose_unitaries_to_givens(
    const std::vector<std::reference_wrapper<const Eigen::MatrixXd>>& matrices);

GivensDecomposition merge_block_givens(
    const std::vector<GivensDecomposition>& decompositions);

void validate_site(const data::MPSSite& site, Eigen::Index ancilla_dim);

// Visit(column, row, value) reads M^p_{ab} at column a, row p * ancilla_dim +
// b. Missing blocks and zero entries are not materialized.
template <typename Visit>
void for_each_nonzero_entry(const data::MPSSite& site, Eigen::Index ancilla_dim,
                            const Visit& visit) {
  const auto& tensor =
      std::get<data::SymmetryBlockedTensor<3, double>>(site.tensor());
  const auto& offsets = site.sector_offsets();
  for (const auto& [labels, block] : tensor.blocks()) {
    const auto left_offset = offsets[0].at(labels[0]);
    const auto physical_offset = offsets[1].at(labels[1]);
    const auto right_offset = offsets[2].at(labels[2]);
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

}  // namespace qdk::chemistry::utils::detail

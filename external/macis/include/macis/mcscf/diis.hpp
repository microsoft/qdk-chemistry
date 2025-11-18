/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <blas.hh>
#include <lapack.hh>
#include <memory>
#include <tuple>

namespace macis {

/**
 * @brief Direct Inversion in the Iterative Subspace (DIIS) acceleration class
 *
 * This class implements the DIIS method for accelerating convergence of
 * iterative procedures by extrapolating from previous iterations using
 * error vectors.
 *
 * @tparam VecType Type of the vectors to be extrapolated (e.g.,
 * std::vector<double>)
 */
template <typename VecType>
class DIIS {
 private:
  /** @brief Maximum number of vector pairs to keep in the DIIS space */
  size_t ndiis_max_;

  /** @brief Storage for (solution vector, error vector) pairs */
  std::vector<std::pair<VecType, VecType>> diis_pairs_;

  /** @brief Logger instance for debugging and information output */
  std::shared_ptr<spdlog::logger> logger_;

 public:
  /**
   * @brief Constructor for DIIS class
   *
   * @param nmax Maximum number of vector pairs to store in DIIS subspace
   * (default: 10)
   */
  DIIS(size_t nmax = 10) : ndiis_max_(nmax) {
    logger_ = spdlog::get("diis");
    if (!logger_) {
      logger_ = spdlog::stdout_color_mt("diis");
    }
  };

  /**
   * @brief Add a vector pair to the DIIS subspace
   *
   * Adds a (solution vector, error vector) pair to the DIIS subspace.
   * If the maximum number of pairs is reached, the oldest pair is removed.
   *
   * @param v Solution vector to add
   * @param e Error vector corresponding to the solution vector
   */
  void add_vector(const VecType& v, const VecType& e) {
    logger_->debug(
        "Appending new vector v_size = {}, e_size = {} e_nrm = {:.5e}",
        v.size(), e.size(), blas::nrm2(e.size(), e.data(), 1));
    if (diis_pairs_.size() < ndiis_max_) {
      diis_pairs_.emplace_back(v, e);
    } else {
      logger_->debug(
          "DIIS has reached its limit for kept vectors, rejecting oldest pair");
      std::copy(diis_pairs_.begin() + 1, diis_pairs_.end(),
                diis_pairs_.begin());
      diis_pairs_[ndiis_max_ - 1] = std::make_pair(v, e);
    }
  }

  /**
   * @brief Perform DIIS extrapolation to obtain an improved solution vector
   *
   * Constructs the DIIS matrix from error vector overlaps, solves the DIIS
   * linear system to obtain optimal coefficients, and returns the extrapolated
   * solution vector as a linear combination of stored solution vectors.
   *
   * @return Extrapolated solution vector
   */
  VecType extrapolate() {
    const size_t ndiis = diis_pairs_.size();
    std::vector<double> B((ndiis + 1) * (ndiis + 1));

    logger_->info("DIIS extrapolation: ndiis = {}", ndiis);

    // Form the DIIS matrix
    for (size_t i = 0; i < ndiis; ++i)
      for (size_t j = i; j < ndiis; ++j) {
        const auto& e_i = diis_pairs_[i].second;
        const auto& e_j = diis_pairs_[j].second;

        const auto val = blas::dot(e_i.size(), e_i.data(), 1, e_j.data(), 1);
        B[i + j * (ndiis + 1)] = val;
        if (i != j) {
          B[j + i * (ndiis + 1)] = val;
        }
      }

    // Pad DIIS matrix
    for (size_t i = 0; i < ndiis; ++i) {
      B[i + ndiis * (ndiis + 1)] = 1.0;
      B[ndiis + i * (ndiis + 1)] = 1.0;
    }

    // Solve Linear System
    std::vector<double> C(ndiis + 1, 0.0);
    C[ndiis] = 1.0;
    std::vector<double> S(ndiis + 1);
    int64_t rank;
    lapack::gelss(ndiis + 1, ndiis + 1, 1, B.data(), ndiis + 1, C.data(),
                  ndiis + 1, S.data(), -1.0, &rank);

    logger_->info("diis_rank = {} / {}", rank, ndiis + 1);

    // Extrapolate
    VecType new_x(diis_pairs_[0].first.size());
    std::fill(new_x.begin(), new_x.end(), 0);
    for (auto i = 0; i < ndiis; ++i) {
      const size_t n = new_x.size();
      blas::axpy(n, C[i], diis_pairs_[i].first.data(), 1, new_x.data(), 1);
    }

    return new_x;
  }
};

}  // namespace macis

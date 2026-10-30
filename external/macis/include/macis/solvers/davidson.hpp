/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <spdlog/sinks/null_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <lobpcgxx/lobpcg.hpp>
#include <macis/util/mpi.hpp>
#include <memory>
#include <sparsexx/matrix_types/csr_matrix.hpp>

#ifdef MACIS_ENABLE_MPI
#include <sparsexx/spblas/pspmbv.hpp>
#endif /* MACIS_ENABLE_MPI */

#include <sparsexx/spblas/spmbv.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(__linux__)
#include <sys/mman.h>

#include <cstdlib>
#endif

namespace macis {

/**
 * @brief Operator wrapper for sparse matrix-vector multiplication operations.
 *
 * This class provides a unified interface for performing matrix-vector
 * multiplication with sparse matrices, supporting both sequential and
 * parallel (MPI) operations.
 *
 * @tparam SpMatType Type of the sparse matrix
 */
template <typename SpMatType>
class SparseMatrixOperator {
  using index_type = typename SpMatType::index_type;

  const SpMatType& m_matrix_;
#ifdef MACIS_ENABLE_MPI
  sparsexx::spblas::spmv_info<index_type> m_spmv_info_;
#endif /* MACIS_ENABLE_MPI */

 public:
  /**
   * @brief Constructor for SparseMatrixOperator.
   *
   * If matrix is distributed, initializes SPMVcommunication info.
   *
   * @param[in] m Reference to the sparse matrix
   */
  SparseMatrixOperator(const SpMatType& m) : m_matrix_(m) {
#ifdef MACIS_ENABLE_MPI
    if constexpr (sparsexx::is_dist_sparse_matrix_v<SpMatType>) {
      m_spmv_info_ = sparsexx::spblas::generate_spmv_comm_info(m);
    }
#endif /* MACIS_ENABLE_MPI */
  }

  /**
   * @brief Perform sparse matrix-vector multiplication:
   *
   * AV = alpha * A * V + beta * AV.
   *
   * Requires that `SpMatType` supports a (p)gespmv overload.
   *
   * @param[in] m      Number of vectors to multiply
   * @param[in] alpha  Scalar multiplier for A*V
   * @param[in] V      Input vector(s)
   * @param[in] LDV    Leading dimension of V
   * @param[in] beta   Scalar multiplier for AV
   * @param[in,out] AV Output vector(s)
   * @param[in] LDAV   Leading dimension of AV
   */
  void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                       double beta, double* AV, size_t LDAV) const {
#ifdef MACIS_ENABLE_MPI
    if constexpr (sparsexx::is_dist_sparse_matrix_v<SpMatType>) {
      sparsexx::spblas::pgespmv(alpha, m_matrix_, V, beta, AV, m_spmv_info_);
    } else {
#endif /* MACIS_ENABLE_MPI */
      sparsexx::spblas::gespmbv(m, alpha, m_matrix_, V, LDV, beta, AV, LDAV);
#ifdef MACIS_ENABLE_MPI
    }
#endif /* MACIS_ENABLE_MPI */
  }
};

/**
 * @brief Generate an initial guess vector based on the diagonal elements of a
 * sparse matrix.
 *
 * This function extracts the diagonal elements from the sparse matrix, finds
 * the minimum element, and sets the corresponding position in the guess vector
 * to 1.
 *
 * @tparam SpMatType Type of the sparse matrix
 * @param[in] N Size of the matrix/vector
 * @param[in] A Sparse matrix to extract diagonal from
 * @param[out] X Initial guess vector (zeroed except for one element)
 */
template <typename SpMatType>
void diagonal_guess(size_t N, const SpMatType& A, double* X) {
  // Extract diagonal and setup guess
  auto D = extract_diagonal_elements(A);
  auto D_min = std::min_element(D.begin(), D.end());
  auto min_idx = std::distance(D.begin(), D_min);
  X[min_idx] = 1.;
}

#ifdef MACIS_ENABLE_MPI
/**
 * @brief Generate an initial guess vector based on diagonal elements in
 * parallel (MPI).
 *
 * This is the MPI-parallel version of diagonal_guess. It gathers diagonal
 * elements from all processes, finds the global minimum, and sets the
 * appropriate element on the owning process.
 *
 * @tparam SpMatType Type of the distributed sparse matrix
 * @param[in] N_local Local size of the matrix/vector on this process
 * @param[in] A Distributed sparse matrix
 * @param[out] X Local portion of the initial guess vector
 */
template <typename SpMatType>
void p_diagonal_guess(size_t N_local, const SpMatType& A, double* X) {
  auto comm = A.comm();
  int world_rank, world_size;
  MPI_Comm_rank(comm, &world_rank);
  MPI_Comm_size(comm, &world_size);

  // Extract diagonal tile
  auto A_diagonal_tile = A.diagonal_tile_ptr();
  if (!A_diagonal_tile) throw std::runtime_error("Diagonal Tile Not Populated");

  // Gather Diagonal
  auto D_local = extract_diagonal_elements(*A_diagonal_tile);

  std::vector<int> remote_counts(world_size), row_starts(world_size + 1, 0);
  for (auto i = 0; i < world_size; ++i) {
    remote_counts[i] = A.row_extent(i);
    row_starts[i + 1] = row_starts[i] + A.row_extent(i);
  }

  std::vector<double> D(row_starts.back());

  MPI_Allgatherv(D_local.data(), D_local.size(), MPI_DOUBLE, D.data(),
                 remote_counts.data(), row_starts.data(), MPI_DOUBLE, comm);

  // Determine min index
  auto D_min = std::min_element(D.begin(), D.end());
  auto min_idx = std::distance(D.begin(), D_min);
  // printf("[rank %d] DMIN %lu %.6e\n", world_rank, min_idx, *D_min);

  // Zero out guess
  for (size_t i = 0; i < N_local; ++i) X[i] = 0.;

  // Get owner rank
  int owner_rank = min_idx / remote_counts[0];
  if (world_rank == owner_rank) {
    X[min_idx - A.local_row_start()] = 1.;
  }
}
#endif /* MACIS_ENABLE_MPI */

/**
 * @brief Perform Gram-Schmidt orthogonalization of a new vector against
 * existing vectors.
 *
 * Orthogonalizes V_new against the K columns of V_old using the classical
 * Gram-Schmidt process with reorthogonalization and normalization.
 *
 * @param[in] N     Dimension of the vectors
 * @param[in] K     Number of existing vectors to orthogonalize against
 * @param[in] V_old Matrix containing K orthonormal columns
 * @param[in] LDV   Leading dimension of V_old
 * @param[in,out] V_new Vector to orthogonalize and normalize
 * @param[in] min_norm Minimum norm threshold below which a vector is considered
 * as zero vector
 */
inline void gram_schmidt(int64_t N, int64_t K, const double* V_old, int64_t LDV,
                         double* V_new, double min_norm = 1e-12) {
  // Early return if no vectors to orthogonalize against - just normalize
  if (K <= 0) {
    auto nrm = blas::nrm2(N, V_new, 1);
    if (nrm > min_norm) {
      blas::scal(N, 1. / nrm, V_new, 1);
    }
    return;
  }

  std::vector<double> inner(K);
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, K,
             1, N, 1., V_old, LDV, V_new, N, 0., inner.data(), K);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N, 1,
             K, -1., V_old, LDV, inner.data(), K, 1., V_new, N);

  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, K,
             1, N, 1., V_old, LDV, V_new, N, 0., inner.data(), K);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N, 1,
             K, -1., V_old, LDV, inner.data(), K, 1., V_new, N);

  auto nrm = blas::nrm2(N, V_new, 1);
  // Protect against division by zero when the new vector is linearly dependent
  // on existing vectors (can happen with very small Hilbert spaces)
  if (nrm > min_norm) {
    blas::scal(N, 1. / nrm, V_new, 1);
  } else {
    // Vector is essentially zero after orthogonalization - try canonical basis
    // vectors until we find one not in the span of existing vectors
    for (int64_t idx = 0; idx < N; ++idx) {
      // Start with canonical basis vector e_idx
      std::fill_n(V_new, N, 0.0);
      V_new[idx] = 1.0;
      // Orthogonalize against existing vectors
      blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
                 K, 1, N, 1., V_old, LDV, V_new, N, 0., inner.data(), K);
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 N, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new, N);
      nrm = blas::nrm2(N, V_new, 1);
      if (nrm > min_norm) {
        blas::scal(N, 1. / nrm, V_new, 1);
        break;
      }
    }
    // If all canonical basis vectors failed, the subspace already spans the
    // entire space (K >= N). This should not happen in normal use.
    if (nrm <= min_norm) {
      throw std::runtime_error(
          "gram_schmidt: Unable to find orthogonal vector - subspace may "
          "already span the entire space");
    }
  }
}

/**
 * @brief Davidson eigensolver for finding the lowest eigenvalue and
 * eigenvector.
 *
 * Implements the Davidson algorithm to find the lowest eigenvalue and
 * corresponding eigenvector of a symmetric matrix using the provided operator
 * and diagonal preconditioner.
 *
 * @tparam Functor Type of the matrix-vector operation functor
 * @param[in] N      Size of the matrix
 * @param[in] max_m  Maximum dimension of the Davidson subspace
 * @param[in] op     Matrix-vector operation functor
 * @param[in] D      Diagonal elements for preconditioning
 * @param[in] tol    Convergence tolerance for residual norm
 * @param[in,out] X  Input: initial guess vector, Output: converged eigenvector
 * @param[in] min_abs_denominator  Minimum absolute value for preconditioner
 * denominator to avoid division by zero
 * @return Pair containing (number of iterations, converged eigenvalue)
 */
template <typename Functor>
auto davidson(int64_t N, int64_t max_m, const Functor& op, const double* D,
              double tol, double* X, double min_abs_denominator = 1e-12) {
  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double, std::milli>;

  if (!X) throw std::runtime_error("Davidson: No Guess Provided");

  auto logger = spdlog::get("davidson");
  if (!logger) {
    logger = spdlog::stdout_color_mt("davidson");
  }
  max_m = std::min(max_m, N);

  logger->info("[Davidson Eigensolver]:");
  logger->info("  {} = {:6}, {} = {:4}, {} = {:10.5e}", "N", N, "MAX_M", max_m,
               "RES_TOL", tol);

  // Handle trivial case of 1x1 matrix by evaluating the operator on the
  // current trial vector (instead of using the preconditioner diagonal D[0]).
  if (N == 1) {
    X[0] = 1.0;
    double AX = 0.0;
    op.operator_action(1, 1., X, N, 0., &AX, N);
    logger->info(
        "iter =    0, LAM(0) = {:20.12e}, RNORM =   0.000000000000e+00", AX);
    return std::make_pair(size_t(0), AX);
  }

  std::vector<double> V(N * (max_m + 1)), AV(N * (max_m + 1)),
      C((max_m + 1) * (max_m + 1)), LAM(max_m + 1);

  // Copy over guess
  std::copy_n(X, N, V.begin());

  // Compute Initial A*V
  op.operator_action(1, 1., V.data(), N, 0., AV.data(), N);

  // Copy AV(:,0) -> V(:,1) and orthogonalize wrt V(:,0)
  std::copy_n(AV.data(), N, V.data() + N);
  gram_schmidt(N, 1, V.data(), N, V.data() + N);

  bool converged = false;
  size_t iter = 1;
  for (int64_t i = 1; i < max_m; ++i, ++iter) {
    const auto k = i + 1;  // Current subspace dimension after new vector

    // AV(:,i) = A * V(:,i)
    auto op_st = hrt_t::now();
    op.operator_action(1, 1., V.data() + i * N, N, 0., AV.data() + i * N, N);
    auto op_en = hrt_t::now();
    dur_t op_dur = op_en - op_st;

    // Rayleigh Ritz
    auto rr_st = hrt_t::now();
    lobpcgxx::rayleigh_ritz(N, k, V.data(), N, AV.data(), N, LAM.data(),
                            C.data(), k);
    auto rr_en = hrt_t::now();
    dur_t rr_dur = rr_en - rr_st;

    // Compute Residual (A - LAM(0)*I) * V(:,0:i) * C(:,0)
    auto res_st = hrt_t::now();
    double* R = V.data() + (i + 1) * N;

    // X = V*C
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N,
               1, k, 1., V.data(), N, C.data(), k, 0., X, N);

    // R = X
    std::copy_n(X, N, R);

    // R = (AV - LAM[0]*V)*C = AV*C - LAM[0]*X = AV*C - LAM[0]*R
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N,
               1, k, 1., AV.data(), N, C.data(), k, -LAM[0], R, N);

    // Compute residual norm
    auto res_nrm = blas::nrm2(N, R, 1);

    auto res_en = hrt_t::now();
    dur_t res_dur = res_en - res_st;

    logger->info("iter = {:4}, LAM(0) = {:20.12e}, RNORM = {:20.12e}", i,
                 LAM[0], res_nrm);
    logger->trace(
        "  * OP_DUR = {:.2e} ms, RR_DUR = {:.2e} ms, RES_DUR = {:.2e} ms",
        op_dur.count(), rr_dur.count(), res_dur.count());

    // Check for convergence
    if (res_nrm < tol) {
      converged = true;
      break;
    }

    // Compute new vector
    // (D - LAM(0)*I) * W = -R ==> W = -(D - LAM(0)*I)**-1 * R
    // When D[j] == LAM[0], add a small shift to avoid division by zero
    for (auto j = 0; j < N; ++j) {
      double denom = D[j] - LAM[0];
      if (std::abs(denom) < min_abs_denominator) {
        denom = (denom >= 0) ? min_abs_denominator : -min_abs_denominator;
      }
      R[j] = -R[j] / denom;
    }

    // Project new vector out from old vectors
    gram_schmidt(N, k, V.data(), N, R);

  }  // Davidson iterations

  if (!converged) throw std::runtime_error("Davidson Did Not Converge!");
  logger->info("Davidson Converged!");

  return std::make_pair(iter, LAM[0]);
}

namespace detail {

struct DavidsonFree {
  void operator()(double* p) const {
#if defined(__linux__)
    std::free(p);
#else
    delete[] p;
#endif
  }
};
using davidson_array = std::unique_ptr<double[], DavidsonFree>;

/// Uninitialized array of n doubles for large work vectors; on Linux it is
/// 2 MiB aligned and advised for transparent huge pages, which saves page
/// faults and TLB misses in the strided accesses of the operator
inline davidson_array davidson_alloc(size_t n) {
#if defined(__linux__)
  void* p = nullptr;
  const size_t bytes = std::max<size_t>(n, 1) * sizeof(double);
  if (posix_memalign(&p, size_t(1) << 21, bytes) != 0) throw std::bad_alloc();
#ifdef MADV_HUGEPAGE
  madvise(p, bytes, MADV_HUGEPAGE);
#endif
  return davidson_array(static_cast<double*>(p));
#else
  return davidson_array(new double[n]);
#endif
}

inline int64_t davidson_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

/// Calls body(thread, begin, end) for a static contiguous partition of
/// [0, n) over the threads of one parallel region, with at least 32768
/// elements per thread so that small problems are not dominated by thread
/// start-up (the partition depends only on n, so arrays first touched
/// through it stay local to the threads)
template <typename Body>
void davidson_partition(int64_t n, Body&& body) {
  const int64_t team =
      std::clamp<int64_t>(n / 32768, 1, davidson_max_threads());
#pragma omp parallel num_threads(team)
  {
#ifdef _OPENMP
    const int64_t nt = omp_get_num_threads(), t = omp_get_thread_num();
#else
    const int64_t nt = 1, t = 0;
#endif
    body(t, n * t / nt, n * (t + 1) / nt);
  }
}

/// result[0:len) = sum over threads of the sums that body(begin, end, acc)
/// accumulates into acc[0:len) for its range of davidson_partition(n);
/// threads are summed in order, so results are reproducible for a fixed
/// team size, and every thread accumulates in cache lines of its own
template <typename Body>
void davidson_reduce(int64_t n, int64_t len, double* result, Body&& body) {
  const int64_t nt = davidson_max_threads();
  const int64_t stride = (len + 7) / 8 * 8;
  std::vector<double> buffer(nt * stride + 8, 0.0);
  double* partial = reinterpret_cast<double*>(
      (reinterpret_cast<uintptr_t>(buffer.data()) + 63) & ~uintptr_t(63));
  davidson_partition(n, [&](int64_t t, int64_t i0, int64_t i1) {
    body(i0, i1, partial + t * stride);
  });
  std::fill_n(result, len, 0.0);
  for (int64_t t = 0; t < nt; ++t)
    for (int64_t l = 0; l < len; ++l) result[l] += partial[t * stride + l];
}

}  // namespace detail

/**
 * @brief Davidson eigensolver with subspace restarts for the lowest eigenpair.
 *
 * Same contract as davidson(), but whenever the search space reaches max_m
 * vectors it is collapsed onto the current and previous Ritz vectors, so the
 * memory footprint stays at (2*max_m + 1) vectors of length N regardless of
 * the iteration count. Length-N work is threaded over a static partition that
 * also places the work arrays on the NUMA domains of their threads.
 *
 * @tparam Functor Type of the matrix-vector operation functor
 * @param[in] N        Size of the matrix
 * @param[in] max_m    Maximum dimension of the search space (>= 2)
 * @param[in] max_iter Maximum number of iterations (operator applications)
 * @param[in] op       Matrix-vector operation functor
 * @param[in] D        Diagonal elements for preconditioning
 * @param[in] tol      Convergence tolerance for residual norm
 * @param[in,out] X  Input: initial guess vector, Output: converged eigenvector
 * @param[in] min_abs_denominator  Minimum absolute value for preconditioner
 * denominator to avoid division by zero
 * @return Pair containing (number of iterations, converged eigenvalue)
 * @throws std::runtime_error if not converged within max_iter iterations
 */
template <typename Functor>
auto restarted_davidson(int64_t N, int64_t max_m, int64_t max_iter,
                        const Functor& op, const double* D, double tol,
                        double* X, double min_abs_denominator = 1e-12) {
  if (!X) throw std::runtime_error("Davidson: No Guess Provided");
  if (max_m < 2)
    throw std::invalid_argument("restarted_davidson: max_m must be >= 2");

  auto logger = spdlog::get("davidson");
  if (!logger) {
    logger = spdlog::stdout_color_mt("davidson");
  }
  max_m = std::min(max_m, N);

  logger->info("[Davidson Eigensolver]:");
  logger->info("  {} = {:6}, {} = {:4}, {} = {:4}, {} = {:10.5e}", "N", N,
               "MAX_M", max_m, "MAX_ITER", max_iter, "RES_TOL", tol);

  if (N == 1) {
    X[0] = 1.0;
    double AX = 0.0;
    op.operator_action(1, 1., X, N, 0., &AX, N);
    return std::make_pair(size_t(0), AX);
  }

  // Elements per inner block of the fused passes (stack buffers)
  constexpr int64_t chunk = 512;
  using detail::davidson_partition;
  using detail::davidson_reduce;

  // Search space V (max_m columns plus one for the new direction) and AV,
  // addressed through column pointers so that the new direction can be
  // renamed after a collapse; first touched with the partition of all passes
  auto v_store = detail::davidson_alloc(N * (max_m + 1));
  auto av_store = detail::davidson_alloc(N * max_m);
  std::vector<double*> V(max_m + 1), AV(max_m);
  for (int64_t j = 0; j <= max_m; ++j) V[j] = v_store.get() + j * N;
  for (int64_t j = 0; j < max_m; ++j) AV[j] = av_store.get() + j * N;
  davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
    for (auto* v : V) std::fill(v + i0, v + i1, 0.0);
    for (auto* v : AV) std::fill(v + i0, v + i1, 0.0);
  });

  std::vector<double> H(max_m * max_m), Y(max_m * max_m), LAM(max_m);
  std::vector<double> y_prev(max_m, 0.0), U(2 * max_m), HU(2 * max_m);
  std::vector<double> sums(max_m + 2), proj(max_m + 1);

  // t -= V[0:k) c, optionally accumulating V[0:k)^T t (after the update)
  // into acc, on the block [c0, c0 + len)
  auto project_out = [&](int64_t k, const double* c, int64_t c0, int64_t len,
                         double* acc) {
    double* t = V[k] + c0;
    for (int64_t j = 0; j < k; ++j) {
      const double* v = V[j] + c0;
      const double cj = c[j];
      for (int64_t i = 0; i < len; ++i) t[i] -= cj * v[i];
    }
    if (acc)
      for (int64_t j = 0; j < k; ++j) {
        const double* v = V[j] + c0;
        double d = 0.0;
        for (int64_t i = 0; i < len; ++i) d += v[i] * t[i];
        acc[j] += d;
      }
  };

  // V[0] = X / |X|
  davidson_reduce(N, 1, sums.data(), [&](int64_t i0, int64_t i1, double* acc) {
    double d = 0.0;
    for (int64_t i = i0; i < i1; ++i) d += X[i] * X[i];
    acc[0] += d;
  });
  if (!(sums[0] > 0.0))
    throw std::runtime_error("restarted_davidson: zero initial guess");
  {
    const double f = 1.0 / std::sqrt(sums[0]);
    davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) V[0][i] = f * X[i];
    });
  }
  op.operator_action(1, 1., V[0], N, 0., AV[0], N);
  davidson_reduce(N, 1, H.data(), [&](int64_t i0, int64_t i1, double* acc) {
    double d = 0.0;
    for (int64_t i = i0; i < i1; ++i) d += V[0][i] * AV[0][i];
    acc[0] += d;
  });

  int64_t k = 1;
  bool have_prev = false;
  bool converged = false;
  size_t iter = 1;
  for (;; ++iter) {
    // Rayleigh-Ritz on the k-dimensional search space
    for (int64_t j = 0; j < k; ++j)
      std::copy_n(H.data() + j * max_m, k, Y.data() + j * k);
    lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, k, Y.data(), k,
                 LAM.data());
    const double* y = Y.data();
    const double lam = LAM[0];

    // Residual r = AV y - lam V y (not stored), new direction
    // t = -(D - lam)^-1 r in V[k] and V[0:k)^T t, in one pass:
    // sums = [|r|^2, V^T t]
    davidson_reduce(
        N, k + 1, sums.data(), [&](int64_t i0, int64_t i1, double* acc) {
          alignas(64) double xs[chunk], rs[chunk];
          for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
            const int64_t len = std::min(chunk, i1 - c0);
            std::fill_n(xs, len, 0.0);
            std::fill_n(rs, len, 0.0);
            for (int64_t j = 0; j < k; ++j) {
              const double* v = V[j] + c0;
              const double* av = AV[j] + c0;
              const double yj = y[j];
              for (int64_t i = 0; i < len; ++i) {
                xs[i] += yj * v[i];
                rs[i] += yj * av[i];
              }
            }
            double* t = V[k] + c0;
            double r2 = 0.0;
            for (int64_t i = 0; i < len; ++i) {
              const double r = rs[i] - lam * xs[i];
              r2 += r * r;
              double denom = D[c0 + i] - lam;
              if (std::abs(denom) < min_abs_denominator)
                denom = denom >= 0 ? min_abs_denominator : -min_abs_denominator;
              t[i] = -r / denom;
            }
            acc[0] += r2;
            for (int64_t j = 0; j < k; ++j) {
              const double* v = V[j] + c0;
              double d = 0.0;
              for (int64_t i = 0; i < len; ++i) d += v[i] * t[i];
              acc[1 + j] += d;
            }
          }
        });
    const double res_nrm = std::sqrt(sums[0]);
    std::copy_n(sums.data() + 1, k, proj.data());

    logger->info("iter = {:4}, LAM(0) = {:20.12e}, RNORM = {:20.12e}", iter,
                 lam, res_nrm);

    if (res_nrm < tol) {
      converged = true;
      break;
    }
    if (iter >= size_t(max_iter)) break;

    if (k == max_m) {
      // Collapse onto U = [y, y_prev orthogonalized against y] (just y when
      // max_m == 2, to leave room for the new direction). Near convergence
      // y_prev is almost parallel to y, so the projection is repeated to keep
      // U orthonormal to working precision.
      std::copy_n(y, k, U.data());
      int64_t kk = 1;
      if (have_prev and max_m > 2) {
        double* u1 = U.data() + k;
        std::copy_n(y_prev.data(), k, u1);
        for (int pass = 0; pass < 2; ++pass)
          blas::axpy(k, -blas::dot(k, y, 1, u1, 1), y, 1, u1, 1);
        const auto nrm = blas::nrm2(k, u1, 1);
        if (nrm > 1e-12) {
          blas::scal(k, 1. / nrm, u1, 1);
          kk = 2;
        }
      }
      // [V|AV][0:kk) <- [V|AV][0:k) U, in place block by block
      davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
        alignas(64) double nv[2][chunk];
        for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
          const int64_t len = std::min(chunk, i1 - c0);
          for (auto* B : {&V, &AV}) {
            for (int64_t c = 0; c < kk; ++c) {
              std::fill_n(nv[c], len, 0.0);
              for (int64_t j = 0; j < k; ++j) {
                const double u = U[j + c * k];
                const double* b = (*B)[j] + c0;
                for (int64_t i = 0; i < len; ++i) nv[c][i] += u * b[i];
              }
            }
            for (int64_t c = 0; c < kk; ++c)
              std::copy_n(nv[c], len, (*B)[c] + c0);
          }
        }
      });
      // H <- U**T * H * U, and V^T t in the new basis
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 k, kk, k, 1., H.data(), max_m, U.data(), k, 0., HU.data(), k);
      blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, kk,
                 kk, k, 1., U.data(), k, HU.data(), k, 0., H.data(), max_m);
      double new_proj[2];
      for (int64_t c = 0; c < kk; ++c)
        new_proj[c] = blas::dot(k, U.data() + c * k, 1, proj.data(), 1);
      std::copy_n(new_proj, kk, proj.data());
      std::swap(V[k], V[kk]);
      k = kk;
      // The current Ritz vector is the first new basis vector
      std::fill(y_prev.begin(), y_prev.end(), 0.0);
      y_prev[0] = 1.0;
    } else {
      std::copy_n(y, k, y_prev.data());
    }
    have_prev = true;

    // Classical Gram-Schmidt, twice: t -= V proj, sums = V^T t;
    // t -= V sums, |t|^2. A direction inside the search space (possible only
    // for tiny N) is replaced by unit vectors until one is not.
    double t_nrm = 0.0;
    for (int64_t unit = -1; unit < N; ++unit) {
      if (unit >= 0) {
        davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
          for (int64_t i = i0; i < i1; ++i) V[k][i] = i == unit ? 1.0 : 0.0;
        });
        davidson_reduce(
            N, k, proj.data(), [&](int64_t i0, int64_t i1, double* acc) {
              for (int64_t j = 0; j < k; ++j)
                acc[j] += unit >= i0 and unit < i1 ? V[j][unit] : 0.0;
            });
      }
      davidson_reduce(
          N, k, sums.data(), [&](int64_t i0, int64_t i1, double* acc) {
            for (int64_t c0 = i0; c0 < i1; c0 += chunk)
              project_out(k, proj.data(), c0, std::min(chunk, i1 - c0), acc);
          });
      double t2;
      davidson_reduce(N, 1, &t2, [&](int64_t i0, int64_t i1, double* acc) {
        for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
          const int64_t len = std::min(chunk, i1 - c0);
          project_out(k, sums.data(), c0, len, nullptr);
          const double* t = V[k] + c0;
          double d = 0.0;
          for (int64_t i = 0; i < len; ++i) d += t[i] * t[i];
          acc[0] += d;
        }
      });
      t_nrm = std::sqrt(t2);
      if (t_nrm > 1e-12) break;
    }

    // AV[k] = H t / |t|; then t /= |t| together with the new row of the
    // projected matrix, sums = V[0:k]^T AV[k]
    op.operator_action(1, 1. / t_nrm, V[k], N, 0., AV[k], N);
    davidson_reduce(N, k + 1, sums.data(),
                    [&](int64_t i0, int64_t i1, double* acc) {
                      const double f = 1.0 / t_nrm;
                      for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
                        const int64_t len = std::min(chunk, i1 - c0);
                        double* t = V[k] + c0;
                        for (int64_t i = 0; i < len; ++i) t[i] *= f;
                        const double* av = AV[k] + c0;
                        for (int64_t j = 0; j <= k; ++j) {
                          const double* v = V[j] + c0;
                          double d = 0.0;
                          for (int64_t i = 0; i < len; ++i) d += v[i] * av[i];
                          acc[j] += d;
                        }
                      }
                    });
    for (int64_t j = 0; j <= k; ++j)
      H[j + k * max_m] = H[k + j * max_m] = sums[j];
    ++k;
  }

  if (!converged) throw std::runtime_error("Davidson Did Not Converge!");
  logger->info("Davidson Converged!");

  // X = V y
  {
    const double* y = Y.data();
    davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
      for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
        const int64_t len = std::min(chunk, i1 - c0);
        double* x = X + c0;
        std::fill_n(x, len, 0.0);
        for (int64_t j = 0; j < k; ++j) {
          const double* v = V[j] + c0;
          const double yj = y[j];
          for (int64_t i = 0; i < len; ++i) x[i] += yj * v[i];
        }
      }
    });
  }

  return std::make_pair(iter, LAM[0]);
}

#ifdef MACIS_ENABLE_MPI
/**
 * @brief Parallel (MPI) Gram-Schmidt orthogonalization.
 *
 * MPI-parallel version of Gram-Schmidt orthogonalization. Each process works
 * on its local portion of the vectors, with global communication for inner
 * products and normalization.
 *
 * @param[in] N_local Local dimension of vectors on this process
 * @param[in] K       Number of existing vectors to orthogonalize against
 * @param[in] V_old   Local portion of matrix containing K orthonormal columns
 * @param[in] LDV     Leading dimension of V_old
 * @param[in,out] V_new Local portion of vector to orthogonalize and normalize
 * @param[in] comm    MPI communicator
 * @param[in] min_norm Minimum norm threshold below which a vector is considered
 * as zero vector
 */
inline void p_gram_schmidt(int64_t N_local, int64_t K, const double* V_old,
                           int64_t LDV, double* V_new, MPI_Comm comm,
                           double min_norm = 1e-12) {
  // Early return if no vectors to orthogonalize against - just normalize
  if (K <= 0) {
    double dot = blas::dot(N_local, V_new, 1, V_new, 1);
    dot = allreduce(dot, MPI_SUM, comm);
    double nrm = std::sqrt(dot);
    if (nrm > min_norm) {
      blas::scal(N_local, 1. / nrm, V_new, 1);
    }
    return;
  }

  std::vector<double> inner(K);
  // Compute local V_old**H * V_new
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, K,
             1, N_local, 1., V_old, LDV, V_new, N_local, 0., inner.data(), K);

  // Reduce inner product
  allreduce(inner.data(), K, MPI_SUM, comm);

  // Project locally
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             N_local, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new,
             N_local);

  // Repeat
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, K,
             1, N_local, 1., V_old, LDV, V_new, N_local, 0., inner.data(), K);
  allreduce(inner.data(), K, MPI_SUM, comm);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             N_local, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new,
             N_local);

  // Normalize
  double dot = blas::dot(N_local, V_new, 1, V_new, 1);
  dot = allreduce(dot, MPI_SUM, comm);
  double nrm = std::sqrt(dot);
  // Protect against division by zero when the new vector is linearly dependent
  // on existing vectors (can happen with very small Hilbert spaces)
  if (nrm > min_norm) {
    blas::scal(N_local, 1. / nrm, V_new, 1);
  } else {
    // Vector is essentially zero after orthogonalization - try canonical basis
    // vectors until we find one not in the span of existing vectors.
    // In parallel, each rank handles its local portion of the global vector.
    int rank = comm_rank(comm);
    int size = comm_size(comm);
    // Gather local sizes to determine global index mapping
    std::vector<int64_t> local_sizes(size);
    int64_t my_size = N_local;
    MPI_Allgather(&my_size, 1, MPI_INT64_T, local_sizes.data(), 1, MPI_INT64_T,
                  comm);
    int64_t global_offset = 0;
    for (int r = 0; r < rank; ++r) global_offset += local_sizes[r];
    int64_t N_global = 0;
    for (int r = 0; r < size; ++r) N_global += local_sizes[r];

    for (int64_t idx = 0; idx < N_global; ++idx) {
      // Start with canonical basis vector e_idx (distributed)
      std::fill_n(V_new, N_local, 0.0);
      if (idx >= global_offset && idx < global_offset + N_local) {
        V_new[idx - global_offset] = 1.0;
      }
      // Orthogonalize against existing vectors
      blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
                 K, 1, N_local, 1., V_old, LDV, V_new, N_local, 0.,
                 inner.data(), K);
      allreduce(inner.data(), K, MPI_SUM, comm);
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 N_local, 1, K, -1., V_old, LDV, inner.data(), K, 1., V_new,
                 N_local);
      dot = blas::dot(N_local, V_new, 1, V_new, 1);
      dot = allreduce(dot, MPI_SUM, comm);
      nrm = std::sqrt(dot);
      if (nrm > min_norm) {
        blas::scal(N_local, 1. / nrm, V_new, 1);
        break;
      }
    }
    // If all canonical basis vectors failed, the subspace already spans the
    // entire space (K >= N_global). This should not happen in normal use.
    if (nrm <= min_norm) {
      throw std::runtime_error(
          "p_gram_schmidt: Unable to find orthogonal vector - subspace may "
          "already span the entire space");
    }
  }
}

/**
 * @brief Parallel (MPI) Rayleigh-Ritz procedure for subspace eigenvalue
 * problems.
 *
 * Performs the Rayleigh-Ritz procedure in parallel: computes the projected
 * matrix X^T * AX, diagonalizes it on root process, and broadcasts the
 * eigenvalues and eigenvectors to all processes.
 *
 * @param[in] N_local Local dimension of vectors on this process
 * @param[in] K       Dimension of the subspace
 * @param[in] X       Local portion of subspace vectors
 * @param[in] LDX     Leading dimension of X
 * @param[in] AX      Local portion of A*X vectors
 * @param[in] LDAX    Leading dimension of AX
 * @param[out] W      Eigenvalues (output on all processes)
 * @param[out] C      Eigenvectors (output on all processes)
 * @param[in] LDC     Leading dimension of C
 * @param[in] comm    MPI communicator
 */
inline void p_rayleigh_ritz(int64_t N_local, int64_t K, const double* X,
                            int64_t LDX, const double* AX, int64_t LDAX,
                            double* W, double* C, int64_t LDC, MPI_Comm comm) {
  int world_rank;
  MPI_Comm_rank(comm, &world_rank);

  // Compute Local inner product
  blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans, K,
             K, N_local, 1., X, LDX, AX, LDAX, 0., C, LDC);

  // Reduce result
  if (LDC != K)
    throw std::runtime_error(
        "Rayleigh-Ritz does not accept submatrices for eigenvector return "
        "storage");
  // allreduce(C, K * K, MPI_SUM, comm);
  std::allocator<double> alloc;
  double* tmp_c = world_rank ? nullptr : alloc.allocate(K * K);
  reduce(C, tmp_c, K * K, MPI_SUM, 0, comm);

  // Do local diagonalization on rank-0
  if (!world_rank) {
    memcpy(C, tmp_c, K * K * sizeof(double));
    lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, K, C, LDC, W);
    alloc.deallocate(tmp_c, K * K);
  }

  // Broadcast results
  MPI_Bcast(W, K, MPI_DOUBLE, 0, comm);
  MPI_Bcast(C, K * K, MPI_DOUBLE, 0, comm);
}

/**
 * @brief Parallel (MPI) Davidson eigensolver for finding the lowest eigenvalue
 * and eigenvector.
 *
 * MPI-parallel implementation of the Davidson algorithm. Each process works on
 * its local portion of vectors, with global communication for inner products,
 * reductions, and broadcasts as needed.
 *
 * @tparam Functor Type of the matrix-vector operation functor
 * @param[in] N_local   Local size of vectors on this process
 * @param[in] max_m     Maximum dimension of the Davidson subspace
 * @param[in] op        Matrix-vector operation functor
 * @param[in] D_local   Local portion of diagonal elements for preconditioning
 * @param[in] tol       Convergence tolerance for residual norm
 * @param[in,out] X_local Input: local portion of initial guess, Output: local
 * portion of eigenvector
 * @param[in] comm      MPI communicator
 * @param[in] min_abs_denominator  Minimum absolute value for preconditioner
 * denominator to avoid division by zero
 * @return Pair containing (number of iterations, converged eigenvalue)
 */
template <typename Functor>
auto p_davidson(int64_t N_local, int64_t max_m, const Functor& op,
                const double* D_local, double tol, double* X_local,
                MPI_Comm comm, double min_abs_denominator = 1e-12) {
  using hrt_t = std::chrono::high_resolution_clock;
  using dur_t = std::chrono::duration<double, std::milli>;

  if (N_local and !X_local)
    throw std::runtime_error("Davidson: No Guess Provided");

  int world_rank, world_size;
  MPI_Comm_rank(comm, &world_rank);
  MPI_Comm_size(comm, &world_size);

  auto logger = spdlog::get("davidson");
  if (!logger) {
    logger = world_rank ? spdlog::null_logger_mt("davidson")
                        : spdlog::stdout_color_mt("davidson");
  }

  logger->info("[Davidson Eigensolver]:");
  logger->info("  {} = {:6}, {} = {:4}, {} = {:10.5e}", "N_LOCAL", N_local,
               "MAX_M", max_m, "RES_TOL", tol);

  // Allocations
  std::vector<double> V_local(N_local * (max_m + 1)),
      AV_local(N_local * (max_m + 1)), C((max_m + 1) * (max_m + 1)),
      LAM(max_m + 1);

  // Copy over guess
  std::copy_n(X_local, N_local, V_local.begin());

  // Compute initial A*V
  op.operator_action(1, 1., V_local.data(), N_local, 0., AV_local.data(),
                     N_local);

  // Copy AV(:,0) -> V(:,1) and orthogonalize wrt V(:,0)
  std::copy_n(AV_local.data(), N_local, V_local.data() + N_local);
  p_gram_schmidt(N_local, 1, V_local.data(), N_local, V_local.data() + N_local,
                 comm);

  bool converged = false;
  int64_t iter = 1;
  for (int64_t i = 1; i < max_m; ++i, ++iter) {
    const auto k = i + 1;  // Current subspace dimension after new vector

    // AV(:,i) = A * V(:,i)
    auto op_st = hrt_t::now();
    op.operator_action(1, 1., V_local.data() + i * N_local, N_local, 0.,
                       AV_local.data() + i * N_local, N_local);
    auto op_en = hrt_t::now();
    dur_t op_dur = op_en - op_st;

    // Rayleigh Ritz
    auto rr_st = hrt_t::now();
    p_rayleigh_ritz(N_local, k, V_local.data(), N_local, AV_local.data(),
                    N_local, LAM.data(), C.data(), k, comm);
    auto rr_en = hrt_t::now();
    dur_t rr_dur = rr_en - rr_st;

    // Compute Residual (A - LAM(0)*I) * V(:,0:i) * C(:,0)
    auto res_st = hrt_t::now();
    double* R_local = V_local.data() + (i + 1) * N_local;

    // X = V*C
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               N_local, 1, k, 1., V_local.data(), N_local, C.data(), k, 0.,
               X_local, N_local);

    // R = X
    std::copy_n(X_local, N_local, R_local);

    // R = (AV - LAM[0]*V)*C = AV*C - LAM[0]*X = AV*C - LAM[0]*R
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               N_local, 1, k, 1., AV_local.data(), N_local, C.data(), k,
               -LAM[0], R_local, N_local);

    // Compute residual norm
    auto res_dot = blas::dot(N_local, R_local, 1, R_local, 1);
    res_dot = allreduce(res_dot, MPI_SUM, comm);
    auto res_nrm = std::sqrt(res_dot);

    auto res_en = hrt_t::now();
    dur_t res_dur = res_en - res_st;

    logger->info("iter = {:4}, LAM(0) = {:20.12e}, RNORM = {:20.12e}", i,
                 LAM[0], res_nrm);
    logger->trace(
        "  * OP_DUR = {:.2e} ms, RR_DUR = {:.2e} ms, RES_DUR = {:.2e} ms",
        op_dur.count(), rr_dur.count(), res_dur.count());

    // Check for convergence
    if (res_nrm < tol) {
      converged = true;
      break;
    }

    // Compute new vector
    // (D - LAM(0)*I) * W = -R ==> W = -(D - LAM(0)*I)**-1 * R
    // When D[j] == LAM[0], add a small shift to avoid division by zero
    double E1_denom = 0, E1_num = 0;
    for (auto j = 0; j < N_local; ++j) {
      double denom = D_local[j] - LAM[0];
      if (std::abs(denom) < min_abs_denominator) {
        denom = (denom >= 0) ? min_abs_denominator : -min_abs_denominator;
      }
      R_local[j] = -R_local[j] / denom;
      E1_num += X_local[j] * R_local[j];
      E1_denom += X_local[j] * X_local[j] / denom;
    }
    E1_denom = allreduce(E1_denom, MPI_SUM, comm);
    if (std::abs(E1_denom) < min_abs_denominator) {
      E1_denom = (E1_denom >= 0) ? min_abs_denominator : -min_abs_denominator;
    }
    E1_num = allreduce(E1_num, MPI_SUM, comm);
    const double E1 = E1_num / E1_denom;

    for (auto j = 0; j < N_local; ++j) {
      double denom = D_local[j] - LAM[0];
      if (std::abs(denom) < min_abs_denominator) {
        denom = (denom >= 0) ? min_abs_denominator : -min_abs_denominator;
      }
      R_local[j] += E1 * X_local[j] / denom;
    }

    // Project new vector out from old vectors
    p_gram_schmidt(N_local, k, V_local.data(), N_local, R_local, comm);

  }  // Davidson iterations

  if (!converged) throw std::runtime_error("Davidson Did Not Converge!");
  logger->info("Davidson Converged!");

  return std::make_pair(iter, LAM[0]);
}
#endif /* MACIS_ENABLE_MPI */

}  // namespace macis

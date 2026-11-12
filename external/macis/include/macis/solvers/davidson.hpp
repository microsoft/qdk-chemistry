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
#include <limits>
#include <lobpcgxx/lobpcg.hpp>
#include <macis/util/mpi.hpp>
#include <memory>
#include <new>
#include <sparsexx/matrix_types/csr_matrix.hpp>
#include <type_traits>

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

/// Alignment of davidson_alloc arrays (a cache line); vectorized kernels may
/// rely on it for aligned loads
inline constexpr size_t davidson_alignment = 64;

struct DavidsonFree {
  template <typename T>
  void operator()(T* p) const {
#if defined(__linux__)
    std::free(p);
#else
    ::operator delete(static_cast<void*>(p),
                      std::align_val_t{davidson_alignment});
#endif
  }
};
template <typename T = double>
using davidson_buffer = std::unique_ptr<T[], DavidsonFree>;
using davidson_array = davidson_buffer<double>;

/// Uninitialized array of n elements for large work vectors, aligned to
/// davidson_alignment. On Linux, arrays of 1 MiB or more are rounded up to
/// whole 2 MiB pages and advised for transparent huge pages, which saves page
/// faults and TLB misses in the strided accesses of the operator (2 MiB is
/// the huge page size of x86-64 and of 4 KiB-page ARM64 kernels; elsewhere
/// the advice is merely ineffective)
template <typename T = double>
inline davidson_buffer<T> davidson_alloc(size_t n) {
  size_t bytes = std::max<size_t>(n, 1) * sizeof(T);
#if defined(__linux__)
  constexpr size_t huge_page = size_t(1) << 21;
  const bool huge = bytes >= huge_page / 2;
  if (huge) bytes = (bytes + huge_page - 1) & ~(huge_page - 1);
  void* p = nullptr;
  if (posix_memalign(&p, huge ? huge_page : davidson_alignment, bytes) != 0)
    throw std::bad_alloc();
#ifdef MADV_HUGEPAGE
  if (huge) madvise(p, bytes, MADV_HUGEPAGE);
#endif
#else
  void* p = ::operator new(bytes, std::align_val_t{davidson_alignment});
#endif
  return davidson_buffer<T>(static_cast<T*>(p));
}

inline int64_t davidson_max_threads() {
#ifdef _OPENMP
  return omp_get_max_threads();
#else
  return 1;
#endif
}

/// Threads of a pass over arrays of n elements: one below 32768 elements,
/// where thread start-up would dominate, else the whole team (one team size
/// for all passes also avoids the cost of growing a team between them)
inline int64_t davidson_team(int64_t n) {
  return n < 32768 ? 1 : davidson_max_threads();
}

/// Calls body(thread, begin, end) for a static contiguous partition of
/// [0, n) over the davidson_team(n) threads of one parallel region (the
/// partition depends only on n and the team, so arrays first touched through
/// it stay local to the threads)
template <typename Body>
void davidson_partition(int64_t n, Body&& body) {
#pragma omp parallel num_threads(davidson_team(n))
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

/// Sum of term(i) over [i0, i1), accumulated in blocks of 512 terms. A single
/// running sum over millions of terms of one sign rounds away those below
/// half an ulp of it, always in the same direction: a Rayleigh quotient over
/// 1.66e8 determinants came out 1e-10 too high.
template <typename Term>
double davidson_sum(int64_t i0, int64_t i1, Term&& term) {
  constexpr int64_t block = 512;
  double sum = 0.0;
  for (int64_t b0 = i0; b0 < i1; b0 += block) {
    const int64_t b1 = std::min(b0 + block, i1);
    double s = 0.0;
    for (int64_t i = b0; i < b1; ++i) s += term(i);
    sum += s;
  }
  return sum;
}

/// Sum of term(i) over [0, len) in eight interleaved partial sums: unlike a
/// single running sum (a chain of dependent additions, which compilers do
/// not reorder), it vectorizes
template <typename Term>
inline double davidson_lane_sum(int64_t len, Term&& term) {
  double s[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t i = 0;
  for (; i + 8 <= len; i += 8)
    for (int l = 0; l < 8; ++l) s[l] += term(i + l);
  for (int l = 0; i < len; ++i, ++l) s[l] += term(i);
  return ((s[0] + s[4]) + (s[1] + s[5])) + ((s[2] + s[6]) + (s[3] + s[7]));
}

}  // namespace detail

/**
 * @brief Exact block of the restarted_davidson preconditioner on a few
 * coordinates, e.g. a P-space of important determinants.
 *
 * With the matrix restricted to the span of the columns of Z (index.size() x
 * lambda.size(), orthonormal columns over the coordinates `index`) equal to
 * Z diag(lambda) Z^T, the new direction on those coordinates is
 * -Z (lambda - lam)^-1 Z^T r instead of the diagonal's -r / (D - lam).
 * Components with |lambda_a - lam| < min_abs_denominator are dropped.
 */
struct DavidsonBlockPreconditioner {
  std::vector<int64_t> index;
  std::vector<double> Z;
  std::vector<double> lambda;
  double min_abs_denominator = 1e-3;
};

/**
 * @brief Davidson eigensolver with subspace restarts for the lowest eigenpair.
 *
 * Same contract as davidson(), but whenever the search space reaches max_m
 * vectors it is collapsed onto the current and previous Ritz vectors, so the
 * memory footprint stays at (2*max_m + 1) vectors of length N regardless of
 * the iteration count.
 *
 * @note Davidson's method (J. Comput. Phys. 17, 87 (1975)), restarted onto the
 * current and previous Ritz vectors as in Stathopoulos & Saad (Electron. Trans.
 * Numer. Anal. 7, 163 (1998)). The anchored mode for an inexact op is a form
 * of iterative refinement (Dongarra, Moler & Wilkinson, SIAM J. Numer. Anal.
 * 20, 23 (1983)), and re-anchoring on the estimated error follows the residual
 * replacement of van der Vorst & Ye (SIAM J. Sci. Comput. 22, 835 (2000)).
 *
 * @tparam TC Storage type of the search space and its images except their
 * first vectors, which are double; float halves the memory of a
 * single-precision op
 * @tparam Functor Type of the matrix-vector operation functor (on vectors of
 * TC)
 * @tparam ExactFunctor Type of the exact operator (on vectors of double)
 * @param[in] N        Size of the matrix
 * @param[in] max_m    Maximum dimension of the search space (>= 2)
 * @param[in] max_iter Maximum number of iterations (operator applications)
 * @param[in] op       Matrix-vector operation functor
 * @param[in] D        Diagonal elements for preconditioning
 * @param[in] tol      Convergence tolerance for residual norm
 * @param[in,out] X  Input: initial guess vector, Output: converged eigenvector
 * @param[in] min_abs_denominator  Minimum absolute value for preconditioner
 * denominator to avoid division by zero
 * @param[in] throw_on_failure  If false, running out of iterations is not an
 * error: X then receives the current Ritz vector
 * @param[out] residual_norm  If not null, the residual norm of the returned
 * Ritz pair as computed from the operator applications
 * @param[in] stagnation_window  If positive, also stop (unconverged) once the
 * residual norm has not halved over this many iterations
 * @param[in] exact  Optional exact operator for an inexact (e.g. reduced
 * precision) op: once the residual falls below anchor_tol (or tol, or
 * stagnates), the image of the Ritz vector is recomputed with exact (the
 * anchor); convergence is only accepted after anchoring
 * @param[in] anchor_tol  Residual norm at which to anchor; infinity anchors
 * the start vector
 * @param[in] block  Optional exact block of the preconditioner
 * @param[in] op_error  With exact: estimated error norm of op's image of a
 * unit vector. If positive, the solve estimates the resulting error of the
 * residual, re-anchors whenever it exceeds a quarter of the residual (or of
 * tol, if larger), accepts convergence only when the residual plus the
 * estimate is below tol, and stops (unconverged) if re-anchoring cannot keep
 * up
 * @return Pair containing (number of iterations, converged eigenvalue)
 * @throws std::runtime_error if not converged within max_iter iterations
 * and throw_on_failure is true
 */
template <typename TC = double, typename Functor,
          typename ExactFunctor = Functor>
auto restarted_davidson(
    int64_t N, int64_t max_m, int64_t max_iter, const Functor& op,
    const double* D, double tol, double* X, double min_abs_denominator = 1e-12,
    bool throw_on_failure = true, double* residual_norm = nullptr,
    int64_t stagnation_window = 0, const ExactFunctor* exact = nullptr,
    double anchor_tol = 0.0, const DavidsonBlockPreconditioner* block = nullptr,
    double op_error = 0.0) {
  static_assert(std::is_same_v<TC, double> or std::is_same_v<TC, float>,
                "restarted_davidson: TC must be double or float");
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
    TC x = 1, ax = 0;
    op.operator_action(1, 1., &x, N, 0., &ax, N);
    if (residual_norm) *residual_norm = 0.0;
    return std::make_pair(size_t(0), double(ax));
  }

  // Elements per inner block of the fused passes (4 KiB stack buffers that
  // stay in L1; unrelated to the vector width)
  constexpr int64_t chunk = 512;
  using detail::davidson_partition;
  using detail::davidson_reduce;

  // Search space V (max_m columns plus one for the new direction) and AV,
  // their first columns in double and the others in TC, addressed through
  // column pointers so that the new direction can be renamed after a
  // collapse. V is first written by the passes below, which all use the
  // static partition (its first touch places it); AV is first written by
  // the operator, so it is first touched here
  auto v0_store = detail::davidson_alloc<double>(N);
  auto av0_store = detail::davidson_alloc<double>(N);
  auto v_store = detail::davidson_alloc<TC>(N * max_m);
  auto av_store = detail::davidson_alloc<TC>(N * (max_m - 1));
  double* const V0 = v0_store.get();
  double* const AV0 = av0_store.get();
  std::vector<TC*> V(max_m + 1, nullptr), AV(max_m, nullptr);  // from 1
  for (int64_t j = 1; j <= max_m; ++j) V[j] = v_store.get() + (j - 1) * N;
  for (int64_t j = 1; j < max_m; ++j) AV[j] = av_store.get() + (j - 1) * N;
  davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
    std::fill(AV0 + i0, AV0 + i1, 0.0);
    for (int64_t j = 1; j < max_m; ++j)
      std::fill(AV[j] + i0, AV[j] + i1, TC(0));
  });

  // f(p) for p the typed pointer to element i of column j of V or AV
  auto vcol = [&](int64_t j, int64_t i, auto&& f) {
    if (j == 0)
      f(V0 + i);
    else
      f(V[j] + i);
  };
  auto avcol = [&](int64_t j, int64_t i, auto&& f) {
    if (j == 0)
      f(AV0 + i);
    else
      f(AV[j] + i);
  };
  // *p = x; single precision flushes values below its normal range to zero
  // (they carry nothing there, and every operation on one is slow on x86)
  auto store = [](auto* p, double x) {
    if constexpr (std::is_same_v<std::remove_pointer_t<decltype(p)>, float>)
      *p = std::abs(x) < double(std::numeric_limits<float>::min()) ? 0.0f
                                                                   : float(x);
    else
      *p = x;
  };

  std::vector<double> H(max_m * max_m), Y(max_m * max_m), LAM(max_m);
  std::vector<double> y_prev(max_m, 0.0), U(2 * max_m), HU(2 * max_m);
  std::vector<double> sums(2 * max_m + 8), proj(max_m + 1);

  // Overlaps G = V^T V of the stored basis. Single-precision storage rounds
  // every stored vector, so V is orthonormal only to about 1e-8, and a
  // Rayleigh-Ritz step that took G = I would leave a residual component
  // inside the search space of about |lam| 1e-8, which no new direction can
  // remove. With float storage G (and every entry of H that involves a
  // rounded vector) is therefore measured on the stored values and the
  // projected problem is H y = lam G y; double storage keeps G = I.
  constexpr bool measure_overlaps = std::is_same_v<TC, float>;
  std::vector<double> G(max_m * max_m, 0.0), GW(max_m * max_m);
  G[0] = 1.0;

  // t -= V[0:k) c, optionally accumulating V[0:k)^T t (after the update)
  // into acc, on the block [c0, c0 + len) of t = V[k]
  auto project_out = [&](int64_t k, const double* c, int64_t c0, int64_t len,
                         double* acc) {
    TC* tk = V[k] + c0;
    alignas(64) double t[chunk];
    for (int64_t i = 0; i < len; ++i) t[i] = tk[i];
    for (int64_t j = 0; j < k; ++j) {
      const double cj = c[j];
      vcol(j, c0, [&](const auto* v) {
        for (int64_t i = 0; i < len; ++i) t[i] -= cj * v[i];
      });
    }
    for (int64_t i = 0; i < len; ++i) {
      store(tk + i, t[i]);
      t[i] = tk[i];
    }
    if (acc)
      for (int64_t j = 0; j < k; ++j)
        vcol(j, c0, [&](const auto* v) {
          acc[j] += detail::davidson_lane_sum(
              len, [&](int64_t i) { return double(v[i]) * t[i]; });
        });
  };

  // V[0] = X / |X|
  davidson_reduce(N, 1, sums.data(), [&](int64_t i0, int64_t i1, double* acc) {
    acc[0] +=
        detail::davidson_sum(i0, i1, [&](int64_t i) { return X[i] * X[i]; });
  });
  if (!(sums[0] > 0.0))
    throw std::runtime_error("restarted_davidson: zero initial guess");
  {
    const double f = 1.0 / std::sqrt(sums[0]);
    davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) V0[i] = f * X[i];
    });
  }
  // Its image: exact when it is the anchor; with op, through the first
  // column of TC (not in use yet) unless that is double
  const bool anchor_start = exact != nullptr and std::isinf(anchor_tol);
  if (anchor_start) {
    exact->operator_action(1, 1., V0, N, 0., AV0, N);
  } else if constexpr (std::is_same_v<TC, double>) {
    op.operator_action(1, 1., V0, N, 0., AV0, N);
  } else {
    davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) store(V[1] + i, V0[i]);
    });
    op.operator_action(1, 1., V[1], N, 0., AV[1], N);
    davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
      for (int64_t i = i0; i < i1; ++i) AV0[i] = AV[1][i];
    });
  }
  davidson_reduce(N, 1, H.data(), [&](int64_t i0, int64_t i1, double* acc) {
    acc[0] +=
        detail::davidson_sum(i0, i1, [&](int64_t i) { return V0[i] * AV0[i]; });
  });

  int64_t k = 1;
  bool have_prev = false;
  bool converged = false;
  bool anchored = exact == nullptr or anchor_start;
  double last_res = std::numeric_limits<double>::infinity();
  std::vector<double> res_history;
  size_t iter = 1;

  // Error estimates of the images of the basis vectors (see op_error)
  const bool track_errors = exact != nullptr and op_error > 0.0;
  std::vector<double> image_error(max_m + 1, 0.0);
  image_error[0] = anchor_start ? 0.0 : op_error;
  int64_t last_anchor = 0, close_anchors = 0;

  // Collapse the search space onto U = [y, y_prev orthogonalized against y]
  // (just y when max_m == 2 or without a previous Ritz vector, leaving room
  // for the new direction): [V|AV][0:kk) <- [V|AV][0:k) U, H <- U^T H U and
  // proj <- U^T proj; returns kk. Near convergence y_prev is almost parallel
  // to y, so the projection is repeated to keep U orthonormal to working
  // precision.
  auto collapse_basis = [&](const double* y) -> int64_t {
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
    return kk;
  };
  // H <- U^T H U and proj <- U^T proj
  auto collapse_projections = [&](int64_t kk) {
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, k,
               kk, k, 1., H.data(), max_m, U.data(), k, 0., HU.data(), k);
    blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans, kk,
               kk, k, 1., U.data(), k, HU.data(), k, 0., H.data(), max_m);
    double new_proj[2];
    for (int64_t c = 0; c < kk; ++c)
      new_proj[c] = blas::dot(k, U.data() + c * k, 1, proj.data(), 1);
    std::copy_n(new_proj, kk, proj.data());
  };
  // image_error[0:kk) of the collapsed basis [V|AV][0:k) U
  auto collapse_errors = [&](int64_t kk) {
    double e[2] = {0.0, 0.0};
    for (int64_t c = 0; c < kk; ++c)
      for (int64_t j = 0; j < k; ++j)
        e[c] += std::pow(U[j + c * k] * image_error[j], 2);
    for (int64_t c = 0; c < kk; ++c) image_error[c] = std::sqrt(e[c]);
  };
  auto collapse = [&](const double* y) -> int64_t {
    const int64_t kk = collapse_basis(y);
    // [V|AV][0:kk) <- [V|AV][0:k) U, in place block by block
    davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
      alignas(64) double nv[2][chunk];
      for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
        const int64_t len = std::min(chunk, i1 - c0);
        for (bool image : {false, true}) {
          auto col = [&](int64_t j, auto&& f) {
            if (image)
              avcol(j, c0, f);
            else
              vcol(j, c0, f);
          };
          for (int64_t c = 0; c < kk; ++c) {
            std::fill_n(nv[c], len, 0.0);
            for (int64_t j = 0; j < k; ++j) {
              const double u = U[j + c * k];
              col(j, [&](const auto* b) {
                for (int64_t i = 0; i < len; ++i) nv[c][i] += u * b[i];
              });
            }
          }
          for (int64_t c = 0; c < kk; ++c)
            col(c, [&](auto* b) {
              for (int64_t i = 0; i < len; ++i) store(b + i, nv[c][i]);
            });
        }
      }
    });
    collapse_projections(kk);
    return kk;
  };

  for (;; ++iter) {
    // Rayleigh-Ritz on the k-dimensional search space
    for (int64_t j = 0; j < k; ++j)
      std::copy_n(H.data() + j * max_m, k, Y.data() + j * k);
    if constexpr (measure_overlaps) {
      for (int64_t j = 0; j < k; ++j)
        std::copy_n(G.data() + j * max_m, k, GW.data() + j * k);
      if (lapack::sygv(1, lapack::Job::Vec, lapack::Uplo::Lower, k, Y.data(), k,
                       GW.data(), k, LAM.data()) != 0)
        throw std::runtime_error("restarted_davidson: Rayleigh-Ritz failed");
    } else {
      lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, k, Y.data(), k,
                   LAM.data());
    }
    const double* y = Y.data();
    const double lam = LAM[0];
    double res_error = 0.0;
    if (track_errors) {
      for (int64_t j = 0; j < k; ++j)
        res_error += std::pow(y[j] * image_error[j], 2);
      res_error = std::sqrt(res_error);
    }

    // A full search space is collapsed in the same pass (fused): onto the
    // Ritz vector, whose image the residual needs anyway, and the previous
    // one (see collapse), saving a pass over all of [V|AV]
    const bool fuse = k == max_m;
    const int64_t kk = fuse ? collapse_basis(y) : k;

    // Residual r = AV y - lam V y (not stored), new direction
    // t = -(D - lam)^-1 r in V[k] and V[0:k)^T t, in one pass:
    // sums = [|r|^2, V^T t, |t|^2], then for a fused collapse with measured
    // overlaps [G00, H00, G01, H01, G11, H11] of the stored new [V0, V[1]]
    const int64_t nsums = k + 2 + (measure_overlaps and fuse ? 6 : 0);
    davidson_reduce(
        N, nsums, sums.data(), [&](int64_t i0, int64_t i1, double* acc) {
          alignas(64) double xs[chunk], rs[chunk], x1[chunk], ax1[chunk];
          for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
            const int64_t len = std::min(chunk, i1 - c0);
            std::fill_n(xs, len, 0.0);
            std::fill_n(rs, len, 0.0);
            for (int64_t j = 0; j < k; ++j) {
              const double yj = y[j];
              vcol(j, c0, [&](const auto* v) {
                for (int64_t i = 0; i < len; ++i) xs[i] += yj * v[i];
              });
              avcol(j, c0, [&](const auto* av) {
                for (int64_t i = 0; i < len; ++i) rs[i] += yj * av[i];
              });
            }
            if (fuse and kk > 1) {
              std::fill_n(x1, len, 0.0);
              std::fill_n(ax1, len, 0.0);
              for (int64_t j = 0; j < k; ++j) {
                const double uj = U[j + k];
                vcol(j, c0, [&](const auto* v) {
                  for (int64_t i = 0; i < len; ++i) x1[i] += uj * v[i];
                });
                avcol(j, c0, [&](const auto* av) {
                  for (int64_t i = 0; i < len; ++i) ax1[i] += uj * av[i];
                });
              }
            }
            TC* t = V[k] + c0;
            for (int64_t i = 0; i < len; ++i) {
              double denom = D[c0 + i] - lam;
              if (std::abs(denom) < min_abs_denominator)
                denom = denom >= 0 ? min_abs_denominator : -min_abs_denominator;
              store(t + i, -(rs[i] - lam * xs[i]) / denom);
            }
            acc[0] += detail::davidson_lane_sum(len, [&](int64_t i) {
              const double r = rs[i] - lam * xs[i];
              return r * r;
            });
            acc[k + 1] += detail::davidson_lane_sum(
                len, [&](int64_t i) { return double(t[i]) * double(t[i]); });
            for (int64_t j = 0; j < k; ++j)
              vcol(j, c0, [&](const auto* v) {
                acc[1 + j] += detail::davidson_lane_sum(len, [&](int64_t i) {
                  return double(v[i]) * double(t[i]);
                });
              });
            if (fuse) {
              std::copy_n(xs, len, V0 + c0);
              std::copy_n(rs, len, AV0 + c0);
              if (kk > 1)
                for (int64_t i = 0; i < len; ++i) {
                  store(V[1] + c0 + i, x1[i]);
                  store(AV[1] + c0 + i, ax1[i]);
                }
              if constexpr (measure_overlaps) {
                double* g = acc + k + 2;
                g[0] += detail::davidson_lane_sum(
                    len, [&](int64_t i) { return xs[i] * xs[i]; });
                g[1] += detail::davidson_lane_sum(
                    len, [&](int64_t i) { return xs[i] * rs[i]; });
                if (kk > 1) {
                  const TC* v1 = V[1] + c0;
                  const TC* av1 = AV[1] + c0;
                  g[2] += detail::davidson_lane_sum(
                      len, [&](int64_t i) { return xs[i] * double(v1[i]); });
                  g[3] += detail::davidson_lane_sum(
                      len, [&](int64_t i) { return rs[i] * double(v1[i]); });
                  g[4] += detail::davidson_lane_sum(len, [&](int64_t i) {
                    return double(v1[i]) * double(v1[i]);
                  });
                  g[5] += detail::davidson_lane_sum(len, [&](int64_t i) {
                    return double(v1[i]) * double(av1[i]);
                  });
                }
              }
            }
          }
        });
    std::copy_n(sums.data() + 1, k, proj.data());
    double t_before2 = sums[k + 1];
    if (fuse) {
      // The basis is now [V|AV][0:kk) with the Ritz vector first, and t
      // moves to V[kk]
      collapse_projections(kk);
      if (track_errors) collapse_errors(kk);
      if constexpr (measure_overlaps) {
        const double* g = sums.data() + k + 2;
        G[0] = g[0];
        H[0] = g[1];
        if (kk > 1) {
          G[1] = G[max_m] = g[2];
          H[1] = H[max_m] = g[3];
          G[1 + max_m] = g[4];
          H[1 + max_m] = g[5];
        }
      }
      std::swap(V[k], V[kk]);
      k = kk;
      std::fill_n(Y.data(), k, 0.0);
      Y[0] = 1.0;
      y = Y.data();
      std::fill(y_prev.begin(), y_prev.end(), 0.0);
    }
    if (block and not block->lambda.empty()) {
      // Exact block: r and the diagonal's direction on the block coordinates,
      // the block's direction there, and its effect on V^T t and |t|^2
      const auto& idx = block->index;
      const size_t nc = idx.size(), ne = block->lambda.size();
      std::vector<double> r_b(nc), t_b(nc, 0.0), c(ne);
      for (size_t i = 0; i < nc; ++i) {
        double x = 0.0, ax = 0.0;
        for (int64_t j = 0; j < k; ++j) {
          vcol(j, idx[i], [&](const auto* v) { x += y[j] * v[0]; });
          avcol(j, idx[i], [&](const auto* av) { ax += y[j] * av[0]; });
        }
        r_b[i] = ax - lam * x;
      }
      for (size_t a = 0; a < ne; ++a) {
        const double den = block->lambda[a] - lam;
        c[a] =
            std::abs(den) < block->min_abs_denominator
                ? 0.0
                : -blas::dot(nc, block->Z.data() + a * nc, 1, r_b.data(), 1) /
                      den;
      }
      // t_b = Z c in the same order for every coordinate, so that rows of Z
      // that are exact mirrors (spin-flip sector functions) give exact
      // mirrors (BLAS gives no such guarantee)
      for (size_t a = 0; a < ne; ++a) {
        const double* z = block->Z.data() + a * nc;
        for (size_t i = 0; i < nc; ++i) t_b[i] += z[i] * c[a];
      }
      for (size_t i = 0; i < nc; ++i) {
        TC& t = V[k][idx[i]];
        const double t_old = t;
        store(&t, t_b[i]);
        const double t_new = t;
        for (int64_t j = 0; j < k; ++j)
          vcol(j, idx[i],
               [&](const auto* v) { proj[j] += v[0] * (t_new - t_old); });
        t_before2 += t_new * t_new - t_old * t_old;
      }
    }
    const double res_nrm = std::sqrt(sums[0]);
    last_res = res_nrm;

    logger->info("iter = {:4}, LAM(0) = {:20.12e}, RNORM = {:20.12e}", iter,
                 lam, res_nrm);

    if (anchored and res_nrm + res_error < tol) {
      converged = true;
      break;
    }
    if (iter >= size_t(max_iter)) break;
    res_history.push_back(res_nrm);
    const bool stagnated =
        stagnation_window > 0 and
        int64_t(res_history.size()) > stagnation_window and
        res_nrm >
            0.5 *
                res_history[res_history.size() - 1 - size_t(stagnation_window)];

    const bool reanchor =
        track_errors and anchored and res_error > 0.25 * std::max(res_nrm, tol);
    if (reanchor) {
      close_anchors = int64_t(iter) - last_anchor <= 2 ? close_anchors + 1 : 0;
      if (close_anchors >= 2) {
        logger->info("Davidson re-anchoring cannot keep up");
        break;
      }
    }
    if ((not anchored and (res_nrm < std::max(tol, anchor_tol) or stagnated)) or
        reanchor) {
      // Collapse onto the Ritz vector (and the previous one) and apply the
      // exact operator to the Ritz vector; its coupling to the second basis
      // vector (AV[0]^T V[1]) is then exact as well, while the image of that
      // vector keeps the inexact operator's error at a small coefficient
      anchored = true;
      last_anchor = int64_t(iter);
      res_history.clear();
      const int64_t ka = fuse ? k : collapse(y);
      if (track_errors and not fuse) collapse_errors(ka);
      image_error[0] = 0.0;
      exact->operator_action(1, 1., V0, N, 0., AV0, N);
      // h = [V0^T AV0, AV0^T V[1]], then with measured overlaps
      // [G00, G01, G11, H11] of the stored collapsed vectors
      double h[6] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
      davidson_reduce(
          N, measure_overlaps ? 6 : ka, h,
          [&](int64_t i0, int64_t i1, double* acc) {
            acc[0] += detail::davidson_sum(
                i0, i1, [&](int64_t i) { return V0[i] * AV0[i]; });
            if (ka > 1)
              acc[1] += detail::davidson_sum(
                  i0, i1, [&](int64_t i) { return AV0[i] * double(V[1][i]); });
            if constexpr (measure_overlaps) {
              acc[2] += detail::davidson_sum(
                  i0, i1, [&](int64_t i) { return V0[i] * V0[i]; });
              if (ka > 1) {
                acc[3] += detail::davidson_sum(
                    i0, i1, [&](int64_t i) { return V0[i] * double(V[1][i]); });
                acc[4] += detail::davidson_sum(i0, i1, [&](int64_t i) {
                  return double(V[1][i]) * double(V[1][i]);
                });
                acc[5] += detail::davidson_sum(i0, i1, [&](int64_t i) {
                  return double(V[1][i]) * double(AV[1][i]);
                });
              }
            }
          });
      H[0] = h[0];
      if (ka > 1) H[1] = H[max_m] = h[1];
      if constexpr (measure_overlaps) {
        G[0] = h[2];
        if (ka > 1) {
          G[1] = G[max_m] = h[3];
          G[1 + max_m] = h[4];
          H[1 + max_m] = h[5];
        }
      }
      k = ka;
      std::fill(y_prev.begin(), y_prev.end(), 0.0);
      y_prev[0] = 1.0;
      have_prev = true;
      logger->info("Davidson anchored");
      continue;
    }
    if (stagnated) {
      logger->info("Davidson stagnated");
      break;
    }

    // After a collapse the Ritz vector is the first basis vector (y = e_0)
    std::copy_n(y, k, y_prev.data());
    have_prev = true;

    // Classical Gram-Schmidt: t -= V proj, sums = [V^T t, |t|^2], and a
    // second pass t -= V sums only where the first removed more than about
    // 30 % of |t| (the criterion |t'|^2 <= |t|^2 / 2 of Daniel, Gragg,
    // Kaufman & Stewart, Math. Comp. 30, 772 (1976)), i.e. where
    // cancellation can have spoilt orthogonality. A direction inside the
    // search space (possible only for tiny N) is replaced by unit vectors
    // until one is not.
    double t_nrm = 0.0;
    for (int64_t unit = -1; unit < N; ++unit) {
      if (unit >= 0) {
        davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
          for (int64_t i = i0; i < i1; ++i) V[k][i] = i == unit ? 1 : 0;
        });
        davidson_reduce(
            N, k, proj.data(), [&](int64_t i0, int64_t i1, double* acc) {
              if (unit >= i0 and unit < i1)
                for (int64_t j = 0; j < k; ++j)
                  vcol(j, unit, [&](const auto* v) { acc[j] += v[0]; });
            });
        t_before2 = 1.0;
      }
      davidson_reduce(
          N, k + 1, sums.data(), [&](int64_t i0, int64_t i1, double* acc) {
            for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
              const int64_t len = std::min(chunk, i1 - c0);
              project_out(k, proj.data(), c0, len, acc);
              const TC* t = V[k] + c0;
              acc[k] += detail::davidson_lane_sum(
                  len, [&](int64_t i) { return double(t[i]) * double(t[i]); });
            }
          });
      double t2 = sums[k];
      if (not(t2 > 0.5 * t_before2)) {
        davidson_reduce(N, 1, &t2, [&](int64_t i0, int64_t i1, double* acc) {
          for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
            const int64_t len = std::min(chunk, i1 - c0);
            project_out(k, sums.data(), c0, len, nullptr);
            const TC* t = V[k] + c0;
            acc[0] += detail::davidson_lane_sum(
                len, [&](int64_t i) { return double(t[i]) * double(t[i]); });
          }
        });
      }
      t_nrm = std::sqrt(t2);
      if (t_nrm > 1e-12) break;
    }

    // AV[k] = H t / |t|; then t /= |t| together with the new row of the
    // projected matrix, H(j, k) = AV[j]^T V[k] for j < k and V[k]^T AV[k].
    // Taking the off-diagonal entries from the older images (equal in exact
    // arithmetic) keeps them exact for an exact anchor image when op is
    // inexact: V[j]^T AV[k] would carry the error of op(t) projected on the
    // anchor, which bounds how far the residual can fall.
    op.operator_action(1, 1. / t_nrm, V[k], N, 0., AV[k], N);
    // With measured overlaps sums also gets G(0:k, k) = V[0:k]^T V[k]
    davidson_reduce(
        N, measure_overlaps ? 2 * k + 2 : k + 1, sums.data(),
        [&](int64_t i0, int64_t i1, double* acc) {
          const double f = 1.0 / t_nrm;
          for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
            const int64_t len = std::min(chunk, i1 - c0);
            TC* t = V[k] + c0;
            for (int64_t i = 0; i < len; ++i) store(t + i, f * t[i]);
            for (int64_t j = 0; j < k; ++j)
              avcol(j, c0, [&](const auto* av) {
                acc[j] += detail::davidson_lane_sum(len, [&](int64_t i) {
                  return double(av[i]) * double(t[i]);
                });
              });
            const TC* av = AV[k] + c0;
            acc[k] += detail::davidson_lane_sum(
                len, [&](int64_t i) { return double(t[i]) * double(av[i]); });
            if constexpr (measure_overlaps)
              for (int64_t j = 0; j <= k; ++j)
                vcol(j, c0, [&](const auto* v) {
                  acc[k + 1 + j] += detail::davidson_lane_sum(
                      len,
                      [&](int64_t i) { return double(v[i]) * double(t[i]); });
                });
          }
        });
    for (int64_t j = 0; j <= k; ++j)
      H[j + k * max_m] = H[k + j * max_m] = sums[j];
    if constexpr (measure_overlaps)
      for (int64_t j = 0; j <= k; ++j)
        G[j + k * max_m] = G[k + j * max_m] = sums[k + 1 + j];
    image_error[k] = op_error;
    ++k;
  }

  if (residual_norm) *residual_norm = last_res;
  if (converged)
    logger->info("Davidson Converged!");
  else if (throw_on_failure)
    throw std::runtime_error("Davidson Did Not Converge!");
  else
    logger->info("Davidson stopped after {} iterations", iter);

  // X = V y; with single-precision corrections V is orthonormal only to their
  // precision, so X is normalized
  {
    const double* y = Y.data();
    auto combine = [&](int64_t c0, int64_t len) {
      double* x = X + c0;
      std::fill_n(x, len, 0.0);
      for (int64_t j = 0; j < k; ++j) {
        const double yj = y[j];
        vcol(j, c0, [&](const auto* v) {
          for (int64_t i = 0; i < len; ++i) x[i] += yj * v[i];
        });
      }
    };
    if constexpr (std::is_same_v<TC, double>) {
      davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
        for (int64_t c0 = i0; c0 < i1; c0 += chunk)
          combine(c0, std::min(chunk, i1 - c0));
      });
    } else {
      double x2 = 0.0;
      davidson_reduce(N, 1, &x2, [&](int64_t i0, int64_t i1, double* acc) {
        for (int64_t c0 = i0; c0 < i1; c0 += chunk) {
          const int64_t len = std::min(chunk, i1 - c0);
          combine(c0, len);
          acc[0] += detail::davidson_sum(
              c0, c0 + len, [&](int64_t i) { return X[i] * X[i]; });
        }
      });
      const double f = 1.0 / std::sqrt(x2);
      davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
        for (int64_t i = i0; i < i1; ++i) X[i] *= f;
      });
    }
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

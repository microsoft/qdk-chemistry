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

#include <fstream>
#include <iomanip>
#include <iostream>
#include <lobpcgxx/lobpcg.hpp>
#include <macis/util/mpi.hpp>
#include <random>
#include <sparsexx/matrix_types/csr_matrix.hpp>

#ifdef MACIS_ENABLE_MPI
#include <sparsexx/spblas/pspmbv.hpp>
#endif /* MACIS_ENABLE_MPI */

#include <sparsexx/spblas/spmbv.hpp>

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
 */
inline void gram_schmidt(int64_t N, int64_t K, const double* V_old, int64_t LDV,
                         double* V_new) {
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
  blas::scal(N, 1. / nrm, V_new, 1);
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
 * @return Pair containing (number of iterations, converged eigenvalue)
 */
template <typename Functor>
auto davidson(int64_t N, int64_t max_m, const Functor& op, const double* D,
              double tol, double* X) {
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
    for (auto j = 0; j < N; ++j) {
      R[j] = -R[j] / (D[j] - LAM[0]);
    }

    // Project new vector out form old vectors
    gram_schmidt(N, k, V.data(), N, R);

  }  // Davidson iterations

  if (!converged) throw std::runtime_error("Davidson Did Not Converge!");
  logger->info("Davidson Converged!");

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
 */
inline void p_gram_schmidt(int64_t N_local, int64_t K, const double* V_old,
                           int64_t LDV, double* V_new, MPI_Comm comm) {
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
  // printf("[rank %d] GS DOT %.6e NRM %.6e\n", comm_rank(comm),
  //   dot, nrm);
  blas::scal(N_local, 1. / nrm, V_new, 1);
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
 * @return Pair containing (number of iterations, converged eigenvalue)
 */
template <typename Functor>
auto p_davidson(int64_t N_local, int64_t max_m, const Functor& op,
                const double* D_local, double tol, double* X_local,
                MPI_Comm comm) {
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
    double E1_denom = 0, E1_num = 0;
    for (auto j = 0; j < N_local; ++j) {
      R_local[j] = -R_local[j] / (D_local[j] - LAM[0]);
      E1_num += X_local[j] * R_local[j];
      E1_denom += X_local[j] * X_local[j] / (D_local[j] - LAM[0]);
    }
    E1_denom = allreduce(E1_denom, MPI_SUM, comm);
    E1_num = allreduce(E1_num, MPI_SUM, comm);
    const double E1 = E1_num / E1_denom;

    for (auto j = 0; j < N_local; ++j) {
      R_local[j] += E1 * X_local[j] / (D_local[j] - LAM[0]);
    }

    // Project new vector out form old vectors
    p_gram_schmidt(N_local, k, V_local.data(), N_local, R_local, comm);

  }  // Davidson iterations

  if (!converged) throw std::runtime_error("Davidson Did Not Converge!");
  logger->info("Davidson Converged!");

  return std::make_pair(iter, LAM[0]);
}
#endif /* MACIS_ENABLE_MPI */

}  // namespace macis

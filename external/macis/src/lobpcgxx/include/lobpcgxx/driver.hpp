#pragma once

#include <algorithm>
#include <blas.hh>
#include <functional>
#include <iostream>
#include <lapack.hh>
#include <vector>

#include "cholqr.hpp"
#include "convergence.hpp"
#include "operator.hpp"
#include "rayleigh_ritz.hpp"
#include "residuals.hpp"
#include "rotate_basis.hpp"
#include "type_traits.hpp"

namespace lobpcgxx {

/**
 *  @brief LOBPCG Settings
 */
struct lobpcg_settings {
  double conv_tol = 1e-3;          ///< Residual convergecne tol
  int64_t maxiter = 500;           ///< Maximum number of iterations
  bool print_iter = true;          ///< Whether to print convegence behaviour
  bool track_convergence = false;  ///< Whether to store convegence behaviour
};

/**
 *  @brief Obtain minimum workspace requirement for LOBPCG
 *
 *  @param[in] N Number of rows in matrix
 *  @param[in] K Numner of cols in basis
 *
 *  @returns Minimum workspace requirement
 */
inline constexpr int64_t lobpcg_lwork(int64_t N, int64_t K) {
  return 8 * N * K + 9 * K * K + 5 * K;
}

/**
 *  @brief LOBPCG Driver
 *
 *  Obtain the lowest NR eigenpairs of the standard eigenvalue problem
 *
 *  A*X = X*L
 *
 *  Where A in N-by-N, X is N-by-NR, and L in NR-by-NR with X orthogonal
 *  (eigenvectors) and L diagonal (eigenvalues).
 *
 *  @param[in] settings    LOBPCG runtime settings
 *  @param[in] N           Problem dimension
 *  @param[in] K           Dimension of trial vector space
 *  @param[in] NR          Number of eigenpairs to obtain
 *  @param[in] op_functor  Functor struct to handle the applocation of the
 *                         matrix and preconditioner onto the trial subspace
 *  @param[out] LAMR       Best approximation to the lowest NR eigenpairs of A
 *  @param[in/out] V       On input, the N-by-K eigenvectors guess
 *                         On output, the N-by-NR set of eigenvectors
 * corresponding to LAMR. The trailing K-NR columns also contain eigenvectors
 * which are not fully converged.
 *  @param[in] LDV         Leading dimension of V
 *  @param[out] res        Relative residual norms of the approximate eigenpairs
 *                           on exit.
 *  @param[in] WORK        WORKSPACE (length LWORK)
 *  @param[in] LWORK       Length of WORKSPACE, use lobpcg_lwork to obtain
 * optimal value
 *  @param[in] conv_check  Functor which checks the convergence of the residuals
 *  @param[out] conv       Convergence data
 */
template <typename T>
void lobpcg(const lobpcg_settings& settings, int64_t N, int64_t K, int64_t NR,
            const lobpcg_operator<T> op_functor, detail::real_t<T>* LAMR, T* V,
            int64_t LDV, detail::real_t<T>* res, T* WORK, int64_t& LWORK,
            const lobpcg_convergence_check<T>& conv_check,
            lobpcg_convergence<T>& conv) {
  const int64_t LWORK_MIN = lobpcg_lwork(N, K);

  if (LWORK < 0) {
    LWORK = LWORK_MIN;
    return;
  } else if (LWORK < LWORK_MIN) {
    throw std::runtime_error("LWORK < LWORK_MIN");
  }

  // Check for invalid options
  if (settings.maxiter < 0) throw std::runtime_error("MAXITER < 0");

  if (settings.conv_tol < 0.0) throw std::runtime_error("CONV_TOL < 0");

  // Reserve space for convergence data
  conv.conv_data.clear();
  conv.conv_data.reserve(settings.maxiter);

  T* S = WORK;            // S  = [X  P  W ]
  T* AS = S + 3 * N * K;  // AS = [AS AP AW]

  T* C = AS + 3 * N * K;         // Ritz coefficients
  T* LAM_T = C + 9 * K * K;      // Ritz values
  T* ABS_RES_T = LAM_T + 3 * K;  // Residual norms
  T* REL_RES_T = ABS_RES_T + K;  // Relative residual norms

  T* WORKSPACE = REL_RES_T + K;  // Remainder of the workspace

  detail::real_t<T>* LAM = reinterpret_cast<detail::real_t<T>*>(LAM_T);
  detail::real_t<T>* ABS_RES = reinterpret_cast<detail::real_t<T>*>(ABS_RES_T);
  detail::real_t<T>* REL_RES = reinterpret_cast<detail::real_t<T>*>(REL_RES_T);

  auto* END_WORK = WORK + LWORK;
  auto* END_WORK_MIN = WORKSPACE + 2 * N * K;
  if (END_WORK_MIN > END_WORK)
    throw std::runtime_error("LWORK IS STILL NOT BIG ENOUGH");

  T* X = S;
  T* P = X + N * K;

  T* AX = AS;
  T* AP = AX + N * K;

  // Copy initial guess to X
  lapack::lacpy(lapack::MatrixType::General, N, K, V, LDV, X, N);

  // Obtain initial AX
  op_functor.apply_matrix(N, K, X, N, AX, N);

  // Initial Rayleigh-Ritz -> C, LAM
  rayleigh_ritz(N, K, X, N, AX, N, LAM, C, K);

  // Rotate X / AX by Ritz coefficients
  rotate_basis(N, K, K, X, N, C, K, WORKSPACE);
  rotate_basis(N, K, K, AX, N, C, K, WORKSPACE);

  // Compute residuals and their norms
  residuals(N, K, X, N, AX, N, LAM, V, LDV, ABS_RES, REL_RES);

  // Main LOBPCG loop
  for (auto iter = 0; iter < settings.maxiter; ++iter) {
    // std::cout << "ITER = " << iter << std::endl;
    //  Append to convergence data structure
    if (settings.track_convergence) {
      conv.conv_data.emplace_back();
      conv.conv_data.back().W.resize(K);
      conv.conv_data.back().res.resize(K);
      conv.conv_data.back().rel_res.resize(K);

      blas::copy(K, LAM, 1, conv.conv_data.back().W.data(), 1);
      blas::copy(K, ABS_RES, 1, conv.conv_data.back().res.data(), 1);
      blas::copy(K, REL_RES, 1, conv.conv_data.back().rel_res.data(), 1);
    }

    // Convergence Check
    if (conv_check(NR, ABS_RES, REL_RES, settings.conv_tol)) {
      // Copy data into return structures
      blas::copy(NR, LAM, 1, LAMR, 1);
      blas::copy(NR, REL_RES, 1, res, 1);
      lapack::lacpy(lapack::MatrixType::General, N, K, X, N, V, LDV);
      return;
    }

    // Things are stored [ X W ] in the first iteration
    // to speed up the RR inner product, but [X P W]
    // in all subsequent iterations
    T* W = P + (!!iter) * N * K;
    T* AW = AP + (!!iter) * N * K;

    const int64_t dim_XP = (!!iter + 1) * K;
    const int64_t dim_S = dim_XP + K;

    // Reverse residuals (for some reason...)
    for (auto i = 0; i < K / 2; ++i) {
      blas::swap(N, V + i * LDV, 1, V + (K - 1 - i) * LDV, 1);
    }

    // Apply preconditioner if it exists, else make a copy
    op_functor.apply_preconditioner(N, K, V, LDV, W, N);

    // Project [X P] from W: W <- W - XP*(XP'*X)
    blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
               dim_XP, K, N, T(1.), X, N, W, N, T(0.), WORKSPACE, dim_XP);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N,
               K, dim_XP, T(-1.), X, N, WORKSPACE, dim_XP, T(1.), W, N);

    // Project [X P] from W (again): W <- W - XP*(XP'*X)
    blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
               dim_XP, K, N, T(1.), X, N, W, N, T(0.), WORKSPACE, dim_XP);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N,
               K, dim_XP, T(-1.), X, N, WORKSPACE, dim_XP, T(1.), W, N);

    // Orthogonalize W via CholQR
    cholqr(N, K, W, N);

    if ((iter + 1) % 20 == 0)
      op_functor.apply_matrix(N, dim_S, S, N, AS,
                              N);  // Recalculate AX, AW and AP
    else
      op_functor.apply_matrix(N, K, W, N, AW, N);  // Calculate AW

    // RR
    rayleigh_ritz(N, dim_S, S, N, AS, N, LAM, C, dim_S);

    // HL ortho of C
    {
      std::vector<T> tau(K);
      std::vector<T> C12(K * dim_XP);

      auto* C12_ptr = C + dim_S * K;

      lapack::lacpy(lapack::MatrixType::General, K, dim_XP, C12_ptr, dim_S,
                    C12.data(), K);
      lapack::gelqf(K, dim_XP, C12.data(), K, tau.data());
      lapack::ormlq(lapack::Side::Right, lapack::Op::ConjTrans, dim_S, dim_XP,
                    K, C12.data(), K, tau.data(), C12_ptr, dim_S);
    }

    // Compute new [X, P]
    rotate_basis(N, 2 * K, dim_S, S, N, C, dim_S, WORKSPACE);
    rotate_basis(N, 2 * K, dim_S, AS, N, C, dim_S, WORKSPACE);

    // Form residuals
    residuals(N, K, X, N, AX, N, LAM, V, LDV, ABS_RES, REL_RES);
  }

  // Copy data into return structures just in case user
  // caught the exception
  blas::copy(NR, LAM, 1, LAMR, 1);
  blas::copy(NR, REL_RES, 1, res, 1);
  lapack::lacpy(lapack::MatrixType::General, N, K, X, N, V, LDV);

  throw std::runtime_error("LOBPCG FAILED TO CONVERGE");
}

}  // namespace lobpcgxx

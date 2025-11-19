#pragma once

#include "convergence.hpp"
#include "driver.hpp"

namespace lobpcgxx {

/**
 * @brief LOBPCG eigenvalue solver wrapper with convergence monitoring and
 * user-provided workspace.
 *
 * This is a simplified wrapper around the main LOBPCG driver that uses the
 * default relative residual convergence check. LOBPCG is an iterative method
 * for computing the smallest eigenvalues and corresponding eigenvectors of
 * large sparse symmetric matrices.
 *
 * @tparam T Numerical data type (float, double, std::complex<float>,
 * std::complex<double>)
 * @param settings Algorithm configuration (convergence tolerance, max
 * iterations, etc.)
 * @param N Dimension of the eigenvalue problem (size of matrix A)
 * @param K Size of the block basis (working set of trial vectors)
 * @param NR Number of converged eigenvalues/eigenvectors to compute (typically
 * ≤ K)
 * @param op_functor Operator object containing matrix A and optional
 * preconditioner
 * @param LAMR Output: computed eigenvalues (size NR)
 * @param V Input/Output: initial guess vectors (input), computed eigenvectors
 * (output)
 * @param LDV Leading dimension of matrix V
 * @param res Output: final residual norms for the computed eigenpairs
 * @param WORK Workspace array for internal computations
 * @param LWORK Size of workspace array (use lobpcg_lwork() for minimum size)
 * @param conv Convergence monitoring object for tracking iteration history
 */
template <typename T>
void lobpcg(const lobpcg_settings& settings, int64_t N, int64_t K, int64_t NR,
            const lobpcg_operator<T> op_functor, detail::real_t<T>* LAMR, T* V,
            int64_t LDV, detail::real_t<T>* res, T* WORK, int64_t& LWORK,
            lobpcg_convergence<T>& conv) {
  lobpcg_convergence_check<T> check = lobpcg_relres_convergence_check<T>;
  lobpcg(settings, N, K, NR, op_functor, LAMR, V, LDV, res, WORK, LWORK, check,
         conv);
}

/**
 * @brief LOBPCG eigenvalue solver wrapper with convergence monitoring and
 * automatic workspace.
 *
 * This wrapper automatically allocates the required workspace memory and uses
 * the default relative residual convergence check. Convenient for users who
 * don't want to manage workspace allocation manually.
 *
 * @tparam T Numerical data type (float, double, std::complex<float>,
 * std::complex<double>)
 * @param settings Algorithm configuration (convergence tolerance, max
 * iterations, etc.)
 * @param N Dimension of the eigenvalue problem (size of matrix A)
 * @param K Size of the block basis (working set of trial vectors)
 * @param NR Number of converged eigenvalues/eigenvectors to compute (typically
 * ≤ K)
 * @param op_functor Operator object containing matrix A and optional
 * preconditioner
 * @param LAMR Output: computed eigenvalues (size NR)
 * @param V Input/Output: initial guess vectors (input), computed eigenvectors
 * (output)
 * @param LDV Leading dimension of matrix V
 * @param res Output: final residual norms for the computed eigenpairs
 * @param conv Convergence monitoring object for tracking iteration history
 */
template <typename T>
void lobpcg(const lobpcg_settings& settings, int64_t N, int64_t K, int64_t NR,
            const lobpcg_operator<T> op_functor, detail::real_t<T>* LAMR, T* V,
            int64_t LDV, detail::real_t<T>* res, lobpcg_convergence<T>& conv) {
  auto LWORK = lobpcg_lwork(N, K);
  std::vector<T> work(LWORK);

  lobpcg(settings, N, K, NR, op_functor, LAMR, V, LDV, res, work.data(), LWORK,
         conv);
}

/**
 * @brief LOBPCG eigenvalue solver wrapper with user workspace and default
 * convergence.
 *
 * This wrapper uses default convergence settings (no tracking) but requires the
 * user to provide workspace memory. Useful when you need control over memory
 * allocation but don't need convergence monitoring.
 *
 * @tparam T Numerical data type (float, double, std::complex<float>,
 * std::complex<double>)
 * @param settings Algorithm configuration (convergence tolerance, max
 * iterations, etc.)
 * @param N Dimension of the eigenvalue problem (size of matrix A)
 * @param K Size of the block basis (working set of trial vectors)
 * @param NR Number of converged eigenvalues/eigenvectors to compute (typically
 * ≤ K)
 * @param op_functor Operator object containing matrix A and optional
 * preconditioner
 * @param LAMR Output: computed eigenvalues (size NR)
 * @param V Input/Output: initial guess vectors (input), computed eigenvectors
 * (output)
 * @param LDV Leading dimension of matrix V
 * @param res Output: final residual norms for the computed eigenpairs
 * @param WORK Workspace array for internal computations
 * @param LWORK Size of workspace array (use lobpcg_lwork() for minimum size)
 */
template <typename T>
void lobpcg(const lobpcg_settings& settings, int64_t N, int64_t K, int64_t NR,
            const lobpcg_operator<T> op_functor, detail::real_t<T>* LAMR, T* V,
            int64_t LDV, detail::real_t<T>* res, T* WORK, int64_t& LWORK) {
  lobpcg_convergence<T> conv;
  lobpcg(settings, N, K, NR, op_functor, LAMR, V, LDV, res, WORK, LWORK, conv);
}

/**
 * @brief Most simplified LOBPCG eigenvalue solver wrapper.
 *
 * This is the most user-friendly interface that automatically handles workspace
 * allocation and uses default convergence settings. Best choice for quick
 * prototyping or when you don't need detailed convergence monitoring.
 *
 * @tparam T Numerical data type (float, double, std::complex<float>,
 * std::complex<double>)
 * @param settings Algorithm configuration (convergence tolerance, max
 * iterations, etc.)
 * @param N Dimension of the eigenvalue problem (size of matrix A)
 * @param K Size of the block basis (working set of trial vectors)
 * @param NR Number of converged eigenvalues/eigenvectors to compute (typically
 * ≤ K)
 * @param op_functor Operator object containing matrix A and optional
 * preconditioner
 * @param LAMR Output: computed eigenvalues (size NR)
 * @param V Input/Output: initial guess vectors (input), computed eigenvectors
 * (output)
 * @param LDV Leading dimension of matrix V
 * @param res Output: final residual norms for the computed eigenpairs
 */
template <typename T>
void lobpcg(const lobpcg_settings& settings, int64_t N, int64_t K, int64_t NR,
            const lobpcg_operator<T> op_functor, detail::real_t<T>* LAMR, T* V,
            int64_t LDV, detail::real_t<T>* res) {
  auto LWORK = lobpcg_lwork(N, K);
  std::vector<T> work(LWORK);

  lobpcg(settings, N, K, NR, op_functor, LAMR, V, LDV, res, work.data(), LWORK);
}

}  // namespace lobpcgxx

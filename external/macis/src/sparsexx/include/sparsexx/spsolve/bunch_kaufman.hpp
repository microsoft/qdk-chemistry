/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once

#include <iostream>
#include <memory>
#include <stdexcept>

namespace sparsexx::spsolve {

namespace detail {
/**
 * @brief Abstract base class for Bunch-Kaufman factorization implementations.
 *
 * This class defines the interface for different Bunch-Kaufman factorization.
 *
 * The Bunch-Kaufman factorization is used for symmetric indefinite matrices
 * and provides a stable decomposition that preserves the inertia of the matrix.
 *
 * @tparam SpMatType The sparse matrix type to be factorized
 */
template <typename SpMatType>
struct bunch_kaufman_pimpl {
  using value_type =
      typename SpMatType::value_type;  ///< Type of matrix elements

  /**
   * @brief Performs Bunch-Kaufman factorization of the input matrix.
   *
   * Computes the Bunch-Kaufman factorization of a symmetric indefinite matrix
   * A. The factorization has the form P*A*P^T = L*D*L^T, where P is a
   * permutation matrix, L is a unit lower triangular matrix, and D is a block
   * diagonal matrix with 1x1 and 2x2 blocks.
   *
   * @param A The symmetric sparse matrix to factorize
   * @throws std::runtime_error If factorization fails or matrix is not suitable
   */
  virtual void factorize(const SpMatType&) = 0;
  /**
   * @brief Solves the linear system A*X = B using the computed factorization.
   *
   * Uses the previously computed Bunch-Kaufman factorization to solve the
   * linear system A*X = B for multiple right-hand sides. The solution is
   * stored in a separate output array.
   *
   * @param NRHS Number of right-hand side vectors
   * @param B Input right-hand side matrix stored in column-major format
   * @param LDB Leading dimension of matrix B (must be >= A.m())
   * @param X Output solution matrix stored in column-major format
   * @param LDX Leading dimension of matrix X (must be >= A.m())
   * @throws std::runtime_error If factorization has not been computed
   */
  virtual void solve(int64_t NRHS, const value_type* B, int64_t LDB,
                     value_type* X, int64_t LDX) = 0;

  /**
   * @brief Solves the linear system A*X = B in-place using the computed
   * factorization.
   *
   * Uses the previously computed Bunch-Kaufman factorization to solve the
   * linear system A*X = B for multiple right-hand sides. The solution
   * overwrites the input right-hand side matrix.
   *
   * @param NRHS Number of right-hand side vectors
   * @param B Input/output matrix: right-hand sides on input, solutions on
   * output
   * @param LDB Leading dimension of matrix B (must be >= A.m())
   * @throws std::runtime_error If factorization has not been computed
   */
  virtual void solve(int64_t NRHS, value_type* B, int64_t LDB) = 0;

  /**
   * @brief Returns the inertia of the factorized matrix.
   *
   * The inertia of a symmetric matrix consists of the number of positive,
   * negative, and zero eigenvalues. This information is available after
   * the Bunch-Kaufman factorization and is useful for determining the
   * definiteness properties of the matrix.
   *
   * @return std::tuple<int64_t, int64_t, int64_t> Tuple containing
   *         (positive eigenvalues, negative eigenvalues, zero eigenvalues)
   * @throws std::runtime_error If factorization has not been computed
   */
  virtual std::tuple<int64_t, int64_t, int64_t> get_inertia() = 0;

  /**
   * @brief Virtual destructor for proper cleanup of derived classes.
   */
  virtual ~bunch_kaufman_pimpl() noexcept = default;
};

/**
 * @brief Exception thrown when operations are attempted on uninitialized
 * Bunch-Kaufman instances.
 *
 * This exception is thrown when trying to perform factorization, solve
 * operations, or query inertia on a Bunch-Kaufman solver that has not been
 * properly initialized with a concrete implementation.
 */
struct bunch_kaufman_init_exception : public std::exception {
  /**
   * @brief Returns the exception message.
   *
   * @return const char* Descriptive error message
   */
  const char* what() const throw() {
    return "Bunch-Kaufman Instance Has Not Been Initialized";
  }
};
}  // namespace detail

/**
 * @brief High-level interface for Bunch-Kaufman factorization of symmetric
 * indefinite matrices.
 *
 * This class provides a user-friendly interface to the Bunch-Kaufman
 * factorization, which is suitable for solving linear systems with symmetric
 * indefinite matrices.
 *
 * The Bunch-Kaufman factorization computes P*A*P^T = L*D*L^T where:
 * - P is a permutation matrix
 * - L is unit lower triangular
 * - D is block diagonal with 1x1 and 2x2 blocks
 *
 * Example usage:
 * @code
 * auto solver = make_bunch_kaufman_solver(matrix);
 * solver.factorize(A);
 * solver.solve(nrhs, B, ldb, X, ldx);
 * auto [pos, neg, zero] = solver.get_inertia();
 * @endcode
 *
 * @tparam SpMatType The sparse matrix type (must be symmetric)
 */
template <typename SpMatType>
class bunch_kaufman {
 public:
  using value_type =
      typename SpMatType::value_type;  ///< Type of matrix elements

 protected:
  using pimpl_type = std::unique_ptr<
      detail::bunch_kaufman_pimpl<SpMatType> >;  ///< PIMPL pointer type
  pimpl_type pimpl_;  ///< Pointer to implementation object

 public:
  /**
   * @brief Virtual destructor for proper cleanup.
   */
  virtual ~bunch_kaufman() noexcept = default;

  /**
   * @brief Constructs a Bunch-Kaufman solver with a specific implementation.
   *
   * @param pimpl Unique pointer to a concrete implementation
   */
  bunch_kaufman(pimpl_type&& pimpl) : pimpl_(std::move(pimpl)) {}

  /**
   * @brief Default constructor creates an uninitialized solver.
   *
   * The solver must be assigned a valid implementation before use.
   */
  bunch_kaufman() : bunch_kaufman(nullptr) {}

  /**
   * @brief Copy constructor is deleted to prevent issues with PIMPL.
   */
  bunch_kaufman(const bunch_kaufman&) = delete;

  /**
   * @brief Move constructor transfers ownership of the implementation.
   */
  bunch_kaufman(bunch_kaufman&&) noexcept = default;

  /**
   * @brief Copy assignment operator is deleted to prevent issues with PIMPL.
   */
  bunch_kaufman& operator=(const bunch_kaufman&) = delete;

  /**
   * @brief Move assignment operator transfers ownership of the implementation.
   */
  bunch_kaufman& operator=(bunch_kaufman&&) noexcept = default;

  /**
   * @brief Computes the Bunch-Kaufman factorization of the input matrix.
   *
   * Performs the numerical factorization P*A*P^T = L*D*L^T where P is a
   * permutation matrix, L is unit lower triangular, and D is block diagonal.
   * This must be called before any solve operations.
   *
   * @param A The symmetric sparse matrix to factorize
   * @throws bunch_kaufman_init_exception If solver is not initialized
   * @throws std::runtime_error If factorization fails
   */
  void factorize(const SpMatType& A) {
    if (pimpl_)
      pimpl_->factorize(A);
    else
      throw detail::bunch_kaufman_init_exception();
  }

  /**
   * @brief Solves A*X = B using the computed factorization.
   *
   * Uses the previously computed Bunch-Kaufman factorization to solve
   * the linear system for multiple right-hand sides. The solution is
   * stored in a separate output array.
   *
   * @param K Number of right-hand side vectors (columns)
   * @param B Pointer to input right-hand side matrix (column-major)
   * @param LDB Leading dimension of matrix B (must be >= matrix size)
   * @param X Pointer to output solution matrix (column-major)
   * @param LDX Leading dimension of matrix X (must be >= matrix size)
   * @throws bunch_kaufman_init_exception If solver is not initialized
   * @throws std::runtime_error If factorization has not been computed
   */
  void solve(int64_t K, const value_type* B, int64_t LDB, value_type* X,
             int64_t LDX) {
    if (pimpl_)
      pimpl_->solve(K, B, LDB, X, LDX);
    else
      throw detail::bunch_kaufman_init_exception();
  };

  /**
   * @brief Solves A*X = B in-place using the computed factorization.
   *
   * Uses the previously computed Bunch-Kaufman factorization to solve
   * the linear system for multiple right-hand sides. The solution overwrites
   * the input matrix.
   *
   * @param K Number of right-hand side vectors (columns)
   * @param B Pointer to input/output matrix: RHS on input, solution on output
   * @param LDB Leading dimension of matrix B (must be >= matrix size)
   * @throws bunch_kaufman_init_exception If solver is not initialized
   * @throws std::runtime_error If factorization has not been computed
   */
  void solve(int64_t K, value_type* B, int64_t LDB) {
    if (pimpl_)
      pimpl_->solve(K, B, LDB);
    else
      throw detail::bunch_kaufman_init_exception();
  };

  /**
   * @brief Returns the inertia of the factorized matrix.
   *
   * The inertia consists of the number of positive, negative, and zero
   * eigenvalues of the symmetric matrix. This information is computed
   * during the Bunch-Kaufman factorization and is useful for:
   * - Determining matrix definiteness
   * - Checking solvability of optimization problems
   * - Verifying numerical stability
   *
   * @return std::tuple<int64_t, int64_t, int64_t> Tuple containing
   *         (number of positive eigenvalues, number of negative eigenvalues,
   * number of zero eigenvalues)
   * @throws bunch_kaufman_init_exception If solver is not initialized
   * @throws std::runtime_error If factorization has not been computed
   */
  std::tuple<int64_t, int64_t, int64_t> get_inertia() {
    if (pimpl_)
      return pimpl_->get_inertia();
    else
      throw detail::bunch_kaufman_init_exception();
  }
};  // class bunch_kaufman

}  // namespace sparsexx::spsolve

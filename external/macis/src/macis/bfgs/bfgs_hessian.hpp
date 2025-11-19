/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <iostream>
#include <memory>

#include "bfgs_traits.hpp"

namespace bfgs {

/**
 * @brief Base BFGS Hessian approximation class implementing the limited-memory
 * BFGS algorithm
 *
 * This class maintains a limited-memory approximation of the inverse Hessian
 * matrix using the BFGS update formula. It stores vectors from recent
 * iterations to compute the Hessian-vector product efficiently without storing
 * the full matrix.
 *
 * @tparam Functor Function object type that provides vector operations (dot,
 * axpy, etc.)
 */
template <typename Functor>
struct BFGSHessian {
  /// Type alias for the argument/vector type used by the functor
  using arg_type = detail::arg_type_t<Functor>;

  /// Storage for step vectors (s_k = x_{k+1} - x_k) from recent iterations
  std::vector<arg_type> sk;

  /// Storage for gradient difference vectors (y_k = grad_{k+1} - grad_k) from
  /// recent iterations
  std::vector<arg_type> yk;

  /// Storage for reciprocal values (1 / y_k^T s_k) used in BFGS updates
  std::vector<double> rhok;

  /// Virtual destructor for proper inheritance
  virtual ~BFGSHessian() noexcept = default;

  /**
   * @brief Updates the BFGS approximation with new step and gradient difference
   * vectors
   *
   * Adds new vectors to the limited-memory storage and computes the
   * corresponding scaling factor for the BFGS update formula.
   *
   * @param s Step vector (x_{k+1} - x_k)
   * @param y Gradient difference vector (grad_{k+1} - grad_k)
   */
  virtual void update(const arg_type& s, const arg_type& y) {
    const auto ys = Functor::dot(y, s);
    rhok.emplace_back(1. / ys);

    sk.emplace_back(s);
    yk.emplace_back(y);
  }

  /**
   * @brief Applies the initial Hessian approximation H_0 to a vector
   *
   * Base implementation does nothing (identity matrix). Derived classes
   * can override to provide custom initial Hessian approximations.
   *
   * @param x Vector to apply H_0 to (modified in-place)
   */
  virtual void apply_H0(arg_type& x) {}  // Null call

  /**
   * @brief Applies the BFGS Hessian approximation to a vector using the
   * two-loop recursion
   *
   * Implements the efficient two-loop recursion algorithm to compute H_k * x
   * without explicitly forming the Hessian matrix.
   *
   * @param x Input vector
   * @return Result of H_k * x
   */
  arg_type apply(const arg_type& x) {
    arg_type q = x;
    const int64_t nk = sk.size();

    std::vector<double> alpha(nk);
    for (int64_t i = nk - 1; i >= 0; i--) {
      alpha[i] = Functor::dot(sk[i], q) * rhok[i];
      Functor::axpy(-alpha[i], yk[i], q);
    }
    apply_H0(q);
    for (int64_t i = 0; i < nk; ++i) {
      const auto beta = Functor::dot(yk[i], q) * rhok[i];
      Functor::axpy(alpha[i] - beta, sk[i], q);
    }
    return q;
  }
};

/**
 * @brief Factory function to create a basic BFGS Hessian with identity initial
 * approximation
 *
 * Creates a standard BFGS Hessian approximation that uses the identity matrix
 * as the initial Hessian approximation (H_0 = I).
 *
 * @tparam Functor Function object type that provides vector operations
 * @return Unique pointer to a BFGSHessian instance
 */
template <typename Functor>
std::unique_ptr<BFGSHessian<Functor>> make_identity_hessian() {
  return std::make_unique<BFGSHessian<Functor>>();
}

/**
 * @brief BFGS Hessian with dynamically updated scaling of the initial
 * approximation
 *
 * This variant implements the scaling strategy from Shanno & Phua (1980)
 * doi:10.1007/BF01589116, where the initial Hessian approximation is scaled
 * by a factor that is updated at each iteration based on the current gradient
 * difference vector.
 *
 * @tparam Functor Function object type that provides vector operations
 */
template <typename Functor>
struct UpdatedScaledBFGSHessian : public BFGSHessian<Functor> {
  using base_type = BFGSHessian<Functor>;
  using arg_type = typename base_type::arg_type;

  /// Infinity constant for numerical computations
  static constexpr double inf = std::numeric_limits<double>::infinity();

  /// Current scaling factor for the initial Hessian approximation (updated each
  /// iteration)
  double gamma_k = 1.0;

  /**
   * @brief Updates the BFGS approximation and computes new scaling parameter
   *
   * Updates the base BFGS storage and computes the new scaling factor
   * gamma_k = ||y_k||^2 * rho_k following Shanno & Phua (1980).
   *
   * @param s Step vector (x_{k+1} - x_k)
   * @param y Gradient difference vector (grad_{k+1} - grad_k)
   */
  void update(const arg_type& s, const arg_type& y) override final {
    base_type::update(s, y);
    const auto y_nrm = Functor::norm(y);
    gamma_k = (y_nrm * y_nrm) * this->rhok.back();
  }

  /**
   * @brief Applies the scaled initial Hessian approximation H_0 = (1/gamma_k) *
   * I
   *
   * Scales the input vector by 1/gamma_k to implement the adaptive scaling
   * of the initial identity matrix approximation.
   *
   * @param x Vector to scale (modified in-place)
   */
  void apply_H0(arg_type& x) override final { Functor::scal(1. / gamma_k, x); }
};

/**
 * @brief Factory function to create a BFGS Hessian with dynamically updated
 * scaling
 *
 * Creates a BFGS Hessian that implements the adaptive scaling strategy from
 * Shanno & Phua (1980), where the initial approximation scaling factor is
 * updated at each iteration.
 *
 * @tparam Functor Function object type that provides vector operations
 * @return Unique pointer to an UpdatedScaledBFGSHessian instance
 */
template <typename Functor>
std::unique_ptr<BFGSHessian<Functor>> make_updated_scaled_hessian() {
  return std::make_unique<UpdatedScaledBFGSHessian<Functor>>();
}

/**
 * @brief BFGS Hessian with static scaling of the initial approximation
 *
 * This variant implements a static scaling strategy where the initial Hessian
 * approximation is scaled by a factor computed only once from the first
 * gradient difference vector, then kept constant throughout the optimization.
 *
 * @tparam Functor Function object type that provides vector operations
 */
template <typename Functor>
struct StaticScaledBFGSHessian : public BFGSHessian<Functor> {
  using base_type = BFGSHessian<Functor>;
  using arg_type = typename base_type::arg_type;

  /// Static scaling factor for initial Hessian (computed once from first
  /// update)
  std::optional<double> gamma_0;

  /**
   * @brief Updates the BFGS approximation and sets static scaling parameter on
   * first call
   *
   * Updates the base BFGS storage and computes the static scaling factor
   * gamma_0 = ||y_0||^2 * rho_0 only on the first update, following Shanno &
   * Phua (1980).
   *
   * @param s Step vector (x_{k+1} - x_k)
   * @param y Gradient difference vector (grad_{k+1} - grad_k)
   */
  void update(const arg_type& s, const arg_type& y) override final {
    base_type::update(s, y);
    if (!gamma_0.has_value()) {
      const auto y_nrm = Functor::norm(y);
      gamma_0 = (y_nrm * y_nrm) * this->rhok.back();
    }
  }

  /**
   * @brief Applies the statically scaled initial Hessian approximation H_0 =
   * (1/gamma_0) * I
   *
   * Scales the input vector by 1/gamma_0 if the scaling factor has been
   * computed, otherwise applies identity (no scaling).
   *
   * @param x Vector to scale (modified in-place)
   */
  void apply_H0(arg_type& x) override final {
    if (gamma_0.has_value()) Functor::scal(1. / gamma_0.value(), x);
  }
};

/**
 * @brief Factory function to create a BFGS Hessian with static scaling
 *
 * Creates a BFGS Hessian that uses static scaling of the initial approximation,
 * where the scaling factor is computed once from the first gradient difference
 * and then kept constant.
 *
 * @tparam Functor Function object type that provides vector operations
 * @return Unique pointer to a StaticScaledBFGSHessian instance
 */
template <typename Functor>
std::unique_ptr<BFGSHessian<Functor>> make_static_scaled_hessian() {
  return std::make_unique<StaticScaledBFGSHessian<Functor>>();
}

/**
 * @brief BFGS Hessian with custom runtime-initialized initial approximation
 *
 * This variant allows users to provide a custom function object that defines
 * the initial Hessian approximation H_0. The function is stored and called
 * whenever the initial approximation needs to be applied.
 *
 * @tparam Functor Function object type that provides vector operations
 */
template <typename Functor>
struct RuntimeInitializedBFGSHessian : public BFGSHessian<Functor> {
  using base_type = BFGSHessian<Functor>;
  using arg_type = typename base_type::arg_type;
  /// Function type for custom initial Hessian application
  using op_type = std::function<void(arg_type&)>;

  /// User-provided function object for applying the initial Hessian
  /// approximation
  op_type m_H0;

  /// Deleted default constructor - requires explicit initialization
  RuntimeInitializedBFGSHessian() = delete;

  /**
   * @brief Constructs with a custom initial Hessian function
   * @param op Function object that applies H_0 to a vector (in-place
   * modification)
   */
  RuntimeInitializedBFGSHessian(const op_type& op) : m_H0(op) {}

  /**
   * @brief Applies the user-defined initial Hessian approximation
   * @param x Vector to apply H_0 to (modified in-place)
   */
  void apply_H0(arg_type& x) override final { m_H0(x); }
};

/**
 * @brief BFGS Hessian with diagonal initial approximation
 *
 * This variant uses a diagonal matrix as the initial Hessian approximation H_0,
 * where the diagonal elements are provided by the user. This is useful when
 * prior knowledge about the problem's scaling is available.
 *
 * @tparam Functor Function object type that provides vector operations
 */
template <typename Functor>
struct DiagInitializedBFGSHessian : public BFGSHessian<Functor> {
  using base_type = BFGSHessian<Functor>;
  using arg_type = typename base_type::arg_type;
  /// Function type for operations (unused in this implementation)
  using op_type = std::function<void(arg_type&)>;

  /// Inverse of diagonal elements for efficient application (stores 1/D_ii)
  std::vector<double> inv_diag;

  /// Deleted default constructor - requires explicit diagonal initialization
  DiagInitializedBFGSHessian() = delete;

  /**
   * @brief Constructs with diagonal initial Hessian from array
   *
   * Takes diagonal elements D and stores their reciprocals for efficient
   * application of the inverse H_0^{-1} = diag(1/D_0, 1/D_1, ...).
   *
   * @param n Size of the diagonal
   * @param D Pointer to array of diagonal elements
   */
  DiagInitializedBFGSHessian(size_t n, double* D) {
    inv_diag.resize(n);
    std::transform(D, D + n, inv_diag.begin(), [](auto x) { return 1. / x; });
  }

  /**
   * @brief Applies the diagonal initial Hessian approximation element-wise
   *
   * Multiplies each element of the input vector by the corresponding
   * inverse diagonal element: x_i *= (1/D_ii).
   *
   * @param x Vector to apply diagonal scaling to (modified in-place)
   */
  void apply_H0(arg_type& x) override final {
    for (size_t i = 0; i < inv_diag.size(); ++i) {
      x[i] *= inv_diag[i];
    }
  }
};

}  // namespace bfgs

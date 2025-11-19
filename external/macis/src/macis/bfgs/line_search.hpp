/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <spdlog/spdlog.h>

#include <iostream>

#include "bfgs_traits.hpp"

namespace bfgs {

/**
 * @brief Backtracking line search
 *
 * Implements a backtracking line search algorithm that finds an acceptable step
 * size along a given search direction.
 *
 * The algorithm starts with a given step size and reduces it by a factor (tau)
 * until the acceptance conditions are satisfied or maximum iterations are
 * reached.
 *
 * @tparam Functor Function object type that provides:
 *         - eval(x): Function evaluation
 *         - grad(x): Gradient computation
 *         - Static methods: dot(), axpy()
 *
 * @param op Function object implementing the objective function and gradient
 * @param x0 Starting point for the line search
 * @param p Search direction (must be a descent direction)
 * @param step Initial step size (will be modified to optimal step)
 * @param x Output: New point x0 + step * p (modified)
 * @param fx Output: Function value at the new point (modified)
 * @param gfx Output: Gradient at the new point (modified)
 *
 * @throws std::runtime_error If search direction is not descent, step is
 * non-positive, or line search fails to find acceptable step
 *
 * @note The search direction p must satisfy dot(p, grad(x0)) < 0 (descent
 * condition)
 * @note Step size is modified in-place and represents the final accepted step
 * @note Uses adaptive parameters: c1=0.5/1e-4, c2=0.8, tau=0.5 for step
 * reduction
 */
template <typename Functor>
void backtracking_line_search(Functor& op,
                              const detail::arg_type_t<Functor>& x0,
                              const detail::arg_type_t<Functor>& p,
                              detail::ret_type_t<Functor>& step,
                              detail::arg_type_t<Functor>& x,
                              detail::ret_type_t<Functor>& fx,
                              detail::arg_type_t<Functor>& gfx) {
  using arg_type = detail::arg_type_t<Functor>;
  using ret_type = detail::ret_type_t<Functor>;

  // Evaluate function and gradient at starting point
  const auto fx0 = op.eval(x0);
  const auto gfx0 = op.grad(x0);
  const auto dg0 = Functor::dot(p, gfx0);  // Directional derivative

  // Validate inputs: ensure descent direction and positive step
  if (dg0 > 0)
    throw std::runtime_error("Step will increase objective function");
  if (step <= 0.0) throw std::runtime_error("Step must be positive");

// Algorithm parameters - conditional compilation for different acceptance
// criteria
#define USE_MODIFIED_ARMIJO
#ifdef USE_MODIFIED_ARMIJO
  constexpr auto c1 =
      0.5;  // Armijo parameter (relaxed for modified validation)
#else
  constexpr auto c1 = 1e-4;  // Standard Armijo parameter
#endif                       /* USE_MODIFIED_ARMIJO */
  constexpr auto c2 = 0.8;   // Wolfe curvature parameter
  constexpr auto tau = 0.5;  // Step reduction factor /*1./1.618033988749894;*/

  // Precompute test thresholds
  const auto t_armijo = -dg0 * c1;           // Armijo threshold
  const auto t_swolfe = c2 * std::abs(dg0);  // Strong Wolfe threshold

  // Define acceptance test functions
  auto test_modified = [t_armijo, fx0](auto _fx, auto _s) -> bool {
    return ((fx0 - _fx) - _s * t_armijo > -1e-6);  // Modified Armijo condition
  };
  auto test_armijo = [t_armijo, fx0](auto _fx, auto _s) -> bool {
    return (_fx < (fx0 + _s * t_armijo));  // Standard Armijo condition
  };
  auto test_swolfe = [t_swolfe, &p](auto _gfx) -> bool {
    return std::abs(Functor::dot(p, _gfx)) <
           t_swolfe;  // Strong Wolfe condition
  };

  auto logger = spdlog::get("line_search");
  logger->debug("");
  logger->debug("Starting Backtracking Line Search");
  logger->debug(" F(X0) = {:15.12f}, dg0 = {:15.12e}", fx, dg0);
  logger->debug(
      "tau = {:.4f}, c1 = {:.4f}, c2 = {:4f}, t_armijo = {:10.7e}, t_swolfe = "
      "{:10.7e}",
      tau, c1, c2, t_armijo, t_swolfe);

  const std::string fmt_str =
      "iter = {:4}, F(X) = {:15.12f}, dF = {:20.12e}, S = {:10.6e}";

  // Initialize search with full step
  step = 1.0;
  x = x0;
  Functor::axpy(step, p, x);  // x = x0 + step * p
  fx = op.eval(x);
  logger->debug(fmt_str, 0, fx, fx - fx0, step);

  size_t max_iter = 100;
  // Main backtracking loop - reduce step until acceptance criteria are met
  for (size_t iter = 0; iter < max_iter; ++iter) {
#ifdef USE_MODIFIED_ARMIJO
    // Modified validation: simplified acceptance test
    if (test_modified(fx, step)) break;
    step *= tau;  // Reduce step size
#else
    // Standard Armijo-Wolfe conditions
    if (test_armijo(fx, step)) {
      logger->debug("  * armijo condition met");
      gfx = op.grad(x);
      if (test_swolfe(gfx)) {
        logger->debug("  * wolfe condition met");
        break;  // Both conditions satisfied - accept step
      }

      logger->debug("  * wolfe condition not met: increasing step by 2.1");
      step *= 2.1;  // Increase step if Armijo satisfied but not Wolfe
    } else {
      logger->debug("  * armijo condition not met: decreasing step by {}", tau);
      step *= tau;  // Decrease step if Armijo not satisfied
    }
#endif /* USE_MODIFIED_ARMIJO */
    // Compute new trial point
    x = x0;
    Functor::axpy(step, p, x);
    fx = op.eval(x);
    logger->debug(fmt_str, iter + 1, fx, (fx - fx0), step);
  }

  // Final validation: ensure function decrease and reasonable step size
  if (fx - fx0 > 0 or step < 1e-6)
    throw std::runtime_error("Line Search Failed");

  gfx = op.grad(x);  // Compute final gradient for output
  logger->debug("Line Search Converged with S = {:10.6e}", step);
  logger->debug("");
}

/**
 * @brief Nocedal-Wright line search with strong Wolfe conditions
 *
 * Implements the line search algorithm described in "Numerical Optimization"
 * by Nocedal and Wright. This is a two-phase algorithm:
 * 1. Bracketing phase: Find an interval containing acceptable step sizes
 * 2. Zoom phase: Refine the interval using interpolation to find optimal step
 *
 * The algorithm satisfies both Armijo (sufficient decrease) and strong Wolfe
 * (curvature) conditions, ensuring global convergence properties for
 * quasi-Newton methods like BFGS.
 *
 * @tparam Functor Function object type that provides:
 *         - eval(x): Function evaluation
 *         - grad(x): Gradient computation
 *         - Static methods: dot(), axpy()
 *
 * @param op Function object implementing the objective function and gradient
 * @param x0 Starting point for the line search
 * @param p Search direction (must be a descent direction)
 * @param step Initial step size (will be modified to optimal step)
 * @param x Output: New point x0 + step * p (modified)
 * @param fx Output: Function value at the new point (modified)
 * @param gfx Output: Gradient at the new point (modified)
 *
 * @throws std::logic_error If search direction is not descent
 * @throws std::runtime_error If step is non-positive or line search fails
 *
 * @note Uses standard parameters: c1=1e-4 (Armijo), c2=0.9 (Wolfe),
 * expansion=2.0
 * @note More robust than backtracking but computationally more expensive
 */
template <typename Functor>
void nocedal_wright_line_search(Functor& op,
                                const detail::arg_type_t<Functor>& x0,
                                const detail::arg_type_t<Functor>& p,
                                detail::ret_type_t<Functor>& step,
                                detail::arg_type_t<Functor>& x,
                                detail::ret_type_t<Functor>& fx,
                                detail::arg_type_t<Functor>& gfx) {
  using arg_type = detail::arg_type_t<Functor>;
  using ret_type = detail::ret_type_t<Functor>;

  const auto fx0 = fx;                    // Starting function value
  const auto dgi = Functor::dot(p, gfx);  // Initial directional derivative

  // Validate inputs
  if (dgi > 0)
    throw std::logic_error(
        "the moving direction increases the objective function value");
  if (step <= 0.0) throw std::runtime_error("Step must be positive");

  // Algorithm parameters (standard Nocedal-Wright values)
  constexpr auto c1 = 1e-4;        // Armijo condition parameter
  constexpr auto c2 = 0.9;         // Strong Wolfe condition parameter
  constexpr auto expansion = 2.0;  // Step expansion factor in bracketing phase

  // Precompute test values
  const auto armijo_test_val = c1 * dgi;   // Armijo threshold
  const auto wolfe_test_curv = -c2 * dgi;  // Wolfe curvature threshold

  // Initialize bracketing variables
  ret_type step_hi, step_lo = 0, fx_hi, fx_lo = fx0, dg_hi, dg_lo = dgi;

  int iter = 0;
  const size_t max_iter = 100;
  bool converged = false;

  // Phase 1: Bracketing - find interval containing acceptable step sizes
  for (;;) {
    // Evaluate function at current trial step
    x = x0;
    Functor::axpy(step, p, x);  // x = x0 + step * p

    // Compute function value and gradient
    fx = op.eval(x);
    gfx = op.grad(x);

    if (iter++ >= max_iter) break;
    auto dg = Functor::dot(gfx, p);  // Directional derivative at current point

    // Check if current step violates Armijo condition or function increased
    if (fx - fx0 > step * armijo_test_val || (0 < step_lo and fx >= fx_lo)) {
      // Set upper bracket - current step is too large
      step_hi = step;
      fx_hi = fx;
      dg_hi = dg;
      break;  // Exit to zoom phase
    }

    // Check if strong Wolfe conditions are satisfied
    if (std::abs(dg) <= wolfe_test_curv) {
      converged = true;  // Found acceptable step
      break;
    }

    // Update bracketing interval - current step becomes lower bound
    step_hi = step_lo;
    fx_hi = fx_lo;
    dg_hi = dg_lo;

    step_lo = step;
    fx_lo = fx;
    dg_lo = dg;

    if (dg >= 0) break;  // Gradient sign change indicates bracket found
    step *= expansion;   // Expand step size for next trial
  }
  if (converged) return;  // Early termination if conditions satisfied

  // Phase 2: Zoom - refine the bracket using interpolation
  iter = 0;
  for (;;) {
    // Compute new trial step using quadratic interpolation
    step = (fx_hi - fx_lo) * step_lo -
           (step_hi * step_hi - step_lo * step_lo) * dg_lo / 2;
    step /= (fx_hi - fx_lo) - (step_hi - step_lo) * dg_lo;

    // Safeguard: use bisection if interpolation gives point outside interval
    if (step <= std::min(step_lo, step_hi) ||
        step >= std::max(step_lo, step_hi))
      step = step_lo / 2 + step_hi / 2;  // Bisection fallback

    // Evaluate function at new trial point
    x = x0;
    Functor::axpy(step, p, x);

    fx = op.eval(x);
    gfx = op.grad(x);

    if (iter++ >= max_iter) break;
    auto dg = Functor::dot(gfx, p);

    // Check if Armijo condition is violated
    if (fx - fx0 > step * armijo_test_val or fx >= fx_lo) {
      if (step == step_hi) throw std::runtime_error("Line Search Failed");
      // Update upper bracket
      step_hi = step;
      fx_hi = fx;
      dg_hi = dg;
    } else {
      // Check if strong Wolfe conditions are satisfied
      if (std::abs(dg) <= wolfe_test_curv) {
        converged = true;
        break;  // Found acceptable step
      }
      // Update bracket based on gradient sign
      if (dg * (step_hi - step_lo) >= 0) {
        step_hi = step_lo;
        fx_hi = fx_lo;
        dg_hi = dg_lo;
      }
      if (step == step_lo) throw std::runtime_error("Line Search Failed");
      // Update lower bracket
      step_lo = step;
      fx_lo = fx;
      dg_lo = dg;
    }
  }

  if (!converged) throw std::runtime_error("Line Search Failed");
}

/**
 * @brief Discrete grid-based line search for robust step size determination
 *
 * Implements a brute-force line search that evaluates the objective function
 * at a discrete grid of points along the search direction. This method is more
 * robust than gradient-based line searches but computationally expensive.
 *
 * The algorithm evaluates the function at ngrid equally-spaced points from
 * step=1.0 down to step=0.0 and selects the step that gives the minimum
 * function value. This approach is particularly useful for non-smooth or
 * difficult objective functions where gradient-based methods may fail.
 *
 * @tparam Functor Function object type that provides:
 *         - eval(x): Function evaluation
 *         - grad(x): Gradient computation
 *         - Static methods: axpy()
 *
 * @param op Function object implementing the objective function and gradient
 * @param x0 Starting point for the line search
 * @param p Search direction (any direction, not required to be descent)
 * @param step Output: Optimal step size found (modified)
 * @param x Output: New point x0 + step * p (modified)
 * @param fx Output: Function value at the optimal point (modified)
 * @param gfx Output: Gradient at the optimal point (modified)
 *
 * @note Uses ngrid=100 evaluation points with uniform spacing ds=1/ngrid
 * @note Does not require descent direction or gradient information for search
 * @note Guaranteed to find the best step among the evaluated grid points
 */
template <typename Functor>
void discrete_line_search(Functor& op, const detail::arg_type_t<Functor>& x0,
                          const detail::arg_type_t<Functor>& p,
                          detail::ret_type_t<Functor>& step,
                          detail::arg_type_t<Functor>& x,
                          detail::ret_type_t<Functor>& fx,
                          detail::arg_type_t<Functor>& gfx) {
  using arg_type = detail::arg_type_t<Functor>;
  using ret_type = detail::ret_type_t<Functor>;

  const auto fx0 = op.eval(x0);   // Function value at starting point
  const auto gfx0 = op.grad(x0);  // Gradient at starting point (for logging)

  // Grid search parameters
  const size_t ngrid = 100;    // Number of grid points to evaluate
  const auto ds = 1. / ngrid;  // Step size between grid points

  auto logger = spdlog::get("line_search");
  logger->debug("");
  logger->debug("Starting Discretized Line Search");
  logger->debug("ngrid = {}, ds = {}", ngrid, ds);

  const std::string fmt_str =
      "iter = {:4}, F(X) = {:15.12f}, dF = {:20.12e}, S = {:10.6e}";

  // Initialize with full step (step = 1.0)
  step = 1.0;
  x = x0;
  Functor::axpy(step, p, x);
  fx = op.eval(x);
  logger->debug(fmt_str, 0, fx, fx - fx0, step);

  // Track the best point found so far
  ret_type min_val = fx;
  auto min_step = step;

  // Grid search: evaluate function at equally-spaced points
  for (size_t ig = 1; ig < ngrid; ++ig) {
    auto temp_step = (1.0 - ig * ds);  // Step decreases from 1.0 to 0.0
    x = x0;
    Functor::axpy(temp_step, p, x);  // x = x0 + temp_step * p
    fx = op.eval(x);
    logger->debug(fmt_str, ig, fx, (fx - fx0), temp_step);

    // Update minimum if better point found
    if (fx < min_val) {
      min_val = fx;
      min_step = temp_step;
    }
  }

  // Set outputs to the best point found
  step = min_step;
  x = x0;
  Functor::axpy(step, p, x);
  fx = min_val;
  gfx = op.grad(x);  // Compute gradient at optimal point

  logger->debug("MinVal at S = {:10.6e} F(X) = {:15.12f}, dF = {:20.12e}", step,
                fx, (fx - fx0));
}

}  // namespace bfgs

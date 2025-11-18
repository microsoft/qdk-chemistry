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
#include <spdlog/stopwatch.h>

#include <vector>

#include "bfgs_hessian.hpp"
#include "line_search.hpp"

namespace bfgs {

/**
 * @brief Configuration settings for BFGS optimization algorithm
 *
 * This structure contains all the configurable parameters for controlling
 * the behavior of the BFGS optimization algorithm. Users can modify these
 * settings to customize the optimization process for their specific problem.
 */
struct BFGSSettings {
  /// Maximum number of BFGS iterations allowed before termination
  size_t max_iter = 100;
};

/**
 * @brief Main BFGS optimization algorithm with custom Hessian approximation
 *
 * Implements the Broyden-Fletcher-Goldfarb-Shanno (BFGS) quasi-Newton
 * optimization algorithm. This variant allows users to provide a custom Hessian
 * approximation strategy through the BFGSHessian parameter.
 *
 * The algorithm iteratively minimizes a function by:
 * 1. Computing search direction using inverse Hessian approximation
 * 2. Performing line search to find optimal step size
 * 3. Updating the Hessian approximation using BFGS formula
 * 4. Checking convergence criteria
 *
 * @tparam Functor Function object type that provides:
 *         - eval(x): Function evaluation
 *         - grad(x): Gradient computation
 *         - converged(x, gfx): Convergence checking
 *         - Static methods: norm(), scal(), subtract(), axpy(), dot()
 *
 * @param op Function object implementing the objective function and its
 * gradient
 * @param x0 Initial guess for the optimization variables
 * @param B Custom Hessian approximation object (modified during optimization)
 * @param settings Configuration parameters for the optimization
 *
 * @return Optimal point found by the algorithm
 *
 * @throws std::runtime_error If line search fails or algorithm doesn't converge
 *
 * @note The functor must provide vector operations as static methods:
 *       - norm(v): Compute vector norm
 *       - scal(alpha, v): Scale vector in-place: v *= alpha
 *       - subtract(a, b): Return a - b
 *       - axpy(alpha, x, y): Compute y += alpha * x
 *       - dot(x, y): Compute inner product
 */
template <typename Functor>
detail::arg_type_t<Functor> bfgs(Functor& op,
                                 const detail::arg_type_t<Functor>& x0,
                                 BFGSHessian<Functor>& B,
                                 BFGSSettings settings) {
  // Type aliases for cleaner code
  using arg_type = detail::arg_type_t<Functor>;
  using ret_type = detail::ret_type_t<Functor>;

  // Set up logging for algorithm monitoring
  auto logger = spdlog::get("bfgs");
  if (!logger) logger = spdlog::stdout_color_mt("bfgs");

  auto ls_logger = spdlog::get("line_search");
  if (!ls_logger) ls_logger = spdlog::stdout_color_mt("line_search");

  // Initialize BFGS algorithm state
  arg_type x = x0;            // Current point
  ret_type fx = op.eval(x);   // Function value at current point
  arg_type gfx = op.grad(x);  // Gradient at current point
  constexpr const char* fmt_string =
      "iter {:4}, F(X) = {:15.12e}, dF = {:20.12e}, |gF(X)| = {:20.12e}";

  logger->info("Starting BFGS Iterations");
  logger->info("|X0| = {:15.12e}", Functor::norm(x));
  logger->info(fmt_string, 0, fx, 0.0, Functor::norm(gfx));

  // Initialize search direction as negative gradient (steepest descent)
  arg_type p = gfx;
  Functor::scal(-1.0, p);
  ret_type step = 1.;  // Initialize step for gradient descent

  bool converged = false;
  // Main BFGS optimization loop
  for (size_t iter = 0; iter < settings.max_iter; ++iter) {
    // Perform line search to find optimal step size along search direction
    arg_type x_new, gfx_new = gfx;
    ret_type f_sav = fx;  // Save current function value for progress tracking
    try {
      backtracking_line_search(op, x, p, step, x_new, fx, gfx_new);
    } catch (...) {
      throw std::runtime_error("Line Search Failed");
    }

    // Compute BFGS update vectors
    arg_type s = Functor::subtract(x_new, x);  // Step vector: x_{k+1} - x_k
    arg_type y = Functor::subtract(
        gfx_new, gfx);  // Gradient difference: grad_{k+1} - grad_k

    // Update current state for next iteration
    x = x_new;
    // Recompute the gradient at the new x point for accuracy
    gfx = op.grad(x);
    step = 1.0;  // Reset step size for next iteration

    logger->info(fmt_string, iter + 1, fx, fx - f_sav, Functor::norm(gfx));

    // Check for convergence using user-defined criteria
    if (op.converged(x, gfx)) {
      converged = true;
      break;
    }

    // Update Hessian approximation using BFGS formula and compute new search
    // direction
    B.update(s, y);          // Update inverse Hessian approximation
    p = B.apply(gfx);        // Compute search direction: p = -H_k * grad_k
    Functor::scal(-1.0, p);  // Negate to get descent direction

    logger->debug(
        "  XNRM = {:10.5e} SNRM = {:10.5e} YNRM = {:10.5e} PNRM = {:10.5e}",
        Functor::norm(x), Functor::norm(s), Functor::norm(y), Functor::norm(p));
  }
  // Report final status and return result
  if (converged) logger->info("BFGS Converged!");

  if (!converged) throw std::runtime_error("BFGS Did Not Converge");
  return x;  // Return the optimal point found
}

/**
 * @brief Convenience BFGS optimization with default identity Hessian
 * approximation
 *
 * This is a convenience wrapper around the main BFGS function that
 * automatically creates a default identity Hessian approximation. This is
 * suitable for most general-purpose optimization problems where no specific
 * Hessian initialization is required.
 *
 * @tparam Functor Function object type that provides:
 *         - eval(x): Function evaluation
 *         - grad(x): Gradient computation
 *         - converged(x, gfx): Convergence checking
 *         - Static methods: norm(), scal(), subtract(), axpy(), dot()
 *
 * @param op Function object implementing the objective function and its
 * gradient
 * @param x0 Initial guess for the optimization variables
 * @param settings Configuration parameters for the optimization
 *
 * @return Optimal point found by the algorithm
 *
 * @throws std::runtime_error If line search fails or algorithm doesn't converge
 *
 * @note This function uses an identity matrix as the initial Hessian
 * approximation (H_0 = I), which makes the first step equivalent to steepest
 * descent.
 */
template <typename Functor>
detail::arg_type_t<Functor> bfgs(Functor& op,
                                 const detail::arg_type_t<Functor>& x0,
                                 BFGSSettings settings) {
  auto B = make_identity_hessian<Functor>();  // Create default identity Hessian
  return bfgs(op, x0, *B, settings);          // Delegate to main BFGS function
}

}  // namespace bfgs

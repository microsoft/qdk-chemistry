// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace qdk::chemistry::scf::impl {

/**
 * @brief Nocedal-Wright line search with strong Wolfe conditions.
 *
 * Implements the line search algorithm described in "Numerical Optimization"
 * by Nocedal and Wright. This is a two-phase algorithm:
 * 1. Bracketing phase: Find an interval containing acceptable step sizes.
 * 2. Zoom phase: Refine the interval using interpolation to find optimal step.
 *
 * Ordinary steps satisfy Armijo and strong Wolfe conditions. Roundoff-limited
 * steps may instead accept indistinguishable energies when both the Wolfe
 * curvature condition and full-gradient reduction hold.
 *
 * @tparam Functor Function object type that provides:
 *         - eval(x): Function evaluation
 *         - grad(x): Gradient computation
 *         - Static methods: dot(), axpy()
 *
 * @param op Function object implementing the objective function and gradient.
 * @param x0 Starting point for the line search.
 * @param p Search direction (must be a descent direction).
 * @param step Initial step size (will be modified to optimal step).
 * @param x Output: New point x0 + step * p (modified).
 * @param fx Output: Function value at the new point (modified).
 * @param gfx Output: Gradient at the new point (modified).
 *
 * @throws std::logic_error If search direction is not descent.
 * @throws std::runtime_error If step is non-positive or line search fails.
 *
 * @note Uses standard parameters: c1=1e-4 (Armijo), c2=0.9 (Wolfe),
 * expansion=2.0.
 * @note More robust than backtracking but computationally more expensive.
 */
template <typename Functor>
void nocedal_wright_line_search(Functor& op,
                                const typename Functor::argument_type& x0,
                                const typename Functor::argument_type& p,
                                typename Functor::return_type& step,
                                typename Functor::argument_type& x,
                                typename Functor::return_type& fx,
                                typename Functor::argument_type& gfx) {
  using ret_type = typename Functor::return_type;

  const auto fx0 = fx;
  const auto dgi = Functor::dot(p, gfx);
  const auto initial_grad_norm_squared = Functor::dot(gfx, gfx);

  if (!std::isfinite(fx0) || !std::isfinite(dgi) ||
      !std::isfinite(initial_grad_norm_squared) || !std::isfinite(step)) {
    throw std::runtime_error("Nonfinite line search starting point");
  }
  if (dgi > 0)
    throw std::logic_error(
        "the moving direction increases the objective function value");
  if (step <= 0.0) throw std::runtime_error("Step must be positive");

  constexpr auto c1 = 1e-4;
  constexpr auto c2 = 0.9;
  constexpr auto expansion = 2.0;

  const auto armijo_test_val = c1 * dgi;
  const auto wolfe_test_curv = -c2 * dgi;

  const auto accept_roundoff_step = [&](ret_type dg,
                                        ret_type grad_norm_squared) {
    // This small roundoff allowance is not a bound on objective-evaluation
    // error. Require a tiny predicted decrease and independent gradient
    // improvement.
    const auto energy_roundoff =
        8 * std::numeric_limits<ret_type>::epsilon() *
        std::max({ret_type(1), std::abs(fx0), std::abs(fx)});
    return dgi < 0 && -step * dgi <= energy_roundoff &&
           fx - fx0 <= energy_roundoff && std::abs(dg) <= wolfe_test_curv &&
           grad_norm_squared <= c2 * c2 * initial_grad_norm_squared;
  };

  ret_type step_hi = 0, step_lo = 0, fx_hi = fx0, fx_lo = fx0, dg_hi = dgi,
           dg_lo = dgi;

  int iter = 0;
  const size_t max_iter = 100;
  bool converged = false;

  for (;;) {
    x = x0;
    Functor::axpy(step, p, x);

    fx = op.eval(x);
    gfx = op.grad(x);

    if (iter++ >= max_iter) break;
    auto dg = Functor::dot(gfx, p);
    const auto grad_norm_squared = Functor::dot(gfx, gfx);
    if (!std::isfinite(fx) || !std::isfinite(dg) ||
        !std::isfinite(grad_norm_squared) || !std::isfinite(step)) {
      throw std::runtime_error("Nonfinite line search trial");
    }

    if (accept_roundoff_step(dg, grad_norm_squared)) {
      converged = true;
      break;
    }

    if (fx - fx0 > step * armijo_test_val || (0 < step_lo and fx >= fx_lo)) {
      step_hi = step;
      fx_hi = fx;
      dg_hi = dg;
      break;
    }

    if (std::abs(dg) <= wolfe_test_curv) {
      converged = true;
      break;
    }

    step_hi = step_lo;
    fx_hi = fx_lo;
    dg_hi = dg_lo;

    step_lo = step;
    fx_lo = fx;
    dg_lo = dg;

    if (dg >= 0) break;
    step *= expansion;
  }
  if (converged) return;

  iter = 0;
  for (;;) {
    step = (fx_hi - fx_lo) * step_lo -
           (step_hi * step_hi - step_lo * step_lo) * dg_lo / 2;
    step /= (fx_hi - fx_lo) - (step_hi - step_lo) * dg_lo;

    if (step <= std::min(step_lo, step_hi) ||
        step >= std::max(step_lo, step_hi))
      step = step_lo / 2 + step_hi / 2;

    x = x0;
    Functor::axpy(step, p, x);

    fx = op.eval(x);
    gfx = op.grad(x);

    if (iter++ >= max_iter) break;
    auto dg = Functor::dot(gfx, p);
    const auto grad_norm_squared = Functor::dot(gfx, gfx);
    if (!std::isfinite(fx) || !std::isfinite(dg) ||
        !std::isfinite(grad_norm_squared) || !std::isfinite(step)) {
      throw std::runtime_error("Nonfinite line search trial");
    }

    if (accept_roundoff_step(dg, grad_norm_squared)) {
      converged = true;
      break;
    }

    if (fx - fx0 > step * armijo_test_val or fx >= fx_lo) {
      if (step == step_hi) throw std::runtime_error("Line Search Failed");
      step_hi = step;
      fx_hi = fx;
      dg_hi = dg;
    } else {
      if (std::abs(dg) <= wolfe_test_curv) {
        converged = true;
        break;
      }
      if (dg * (step_hi - step_lo) >= 0) {
        step_hi = step_lo;
        fx_hi = fx_lo;
        dg_hi = dg_lo;
      }
      if (step == step_lo) throw std::runtime_error("Line Search Failed");
      step_lo = step;
      fx_lo = fx;
      dg_lo = dg;
    }
  }

  if (!converged) throw std::runtime_error("Line Search Failed");
}

}  // namespace qdk::chemistry::scf::impl

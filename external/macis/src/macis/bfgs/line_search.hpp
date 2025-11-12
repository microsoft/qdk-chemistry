/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <spdlog/spdlog.h>

#include <iostream>

#include "bfgs_traits.hpp"

namespace bfgs {

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

  const auto fx0 = op.eval(x0);
  const auto gfx0 = op.grad(x0);
  const auto dg0 = Functor::dot(p, gfx0);
  if (dg0 > 0)
    throw std::runtime_error("Step will increase objective function");
  if (step <= 0.0) throw std::runtime_error("Step must be positive");

#define USE_CARLOS_VAL
#ifdef USE_CARLOS_VAL
  constexpr auto c1 = 0.5;
#else
  constexpr auto c1 = 1e-4;
#endif /* USE_CARLOS_VAL */
  constexpr auto c2 = 0.8;
  constexpr auto tau = 0.5; /*1./1.618033988749894;*/

  const auto t_armijo = -dg0 * c1;
  const auto t_swolfe = c2 * std::abs(dg0);
  auto test_carlos = [t_armijo, fx0](auto _fx, auto _s) -> bool {
    return ((fx0 - _fx) - _s * t_armijo > -1e-6);
  };
  auto test_armijo = [t_armijo, fx0](auto _fx, auto _s) -> bool {
    return (_fx < (fx0 + _s * t_armijo));
  };
  auto test_swolfe = [t_swolfe, &p](auto _gfx) -> bool {
    return std::abs(Functor::dot(p, _gfx)) < t_swolfe;
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

  // Initialization
  step = 1.0;
  x = x0;
  Functor::axpy(step, p, x);
  fx = op.eval(x);
  logger->debug(fmt_str, 0, fx, fx - fx0, step);

  size_t max_iter = 100;
  for (size_t iter = 0; iter < max_iter; ++iter) {
#ifdef USE_CARLOS_VAL
    if (test_carlos(fx, step)) break;
    step *= tau;
#else
    if (test_armijo(fx, step)) {
      logger->debug("  * armijo condition met");
      gfx = op.grad(x);
      if (test_swolfe(gfx)) {
        logger->debug("  * wolfe condition met");
        break;
      }

      logger->debug("  * wolfe condition not met: increasing step by 2.1");
      step *= 2.1;
    } else {
      logger->debug("  * armijo condition not met: decreasing step by {}", tau);
      step *= tau;
    }
#endif /* USE_CARLOS_VAL */
    x = x0;
    Functor::axpy(step, p, x);
    fx = op.eval(x);
    logger->debug(fmt_str, iter + 1, fx, (fx - fx0), step);
  }

  if (fx - fx0 > 0 or step < 1e-6)
    throw std::runtime_error("Line Search Failed");

  gfx = op.grad(x);
  logger->debug("Line Search Converged with S = {:10.6e}", step);
  logger->debug("");
}

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

  const auto fx0 = fx;
  const auto dgi = Functor::dot(p, gfx);
  if (dgi > 0)
    throw std::logic_error(
        "the moving direction increases the objective function value");
  if (step <= 0.0) throw std::runtime_error("Step must be positive");

  constexpr auto c1 = 1e-4;
  constexpr auto c2 = 0.9;
  constexpr auto expansion = 2.0;

  const auto armijo_test_val = c1 * dgi;
  const auto wolfe_test_curv = -c2 * dgi;

  ret_type step_hi, step_lo = 0, fx_hi, fx_lo = fx0, dg_hi, dg_lo = dgi;

  int iter = 0;
  const size_t max_iter = 100;
  bool converged = false;
  // Bracketing Phase
  for (;;) {
    // Compute new x
    x = x0;
    Functor::axpy(step, p, x);

    // Evaluate new f(x) and gradiend
    fx = op.eval(x);
    gfx = op.grad(x);

    if (iter++ >= max_iter) break;
    auto dg = Functor::dot(gfx, p);

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

  // Zoom Phase
  iter = 0;
  for (;;) {
    step = (fx_hi - fx_lo) * step_lo -
           (step_hi * step_hi - step_lo * step_lo) * dg_lo / 2;
    step /= (fx_hi - fx_lo) - (step_hi - step_lo) * dg_lo;

    if (step <= std::min(step_lo, step_hi) ||
        step >= std::max(step_lo, step_hi))
      step = step_lo / 2 + step_hi / 2;

    // Compute new x
    x = x0;
    Functor::axpy(step, p, x);

    // Evaluate new f(x) and gradiend
    fx = op.eval(x);
    gfx = op.grad(x);

    if (iter++ >= max_iter) break;
    auto dg = Functor::dot(gfx, p);

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

template <typename Functor>
void discrete_line_search(Functor& op, const detail::arg_type_t<Functor>& x0,
                          const detail::arg_type_t<Functor>& p,
                          detail::ret_type_t<Functor>& step,
                          detail::arg_type_t<Functor>& x,
                          detail::ret_type_t<Functor>& fx,
                          detail::arg_type_t<Functor>& gfx) {
  using arg_type = detail::arg_type_t<Functor>;
  using ret_type = detail::ret_type_t<Functor>;

  const auto fx0 = op.eval(x0);
  const auto gfx0 = op.grad(x0);

  const size_t ngrid = 100;
  const auto ds = 1. / ngrid;

  auto logger = spdlog::get("line_search");
  logger->debug("");
  logger->debug("Starting Discretized Line Search");
  logger->debug("ngrid = {}, ds = {}", ngrid, ds);

  const std::string fmt_str =
      "iter = {:4}, F(X) = {:15.12f}, dF = {:20.12e}, S = {:10.6e}";

  // Initialization
  step = 1.0;
  x = x0;
  Functor::axpy(step, p, x);
  fx = op.eval(x);
  logger->debug(fmt_str, 0, fx, fx - fx0, step);

  ret_type min_val = fx;
  auto min_step = step;
  for (size_t ig = 1; ig < ngrid; ++ig) {
    auto temp_step = (1.0 - ig * ds);
    x = x0;
    Functor::axpy(temp_step, p, x);
    fx = op.eval(x);
    logger->debug(fmt_str, ig, fx, (fx - fx0), temp_step);

    if (fx < min_val) {
      min_val = fx;
      min_step = temp_step;
    }
  }

  step = min_step;
  x = x0;
  Functor::axpy(step, p, x);
  fx = min_val;
  gfx = op.grad(x);

  logger->debug("MinVal at S = {:10.6e} F(X) = {:15.12f}, dF = {:20.12e}", step,
                fx, (fx - fx0));
}

}  // namespace bfgs

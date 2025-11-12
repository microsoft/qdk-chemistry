/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
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

struct BFGSSettings {
  size_t max_iter = 100;
};

template <typename Functor>
detail::arg_type_t<Functor> bfgs(Functor& op,
                                 const detail::arg_type_t<Functor>& x0,
                                 BFGSHessian<Functor>& B,
                                 BFGSSettings settings) {
  using arg_type = detail::arg_type_t<Functor>;
  using ret_type = detail::ret_type_t<Functor>;
  auto logger = spdlog::get("bfgs");
  if (!logger) logger = spdlog::stdout_color_mt("bfgs");

  auto ls_logger = spdlog::get("line_search");
  if (!ls_logger) ls_logger = spdlog::stdout_color_mt("line_search");

  // Initialize BFGS
  arg_type x = x0;
  ret_type fx = op.eval(x);
  arg_type gfx = op.grad(x);
  constexpr const char* fmt_string =
      "iter {:4}, F(X) = {:15.12e}, dF = {:20.12e}, |gF(X)| = {:20.12e}";

  logger->info("Starting BFGS Iterations");
  logger->info("|X0| = {:15.12e}", Functor::norm(x));
  logger->info(fmt_string, 0, fx, 0.0, Functor::norm(gfx));

  // Initial Hessian
  arg_type p = gfx;
  Functor::scal(-1.0, p);
  ret_type step = 1.;  // Initialize step for gradient descent

  bool converged = false;
  for (size_t iter = 0; iter < settings.max_iter; ++iter) {
    // Do line search
    arg_type x_new, gfx_new = gfx;
    ret_type f_sav = fx;
    try {
      backtracking_line_search(op, x, p, step, x_new, fx, gfx_new);
    } catch (...) {
#if 0
      // #TODO: Replace this whole debugging block with logger-driven diagnostics instead of std::cout and disabled code.
      logger->info("line search failed, trying gradient descent");
      try{
        p = op.grad(x); Functor::scal(-1.0,p);
        backtracking_line_search(op, x, p, step, x_new, fx, gfx_new);
      } catch(...) {
        size_t ngrid = 300;
        double ds    = 0.001;
        std::cout << "XNRM = " << Functor::norm(x) << std::endl;
        std::cout << "d = [" << std::endl;
        for(size_t i = 0; i <= ngrid; ++i) {
          arg_type xp = x; Functor::axpy(i * ds, p, xp);
          auto fp = op.eval(xp);
          arg_type xg = x; Functor::axpy(i * -ds, op.grad(x), xg);
          auto fg = op.eval(xg);

          std::cout << i*ds << ", " << fp << ", " << fg << std::endl;
        }
        std::cout << "];" << std::endl;
        throw std::runtime_error("Line Search Failed");
      }
#else
      throw std::runtime_error("Line Search Failed");
#endif /* 0 */
    }

    // Compute update steps
    arg_type s = Functor::subtract(x_new, x);
    arg_type y = Functor::subtract(gfx_new, gfx);

    // Save quantities for possible next step or return
    x = x_new;
    // #TODO: As gfx_new already existing, do we need to recompute the gradient
    // still? Seems redundant?
    gfx = op.grad(x) /*gfx_new*/;
    step = 1.0;
    // #TODO: Drop the below commented out debug printing; the logger below
    // already reports the relevant diagnostics.
    // std::cout << iter << ", " << fx <<  ", " << fx-f_sav << ", "
    //           << Functor::norm(gfx)  << std::endl;

    logger->info(fmt_string, iter + 1, fx, fx - f_sav, Functor::norm(gfx));

    // Check for convergence
    if (op.converged(x, gfx)) {
      converged = true;
      break;
    }

    // Update and apply Hessian
    B.update(s, y);
    p = B.apply(gfx);
    Functor::scal(-1.0, p);

    logger->debug(
        "  XNRM = {:10.5e} SNRM = {:10.5e} YNRM = {:10.5e} PNRM = {:10.5e}",
        Functor::norm(x), Functor::norm(s), Functor::norm(y), Functor::norm(p));
  }
  if (converged) logger->info("BFGS Converged!");

  if (!converged) throw std::runtime_error("BFGS Did Not Converge");
  return x;
}

template <typename Functor>
detail::arg_type_t<Functor> bfgs(Functor& op,
                                 const detail::arg_type_t<Functor>& x0,
                                 BFGSSettings settings) {
  auto B = make_identity_hessian<Functor>();
  return bfgs(op, x0, *B, settings);
}

}  // namespace bfgs

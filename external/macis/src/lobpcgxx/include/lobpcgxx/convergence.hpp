/**
 *  Copyright (c) 2020 The Regents of the University of California,
 *  through Lawrence Berkeley National Laboratory.
 *
 *  Author: David Williams-Young
 *
 *  This file is part of LOBPCGXX. All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *
 *  (1) Redistributions of source code must retain the above copyright notice,
 * this list of conditions and the following disclaimer.
 *
 *  (2) Redistributions in binary form must reproduce the above copyright
 * notice, this list of conditions and the following disclaimer in the
 * documentation and/or other materials provided with the distribution.
 *
 *  (3) Neither the name of the University of California, Lawrence Berkeley
 *  National Laboratory, U.S. Dept. of Energy nor the names of its contributors
 * may be used to endorse or promote products derived from this software without
 *  specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 *  You are under no obligation whatsoever to provide any bug fixes, patches, or
 *  upgrades to the features, functionality or performance of the source code
 *  ("Enhancements") to anyone; however, if you choose to make your Enhancements
 *  available either publicly, or directly to Lawrence Berkeley National
 *  Laboratory, without imposing a separate written license agreement for such
 *  Enhancements, then you hereby grant the following license: a non-exclusive,
 *  royalty-free perpetual license to install, use, modify, prepare derivative
 *  works, incorporate into other computer software, distribute, and sublicense
 *  such enhancements or derivative works thereof, in binary and source code
 * form.
 */
#pragma once

#include <vector>

#include "type_traits.hpp"

namespace lobpcgxx {

/**
 *  @brief Typedef for prototype convergence check routine.
 */
template <typename T>
using lobpcg_convergence_check =
    std::function<bool(int64_t, const T*, const T*, T)>;

/**
 *  @brief Check convergence of LOBPCG based on relative residual norms.
 *
 *  Checks if all of the relative residual norms are under the specified
 *  threshold.
 *
 *  @param[in] NR      Number of residuals to check
 *  @param[in] ABS_RES Absolute residual norms
 *  @param[in] REL_RES Relative residual norms
 *  @param[in] TOL     Residual convergence tolerance
 *
 *  @returns True if all residuals are under the specified threshold,
 *           False otherwise.
 */
template <typename T>
bool lobpcg_relres_convergence_check(int64_t NR, const T* ABS_RES,
                                     const T* REL_RES, T TOL) {
  return std::all_of(REL_RES, REL_RES + NR,
                     [&](const auto r) { return r < TOL; });

}  // lobpcg_relres_convergence_check

/**
 *  @brief Check convergence of LOBPCG based on absolute residual norms.
 *
 *  Checks if all of the absolute residual norms are under the specified
 *  threshold.
 *
 *  @param[in] NR      Number of residuals to check
 *  @param[in] ABS_RES Absolute residual norms
 *  @param[in] REL_RES Relative residual norms
 *  @param[in] TOL     Residual convergence tolerance
 *
 *  @returns True if all residuals are under the specified threshold,
 *           False otherwise.
 */
template <typename T>
bool lobpcg_absres_convergence_check(int64_t NR, const T* ABS_RES,
                                     const T* REL_RES, T TOL) {
  return std::all_of(ABS_RES, ABS_RES + NR,
                     [&](const auto r) { return r < TOL; });

}  // lobpcg_absres_convergence_check

/**
 *  @brief Struct to track the convergence of LOBPCG
 */
template <typename T>
struct lobpcg_convergence {
  /**
   *  @brief Struct to hold the convergence info for a particular
   *         LOBPCG iteration
   */
  struct lobpcg_iteration {
    std::vector<detail::real_t<T>> W;        ///< Eigenvalue approximations
    std::vector<detail::real_t<T>> res;      ///< Absolute residual norms
    std::vector<detail::real_t<T>> rel_res;  ///< Relative residual norms
  };

  std::vector<lobpcg_iteration> conv_data;
  ///< Tracked convergence data

};  // struct lobpcg_convergence

}  // namespace lobpcgxx

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

#include <functional>

namespace lobpcgxx {

/**
 *  @brief Typedef for prototype operator action routine.
 */
template <typename T>
using operator_action_type =
    std::function<void(int64_t, int64_t, const T*, int64_t, T*, int64_t)>;

/**
 *  @brief A struct to manage the application of an operator
 *  onto a subspace in LOBPCG.
 *
 *  Includes an optional preconditioner.
 *
 *  @tparam T Field overwhich the operator is defined
 */
template <typename T>
class lobpcg_operator {
  operator_action_type<T> Aop_;  ///< Matrix operator functor
  operator_action_type<T> Kop_;  ///< Preconditioner operator functor

  /**
   *  @brief Default preconditioner.
   *
   *  Does nothing to the subspace and copies the
   *  original data into the result.
   *
   *  @param[in]  N    The number of rows of V / KV
   *  @param[in]  K    The number of columns of V / KV
   *  @param[in]  V    Input subspace
   *  @param[in]  LDV  Leading dimension of V
   *  @param[out] KV   Preconditioned subspace
   *  @param[in]  LDKV Leading dimension of KV
   */
  static void default_preconditioner(int64_t N, int64_t K, const T* V,
                                     int64_t LDV, T* KV, int64_t LDKV) {
    lapack::lacpy(lapack::MatrixType::General, N, K, V, LDV, KV, LDKV);
  }

 public:
  lobpcg_operator() = delete;  // No default state

  /**
   *  @brief Construct a lobpcg_operator instance with both
   *  matrix and preconditioner
   *
   *  @param[in] Aop Matrix functor
   *  @param[in] Kop Preconditioner functor
   */
  lobpcg_operator(operator_action_type<T> Aop, operator_action_type<T> Kop)
      : Aop_(Aop), Kop_(Kop) {
    if (not Aop_) throw std::runtime_error("A cannot be a null op");
  };

  /**
   *  @brief Construct a lobpcg_operator instance with matrix only
   *
   #  Defaults preconditioner to default_preconditioner
   *
   *  @param[in] Aop Matrix functor
   */
  lobpcg_operator(operator_action_type<T> Aop)
      : lobpcg_operator(Aop, default_preconditioner) {}

  /**
   *  @brief Apply matrix to subspace.
   *
   *  @tparam Args Parameter pack to be forwarded to matrix functor.
   *
   *  @param[in/out] Parameter pack to be forwarded to matrix functor.
   */
  template <typename... Args>
  void apply_matrix(Args&&... args) const {
    Aop_(std::forward<Args>(args)...);
  }

  /**
   *  @brief Apply preconditioner to subspace.
   *
   *  @tparam Args Parameter pack to be forwarded to preconditioner functor.
   *
   *  @param[in/out] Parameter pack to be forwarded to preconditioner functor.
   */
  template <typename... Args>
  void apply_preconditioner(Args&&... args) const {
    Kop_(std::forward<Args>(args)...);
  }

};  // class lobpcg_operator

}  // namespace lobpcgxx

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

#include <blas.hh>
#include <lapack.hh>

#include "type_traits.hpp"

namespace lobpcgxx {

/**
 *  @brief Compute residuals and residual norms of approximate eigenvectors
 *
 *  @param[in]  N    Number of rows in X / AX
 *  @param[in]  K    Number of columns in X / AX
 *  @param[in]  X    Approximate eigenvectors (LDX*K)
 *  @param[in]  LDX  Leading dimension of X ( > N)
 *  @param[in]  AX   Precomputed A * X (LDAX*K)
 *  @param[in]  LDAX Leading dimension of AX ( > N )
 *  @param[out] W    Eigenvalue approximations (K)
 *  @param[out] R    Residuals (LDR*K)
 *  @param[in]  LDR  Leading dimension of R ( > N)
 *  @param[out] ABS_R Absolute residual norms
 *  @param[out] REL_R Relative residual norms
 */
template <typename T>
void residuals(int64_t N, int64_t K, const T* X, int64_t LDX, const T* AX,
               int64_t LDAX, detail::real_t<T>* W, T* R, int64_t LDR,
               detail::real_t<T>* ABS_R, detail::real_t<T>* REL_R) {
  for (int64_t i = 0; i < K; ++i) {
    blas::copy(N, AX + i * LDAX, 1, R + i * LDR, 1);
    blas::axpy(N, -W[i], X + i * LDX, 1, R + i * LDR, 1);
    ABS_R[i] = blas::nrm2(N, R + i * LDR, 1);
    REL_R[i] = ABS_R[i] / (blas::nrm2(N, AX + i * LDAX, 1) + std::abs(W[i]));
  }

}  // residuals

}  // namespace lobpcgxx

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
 *  @brief Rotate a set of vectors
 *
 *  @param[in]  N      Number of rows in X
 *  @param[in]  K      Number of columns in X
 *  @param[in/out]  X  Input: vectors to be rotated, Output: rotated vectors
 * (LDX*K)
 *  @param[in]  LDX    Leading dimension of X ( > N)
 *  @param[in]  C      Rotation matrix ( K * LDC )
 *  @param[in]  LDC    Leading dimension of C
 *  @param[in/out] WORK Workspace of at least K*N
 */

template <typename T>
void rotate_basis(int64_t N, int64_t K, int64_t M, T* X, int64_t LDX, T* C,
                  int64_t LDC, T* WORK) {
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans, N, K,
             M, T(1.), X, LDX, C, LDC, T(0.), WORK, N);

  lapack::lacpy(lapack::MatrixType::General, N, K, WORK, N, X, LDX);
}

}  // namespace lobpcgxx

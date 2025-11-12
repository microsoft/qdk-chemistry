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

namespace lobpcgxx {

/**
 *  @brief Orthogonalize a subspace via Cholesky QR.
 *
 *  Produces an orthogonal subspace from a non-orthogonal input. Input
 *  data is assumed to be dense and contiguous. The procedure will fail
 *  if the Grammian of the subspace is ill-conditioned.
 *
 *  @tparam T Type for the field on which the vector space is defined.
 *
 *  @param[in]     N The number of rows in V
 *  @param[in]     K The number of columns in V
 *  @param[in/out] V On input, the subspace to be orthogonalized stored
 * column-major, On output, the orthogonal subspace (if sucessful).
 *  @param[in]     LDV The leading dimension of the subspace.
 *  @param[in]     n_rep The number of reorthogonalizations to perform (default
 * = 2)
 */
template <typename T>
void cholqr(int64_t N, int64_t K, T* V, int64_t LDV, int64_t n_rep = 2) {
  std::vector<T> L(K * K);

  for (int64_t i = 0; i < n_rep; ++i) {
    blas::gemm(blas::Layout::ColMajor, blas::Op::ConjTrans, blas::Op::NoTrans,
               K, K, N, T(1.), V, LDV, V, LDV, T(0.), L.data(), K);

    auto info = lapack::potrf(lapack::Uplo::Lower, K, L.data(), K);
    if (info) throw std::runtime_error("Cholesky failed in CholQR");

    blas::trsm(blas::Layout::ColMajor, blas::Side::Right, blas::Uplo::Lower,
               blas::Op::ConjTrans, blas::Diag::NonUnit, N, K, T(1.), L.data(),
               K, V, LDV);
  }

}  // cholqr

}  // namespace lobpcgxx

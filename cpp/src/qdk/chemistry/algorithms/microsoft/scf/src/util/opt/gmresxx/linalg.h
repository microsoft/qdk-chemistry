// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cassert>
#include <complex>

#include "type_traits.h"
#include "types.h"

namespace qdk::chemistry::scf {
/**
 * @brief Computes the Euclidean (L2) norm of a vector
 *
 * Calculates the L2 norm (Euclidean norm) of a vector X with stride INCX.
 * For complex vectors, uses the magnitude of each element before squaring.
 * The result is always real-valued even for complex input vectors.
 *
 * @tparam T Numeric type (real or complex)
 * @param N Number of vector elements
 * @param X Pointer to vector data
 * @param INCX Stride between consecutive elements
 * @return L2 norm as real-valued type
 */
template <typename T>
detail::real_t<T> two_norm(int32_t N, const T* X, int32_t INCX) {
  detail::real_t<T> nrm = 0.;
  for (int32_t i = 0; i < N; ++i) {
    const auto a = std::abs(X[i * INCX]);
    nrm += a * a;
  }
  return std::sqrt(nrm);
}

/**
 * @brief Computes the inner product (dot product) of two vectors with
 * conjugation
 *
 * Calculates the inner product of vectors X and Y, applying complex conjugation
 * to the first vector. For real vectors, this is equivalent to a standard dot
 * product. For complex vectors, computes ∑ conj(X[i]) * Y[i].
 *
 * @tparam T Numeric type (real or complex)
 * @param N Number of vector elements
 * @param X Pointer to first vector data (will be conjugated)
 * @param INCX Stride between consecutive elements in X
 * @param Y Pointer to second vector data
 * @param INCY Stride between consecutive elements in Y
 * @return Inner product of the vectors
 */
template <typename T>
T inner(int32_t N, const T* X, int32_t INCX, const T* Y, int32_t INCY) {
  T prd = 0.;
  for (int32_t i = 0; i < N; ++i) {
    prd += detail::smart_conj<T>(X[i * INCX]) * Y[i * INCY];
  }
  return prd;
}

/**
 * @brief Linear combination with scaling: B = BETA * B + ALPHA * A (in-place)
 *
 * Performs the matrix operation B = BETA * B + ALPHA * A, storing the result
 * in matrix B. This is a generalized matrix scaling and addition operation
 * commonly used in iterative methods and linear algebra algorithms.
 *
 * @tparam T Numeric type (real or complex)
 * @param M Number of rows
 * @param N Number of columns
 * @param ALPHA Scaling factor for matrix A
 * @param A Source matrix A
 * @param LDA Leading dimension of matrix A
 * @param BETA Scaling factor for matrix B
 * @param B Target matrix B (modified in-place)
 * @param LDB Leading dimension of matrix B
 */
// B = BETA * B + ALPHA * A
template <typename T>
void laxpby(int32_t M, int32_t N, T ALPHA, const T* A, int32_t LDA, T BETA,
            T* B, int32_t LDB) {
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i)
      B[i + j * LDB] = BETA * B[i + j * LDB] + ALPHA * A[i + j * LDA];
}

/**
 * @brief Linear combination with scaling: C = BETA * B + ALPHA * A
 * (out-of-place)
 *
 * Performs the matrix operation C = BETA * B + ALPHA * A, storing the result
 * in a separate output matrix C. This version does not modify the input
 * matrices.
 *
 * @tparam T Numeric type (real or complex)
 * @param M Number of rows
 * @param N Number of columns
 * @param ALPHA Scaling factor for matrix A
 * @param A Source matrix A
 * @param LDA Leading dimension of matrix A
 * @param BETA Scaling factor for matrix B
 * @param B Source matrix B
 * @param LDB Leading dimension of matrix B
 * @param C Output matrix C
 * @param LDC Leading dimension of matrix C
 */
// C = BETA * B + ALPHA * A
template <typename T>
void laxpby(int32_t M, int32_t N, T ALPHA, const T* A, int32_t LDA, T BETA,
            const T* B, int32_t LDB, T* C, int32_t LDC) {
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i)
      C[i + j * LDC] = BETA * B[i + j * LDB] + ALPHA * A[i + j * LDA];
}

/**
 * @brief Scales matrix A by scalar ALPHA in-place
 *
 * Multiplies every element of matrix A by the scalar value ALPHA.
 * The operation is performed in-place, modifying the original matrix.
 *
 * @tparam T Numeric type (real or complex)
 * @param M Number of rows
 * @param N Number of columns
 * @param ALPHA Scaling factor
 * @param A Matrix to scale (modified in-place)
 * @param LDA Leading dimension of matrix A
 */
template <typename T>
void lascal(int32_t M, int32_t N, T ALPHA, T* A, int32_t LDA) {
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i) A[i + j * LDA] *= ALPHA;
}

/**
 * @brief Scales matrix A by scalar ALPHA and stores result in B
 *
 * Multiplies every element of matrix A by the scalar value ALPHA and
 * stores the result in matrix B. The original matrix A is not modified.
 *
 * @tparam T Numeric type (real or complex)
 * @param M Number of rows
 * @param N Number of columns
 * @param ALPHA Scaling factor
 * @param A Source matrix A
 * @param LDA Leading dimension of matrix A
 * @param B Output matrix B
 * @param LDB Leading dimension of matrix B
 */
template <typename T>
void lascal(int32_t M, int32_t N, T ALPHA, const T* A, int32_t LDA, T* B,
            int32_t LDB) {
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i) B[i + j * LDB] = A[i + j * LDA] * ALPHA;
}

/**
 * @brief Copies matrix A to matrix B
 *
 * Performs element-wise copy from matrix A to matrix B. Both matrices
 * must have the same dimensions but can have different leading dimensions.
 *
 * @tparam T Numeric type (real or complex)
 * @param M Number of rows
 * @param N Number of columns
 * @param A Source matrix A
 * @param LDA Leading dimension of matrix A
 * @param B Destination matrix B
 * @param LDB Leading dimension of matrix B
 */
template <typename T>
void lacpy(int32_t M, int32_t N, const T* A, int32_t LDA, T* B, int32_t LDB) {
  for (int32_t j = 0; j < N; ++j)
    for (int32_t i = 0; i < M; ++i) B[i + j * LDB] = A[i + j * LDA];
}

/**
 * @brief Solves linear system AX = B using LU decomposition (simplified
 * interface)
 *
 * Convenience wrapper that allocates pivot array internally and calls the
 * full gesv routine. This version is simpler to use when pivot information
 * is not needed by the caller.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 * @param N Order of matrix A and number of rows in B
 * @param NRHS Number of right-hand side vectors
 * @param A Coefficient matrix (overwritten with LU factorization)
 * @param LDA Leading dimension of matrix A
 * @param B Right-hand side matrix, overwritten with solution
 * @param LDB Leading dimension of matrix B
 *
 * @throws std::runtime_error If the LU factorization fails
 */
template <typename T>
void gesv(int32_t N, int32_t NRHS, T* A, int32_t LDA, T* B, int32_t LDB) {
  std::vector<int64_t> IPIV(N);
  lapack::gesv(N, NRHS, A, LDA, IPIV.data(), B, LDB);
}

/**
 * @brief Generates plane rotation for Givens rotations
 *
 * Wrapper around BLAS DROTG/ZROTG that constructs a plane rotation to
 * eliminate the second component of a 2-vector. Commonly used in QR
 * factorizations and iterative methods like GMRES.
 *
 * The rotation is defined by parameters c (cosine) and s (sine) such that:
 * [c  s] [x] = [r]
 * [-s c] [y]   [0]
 *
 * @tparam T Numeric type (double or std::complex<double>)
 * @param x First component of input vector
 * @param y Second component of input vector
 * @return Pair containing (c, s) rotation parameters
 *
 * @note For complex types, c is always real while s may be complex
 */
template <typename T>
auto rotg(T x, T y) {
  T c, s;

  double rc;
  blas::rotg(&x, &y, &rc, &s);
  c = rc;

  return std::pair(c, s);
}
}  // namespace qdk::chemistry::scf

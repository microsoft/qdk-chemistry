// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cassert>
#include <complex>

#include "type_traits.h"
#include "types.h"

extern "C" {

void dsyevd_(const char*, const char*, const int32_t*, double*, const int32_t*,
             double*, double*, const int32_t*, int32_t*, const int32_t*,
             int32_t*);
void dgeev_(const char*, const char*, const int32_t*, double*, const int32_t*,
            double*, double*, double*, const int32_t*, double*, const int32_t*,
            double*, const int32_t*, int32_t*);

void dgemm_(const char*, const char*, const int32_t*, const int32_t*,
            const int32_t*, const double*, const double*, const int32_t*,
            const double*, const int32_t*, const double*, double*,
            const int32_t*);
void zgemm_(const char*, const char*, const int32_t*, const int32_t*,
            const int32_t*, const std::complex<double>*,
            const std::complex<double>*, const int32_t*,
            const std::complex<double>*, const int32_t*,
            const std::complex<double>*, std::complex<double>*, const int32_t*);

void drotg_(const double*, const double*, double*, double*);
void zrotg_(const std::complex<double>*, const std::complex<double>*, double*,
            std::complex<double>*);

void dgesv_(const int32_t*, const int32_t*, double*, const int32_t*, int32_t*,
            double*, const int32_t*, int32_t*);
void zgesv_(const int32_t*, const int32_t*, std::complex<double>*,
            const int32_t*, int32_t*, std::complex<double>*, const int32_t*,
            int32_t*);
};

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
 * @brief General matrix multiplication: C = ALPHA * op(A) * op(B) + BETA * C
 *
 * Performs general matrix multiplication with optional transposition and
 * scaling. This is a wrapper around BLAS DGEMM/ZGEMM routines with automatic
 * type dispatch. The operation performed is: C = ALPHA * op(A) * op(B) + BETA *
 * C where op(X) can be X or X^T depending on the transpose flags.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 * @param TA Transpose flag for matrix A ('N' for no transpose, 'T' for
 * transpose)
 * @param TB Transpose flag for matrix B ('N' for no transpose, 'T' for
 * transpose)
 * @param M Number of rows in op(A) and C
 * @param N Number of columns in op(B) and C
 * @param K Number of columns in op(A) and rows in op(B)
 * @param ALPHA Scaling factor for the matrix product
 * @param A Matrix A
 * @param LDA Leading dimension of matrix A
 * @param B Matrix B
 * @param LDB Leading dimension of matrix B
 * @param BETA Scaling factor for matrix C
 * @param C Matrix C (input/output)
 * @param LDC Leading dimension of matrix C
 */
template <typename T>
void gemm(char TA, char TB, int32_t M, int32_t N, int32_t K, T ALPHA,
          const T* A, int32_t LDA, const T* B, int32_t LDB, T BETA, T* C,
          int32_t LDC) {
  if constexpr (std::is_same_v<T, double>)
    dgemm_(&TA, &TB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
  else
    zgemm_(&TA, &TB, &M, &N, &K, &ALPHA, A, &LDA, B, &LDB, &BETA, C, &LDC);
}

/**
 * @brief Computes eigenvalues and optionally eigenvectors of a symmetric matrix
 *
 * Wrapper around LAPACK DSYEVD for computing eigenvalues and eigenvectors of
 * a real symmetric matrix using a divide-and-conquer algorithm. This routine
 * is typically faster than the QR algorithm for large matrices.
 *
 * @param JOBZ 'N' to compute eigenvalues only, 'V' to compute eigenvalues and
 * eigenvectors
 * @param UPLO 'U' if upper triangle is stored, 'L' if lower triangle is stored
 * @param N Order of the matrix A
 * @param A Input symmetric matrix, on exit contains eigenvectors if JOBZ='V'
 * @param LDA Leading dimension of matrix A
 * @param W Output array of eigenvalues in ascending order
 *
 * @note The input matrix A is overwritten with eigenvectors if JOBZ='V'
 */
void syevd(char JOBZ, char UPLO, int32_t N, double* A, int32_t LDA, double* W) {
  std::vector<double> WORK(1);
  std::vector<int32_t> IWORK(1);

  int32_t INFO, LWORK = -1, LIWORK = -1;
  dsyevd_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK.data(), &LWORK, IWORK.data(),
          &LIWORK, &INFO);

  LWORK = int32_t(WORK[0]);
  LIWORK = int32_t(IWORK[0]);
  WORK.resize(LWORK);
  IWORK.resize(LIWORK);

  dsyevd_(&JOBZ, &UPLO, &N, A, &LDA, W, WORK.data(), &LWORK, IWORK.data(),
          &LIWORK, &INFO);
}

/**
 * @brief Computes eigenvalues and optionally eigenvectors of a general matrix
 *
 * Wrapper around LAPACK DGEEV for computing eigenvalues and eigenvectors of
 * a general real matrix using the QR algorithm. Handles non-symmetric matrices
 * and returns potentially complex eigenvalues as separate real and imaginary
 * parts.
 *
 * @param JOBVL 'N' to not compute left eigenvectors, 'V' to compute them
 * @param JOBVR 'N' to not compute right eigenvectors, 'V' to compute them
 * @param N Order of the matrix A
 * @param A Input matrix (overwritten during computation)
 * @param LDA Leading dimension of matrix A
 * @param WR Real parts of eigenvalues
 * @param WI Imaginary parts of eigenvalues
 * @param VL Left eigenvectors (if JOBVL='V')
 * @param LDVL Leading dimension of VL
 * @param VR Right eigenvectors (if JOBVR='V')
 * @param LDVR Leading dimension of VR
 *
 * @note Complex eigenvalues appear in conjugate pairs
 */
void geev(char JOBVL, char JOBVR, int32_t N, double* A, int32_t LDA, double* WR,
          double* WI, double* VL, int32_t LDVL, double* VR, int32_t LDVR) {
  std::vector<double> WORK(1);

  int32_t INFO, LWORK = -1, LIWORK = -1;
  dgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK.data(),
         &LWORK, &INFO);

  LWORK = int32_t(WORK[0]);
  WORK.resize(LWORK);

  dgeev_(&JOBVL, &JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, WORK.data(),
         &LWORK, &INFO);
}

/**
 * @brief Solves linear system AX = B using LU decomposition with partial
 * pivoting
 *
 * Wrapper around LAPACK DGESV/ZGESV that solves the linear system AX = B
 * by computing the LU factorization of A and using it to solve the system.
 * The matrix A is overwritten with its LU factorization.
 *
 * @tparam T Numeric type (double or std::complex<double>)
 * @param N Order of matrix A and number of rows in B
 * @param NRHS Number of right-hand side vectors
 * @param A Coefficient matrix (overwritten with LU factorization)
 * @param LDA Leading dimension of matrix A
 * @param IPIV Pivot indices from LU factorization
 * @param B Right-hand side matrix, overwritten with solution
 * @param LDB Leading dimension of matrix B
 *
 * @throws std::runtime_error If the LU factorization fails
 */
template <typename T>
void gesv(int32_t N, int32_t NRHS, T* A, int32_t LDA, int32_t* IPIV, T* B,
          int32_t LDB) {
  int32_t INFO;
  if constexpr (std::is_same_v<double, T>)
    dgesv_(&N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
  else
    zgesv_(&N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
  if (INFO) throw std::runtime_error("GESV FAILED");
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
  std::vector<int32_t> IPIV(N);
  gesv(N, NRHS, A, LDA, IPIV.data(), B, LDB);
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

  if constexpr (std::is_same_v<T, double>)
    drotg_(&x, &y, &c, &s);
  else {
    double rc;
    zrotg_(&x, &y, &rc, &s);
    c = rc;
  }

  return std::pair(c, s);
}
}  // namespace qdk::chemistry::scf

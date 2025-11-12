// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

namespace qdk::chemistry::scf::lapack {

/**
 * @brief LU factorization of a general matrix
 * @see https://netlib.org/lapack/explore-html/db/d04/group__getrf.html for
 * details
 */
void getrf(int M, int N, double* A, int LDA, int* IPIV);

/**
 * @brief Solve system using LU factorization
 * @see https://netlib.org/lapack/explore-html/df/d36/group__getrs.html for
 * details
 */
void getrs(const char* TRANS, int N, int NRHS, const double* A, int LDA,
           const int* IPIV, double* B, int LDB);

/**
 * @brief Cholesky factorization of a symmetric positive definite matrix
 * @see https://netlib.org/lapack/explore-html/d2/d09/group__potrf.html for
 * details
 */
void potrf(const char* UPLO, int N, double* A, int LDA);

/**
 * @brief Solve system using Cholesky factorization
 * @see https://netlib.org/lapack/explore-html/d3/dc8/group__potrs.html for
 * details
 */
void potrs(const char* UPLO, int N, int NRHS, const double* A, int LDA,
           double* B, int LDB);

/**
 * @brief Compute eigenvalues and eigenvectors of symmetric matrix
 * @see https://netlib.org/lapack/explore-html/d8/d1c/group__heev.html for
 * details
 */
void syev(const char* JOBZ, const char* UPLO, int N, double* A, int LDA,
          double* W);

/**
 * @brief Compute eigenvalues and eigenvectors of general matrix
 * @see https://netlib.org/lapack/explore-html/d4/d68/group__geev.html for
 * details
 */
void geev(const char* JOBVL, const char* JOBVR, int N, double* A, int LDA,
          double* WR, double* WI, double* VL, int LDVL, double* VR, int LDVR);

/**
 * @brief Solve general linear system A*X=B using LU factorization
 * @see https://netlib.org/lapack/explore-html/d8/da6/group__gesv.html for
 * details
 */
void gesv(int N, int NRHS, double* A, int LDA, int* IPIV, double* B, int LDB);

/**
 * @brief Solve linear least squares problem using DGELSS
 * @see
 * https://netlib.org/lapack/explore-html/da/d55/group__gelss_gac6159de3953ae0386c2799294745ac90.html
 * for details
 */
int dgelss(int M, int N, int NRHS, double* A, int LDA, double* B, int LDB,
           double* S, double RCOND);

}  // namespace qdk::chemistry::scf::lapack

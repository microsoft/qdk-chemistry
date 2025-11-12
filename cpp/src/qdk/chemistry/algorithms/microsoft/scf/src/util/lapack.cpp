// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "util/lapack.h"

#include <stdexcept>

/**
 * @brief External LAPACK function declarations for Fortran interface
 *
 * These declarations provide C++ access to Fortran LAPACK routines for
 * linear algebra operations. The underscore suffix follows the standard
 * Fortran name mangling convention. All parameters are passed by reference
 * as required by the Fortran calling convention.
 *
 * @note These are low-level Fortran interface declarations. Use the C++
 *       wrapper functions below for type-safe and exception-safe access.
 */
extern "C" {
void dpotrf_(const char* UPLO, const int* N, double* A, const int* LDA,
             int* INFO);
void dgetrf_(const int* M, const int* N, double* A, const int* LDA, int* IPIV,
             int* INFO);
void dgetrs_(const char* TRANS, const int* N, const int* NRHS, const double* A,
             const int* LDA, const int* IPIV, double* B, const int* LDB,
             int* INFO);
void dpotrs_(const char* UPLO, const int* N, const int* NRHS, const double* A,
             const int* LDA, double* B, const int* LDB, int* INFO);
void dsyev_(const char* JOBZ, const char* UPLO, const int* N, double* A,
            const int* LDA, double* W, double* WORK, const int* LWORK,
            int* INFO);
void dgeev_(const char* JOBVL, const char* JOBVR, const int* N, double* A,
            const int* LDA, double* WR, double* WI, double* VL, const int* LDVL,
            double* VR, const int* LDVR, double* WORK, const int* LWORK,
            int* INFO);
void dgesv_(const int* N, const int* NRHS, double* A, const int* LDA, int* IPIV,
            double* B, const int* LDB, int* INFO);
void dgelss_(const int* M, const int* N, const int* NRHS, double* A,
             const int* LDA, double* B, const int* LDB, double* S,
             const double* RCOND, int* RANK, double* WORK, const int* LWORK,
             int* INFO);
}

namespace qdk::chemistry::scf::lapack {

void getrf(int M, int N, double* A, int LDA, int* IPIV) {
  int INFO;
  dgetrf_(&M, &N, A, &LDA, IPIV, &INFO);
  if (INFO)
    throw std::runtime_error("DGETRF Failed with INFO = " +
                             std::to_string(INFO));
}

void getrs(const char* TRANS, int N, int NRHS, const double* A, int LDA,
           const int* IPIV, double* B, int LDB) {
  int INFO;
  dgetrs_(TRANS, &N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
  if (INFO)
    throw std::runtime_error("DGETRS Failed with INFO = " +
                             std::to_string(INFO));
}

void potrf(const char* UPLO, int N, double* A, int LDA) {
  int INFO;
  dpotrf_(UPLO, &N, A, &LDA, &INFO);
  if (INFO)
    throw std::runtime_error("DPOTRF Failed with INFO = " +
                             std::to_string(INFO));
}

void potrs(const char* UPLO, int N, int NRHS, const double* A, int LDA,
           double* B, int LDB) {
  int INFO;
  dpotrs_(UPLO, &N, &NRHS, A, &LDA, B, &LDB, &INFO);
  if (INFO)
    throw std::runtime_error("DPOTRS Failed with INFO = " +
                             std::to_string(INFO));
}

void syev(const char* JOBZ, const char* UPLO, int N, double* A, int LDA,
          double* W) {
  int INFO, lwork = -1;
  double wkopt;
  double* work;
  // Query and allocate the optimal workspace
  dsyev_(JOBZ, UPLO, &N, A, &LDA, W, &wkopt, &lwork, &INFO);
  lwork = (int)wkopt;
  work = new double[lwork];
  // Solve the eigenvalue problem
  dsyev_(JOBZ, UPLO, &N, A, &LDA, W, work, &lwork, &INFO);
  if (INFO)
    throw std::runtime_error("DSYEV Failed with INFO = " +
                             std::to_string(INFO));
  delete[] work;
}

void geev(const char* JOBVL, const char* JOBVR, int N, double* A, int LDA,
          double* WR, double* WI, double* VL, int LDVL, double* VR, int LDVR) {
  int INFO, lwork = -1;
  double wkopt;
  double* work;
  // Query and allocate the optimal workspace
  dgeev_(JOBVL, JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, &wkopt,
         &lwork, &INFO);
  lwork = (int)wkopt;
  work = new double[lwork];
  // Solve the eigenvalue problem
  dgeev_(JOBVL, JOBVR, &N, A, &LDA, WR, WI, VL, &LDVL, VR, &LDVR, work, &lwork,
         &INFO);
  if (INFO)
    throw std::runtime_error("DGEEV Failed with INFO = " +
                             std::to_string(INFO));
  delete[] work;
}

void gesv(int N, int NRHS, double* A, int LDA, int* IPIV, double* B, int LDB) {
  int INFO;
  dgesv_(&N, &NRHS, A, &LDA, IPIV, B, &LDB, &INFO);
  if (INFO)
    throw std::runtime_error("DGESV Failed with INFO = " +
                             std::to_string(INFO));
}

int dgelss(int M, int N, int NRHS, double* A, int LDA, double* B, int LDB,
           double* S, double RCOND) {
  int LWORK = -1, INFO, RANK;
  double wkopt;
  double* work;

  dgelss_(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, &wkopt, &LWORK,
          &INFO);
  LWORK = int(wkopt);
  work = new double[LWORK];
  dgelss_(&M, &N, &NRHS, A, &LDA, B, &LDB, S, &RCOND, &RANK, work, &LWORK,
          &INFO);
  if (INFO)
    throw std::runtime_error("DGELSS FAILED WITH INFO = " +
                             std::to_string(INFO));
  delete[] work;

  return RANK;
}

}  // namespace qdk::chemistry::scf::lapack

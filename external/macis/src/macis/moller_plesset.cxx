/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <iostream>
#include <lapack.hh>
#include <macis/mcscf/orbital_energies.hpp>
#include <macis/util/moller_plesset.hpp>

namespace macis {

void mp2_t2(NumCanonicalOccupied _nocc, NumCanonicalVirtual _nvir,
            const double* V, size_t LDV, const double* eps, double* T2) {
  const size_t nocc = _nocc.get();
  const size_t nvir = _nvir.get();

  const size_t nocc2 = nocc * nocc;
  const size_t nocc2v = nocc2 * nvir;
  const size_t LDV2 = LDV * LDV;
  const size_t LDV3 = LDV2 * LDV;

  // T2(i,j,a,b) = (ia|jb) / (eps[i] + eps[j] - eps[a] - eps[b])
  for (auto i = 0ul; i < nocc; ++i)
    for (auto j = 0ul; j < nocc; ++j)
      for (auto a = 0ul; a < nvir; ++a)
        for (auto b = 0ul; b < nvir; ++b) {
          const auto a_off = a + nocc;
          const auto b_off = b + nocc;

          T2[i + j * nocc + a * nocc2 + b * nocc2v] =
              V[i + a_off * LDV + j * LDV2 + b_off * LDV3] /
              // #TODO: Denominator can hit zero for near-degenerate orbitals;
              // add guard or regularization.
              (eps[i] + eps[j] - eps[a_off] - eps[b_off]);
        }
}

void mp2_1rdm(NumOrbital _norb, NumCanonicalOccupied _nocc,
              NumCanonicalVirtual _nvir, const double* T, size_t LDT,
              const double* V, size_t LDV, double* ORDM, size_t LDD) {
  const size_t norb = _norb.get();
  const size_t nocc = _nocc.get();
  const size_t nvir = _nvir.get();

  const size_t nocc2 = nocc * nocc;
  const size_t nocc2v = nocc2 * nvir;
  const size_t LDV2 = LDV * LDV;
  const size_t LDV3 = LDV2 * LDV;

  // Compute canonical eigenenergies
  // XXX: This will not generally replicate full precision
  // with respect to those returned by the eigen solver
  std::vector<double> eps(norb);
  canonical_orbital_energies(_norb, NumInactive(nocc), T, LDT, V, LDV,
                             eps.data());

  // Compute T2
  std::vector<double> T2(nocc2v * nvir);
  mp2_t2(_nocc, _nvir, V, LDV, eps.data(), T2.data());

  // Compute MP2 energy XXX: This is not required, just a check
  double EMP2 = 0.0;
  for (auto i = 0ul; i < nocc; ++i)
    for (auto j = 0ul; j < nocc; ++j)
      for (auto a = 0ul; a < nvir; ++a)
        for (auto b = 0ul; b < nvir; ++b) {
          const auto a_off = a + nocc;
          const auto b_off = b + nocc;
          const double V_abij = V[a_off + i * norb + b_off * LDV2 + j * LDV3];
          const double V_abji = V[a_off + j * norb + b_off * LDV2 + i * LDV3];

          const double t2_ijab = T2[i + j * nocc + a * nocc2 + b * nocc2v];
          EMP2 += t2_ijab * (2 * V_abij - V_abji);
        }

  // #TODO: Replace std::cout with the project logger and allow disabling this
  // expensive diagnostic in production.
  std::cout << "EMP2 = " << EMP2 << std::endl;

  // P(MP2) OO-block
  // D(i,j) -= T2(i,k,a,b) * (2*T2(j,k,a,b) - T2(j,k,b,a))
  for (auto i = 0ul; i < nocc; ++i)
    for (auto j = 0ul; j < nocc; ++j) {
      double tmp = 0.0;
      for (auto k = 0ul; k < nocc; ++k)
        for (auto a = 0ul; a < nvir; ++a)
          for (auto b = 0ul; b < nvir; ++b) {
            tmp += T2[i + k * nocc + a * nocc2 + b * nocc2v] *
                   (2 * T2[j + k * nocc + a * nocc2 + b * nocc2v] -
                    T2[j + k * nocc + b * nocc2 + a * nocc2v]);
          }
      ORDM[i + j * LDD] = -2 * tmp;
      if (i == j) ORDM[i + j * LDD] += 2.0;  // HF contribution
    }

  // P(MP2) VV-block
  // D(a,b) -= T2(i,j,c,a) * (2*T2(i,j,c,b) - T2(i,j,b,c))
  for (auto a = 0ul; a < nvir; ++a)
    for (auto b = 0ul; b < nvir; ++b) {
      double tmp = 0;
      for (auto i = 0ul; i < nocc; ++i)
        for (auto j = 0ul; j < nocc; ++j)
          for (auto c = 0ul; c < nvir; ++c) {
            tmp += T2[i + j * nocc + c * nocc2 + a * nocc2v] *
                   (2 * T2[i + j * nocc + c * nocc2 + b * nocc2v] -
                    T2[i + j * nocc + b * nocc2 + c * nocc2v]);
          }
      ORDM[a + nocc + (b + nocc) * LDD] = 2 * tmp;
    }
}

void mp2_natural_orbitals(NumOrbital norb, NumCanonicalOccupied nocc,
                          NumCanonicalVirtual nvir, const double* T, size_t LDT,
                          const double* V, size_t LDV, double* ON, double* NO_C,
                          size_t LDC) {
  // Compute MP2 1-RDM
  mp2_1rdm(norb, nocc, nvir, T, LDT, V, LDV, NO_C, LDC);

  // Compute MP2 Natural Orbitals

  // 1. First negate to ensure diagonalization sorts eigenvalues in
  //    decending order
  for (size_t i = 0; i < norb.get(); ++i)
    for (size_t j = 0; j < norb.get(); ++j) {
      NO_C[i + j * LDC] *= -1.0;
    }

  // 2. Solve eigenvalue problem PC = CO
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, norb.get(), NO_C, LDC,
               ON);

  // 3. Undo negation
  for (size_t i = 0; i < norb.get(); ++i) ON[i] *= -1.0;
}

}  // namespace macis

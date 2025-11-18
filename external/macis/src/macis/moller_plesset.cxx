/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <iostream>
#include <lapack.hh>
#include <macis/mcscf/orbital_energies.hpp>
#include <macis/util/moller_plesset.hpp>

namespace macis {

void mp2_t2(NumCanonicalOccupied _num_occupied_orbitals,
            NumCanonicalVirtual _num_virtual_orbitals, const double* V,
            size_t LDV, const double* eps, double* T2, double shift) {
  const size_t num_occupied_orbitals = _num_occupied_orbitals.get();
  const size_t num_virtual_orbitals = _num_virtual_orbitals.get();

  const size_t num_occupied_orbitals2 =
      num_occupied_orbitals * num_occupied_orbitals;
  const size_t nocc2v = num_occupied_orbitals2 * num_virtual_orbitals;
  const size_t LDV2 = LDV * LDV;
  const size_t LDV3 = LDV2 * LDV;

  // T2(i,j,a,b) = (ia|jb) / (eps[i] + eps[j] - eps[a] - eps[b])
  for (auto i = 0ul; i < num_occupied_orbitals; ++i)
    for (auto j = 0ul; j < num_occupied_orbitals; ++j)
      for (auto a = 0ul; a < num_virtual_orbitals; ++a)
        for (auto b = 0ul; b < num_virtual_orbitals; ++b) {
          const auto a_off = a + num_occupied_orbitals;
          const auto b_off = b + num_occupied_orbitals;

          T2[i + j * num_occupied_orbitals + a * num_occupied_orbitals2 +
             b * nocc2v] = V[i + a_off * LDV + j * LDV2 + b_off * LDV3] /
                           (eps[i] + eps[j] - eps[a_off] - eps[b_off] + shift);
        }
}

void mp2_1rdm(NumOrbital _norb, NumCanonicalOccupied _num_occupied_orbitals,
              NumCanonicalVirtual _num_virtual_orbitals, const double* T,
              size_t LDT, const double* V, size_t LDV, double* ORDM, size_t LDD,
              double shift) {
  const size_t norb = _norb.get();
  const size_t num_occupied_orbitals = _num_occupied_orbitals.get();
  const size_t num_virtual_orbitals = _num_virtual_orbitals.get();

  const size_t num_occupied_orbitals2 =
      num_occupied_orbitals * num_occupied_orbitals;
  const size_t nocc2v = num_occupied_orbitals2 * num_virtual_orbitals;
  const size_t LDV2 = LDV * LDV;
  const size_t LDV3 = LDV2 * LDV;

  // Compute canonical eigenenergies
  // XXX: This will not generally replicate full precision
  // with respect to those returned by the eigen solver
  std::vector<double> eps(norb);
  canonical_orbital_energies(_norb, NumInactive(num_occupied_orbitals), T, LDT,
                             V, LDV, eps.data());

  // Compute T2
  std::vector<double> T2(nocc2v * num_virtual_orbitals);
  mp2_t2(_num_occupied_orbitals, _num_virtual_orbitals, V, LDV, eps.data(),
         T2.data(), shift);

  // P(MP2) OO-block
  // D(i,j) -= T2(i,k,a,b) * (2*T2(j,k,a,b) - T2(j,k,b,a))
  for (auto i = 0ul; i < num_occupied_orbitals; ++i)
    for (auto j = 0ul; j < num_occupied_orbitals; ++j) {
      double tmp = 0.0;
      for (auto k = 0ul; k < num_occupied_orbitals; ++k)
        for (auto a = 0ul; a < num_virtual_orbitals; ++a)
          for (auto b = 0ul; b < num_virtual_orbitals; ++b) {
            tmp += T2[i + k * num_occupied_orbitals +
                      a * num_occupied_orbitals2 + b * nocc2v] *
                   (2 * T2[j + k * num_occupied_orbitals +
                           a * num_occupied_orbitals2 + b * nocc2v] -
                    T2[j + k * num_occupied_orbitals +
                       b * num_occupied_orbitals2 + a * nocc2v]);
          }
      ORDM[i + j * LDD] = -2 * tmp;
      if (i == j) ORDM[i + j * LDD] += 2.0;  // HF contribution
    }

  // P(MP2) VV-block
  // D(a,b) -= T2(i,j,c,a) * (2*T2(i,j,c,b) - T2(i,j,b,c))
  for (auto a = 0ul; a < num_virtual_orbitals; ++a)
    for (auto b = 0ul; b < num_virtual_orbitals; ++b) {
      double tmp = 0;
      for (auto i = 0ul; i < num_occupied_orbitals; ++i)
        for (auto j = 0ul; j < num_occupied_orbitals; ++j)
          for (auto c = 0ul; c < num_virtual_orbitals; ++c) {
            tmp += T2[i + j * num_occupied_orbitals +
                      c * num_occupied_orbitals2 + a * nocc2v] *
                   (2 * T2[i + j * num_occupied_orbitals +
                           c * num_occupied_orbitals2 + b * nocc2v] -
                    T2[i + j * num_occupied_orbitals +
                       b * num_occupied_orbitals2 + c * nocc2v]);
          }
      ORDM[a + num_occupied_orbitals + (b + num_occupied_orbitals) * LDD] =
          2 * tmp;
    }
}

void mp2_natural_orbitals(NumOrbital norb,
                          NumCanonicalOccupied num_occupied_orbitals,
                          NumCanonicalVirtual num_virtual_orbitals,
                          const double* T, size_t LDT, const double* V,
                          size_t LDV, double* ON, double* NO_C, size_t LDC,
                          double shift) {
  // Compute MP2 1-RDM
  mp2_1rdm(norb, num_occupied_orbitals, num_virtual_orbitals, T, LDT, V, LDV,
           NO_C, LDC, shift);

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

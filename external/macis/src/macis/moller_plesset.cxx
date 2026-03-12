/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <spdlog/spdlog.h>

#include <chrono>
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
  auto t0_eps = std::chrono::high_resolution_clock::now();
  std::vector<double> eps(norb);
  canonical_orbital_energies(_norb, NumInactive(num_occupied_orbitals), T, LDT,
                             V, LDV, eps.data());
  auto t1_eps = std::chrono::high_resolution_clock::now();

  // Compute T2
  auto t0_t2 = std::chrono::high_resolution_clock::now();
  std::vector<double> T2(nocc2v * num_virtual_orbitals);
  mp2_t2(_num_occupied_orbitals, _num_virtual_orbitals, V, LDV, eps.data(),
         T2.data(), shift);
  auto t1_t2 = std::chrono::high_resolution_clock::now();

  // P(MP2) OO-block
  // D(i,j) -= T2(i,k,a,b) * (2*T2(j,k,a,b) - T2(j,k,b,a))
  auto t0_oo = std::chrono::high_resolution_clock::now();
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
  auto t1_oo = std::chrono::high_resolution_clock::now();

  // P(MP2) VV-block
  // D(a,b) -= T2(i,j,c,a) * (2*T2(i,j,c,b) - T2(i,j,b,c))
  auto t0_vv = std::chrono::high_resolution_clock::now();
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
  auto t1_vv = std::chrono::high_resolution_clock::now();

  spdlog::info("[macis::mp2_1rdm] n_occ={} n_virt={} n_orb={}",
               num_occupied_orbitals, num_virtual_orbitals, norb);
  spdlog::info(
      "[macis::mp2_1rdm]   canonical_energies: {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_eps - t0_eps).count());
  spdlog::info(
      "[macis::mp2_1rdm]   t2_amplitudes:      {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_t2 - t0_t2).count());
  spdlog::info(
      "[macis::mp2_1rdm]   1rdm_oo_block:      {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_oo - t0_oo).count());
  spdlog::info(
      "[macis::mp2_1rdm]   1rdm_vv_block:      {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_vv - t0_vv).count());
}

void mp2_natural_orbitals(NumOrbital norb,
                          NumCanonicalOccupied num_occupied_orbitals,
                          NumCanonicalVirtual num_virtual_orbitals,
                          const double* T, size_t LDT, const double* V,
                          size_t LDV, double* ON, double* NO_C, size_t LDC,
                          double shift) {
  auto t0_total = std::chrono::high_resolution_clock::now();

  // Compute MP2 1-RDM
  auto t0_1rdm = std::chrono::high_resolution_clock::now();
  mp2_1rdm(norb, num_occupied_orbitals, num_virtual_orbitals, T, LDT, V, LDV,
           NO_C, LDC, shift);
  auto t1_1rdm = std::chrono::high_resolution_clock::now();

  // Compute MP2 Natural Orbitals

  // 1. First negate to ensure diagonalization sorts eigenvalues in
  //    decending order
  for (size_t i = 0; i < norb.get(); ++i)
    for (size_t j = 0; j < norb.get(); ++j) {
      NO_C[i + j * LDC] *= -1.0;
    }

  // 2. Solve eigenvalue problem PC = CO
  auto t0_syev = std::chrono::high_resolution_clock::now();
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, norb.get(), NO_C, LDC,
               ON);
  auto t1_syev = std::chrono::high_resolution_clock::now();

  // 3. Undo negation
  for (size_t i = 0; i < norb.get(); ++i) ON[i] *= -1.0;

  auto t1_total = std::chrono::high_resolution_clock::now();
  spdlog::info("[macis::mp2_natural_orbitals] n_orb={}", norb.get());
  spdlog::info(
      "[macis::mp2_natural_orbitals]   1rdm_total: {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_1rdm - t0_1rdm).count());
  spdlog::info(
      "[macis::mp2_natural_orbitals]   syev:       {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_syev - t0_syev).count());
  spdlog::info(
      "[macis::mp2_natural_orbitals]   total:      {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_total - t0_total).count());
}

void mp2_t2_ov(NumCanonicalOccupied _nocc, NumCanonicalVirtual _nvirt,
               const double* V_iajb, const double* eps_occ,
               const double* eps_virt, double* T2, double shift) {
  const size_t nocc = _nocc.get();
  const size_t nvirt = _nvirt.get();
  const size_t nocc2 = nocc * nocc;
  const size_t nocc2v = nocc2 * nvirt;

  // V_iajb layout: V_iajb[i + a*nocc + j*nocc*nvirt + b*nocc*nvirt*nocc]
  const size_t LDV1 = nocc;          // stride for a
  const size_t LDV2 = nocc * nvirt;  // stride for j
  const size_t LDV3 = LDV2 * nocc;   // stride for b

  // T2 layout: T2[i + j*nocc + a*nocc² + b*nocc²*nvirt]
  for (size_t i = 0; i < nocc; ++i)
    for (size_t j = 0; j < nocc; ++j)
      for (size_t a = 0; a < nvirt; ++a)
        for (size_t b = 0; b < nvirt; ++b) {
          T2[i + j * nocc + a * nocc2 + b * nocc2v] =
              V_iajb[i + a * LDV1 + j * LDV2 + b * LDV3] /
              (eps_occ[i] + eps_occ[j] - eps_virt[a] - eps_virt[b] + shift);
        }
}

void mp2_natural_orbitals_ov(NumOrbital _norb, NumCanonicalOccupied _nocc,
                             NumCanonicalVirtual _nvirt, const double* eps_occ,
                             const double* eps_virt, const double* V_iajb,
                             double* ON, double* NO_C, size_t LDC,
                             double shift) {
  const size_t norb = _norb.get();
  const size_t nocc = _nocc.get();
  const size_t nvirt = _nvirt.get();
  const size_t nocc2 = nocc * nocc;
  const size_t nocc2v = nocc2 * nvirt;

  auto t0_total = std::chrono::high_resolution_clock::now();

  // Compute T2 amplitudes
  auto t0_t2 = std::chrono::high_resolution_clock::now();
  std::vector<double> T2(nocc2 * nvirt * nvirt);
  mp2_t2_ov(_nocc, _nvirt, V_iajb, eps_occ, eps_virt, T2.data(), shift);
  auto t1_t2 = std::chrono::high_resolution_clock::now();

  // Zero the 1-RDM (stored in NO_C)
  for (size_t i = 0; i < norb; ++i)
    for (size_t j = 0; j < norb; ++j) NO_C[i + j * LDC] = 0.0;

  // OO-block: D(i,j) = -2 * sum_{kab} T2(i,k,a,b) *
  //                     (2*T2(j,k,a,b) - T2(j,k,b,a)) + 2*delta(i,j)
  auto t0_oo = std::chrono::high_resolution_clock::now();
  for (size_t i = 0; i < nocc; ++i)
    for (size_t j = 0; j < nocc; ++j) {
      double tmp = 0.0;
      for (size_t k = 0; k < nocc; ++k)
        for (size_t a = 0; a < nvirt; ++a)
          for (size_t b = 0; b < nvirt; ++b) {
            tmp += T2[i + k * nocc + a * nocc2 + b * nocc2v] *
                   (2 * T2[j + k * nocc + a * nocc2 + b * nocc2v] -
                    T2[j + k * nocc + b * nocc2 + a * nocc2v]);
          }
      NO_C[i + j * LDC] = -2 * tmp;
      if (i == j) NO_C[i + j * LDC] += 2.0;
    }
  auto t1_oo = std::chrono::high_resolution_clock::now();

  // VV-block: D(a,b) = 2 * sum_{ijc} T2(i,j,c,a) *
  //                     (2*T2(i,j,c,b) - T2(i,j,b,c))
  auto t0_vv = std::chrono::high_resolution_clock::now();
  for (size_t a = 0; a < nvirt; ++a)
    for (size_t b = 0; b < nvirt; ++b) {
      double tmp = 0.0;
      for (size_t i = 0; i < nocc; ++i)
        for (size_t j = 0; j < nocc; ++j)
          for (size_t c = 0; c < nvirt; ++c) {
            tmp += T2[i + j * nocc + c * nocc2 + a * nocc2v] *
                   (2 * T2[i + j * nocc + c * nocc2 + b * nocc2v] -
                    T2[i + j * nocc + b * nocc2 + c * nocc2v]);
          }
      NO_C[a + nocc + (b + nocc) * LDC] = 2 * tmp;
    }
  auto t1_vv = std::chrono::high_resolution_clock::now();

  // Negate for descending eigenvalue order
  for (size_t i = 0; i < norb; ++i)
    for (size_t j = 0; j < norb; ++j) NO_C[i + j * LDC] *= -1.0;

  // Diagonalize
  auto t0_syev = std::chrono::high_resolution_clock::now();
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, norb, NO_C, LDC, ON);
  auto t1_syev = std::chrono::high_resolution_clock::now();

  // Undo negation of eigenvalues
  for (size_t i = 0; i < norb; ++i) ON[i] *= -1.0;

  auto t1_total = std::chrono::high_resolution_clock::now();
  spdlog::info("[macis::mp2_natural_orbitals_ov] n_occ={} n_virt={} n_orb={}",
               nocc, nvirt, norb);
  spdlog::info(
      "[macis::mp2_natural_orbitals_ov]   t2_amplitudes:  {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_t2 - t0_t2).count());
  spdlog::info(
      "[macis::mp2_natural_orbitals_ov]   1rdm_oo_block:  {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_oo - t0_oo).count());
  spdlog::info(
      "[macis::mp2_natural_orbitals_ov]   1rdm_vv_block:  {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_vv - t0_vv).count());
  spdlog::info(
      "[macis::mp2_natural_orbitals_ov]   syev:           {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_syev - t0_syev).count());
  spdlog::info(
      "[macis::mp2_natural_orbitals_ov]   total:          {:.3f} ms",
      std::chrono::duration<double, std::milli>(t1_total - t0_total).count());
}

}  // namespace macis

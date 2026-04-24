// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

// Test compute_s_squared for a 2-electron, 2-orbital RHF singlet wavefunction.
//
// For RHF with one doubly-occupied orbital out of norbs=2:
//   gamma^a_{00} = gamma^b_{00} = 1,  gamma^a_{11} = gamma^b_{11} = 0
//
// 2-RDM (QDK convention): Gamma(p,q,r,s) = <a†_p a†_r a_s a_q>
//   For a single determinant:
//   aabb block: Gamma^{aabb}(p,q,r,s) = gamma^a(p,q) * gamma^b(r,s)
//   aaaa block: Gamma^{aaaa}(p,q,r,s) = gamma^a(p,q) * gamma^a(r,s) -
//   gamma^a(p,s) * gamma^a(r,q) bbbb block: Gamma^{bbbb}(p,q,r,s) =
//   gamma^b(p,q) * gamma^b(r,s) - gamma^b(p,s) * gamma^b(r,q)
//
// Expected <S^2> = 0 for a singlet.
TEST(SSquared, RHFSinglet) {
  const int norbs = 2;
  const int norbs4 = norbs * norbs * norbs * norbs;

  // 1-RDMs: only orbital 0 occupied
  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  one_rdm_aa(0, 0) = 1.0;
  one_rdm_bb(0, 0) = 1.0;

  // Build 2-RDMs from 1-RDMs
  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  for (int p = 0; p < norbs; ++p)
    for (int q = 0; q < norbs; ++q)
      for (int r = 0; r < norbs; ++r)
        for (int s = 0; s < norbs; ++s) {
          int idx =
              p * norbs * norbs * norbs + q * norbs * norbs + r * norbs + s;
          // aabb: no exchange (different spins)
          two_rdm_aabb[idx] = one_rdm_aa(p, q) * one_rdm_bb(r, s);
          // aaaa: with exchange
          two_rdm_aaaa[idx] = one_rdm_aa(p, q) * one_rdm_aa(r, s) -
                              one_rdm_aa(p, s) * one_rdm_aa(r, q);
          // bbbb: with exchange
          two_rdm_bbbb[idx] = one_rdm_bb(p, q) * one_rdm_bb(r, s) -
                              one_rdm_bb(p, s) * one_rdm_bb(r, q);
        }

  // Create wavefunction with these RDMs
  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  Eigen::VectorXd coeffs(1);
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets = {Configuration("ud")};

  auto wf = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals,
      std::nullopt,                        // no spin-traced 1-RDM
      std::make_optional(one_rdm_aa),      // alpha 1-RDM
      std::make_optional(one_rdm_bb),      // beta 1-RDM
      std::nullopt,                        // no spin-traced 2-RDM
      std::make_optional(two_rdm_aabb),    // mixed-spin 2-RDM
      std::make_optional(two_rdm_aaaa),    // alpha-alpha 2-RDM
      std::make_optional(two_rdm_bbbb)));  // beta-beta 2-RDM

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.0, 1e-12);
}

// Test compute_s_squared for a single alpha electron (doublet, S=1/2).
//
// 1 electron in 1 orbital, alpha only.
//   gamma^a_{00} = 1, gamma^b_{00} = 0
//   All 2-RDMs are zero (only 1 electron).
//
// Expected <S^2> = S(S+1) = 0.75
TEST(SSquared, SingleElectronDoublet) {
  const int norbs = 1;
  const int norbs4 = 1;

  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  one_rdm_aa(0, 0) = 1.0;

  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  auto orbitals = testing::create_test_orbitals(2, norbs, true);

  Eigen::VectorXd coeffs(1);
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets = {Configuration("u")};

  auto wf = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.75, 1e-12);
}

// Test compute_s_squared for a triplet state (S=1, M_S=0).
//
// 2 orbitals, 2 electrons. The triplet M_S=0 state is:
//   |T,0> = (1/sqrt(2)) (|ud> - |du>)    -- but as orbital occupations:
//   |T,0> = (1/sqrt(2)) (a†_{0α} a†_{1β} - a†_{0β} a†_{1α}) |vac>
//
// 1-RDMs: gamma^a_{00} = gamma^a_{11} = 0.5, gamma^b_{00} = gamma^b_{11} = 0.5
//
// Expected <S^2> = S(S+1) = 2.0
TEST(SSquared, TripletMSZero) {
  const int norbs = 2;
  const int norbs4 = norbs * norbs * norbs * norbs;

  // 1-RDMs for triplet M_S=0 (each orbital half-occupied in each spin)
  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  one_rdm_aa(0, 0) = 0.5;
  one_rdm_aa(1, 1) = 0.5;
  one_rdm_bb(0, 0) = 0.5;
  one_rdm_bb(1, 1) = 0.5;

  // 2-RDMs for the triplet M_S=0 state (QDK convention)
  // |T,0> = (1/sqrt(2)) (|0α,1β> - |1α,0β>)
  //
  // Gamma^{aabb}(p,q,r,s) = <a†_{pα} a†_{rβ} a_{sβ} a_{qα}>
  //
  // For the triplet M_S=0 state, same-spin 2-RDMs are zero (one electron per
  // spin).
  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  // Non-zero aabb elements for |T,0> = (1/sqrt(2))(|0α,1β> - |1α,0β>):
  //
  // Gamma^{aabb}(0,0,1,1) = <a†_{0α} a†_{1β} a_{1β} a_{0α}> =  0.5 (Coulomb)
  // Gamma^{aabb}(1,1,0,0) = <a†_{1α} a†_{0β} a_{0β} a_{1α}> =  0.5 (Coulomb)
  // Gamma^{aabb}(0,1,1,0) = <a†_{0α} a†_{1β} a_{0β} a_{1α}> = -0.5 (exchange)
  // Gamma^{aabb}(1,0,0,1) = <a†_{1α} a†_{0β} a_{1β} a_{0α}> = -0.5 (exchange)
  auto idx = [norbs](int p, int q, int r, int s) {
    return p * norbs * norbs * norbs + q * norbs * norbs + r * norbs + s;
  };

  two_rdm_aabb[idx(0, 0, 1, 1)] = 0.5;   // Coulomb
  two_rdm_aabb[idx(1, 1, 0, 0)] = 0.5;   // Coulomb
  two_rdm_aabb[idx(0, 1, 1, 0)] = -0.5;  // exchange
  two_rdm_aabb[idx(1, 0, 0, 1)] = -0.5;  // exchange

  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  Eigen::VectorXd coeffs(2);
  coeffs(0) = 1.0 / std::sqrt(2.0);
  coeffs(1) = -1.0 / std::sqrt(2.0);
  Wavefunction::DeterminantVector dets = {Configuration("ud"),
                                          Configuration("du")};

  // Note: We override the RDMs manually here to match the triplet state.
  // The CAS container doesn't recompute them from coefficients.
  auto wf = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 2.0, 1e-12);
}

// Test compute_s_squared for triplet M_S=+1: two alpha electrons in 2 orbitals.
//
// |T,+1> = a†_{0α} a†_{1α} |vac>
//   gamma^a_{00} = gamma^a_{11} = 1, gamma^b = 0
//   Gamma^{aaaa}_{0110} - Gamma^{aaaa}_{0101} etc. from antisymmetry
//
// Expected <S^2> = S(S+1) = 2.0
TEST(SSquared, TripletMSPlusOne) {
  const int norbs = 2;
  const int norbs4 = norbs * norbs * norbs * norbs;

  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  one_rdm_aa(0, 0) = 1.0;
  one_rdm_aa(1, 1) = 1.0;

  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  // Build aaaa 2-RDM from 1-RDM (QDK convention):
  // Gamma^{aaaa}(p,q,r,s) = gamma^a(p,q)*gamma^a(r,s) -
  // gamma^a(p,s)*gamma^a(r,q)
  for (int p = 0; p < norbs; ++p)
    for (int q = 0; q < norbs; ++q)
      for (int r = 0; r < norbs; ++r)
        for (int s = 0; s < norbs; ++s) {
          int idx =
              p * norbs * norbs * norbs + q * norbs * norbs + r * norbs + s;
          two_rdm_aaaa[idx] = one_rdm_aa(p, q) * one_rdm_aa(r, s) -
                              one_rdm_aa(p, s) * one_rdm_aa(r, q);
        }

  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  Eigen::VectorXd coeffs(1);
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets = {Configuration("uu")};

  auto wf = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 2.0, 1e-12);
}

// Test compute_s_squared for a 3-electron quartet (S=3/2, M_S=+3/2).
//
// 3 alpha electrons in 3 orbitals.
//   gamma^a = I_3, gamma^b = 0
//
// Expected <S^2> = S(S+1) = 3.75
TEST(SSquared, QuartetMSPlusThreeHalf) {
  const int norbs = 3;
  const int norbs4 = norbs * norbs * norbs * norbs;

  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Identity(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);

  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  // QDK convention: Gamma^{aaaa}(p,q,r,s) = gamma^a(p,q)*gamma^a(r,s) -
  // gamma^a(p,s)*gamma^a(r,q)
  for (int p = 0; p < norbs; ++p)
    for (int q = 0; q < norbs; ++q)
      for (int r = 0; r < norbs; ++r)
        for (int s = 0; s < norbs; ++s) {
          int idx =
              p * norbs * norbs * norbs + q * norbs * norbs + r * norbs + s;
          two_rdm_aaaa[idx] = one_rdm_aa(p, q) * one_rdm_aa(r, s) -
                              one_rdm_aa(p, s) * one_rdm_aa(r, q);
        }

  auto orbitals = testing::create_test_orbitals(6, norbs, true);

  Eigen::VectorXd coeffs(1);
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets = {Configuration("uuu")};

  auto wf = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 3.75, 1e-12);
}

// Test compute_s_squared for a 4-electron singlet: two doubly-occupied orbitals
// in a 3-orbital active space.
//
//   gamma^a_{00} = gamma^a_{11} = 1, gamma^b_{00} = gamma^b_{11} = 1
//   Orbital 2 is virtual (unoccupied).
//
// Expected <S^2> = 0.0
TEST(SSquared, FourElectronSinglet) {
  const int norbs = 3;
  const int norbs4 = norbs * norbs * norbs * norbs;

  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  one_rdm_aa(0, 0) = 1.0;
  one_rdm_aa(1, 1) = 1.0;
  one_rdm_bb(0, 0) = 1.0;
  one_rdm_bb(1, 1) = 1.0;

  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  // QDK convention
  for (int p = 0; p < norbs; ++p)
    for (int q = 0; q < norbs; ++q)
      for (int r = 0; r < norbs; ++r)
        for (int s = 0; s < norbs; ++s) {
          int idx =
              p * norbs * norbs * norbs + q * norbs * norbs + r * norbs + s;
          two_rdm_aabb[idx] = one_rdm_aa(p, q) * one_rdm_bb(r, s);
          two_rdm_aaaa[idx] = one_rdm_aa(p, q) * one_rdm_aa(r, s) -
                              one_rdm_aa(p, s) * one_rdm_aa(r, q);
          two_rdm_bbbb[idx] = one_rdm_bb(p, q) * one_rdm_bb(r, s) -
                              one_rdm_bb(p, s) * one_rdm_bb(r, q);
        }

  auto orbitals = testing::create_test_orbitals(6, norbs, true);

  Eigen::VectorXd coeffs(1);
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets = {Configuration("220")};

  auto wf = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.0, 1e-12);
}

// Test compute_s_squared for a single beta electron (doublet, S=1/2).
//
// Expected <S^2> = 0.75 (same as single alpha electron by symmetry).
TEST(SSquared, SingleBetaElectronDoublet) {
  const int norbs = 1;
  const int norbs4 = 1;

  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  one_rdm_bb(0, 0) = 1.0;

  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  auto orbitals = testing::create_test_orbitals(2, norbs, true);

  Eigen::VectorXd coeffs(1);
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets = {Configuration("d")};

  auto wf = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.75, 1e-12);
}

// Test compute_s_squared for an open-shell singlet (S=0, M_S=0).
//
// 2 orbitals, 2 electrons. The open-shell singlet is:
//   |S,0> = (1/sqrt(2)) (|0α,1β> + |1α,0β>)
//
// 1-RDMs are the same as the triplet M_S=0: gamma^a = gamma^b = 0.5 * I
// But the 2-RDM cross terms have opposite sign.
//
// Expected <S^2> = 0.0
TEST(SSquared, OpenShellSinglet) {
  const int norbs = 2;
  const int norbs4 = norbs * norbs * norbs * norbs;

  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  one_rdm_aa(0, 0) = 0.5;
  one_rdm_aa(1, 1) = 0.5;
  one_rdm_bb(0, 0) = 0.5;
  one_rdm_bb(1, 1) = 0.5;

  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  auto idx = [norbs](int p, int q, int r, int s) {
    return p * norbs * norbs * norbs + q * norbs * norbs + r * norbs + s;
  };

  // Coulomb terms are the same as the triplet
  two_rdm_aabb[idx(0, 0, 1, 1)] = 0.5;
  two_rdm_aabb[idx(1, 1, 0, 0)] = 0.5;
  // Exchange terms have OPPOSITE sign compared to triplet (+0.5 instead of
  // -0.5)
  two_rdm_aabb[idx(0, 1, 1, 0)] = 0.5;
  two_rdm_aabb[idx(1, 0, 0, 1)] = 0.5;

  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  Eigen::VectorXd coeffs(2);
  coeffs(0) = 1.0 / std::sqrt(2.0);
  coeffs(1) = 1.0 / std::sqrt(2.0);
  Wavefunction::DeterminantVector dets = {Configuration("ud"),
                                          Configuration("du")};

  auto wf = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.0, 1e-12);
}

// Test compute_s_squared for the vacuum (0 electrons).
//
// All RDMs are zero. Expected <S^2> = 0.
TEST(SSquared, Vacuum) {
  const int norbs = 2;
  const int norbs4 = norbs * norbs * norbs * norbs;

  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  Eigen::VectorXd coeffs(1);
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets = {Configuration("00")};

  auto wf = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 0.0, 1e-12);
}

// Test that compute_s_squared throws when RDMs are missing
TEST(SSquared, ThrowsWithoutRDMs) {
  auto orbitals = testing::create_test_orbitals(4, 2, true);
  Eigen::VectorXd coeffs(1);
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets = {Configuration("ud")};

  auto wf = Wavefunction(
      std::make_unique<CasWavefunctionContainer>(coeffs, dets, orbitals));

  EXPECT_THROW(wf.compute_s_squared(), std::runtime_error);
}

// ---------------------------------------------------------------------------
// Helper: run SCF → Hamiltonian → CAS/SCI with RDMs
// ---------------------------------------------------------------------------

// Force unrestricted orbitals to restricted (MACIS requires restricted)
static std::shared_ptr<Orbitals> make_restricted(
    std::shared_ptr<Orbitals> orbitals) {
  if (orbitals->is_restricted()) return orbitals;
  return std::make_shared<Orbitals>(
      orbitals->get_coefficients().first, orbitals->get_energies().first,
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      std::make_tuple(orbitals->get_active_space_indices().first,
                      orbitals->get_inactive_space_indices().first));
}

// Run SCF + CAS-FCI with RDMs and return the wavefunction
static std::shared_ptr<Wavefunction> run_cas_with_rdms(
    std::shared_ptr<Structure> structure, int charge, int multiplicity,
    const std::string& basis, int nalpha, int nbeta,
    std::shared_ptr<Orbitals> custom_orbitals = nullptr) {
  std::shared_ptr<Orbitals> orbitals;
  if (custom_orbitals) {
    orbitals = custom_orbitals;
  } else {
    auto scf_solver = ScfSolverFactory::create();
    scf_solver->settings().set("scf_type", std::string("auto"));
    scf_solver->settings().set("enable_gdm", true);
    auto [E_scf, wfn_scf] =
        scf_solver->run(structure, charge, multiplicity, basis);
    orbitals = make_restricted(wfn_scf->get_orbitals());
  }

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create();  // macis_cas
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  auto [E, wfn] = mc->run(H, nalpha, nbeta);
  return wfn;
}

// Run SCF + SCI (ASCI) with RDMs and return the wavefunction
static std::shared_ptr<Wavefunction> run_sci_with_rdms(
    std::shared_ptr<Structure> structure, int charge, int multiplicity,
    const std::string& basis, int nalpha, int nbeta, int ntdets_max = 128,
    std::shared_ptr<Orbitals> custom_orbitals = nullptr) {
  std::shared_ptr<Orbitals> orbitals;
  if (custom_orbitals) {
    orbitals = custom_orbitals;
  } else {
    auto scf_solver = ScfSolverFactory::create();
    scf_solver->settings().set("scf_type", std::string("auto"));
    scf_solver->settings().set("enable_gdm", true);
    auto [E_scf, wfn_scf] =
        scf_solver->run(structure, charge, multiplicity, basis);
    orbitals = make_restricted(wfn_scf->get_orbitals());
  }

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create("macis_asci");
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  mc->settings().set("ntdets_max", ntdets_max);
  mc->settings().set("max_refine_iter", 0);
  mc->settings().set("grow_factor", 2);
  mc->settings().set("core_selection_strategy", std::string("fixed"));
  auto [E, wfn] = mc->run(H, nalpha, nbeta);
  return wfn;
}

// ---------------------------------------------------------------------------
// Tests using SCF + CAS (full CI) with RDMs from MACIS
// ---------------------------------------------------------------------------

// H atom: single electron, doublet, <S^2> = 0.75
TEST(SSquaredCAS, H_Doublet) {
  auto wfn = run_cas_with_rdms(testing::create_hydrogen_structure(), 0, 2,
                               "sto-3g", 1, 0);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.75, 1e-6);
}

// H2+ (charge=+1): single electron, doublet, <S^2> = 0.75
TEST(SSquaredCAS, H2Plus_Doublet) {
  // H2 at 1.4 Bohr
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {0., 0., 1.4}};
  std::vector<std::string> symbols = {"H", "H"};
  auto h2 = std::make_shared<Structure>(coords, symbols);
  auto wfn = run_cas_with_rdms(h2, 1, 2, "sto-3g", 1, 0);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.75, 1e-6);
}

// H2 (neutral): singlet, <S^2> = 0
TEST(SSquaredCAS, H2_Singlet) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {0., 0., 1.4}};
  std::vector<std::string> symbols = {"H", "H"};
  auto h2 = std::make_shared<Structure>(coords, symbols);
  auto wfn = run_cas_with_rdms(h2, 0, 1, "sto-3g", 1, 1);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.0, 1e-6);
}

// H2O: closed-shell singlet, <S^2> = 0
// Full CI in sto-3g (7 MOs, 5α+5β) — no active space truncation
TEST(SSquaredCAS, Water_Singlet) {
  auto water = testing::create_water_structure();
  auto wfn = run_cas_with_rdms(water, 0, 1, "sto-3g", 5, 5);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.0, 1e-6);
}

// Li atom: doublet, <S^2> = 0.75
TEST(SSquaredCAS, Li_Doublet) {
  auto li = testing::create_li_structure();
  auto wfn = run_cas_with_rdms(li, 0, 2, "sto-3g", 2, 1);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.75, 1e-6);
}

// O2: triplet ground state, <S^2> = 2.0
// CAS(2,2) with 6 frozen (core + doubly-occupied valence)
TEST(SSquaredCAS, O2_Triplet) {
  auto o2 = testing::create_o2_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("auto"));
  scf_solver->settings().set("enable_gdm", true);
  auto [E_scf, wfn_scf] = scf_solver->run(o2, 0, 3, "sto-3g");
  auto orbitals = testing::with_active_space(
      wfn_scf->get_orbitals(), std::vector<size_t>{6, 7, 8, 9},
      std::vector<size_t>{0, 1, 2, 3, 4, 5});
  auto restricted = make_restricted(orbitals);
  auto wfn = run_cas_with_rdms(o2, 0, 3, "sto-3g", 3, 1, restricted);
  EXPECT_NEAR(wfn->compute_s_squared(), 2.0, 1e-6);
}

// N atom: quartet S=3/2, <S^2> = 3.75
// CAS in 2s2p shell (4 orbitals, freeze 1s)
TEST(SSquaredCAS, N_Quartet) {
  auto n_atom = testing::create_nitrogen_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("auto"));
  scf_solver->settings().set("enable_gdm", true);
  auto [E_scf, wfn_scf] = scf_solver->run(n_atom, 0, 4, "sto-3g");
  auto orbitals = testing::with_active_space(wfn_scf->get_orbitals(),
                                             std::vector<size_t>{1, 2, 3, 4},
                                             std::vector<size_t>{0});
  auto restricted = make_restricted(orbitals);
  auto wfn = run_cas_with_rdms(n_atom, 0, 4, "sto-3g", 4, 1, restricted);
  EXPECT_NEAR(wfn->compute_s_squared(), 3.75, 1e-6);
}

// ---------------------------------------------------------------------------
// Tests using SCF + SCI (ASCI) with RDMs from MACIS
// ---------------------------------------------------------------------------

// H atom via SCI: doublet, <S^2> = 0.75
TEST(SSquaredSCI, H_Doublet) {
  auto wfn = run_sci_with_rdms(testing::create_hydrogen_structure(), 0, 2,
                               "sto-3g", 1, 0, 1);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.75, 1e-6);
}

// H2 via SCI: singlet, <S^2> = 0
TEST(SSquaredSCI, H2_Singlet) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {0., 0., 1.4}};
  std::vector<std::string> symbols = {"H", "H"};
  auto h2 = std::make_shared<Structure>(coords, symbols);
  auto wfn = run_sci_with_rdms(h2, 0, 1, "sto-3g", 1, 1, 4);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.0, 1e-6);
}

// Li atom via SCI: doublet, <S^2> = 0.75
TEST(SSquaredSCI, Li_Doublet) {
  auto li = testing::create_li_structure();
  auto wfn = run_sci_with_rdms(li, 0, 2, "sto-3g", 2, 1, 50);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.75, 1e-6);
}

// Water via SCI: singlet, <S^2> = 0
// Full orbital space in sto-3g
TEST(SSquaredSCI, Water_Singlet) {
  auto water = testing::create_water_structure();
  auto wfn = run_sci_with_rdms(water, 0, 1, "sto-3g", 5, 5, 128);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.0, 1e-6);
}

// N atom via SCI: quartet, <S^2> = 3.75
TEST(SSquaredSCI, N_Quartet) {
  auto n_atom = testing::create_nitrogen_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("auto"));
  scf_solver->settings().set("enable_gdm", true);
  auto [E_scf, wfn_scf] = scf_solver->run(n_atom, 0, 4, "sto-3g");
  auto orbitals = testing::with_active_space(wfn_scf->get_orbitals(),
                                             std::vector<size_t>{1, 2, 3, 4},
                                             std::vector<size_t>{0});
  auto restricted = make_restricted(orbitals);
  auto wfn = run_sci_with_rdms(n_atom, 0, 4, "sto-3g", 4, 1, 20, restricted);
  EXPECT_NEAR(wfn->compute_s_squared(), 3.75, 1e-6);
}

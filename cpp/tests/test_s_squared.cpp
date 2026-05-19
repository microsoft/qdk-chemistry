// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

// Test compute_s_squared for a 2-electron, 2-orbital RHF singlet wavefunction.
//
// For RHF with one doubly-occupied orbital out of norbs=2:
//   gamma^a_{00} = gamma^b_{00} = 1,  gamma^a_{11} = gamma^b_{11} = 0
//
// 2-RDM (QDK convention): Gamma(p,q,r,s) = <a†_p a†_r a_s a_q>
//   For a single determinant:
//   aabb block: Gamma^{aabb}(p,q,r,s) = gamma^a(p,q) * gamma^b(r,s)
//   aaaa block: Gamma^{aaaa}(p,q,r,s) = gamma^a(p,q) * gamma^a(r,s) -
//   gamma^a(p,s) * gamma^a(r,q)
//   bbbb block: Gamma^{bbbb}(p,q,r,s) = gamma^b(p,q) * gamma^b(r,s) -
//   gamma^b(p,s) * gamma^b(r,q)
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
  Wavefunction::DeterminantVector dets = {Configuration("20")};

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
  EXPECT_NEAR(s_squared, 0.0, testing::numerical_zero_tolerance);
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
  EXPECT_NEAR(s_squared, 0.75, testing::numerical_zero_tolerance);
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
  EXPECT_NEAR(s_squared, 2.0, testing::numerical_zero_tolerance);
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
  EXPECT_NEAR(s_squared, 2.0, testing::numerical_zero_tolerance);
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
  EXPECT_NEAR(s_squared, 3.75, testing::numerical_zero_tolerance);
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
  EXPECT_NEAR(s_squared, 0.0, testing::numerical_zero_tolerance);
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
  EXPECT_NEAR(s_squared, 0.75, testing::numerical_zero_tolerance);
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
  EXPECT_NEAR(s_squared, 0.0, testing::numerical_zero_tolerance);
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
  EXPECT_NEAR(s_squared, 0.0, testing::numerical_zero_tolerance);
}

// Stretched H2 broken-symmetry UHF: hardcoded textbook example.
// Single determinant with α in orbital 0, β in orbital 1 (orthogonal spatial
// orbitals). This models the dissociation limit where α localizes on atom A
// and β on atom B. The state is a 50/50 singlet-triplet mixture.
// Expected <S²> = 1.0
TEST(SSquared, SpinContaminatedUHF_StretchedH2) {
  const int norbs = 2;
  const int norbs4 = norbs * norbs * norbs * norbs;

  // α electron in orbital 0, β electron in orbital 1
  Eigen::MatrixXd one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
  Eigen::MatrixXd one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
  one_rdm_aa(0, 0) = 1.0;
  one_rdm_bb(1, 1) = 1.0;

  // 2-RDMs: only aabb is non-zero (1α, 1β → no same-spin pairs)
  Eigen::VectorXd two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
  Eigen::VectorXd two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);

  // Γ^{aabb}(p,q,r,s) = γ^a(p,q) · γ^b(r,s)
  for (int p = 0; p < norbs; ++p)
    for (int q = 0; q < norbs; ++q)
      for (int r = 0; r < norbs; ++r)
        for (int s = 0; s < norbs; ++s) {
          int idx =
              p * norbs * norbs * norbs + q * norbs * norbs + r * norbs + s;
          two_rdm_aabb[idx] = one_rdm_aa(p, q) * one_rdm_bb(r, s);
        }

  auto orbitals = testing::create_test_orbitals(4, norbs, true);

  Eigen::VectorXd coeffs(1);
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets = {Configuration("ud")};

  auto wf = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  double s_squared = wf.compute_s_squared();
  EXPECT_NEAR(s_squared, 1.0, testing::numerical_zero_tolerance);
}

// Test that compute_s_squared throws when RDMs are missing
TEST(SSquared, ThrowsWithoutRDMs) {
  auto orbitals = testing::create_test_orbitals(4, 2, true);
  Eigen::VectorXd coeffs(1);
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets = {Configuration("20")};

  auto wf = Wavefunction(
      std::make_unique<CasWavefunctionContainer>(coeffs, dets, orbitals));

  EXPECT_THROW(wf.compute_s_squared(), std::runtime_error);
}

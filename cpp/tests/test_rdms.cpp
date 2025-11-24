// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <complex>
#include <cstdio>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <stdexcept>
#include <tuple>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

class WavefunctionRDMTest : public ::testing::Test {
 protected:
  void SetUp() override {
    // Set up basic wavefunction data
    int size = 10;
    norbs = 3;

    // Create orbitals with enough MOs for our configurations (3 orbitals
    // needed)
    base_orbitals = testing::create_test_orbitals(6, 3, true);
    // Create dummy coefficients and determinants for basic wavefunction
    // construction - configurations need 3 orbitals to match norbs
    Eigen::VectorXd dummy_coeffs(size);
    dummy_coeffs.setZero();
    dummy_coeffs(0) = 1.0;

    Wavefunction::DeterminantVector dummy_dets(size);
    for (int i = 0; i < size; ++i) {
      // Use 3-orbital configurations to match active space size
      dummy_dets[i] = Configuration("ud0");
    }

    // Create sample RDMs
    // One-body RDMs
    one_rdm_aa = Eigen::MatrixXd::Zero(norbs, norbs);
    one_rdm_bb = Eigen::MatrixXd::Zero(norbs, norbs);
    one_rdm_spin_traced_restricted = Eigen::MatrixXd::Zero(norbs, norbs);
    one_rdm_spin_traced_unrestricted = Eigen::MatrixXd::Zero(norbs, norbs);

    // Fill with sample values
    for (int i = 0; i < norbs; ++i) {
      for (int j = 0; j < norbs; ++j) {
        one_rdm_aa(i, j) = 0.1 * (i + 1) * (j + 1);
        one_rdm_bb(i, j) = 0.05 * (i + 1) * (j + 1);
        one_rdm_spin_traced_restricted(i, j) =
            one_rdm_aa(i, j) + one_rdm_aa(i, j);
        one_rdm_spin_traced_unrestricted(i, j) =
            one_rdm_aa(i, j) + one_rdm_bb(i, j);
      }
    }

    // Two-body RDMs
    int norbs2 = norbs * norbs;
    int norbs4 = norbs2 * norbs2;
    two_rdm_aabb = Eigen::VectorXd::Zero(norbs4);
    two_rdm_bbaa = Eigen::VectorXd::Zero(norbs4);
    two_rdm_aaaa = Eigen::VectorXd::Zero(norbs4);
    two_rdm_bbbb = Eigen::VectorXd::Zero(norbs4);
    two_rdm_spin_traced_restricted = Eigen::VectorXd::Zero(norbs4);
    two_rdm_spin_traced_unrestricted = Eigen::VectorXd::Zero(norbs4);

    // Fill mixed spin blocks
    for (int i = 0; i < norbs; ++i)
      for (int j = 0; j < norbs; ++j)
        for (int k = 0; k < norbs; ++k)
          for (int l = 0; l < norbs; ++l) {
            int idx = i * norbs * norbs2 + j * norbs2 + k * norbs + l;
            two_rdm_aabb(idx) = 0.01 * (i + 5) * (j + 2) * (k + 3) * (l + 10);
          }

    for (int i = 0; i < norbs; ++i)
      for (int j = 0; j < norbs; ++j)
        for (int k = 0; k < norbs; ++k)
          for (int l = 0; l < norbs; ++l) {
            int idx_t = k * norbs * norbs2 + l * norbs2 + i * norbs + j;
            int idx = i * norbs * norbs2 + j * norbs2 + k * norbs + l;
            two_rdm_bbaa(idx_t) = two_rdm_aabb(idx);
          }

    // Fill with sample values
    for (int i = 0; i < norbs; ++i) {
      for (int j = 0; j < norbs; ++j) {
        for (int k = 0; k < norbs; ++k) {
          for (int l = 0; l < norbs; ++l) {
            int idx = i * norbs * norbs2 + j * norbs2 + k * norbs + l;
            two_rdm_aaaa(idx) = 0.005 * (i + 1) * (j + 1) * (k + 1) * (l + 1);
            two_rdm_bbbb(idx) = 0.002 * (i + 1) * (j + 1) * (k + 1) * (l + 1);

            two_rdm_spin_traced_unrestricted(idx) =
                two_rdm_aabb(idx) + two_rdm_bbaa(idx) + two_rdm_aaaa(idx) +
                two_rdm_bbbb(idx);
            two_rdm_spin_traced_restricted(idx) =
                two_rdm_aabb(idx) * 2 + two_rdm_aaaa(idx) * 2;
          }
        }
      }
    }

    // Create basic wavefunctions without RDMs
    wf_restricted = std::make_unique<Wavefunction>(
        std::make_unique<CasWavefunctionContainer>(dummy_coeffs, dummy_dets,
                                                   base_orbitals));
    wf_unrestricted = std::make_unique<Wavefunction>(
        std::make_unique<CasWavefunctionContainer>(dummy_coeffs, dummy_dets,
                                                   base_orbitals));

    // entropies
    entropies_restricted = Eigen::VectorXd::Zero(norbs);
    entropies_unrestricted = Eigen::VectorXd::Zero(norbs);

    // Create lambda functions to get the two-body RDM elements
    auto get_active_two_rdm_element = [this](int i, int j, int k, int l) {
      int norbs2 = norbs * norbs;
      return two_rdm_aabb(i * norbs * norbs2 + j * norbs2 + k * norbs + l);
    };

    // Calculate for restricted case
    for (std::size_t i = 0; i < norbs; ++i) {
      auto ordm1 = 1 - one_rdm_aa(i, i) - one_rdm_aa(i, i) +
                   get_active_two_rdm_element(i, i, i, i);
      if (ordm1 > 0) {
        entropies_restricted(i) -= ordm1 * std::log(ordm1);
      }
      auto ordm2 = one_rdm_aa(i, i) - get_active_two_rdm_element(i, i, i, i);
      if (ordm2 > 0) {
        entropies_restricted(i) -= ordm2 * std::log(ordm2);
      }
      auto ordm3 = one_rdm_aa(i, i) - get_active_two_rdm_element(i, i, i, i);
      if (ordm3 > 0) {
        entropies_restricted(i) -= ordm3 * std::log(ordm3);
      }
      auto ordm4 = get_active_two_rdm_element(i, i, i, i);
      if (ordm4 > 0) {
        entropies_restricted(i) -= ordm4 * std::log(ordm4);
      }
    }

    // Calculate for unrestricted case
    for (std::size_t i = 0; i < norbs; ++i) {
      auto ordm1 = 1 - one_rdm_aa(i, i) - one_rdm_bb(i, i) +
                   get_active_two_rdm_element(i, i, i, i);
      if (ordm1 > 0) {
        entropies_unrestricted(i) -= ordm1 * std::log(ordm1);
      }
      auto ordm2 = one_rdm_aa(i, i) - get_active_two_rdm_element(i, i, i, i);
      if (ordm2 > 0) {
        entropies_unrestricted(i) -= ordm2 * std::log(ordm2);
      }
      auto ordm3 = one_rdm_bb(i, i) - get_active_two_rdm_element(i, i, i, i);
      if (ordm3 > 0) {
        entropies_unrestricted(i) -= ordm3 * std::log(ordm3);
      }
      auto ordm4 = get_active_two_rdm_element(i, i, i, i);
      if (ordm4 > 0) {
        entropies_unrestricted(i) -= ordm4 * std::log(ordm4);
      }
    }
  }

  std::unique_ptr<Wavefunction> wf_restricted;
  std::unique_ptr<Wavefunction> wf_unrestricted;
  std::shared_ptr<Orbitals> base_orbitals;
  int norbs;

  Eigen::MatrixXd one_rdm_aa;
  Eigen::MatrixXd one_rdm_bb;
  Eigen::MatrixXd one_rdm_spin_traced_restricted;
  Eigen::MatrixXd one_rdm_spin_traced_unrestricted;

  Eigen::VectorXd two_rdm_aabb;
  Eigen::VectorXd two_rdm_bbaa;
  Eigen::VectorXd two_rdm_aaaa;
  Eigen::VectorXd two_rdm_bbbb;
  Eigen::VectorXd two_rdm_spin_traced_restricted;
  Eigen::VectorXd two_rdm_spin_traced_unrestricted;

  // Pre-computed entropy values
  Eigen::VectorXd entropies_restricted;
  Eigen::VectorXd entropies_unrestricted;
};

// Setting and retrieving one-body RDMs (spin-traced)
TEST_F(WavefunctionRDMTest, OneRDMSpinTraced) {
  EXPECT_FALSE(wf_restricted->has_one_rdm_spin_traced());
  EXPECT_FALSE(wf_unrestricted->has_one_rdm_spin_traced());

  // Create wavefunctions with spin-traced RDMs
  Eigen::VectorXd coeffs(10);
  coeffs.setZero();
  coeffs(0) = 1.0;
  Wavefunction::DeterminantVector dets(10);
  for (int i = 0; i < 10; ++i) {
    dets[i] = Configuration("ud0");
  }

  auto wf_r_traced = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, one_rdm_spin_traced_restricted,
      std::nullopt));
  auto wf_u_traced = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, one_rdm_spin_traced_unrestricted,
      std::nullopt));

  EXPECT_TRUE(wf_r_traced.has_one_rdm_spin_traced());
  EXPECT_TRUE(wf_u_traced.has_one_rdm_spin_traced());
  EXPECT_EQ(
      std::get<Eigen::MatrixXd>(wf_r_traced.get_active_one_rdm_spin_traced()),
      one_rdm_spin_traced_restricted);
  EXPECT_EQ(
      std::get<Eigen::MatrixXd>(wf_u_traced.get_active_one_rdm_spin_traced()),
      one_rdm_spin_traced_unrestricted);
}
// Setting and retrieving one-body RDMs (spin-dependent)
TEST_F(WavefunctionRDMTest, OneRDMSpinDependent) {
  EXPECT_FALSE(wf_restricted->has_one_rdm_spin_dependent());
  EXPECT_FALSE(wf_unrestricted->has_one_rdm_spin_dependent());

  Eigen::VectorXcd coeffs(10);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(10);
  for (int i = 0; i < 10; ++i) {
    dets[i] = Configuration("ud0");
  }

  auto wf_r_dependent = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_aa), std::nullopt, std::nullopt, std::nullopt,
      std::nullopt));
  auto wf_u_dependent = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt, std::nullopt, std::nullopt,
      std::nullopt));

  EXPECT_TRUE(wf_r_dependent.has_one_rdm_spin_dependent());
  EXPECT_TRUE(wf_u_dependent.has_one_rdm_spin_dependent());

  auto [aa_r, bb_r] = wf_r_dependent.get_active_one_rdm_spin_dependent();
  auto [aa_u, bb_u] = wf_u_dependent.get_active_one_rdm_spin_dependent();

  // For restricted case, both should be equal to one_rdm_aa
  EXPECT_EQ(std::get<Eigen::MatrixXd>(aa_r), one_rdm_aa);
  EXPECT_EQ(std::get<Eigen::MatrixXd>(bb_r), one_rdm_aa);

  EXPECT_EQ(std::get<Eigen::MatrixXd>(aa_u), one_rdm_aa);
  EXPECT_EQ(std::get<Eigen::MatrixXd>(bb_u), one_rdm_bb);
}

// Test single-argument create_with_one_rdm_spin_dependent (restricted case)
TEST_F(WavefunctionRDMTest, OneRDMSpinDependentSingleArgument) {
  EXPECT_FALSE(wf_restricted->has_one_rdm_spin_dependent());

  // Create new wavefunction using single argument (for restricted case)
  Eigen::VectorXcd coeffs(10);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(10);
  for (int i = 0; i < 10; ++i) {
    dets[i] = Configuration("ud0");
  }

  auto wf_with_rdm = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_aa), std::nullopt, std::nullopt, std::nullopt,
      std::nullopt));

  EXPECT_TRUE(wf_with_rdm.has_one_rdm_spin_dependent());

  auto [aa, bb] = wf_with_rdm.get_active_one_rdm_spin_dependent();

  // Both alpha and beta should be equal to the input matrix
  EXPECT_EQ(std::get<Eigen::MatrixXd>(aa), one_rdm_aa);
  EXPECT_EQ(std::get<Eigen::MatrixXd>(bb), one_rdm_aa);
}

// One-body RDM conversions from spin traced
TEST_F(WavefunctionRDMTest, OneRDMConversionsFromSpinTraced) {
  Eigen::VectorXcd coeffs(10);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(10);
  for (int i = 0; i < 10; ++i) {
    dets[i] = Configuration("ud0");
  }

  auto wf_r_traced = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals,
      std::make_optional(one_rdm_spin_traced_restricted), std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt));
  auto wf_u_traced = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, testing::create_test_orbitals(3, 2, false),
      std::make_optional(one_rdm_spin_traced_unrestricted), std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt));

  EXPECT_TRUE(wf_r_traced.has_one_rdm_spin_dependent());
  EXPECT_FALSE(wf_u_traced.has_one_rdm_spin_dependent());
  EXPECT_THROW(wf_u_traced.get_active_one_rdm_spin_dependent(),
               std::runtime_error);

  auto [aa, bb] = wf_r_traced.get_active_one_rdm_spin_dependent();

  // For restricted case, both should be equal to one_rdm_spin_traced * 0.5
  Eigen::MatrixXd expected = one_rdm_spin_traced_restricted * 0.5;
  for (int i = 0; i < norbs; ++i) {
    for (int j = 0; j < norbs; ++j) {
      EXPECT_NEAR(std::get<Eigen::MatrixXd>(aa)(i, j), expected(i, j),
                  testing::wf_tolerance);
      EXPECT_NEAR(std::get<Eigen::MatrixXd>(bb)(i, j), expected(i, j),
                  testing::wf_tolerance);
    }
  }
}

// One-body RDM conversions from spin dependent
TEST_F(WavefunctionRDMTest, OneRDMConversionsFromSpinDependent) {
  Eigen::VectorXcd coeffs(10);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(10);
  for (int i = 0; i < 10; ++i) {
    dets[i] = Configuration("ud0");
  }

  auto wf_r_dependent = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_aa), std::nullopt, std::nullopt, std::nullopt,
      std::nullopt));
  auto wf_u_dependent = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt, std::nullopt, std::nullopt,
      std::nullopt));

  EXPECT_TRUE(wf_r_dependent.has_one_rdm_spin_traced());
  EXPECT_TRUE(wf_u_dependent.has_one_rdm_spin_traced());

  auto spin_traced_r = wf_r_dependent.get_active_one_rdm_spin_traced();
  auto spin_traced_u = wf_u_dependent.get_active_one_rdm_spin_traced();

  // For restricted case, spin-traced should be double the AA RDM
  Eigen::MatrixXd expected_traced_r = one_rdm_aa + one_rdm_aa;
  Eigen::MatrixXd expected_traced_u = one_rdm_aa + one_rdm_bb;

  for (int i = 0; i < norbs; ++i) {
    for (int j = 0; j < norbs; ++j) {
      EXPECT_NEAR(std::get<Eigen::MatrixXd>(spin_traced_r)(i, j),
                  expected_traced_r(i, j), testing::wf_tolerance);
      EXPECT_NEAR(std::get<Eigen::MatrixXd>(spin_traced_u)(i, j),
                  expected_traced_u(i, j), testing::wf_tolerance);
    }
  }
}

// Setting and retrieving two-body RDMs (spin-traced)
TEST_F(WavefunctionRDMTest, TwoRDMSpinTraced) {
  EXPECT_FALSE(wf_restricted->has_two_rdm_spin_traced());
  EXPECT_FALSE(wf_unrestricted->has_two_rdm_spin_traced());

  Eigen::VectorXcd coeffs(10);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(10);
  for (int i = 0; i < 10; ++i) {
    dets[i] = Configuration("ud0");
  }

  auto wf_r_traced = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt,
      std::make_optional(two_rdm_spin_traced_restricted)));
  auto wf_u_traced = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt,
      std::make_optional(two_rdm_spin_traced_unrestricted)));

  EXPECT_TRUE(wf_r_traced.has_two_rdm_spin_traced());
  EXPECT_TRUE(wf_u_traced.has_two_rdm_spin_traced());

  EXPECT_EQ(
      std::get<Eigen::VectorXd>(wf_r_traced.get_active_two_rdm_spin_traced()),
      two_rdm_spin_traced_restricted);
  EXPECT_EQ(
      std::get<Eigen::VectorXd>(wf_u_traced.get_active_two_rdm_spin_traced()),
      two_rdm_spin_traced_unrestricted);
}

// Setting and retrieving two-body RDMs (spin-dependent)
TEST_F(WavefunctionRDMTest, TwoRDMSpinDependent) {
  EXPECT_FALSE(wf_restricted->has_two_rdm_spin_dependent());
  EXPECT_FALSE(wf_unrestricted->has_two_rdm_spin_dependent());

  Eigen::VectorXcd coeffs(10);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(10);
  for (int i = 0; i < 10; ++i) {
    dets[i] = Configuration("ud0");
  }

  auto wf_r_dependent = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::make_optional(two_rdm_aabb),
      std::make_optional(two_rdm_aaaa), std::make_optional(two_rdm_aaaa)));
  auto wf_u_dependent = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::make_optional(two_rdm_aabb),
      std::make_optional(two_rdm_aaaa), std::make_optional(two_rdm_bbbb)));

  EXPECT_TRUE(wf_r_dependent.has_two_rdm_spin_dependent());
  EXPECT_TRUE(wf_u_dependent.has_two_rdm_spin_dependent());

  auto [aabb_r, aaaa_r, bbbb_r] =
      wf_r_dependent.get_active_two_rdm_spin_dependent();
  auto [aabb_u, aaaa_u, bbbb_u] =
      wf_u_dependent.get_active_two_rdm_spin_dependent();

  // For restricted case, bbbb should be equal to aaaa
  EXPECT_EQ(std::get<Eigen::VectorXd>(aabb_r), two_rdm_aabb);
  EXPECT_EQ(std::get<Eigen::VectorXd>(aaaa_r), two_rdm_aaaa);
  EXPECT_EQ(std::get<Eigen::VectorXd>(bbbb_r), two_rdm_aaaa);

  EXPECT_EQ(std::get<Eigen::VectorXd>(aabb_u), two_rdm_aabb);
  EXPECT_EQ(std::get<Eigen::VectorXd>(aaaa_u), two_rdm_aaaa);
  EXPECT_EQ(std::get<Eigen::VectorXd>(bbbb_u), two_rdm_bbbb);
}

// Test create_with_two_rdm_spin_dependent with two arguments (restricted case)
TEST_F(WavefunctionRDMTest, TwoRDMSpinDependentTwoArguments) {
  EXPECT_FALSE(wf_restricted->has_two_rdm_spin_dependent());

  // Create using two arguments (for restricted case, bbbb = aaaa)
  Eigen::VectorXcd coeffs(10);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(10);
  for (int i = 0; i < 10; ++i) {
    dets[i] = Configuration("ud0");
  }

  auto wf_with_rdm = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::make_optional(two_rdm_aabb),
      std::make_optional(two_rdm_aaaa), std::make_optional(two_rdm_aaaa)));

  EXPECT_TRUE(wf_with_rdm.has_two_rdm_spin_dependent());

  auto [aabb, aaaa, bbbb] = wf_with_rdm.get_active_two_rdm_spin_dependent();

  // aabb and aaaa should match input, bbbb should equal aaaa for restricted
  EXPECT_EQ(std::get<Eigen::VectorXd>(aabb), two_rdm_aabb);
  EXPECT_EQ(std::get<Eigen::VectorXd>(aaaa), two_rdm_aaaa);
  EXPECT_EQ(std::get<Eigen::VectorXd>(bbbb),
            two_rdm_aaaa);  // For restricted case
}

// Two-body RDM conversions
TEST_F(WavefunctionRDMTest, TwoRDMConversions) {
  Eigen::VectorXcd coeffs(10);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets_r(10);
  for (int i = 0; i < 10; ++i) {
    dets_r[i] = Configuration("200");
  }
  Wavefunction::DeterminantVector dets_u(10);
  for (int i = 0; i < 10; ++i) {
    dets_u[i] = Configuration("2u0");
  }

  auto wf_r_dependent = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets_r, testing::create_test_orbitals(3, norbs, true),
      std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::nullopt));
  auto wf_u_dependent = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets_u, testing::create_test_orbitals(3, norbs, true),
      std::nullopt, std::nullopt, std::nullopt, std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  EXPECT_TRUE(wf_r_dependent.has_two_rdm_spin_traced());
  EXPECT_TRUE(wf_u_dependent.has_two_rdm_spin_traced());

  auto spin_traced_r = wf_r_dependent.get_active_two_rdm_spin_traced();
  auto spin_traced_u = wf_u_dependent.get_active_two_rdm_spin_traced();

  // check transpose function
  ContainerTypes::VectorVariant two_rdm_aabb_variant = two_rdm_aabb;
  auto two_rdm_bbaa_ptr =
      qdk::chemistry::data::detail::transpose_ijkl_klij_vector_variant(
          two_rdm_aabb_variant, norbs);
  auto two_rdm_bbaa_test = std::get<Eigen::VectorXd>(*two_rdm_bbaa_ptr);
  for (int i = 0; i < two_rdm_bbaa.size(); ++i) {
    EXPECT_NEAR(two_rdm_bbaa_test(i), two_rdm_bbaa(i), testing::wf_tolerance);
  }

  Eigen::VectorXd expected_r = two_rdm_aabb * 2 + two_rdm_aaaa * 2;
  Eigen::VectorXd expected_u =
      two_rdm_bbaa + two_rdm_aabb + two_rdm_aaaa + two_rdm_bbbb;

  for (int i = 0; i < expected_r.size(); ++i) {
    EXPECT_NEAR(std::get<Eigen::VectorXd>(spin_traced_r)(i), expected_r(i),
                testing::wf_tolerance);
  }

  for (int i = 0; i < expected_u.size(); ++i) {
    EXPECT_NEAR(std::get<Eigen::VectorXd>(spin_traced_u)(i), expected_u(i),
                testing::wf_tolerance);
  }
}

// Error handling for missing RDMs
TEST_F(WavefunctionRDMTest, ErrorHandlingMissingRDMs) {
  EXPECT_THROW(wf_restricted->get_active_one_rdm_spin_traced(),
               std::runtime_error);
  EXPECT_THROW(wf_restricted->get_active_one_rdm_spin_dependent(),
               std::runtime_error);
  EXPECT_THROW(wf_restricted->get_active_two_rdm_spin_traced(),
               std::runtime_error);
  EXPECT_THROW(wf_restricted->get_active_two_rdm_spin_dependent(),
               std::runtime_error);

  EXPECT_THROW(wf_unrestricted->get_active_one_rdm_spin_traced(),
               std::runtime_error);
  EXPECT_THROW(wf_unrestricted->get_active_one_rdm_spin_dependent(),
               std::runtime_error);
  EXPECT_THROW(wf_unrestricted->get_active_two_rdm_spin_traced(),
               std::runtime_error);
  EXPECT_THROW(wf_unrestricted->get_active_two_rdm_spin_dependent(),
               std::runtime_error);
}

// Orbital entropies computation
TEST_F(WavefunctionRDMTest, OrbitalEntropies) {
  Eigen::VectorXcd coeffs(10);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(10);
  for (int i = 0; i < 10; ++i) {
    dets[i] = Configuration("ud0");
  }

  // Create wavefunctions with both 1-RDM and 2-RDM data for entropy
  // calculation
  auto wf_r_full = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_aa), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_aaaa)));

  auto wf_u_full = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::make_optional(one_rdm_aa),
      std::make_optional(one_rdm_bb), std::nullopt,
      std::make_optional(two_rdm_aabb), std::make_optional(two_rdm_aaaa),
      std::make_optional(two_rdm_bbbb)));

  // Get the entropies
  Eigen::VectorXd entropies_r = wf_r_full.get_single_orbital_entropies();
  Eigen::VectorXd entropies_u = wf_u_full.get_single_orbital_entropies();

  // Verify the size of the entropies vectors
  EXPECT_EQ(entropies_r.size(), norbs);
  EXPECT_EQ(entropies_u.size(), norbs);

  // Compare with pre-computed values from SetUp
  for (int i = 0; i < norbs; ++i) {
    EXPECT_NEAR(entropies_r(i), entropies_restricted(i), testing::wf_tolerance);
    EXPECT_NEAR(entropies_u(i), entropies_unrestricted(i),
                testing::wf_tolerance);
  }
}

// Additional tests for edge cases in has_* methods and entropy calculation
TEST_F(WavefunctionRDMTest, HasMethodsRestrictedEdgeCases) {
  // Test has_* methods for restricted orbitals
  // These require having only alpha or only beta RDM set in restricted
  // systems. Since the public API doesn't allow this directly, we'll test the
  // achievable paths

  // Test restricted system with spin-dependent RDMs
  Eigen::VectorXcd coeffs(2);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(2);
  dets[0] = Configuration("ud");
  dets[1] = Configuration("du");

  Eigen::MatrixXd test_rdm = Eigen::MatrixXd::Random(2, 2);

  // Create wavefunction with spin-dependent RDM (both alpha and beta will be
  // the same)
  auto restricted_test =
      Wavefunction(std::make_unique<CasWavefunctionContainer>(
          coeffs, dets, base_orbitals, std::nullopt,
          std::make_optional(test_rdm), std::make_optional(test_rdm),
          std::nullopt, std::nullopt, std::nullopt, std::nullopt));

  // This should return true through the normal path
  EXPECT_TRUE(restricted_test.has_one_rdm_spin_dependent());
  EXPECT_TRUE(restricted_test.has_one_rdm_spin_traced());

  // Create wavefunction with only spin-traced RDM set
  auto restricted_traced =
      Wavefunction(std::make_unique<CasWavefunctionContainer>(
          coeffs, dets, base_orbitals, std::make_optional(test_rdm),
          std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
          std::nullopt));

  EXPECT_TRUE(restricted_traced.has_one_rdm_spin_dependent());
  EXPECT_TRUE(restricted_traced.has_one_rdm_spin_traced());
}

TEST_F(WavefunctionRDMTest, HasMethodsTwoRDMEdgeCases) {
  // Test has_two_rdm_* methods for edge cases

  // Test restricted system with two-argument RDM setting
  Eigen::VectorXcd coeffs(2);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(2);
  dets[0] = Configuration("ud");
  dets[1] = Configuration("du");

  Eigen::VectorXd test_aabb = Eigen::VectorXd::Random(16);
  Eigen::VectorXd test_aaaa = Eigen::VectorXd::Random(16);

  // Create wavefunction with two-RDM using two arguments (bbbb will be set to
  // aaaa for restricted)
  auto restricted_two_rdm =
      Wavefunction(std::make_unique<CasWavefunctionContainer>(
          coeffs, dets, base_orbitals, std::nullopt, std::nullopt, std::nullopt,
          std::nullopt, std::make_optional(test_aabb),
          std::make_optional(test_aaaa), std::make_optional(test_aaaa)));

  // This should return true through normal path
  EXPECT_TRUE(restricted_two_rdm.has_two_rdm_spin_dependent());
  EXPECT_TRUE(restricted_two_rdm.has_two_rdm_spin_traced());

  // Test unrestricted system with partial RDMs
  auto unrestricted_partial =
      Wavefunction(std::make_unique<CasWavefunctionContainer>(
          coeffs, dets, base_orbitals, std::nullopt, std::nullopt, std::nullopt,
          std::nullopt, std::make_optional(test_aabb),
          std::make_optional(test_aaaa), std::make_optional(test_aaaa)));

  // For unrestricted with only two arguments, bbbb = aaaa, so normal path
  // applies
  EXPECT_TRUE(unrestricted_partial.has_two_rdm_spin_dependent());
  EXPECT_TRUE(unrestricted_partial.has_two_rdm_spin_traced());
}

TEST_F(WavefunctionRDMTest, OrbitEntropyErrorHandling) {
  // Error cases in get_single_orbital_entropies

  // Create basic coefficients and determinants for testing
  Eigen::VectorXcd coeffs(2);
  coeffs.setZero();
  coeffs(0) = std::complex<double>(1.0, 0.0);
  Wavefunction::DeterminantVector dets(2);
  dets[0] = Configuration("ud");
  dets[1] = Configuration("du");

  // Test case 1: Missing one-body RDMs
  Wavefunction empty_one_rdm(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, base_orbitals, std::nullopt, std::nullopt, std::nullopt,
      std::nullopt, std::nullopt, std::nullopt, std::nullopt));
  EXPECT_THROW(empty_one_rdm.get_single_orbital_entropies(),
               std::runtime_error);

  // Test case 2: Missing two-body RDMs
  Eigen::MatrixXd one_rdm = Eigen::MatrixXd::Random(2, 2);
  auto no_two_rdm = Wavefunction(std::make_unique<CasWavefunctionContainer>(
      coeffs, dets, testing::create_test_orbitals(), std::nullopt,
      std::make_optional(one_rdm), std::make_optional(one_rdm), std::nullopt,
      std::nullopt, std::nullopt, std::nullopt));
  EXPECT_THROW(no_two_rdm.get_single_orbital_entropies(), std::runtime_error);
}

class WavefunctionRealRDMsTest : public ::testing::Test {
 protected:
  void SetUp() override {}
  // Lambda to extract a spin-dependent one-particle RDM from a two-particle
  // mixed spin RDM D_a[p,q] =  sum_r D_ab[p,q,r,r]/(nele_b)
 public:
  // helper function to extract spin-dependent one-particle RDM from
  Eigen::MatrixXd extract_one_rdm_spin_from_mixed_two(
      const Eigen::VectorXd &two_rdm, int norb, int nele_spin) {
    Eigen::MatrixXd one_rdm = Eigen::MatrixXd::Zero(norb, norb);
    for (int p = 0; p < norb; ++p) {
      for (int q = 0; q < norb; ++q) {
        double val = 0.0;
        for (int r = 0; r < norb; ++r) {
          int idx = p * norb * norb * norb + q * norb * norb + r * norb + r;
          val += two_rdm(idx);
        }
        one_rdm(p, q) = val / static_cast<double>(nele_spin);
      }
    }
    return one_rdm;
  }

  // helper function to extract spin-traced one-particle RDM from two-particle
  // RDM
  Eigen::MatrixXd extract_one_rdm_from_two(const Eigen::VectorXd &two_rdm,
                                           int norb, int nele) {
    Eigen::MatrixXd one_rdm = Eigen::MatrixXd::Zero(norb, norb);
    for (int p = 0; p < norb; ++p) {
      for (int q = 0; q < norb; ++q) {
        double val = 0.0;
        for (int r = 0; r < norb; ++r) {
          int idx = p * norb * norb * norb + q * norb * norb + r * norb + r;
          val += two_rdm(idx);
        }
        one_rdm(p, q) = val / static_cast<double>(nele - 1);
      }
    }
    return one_rdm;
  }

  // helper function to transpose two-rdm indices
  Eigen::VectorXd transpose_two_rdm_indices(
      const Eigen::VectorXd &two_rdm, int norb,
      const std::array<int, 4> &new_order) {
    Eigen::VectorXd transposed = Eigen::VectorXd::Zero(two_rdm.size());
    for (int p = 0; p < norb; ++p) {
      for (int q = 0; q < norb; ++q) {
        for (int r = 0; r < norb; ++r) {
          for (int s = 0; s < norb; ++s) {
            int old_idx =
                p * norb * norb * norb + q * norb * norb + r * norb + s;
            std::array<int, 4> indices = {p, q, r, s};
            int new_p = indices[new_order[0]];
            int new_q = indices[new_order[1]];
            int new_r = indices[new_order[2]];
            int new_s = indices[new_order[3]];
            int new_idx = new_p * norb * norb * norb + new_q * norb * norb +
                          new_r * norb + new_s;
            transposed(new_idx) = two_rdm(old_idx);
          }
        }
      }
    }
    return transposed;
  }
};

TEST_F(WavefunctionRealRDMsTest, N2_Singlet) {
  auto n2 = testing::create_stretched_n2_structure();
  auto scf = ScfSolverFactory::create();
  auto [scf_energy, scf_wfn] = scf->run(n2, 0, 1, "sto-3g");

  auto cas_selector = ActiveSpaceSelectorFactory::create("qdk_valence");
  cas_selector->settings().set("num_active_electrons", 6);
  cas_selector->settings().set("num_active_orbitals", 6);
  auto active_space = cas_selector->run(scf_wfn);

  auto h_creator = HamiltonianConstructorFactory::create();
  auto hamiltonian = h_creator->run(scf_wfn->get_orbitals());

  auto macis_cas = MultiConfigurationCalculatorFactory::create("macis_cas");
  macis_cas->settings().set("calculate_one_rdm", true);
  macis_cas->settings().set("calculate_two_rdm", true);
  auto [energy, wavefunction] = macis_cas->run(hamiltonian, 3, 3);

  EXPECT_TRUE(wavefunction->has_one_rdm_spin_dependent());
  EXPECT_TRUE(wavefunction->has_two_rdm_spin_dependent());
  EXPECT_TRUE(wavefunction->has_one_rdm_spin_traced());
  EXPECT_TRUE(wavefunction->has_two_rdm_spin_traced());

  // get rdm and other wavefunction properties
  int norb =
      wavefunction->get_orbitals()->get_active_space_indices().first.size();
  auto [alpha_elec, beta_elec] = wavefunction->get_active_num_electrons();
  auto rdm1 = wavefunction->get_active_one_rdm_spin_traced();
  auto [aa, bb] = wavefunction->get_active_one_rdm_spin_dependent();
  auto rdm2 = wavefunction->get_active_two_rdm_spin_traced();
  auto [aabb, aaaa, bbbb] = wavefunction->get_active_two_rdm_spin_dependent();

  // transpose aabb to get bbaa
  auto bbaa = transpose_two_rdm_indices(std::get<Eigen::VectorXd>(aabb), norb,
                                        {2, 3, 0, 1});

  // compare spin traced and spin dependent RDMs
  auto manual_rdm1 =
      (std::get<Eigen::MatrixXd>(aa) + std::get<Eigen::MatrixXd>(bb));
  for (int i = 0; i < manual_rdm1.rows(); ++i) {
    for (int j = 0; j < manual_rdm1.cols(); ++j) {
      EXPECT_NEAR(manual_rdm1(i, j), std::get<Eigen::MatrixXd>(rdm1)(i, j),
                  testing::rdm_tolerance);
    }
  }
  auto manual_rdm2 =
      (std::get<Eigen::VectorXd>(aaaa) + std::get<Eigen::VectorXd>(bbbb)) +
      (std::get<Eigen::VectorXd>(aabb) + bbaa);
  for (int i = 0; i < manual_rdm2.size(); ++i) {
    EXPECT_NEAR(manual_rdm2(i), std::get<Eigen::VectorXd>(rdm2)(i),
                testing::rdm_tolerance);
  }

  // check that one rdm matches that extracted from two rdm
  Eigen::MatrixXd one_from_two = extract_one_rdm_from_two(
      std::get<Eigen::VectorXd>(rdm2), norb, alpha_elec + beta_elec);
  // Basic sanity: diagonal elements should be close
  for (int i = 0; i < norb; ++i) {
    for (int j = 0; j < norb; ++j) {
      EXPECT_NEAR(one_from_two(i, j), std::get<Eigen::MatrixXd>(rdm1)(i, j),
                  testing::rdm_tolerance);
    }
  }
  Eigen::MatrixXd one_alpha_from_two_mixed =
      extract_one_rdm_spin_from_mixed_two(std::get<Eigen::VectorXd>(aabb), norb,
                                          beta_elec);
  for (int i = 0; i < norb; ++i) {
    for (int j = 0; j < norb; ++j) {
      EXPECT_NEAR(one_alpha_from_two_mixed(i, j),
                  std::get<Eigen::MatrixXd>(aa)(i, j), testing::rdm_tolerance);
    }
  }
  Eigen::MatrixXd one_beta_from_two_mixed =
      extract_one_rdm_spin_from_mixed_two(bbaa, norb, alpha_elec);
  for (int i = 0; i < norb; ++i) {
    for (int j = 0; j < norb; ++j) {
      EXPECT_NEAR(one_beta_from_two_mixed(i, j),
                  std::get<Eigen::MatrixXd>(bb)(i, j), testing::rdm_tolerance);
    }
  }
}

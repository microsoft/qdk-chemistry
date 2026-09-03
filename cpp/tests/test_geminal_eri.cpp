// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/util/geminal_eri.h>

#include <cmath>
#include <libint2.hpp>
#include <limits>
#include <vector>

using qdk::chemistry::scf::geminal::stg_geminal_eri;

namespace {

// Two-primitive s-shell basis used as the reference probe (matches the values
// computed directly from libint2 in the design's feasibility gate G1).
libint2::BasisSet make_two_shell_basis() {
  libint2::Shell a({1.2}, {{0, false, {1.0}}}, {{0.0, 0.0, 0.0}});
  libint2::Shell b({0.8}, {{0, false, {1.0}}}, {{0.0, 0.0, 1.4}});
  return libint2::BasisSet({a, b});
}

libint2::BasisSet make_one_shell_basis() {
  libint2::Shell c({0.5}, {{0, false, {1.0}}}, {{0.3, 0.0, 0.7}});
  return libint2::BasisSet({c});
}

// Independent dense (no-symmetry) evaluation of (p q | Op | r s) over four
// basis sets, used as the oracle for the helper.
template <libint2::Operator Op>
std::vector<double> reference_geminal_eri(const libint2::BasisSet& bs1,
                                          const libint2::BasisSet& bs2,
                                          const libint2::BasisSet& bs3,
                                          const libint2::BasisSet& bs4,
                                          double gamma) {
  const size_t n1 = bs1.nbf(), n2 = bs2.nbf(), n3 = bs3.nbf(), n4 = bs4.nbf();
  std::vector<double> out(n1 * n2 * n3 * n4, 0.0);
  const auto max_nprim = std::max(
      {bs1.max_nprim(), bs2.max_nprim(), bs3.max_nprim(), bs4.max_nprim()});
  const auto max_l =
      std::max({bs1.max_l(), bs2.max_l(), bs3.max_l(), bs4.max_l()});
  libint2::Engine engine(Op, max_nprim, static_cast<int>(max_l), 0,
                         std::numeric_limits<double>::epsilon(), gamma);
  const auto s1 = bs1.shell2bf(), s2 = bs2.shell2bf(), s3 = bs3.shell2bf(),
             s4 = bs4.shell2bf();
  const auto& buf = engine.results();
  for (size_t i = 0; i < bs1.size(); ++i)
    for (size_t j = 0; j < bs2.size(); ++j)
      for (size_t k = 0; k < bs3.size(); ++k)
        for (size_t l = 0; l < bs4.size(); ++l) {
          engine.compute2<Op, libint2::BraKet::xx_xx, 0>(bs1[i], bs2[j], bs3[k],
                                                         bs4[l]);
          const auto* data = buf[0];
          if (data == nullptr) continue;
          for (size_t a = 0, idx = 0; a < bs1[i].size(); ++a)
            for (size_t b = 0; b < bs2[j].size(); ++b)
              for (size_t c = 0; c < bs3[k].size(); ++c)
                for (size_t d = 0; d < bs4[l].size(); ++d, ++idx)
                  out[((s1[i] + a) * n2 + (s2[j] + b)) * n3 * n4 +
                      (s3[k] + c) * n4 + (s4[l] + d)] = data[idx];
        }
  return out;
}

}  // namespace

TEST(GeminalEri, StgMatchesDirectLibint) {
  libint2::initialize();
  const auto bs = make_two_shell_basis();
  const double gamma = 1.5;
  auto got = stg_geminal_eri(libint2::Operator::stg, gamma, bs, bs, bs, bs);
  auto ref =
      reference_geminal_eri<libint2::Operator::stg>(bs, bs, bs, bs, gamma);
  for (size_t i = 0; i < ref.size(); ++i)
    EXPECT_NEAR(got[i], ref[i], 1e-12) << "element " << i;
  libint2::finalize();
}

TEST(GeminalEri, StgTimesCoulombMatchesDirectLibint) {
  libint2::initialize();
  const auto bs = make_two_shell_basis();
  const double gamma = 1.5;
  auto got =
      stg_geminal_eri(libint2::Operator::stg_x_coulomb, gamma, bs, bs, bs, bs);
  auto ref = reference_geminal_eri<libint2::Operator::stg_x_coulomb>(bs, bs, bs,
                                                                     bs, gamma);
  for (size_t i = 0; i < ref.size(); ++i)
    EXPECT_NEAR(got[i], ref[i], 1e-12) << "element " << i;
  libint2::finalize();
}

// Cross-check against the known-good values from the G1 feasibility probe
// (gamma = 1.0, the (a b | a b) block of the two-shell basis).
TEST(GeminalEri, ReferenceValuesG1) {
  libint2::initialize();
  const auto bs = make_two_shell_basis();
  const size_t n = bs.nbf();  // 2
  const size_t abab = ((0 * n + 1) * n + 0) * n + 1;
  auto stg = stg_geminal_eri(libint2::Operator::stg, 1.0, bs, bs, bs, bs);
  auto stgc =
      stg_geminal_eri(libint2::Operator::stg_x_coulomb, 1.0, bs, bs, bs, bs);
  EXPECT_NEAR(stg[abab], 5.149334e-02, 1e-7);
  EXPECT_NEAR(stgc[abab], 7.346698e-02, 1e-7);
  libint2::finalize();
}

// Exercise the mixed-dimension (CABS-like) path with four index spaces of
// different sizes.
TEST(GeminalEri, MixedBasisDimensions) {
  libint2::initialize();
  const auto obs = make_two_shell_basis();   // nbf = 2
  const auto cabs = make_one_shell_basis();  // nbf = 1
  const double gamma = 1.5;
  auto got =
      stg_geminal_eri(libint2::Operator::stg, gamma, obs, cabs, obs, cabs);
  auto ref = reference_geminal_eri<libint2::Operator::stg>(obs, cabs, obs, cabs,
                                                           gamma);
  ASSERT_EQ(ref.size(), static_cast<size_t>(2 * 1 * 2 * 1));
  for (size_t i = 0; i < ref.size(); ++i)
    EXPECT_NEAR(got[i], ref[i], 1e-12) << "element " << i;
  libint2::finalize();
}

TEST(GeminalEri, RejectsNonSlaterOperator) {
  libint2::initialize();
  const auto bs = make_two_shell_basis();
  EXPECT_THROW(stg_geminal_eri(libint2::Operator::coulomb, 1.0, bs, bs, bs, bs),
               std::invalid_argument);
  libint2::finalize();
}

// The four-index MO transform must match an independent naive contraction with
// four distinct (non-square) coefficient blocks.
TEST(GeminalEri, MoTransformMatchesNaive) {
  using qdk::chemistry::scf::geminal::mo_transform_4index;
  const std::size_t n1 = 3, n2 = 4, n3 = 2, n4 = 3;
  const int m1 = 2, m2 = 3, m3 = 2, m4 = 1;
  std::vector<double> ao(n1 * n2 * n3 * n4);
  for (std::size_t i = 0; i < ao.size(); ++i)
    ao[i] = std::sin(0.7 * static_cast<double>(i) + 0.3);

  auto rnd = [](int r, int c, double seed) {
    Eigen::MatrixXd m(r, c);
    for (int a = 0; a < r; ++a)
      for (int b = 0; b < c; ++b) m(a, b) = std::cos(seed + 0.9 * a - 0.4 * b);
    return m;
  };
  const Eigen::MatrixXd c1 = rnd(n1, m1, 0.1);
  const Eigen::MatrixXd c2 = rnd(n2, m2, 1.1);
  const Eigen::MatrixXd c3 = rnd(n3, m3, 2.1);
  const Eigen::MatrixXd c4 = rnd(n4, m4, 3.1);

  auto got = mo_transform_4index(ao.data(), n1, n2, n3, n4, c1, c2, c3, c4);

  for (int p = 0; p < m1; ++p)
    for (int q = 0; q < m2; ++q)
      for (int r = 0; r < m3; ++r)
        for (int s = 0; s < m4; ++s) {
          double ref = 0.0;
          for (std::size_t mu = 0; mu < n1; ++mu)
            for (std::size_t nu = 0; nu < n2; ++nu)
              for (std::size_t la = 0; la < n3; ++la)
                for (std::size_t si = 0; si < n4; ++si)
                  ref += c1(mu, p) * c2(nu, q) * c3(la, r) * c4(si, s) *
                         ao[((mu * n2 + nu) * n3 + la) * n4 + si];
          const std::size_t idx =
              static_cast<std::size_t>(((p * m2 + q) * m3 + r) * m4 + s);
          EXPECT_NEAR(got[idx], ref, 1e-12)
              << p << "," << q << "," << r << "," << s;
        }
}

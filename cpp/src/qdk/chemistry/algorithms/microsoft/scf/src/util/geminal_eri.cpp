// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/util/geminal_eri.h>

#include <algorithm>
#include <array>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qdk::chemistry::scf::geminal {

namespace {

template <::libint2::Operator Op>
void compute_four_center(const ::libint2::BasisSet& bs1,
                         const ::libint2::BasisSet& bs2,
                         const ::libint2::BasisSet& bs3,
                         const ::libint2::BasisSet& bs4,
                         ::libint2::Engine base_engine, double* out) {
  const size_t n2 = bs2.nbf();
  const size_t n3 = bs3.nbf();
  const size_t n4 = bs4.nbf();
  const size_t stride1 = n2 * n3 * n4;
  const size_t stride2 = n3 * n4;
  const size_t stride3 = n4;

  const auto sh2bf1 = bs1.shell2bf();
  const auto sh2bf2 = bs2.shell2bf();
  const auto sh2bf3 = bs3.shell2bf();
  const auto sh2bf4 = bs4.shell2bf();
  const size_t ns1 = bs1.size();
  const size_t ns2 = bs2.size();
  const size_t ns3 = bs3.size();
  const size_t ns4 = bs4.size();

#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  std::vector<::libint2::Engine> engines(nthreads, base_engine);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    auto& engine = engines[thread_id];
    const auto& buf = engine.results();

    for (size_t s1 = 0; s1 < ns1; ++s1) {
      if (static_cast<int>(s1) % nthreads != thread_id) continue;
      const size_t bf1 = sh2bf1[s1];
      const size_t nb1 = bs1[s1].size();
      for (size_t s2 = 0; s2 < ns2; ++s2) {
        const size_t bf2 = sh2bf2[s2];
        const size_t nb2 = bs2[s2].size();
        for (size_t s3 = 0; s3 < ns3; ++s3) {
          const size_t bf3 = sh2bf3[s3];
          const size_t nb3 = bs3[s3].size();
          for (size_t s4 = 0; s4 < ns4; ++s4) {
            const size_t bf4 = sh2bf4[s4];
            const size_t nb4 = bs4[s4].size();

            engine.compute2<Op, ::libint2::BraKet::xx_xx, 0>(bs1[s1], bs2[s2],
                                                             bs3[s3], bs4[s4]);
            const auto* data = buf[0];
            if (data == nullptr) continue;

            for (size_t i1 = 0, idx = 0; i1 < nb1; ++i1)
              for (size_t i2 = 0; i2 < nb2; ++i2)
                for (size_t i3 = 0; i3 < nb3; ++i3)
                  for (size_t i4 = 0; i4 < nb4; ++i4, ++idx)
                    out[(bf1 + i1) * stride1 + (bf2 + i2) * stride2 +
                        (bf3 + i3) * stride3 + (bf4 + i4)] = data[idx];
          }
        }
      }
    }
  }
}

}  // namespace

namespace {
::libint2::Engine make_engine(::libint2::Operator op,
                              const ::libint2::BasisSet& bs1,
                              const ::libint2::BasisSet& bs2,
                              const ::libint2::BasisSet& bs3,
                              const ::libint2::BasisSet& bs4, double gamma) {
  const auto max_nprim = std::max(
      {bs1.max_nprim(), bs2.max_nprim(), bs3.max_nprim(), bs4.max_nprim()});
  const auto max_l =
      std::max({bs1.max_l(), bs2.max_l(), bs3.max_l(), bs4.max_l()});
  // The geminal operators take the Slater exponent as the libint2 Params
  // argument; the Coulomb operator must use default (empty) parameters.
  if (op == ::libint2::Operator::coulomb)
    return ::libint2::Engine(op, max_nprim, static_cast<int>(max_l), 0);
  return ::libint2::Engine(op, max_nprim, static_cast<int>(max_l), 0,
                           std::numeric_limits<double>::epsilon(), gamma);
}
}  // namespace

std::unique_ptr<double[]> stg_geminal_eri(::libint2::Operator op, double gamma,
                                          const ::libint2::BasisSet& bs1,
                                          const ::libint2::BasisSet& bs2,
                                          const ::libint2::BasisSet& bs3,
                                          const ::libint2::BasisSet& bs4) {
  QDK_LOG_TRACE_ENTERING();

  const size_t total = bs1.nbf() * bs2.nbf() * bs3.nbf() * bs4.nbf();
  auto out = std::make_unique<double[]>(total);
  std::fill_n(out.get(), total, 0.0);

  switch (op) {
    case ::libint2::Operator::stg:
      compute_four_center<::libint2::Operator::stg>(
          bs1, bs2, bs3, bs4,
          make_engine(::libint2::Operator::stg, bs1, bs2, bs3, bs4, gamma),
          out.get());
      break;
    case ::libint2::Operator::stg_x_coulomb:
      compute_four_center<::libint2::Operator::stg_x_coulomb>(
          bs1, bs2, bs3, bs4,
          make_engine(::libint2::Operator::stg_x_coulomb, bs1, bs2, bs3, bs4,
                      gamma),
          out.get());
      break;
    default:
      throw std::invalid_argument(
          "stg_geminal_eri: operator must be stg or stg_x_coulomb");
  }

  return out;
}

std::unique_ptr<double[]> four_center_coulomb(const ::libint2::BasisSet& bs1,
                                              const ::libint2::BasisSet& bs2,
                                              const ::libint2::BasisSet& bs3,
                                              const ::libint2::BasisSet& bs4) {
  QDK_LOG_TRACE_ENTERING();
  const size_t total = bs1.nbf() * bs2.nbf() * bs3.nbf() * bs4.nbf();
  auto out = std::make_unique<double[]>(total);
  std::fill_n(out.get(), total, 0.0);
  compute_four_center<::libint2::Operator::coulomb>(
      bs1, bs2, bs3, bs4,
      make_engine(::libint2::Operator::coulomb, bs1, bs2, bs3, bs4, 0.0),
      out.get());
  return out;
}

namespace {
Eigen::MatrixXd one_body_matrix(const ::libint2::BasisSet& bs,
                                ::libint2::Engine& engine) {
  const size_t n = bs.nbf();
  Eigen::MatrixXd m = Eigen::MatrixXd::Zero(n, n);
  const auto sh2bf = bs.shell2bf();
  const auto& buf = engine.results();
  for (size_t s1 = 0; s1 < bs.size(); ++s1) {
    const size_t bf1 = sh2bf[s1];
    const size_t nb1 = bs[s1].size();
    for (size_t s2 = 0; s2 <= s1; ++s2) {
      const size_t bf2 = sh2bf[s2];
      const size_t nb2 = bs[s2].size();
      engine.compute(bs[s1], bs[s2]);
      const auto* data = buf[0];
      if (data == nullptr) continue;
      for (size_t i = 0, idx = 0; i < nb1; ++i)
        for (size_t j = 0; j < nb2; ++j, ++idx) {
          m(bf1 + i, bf2 + j) = data[idx];
          m(bf2 + j, bf1 + i) = data[idx];
        }
    }
  }
  return m;
}
}  // namespace

Eigen::MatrixXd kinetic_matrix(const ::libint2::BasisSet& bs) {
  QDK_LOG_TRACE_ENTERING();
  ::libint2::Engine engine(::libint2::Operator::kinetic, bs.max_nprim(),
                           static_cast<int>(bs.max_l()), 0);
  return one_body_matrix(bs, engine);
}

Eigen::MatrixXd nuclear_matrix(
    const ::libint2::BasisSet& bs,
    const std::vector<std::pair<double, std::array<double, 3>>>& charges) {
  QDK_LOG_TRACE_ENTERING();
  ::libint2::Engine engine(::libint2::Operator::nuclear, bs.max_nprim(),
                           static_cast<int>(bs.max_l()), 0);
  engine.set_params(charges);
  return one_body_matrix(bs, engine);
}

std::unique_ptr<double[]> mo_transform_4index(
    const double* ao, std::size_t n1, std::size_t n2, std::size_t n3,
    std::size_t n4, const Eigen::MatrixXd& c1, const Eigen::MatrixXd& c2,
    const Eigen::MatrixXd& c3, const Eigen::MatrixXd& c4) {
  QDK_LOG_TRACE_ENTERING();
  using RowMajorMat =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

  const std::array<std::size_t, 4> dims = {n1, n2, n3, n4};
  const std::array<const Eigen::MatrixXd*, 4> coeffs = {&c1, &c2, &c3, &c4};

  std::vector<double> data(ao, ao + n1 * n2 * n3 * n4);
  std::array<std::size_t, 4> shape = dims;

  // Transform the leading axis with its coefficient block, then rotate axes so
  // the next untransformed axis becomes leading; four steps restore the order.
  for (int step = 0; step < 4; ++step) {
    const std::size_t a = shape[0];
    const std::size_t b = shape[1];
    const std::size_t c = shape[2];
    const std::size_t d = shape[3];
    const Eigen::MatrixXd& cmat = *coeffs[step];
    const std::size_t m = static_cast<std::size_t>(cmat.cols());

    Eigen::Map<const RowMajorMat> mat(data.data(), a, b * c * d);
    const RowMajorMat r = cmat.transpose() * mat;  // [m, b*c*d]

    std::vector<double> rotated(b * c * d * m);
    for (std::size_t i = 0; i < m; ++i)
      for (std::size_t bcd = 0; bcd < b * c * d; ++bcd)
        rotated[bcd * m + i] =
            r(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(bcd));

    data.swap(rotated);
    shape = {b, c, d, m};
  }

  auto out = std::make_unique<double[]>(data.size());
  std::copy(data.begin(), data.end(), out.get());
  return out;
}

}  // namespace qdk::chemistry::scf::geminal

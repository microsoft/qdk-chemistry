// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/util/cabs.h>

#include <algorithm>
#include <qdk/chemistry/utils/logger.hpp>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qdk::chemistry::scf::cabs {

Eigen::MatrixXd ao_overlap(const ::libint2::BasisSet& bs1,
                           const ::libint2::BasisSet& bs2) {
  QDK_LOG_TRACE_ENTERING();
  const size_t n1 = bs1.nbf();
  const size_t n2 = bs2.nbf();
  Eigen::MatrixXd s = Eigen::MatrixXd::Zero(n1, n2);

  const auto max_nprim = std::max(bs1.max_nprim(), bs2.max_nprim());
  const auto max_l = std::max(bs1.max_l(), bs2.max_l());
  ::libint2::Engine base_engine(::libint2::Operator::overlap, max_nprim,
                                static_cast<int>(max_l), 0);

  const auto sh2bf1 = bs1.shell2bf();
  const auto sh2bf2 = bs2.shell2bf();

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
    for (size_t s1 = 0; s1 < bs1.size(); ++s1) {
      if (static_cast<int>(s1) % nthreads != thread_id) continue;
      const size_t bf1 = sh2bf1[s1];
      const size_t nb1 = bs1[s1].size();
      for (size_t s2 = 0; s2 < bs2.size(); ++s2) {
        const size_t bf2 = sh2bf2[s2];
        const size_t nb2 = bs2[s2].size();
        engine.compute(bs1[s1], bs2[s2]);
        const auto* data = buf[0];
        if (data == nullptr) continue;
        for (size_t i = 0, idx = 0; i < nb1; ++i)
          for (size_t j = 0; j < nb2; ++j, ++idx)
            s(bf1 + i, bf2 + j) = data[idx];
      }
    }
  }
  return s;
}

namespace {

// Canonical orthonormalizer X (n x m, m <= n) of a symmetric positive
// (semi-)definite overlap S such that X^T S X = I, dropping directions with
// eigenvalue below tol.
Eigen::MatrixXd canonical_orthonormalizer(const Eigen::MatrixXd& s,
                                          double tol) {
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(s);
  const auto& evals = es.eigenvalues();
  const auto& evecs = es.eigenvectors();
  std::vector<int> keep;
  for (int i = 0; i < evals.size(); ++i)
    if (evals(i) > tol) keep.push_back(i);
  Eigen::MatrixXd x(s.rows(), static_cast<int>(keep.size()));
  for (size_t c = 0; c < keep.size(); ++c)
    x.col(static_cast<int>(c)) = evecs.col(keep[c]) / std::sqrt(evals(keep[c]));
  return x;
}

}  // namespace

CabsResult build_cabs(const ::libint2::BasisSet& obs,
                      const ::libint2::BasisSet& aux, double lindep_tol) {
  QDK_LOG_TRACE_ENTERING();

  std::vector<::libint2::Shell> union_shells(obs.begin(), obs.end());
  union_shells.insert(union_shells.end(), aux.begin(), aux.end());
  ::libint2::BasisSet ri_basis(union_shells);

  const int n_obs = static_cast<int>(obs.nbf());
  const int n_ri = static_cast<int>(ri_basis.nbf());

  const Eigen::MatrixXd s_ri = ao_overlap(ri_basis, ri_basis);

  // Orthonormal orbital basis embedded in the union AO space.
  const Eigen::MatrixXd c_obs_small =
      canonical_orthonormalizer(s_ri.topLeftCorner(n_obs, n_obs), lindep_tol);
  Eigen::MatrixXd c_obs = Eigen::MatrixXd::Zero(n_ri, c_obs_small.cols());
  c_obs.topRows(n_obs) = c_obs_small;

  // Orthonormal basis of the full union space.
  const Eigen::MatrixXd c_ri = canonical_orthonormalizer(s_ri, lindep_tol);

  // Overlap of the orthonormal orbital basis with the orthonormal union basis;
  // its right null space is the orbital-orthogonal (CABS) subspace.
  const Eigen::MatrixXd m = c_obs.transpose() * s_ri * c_ri;
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(m, Eigen::ComputeFullV);
  const auto& sigma = svd.singularValues();
  const Eigen::MatrixXd v = svd.matrixV();

  std::vector<int> cabs_cols;
  for (int j = 0; j < v.cols(); ++j)
    if (j >= sigma.size() || sigma(j) < 1.0 - lindep_tol)
      cabs_cols.push_back(j);

  Eigen::MatrixXd v_cabs(v.rows(), static_cast<int>(cabs_cols.size()));
  for (size_t c = 0; c < cabs_cols.size(); ++c)
    v_cabs.col(static_cast<int>(c)) = v.col(cabs_cols[c]);

  CabsResult result{ri_basis, c_ri * v_cabs};
  return result;
}

}  // namespace qdk::chemistry::scf::cabs

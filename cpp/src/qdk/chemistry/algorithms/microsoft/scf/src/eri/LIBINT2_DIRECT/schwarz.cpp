// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "schwarz.h"

#include <qdk/chemistry/utils/logger.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qdk::chemistry::scf::libint2::direct {

using qdk::chemistry::scf::RowMajorMatrix;

RowMajorMatrix compute_schwarz_ints(const ::libint2::BasisSet& obs,
                                    bool use_2norm) {
  QDK_LOG_TRACE_ENTERING();

  const size_t nsh = obs.size();

  // Setup the engine
#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
#else
  const int nthreads = 1;
#endif
  std::vector<::libint2::Engine> engines(nthreads);
  engines[0] = ::libint2::Engine(::libint2::Operator::coulomb, obs.max_nprim(),
                                 obs.max_l(), 0, 0.0);
  for (int i = 1; i < nthreads; ++i) engines[i] = engines[0];

  RowMajorMatrix K(nsh, nsh);
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    auto& engine = engines[omp_get_thread_num()];
#else
    auto& engine = engines[0];
#endif
    const auto& buf = engine.results();
#ifdef _OPENMP
#pragma omp for collapse(2)
#endif
    for (auto i = 0; i < nsh; ++i) {
      for (auto j = 0; j <= i; ++j) {
        const size_t ni = obs[i].size();
        const size_t nj = obs[j].size();
        const size_t nij = ni * nj;
        engine.compute2<::libint2::Operator::coulomb, ::libint2::BraKet::xx_xx,
                        0>(obs[i], obs[j], obs[i], obs[j]);

        Eigen::Map<const RowMajorMatrix> bmap(buf[0], nij, nij);
        auto norm = use_2norm ? bmap.norm() : bmap.lpNorm<Eigen::Infinity>();
        K(i, j) = std::sqrt(norm);
        K(j, i) = K(i, j);
      }
    }
  }

  return K;
}

}  // namespace qdk::chemistry::scf::libint2::direct

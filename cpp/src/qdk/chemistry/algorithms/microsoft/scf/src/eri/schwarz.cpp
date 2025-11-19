// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "schwarz.h"

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/types.h>
#include <qdk/chemistry/scf/util/libint2_util.h>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cassert>
#include <qdk/chemistry/utils/omp_utils.hpp>
#include <vector>

#include "util/timer.h"

namespace qdk::chemistry::scf {
void schwarz_integral(const BasisSet* iobs, const ParallelConfig& mpi,
                      double* res) {
  AutoTimer timer("schwarz_integral");
  auto obs = libint2_util::convert_to_libint_basisset(*iobs);

  using libint2::Engine;
  int nthreads = omp_get_max_threads();
  RowMajorMatrix S = RowMajorMatrix::Zero(obs.size(), obs.size());
  std::vector<Engine> engines(
      nthreads,
      Engine(libint2::Operator::coulomb, obs.max_nprim(), obs.max_l(), 0, 0.0));
#pragma omp parallel
  {
    int local_thread_id = omp_get_thread_num();
    int world_thread_size = mpi.world_size * nthreads;
    int world_thread_id = mpi.world_rank * nthreads + local_thread_id;
    Engine& engine = engines[local_thread_id];

    int job_id = 0;
    for (size_t i = 0; i < obs.size(); i++) {
      for (size_t j = 0; j <= i; j++) {
        if (job_id++ % world_thread_size != world_thread_id) continue;
        auto buf = engine.compute(obs[i], obs[j], obs[i], obs[j]);
        assert(buf[0] != nullptr);
        size_t n1 = obs[i].size(), n2 = obs[j].size();
        Eigen::Map<const RowMajorMatrix> buf_mat(buf[0], n1 * n1, n2 * n2);
        S(i, j) = S(j, i) = std::sqrt(buf_mat.lpNorm<Eigen::Infinity>());
      }
    }
  }
  if (mpi.world_size > 1) {
#ifdef QDK_CHEMISTRY_ENABLE_MPI
    MPI_Allreduce(MPI_IN_PLACE, S.data(), S.size(), MPI_DOUBLE, MPI_SUM,
                  MPI_COMM_WORLD);
#endif
  }

  memcpy(res, S.data(), sizeof(double) * S.size());
}

void compute_shell_norm(const BasisSet* obs, const double* D, double* res) {
  auto nsh = obs->shells.size();
  auto num_basis_funcs = obs->num_basis_funcs;
  for (size_t i = 0, pi = 0; i < nsh; i++) {
    size_t ag_i = obs->shells[i].angular_momentum;
    size_t num_basis_funcs_i =
        obs->pure ? ag_i * 2 + 1 : (ag_i + 1) * (ag_i + 2) / 2;

    for (size_t j = 0, pj = 0; j < nsh; j++) {
      size_t ag_j = obs->shells[j].angular_momentum;
      size_t num_basis_funcs_j =
          obs->pure ? ag_j * 2 + 1 : (ag_j + 1) * (ag_j + 2) / 2;

      double mx = std::fabs(D[pi * num_basis_funcs + pj]);
      for (size_t ii = pi; ii < pi + num_basis_funcs_i; ii++) {
        for (size_t jj = pj; jj < pj + num_basis_funcs_j; jj++) {
          mx = std::max(mx, std::fabs(D[ii * num_basis_funcs + jj]));
        }
      }
      res[i * nsh + j] = mx;
      pj += num_basis_funcs_j;
    }
    pi += num_basis_funcs_i;
  }
}
}  // namespace qdk::chemistry::scf

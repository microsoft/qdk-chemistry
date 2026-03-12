// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/eri.h>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif

#include <Eigen/Core>
#include <blas.hh>
#include <cstring>
#include <memory>
#include <vector>
#ifdef ENABLE_NVTX3
#include <nvtx3/nvtx3.hpp>
#endif

// These are external headers provided in the addons package
#ifdef QDK_CHEMISTRY_ENABLE_HGP
#include "eri/HGP/eri_hgp.h"
#endif
#ifdef QDK_CHEMISTRY_ENABLE_RYS
#include "eri/RYS/rys.h"
#endif
#ifdef QDK_CHEMISTRY_ENABLE_LIBINTX
#include "eri/LIBINTX/libintx.h"
#endif

#include <qdk/chemistry/utils/logger.hpp>

#include "eri/INCORE/incore.h"
#include "eri/LIBINT2_DIRECT/libint2_direct.h"
#include "eri/SNK/snk.h"
#include "util/macros.h"
#include "util/timer.h"

namespace qdk::chemistry::scf {
std::shared_ptr<ERI> ERI::create(BasisSet& basis_set, const SCFConfig& cfg,
                                 double omega) {
  QDK_LOG_TRACE_ENTERING();

#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{nvtx3::rgb{127, 0, 255}, "ERI::create"};
#endif
  AutoTimer t("ERI::create");
  switch (cfg.eri.method) {
#ifdef QDK_CHEMISTRY_ENABLE_RYS
    case ERIMethod::Rys:
      return std::make_shared<ERIRYS>(
          cfg.scf_orbital_type, cfg.eri.eri_threshold, basis_set, cfg.mpi);
#endif
#ifdef QDK_CHEMISTRY_ENABLE_HGP
    case ERIMethod::HGP:
      return std::make_shared<ERIHGP>(
          cfg.scf_orbital_type, cfg.eri.eri_threshold, basis_set, cfg.mpi);
#endif
    case ERIMethod::Incore:
      return std::make_shared<ERIINCORE>(cfg.scf_orbital_type, basis_set,
                                         cfg.mpi, omega);
    case ERIMethod::SnK:
      return std::make_shared<SNK>(cfg.scf_orbital_type, basis_set,
                                   cfg.snk_input, cfg.exc.xc_name, cfg.mpi);
    case ERIMethod::Libint2Direct:
      return std::make_shared<LIBINT2_DIRECT>(cfg.scf_orbital_type, basis_set,
                                              cfg.mpi, cfg.eri.use_atomics);
    default:
      throw std::runtime_error("Invalid ERI Method");
  }
  return nullptr;
}

std::shared_ptr<ERI> ERI::create(BasisSet& basis_set, BasisSet& aux_basis_set,
                                 const SCFConfig& cfg, double omega) {
  QDK_LOG_TRACE_ENTERING();

#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{nvtx3::rgb{127, 0, 255}, "ERI::create"};
#endif
  AutoTimer t("ERI::create");
  switch (cfg.eri.method) {
    case ERIMethod::Incore:
      return std::make_shared<ERIINCORE_DF>(cfg.scf_orbital_type, basis_set,
                                            aux_basis_set, cfg.mpi);
#ifdef QDK_CHEMISTRY_ENABLE_LIBINTX
    case ERIMethod::LibintX:
      return std::make_shared<LIBINTX_DF>(cfg.scf_orbital_type, basis_set,
                                          aux_basis_set, cfg.mpi,
                                          cfg.libintx_config.min_tile_size);
#endif
    default:
      throw std::runtime_error("Invalid DF-ERI Method");
  }
  return nullptr;
}

void ERI::build_JK(const double* P, double* J, double* K, double alpha,
                   double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();

#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{nvtx3::rgb{127, 0, 255}, "ERI::build_JK"};
#endif
  AutoTimer t("ERI::build_JK");
  build_JK_impl_(P, J, K, alpha, beta, omega);

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  int num_density_matrices = has_spin_split_density() ? 2 : 1;
  int size = num_density_matrices * basis_set_.num_atomic_orbitals *
             basis_set_.num_atomic_orbitals;
  if (mpi_.world_size > 1) {
    MPI_Barrier(MPI_COMM_WORLD);
    AutoTimer t("ERI::build_JK->MPI_Reduce");
    if (J)
      MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : J, J, size, MPI_DOUBLE,
                 MPI_SUM, 0, MPI_COMM_WORLD);
    if (K)
      MPI_Reduce(mpi_.world_rank == 0 ? MPI_IN_PLACE : K, K, size, MPI_DOUBLE,
                 MPI_SUM, 0, MPI_COMM_WORLD);
  }
#endif
}

void ERI::quarter_trans(size_t nt, const double* C, double* out) {
  QDK_LOG_TRACE_ENTERING();

  quarter_trans_impl(nt, C, out);
}

void ERI::half_trans(size_t ni, const double* Ci, size_t nj, const double* Cj,
                     double* out) {
  QDK_LOG_TRACE_ENTERING();

  half_trans_impl(ni, Ci, nj, Cj, out);
}

void ERI::half_trans_impl(size_t ni, const double* Ci, size_t nj,
                          const double* Cj, double* out) {
  QDK_LOG_TRACE_ENTERING();

  // Batched Q1 + immediate Q2 to keep memory bounded.
  // Each batch transforms bp orbitals via quarter_trans_impl, then
  // contracts the second index via BLAS before the next batch.
  const size_t nb = basis_set_.num_atomic_orbitals;
  const size_t nb2 = nb * nb;
  const size_t nb3 = nb2 * nb;
  const size_t nij = ni * nj;

  // Budget: quarter_trans internally allocates nthreads copies of the output
  // for thread-local accumulation, so actual memory is bp*nb³*(1+nthreads).
  // Target ~1 GB for the Q1 buffer itself to stay safe on 16 GB systems.
  constexpr size_t max_bytes = 1ULL * 1024 * 1024 * 1024;
  const size_t max_batch =
      std::max<size_t>(1, max_bytes / (nb3 * sizeof(double)));

  // Cj row-major (nb × nj) → col-major for BLAS
  Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic,
                                 Eigen::RowMajor>>
      Cj_rm(Cj, nb, nj);
  Eigen::MatrixXd Cj_cm = Cj_rm;

  const size_t n_batches = (ni + max_batch - 1) / max_batch;
  QDK_LOGGER().debug(
      "half_trans: nb={}, ni={}, nj={}, batch_size={}, n_batches={}, "
      "Q1_buf={:.2f} GB, out={:.2f} GB",
      nb, ni, nj, max_batch, n_batches,
      double(max_batch * nb3 * sizeof(double)) / (1024.0 * 1024 * 1024),
      double(nij * nb2 * sizeof(double)) / (1024.0 * 1024 * 1024));

  std::memset(out, 0, nij * nb2 * sizeof(double));

  for (size_t p0 = 0, batch_idx = 0; p0 < ni; p0 += max_batch, ++batch_idx) {
    const size_t bp = std::min(max_batch, ni - p0);

    QDK_LOGGER().debug("half_trans: batch {}/{}, orbitals [{}, {})",
                       batch_idx + 1, n_batches, p0, p0 + bp);

    // Extract columns p0..p0+bp from Ci (row-major nb×ni) into contiguous
    // row-major buffer (nb×bp)
    std::vector<double> Ci_batch(nb * bp);
    for (size_t mu = 0; mu < nb; ++mu)
      for (size_t p = 0; p < bp; ++p)
        Ci_batch[mu * bp + p] = Ci[mu * ni + p0 + p];

    // Q1 for this batch: quarter_trans produces (bp, nb, nb, nb) col-major
    std::vector<double> tmp(bp * nb3);
    quarter_trans_impl(bp, Ci_batch.data(), tmp.data());

    // Q2: for each (ν,μ) pair, gemm contracts λ→q
    //   out_{νμ}(p_full, q) += tmp_{νμ}(p_local, λ) × Cj(λ, q)
    // Output stride: p sits within the full ni dimension
    for (size_t vm = 0; vm < nb2; ++vm) {
      const double* TMP_vm = tmp.data() + vm * nb * bp;
      double* OUT_vm = out + vm * nij + p0;
      blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
                 bp, nj, nb, 1.0, TMP_vm, bp, Cj_cm.data(), nb, 0.0, OUT_vm,
                 ni);
    }
  }
}
}  // namespace qdk::chemistry::scf

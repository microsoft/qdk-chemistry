// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/eri.h>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif

#include <memory>
#ifdef ENABLE_NVTX3
#include <nvtx3/nvtx3.hpp>
#endif
#include <spdlog/spdlog.h>

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

#include "eri/INCORE/incore.h"
#include "eri/LIBINT2_DIRECT/libint2_direct.h"
#include "eri/SNK/snk.h"
#include "util/macros.h"
#include "util/timer.h"

namespace qdk::chemistry::scf {
std::shared_ptr<ERI> ERI::create(BasisSet& basis_set, const SCFConfig& cfg,
                                 double omega) {
#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{nvtx3::rgb{127, 0, 255}, "ERI::create"};
#endif
  AutoTimer t("ERI::create");
  switch (cfg.eri.method) {
#ifdef QDK_CHEMISTRY_ENABLE_RYS
    case ERIMethod::Rys:
      return std::make_shared<ERIRYS>(cfg.unrestricted, cfg.eri.eri_threshold,
                                      basis_set, cfg.mpi);
#endif
#ifdef QDK_CHEMISTRY_ENABLE_HGP
    case ERIMethod::HGP:
      return std::make_shared<ERIHGP>(cfg.unrestricted, cfg.eri.eri_threshold,
                                      basis_set, cfg.mpi);
#endif
    case ERIMethod::Incore:
      return std::make_shared<ERIINCORE>(cfg.unrestricted, basis_set, cfg.mpi,
                                         omega);
    case ERIMethod::SnK:
      return std::make_shared<SNK>(cfg.unrestricted, basis_set, cfg.snk_input,
                                   cfg.exc.xc_name, cfg.mpi);
    case ERIMethod::Libint2Direct:
      return std::make_shared<LIBINT2_DIRECT>(cfg.unrestricted, basis_set,
                                              cfg.mpi);
    default:
      throw std::runtime_error("Invalid ERI Method");
  }
  return nullptr;
}

std::shared_ptr<ERI> ERI::create(BasisSet& basis_set, BasisSet& aux_basis_set,
                                 const SCFConfig& cfg, double omega) {
#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{nvtx3::rgb{127, 0, 255}, "ERI::create"};
#endif
  AutoTimer t("ERI::create");
  switch (cfg.eri.method) {
    case ERIMethod::Incore:
      return std::make_shared<ERIINCORE_DF>(cfg.unrestricted, basis_set,
                                            aux_basis_set, cfg.mpi);
#ifdef QDK_CHEMISTRY_ENABLE_LIBINTX
    case ERIMethod::LibintX:
      return std::make_shared<LIBINTX_DF>(cfg.unrestricted, basis_set,
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
#ifdef ENABLE_NVTX3
  nvtx3::scoped_range r{nvtx3::rgb{127, 0, 255}, "ERI::build_JK"};
#endif
  AutoTimer t("ERI::build_JK");
  build_JK_impl_(P, J, K, alpha, beta, omega);

#ifdef QDK_CHEMISTRY_ENABLE_MPI
  int num_density_matrices = unrestricted_ ? 2 : 1;
  int size = num_density_matrices * basis_set_.num_basis_funcs *
             basis_set_.num_basis_funcs;
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
  quarter_trans_impl(nt, C, out);
}
}  // namespace qdk::chemistry::scf

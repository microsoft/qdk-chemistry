// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scf/ks_impl.h"

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/eri/eri_multiplexer.h>
#include <qdk/chemistry/scf/scf/scf_solver.h>
#include <qdk/chemistry/scf/util/env_helper.h>
#include <qdk/chemistry/scf/util/gauxc_util.h>
#include <spdlog/spdlog.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <thread>

#include "scf/scf_impl.h"
#include "util/macros.h"
#include "util/timer.h"

#ifdef ENABLE_NVTX3
#include <nvtx3/nvtx3.hpp>
#endif

namespace qdk::chemistry::scf {
KSImpl::KSImpl(std::shared_ptr<Molecule> mol, const SCFConfig& cfg)
    : SCFImpl(mol, cfg, true) {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  auto& bs = ctx_.basis_set;
  if (cfg.mpi.world_rank == 0) {
    spdlog::info("xc={}, grid_level={}", cfg.exc.xc_name,
                 gauxc_util::to_string(cfg.xc_input.grid_spec));
  }
  XC_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                             num_atomic_orbitals_);
  TIMEIT(exc_ = EXC::create(bs, cfg), "KSImpl::KS->EXC::create");

  // Update SCFConfig w/ RSX data
  double omega;
  {
    double alpha, beta;
    std::tie(alpha, beta, omega) = get_hyb_coeff_();
  }
  if (cfg.do_dfj) {
    TIMEIT(eri_ = ERIMultiplexer::create(*ctx_.basis_set, *ctx_.aux_basis_set,
                                         cfg, omega),
           "SCFImpl::SCFImpl->ERI::create");
  } else {
    TIMEIT(eri_ = ERIMultiplexer::create(*ctx_.basis_set, cfg, omega),
           "SCFImpl::SCFImpl->ERI::create");
  }

  if (cfg.require_polarizability) {
    // Host allocations for CPSCF/TDDFT quantities
    tXC_ = RowMajorMatrix::Zero(num_density_matrices_ * num_atomic_orbitals_,
                                num_atomic_orbitals_);
  }
}
KSImpl::KSImpl(std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
               const RowMajorMatrix& dm)
    : KSImpl(mol, cfg) {
  VERIFY(dm.rows() == num_density_matrices_ * num_atomic_orbitals_ &&
         dm.cols() == num_atomic_orbitals_);
  P_ = dm;
  density_matrix_initialized_ = true;
}

void KSImpl::update_fock_() {
#ifdef ENABLE_NVTX3
  NVTX3_FUNC_RANGE();
#endif
  if (ctx_.cfg->mpi.world_rank == 0) {
    F_ -= XC_;
  }
  TIMEIT(exc_->build_XC(P_.data(), XC_.data(), &ctx_.result.scf_xc_energy),
         "EXC::build_XC");
  SCFImpl::update_fock_();
  if (ctx_.cfg->mpi.world_rank == 0) {
    F_ += XC_;
  }
}

void KSImpl::reset_fock_() {
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  F_ = H_ + XC_ + Vpcm_;
#else
  F_ = H_ + XC_;
#endif
}

double KSImpl::total_energy_() {
  auto& res = ctx_.result;
  res.scf_one_electron_energy = P_.cwiseProduct(H_).sum();
  res.scf_two_electron_energy = 0.5 * P_.cwiseProduct(F_ - H_ - XC_).sum();
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  res.scf_two_electron_energy -= 0.5 * P_.cwiseProduct(Vpcm_).sum();
#endif

  double _total = res.nuclear_repulsion_energy + res.scf_one_electron_energy +
                  res.scf_two_electron_energy + res.scf_xc_energy;
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  _total += res.scf_pcm_energy;
#endif
#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
  _total += res.scf_dispersion_correction_energy;
#endif
  return _total;
}

std::tuple<double, double, double> KSImpl::get_hyb_coeff_() const {
  return exc_->get_hyb();
}

const RowMajorMatrix KSImpl::get_vxc_grad_() const {
  if (ctx_.cfg->mpi.world_rank == 0) {
    spdlog::trace("Calculating EXC gradients");
  }
  RowMajorMatrix grad = RowMajorMatrix::Zero(3, ctx_.mol->n_atoms);
  TIMEIT(exc_->get_gradients(P_.data(), grad.data()), "EXC::get_gradients");
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  if (pcm_ != nullptr) {
    RowMajorMatrix grad_pcm = RowMajorMatrix::Zero(3, ctx_.mol->n_atoms);
    TIMEIT(pcm_->get_gradients(P_.data(), grad_pcm.data()),
           "PCM::get_gradients");
    grad += grad_pcm;
  }
#endif

  return grad;
}
}  // namespace qdk::chemistry::scf

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/eri/eri_multiplexer.h>
#include <spdlog/spdlog.h>

namespace qdk::chemistry::scf {
ERIMultiplexer::ERIMultiplexer(BasisSet& basis, BasisSet& aux_basis,
                               const SCFConfig& cfg, double omega)
    : ERI(cfg.unrestricted, cfg.eri.eri_threshold, basis, cfg.mpi) {
  if (not cfg.do_dfj) {
    j_impl_ = ERI::create(basis, cfg, omega);
    k_impl_ = j_impl_;
  } else {
    j_impl_ = ERI::create(basis, aux_basis, cfg, 0.0);
    SCFConfig k_cfg(cfg);
    k_cfg.eri = cfg.k_eri;
    k_impl_ = ERI::create(basis, k_cfg, omega);
  }
  qt_impl_ = k_impl_;
  if (cfg.grad_eri.method != cfg.eri.method and cfg.require_gradient) {
    SCFConfig grad_cfg(cfg);
    grad_cfg.eri = cfg.grad_eri;
    grad_impl_ = ERI::create(basis, grad_cfg, omega);
  } else {
    grad_impl_ = j_impl_;
  }
}

ERIMultiplexer::ERIMultiplexer(BasisSet& basis, const SCFConfig& cfg,
                               double omega)
    : ERI(cfg.unrestricted, cfg.eri.eri_threshold, basis, cfg.mpi) {
  if (cfg.do_dfj)
    throw std::runtime_error("An AUX basis must be specified for DFJ");

  j_impl_ = ERI::create(basis, cfg, omega);
  if (cfg.eri.method == cfg.k_eri.method) {
    k_impl_ = j_impl_;
  } else {
    SCFConfig k_cfg(cfg);
    k_cfg.eri = cfg.k_eri;
    k_impl_ = ERI::create(basis, k_cfg, omega);
  }
  qt_impl_ = j_impl_;
  if (cfg.grad_eri.method != cfg.eri.method and cfg.require_gradient) {
    SCFConfig grad_cfg(cfg);
    grad_cfg.eri = cfg.grad_eri;
    grad_impl_ = ERI::create(basis, grad_cfg, omega);
  } else {
    grad_impl_ = j_impl_;
  }
}

void ERIMultiplexer::build_JK(const double* P, double* J, double* K,
                              double alpha, double beta, double omega) {
  // jk_impl_->build_JK(P, J, K, alpha, beta, omega);
  if (j_impl_ == k_impl_) {
    j_impl_->build_JK(P, J, K, alpha, beta, omega);
  } else {
    j_impl_->build_JK(P, J, nullptr, alpha, beta, 0.0);
    k_impl_->build_JK(P, nullptr, K, alpha, beta, omega);
  }
}
void ERIMultiplexer::quarter_trans(size_t nt, const double* C, double* out) {
  qt_impl_->quarter_trans(nt, C, out);
}
void ERIMultiplexer::get_gradients(const double* P, double* dJ, double* dK,
                                   double alpha, double beta, double omega) {
  if (grad_impl_ == j_impl_) {
    if (j_impl_ == k_impl_) {
      j_impl_->get_gradients(P, dJ, dK, alpha, beta, omega);
    } else {
      j_impl_->get_gradients(P, dJ, nullptr, alpha, beta, 0.0);
      k_impl_->get_gradients(P, nullptr, dK, alpha, beta, omega);
    }
  } else {
    grad_impl_->get_gradients(P, dJ, dK, alpha, beta, omega);
  }
}
}  // namespace qdk::chemistry::scf

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/eri/eri_multiplexer.h>

#include <algorithm>
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::scf {
ERIMultiplexer::ERIMultiplexer(BasisSet& basis, BasisSet& aux_basis,
                               const SCFConfig& cfg, double omega,
                               const bool needs_exchange)
    : ERI(cfg.scf_orbital_type, cfg.eri.eri_threshold, basis, cfg.mpi),
      omega_(omega) {
  QDK_LOG_TRACE_ENTERING();

  if (not cfg.do_dfj) {
    j_impl_ = ERI::create(basis, cfg, omega);
    k_impl_ = needs_exchange ? j_impl_ : nullptr;
  } else {
    j_impl_ = ERI::create(basis, aux_basis, cfg, 0.0);
    if (needs_exchange) {
      SCFConfig k_cfg(cfg);
      k_cfg.eri = cfg.k_eri;
      k_impl_ = ERI::create(basis, k_cfg, omega);
    } else {
      deferred_qt_config_ = cfg;
      deferred_qt_config_->eri = cfg.k_eri;
    }
  }
  qt_impl_ = cfg.do_dfj ? k_impl_ : j_impl_;
  if (cfg.grad_eri.method != cfg.eri.method and cfg.require_gradient) {
    SCFConfig grad_cfg(cfg);
    grad_cfg.eri = cfg.grad_eri;
    grad_impl_ = ERI::create(basis, grad_cfg, omega);
  } else {
    grad_impl_ = j_impl_;
  }
}

ERIMultiplexer::ERIMultiplexer(BasisSet& basis, const SCFConfig& cfg,
                               double omega, const bool needs_exchange)
    : ERI(cfg.scf_orbital_type, cfg.eri.eri_threshold, basis, cfg.mpi),
      omega_(omega) {
  QDK_LOG_TRACE_ENTERING();

  if (cfg.do_dfj)
    throw std::runtime_error("An AUX basis must be specified for DFJ");

  j_impl_ = ERI::create(basis, cfg, omega);
  if (!needs_exchange) {
    k_impl_ = nullptr;
  } else if (cfg.eri.method == cfg.k_eri.method) {
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
  QDK_LOG_TRACE_ENTERING();

  if (!k_impl_) {
    j_impl_->build_JK(P, J, nullptr, 0.0, 0.0, 0.0);
    if (K) {
      const size_t density_blocks = has_spin_split_density() ? 2 : 1;
      const size_t num_orbitals = basis_set_.num_atomic_orbitals;
      std::fill_n(K, density_blocks * num_orbitals * num_orbitals, 0.0);
    }
  } else if (j_impl_ == k_impl_) {
    j_impl_->build_JK(P, J, K, alpha, beta, omega);
  } else {
    j_impl_->build_JK(P, J, nullptr, alpha, beta, 0.0);
    k_impl_->build_JK(P, nullptr, K, alpha, beta, omega);
  }
}

void ERIMultiplexer::quarter_trans(size_t nt, const double* C, double* out) {
  QDK_LOG_TRACE_ENTERING();

  if (!qt_impl_) {
    if (!deferred_qt_config_) {
      throw std::runtime_error(
          "No conventional ERI backend is available for transformation");
    }
    qt_impl_ = ERI::create(basis_set_, *deferred_qt_config_, omega_);
  }
  qt_impl_->quarter_trans(nt, C, out);
}

void ERIMultiplexer::get_gradients(const double* P, double* dJ, double* dK,
                                   double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();

  if (!k_impl_) {
    grad_impl_->get_gradients(P, dJ, nullptr, 0.0, 0.0, 0.0);
    if (dK) {
      std::fill_n(dK, 3 * basis_set_.mol->n_atoms, 0.0);
    }
  } else if (grad_impl_ == j_impl_) {
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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "incore.h"

#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

#include "incore_impl.h"

namespace qdk::chemistry::scf {

ERIINCORE::ERIINCORE(SCFOrbitalType scf_orbital_type, BasisSet& basis_set,
                     ParallelConfig _mpi, double omega)
    : ERI(scf_orbital_type, 0.0, basis_set, _mpi),
      eri_impl_(incore::ERI::make_incore_eri(
          scf_orbital_type == SCFOrbitalType::Restricted ? 1 : 2, basis_set,
          _mpi, omega)) {
  QDK_LOG_TRACE_ENTERING();
}

ERIINCORE::~ERIINCORE() noexcept = default;

void ERIINCORE::build_JK_impl_(const double* P, double* J, double* K,
                               double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();

  if (!eri_impl_) throw std::runtime_error("ERIINCORE NOT INITIALIZED");
  eri_impl_->build_JK(P, J, K, alpha, beta, omega);
}

void ERIINCORE::quarter_trans_impl(size_t nt, const double* C, double* out) {
  QDK_LOG_TRACE_ENTERING();

  if (!eri_impl_) throw std::runtime_error("ERIINCORE NOT INITIALIZED");
  eri_impl_->quarter_trans(nt, C, out);
}

void ERIINCORE::get_gradients(const double* P, double* dJ, double* dK,
                              double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();

  throw std::runtime_error("INCORE GRADIENTS NYI");
}

ERIINCORE_DF::ERIINCORE_DF(SCFOrbitalType scf_orbital_type, BasisSet& obs,
                           BasisSet& abs, ParallelConfig _mpi)
    : ERI(scf_orbital_type, 0.0, obs, _mpi),
      eri_impl_(incore::ERI_DF::make_incore_eri(
          scf_orbital_type != SCFOrbitalType::Restricted, obs, abs, _mpi)) {
  QDK_LOG_TRACE_ENTERING();
}

ERIINCORE_DF::~ERIINCORE_DF() noexcept = default;

void ERIINCORE_DF::build_JK_impl_(const double* P, double* J, double* K,
                                  double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();

  if (!eri_impl_) throw std::runtime_error("ERIINCORE_DF NOT INITIALIZED");
  eri_impl_->build_JK(P, J, K, alpha, beta, omega);
}

void ERIINCORE_DF::get_gradients(const double* P, double* dJ, double* dK,
                                 double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();

  if (!eri_impl_) throw std::runtime_error("ERIINCORE_DF NOT INITIALIZED");
  eri_impl_->get_gradients(P, dJ, dK, alpha, beta, omega);
}

void ERIINCORE_DF::quarter_trans_impl(size_t nt, const double* C, double* out) {
  QDK_LOG_TRACE_ENTERING();

  if (!eri_impl_) throw std::runtime_error("ERIINCORE_DF NOT INITIALIZED");
  eri_impl_->quarter_trans(nt, C, out);
};

}  // namespace qdk::chemistry::scf

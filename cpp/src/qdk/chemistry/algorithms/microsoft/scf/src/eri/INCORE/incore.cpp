// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "incore.h"

#include <stdexcept>

#include "incore_impl.h"

namespace qdk::chemistry::scf {

ERIINCORE::ERIINCORE(bool unr, BasisSet& basis_set, ParallelConfig _mpi,
                     double omega)
    : ERI(unr, 0.0, basis_set, _mpi),
      eri_impl_(incore::ERI::make_incore_eri(unr, basis_set, _mpi, omega)) {}

ERIINCORE::~ERIINCORE() noexcept = default;

void ERIINCORE::build_JK_impl_(const double* P, double* J, double* K,
                               double alpha, double beta, double omega) {
  if (!eri_impl_) throw std::runtime_error("ERIINCORE NOT INITIALIZED");
  eri_impl_->build_JK(P, J, K, alpha, beta, omega);
}

void ERIINCORE::quarter_trans_impl(size_t nt, const double* C, double* out) {
  if (!eri_impl_) throw std::runtime_error("ERIINCORE NOT INITIALIZED");
  eri_impl_->quarter_trans(nt, C, out);
}

void ERIINCORE::get_gradients(const double* P, double* dJ, double* dK,
                              double alpha, double beta, double omega) {
  throw std::runtime_error("INCORE GRADIENTS NYI");
}

ERIINCORE_DF::ERIINCORE_DF(bool unr, BasisSet& obs, BasisSet& abs,
                           ParallelConfig _mpi)
    : ERI(unr, 0.0, obs, _mpi),
      eri_impl_(incore::ERI_DF::make_incore_eri(unr, obs, abs, _mpi)) {}

ERIINCORE_DF::~ERIINCORE_DF() noexcept = default;

void ERIINCORE_DF::build_JK_impl_(const double* P, double* J, double* K,
                                  double alpha, double beta, double omega) {
  if (!eri_impl_) throw std::runtime_error("ERIINCORE_DF NOT INITIALIZED");
  eri_impl_->build_JK(P, J, K, alpha, beta, omega);
}

void ERIINCORE_DF::get_gradients(const double* P, double* dJ, double* dK,
                                 double alpha, double beta, double omega) {
  if (!eri_impl_) throw std::runtime_error("ERIINCORE_DF NOT INITIALIZED");
  eri_impl_->get_gradients(P, dJ, dK, alpha, beta, omega);
}

void ERIINCORE_DF::quarter_trans_impl(size_t nt, const double* C, double* out) {
  if (!eri_impl_) throw std::runtime_error("ERIINCORE_DF NOT INITIALIZED");
  eri_impl_->quarter_trans(nt, C, out);
};

}  // namespace qdk::chemistry::scf

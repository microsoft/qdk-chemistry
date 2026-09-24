// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <cstddef>
#include <memory>
#include <qdk/chemistry/algorithms/symmetry_shift.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>

#include "microsoft/symmetry_shift/fermionic_low_rank.hpp"
#include "symmetry_shift_detail.hpp"

namespace qdk::chemistry::algorithms {

// ---------------------------------------------------------------------------
// detail: the single source of truth for the two-body correction dg_ijkl
// implied by (mu2, xi). Declared in the private header
// symmetry_shift_detail.hpp and defined side by side so they cannot drift
// apart. The SymmetryShifter implementations use the Coulomb/exchange
// contractions; the full dg tensor is the reference definition, and the
// oracle the tests check the shifted Hamiltonians against.
// ---------------------------------------------------------------------------

namespace detail {

void add_coulomb_contraction(Eigen::MatrixXd& coulomb, double mu2,
                             const Eigen::MatrixXd& xi) {
  const double norb = static_cast<double>(coulomb.rows());
  coulomb -= norb * xi;
  coulomb.diagonal().array() -= 2.0 * mu2 * norb + xi.trace();
}

void add_exchange_contraction(Eigen::MatrixXd& exchange, double mu2,
                              const Eigen::MatrixXd& xi) {
  exchange -= 2.0 * xi;
  exchange.diagonal().array() -= 2.0 * mu2;
}

void add_two_body_correction(Eigen::VectorXd& g, Eigen::Index norb, double mu2,
                             const Eigen::MatrixXd& xi) {
  const auto index = [norb](Eigen::Index i, Eigen::Index j, Eigen::Index k,
                            Eigen::Index l) {
    return ((i * norb + j) * norb + k) * norb + l;
  };

  // Term 1: -2*mu2 * delta_ij * delta_kl  (i==j and k==l).
  for (Eigen::Index i = 0; i < norb; ++i) {
    for (Eigen::Index k = 0; k < norb; ++k) {
      g[index(i, i, k, k)] -= 2.0 * mu2;
    }
  }

  // Term 2: -xi_ij * delta_kl  (k==l, all i,j).
  for (Eigen::Index i = 0; i < norb; ++i) {
    for (Eigen::Index j = 0; j < norb; ++j) {
      const double xi_ij = xi(i, j);
      for (Eigen::Index k = 0; k < norb; ++k) {
        g[index(i, j, k, k)] -= xi_ij;
      }
    }
  }

  // Term 3: -delta_ij * xi_kl  (i==j, all k,l).
  for (Eigen::Index i = 0; i < norb; ++i) {
    for (Eigen::Index k = 0; k < norb; ++k) {
      for (Eigen::Index l = 0; l < norb; ++l) {
        g[index(i, i, k, l)] -= xi(k, l);
      }
    }
  }
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Factory registration.
// ---------------------------------------------------------------------------

namespace {

std::unique_ptr<SymmetryShifter> make_fermionic_low_rank_shifter() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<microsoft::FermionicLowRankShifter>();
}

}  // namespace

void SymmetryShifterFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  SymmetryShifterFactory::register_instance(&make_fermionic_low_rank_shifter);
}

}  // namespace qdk::chemistry::algorithms

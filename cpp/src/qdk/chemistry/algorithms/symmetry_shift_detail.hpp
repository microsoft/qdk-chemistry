// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>

namespace qdk::chemistry::algorithms::detail {

// The two-body correction that the aggregated shift (mu2, xi) adds to the
// two-electron integral tensor g_ijkl:
//
//   dg_ijkl = -2*mu2*delta_ij*delta_kl - xi_ij*delta_kl - delta_ij*xi_kl
//
// Single source of truth for dg: rebuild_shifted_hamiltonian() folds the full
// tensor in via add_two_body_correction(), while a one-electron shift solver
// needs only dg's Coulomb/exchange contractions. Deriving all three from the
// one definition above, and defining them side by side in symmetry_shift.cpp,
// keeps them from silently drifting apart.
//
// Private to the library and shared between rebuild_shifted_hamiltonian() and
// the SymmetryShifter implementations; not part of the public API, which
// exchanges shifts only as SymmetryShift{mu1, mu2, xi}.

/// Fold the Coulomb-type contraction of dg into `coulomb` in place, so on
/// return it holds coul(g~) = coul(g) + Sum_k dg_ijkk, where
///   Sum_k dg_ijkk = -(2*mu2*norb + tr(xi))*delta_ij - norb*xi_ij.
/// `norb` is taken from the caller's matrix; nothing is allocated.
void add_coulomb_contraction(Eigen::MatrixXd& coulomb, double mu2,
                             const Eigen::MatrixXd& xi);

/// Fold the exchange-type contraction of dg into `exchange` in place, so on
/// return it holds exch(g~) = exch(g) + Sum_k dg_ikkj, where
///   Sum_k dg_ikkj = -2*mu2*delta_ij - 2*xi_ij.
/// Nothing is allocated.
void add_exchange_contraction(Eigen::MatrixXd& exchange, double mu2,
                              const Eigen::MatrixXd& xi);

/// Fold dg onto `g` in place, so on return `g` holds g~ = g + dg. `g` is the
/// flattened tensor ((i*norb+j)*norb+k)*norb+l with side length `norb`; no
/// separate O(norb^4) dg tensor is materialized.
///
/// Each term of dg is delta-supported, so only the non-zero blocks are
/// touched: O(norb^2) for the mu2 term and O(norb^3) for the two xi terms.
void add_two_body_correction(Eigen::VectorXd& g, Eigen::Index norb, double mu2,
                             const Eigen::MatrixXd& xi);

}  // namespace qdk::chemistry::algorithms::detail

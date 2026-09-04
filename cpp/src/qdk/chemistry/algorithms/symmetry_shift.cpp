// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <cmath>
#include <cstddef>
#include <memory>
#include <qdk/chemistry/algorithms/symmetry_shift.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>

#include "microsoft/symmetry_shift/fermionic_low_rank.hpp"
#include "symmetry_shift_detail.hpp"

namespace qdk::chemistry::algorithms {

// ---------------------------------------------------------------------------
// detail: the single source of truth for the two-body correction dg_ijkl
// implied by (mu2, xi). Declared in the private header
// symmetry_shift_detail.hpp and shared with the SymmetryShifter
// implementations, which need only dg's Coulomb/exchange contractions. The
// three are defined side by side so they cannot drift apart.
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
// rebuild_shifted_hamiltonian: apply a SymmetryShift to the dense integrals
// and rebuild. Independent of how `shift` was computed (a SymmetryShifter
// implementation or an external source).
// ---------------------------------------------------------------------------

std::shared_ptr<data::Hamiltonian> rebuild_shifted_hamiltonian(
    const data::Hamiltonian& original, const SymmetryShift& shift,
    unsigned int num_electrons) {
  using qdk::chemistry::data::CanonicalFourCenterHamiltonianContainer;
  using qdk::chemistry::data::Hamiltonian;

  if (!original.is_restricted()) {
    throw std::invalid_argument(
        "rebuild_shifted_hamiltonian currently only supports restricted "
        "(spin-restricted) Hamiltonians.");
  }

  // The BLISS operator only annihilates the Ne-electron sector (leaving its
  // energy invariant) when Ne is a non-negative integer electron count.

  auto [h_alpha, h_beta] = original.get_one_body_integrals();
  (void)h_beta;
  auto [g_aaaa, g_aabb, g_bbbb] = original.get_two_body_integrals();
  (void)g_aabb;
  (void)g_bbbb;

  const std::size_t norb = static_cast<std::size_t>(h_alpha.rows());

  if (shift.xi.rows() != static_cast<Eigen::Index>(norb) ||
      shift.xi.cols() != static_cast<Eigen::Index>(norb)) {
    throw std::invalid_argument(
        "rebuild_shifted_hamiltonian: shift.xi must be norb x norb.");
  }

  const double mu1 = shift.mu1;
  const double mu2 = shift.mu2;
  const Eigen::MatrixXd& xi = shift.xi;
  const double ne = static_cast<double>(num_electrons);

  // One-body part: h_tilde_ij = h_ij + (Ne-1)*xi_ij - (mu1+mu2)*delta_ij.
  Eigen::MatrixXd h_tilde = h_alpha + (ne - 1.0) * xi;
  h_tilde.diagonal().array() -= (mu1 + mu2);

  // Two-body part: g_tilde = g + dg, folded in place (single source of truth
  // in symmetry_shift_detail.hpp) to avoid a separate O(norb^4) allocation.
  Eigen::VectorXd g_tilde = g_aaaa;
  detail::add_two_body_correction(g_tilde, static_cast<Eigen::Index>(norb), mu2,
                                  xi);

  // Constant part of -K in the Ne-electron sector: +mu1*Ne + mu2*Ne^2.
  const double core_energy_new =
      original.get_core_energy() + mu1 * ne + mu2 * ne * ne;

  const Eigen::MatrixXd inactive_fock =
      original.has_inactive_fock_matrix()
          ? original.get_inactive_fock_matrix().first
          : Eigen::MatrixXd(0, 0);

  auto container = std::make_unique<CanonicalFourCenterHamiltonianContainer>(
      h_tilde, g_tilde, original.get_orbitals(), core_energy_new, inactive_fock,
      original.get_type());

  return std::make_shared<Hamiltonian>(std::move(container));
}

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

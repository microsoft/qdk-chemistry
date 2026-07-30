// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2.hpp"

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace kern = qdk::chemistry::algorithms::microsoft::swpt2;

SchriefferWolffPT2Settings::SchriefferWolffPT2Settings() {
  set_default("denom_floor", 1e-8,
              "Hard cutoff: skip couplings with |Delta| < floor.");
  set_default("denom_shift", 0.0,
              "CASPT2-like level shift: 1/D -> D / (D^2 + shift^2).");
  set_default("denom_flow", 1.0,
              "DSRG-style flow-parameter regularizer 1/D -> (1-exp(-s*D^2))/D "
              "(units Eh^-2); on by default to tame near-degenerate intruder "
              "channels; <= 0 disables (bare second-order PT).");
  set_default(
      "intruder_warn_amplitude", 1.0,
      "Warn when the largest raw excitation amplitude |V/Delta| exceeds "
      "this (an intruder the regularizer is compensating for); "
      "<= 0 disables the warning.");
}

std::shared_ptr<data::Hamiltonian> SchriefferWolffPT2Constructor::_run_impl(
    std::shared_ptr<data::Wavefunction> reference,
    std::shared_ptr<data::Hamiltonian> hamiltonian) const {
  // Spin-restricted method: H0 is the spin-averaged diagonal Fock and the
  // effective two-body is emitted as a single (spin-free) chemist tensor, so an
  // unrestricted input would be silently collapsed. Reject it up front (matches
  // the downstream MacisCas/Asci/Pmc solvers, which do the same).
  if (hamiltonian->is_unrestricted())
    throw std::runtime_error(
        "SchriefferWolffPT2 does not support unrestricted orbitals. "
        "Only restricted orbitals are supported.");

  // --- window integrals (spin-restricted) ---
  const auto [h1a, h1b] = hamiltonian->get_one_body_integrals();
  const auto [g_aaaa, g_aabb, g_bbbb] = hamiltonian->get_two_body_integrals();
  const double e_core = hamiltonian->get_core_energy();
  const int norb = static_cast<int>(h1a.rows());

  auto win_orbitals = hamiltonian->get_orbitals();
  auto ref_orbitals = reference->get_orbitals();

  // spatial index lists (alpha channel == spatial orbital for restricted)
  const auto W_global = data::spin_channel_indices(
      win_orbitals->active_indices(), data::axes::alpha());
  const auto P_global = data::spin_channel_indices(
      ref_orbitals->active_indices(), data::axes::alpha());
  const auto inact_global = data::spin_channel_indices(
      ref_orbitals->inactive_indices(), data::axes::alpha());

  std::unordered_map<std::size_t, int> p_pos;
  for (int k = 0; k < static_cast<int>(P_global.size()); ++k)
    p_pos[P_global[k]] = k;
  const std::unordered_set<std::size_t> inact_set(inact_global.begin(),
                                                  inact_global.end());

  // Reference active-space occupations (per active orbital, spin-traced).
  // A correlated reference exposes them as its active 1-RDM diagonal (MO
  // basis); a mean-field/HF single-determinant reference has no 1-RDM, but its
  // occupations are read directly from the reference determinant.
  Eigen::VectorXd p_occ;
  if (reference->has_active_one_rdm()) {
    const auto& rdm =
        std::get<Eigen::MatrixXd>(reference->get_active_one_rdm_spin_traced());
    p_occ = rdm.diagonal();
  } else {
    try {
      const auto [occ_a, occ_b] = reference->get_active_orbital_occupations();
      p_occ = occ_a + occ_b;
    } catch (const std::runtime_error&) {
      throw std::runtime_error(
          "SchriefferWolffPT2: reference wavefunction exposes neither an "
          "active "
          "1-RDM nor active orbital occupations");
    }
  }

  // classify every window orbital: active (P) / inactive (domo) / virtual
  std::vector<int> active_spatial;
  std::vector<double> occupation(norb, 0.0);
  for (int i = 0; i < norb; ++i) {
    const std::size_t g = W_global[i];
    const auto it = p_pos.find(g);
    if (it != p_pos.end()) {
      active_spatial.push_back(i);
      occupation[i] = p_occ(it->second);
    } else if (inact_set.count(g)) {
      occupation[i] = 2.0;
    } else {
      occupation[i] = 0.0;
    }
  }

  // The emitted active integrals are ordered by window index; the reused
  // reference orbitals expect them in reference-active order. Require the two
  // to agree (both ascending) -- general reordering is not yet supported.
  for (int k = 0; k < static_cast<int>(active_spatial.size()); ++k)
    if (W_global[active_spatial[k]] != P_global[k])
      throw std::runtime_error(
          "SchriefferWolffPT2: active-orbital ordering in the window does not "
          "match the reference active space (not yet supported)");

  // --- kernel pipeline (spin-blocked: the antisymmetric two-body tensor is
  // stored as spatial spin blocks and every element is formed on the fly, so
  // the dense n_so^4 objects are never materialized) ---
  const auto blk = kern::build_two_body_blocked(g_aaaa, g_aabb, g_bbbb, norb);
  const auto f = kern::spin_orbital_one_body(h1a, h1b, norb);
  const auto part = kern::make_partition(norb, active_spatial, occupation);

  Eigen::VectorXd na = Eigen::VectorXd::Zero(norb);
  Eigen::VectorXd nb = Eigen::VectorXd::Zero(norb);
  for (int i = 0; i < norb; ++i) {
    na(i) = 0.5 * occupation[i];
    nb(i) = 0.5 * occupation[i];
  }
  const auto eps = kern::diagonal_fock_energies(h1a, g_aaaa, na, nb, norb);

  kern::RegOptions reg;
  reg.denom_floor = _settings->get<double>("denom_floor");
  reg.denom_shift = _settings->get<double>("denom_shift");
  reg.denom_flow = _settings->get<double>("denom_flow");

  const auto down = kern::downfold_blocked(f, blk, eps, part, reg, e_core);

  // Intruder warning is gated on the RAW (unregularized) amplitude: the
  // regularizer damps the operator, so warning on regularized amplitudes would
  // hide exactly the near-degenerate channels it is compensating for. A large
  // raw |V/Delta| means the result leans on regularization -- surface it.
  const double warn_amp = _settings->get<double>("intruder_warn_amplitude");
  if (warn_amp > 0.0 && down.max_amplitude > warn_amp) {
    QDK_LOGGER().warn(
        "swpt2 downfold: large excitation amplitude {:.3g} (smallest energy "
        "denominator {:.3g} Eh) -- a near-degenerate/intruder channel. The "
        "result relies on denominator regularization (denom_flow={:.3g}, "
        "denom_shift={:.3g}); consider enlarging the active space so "
        "near-degenerate orbitals are not split across the active/external "
        "boundary.",
        down.max_amplitude, down.min_denominator, reg.denom_flow,
        reg.denom_shift);
  }

  const auto active = kern::to_spatial_chemist(down, part);

  // --- emit over P, reusing the reference orbitals (active_indices == P) ---
  const Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);
  return std::make_shared<data::Hamiltonian>(
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          active.one_body, active.two_body, ref_orbitals, active.core_energy,
          empty_fock));
}

}  // namespace qdk::chemistry::algorithms::microsoft

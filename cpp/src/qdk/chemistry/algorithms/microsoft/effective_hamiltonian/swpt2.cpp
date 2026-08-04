// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <limits>
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
  set_default("regularizer", std::string("flow"),
              "Energy-denominator regularizer: 'flow' (default), 'shift', "
              "or 'bare'.",
              data::ListConstraint<std::string>{
                  {std::vector<std::string>{"flow", "shift", "bare"}}});
  set_default(
      "denom_floor", 1e-8,
      "Hard cutoff used by bare denominators and raw-amplitude "
      "diagnostics.",
      data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
  set_default(
      "denom_shift", 0.0,
      "CASPT2-like level shift used when regularizer='shift': "
      "1/D -> D / (D^2 + shift^2).",
      data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
  set_default(
      "denom_flow", 1.0,
      "Flow-parameter denominator regularizer "
      "1/D -> (1-exp(-s*D^2))/D (units Eh^-2), used when "
      "regularizer='flow'. This borrows the DSRG damping form; it does not "
      "turn the downfold into a full DSRG calculation.",
      data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
  set_default(
      "intruder_warn_amplitude", 1.0,
      "Warn when the largest raw excitation amplitude |V/Delta| exceeds "
      "this (an intruder the regularizer is compensating for); "
      "<= 0 disables the warning.");
  set_default("semicanonicalize", true,
              "Diagonalize the generalized Fock independently within the "
              "inactive, active, and virtual blocks before forming Fock "
              "denominators.");
  set_default(
      "semicanonical_tolerance", 1e-10,
      "Skip a block rotation when its largest off-diagonal Fock element does "
      "not exceed this threshold.",
      data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
}

std::shared_ptr<data::Hamiltonian> SchriefferWolffPT2Constructor::_run_impl(
    std::shared_ptr<data::Wavefunction> reference,
    std::shared_ptr<data::Hamiltonian> hamiltonian) const {
  if (!reference || !hamiltonian)
    throw std::invalid_argument(
        "SchriefferWolffPT2 requires non-null reference and Hamiltonian "
        "inputs.");

  // Spin-restricted method: H0 is the spin-averaged diagonal Fock and the
  // effective two-body is emitted as a single (spin-free) chemist tensor, so an
  // unrestricted input would be silently collapsed. Reject it up front (matches
  // the downstream MacisCas/Asci/Pmc solvers, which do the same).
  if (hamiltonian->is_unrestricted())
    throw std::runtime_error(
        "SchriefferWolffPT2 does not support unrestricted orbitals. "
        "Only restricted orbitals are supported.");

  // --- window integrals (spin-restricted) ---
  const auto [h1a_input, h1b_input] = hamiltonian->get_one_body_integrals();
  const auto two_body_inputs = hamiltonian->get_two_body_integrals();
  const auto& g_aaaa_input = std::get<0>(two_body_inputs);
  Eigen::MatrixXd h1a = h1a_input;
  Eigen::MatrixXd h1b = h1b_input;
  // Restricted Hamiltonians alias all three spin channels. Keep one spatial
  // chemist tensor so semicanonicalization performs one N^4 copy/rotation.
  Eigen::VectorXd g_aaaa = g_aaaa_input;
  const double e_core = hamiltonian->get_core_energy();
  const int norb = static_cast<int>(h1a.rows());

  auto win_orbitals = hamiltonian->get_orbitals();
  auto ref_orbitals = reference->get_orbitals();
  if (ref_orbitals->is_unrestricted())
    throw std::invalid_argument(
        "SchriefferWolffPT2 does not support unrestricted reference "
        "orbitals. Only restricted orbitals are supported.");

  // Global orbital labels are meaningful only when both inputs use the same
  // MO basis. Different active/inactive index sets are expected; different MO
  // coefficient matrices are not.
  const auto& win_coeff = win_orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  const auto& ref_coeff = ref_orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  if (win_coeff.rows() != ref_coeff.rows() ||
      win_coeff.cols() != ref_coeff.cols() ||
      !win_coeff.isApprox(ref_coeff, 1e-10))
    throw std::invalid_argument(
        "SchriefferWolffPT2 requires the reference and window Hamiltonian "
        "to use the same molecular-orbital basis.");

  // spatial index lists (alpha channel == spatial orbital for restricted)
  const auto W_global = data::spin_channel_indices(
      win_orbitals->active_indices(), data::axes::alpha());
  const auto P_global = data::spin_channel_indices(
      ref_orbitals->active_indices(), data::axes::alpha());
  const auto inact_global = data::spin_channel_indices(
      ref_orbitals->inactive_indices(), data::axes::alpha());
  if (static_cast<int>(W_global.size()) != norb)
    throw std::invalid_argument(
        "SchriefferWolffPT2: window active-space size does not match the "
        "Hamiltonian integral dimensions.");

  const auto [total_alpha, total_beta] = reference->get_total_num_electrons();
  const auto [active_alpha, active_beta] =
      reference->get_active_num_electrons();
  if (active_alpha > total_alpha || active_beta > total_beta ||
      total_alpha - active_alpha != inact_global.size() ||
      total_beta - active_beta != inact_global.size())
    throw std::invalid_argument(
        "SchriefferWolffPT2 requires all singly occupied orbitals in an "
        "open-shell reference to belong to the active space; inactive "
        "orbitals must be doubly occupied.");

  std::unordered_map<std::size_t, int> p_pos;
  for (int k = 0; k < static_cast<int>(P_global.size()); ++k)
    p_pos[P_global[k]] = k;
  const std::unordered_set<std::size_t> inact_set(inact_global.begin(),
                                                  inact_global.end());

  // Reference active-space occupations (per active orbital, spin-traced).
  // For ROHF this maps each singly occupied active orbital to occupation one;
  // the spin-free H0 then preserves S^2 and the active solve selects the
  // desired spin sector.
  // A correlated reference exposes them as its active 1-RDM diagonal (MO
  // basis); a mean-field/HF single-determinant reference has no 1-RDM, but its
  // occupations are read directly from the reference determinant.
  Eigen::MatrixXd p_density;
  if (reference->has_active_one_rdm()) {
    const auto& rdm_variant = reference->get_active_one_rdm_spin_traced();
    const auto* rdm = std::get_if<Eigen::MatrixXd>(&rdm_variant);
    if (!rdm)
      throw std::invalid_argument(
          "SchriefferWolffPT2 requires a real-valued active 1-RDM.");
    if (rdm->rows() != static_cast<Eigen::Index>(P_global.size()) ||
        rdm->cols() != static_cast<Eigen::Index>(P_global.size()))
      throw std::invalid_argument(
          "SchriefferWolffPT2: active 1-RDM dimensions do not match the "
          "reference active-space size.");
    p_density = 0.5 * (*rdm + rdm->transpose()).eval();
  } else {
    try {
      const auto [occ_a, occ_b] = reference->get_active_orbital_occupations();
      if (occ_a.size() != static_cast<Eigen::Index>(P_global.size()) ||
          occ_b.size() != static_cast<Eigen::Index>(P_global.size()))
        throw std::invalid_argument(
            "SchriefferWolffPT2: active occupation dimensions do not match "
            "the reference active-space size.");
      p_density = (occ_a + occ_b).asDiagonal();
    } catch (const std::runtime_error&) {
      throw std::runtime_error(
          "SchriefferWolffPT2: reference wavefunction exposes neither an "
          "active "
          "1-RDM nor active orbital occupations");
    }
  }

  // classify every window orbital: active (P) / inactive (domo) / virtual
  std::vector<int> active_spatial, inactive_spatial, virtual_spatial;
  std::vector<double> occupation(norb, 0.0);
  Eigen::MatrixXd density = Eigen::MatrixXd::Zero(norb, norb);
  for (int i = 0; i < norb; ++i) {
    const std::size_t g = W_global[i];
    const auto it = p_pos.find(g);
    if (it != p_pos.end()) {
      active_spatial.push_back(i);
      occupation[i] = p_density(it->second, it->second);
    } else if (inact_set.count(g)) {
      inactive_spatial.push_back(i);
      occupation[i] = 2.0;
      density(i, i) = 2.0;
    } else {
      virtual_spatial.push_back(i);
      occupation[i] = 0.0;
    }
  }

  for (int i : active_spatial)
    for (int j : active_spatial)
      density(i, j) = p_density(p_pos.at(W_global[i]), p_pos.at(W_global[j]));

  if (active_spatial.size() != P_global.size())
    throw std::invalid_argument(
        "SchriefferWolffPT2: the reference active space is not fully "
        "contained in the window Hamiltonian.");

  // The emitted active integrals follow window order, while the reused
  // reference orbitals expect reference-active order. Require them to agree;
  // permutation into a different reference order is not implemented.
  for (int k = 0; k < static_cast<int>(active_spatial.size()); ++k)
    if (W_global[active_spatial[k]] != P_global[k])
      throw std::runtime_error(
          "SchriefferWolffPT2: active-orbital ordering in the window does not "
          "match the reference active-space ordering");

  Eigen::MatrixXd semicanonical_rotation =
      Eigen::MatrixXd::Identity(norb, norb);
  bool semicanonical_rotation_applied = false;
  Eigen::VectorXd eps;
  if (_settings->get<bool>("semicanonicalize")) {
    const double tolerance = _settings->get<double>("semicanonical_tolerance");
    if (!std::isfinite(tolerance) || tolerance < 0.0)
      throw std::invalid_argument(
          "SchriefferWolffPT2: semicanonical_tolerance must be finite and "
          "non-negative.");
    const auto fock = kern::generalized_fock_matrix(h1a, g_aaaa, density, norb);
    semicanonical_rotation = kern::semicanonical_rotation(
        fock, {inactive_spatial, active_spatial, virtual_spatial}, tolerance);
    semicanonical_rotation_applied = !semicanonical_rotation.isIdentity(0.0);
    Eigen::MatrixXd fock_rotated = fock;
    if (semicanonical_rotation_applied) {
      h1a = kern::rotate_one_body(h1a, semicanonical_rotation);
      h1b = kern::rotate_one_body(h1b, semicanonical_rotation);
      g_aaaa = kern::rotate_two_body(g_aaaa, semicanonical_rotation, norb);
      density = kern::rotate_one_body(density, semicanonical_rotation);
      fock_rotated = kern::rotate_one_body(fock, semicanonical_rotation);
    }
    eps = Eigen::VectorXd::Zero(2 * norb);
    for (int i = 0; i < norb; ++i) {
      occupation[i] = density(i, i);
      eps(2 * i) = fock_rotated(i, i);
      eps(2 * i + 1) = fock_rotated(i, i);
    }
  }

  // --- kernel pipeline (spin-blocked: the antisymmetric two-body tensor is
  // stored as spatial spin blocks and every element is formed on the fly, so
  // the dense n_so^4 objects are never materialized) ---
  const auto blk = kern::build_two_body_blocked_restricted(g_aaaa, norb);
  const auto f = kern::spin_orbital_one_body(h1a, h1b, norb);
  const auto part = kern::make_partition(norb, active_spatial, occupation);

  if (eps.size() == 0) {
    Eigen::VectorXd na = Eigen::VectorXd::Zero(norb);
    Eigen::VectorXd nb = Eigen::VectorXd::Zero(norb);
    for (int i = 0; i < norb; ++i) {
      na(i) = 0.5 * occupation[i];
      nb(i) = 0.5 * occupation[i];
    }
    eps = kern::diagonal_fock_energies(h1a, g_aaaa, na, nb, norb);
  }

  kern::RegOptions reg;
  reg.denom_floor = _settings->get<double>("denom_floor");
  if (!std::isfinite(reg.denom_floor) || reg.denom_floor <= 0.0)
    throw std::invalid_argument(
        "SchriefferWolffPT2: denom_floor must be finite and positive.");
  const std::string regularizer = _settings->get<std::string>("regularizer");
  if (regularizer == "flow") {
    reg.denom_flow = _settings->get<double>("denom_flow");
    reg.denom_shift = 0.0;
    if (!std::isfinite(reg.denom_flow) || reg.denom_flow <= 0.0)
      throw std::invalid_argument(
          "SchriefferWolffPT2: denom_flow must be finite and positive when "
          "regularizer='flow'.");
  } else if (regularizer == "shift") {
    reg.denom_flow = -1.0;
    reg.denom_shift = _settings->get<double>("denom_shift");
    if (!std::isfinite(reg.denom_shift) || reg.denom_shift <= 0.0)
      throw std::invalid_argument(
          "SchriefferWolffPT2: denom_shift must be finite and positive when "
          "regularizer='shift'.");
  } else {
    reg.denom_flow = -1.0;
    reg.denom_shift = 0.0;
  }

  const auto down = kern::downfold_blocked(f, blk, eps, part, reg, e_core);

  // Intruder warning is gated on the RAW (unregularized) amplitude: the
  // regularizer damps the operator, so warning on regularized amplitudes would
  // hide exactly the near-degenerate channels it is compensating for. A large
  // raw |V/Delta| means the result leans on regularization -- surface it.
  const double warn_amp = _settings->get<double>("intruder_warn_amplitude");
  if (!std::isfinite(warn_amp))
    throw std::invalid_argument(
        "SchriefferWolffPT2: intruder_warn_amplitude must be finite.");
  QDK_LOGGER().info(
      "SW-PT2 downfold complete: regularizer='{}', minimum denominator={:.3g} "
      "Eh, maximum raw amplitude={:.3g}, semicanonical rotation applied={}",
      regularizer, down.min_denominator, down.max_amplitude,
      semicanonical_rotation_applied);
  if (warn_amp > 0.0 && down.max_amplitude > warn_amp) {
    if (regularizer == "bare")
      QDK_LOGGER().warn(
          "swpt2 downfold: large excitation amplitude {:.3g} (smallest "
          "energy denominator {:.3g} Eh) -- the bare second-order result may "
          "be unreliable. Consider enlarging the active space or selecting a "
          "denominator regularizer.",
          down.max_amplitude, down.min_denominator);
    else
      QDK_LOGGER().warn(
          "swpt2 downfold: large excitation amplitude {:.3g} (smallest energy "
          "denominator {:.3g} Eh) -- a near-degenerate/intruder channel. The "
          "result relies on '{}' denominator regularization; consider "
          "enlarging the active space so near-degenerate orbitals are not "
          "split across the active/external boundary.",
          down.max_amplitude, down.min_denominator, regularizer);
  }

  auto active = kern::to_spatial_chemist(down, part);
  if (semicanonical_rotation_applied) {
    const int nactive = static_cast<int>(active_spatial.size());
    Eigen::MatrixXd active_rotation(nactive, nactive);
    for (int i = 0; i < nactive; ++i)
      for (int j = 0; j < nactive; ++j)
        active_rotation(i, j) =
            semicanonical_rotation(active_spatial[i], active_spatial[j]);
    active.one_body =
        kern::rotate_one_body(active.one_body, active_rotation.transpose());
    active.two_body = kern::rotate_two_body(
        active.two_body, active_rotation.transpose(), nactive);
  }

  // --- emit over P, reusing the reference orbitals (active_indices == P) ---
  const Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);
  return std::make_shared<data::Hamiltonian>(
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          active.one_body, active.two_body, ref_orbitals, active.core_energy,
          empty_fock));
}

}  // namespace qdk::chemistry::algorithms::microsoft

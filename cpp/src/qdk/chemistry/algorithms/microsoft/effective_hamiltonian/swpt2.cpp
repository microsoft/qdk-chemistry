// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
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
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices) const {
  if (!reference || !hamiltonian)
    throw std::invalid_argument(
        "SchriefferWolffPT2 requires non-null reference and Hamiltonian "
        "inputs.");
  if (!p_indices)
    throw std::invalid_argument(
        "SchriefferWolffPT2 requires a non-null p_indices argument: the "
        "kept space P as a SymmetryBlockedIndexSet of global (spatial) orbital "
        "indices into the window Hamiltonian's active space W = P u Q.");
  // Kept space P as global (spatial) MO indices; the alpha channel is the
  // spatial orbital for a restricted method.
  const std::vector<std::size_t> kept_global =
      data::spin_channel_indices(p_indices, data::axes::alpha());
  if (kept_global.empty())
    throw std::invalid_argument(
        "SchriefferWolffPT2 requires a non-empty p_indices argument: the "
        "kept space P as a SymmetryBlockedIndexSet of global (spatial) orbital "
        "indices into the window Hamiltonian's active space W = P u Q.");

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

  // Reference density and occupation over the whole window W, taken from the
  // reference roles (active 1-RDM / doubly-occupied inactive / empty virtual).
  // This is independent of the kept space P chosen below.
  std::vector<double> occupation(norb, 0.0);
  Eigen::MatrixXd density = Eigen::MatrixXd::Zero(norb, norb);
  int ref_active_in_window = 0;
  for (int i = 0; i < norb; ++i) {
    const std::size_t g = W_global[i];
    const auto it = p_pos.find(g);
    if (it != p_pos.end()) {
      occupation[i] = p_density(it->second, it->second);
      ++ref_active_in_window;
    } else if (inact_set.count(g)) {
      occupation[i] = 2.0;
      density(i, i) = 2.0;
    }
  }
  for (int i = 0; i < norb; ++i) {
    const auto ii = p_pos.find(W_global[i]);
    if (ii == p_pos.end()) continue;
    for (int j = 0; j < norb; ++j) {
      const auto jj = p_pos.find(W_global[j]);
      if (jj != p_pos.end()) density(i, j) = p_density(ii->second, jj->second);
    }
  }
  if (ref_active_in_window != static_cast<int>(P_global.size()))
    throw std::invalid_argument(
        "SchriefferWolffPT2: the reference active space is not fully "
        "contained in the window Hamiltonian.");

  // Kept space P: the mandatory explicit index set of global (spatial) MO
  // indices into the window W = P u Q (a run() argument, extracted above). The
  // reference wavefunction supplies the density over W; P selects which
  // orbitals are kept exactly.
  const std::unordered_set<std::size_t> kept_set(kept_global.begin(),
                                                 kept_global.end());
  if (kept_set.size() != kept_global.size())
    throw std::invalid_argument(
        "SchriefferWolffPT2: p_indices contains duplicate orbitals.");
  const std::unordered_set<std::size_t> window_set(W_global.begin(),
                                                   W_global.end());
  for (std::size_t g : kept_global)
    if (!window_set.count(g))
      throw std::invalid_argument(
          "SchriefferWolffPT2: every p_indices orbital must lie in the "
          "window Hamiltonian's active space W = P u Q.");

  // Partition the window: kept -> active P; the folded rest -> Q, split by the
  // reference occupation into inactive (doubly occupied) / virtual (empty).
  // Folded orbitals must be closed-shell; keep open-shell/fractionally occupied
  // orbitals in P.
  constexpr double occupation_tolerance = 1e-6;
  std::vector<int> active_spatial, inactive_spatial, virtual_spatial;
  for (int i = 0; i < norb; ++i) {
    if (kept_set.count(W_global[i]))
      active_spatial.push_back(i);
    else if (std::abs(occupation[i] - 2.0) <= occupation_tolerance)
      inactive_spatial.push_back(i);
    else if (std::abs(occupation[i]) <= occupation_tolerance)
      virtual_spatial.push_back(i);
    else
      throw std::invalid_argument(
          "SchriefferWolffPT2: a folded external orbital has fractional "
          "reference occupation; keep open-shell / partially occupied "
          "orbitals in the active space P (active_indices).");
  }
  if (active_spatial.empty())
    throw std::invalid_argument(
        "SchriefferWolffPT2: the kept active space P is empty.");

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
    // Denominators from the full-density generalized Fock (the same operator as
    // the semicanonical branch), so a correlated reference's off-diagonal 1-RDM
    // is retained rather than dropped by a diagonal-occupation Fock.
    const auto fock = kern::generalized_fock_matrix(h1a, g_aaaa, density, norb);
    eps = Eigen::VectorXd::Zero(2 * norb);
    for (int i = 0; i < norb; ++i) {
      eps(2 * i) = fock(i, i);
      eps(2 * i + 1) = fock(i, i);
    }
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

  // --- emit over P ---
  // The effective operator lives on P, so it must be labeled by orbitals whose
  // active index set is P. `Orbitals` is immutable, so reuse the reference
  // orbitals' MO coefficients / energies / overlap / basis and only relabel the
  // active (P) and inactive (folded doubly-occupied core) index sets; no new
  // orbitals are computed.
  std::vector<std::size_t> emit_active, emit_inactive;
  for (int i : active_spatial) emit_active.push_back(W_global[i]);
  for (int i : inactive_spatial) emit_inactive.push_back(W_global[i]);
  // The reference core (folded into the window's core energy) is inactive too,
  // except any core orbital kept in P. It may overlap the folded window
  // orbitals when the window already spans the core, so deduplicate.
  for (std::size_t g : inact_global)
    if (!kept_set.count(g)) emit_inactive.push_back(g);
  std::sort(emit_active.begin(), emit_active.end());
  std::sort(emit_inactive.begin(), emit_inactive.end());
  emit_inactive.erase(std::unique(emit_inactive.begin(), emit_inactive.end()),
                      emit_inactive.end());

  const auto ref_active_set = ref_orbitals->active_indices();
  const auto make_index_set = [&](const std::vector<std::size_t>& idx) {
    std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>> indices;
    for (const auto& label : ref_active_set->labels())
      indices[label] = std::vector<std::uint32_t>(idx.begin(), idx.end());
    return std::make_shared<const data::SymmetryBlockedIndexSet>(
        ref_active_set->symmetries(), ref_active_set->extents(),
        std::move(indices));
  };

  const Eigen::MatrixXd ref_coeffs = ref_orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  std::optional<Eigen::VectorXd> energies;
  if (ref_orbitals->has_energies())
    energies = ref_orbitals->energies()->block({data::axes::alpha()});
  std::optional<Eigen::MatrixXd> overlap;
  if (ref_orbitals->has_overlap_matrix())
    overlap = ref_orbitals->get_overlap_matrix();
  std::shared_ptr<data::BasisSet> basis;
  if (ref_orbitals->has_basis_set()) basis = ref_orbitals->get_basis_set();
  auto emit_orbitals = std::make_shared<data::Orbitals>(
      ref_coeffs, energies, overlap, basis, make_index_set(emit_active),
      make_index_set(emit_inactive));

  const Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);
  return std::make_shared<data::Hamiltonian>(
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          active.one_body, active.two_body, emit_orbitals, active.core_energy,
          empty_fock));
}

}  // namespace qdk::chemistry::algorithms::microsoft

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace sw = qdk::chemistry::algorithms::microsoft::swpt2;

namespace {

/// The reference's spin-traced 1-RDM over its own active space, from the
/// active 1-RDM when available and from the reference determinant otherwise.
Eigen::MatrixXd reference_active_density(const data::Wavefunction& reference,
                                         std::size_t n_ref_active) {
  const auto size = static_cast<Eigen::Index>(n_ref_active);
  if (reference.has_active_one_rdm()) {
    const auto& rdm_variant = reference.get_active_one_rdm_spin_traced();
    const auto* rdm = std::get_if<Eigen::MatrixXd>(&rdm_variant);
    if (!rdm)
      throw std::invalid_argument(
          "SchriefferWolffPT2 requires a real-valued active 1-RDM.");
    if (rdm->rows() != size || rdm->cols() != size)
      throw std::invalid_argument(
          "SchriefferWolffPT2: active 1-RDM dimensions do not match the "
          "reference active-space size.");
    return 0.5 * (*rdm + rdm->transpose()).eval();
  }

  try {
    const auto [occ_a, occ_b] = reference.get_active_orbital_occupations();
    if (occ_a.size() != size || occ_b.size() != size)
      throw std::invalid_argument(
          "SchriefferWolffPT2: active occupation dimensions do not match "
          "the reference active-space size.");
    // This accessor is per-active-index only for a determinant; for a
    // multi-determinant state it returns 1-RDM eigenvalues sorted descending,
    // which need not be the MO-basis diagonal these indices label. Rejecting
    // fractional values catches the common case.
    constexpr double determinant_tolerance = 1e-6;
    for (Eigen::Index k = 0; k < occ_a.size(); ++k)
      for (double n : {occ_a(k), occ_b(k)})
        if (std::abs(n) > determinant_tolerance &&
            std::abs(n - 1.0) > determinant_tolerance)
          throw std::invalid_argument(
              "SchriefferWolffPT2: the reference reports fractional active "
              "orbital occupations but exposes no active 1-RDM, so they "
              "cannot be assigned to orbitals. Enable one-RDM calculation "
              "on the reference.");
    return Eigen::MatrixXd((occ_a + occ_b).asDiagonal());
  } catch (const std::runtime_error&) {
    throw std::runtime_error(
        "SchriefferWolffPT2: reference wavefunction exposes neither an "
        "active 1-RDM nor active orbital occupations");
  }
}

/// The reference's spin-traced 1-RDM over the whole window W: its active 1-RDM
/// block, 2 on the diagonal for doubly-occupied inactive orbitals, 0 for empty
/// virtuals. Off-diagonals are therefore confined to the active block, and the
/// diagonal is the occupation, so the trace is the window electron count.
Eigen::MatrixXd window_density(
    const std::vector<std::size_t>& window_global,
    const std::vector<std::size_t>& ref_active_global,
    const std::unordered_set<std::size_t>& ref_inactive_set,
    const Eigen::MatrixXd& ref_active_density) {
  std::unordered_map<std::size_t, int> ref_active_pos;
  for (int k = 0; k < static_cast<int>(ref_active_global.size()); ++k)
    ref_active_pos[ref_active_global[k]] = k;

  const int norb = static_cast<int>(window_global.size());
  Eigen::MatrixXd density = Eigen::MatrixXd::Zero(norb, norb);
  std::vector<int> active_position(norb, -1);
  for (int i = 0; i < norb; ++i) {
    const auto it = ref_active_pos.find(window_global[i]);
    if (it != ref_active_pos.end())
      active_position[i] = it->second;
    else if (ref_inactive_set.count(window_global[i]))
      density(i, i) = 2.0;
  }

  for (int i = 0; i < norb; ++i) {
    if (active_position[i] < 0) continue;
    for (int j = 0; j < norb; ++j)
      if (active_position[j] >= 0)
        density(i, j) =
            ref_active_density(active_position[i], active_position[j]);
  }

  constexpr double occupation_bound_tolerance = 1e-6;
  for (int i = 0; i < norb; ++i)
    if (density(i, i) < -occupation_bound_tolerance ||
        density(i, i) > 2.0 + occupation_bound_tolerance)
      throw std::invalid_argument("SchriefferWolffPT2: window orbital " +
                                  std::to_string(window_global[i]) +
                                  " has unphysical reference occupation " +
                                  std::to_string(density(i, i)) +
                                  "; expected a value in [0, 2].");
  return density;
}

/// Label the effective operator with `p_indices` as its active index set.
/// `Orbitals` is immutable, so the reference orbitals' coefficients, energies,
/// overlap, and basis are reused and only the index sets are relabeled.
std::shared_ptr<data::Orbitals> relabeled_orbitals(
    const data::Orbitals& ref_orbitals,
    const std::shared_ptr<const data::SymmetryBlockedIndexSet>& p_indices,
    const std::vector<std::size_t>& emit_inactive) {
  std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>> inactive;
  for (const auto& label : p_indices->labels())
    inactive[label] =
        std::vector<std::uint32_t>(emit_inactive.begin(), emit_inactive.end());

  std::optional<Eigen::VectorXd> energies;
  if (ref_orbitals.has_energies())
    energies = ref_orbitals.energies()->block({data::axes::alpha()});
  std::optional<Eigen::MatrixXd> overlap;
  if (ref_orbitals.has_overlap_matrix())
    overlap = ref_orbitals.get_overlap_matrix();
  std::shared_ptr<data::BasisSet> basis;
  if (ref_orbitals.has_basis_set()) basis = ref_orbitals.get_basis_set();
  return std::make_shared<data::Orbitals>(
      ref_orbitals.coefficients()->block(
          {data::axes::alpha(), data::axes::alpha()}),
      energies, overlap, basis, p_indices,
      std::make_shared<const data::SymmetryBlockedIndexSet>(
          p_indices->symmetries(), p_indices->extents(), std::move(inactive)));
}

/// Square sub-block of `matrix` on the given window positions.
Eigen::MatrixXd sub_block(const Eigen::MatrixXd& matrix,
                          const std::vector<int>& positions) {
  const int n = static_cast<int>(positions.size());
  Eigen::MatrixXd block(n, n);
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j)
      block(i, j) = matrix(positions[i], positions[j]);
  return block;
}

void log_fold_rounding(const sw::WindowPartition& roles) {
  QDK_LOGGER().info(
      "SW-PT2 partition: active={}, folded inactive={}, folded virtual={}; "
      "active electrons={}, largest folded occupation deviation={:.3g}, "
      "folded core electron excess={:.3g}",
      roles.active_spatial.size(), roles.inactive_spatial.size(),
      roles.virtual_spatial.size(), roles.active_electrons,
      roles.worst_deviation, roles.folded_charge_error);

  // Roundings of opposite sign cancel, so a correlated pair folded together is
  // benign; a net charge error is not.
  constexpr double charge_error_warning = 0.01;
  // The default occupation_threshold of OccupationActiveSpaceSelector: an
  // orbital that selector would have kept active was folded anyway.
  constexpr double deviation_warning = 0.1;
  if (std::abs(roles.folded_charge_error) <= charge_error_warning &&
      roles.worst_deviation <= deviation_warning)
    return;
  QDK_LOGGER().warn(
      "swpt2 downfold: folded orbital {} has fractional reference "
      "occupation {:.4f} (deviation {:.3g}), and the folded core carries "
      "{:.3g} electrons more than the reference density. Rounding the "
      "folded density perturbs the mean field the active space feels at "
      "first order -- an error the regularizer does not damp. Keeping a "
      "correlated pair together on the folded side makes its roundings "
      "cancel.",
      roles.worst_orbital, roles.worst_occupation, roles.worst_deviation,
      roles.folded_charge_error);
}

void warn_on_intruders(const sw::ActiveDownfoldResult& down, bool regularized) {
  if (!regularized && down.min_denominator < sw::bare_denominator_floor)
    QDK_LOGGER().warn(
        "swpt2 downfold: coupled channels with energy denominators below "
        "{:.3g} Eh were omitted by the guarded bare pseudoinverse. The "
        "remaining channels were downfolded, but the result depends on this "
        "cutoff; consider enlarging the active space or setting "
        "regularizer_sigma2.",
        sw::bare_denominator_floor);

  // Warn on the RAW amplitude: the regularizer damps the operator, so a
  // regularized amplitude would hide the very channels it compensates for.
  // 1.0 is where the perturbation series stops contracting, and it sits in a
  // wide empirical gap -- benign folds measured here top out near 0.51, a
  // mismatched kept space reaches 1.6-3.0.
  constexpr double intruder_warn_amplitude = 1.0;
  if (down.max_amplitude <= intruder_warn_amplitude) return;
  if (!regularized)
    QDK_LOGGER().warn(
        "swpt2 downfold: large excitation amplitude {:.3g} (smallest "
        "energy denominator {:.3g} Eh) -- the unregularized second-order "
        "result may be unreliable. Consider enlarging the active space, or "
        "setting regularizer_sigma2.",
        down.max_amplitude, down.min_denominator);
  else
    QDK_LOGGER().warn(
        "swpt2 downfold: large excitation amplitude {:.3g} (smallest energy "
        "denominator {:.3g} Eh) -- a near-degenerate/intruder channel. The "
        "result relies on sigma^2 denominator regularization; consider "
        "enlarging the active space so near-degenerate orbitals are not "
        "split across the active/external boundary.",
        down.max_amplitude, down.min_denominator);
}

}  // namespace

std::shared_ptr<data::Hamiltonian> SchriefferWolffPT2Constructor::_run_impl(
    std::shared_ptr<data::Wavefunction> reference,
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices) const {
  QDK_LOG_TRACE_ENTERING();
  if (!reference || !hamiltonian || !p_indices)
    throw std::invalid_argument(
        "SchriefferWolffPT2: reference, hamiltonian, and p_indices must all "
        "be non-null.");
  if (hamiltonian->is_unrestricted())
    throw std::runtime_error(
        "SchriefferWolffPT2 does not support unrestricted orbitals. "
        "Only restricted orbitals are supported.");
  auto win_orbitals = hamiltonian->get_orbitals();
  auto ref_orbitals = reference->get_orbitals();
  if (ref_orbitals->is_unrestricted())
    throw std::invalid_argument(
        "SchriefferWolffPT2 does not support unrestricted reference "
        "orbitals. Only restricted orbitals are supported.");
  // Global orbital labels are only meaningful in a shared MO basis; this also
  // checks P and the reference active space against the window.
  _validate_inputs(reference, hamiltonian, p_indices);

  // The alpha channel is the spatial orbital for a restricted method.
  const std::vector<std::size_t> kept_global =
      data::spin_channel_indices(p_indices, data::axes::alpha());
  if (kept_global.empty())
    throw std::invalid_argument(
        "SchriefferWolffPT2 requires a non-empty p_indices argument: the "
        "kept space P as a SymmetryBlockedIndexSet of global (spatial) orbital "
        "indices into the window Hamiltonian's active space W = P u Q.");
  if (data::spin_channel_indices(p_indices, data::axes::beta()) != kept_global)
    throw std::invalid_argument(
        "SchriefferWolffPT2 requires p_indices to select the same orbitals in "
        "both spin channels; this is a spin-restricted method.");

  // Unrestricted input is rejected above, so the spin channels alias: one
  // spatial tensor suffices and is rotated once.
  Eigen::MatrixXd h1a = std::get<0>(hamiltonian->get_one_body_integrals());
  Eigen::VectorXd g_aaaa = std::get<0>(hamiltonian->get_two_body_integrals());
  const double e_core = hamiltonian->get_core_energy();
  const int norb = static_cast<int>(h1a.rows());

  const auto W_global = data::spin_channel_indices(
      win_orbitals->active_indices(), data::axes::alpha());
  if (W_global.size() != static_cast<std::size_t>(norb) ||
      data::spin_channel_indices(win_orbitals->active_indices(),
                                 data::axes::beta()) != W_global)
    throw std::invalid_argument(
        "SchriefferWolffPT2: the window Hamiltonian's active index set must "
        "be spin-independent and match the rank of its integrals.");
  const auto win_inactive_global = data::spin_channel_indices(
      win_orbitals->inactive_indices(), data::axes::alpha());
  const auto ref_active_global = data::spin_channel_indices(
      ref_orbitals->active_indices(), data::axes::alpha());
  const auto ref_inactive_global = data::spin_channel_indices(
      ref_orbitals->inactive_indices(), data::axes::alpha());

  if (data::spin_channel_indices(ref_orbitals->inactive_indices(),
                                 data::axes::beta()) != ref_inactive_global)
    throw std::invalid_argument(
        "SchriefferWolffPT2 requires all singly occupied orbitals in an "
        "open-shell reference to belong to the active space; inactive "
        "orbitals must be doubly occupied.");

  const std::unordered_set<std::size_t> ref_inactive_set(
      ref_inactive_global.begin(), ref_inactive_global.end());

  // The reference 1-RDM over W, spin-traced. For ROHF that gives each singly
  // occupied active orbital occupation one; the spin-free H0 then preserves
  // S^2 and the active solve selects the desired spin sector.
  const Eigen::MatrixXd density = window_density(
      W_global, ref_active_global, ref_inactive_set,
      reference_active_density(*reference, ref_active_global.size()));

  const std::unordered_set<std::size_t> kept_set(kept_global.begin(),
                                                 kept_global.end());
  const std::unordered_set<std::size_t> window_set(W_global.begin(),
                                                   W_global.end());

  // The window Hamiltonian already folded its own inactive orbitals into
  // `e_core`, so those must be exactly the reference core orbitals lying
  // outside W; otherwise the core energy and the reference density disagree.
  std::vector<std::size_t> expected_win_inactive;
  for (std::size_t g : ref_inactive_global)
    if (!window_set.count(g)) expected_win_inactive.push_back(g);
  if (win_inactive_global != expected_win_inactive)
    throw std::invalid_argument(
        "SchriefferWolffPT2: the window Hamiltonian's inactive orbitals must "
        "be exactly the reference core orbitals outside the window.");

  // Every fractionally occupied reference orbital lies inside the window (the
  // containment check above), and the rest are exactly doubly occupied or
  // empty, so the window must carry an integer number of electrons.
  const double window_electrons = density.diagonal().sum();
  const double window_electrons_integer = std::round(window_electrons);
  constexpr double electron_count_tolerance = 1e-6;
  if (std::abs(window_electrons - window_electrons_integer) >
      electron_count_tolerance)
    throw std::invalid_argument(
        "SchriefferWolffPT2: the reference density over the window does not "
        "carry an integer number of electrons (" +
        std::to_string(window_electrons) + ").");

  const Eigen::VectorXd diagonal = density.diagonal();
  const std::vector<double> occupation(diagonal.data(),
                                       diagonal.data() + diagonal.size());
  const sw::WindowPartition window_roles = sw::partition_window(
      occupation, W_global, kept_set,
      static_cast<int>(window_electrons_integer),
      _settings->get<double>("max_folded_occupation_deviation"));
  const std::vector<int>& active_spatial = window_roles.active_spatial;
  const std::vector<int>& inactive_spatial = window_roles.inactive_spatial;
  const std::vector<int>& virtual_spatial = window_roles.virtual_spatial;
  log_fold_rounding(window_roles);

  // Built from the full 1-RDM, so a correlated reference's off-diagonal
  // density survives into the denominators.
  Eigen::MatrixXd fock =
      sw::generalized_fock_matrix(h1a, g_aaaa, density, norb);
  // The fold reads occupations in the basis it runs in, so the 1-RDM follows
  // the rotation instead of being dropped here.
  Eigen::MatrixXd folded_density = density;
  Eigen::MatrixXd semicanonical_transform =
      Eigen::MatrixXd::Identity(norb, norb);
  bool semicanonical_applied = false;
  if (_settings->get<bool>("semicanonicalize")) {
    // Below this the block is already diagonal to working precision and the
    // rotation would be a no-op.
    constexpr double semicanonical_tolerance = 1e-10;
    semicanonical_transform = sw::semicanonical_rotation(
        fock, {inactive_spatial, active_spatial, virtual_spatial},
        semicanonical_tolerance);
    semicanonical_applied = !semicanonical_transform.isIdentity(0.0);
    if (semicanonical_applied) {
      h1a = sw::rotate_one_body(h1a, semicanonical_transform);
      g_aaaa = sw::rotate_two_body(g_aaaa, semicanonical_transform, norb);
      fock = sw::rotate_one_body(fock, semicanonical_transform);
      folded_density = sw::rotate_one_body(density, semicanonical_transform);
    }
  }

  Eigen::VectorXd eps(2 * norb);
  for (int i = 0; i < norb; ++i) eps(2 * i) = eps(2 * i + 1) = fock(i, i);

  const auto blk = sw::build_two_body_blocked(g_aaaa, norb);
  const auto f = sw::spin_orbital_one_body(h1a, h1a, norb);
  const auto part = sw::make_partition(norb, active_spatial, inactive_spatial,
                                       virtual_spatial);

  // Terms above two-body cannot be emitted, and discarding them throws away
  // their reference contractions, which dominate. `folded_density` lets the
  // kernel fold them onto the reference instead; it is read in the
  // semicanonical basis, so it had to follow the rotation above. Below three
  // active electrons a three-body operator has no matrix elements at all, so
  // folding would only add reference-specific terms at several times the cost.
  const bool fold_pays = window_roles.active_electrons > 2;
  const bool fold_requested = _settings->get<bool>("fold_above_two_body");
  const bool fold = fold_requested && fold_pays;
  // A determinant reference leaves the kept-space density idempotent, D^2 = 2D;
  // a correlated one loses the two-body cumulant to the fold.
  const Eigen::MatrixXd kept_density =
      sub_block(folded_density, active_spatial);
  const bool determinant_reference =
      (kept_density * kept_density - 2.0 * kept_density).cwiseAbs().maxCoeff() <
      1e-6;

  sw::RegularizerOptions reg;
  reg.sigma2 = _settings->get<double>("regularizer_sigma2");
  // The bound constraint admits NaN, which would then read as "not positive"
  // and silently select the bare path instead of the requested regularizer.
  if (!std::isfinite(reg.sigma2))
    throw std::invalid_argument(
        "SchriefferWolffPT2: regularizer_sigma2 must be finite.");
  const bool regularized = reg.sigma2 > 0.0;

  const auto down =
      sw::downfold_blocked(f, blk, eps, part, reg, e_core, {},
                           fold ? folded_density : Eigen::MatrixXd());

  QDK_LOGGER().info(
      "SW-PT2 downfold complete: regularization={}, minimum denominator={:.3g} "
      "Eh, maximum raw amplitude={:.3g}, semicanonical rotation applied={}, "
      "above two-body={}",
      regularized ? "sigma^2" : "none", down.min_denominator,
      down.max_amplitude, semicanonical_applied,
      !fold ? (fold_requested ? "not folded (kept space holds at most two "
                                "electrons)"
                              : "discarded (fold_above_two_body is off)")
      : determinant_reference ? "folded onto a determinant reference"
                              : "folded onto a correlated reference");
  warn_on_intruders(down, regularized);

  auto active = sw::to_spatial_chemist(down, part);
  if (semicanonical_applied) {
    const Eigen::MatrixXd active_rotation =
        sub_block(semicanonical_transform, active_spatial).transpose();
    active.one_body = sw::rotate_one_body(active.one_body, active_rotation);
    active.two_body =
        sw::rotate_two_body(active.two_body, active_rotation,
                            static_cast<int>(active_spatial.size()));
  }

  // The orbitals the window Hamiltonian folded into its core energy are
  // inactive in the emitted operator too; they lie outside W, so they cannot
  // collide with the folded window orbitals above.
  std::vector<std::size_t> emit_inactive;
  for (int i : inactive_spatial) emit_inactive.push_back(W_global[i]);
  emit_inactive.insert(emit_inactive.end(), win_inactive_global.begin(),
                       win_inactive_global.end());
  std::sort(emit_inactive.begin(), emit_inactive.end());

  const Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);
  return std::make_shared<data::Hamiltonian>(
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          active.one_body, active.two_body,
          relabeled_orbitals(*ref_orbitals, p_indices, emit_inactive),
          active.core_energy, empty_fock));
}

}  // namespace qdk::chemistry::algorithms::microsoft

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "swpt2.hpp"

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

#include "swpt2_kernel.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace sw = qdk::chemistry::algorithms::microsoft::swpt2;

namespace {

/// The reference's per-spin 1-RDMs over its own active space, from its active
/// 1-RDMs when available and from the reference determinant otherwise. A
/// restricted reference splits its spin-traced density evenly, so H0 stays spin
/// free and preserves S^2; for ROHF the active-space solve then selects the
/// spin sector.
std::array<Eigen::MatrixXd, 2> reference_active_density(
    const data::Wavefunction& reference, std::size_t n_ref_active,
    bool unrestricted) {
  const auto size = static_cast<Eigen::Index>(n_ref_active);
  const auto real_rdm = [&](const auto& variant) {
    const auto* rdm = std::get_if<Eigen::MatrixXd>(&variant);
    if (!rdm)
      throw std::invalid_argument(
          "SW-PT2 requires a real-valued active 1-RDM.");
    if (rdm->rows() != size || rdm->cols() != size)
      throw std::invalid_argument(
          "SW-PT2: active 1-RDM dimensions do not match the "
          "reference active-space size.");
    if (!rdm->allFinite())
      throw std::invalid_argument(
          "SW-PT2: the active 1-RDM contains non-finite values.");
    return Eigen::MatrixXd(0.5 * (*rdm + rdm->transpose()));
  };
  if (unrestricted && reference.has_one_rdm_spin_dependent()) {
    const auto [alpha, beta] = reference.get_active_one_rdm_spin_dependent();
    return {real_rdm(alpha), real_rdm(beta)};
  }
  if (!unrestricted && reference.has_one_rdm_spin_traced()) {
    const Eigen::MatrixXd half =
        0.5 * real_rdm(reference.get_active_one_rdm_spin_traced());
    return {half, half};
  }

  try {
    const auto [occ_a, occ_b] = reference.get_active_orbital_occupations();
    if (occ_a.size() != size || occ_b.size() != size)
      throw std::invalid_argument(
          "SW-PT2: active occupation dimensions do not match "
          "the reference active-space size.");
    if (!occ_a.allFinite() || !occ_b.allFinite())
      throw std::invalid_argument(
          "SW-PT2: the active orbital occupations contain "
          "non-finite values.");
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
              "SW-PT2: the reference reports fractional active "
              "orbital occupations but exposes no active 1-RDM, so they "
              "cannot be assigned to orbitals. Enable one-RDM calculation "
              "on the reference.");
    if (unrestricted)
      return {Eigen::MatrixXd(occ_a.asDiagonal()),
              Eigen::MatrixXd(occ_b.asDiagonal())};
    const Eigen::MatrixXd half = (0.5 * (occ_a + occ_b)).asDiagonal();
    return {half, half};
  } catch (const std::runtime_error&) {
    throw std::runtime_error(
        "SW-PT2: reference wavefunction exposes neither an "
        "active 1-RDM nor active orbital occupations");
  }
}

/// A per-spin reference 1-RDM over the whole window W: the reference's active
/// 1-RDM block, 1 on the diagonal of its inactive orbitals, and 0 elsewhere.
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
  std::vector<int> window_rows, reference_rows;
  for (int i = 0; i < norb; ++i) {
    const auto it = ref_active_pos.find(window_global[i]);
    if (it != ref_active_pos.end()) {
      window_rows.push_back(i);
      reference_rows.push_back(it->second);
    } else if (ref_inactive_set.count(window_global[i])) {
      density(i, i) = 1.0;
    }
  }
  density(window_rows, window_rows) =
      ref_active_density(reference_rows, reference_rows);

  constexpr double occupation_bound_tolerance = 1e-6;
  for (int i = 0; i < norb; ++i)
    if (density(i, i) < -occupation_bound_tolerance ||
        density(i, i) > 1.0 + occupation_bound_tolerance)
      throw std::invalid_argument("SW-PT2: window orbital " +
                                  std::to_string(window_global[i]) +
                                  " has unphysical reference occupation " +
                                  std::to_string(density(i, i)) +
                                  " per spin; expected a value in [0, 1].");
  return density;
}

/// The reference orbitals relabeled with `p_indices` as active and
/// `emit_inactive` as inactive orbitals.
std::shared_ptr<data::Orbitals> relabeled_orbitals(
    const data::Orbitals& ref_orbitals,
    const std::shared_ptr<const data::SymmetryBlockedIndexSet>& p_indices,
    const std::vector<std::size_t>& emit_inactive) {
  std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>> inactive;
  for (const auto& label : p_indices->labels())
    inactive[label] =
        std::vector<std::uint32_t>(emit_inactive.begin(), emit_inactive.end());

  std::optional<Eigen::MatrixXd> overlap;
  if (ref_orbitals.has_overlap_matrix())
    overlap = ref_orbitals.get_overlap_matrix();
  std::shared_ptr<data::BasisSet> basis;
  if (ref_orbitals.has_basis_set()) basis = ref_orbitals.get_basis_set();
  return std::make_shared<data::Orbitals>(
      ref_orbitals.coefficients(),
      ref_orbitals.has_energies() ? ref_orbitals.energies() : nullptr, overlap,
      basis, p_indices,
      std::make_shared<const data::SymmetryBlockedIndexSet>(
          p_indices->symmetries(), p_indices->extents(), std::move(inactive)));
}

/// Window roles chosen by the folding policy, plus the rounding diagnostics a
/// caller needs to judge how much the fold perturbed the reference density.
struct WindowPartition {
  /// Spatial positions into the window.
  std::vector<int> active_spatial, inactive_spatial, virtual_spatial;
  int active_electrons = 0;       ///< window electrons the rounded fold leaves
  double worst_deviation = 0.0;   ///< largest folded |occupation - rounded|
  double worst_occupation = 0.0;  ///< occupation attaining `worst_deviation`
  std::size_t worst_orbital = 0;  ///< global index attaining it
  /// Rounded folded electron count minus the reference occupation over the
  /// folded orbitals; roundings of opposite sign cancel here.
  double folded_charge_error = 0.0;
};

/// Keep `kept_global` active and fold every other window orbital by rounding
/// its reference occupation (`occupation`, by window position) to 2 or 0.
WindowPartition partition_window(
    const std::vector<double>& occupation,
    const std::vector<std::size_t>& window_global,
    const std::unordered_set<std::size_t>& kept_global, int window_electrons,
    double max_folded_occupation_deviation) {
  // The setting's bound constraint admits NaN.
  if (!(max_folded_occupation_deviation >= 0.0 &&
        max_folded_occupation_deviation < 1.0))
    throw std::invalid_argument(
        "SW-PT2: max_folded_occupation_deviation must lie in "
        "[0, 1); a half-occupied orbital cannot be folded unambiguously.");

  WindowPartition out;
  const int norb = static_cast<int>(occupation.size());
  for (int i = 0; i < norb; ++i) {
    if (kept_global.count(window_global[i])) {
      out.active_spatial.push_back(i);
      continue;
    }
    const double to_occupied = std::abs(occupation[i] - 2.0);
    const double to_empty = std::abs(occupation[i]);
    const double deviation = std::min(to_occupied, to_empty);
    if (deviation > max_folded_occupation_deviation)
      throw std::invalid_argument(
          "SW-PT2: folded external orbital " +
          std::to_string(window_global[i]) + " has reference occupation " +
          std::to_string(occupation[i]) +
          ", which deviates from an integer occupation by more than "
          "max_folded_occupation_deviation. Keep strongly correlated or "
          "open-shell orbitals in the active space P (p_indices), or raise "
          "the setting to accept the rounding.");
    if (deviation > out.worst_deviation) {
      out.worst_deviation = deviation;
      out.worst_occupation = occupation[i];
      out.worst_orbital = window_global[i];
    }
    (to_occupied <= to_empty ? out.inactive_spatial : out.virtual_spatial)
        .push_back(i);
  }

  // The active space takes the electrons the rounded fold leaves, which need
  // not equal the reference occupation of P.
  const int folded_electrons =
      2 * static_cast<int>(out.inactive_spatial.size());
  out.active_electrons = window_electrons - folded_electrons;
  double folded_occupation = 0.0;
  for (int i : out.inactive_spatial) folded_occupation += occupation[i];
  for (int i : out.virtual_spatial) folded_occupation += occupation[i];
  out.folded_charge_error = folded_electrons - folded_occupation;
  return out;
}

void log_fold_rounding(const WindowPartition& roles) {
  QDK_LOGGER().info(
      "SW-PT2 partition: active={}, folded inactive={}, folded virtual={}; "
      "active electrons={}, largest folded occupation deviation={:.3g}, "
      "folded core electron excess={:.3g}",
      roles.active_spatial.size(), roles.inactive_spatial.size(),
      roles.virtual_spatial.size(), roles.active_electrons,
      roles.worst_deviation, roles.folded_charge_error);

  constexpr double charge_error_warning = 0.01;
  // The default occupation_threshold of OccupationActiveSpaceSelector.
  constexpr double deviation_warning = 0.1;
  if (std::abs(roles.folded_charge_error) > charge_error_warning ||
      roles.worst_deviation > deviation_warning)
    QDK_LOGGER().warn(
        "SW-PT2: folded orbital {} has reference occupation {:.4f} and the "
        "folded core is off by {:.3g} electrons; consider adding it to "
        "p_indices.",
        roles.worst_orbital, roles.worst_occupation, roles.folded_charge_error);
}

void warn_on_intruders(const sw::DownfoldResult& down, bool regularized) {
  if (!regularized && down.min_denominator < sw::bare_denominator_floor)
    QDK_LOGGER().warn(
        "SW-PT2: channels with denominators below {:.3g} Eh were omitted; "
        "consider a larger P or setting regularizer_sigma2.",
        sw::bare_denominator_floor);
  // A heuristic flag on the undamped amplitude, not a convergence bound.
  constexpr double intruder_warn_amplitude = 1.0;
  if (down.max_amplitude > intruder_warn_amplitude)
    QDK_LOGGER().warn(
        "SW-PT2: large amplitude {:.3g} (smallest denominator {:.3g} Eh); "
        "consider a larger P{}.",
        down.max_amplitude, down.min_denominator,
        regularized ? "" : " or setting regularizer_sigma2");
}

}  // namespace

std::shared_ptr<data::Hamiltonian> SchriefferWolffPT2Constructor::_run_impl(
    std::shared_ptr<data::Wavefunction> reference,
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices) const {
  QDK_LOG_TRACE_ENTERING();
  // Checks a shared MO basis and spin restriction, and P and the reference
  // active space against the window.
  _validate_inputs(reference, hamiltonian, p_indices);
  if (!hamiltonian->is_hermitian())
    throw std::invalid_argument("SW-PT2 requires a Hermitian Hamiltonian.");
  auto win_orbitals = hamiltonian->get_orbitals();
  auto ref_orbitals = reference->get_orbitals();
  const bool unrestricted =
      hamiltonian->is_unrestricted() || ref_orbitals->is_unrestricted();

  // The spatial orbitals an index set selects; both spins must agree.
  const auto spatial = [](const auto& indices, const std::string& what) {
    auto alpha = data::spin_channel_indices(indices, data::axes::alpha());
    if (data::spin_channel_indices(indices, data::axes::beta()) != alpha)
      throw std::invalid_argument("SW-PT2: " + what +
                                  " must select the same orbitals in both "
                                  "spin channels.");
    return alpha;
  };
  const auto kept_global = spatial(p_indices, "p_indices");
  if (kept_global.empty())
    throw std::invalid_argument(
        "SW-PT2 requires a non-empty p_indices argument: the "
        "kept space P as a SymmetryBlockedIndexSet of global orbital indices "
        "into the window Hamiltonian's active space W = P u Q.");

  const auto [h_alpha, h_beta] = hamiltonian->get_one_body_integrals();
  std::array<Eigen::MatrixXd, 2> one_body{h_alpha, h_beta};
  // The kernel reads the Hamiltonian's tensors in place; a semicanonical
  // rotation below replaces them with rotated copies. A restricted Hamiltonian
  // keeps one tensor for all three spin blocks.
  const auto [g_aaaa, g_aabb, g_bbbb] = hamiltonian->get_two_body_integrals();
  std::array<const Eigen::VectorXd*, 3> two_body{&g_aaaa, &g_aaaa, &g_aaaa};
  if (unrestricted) two_body = {&g_aaaa, &g_aabb, &g_bbbb};
  std::array<Eigen::VectorXd, 3> rotated_two_body;
  const auto two_body_blocks = [&]() -> sw::TwoBodyBlocks {
    return {std::cref(*two_body[0]), std::cref(*two_body[1]),
            std::cref(*two_body[2])};
  };
  const double e_core = hamiltonian->get_core_energy();
  const int norb = static_cast<int>(h_alpha.rows());

  const auto W_global = spatial(win_orbitals->active_indices(),
                                "the window Hamiltonian's active index set");
  if (W_global.size() != static_cast<std::size_t>(norb))
    throw std::invalid_argument(
        "SW-PT2: the window Hamiltonian's active index set must match the "
        "rank of its integrals.");
  const auto win_inactive_global =
      spatial(win_orbitals->inactive_indices(),
              "the window Hamiltonian's inactive index set");
  const auto ref_inactive_global =
      spatial(ref_orbitals->inactive_indices(),
              "the reference inactive space (singly occupied orbitals must "
              "be active)");
  const auto ref_active_global =
      spatial(ref_orbitals->active_indices(), "the reference active space");

  const std::unordered_set<std::size_t> ref_inactive_set(
      ref_inactive_global.begin(), ref_inactive_global.end());

  const auto ref_density = reference_active_density(
      *reference, ref_active_global.size(), unrestricted);
  std::array<Eigen::MatrixXd, 2> density;
  for (int spin = 0; spin < 2; ++spin)
    density[spin] = window_density(W_global, ref_active_global,
                                   ref_inactive_set, ref_density[spin]);

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
        "SW-PT2: the window Hamiltonian's inactive orbitals must "
        "be exactly the reference core orbitals outside the window.");

  // Fractionally occupied reference orbitals all lie in the window and the
  // rest are doubly occupied or empty, so the window holds an integer number
  // of electrons.
  const Eigen::VectorXd diagonal =
      density[0].diagonal() + density[1].diagonal();
  const double window_electrons = diagonal.sum();
  const double window_electrons_integer = std::round(window_electrons);
  constexpr double electron_count_tolerance = 1e-6;
  if (std::abs(window_electrons - window_electrons_integer) >
      electron_count_tolerance)
    throw std::invalid_argument(
        "SW-PT2: the reference density over the window does not "
        "carry an integer number of electrons (" +
        std::to_string(window_electrons) + ").");

  const std::vector<double> occupation(diagonal.data(),
                                       diagonal.data() + diagonal.size());
  const WindowPartition window_roles = partition_window(
      occupation, W_global, kept_set,
      static_cast<int>(window_electrons_integer),
      _settings->get<double>("max_folded_occupation_deviation"));
  const std::vector<int>& active_spatial = window_roles.active_spatial;
  const std::vector<int>& inactive_spatial = window_roles.inactive_spatial;
  const std::vector<int>& virtual_spatial = window_roles.virtual_spatial;
  // The folded core is closed-shell, so the active space keeps the
  // reference's spin imbalance; a restricted reference's spin-averaged density
  // cannot show it, so the counts come from the reference itself.
  const auto [ref_alpha, ref_beta] = reference->get_active_num_electrons();
  const auto core_in_window = static_cast<long long>(
      std::count_if(ref_inactive_global.begin(), ref_inactive_global.end(),
                    [&](std::size_t g) { return window_set.count(g) != 0; }));
  const auto folded_pairs = static_cast<long long>(inactive_spatial.size());
  const long long active_alpha =
      static_cast<long long>(ref_alpha) + core_in_window - folded_pairs;
  const long long active_beta =
      static_cast<long long>(ref_beta) + core_in_window - folded_pairs;
  const auto kept_count = static_cast<long long>(active_spatial.size());
  if (std::min(active_alpha, active_beta) < 0 ||
      std::max(active_alpha, active_beta) > kept_count)
    throw std::invalid_argument(
        "SW-PT2: rounding the folded occupations leaves " +
        std::to_string(active_alpha) + " alpha and " +
        std::to_string(active_beta) + " beta electrons in " +
        std::to_string(kept_count) +
        " active orbitals. Keep the partially occupied orbitals in the "
        "active space P (p_indices).");
  log_fold_rounding(window_roles);

  sw::RegularizerOptions reg;
  reg.sigma2 = _settings->get<double>("regularizer_sigma2");
  // The bound constraint admits NaN, which would then read as "not positive"
  // and silently select the bare path instead of the requested regularizer.
  if (!std::isfinite(reg.sigma2))
    throw std::invalid_argument("SW-PT2: regularizer_sigma2 must be finite.");
  const bool regularized = reg.sigma2 > 0.0;

  // Built from the full 1-RDMs, so a correlated reference's off-diagonal
  // density survives into the denominators.
  auto fock = sw::generalized_fock_matrices(one_body, two_body_blocks(),
                                            density, norb, !unrestricted);
  // The density follows the rotation, since the fold reads it in the
  // semicanonical basis.
  std::array<Eigen::MatrixXd, 2> rotation{
      Eigen::MatrixXd::Identity(norb, norb),
      Eigen::MatrixXd::Identity(norb, norb)};
  bool semicanonical_applied = false;
  if (_settings->get<bool>("semicanonicalize")) {
    // Leaving a block unrotated neglects its off-diagonal Fock elements e,
    // which shifts H_eff by about e * V^2 * max|dr/dD| for the inverse
    // denominator r(D): at most sigma with the regularizer, unbounded without
    // it. A converged SCF leaves e near its orbital-gradient threshold.
    const double semicanonical_tolerance =
        regularized ? 1e-6 / std::max(1.0, reg.sigma2) : 1e-10;
    for (int spin = 0; spin < (unrestricted ? 2 : 1); ++spin)
      rotation[spin] = sw::semicanonical_rotation(
          fock[spin], {inactive_spatial, active_spatial, virtual_spatial},
          semicanonical_tolerance);
    if (!unrestricted) rotation[1] = rotation[0];
    semicanonical_applied =
        !rotation[0].isIdentity(0.0) || !rotation[1].isIdentity(0.0);
    if (semicanonical_applied) {
      for (int spin = 0; spin < 2; ++spin) {
        one_body[spin] = sw::rotate_one_body(one_body[spin], rotation[spin]);
        fock[spin] = sw::rotate_one_body(fock[spin], rotation[spin]);
        density[spin] = sw::rotate_one_body(density[spin], rotation[spin]);
      }
      rotated_two_body[0] =
          sw::rotate_two_body(*two_body[0], rotation[0], norb);
      if (unrestricted) {
        rotated_two_body[1] =
            sw::rotate_two_body(*two_body[1], rotation[0], rotation[1], norb);
        rotated_two_body[2] =
            sw::rotate_two_body(*two_body[2], rotation[1], norb);
        two_body = {&rotated_two_body[0], &rotated_two_body[1],
                    &rotated_two_body[2]};
      } else {
        two_body.fill(&rotated_two_body[0]);
      }
    }
  }

  // A three-body operator has no matrix elements below three active
  // electrons, so there the fold would only add reference-specific terms.
  const bool fold_pays = window_roles.active_electrons > 2;
  const bool fold_requested = _settings->get<bool>("fold_above_two_body");
  const bool fold = fold_requested && fold_pays;
  const char* fold_description =
      fold             ? "Gaussian 1-RDM fold"
      : fold_requested ? "not folded (kept space holds at most two electrons)"
                       : "discarded (fold_above_two_body is off)";

  auto down = sw::downfold(one_body, two_body_blocks(), fock, active_spatial,
                           inactive_spatial, virtual_spatial, reg, e_core,
                           !unrestricted,
                           fold ? density : std::array<Eigen::MatrixXd, 2>{});

  QDK_LOGGER().info(
      "SW-PT2 downfold complete: regularization={}, minimum denominator={:.3g} "
      "Eh, maximum raw amplitude={:.3g}, semicanonical rotation applied={}, "
      "above two-body={}",
      regularized ? "sigma^2" : "none", down.min_denominator,
      down.max_amplitude, semicanonical_applied, fold_description);
  warn_on_intruders(down, regularized);

  if (semicanonical_applied) {
    const int active_count = static_cast<int>(active_spatial.size());
    std::array<Eigen::MatrixXd, 2> active_rotation;
    for (int spin = 0; spin < 2; ++spin) {
      active_rotation[spin] =
          rotation[spin](active_spatial, active_spatial).transpose();
      down.one_body[spin] =
          sw::rotate_one_body(down.one_body[spin], active_rotation[spin]);
      if (unrestricted)
        down.same_spin_two_body[spin] = sw::rotate_two_body(
            down.same_spin_two_body[spin], active_rotation[spin], active_count);
    }
    down.two_body = sw::rotate_two_body(down.two_body, active_rotation[0],
                                        active_rotation[1], active_count);
  }

  // The orbitals the window Hamiltonian folded into its core energy are
  // inactive in the emitted operator too; they lie outside W, so they cannot
  // collide with the folded window orbitals above.
  std::vector<std::size_t> emit_inactive;
  for (int i : inactive_spatial) emit_inactive.push_back(W_global[i]);
  emit_inactive.insert(emit_inactive.end(), win_inactive_global.begin(),
                       win_inactive_global.end());
  std::sort(emit_inactive.begin(), emit_inactive.end());

  auto orbitals = relabeled_orbitals(*ref_orbitals, p_indices, emit_inactive);
  const Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);
  if (!unrestricted)
    return std::make_shared<data::Hamiltonian>(
        std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
            down.one_body[0], down.two_body, orbitals, down.scalar,
            empty_fock));
  return std::make_shared<data::Hamiltonian>(
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          down.one_body[0], down.one_body[1], down.same_spin_two_body[0],
          down.two_body, down.same_spin_two_body[1], orbitals, down.scalar,
          empty_fock, empty_fock));
}

}  // namespace qdk::chemistry::algorithms::microsoft

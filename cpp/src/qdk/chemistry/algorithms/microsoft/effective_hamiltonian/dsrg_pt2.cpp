// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "dsrg_pt2.hpp"

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
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../../data/two_body_symmetry.hpp"
#include "dsrg_pt2_kernel.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace dsrg = qdk::chemistry::algorithms::microsoft::dsrg_pt2;

namespace {

/// Per-spin 1-RDMs over the reference active space for a single-determinant
/// reference. Restricted orbitals require a closed-shell determinant;
/// unrestricted orbitals accept any spin-1/2 single determinant.
std::array<Eigen::MatrixXd, 2> single_determinant_active_density(
    const data::Wavefunction& reference, std::size_t n_ref_active,
    bool unrestricted) {
  if (!reference.has_container_type<data::StateVectorContainer>())
    throw std::invalid_argument(
        "DSRG-PT2 requires a single-determinant StateVectorContainer "
        "reference.");

  const auto& state = reference.get_container<data::StateVectorContainer>();
  if (state.get_type() != data::WavefunctionType::SelfDual)
    throw std::invalid_argument("DSRG-PT2 requires a self-dual reference.");

  const auto* coeffs = std::get_if<Eigen::VectorXd>(&state.get_coefficients());
  if (!coeffs)
    throw std::invalid_argument("DSRG-PT2 requires real CI coefficients.");

  constexpr double normalization_tolerance = 1e-12;
  const double norm_squared = coeffs->squaredNorm();
  if (!std::isfinite(norm_squared) ||
      std::abs(norm_squared - 1.0) > normalization_tolerance)
    throw std::invalid_argument("DSRG-PT2 requires a normalized reference.");

  std::size_t det_index = 0;
  std::size_t nonzero_count = 0;
  for (Eigen::Index k = 0; k < coeffs->size(); ++k) {
    if ((*coeffs)[k] != 0.0) {
      det_index = static_cast<std::size_t>(k);
      ++nonzero_count;
    }
  }
  if (nonzero_count != 1)
    throw std::invalid_argument(
        "DSRG-PT2 requires a single-determinant reference.");

  const auto& det = state.get_active_determinants()[det_index];
  if (det.bits_per_mode() != 2)
    throw std::invalid_argument("DSRG-PT2 requires spin-1/2 determinants.");

  const auto size = static_cast<Eigen::Index>(n_ref_active);
  Eigen::VectorXd occ_a(size), occ_b(size);
  for (std::size_t r = 0; r < n_ref_active; ++r) {
    const std::uint8_t mode = det.get_mode_state(r);
    occ_a(static_cast<Eigen::Index>(r)) = static_cast<double>(mode & 1u);
    occ_b(static_cast<Eigen::Index>(r)) = static_cast<double>(mode >> 1u);
  }
  if (!unrestricted && occ_a != occ_b)
    throw std::invalid_argument(
        "DSRG-PT2 requires a closed-shell determinant on restricted orbitals.");

  constexpr double rdm_tolerance = 1e-10;
  const auto check_rdm = [&](const auto& variant,
                             const Eigen::VectorXd& expected_diag) {
    const auto* rdm = std::get_if<Eigen::MatrixXd>(&variant);
    if (!rdm || rdm->rows() != size || rdm->cols() != size || !rdm->allFinite())
      throw std::invalid_argument(
          "DSRG-PT2 requires a finite real active 1-RDM matching the "
          "reference determinant.");
    const Eigen::MatrixXd expected = expected_diag.asDiagonal();
    if (size > 0 && (*rdm - expected).cwiseAbs().maxCoeff() > rdm_tolerance)
      throw std::invalid_argument(
          "DSRG-PT2 requires a single-determinant reference.");
  };
  if (reference.has_one_rdm_spin_dependent()) {
    const auto [rdm_a, rdm_b] = reference.get_active_one_rdm_spin_dependent();
    check_rdm(rdm_a, occ_a);
    check_rdm(rdm_b, occ_b);
  }
  if (reference.has_one_rdm_spin_traced()) {
    check_rdm(reference.get_active_one_rdm_spin_traced(), occ_a + occ_b);
  }

  return {Eigen::MatrixXd(occ_a.asDiagonal()),
          Eigen::MatrixXd(occ_b.asDiagonal())};
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
  return density;
}

/// The reference orbitals relabeled with `p_indices` as active and
/// `emit_inactive` as inactive orbitals. Orbital energies are dropped: they
/// describe the input Fock operator, not the effective Hamiltonian.
std::shared_ptr<data::Orbitals> relabeled_orbitals(
    const data::Orbitals& ref_orbitals,
    const std::shared_ptr<const data::SymmetryBlockedIndexSet>& p_indices,
    const std::vector<std::size_t>& emit_inactive) {
  std::vector<std::uint32_t> inactive_indices;
  inactive_indices.reserve(emit_inactive.size());
  for (const std::size_t index : emit_inactive)
    inactive_indices.push_back(static_cast<std::uint32_t>(index));
  std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>> inactive;
  for (const auto& label : p_indices->labels())
    inactive[label] = inactive_indices;
  const auto inactive_set =
      std::make_shared<const data::SymmetryBlockedIndexSet>(
          p_indices->symmetries(), p_indices->extents(), std::move(inactive));
  if (dynamic_cast<const data::ModelOrbitals*>(&ref_orbitals))
    return std::make_shared<data::ModelOrbitals>(p_indices, inactive_set);

  std::optional<Eigen::MatrixXd> overlap;
  if (ref_orbitals.has_overlap_matrix())
    overlap = ref_orbitals.get_overlap_matrix();
  std::shared_ptr<data::BasisSet> basis;
  if (ref_orbitals.has_basis_set()) basis = ref_orbitals.get_basis_set();
  return std::make_shared<data::Orbitals>(ref_orbitals.coefficients(), nullptr,
                                          overlap, basis, p_indices,
                                          inactive_set);
}

struct WindowPartition {
  std::vector<int> active_spatial, inactive_spatial, virtual_spatial;
};

WindowPartition partition_window(
    const std::array<Eigen::MatrixXd, 2>& density,
    const std::vector<std::size_t>& window_global,
    const std::unordered_set<std::size_t>& kept_global) {
  WindowPartition out;
  const int norb = static_cast<int>(window_global.size());
  for (int i = 0; i < norb; ++i) {
    if (kept_global.count(window_global[i])) {
      out.active_spatial.push_back(i);
      continue;
    }
    const double na = density[0](i, i), nb = density[1](i, i);
    if (na == 1.0 && nb == 1.0)
      out.inactive_spatial.push_back(i);
    else if (na == 0.0 && nb == 0.0)
      out.virtual_spatial.push_back(i);
    else
      throw std::invalid_argument("DSRG-PT2: singly occupied orbital " +
                                  std::to_string(window_global[i]) +
                                  " must remain in p_indices.");
  }
  return out;
}

}  // namespace

std::shared_ptr<data::Hamiltonian> DsrgPT2Constructor::_run_impl(
    std::shared_ptr<data::Wavefunction> reference,
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices) const {
  QDK_LOG_TRACE_ENTERING();
  _validate_inputs(reference, hamiltonian, p_indices);
  if (!hamiltonian->is_hermitian())
    throw std::invalid_argument("DSRG-PT2 requires a Hermitian Hamiltonian.");

  const double flow = _settings->get<double>("flow_parameter");
  if (!std::isfinite(flow) || flow < 0.0)
    throw std::invalid_argument(
        "DSRG-PT2: flow_parameter must be finite and nonnegative.");

  auto win_orbitals = hamiltonian->get_orbitals();
  auto ref_orbitals = reference->get_orbitals();
  const bool unrestricted = hamiltonian->is_unrestricted();

  // The spatial orbitals an index set selects; both spins must agree.
  const auto spatial = [](const auto& indices, const std::string& what) {
    auto alpha = data::spin_channel_indices(indices, data::axes::alpha());
    if (data::spin_channel_indices(indices, data::axes::beta()) != alpha)
      throw std::invalid_argument("DSRG-PT2: " + what +
                                  " must select the same orbitals in both "
                                  "spin channels.");
    return alpha;
  };
  const auto kept_global = spatial(p_indices, "p_indices");
  if (kept_global.empty())
    throw std::invalid_argument("DSRG-PT2 requires a non-empty p_indices.");

  const auto [h_alpha, h_beta] = hamiltonian->get_one_body_integrals();
  if (!h_alpha.allFinite() || !h_beta.allFinite())
    throw std::invalid_argument(
        "DSRG-PT2: the one-body integrals contain a non-finite value.");
  std::array<Eigen::MatrixXd, 2> one_body{h_alpha, h_beta};
  const int norb = static_cast<int>(h_alpha.rows());
  // The aaaa, aabb and bbbb blocks are read as stored; `rotated` holds their
  // semicanonical rotation.
  const auto [g_aaaa, g_aabb, g_bbbb] = hamiltonian->get_two_body_integrals();
  std::array<const Eigen::VectorXd*, 3> two_body{&g_aaaa, &g_aabb, &g_bbbb};
  if (!unrestricted) two_body.fill(&g_aaaa);
  for (int block = 0; block < (unrestricted ? 3 : 1); ++block)
    data::detail::require_finite_two_body(
        two_body[block]->data(),
        static_cast<std::size_t>(two_body[block]->size()), "DSRG-PT2");
  std::array<Eigen::VectorXd, 3> rotated;
  const auto two_body_blocks = [&]() -> dsrg::TwoBodyBlocks {
    return {std::cref(*two_body[0]), std::cref(*two_body[1]),
            std::cref(*two_body[2])};
  };
  const double e_core = hamiltonian->get_core_energy();

  const auto W_global = spatial(win_orbitals->active_indices(),
                                "the window Hamiltonian's active index set");
  const auto win_inactive_global =
      spatial(win_orbitals->inactive_indices(),
              "the window Hamiltonian's inactive index set");
  const auto ref_inactive_global =
      spatial(ref_orbitals->inactive_indices(), "the reference inactive space");
  const auto ref_active_global =
      spatial(ref_orbitals->active_indices(), "the reference active space");

  const std::unordered_set<std::size_t> ref_inactive_set(
      ref_inactive_global.begin(), ref_inactive_global.end());

  const auto ref_density = single_determinant_active_density(
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
        "DSRG-PT2: window inactive orbitals must be the reference core "
        "outside the window.");

  const WindowPartition window_roles =
      partition_window(density, W_global, kept_set);
  const std::vector<int>& active_spatial = window_roles.active_spatial;
  const std::vector<int>& inactive_spatial = window_roles.inactive_spatial;
  const std::vector<int>& virtual_spatial = window_roles.virtual_spatial;

  std::array<Eigen::VectorXd, 2> orbital_energies{Eigen::VectorXd::Zero(norb),
                                                  Eigen::VectorXd::Zero(norb)};
  std::array<dsrg::SemicanonicalRotation, 2> rotation;
  if (flow > 0.0) {
    auto fock = dsrg::generalized_fock_matrices(one_body, two_body_blocks(),
                                                density, norb, !unrestricted);
    constexpr double semicanonical_tolerance = 1e-12;
    for (int spin = 0; spin < (unrestricted ? 2 : 1); ++spin)
      rotation[spin] = dsrg::semicanonical_rotation(
          fock[spin], {inactive_spatial, active_spatial, virtual_spatial},
          semicanonical_tolerance);
    if (!unrestricted) rotation[1] = rotation[0];
    for (int spin = 0; spin < 2; ++spin) {
      if (rotation[spin].rotated) {
        one_body[spin] =
            dsrg::rotate_one_body(one_body[spin], rotation[spin].matrix);
        fock[spin] = dsrg::rotate_one_body(fock[spin], rotation[spin].matrix);
        density[spin] =
            dsrg::rotate_one_body(density[spin], rotation[spin].matrix);
      }
      orbital_energies[spin] = fock[spin].diagonal();
    }
    if (rotation[0].rotated || rotation[1].rotated) {
      const std::array<std::vector<int>, 3> roles{
          inactive_spatial, active_spatial, virtual_spatial};
      // aaaa, aabb and bbbb rotate their (pq) and (rs) pairs by the alpha and
      // beta rotations as (alpha, alpha), (alpha, beta) and (beta, beta).
      for (int block = 0; block < (unrestricted ? 3 : 1); ++block) {
        rotated[block] = dsrg::rotate_two_body(
            *two_body[block], rotation[block / 2].matrix,
            rotation[(block + 1) / 2].matrix, roles, norb);
        two_body[block] = &rotated[block];
      }
      if (!unrestricted) two_body.fill(&rotated[0]);
    }
  }

  auto down = dsrg::downfold(one_body, two_body_blocks(), orbital_energies,
                             active_spatial, inactive_spatial, virtual_spatial,
                             flow, e_core, !unrestricted, density);

  QDK_LOGGER().info(
      "DSRG-PT2 downfold complete: flow_parameter={}, active={}, "
      "folded inactive={}, folded virtual={}, minimum gap={:.3g} Eh, "
      "maximum amplitude={:.3g}",
      flow, active_spatial.size(), inactive_spatial.size(),
      virtual_spatial.size(), down.min_denominator, down.max_amplitude);
  constexpr double intruder_warn_amplitude = 1.0;
  if (down.max_amplitude > intruder_warn_amplitude)
    QDK_LOGGER().warn(
        "DSRG-PT2: large amplitude {:.3g} (smallest gap {:.3g} Eh); consider "
        "a larger P or smaller flow_parameter.",
        down.max_amplitude, down.min_denominator);

  if (rotation[0].rotated || rotation[1].rotated) {
    const int active_count = static_cast<int>(active_spatial.size());
    std::array<Eigen::MatrixXd, 2> active_rotation;
    for (int spin = 0; spin < 2; ++spin) {
      active_rotation[spin] =
          rotation[spin].matrix(active_spatial, active_spatial).transpose();
      down.one_body[spin] =
          dsrg::rotate_one_body(down.one_body[spin], active_rotation[spin]);
      if (unrestricted)
        down.same_spin_two_body[spin] = dsrg::rotate_two_body(
            down.same_spin_two_body[spin], active_rotation[spin], active_count);
    }
    down.two_body = dsrg::rotate_two_body(down.two_body, active_rotation[0],
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

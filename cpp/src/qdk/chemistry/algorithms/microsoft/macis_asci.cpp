// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "macis_asci.hpp"

#include <limits>
#include <macis/asci/determinant_search.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::algorithms::microsoft {

MacisAsciSettings::MacisAsciSettings() {
  // Use MACIS library defaults directly
  macis::ASCISettings macis_defaults;

  // ASCI determinant control parameters
  set_default<int64_t>(
      "ntdets_max", macis_defaults.ntdets_max,
      "Maximum number of trial determinants in the variational space",
      data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});
  set_default<int64_t>(
      "ntdets_min", macis_defaults.ntdets_min,
      "Minimum number of trial determinants required",
      data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});
  set_default<int64_t>(
      "ncdets_max", macis_defaults.ncdets_max,
      "Maximum number of core determinants",
      data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});

  // Tolerance parameters
  set_default<double>("search_matel_tol", macis_defaults.h_el_tol,
                      "Hamiltonian matrix element magnitude threshold",
                      data::BoundConstraint<double>{0.0, 1.0});
  set_default<double>("rv_prune_tol", macis_defaults.rv_prune_tol,
                      "Ratio value pruning threshold",
                      data::BoundConstraint<double>{0.0, 1.0});
  set_default<double>("pt2_tol", macis_defaults.pt2_tol,
                      "PT2 correction tolerance",
                      data::BoundConstraint<double>{0.0, 1.0});
  set_default<double>("refine_energy_tol", macis_defaults.refine_energy_tol,
                      "Energy convergence tolerance for refinement",
                      data::BoundConstraint<double>{0.0, 1.0});

  // PT2 correction parameters
  set_default<int64_t>(
      "pt2_reserve_count", macis_defaults.pt2_reserve_count,
      "Reserve count for PT2 calculations",
      data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
  set_default<bool>("pt2_prune", macis_defaults.pt2_prune);
  set_default<bool>("pt2_precompute_eps", macis_defaults.pt2_precompute_eps);
  set_default<bool>("pt2_precompute_idx", macis_defaults.pt2_precompute_idx);
  set_default<bool>("pt2_print_progress", macis_defaults.pt2_print_progress);
  set_default<int64_t>(
      "pt2_bigcon_thresh", macis_defaults.pt2_bigcon_thresh,
      "Threshold for using bigcon PT2 algorithm",
      data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});

  // Algorithm control parameters
  set_default<int64_t>(
      "pair_size_max", macis_defaults.pair_size_max,
      "Maximum number of ASCI contribution pairs to store in memory",
      data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});
  set_default<int64_t>(
      "nxtval_bcount_thresh", macis_defaults.nxtval_bcount_thresh,
      "Threshold for next value batch count",
      data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});
  set_default<int64_t>(
      "nxtval_bcount_inc", macis_defaults.nxtval_bcount_inc,
      "Increment for next value batch count",
      data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});
  set_default<bool>("just_singles", macis_defaults.just_singles);
  set_default<double>(
      "grow_factor", macis_defaults.grow_factor,
      "Factor by which to grow the variational space",
      data::BoundConstraint<double>{1.0e0, std::numeric_limits<double>::max()});
  set_default<double>(
      "min_grow_factor", macis_defaults.min_grow_factor,
      "Minimum allowed growth factor",
      data::BoundConstraint<double>{1.0, std::numeric_limits<double>::max()});
  set_default<double>("growth_backoff_rate", macis_defaults.growth_backoff_rate,
                      "Rate to reduce grow_factor on failure",
                      data::BoundConstraint<double>{0.0, 1.0});
  set_default<double>(
      "growth_recovery_rate", macis_defaults.growth_recovery_rate,
      "Rate to restore grow_factor on success",
      data::BoundConstraint<double>{1.0, std::numeric_limits<double>::max()});
  set_default<int64_t>(
      "max_refine_iter", macis_defaults.max_refine_iter,
      "Maximum number of refinement iterations",
      data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
  set_default<bool>("grow_with_rot", macis_defaults.grow_with_rot);
  set_default<int64_t>(
      "rot_size_start", macis_defaults.rot_size_start,
      "Starting size for rotations",
      data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});

  // Constraint parameters
  set_default<int64_t>(
      "constraint_level", macis_defaults.constraint_level,
      "Constraint level for excitation generation",
      data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
  set_default<int64_t>(
      "pt2_max_constraint_level", macis_defaults.pt2_max_constraint_level,
      "Maximum constraint level for PT2 calculations",
      data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
  set_default<int64_t>(
      "pt2_min_constraint_level", macis_defaults.pt2_min_constraint_level,
      "Minimum constraint level for PT2 calculations",
      data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});
  set_default<int64_t>(
      "pt2_constraint_refine_force", macis_defaults.pt2_constraint_refine_force,
      "Force constraint refinement for PT2 calculations",
      data::BoundConstraint<int64_t>{0, std::numeric_limits<int64_t>::max()});

  // Core selection strategy parameter
  set_default<std::string>(
      "core_selection_strategy", "percentage",
      "Core determinant selection strategy: 'percentage' uses cumulative "
      "weight threshold, 'fixed' uses a fixed number of determinants",
      data::ListConstraint<std::string>{
          {std::vector<std::string>{"percentage", "fixed"}}});
  set_default<double>("core_selection_threshold",
                      macis_defaults.core_selection_threshold,
                      "Cumulative weight threshold for core selection",
                      data::BoundConstraint<double>{
                          std::numeric_limits<double>::epsilon(), 1.0});

  // Warm-start and tolerance controls
  set_default<bool>("warm_start_davidson", macis_defaults.warm_start_davidson,
                    "Warm-start Davidson from previous eigenvector");
  set_default<double>(
      "min_warm_start_overlap", macis_defaults.min_warm_start_overlap,
      "Minimum projected vector norm for warm-start Davidson. "
      "Measures how much of the previous eigenvector's weight is "
      "retained in the new determinant set (0=never, 1=always reject)",
      data::BoundConstraint<double>{0.0, 1.0});
  set_default<double>("min_patch_overlap", macis_defaults.min_patch_overlap,
                      "Minimum determinant overlap for incremental H build. "
                      "The cache persists across iterations; a full rebuild is "
                      "triggered only when overlap drops below this threshold",
                      data::BoundConstraint<double>{0.0, 1.0});
  set_default<double>(
      "grow_ci_residual_tolerance", macis_defaults.grow_ci_residual_tolerance,
      "CI residual tolerance during grow phase (0 = use refine tolerance)");
  set_default<double>(
      "taper_grow_factor", macis_defaults.taper_grow_factor,
      "Growth factor for final expansion near ntdets_max (0 = disabled)");

  // Hamiltonian build algorithm selection
  set_default<std::string>("hamiltonian_build_algorithm",
                           macis_defaults.hamiltonian_build_algorithm,
                           "Algorithm for diagonal Hamiltonian construction: "
                           "'' or 'sorted_double_loop' (default), "
                           "'residue_arrays', 'dynamic_bit_masking'");
  set_default<size_t>("dynamic_bit_masking_num_masks", 0,
                      "Number of bit masks for dynamic_bit_masking generator "
                      "(0 = use generator default)");
}

std::pair<double, std::shared_ptr<data::Wavefunction>> MacisAsci::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian, unsigned int nalpha,
    unsigned int nbeta) const {
  QDK_LOG_TRACE_ENTERING();

  QDK_LOGGER().debug("MacisAsci::_run_impl starting: nalpha={}, nbeta={}",
                     nalpha, nbeta);

  const auto& orbitals = hamiltonian->get_orbitals();
  if (hamiltonian->is_unrestricted()) {
    throw std::runtime_error(
        "MacisAsci does not support unrestricted orbitals. "
        "Only restricted orbitals are supported.");
  }
  const auto active_ai = orbitals->active_indices();
  const auto active_indices =
      data::spin_channel_indices(active_ai, data::axes::alpha());
  const auto active_indices_beta =
      data::spin_channel_indices(active_ai, data::axes::beta());
  // check that alpha and beta active space indices are the same
  if (active_indices != active_indices_beta) {
    throw std::runtime_error(
        "MacisAsci only supports identical alpha and beta active "
        "space indices.");
  }

  QDK_LOGGER().debug("MacisAsci::_run_impl dispatching for {} active orbitals.",
                     active_indices.size());

  auto result = dispatch_by_norb<asci_helper>(
      active_indices.size(), *hamiltonian, *_settings, nalpha, nbeta);
  return std::make_pair(result.first, std::make_shared<data::Wavefunction>(
                                          std::move(result.second)));
}

}  // namespace qdk::chemistry::algorithms::microsoft

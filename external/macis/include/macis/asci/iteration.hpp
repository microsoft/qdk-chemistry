/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <blas.hh>
#include <macis/asci/determinant_search.hpp>
#include <macis/mcscf/mcscf.hpp>
#include <macis/solvers/selected_ci_diag.hpp>
#include <numeric>

namespace macis {

/**
 * @brief Perform a single ASCI (Adaptive Sampling Configuration Interaction)
 * iteration
 *
 * This function executes one complete iteration of the ASCI algorithm,
 * including determinant sorting, space expansion through search, and
 * Hamiltonian rediagonalization. It represents the core computational cycle of
 * ASCI calculations.
 *
 * @tparam N Size of the wavefunction bitset representation
 * @tparam index_t Integer type for indexing operations
 *
 * @param[in] asci_settings ASCI algorithm parameters
 * @param[in] mcscf_settings MCSCF parameters for CI diagonalization
 * @param[in] ndets_max Maximum number of determinants for expanded space
 * @param[in] E0 Reference energy from previous iteration
 * @param[in] wfn Current wavefunction determinants
 * @param[in] X Current CI coefficients corresponding to wavefunction
 * @param[in,out] ham_gen Hamiltonian generator containing integrals and methods
 * @param[in] norb Number of molecular orbitals
 * @param[in] comm MPI communicator for parallel execution (if MPI enabled)
 *
 * @return Tuple containing:
 *   - New ground state energy
 *   - Expanded and rediagonalized wavefunction determinants
 *   - New CI coefficients
 *
 * @see asci_search, selected_ci_diag, reorder_ci_on_coeff, reorder_ci_on_alpha
 */
template <size_t N, typename index_t>
auto asci_iter(ASCISettings asci_settings, MCSCFSettings mcscf_settings,
               size_t ndets_max, double E0, std::vector<wfn_t<N>> wfn,
               std::vector<double> X, HamiltonianGenerator<wfn_t<N>>& ham_gen,
               size_t norb,
               CachedHamiltonianState<wfn_t<N>, index_t>* h_cache =
                   nullptr MACIS_MPI_CODE(, MPI_Comm comm = MPI_COMM_WORLD)) {
  // Sort wfn on coefficient weights
  if (wfn.size() > 1) reorder_ci_on_coeff(wfn, X);

  auto logger = spdlog::get("asci_search");

  size_t nkeep = 0;
  switch (asci_settings.core_selection_strategy) {
    case CoreSelectionStrategy::Fixed:
      // Use fixed number of determinants
      nkeep = std::min(asci_settings.ncdets_max, wfn.size());
      if (logger) {
        logger->trace("  * Core selection: nkeep={}", nkeep);
      }
      break;
    case CoreSelectionStrategy::Percentage: {
      // Validate core_selection_threshold only when using Percentage strategy
      if (asci_settings.core_selection_threshold <
              std::numeric_limits<double>::epsilon() ||
          asci_settings.core_selection_threshold > 1.0) {
        throw std::invalid_argument(
            "core_selection_threshold must be in [epsilon, 1.0], got " +
            std::to_string(asci_settings.core_selection_threshold));
      }
      // Use percentage-based selection - keep determinants until cumulative
      // weight reaches threshold (not capped by ncdets_max).
      // Note: If the threshold is never reached (e.g., very small
      // coefficients), all determinants will be kept (nkeep == wfn.size()).
      double core_weight = 0.0;
      for (size_t i = 0; i < wfn.size(); ++i) {
        core_weight += X[i] * X[i];
        nkeep++;
        if (core_weight >= asci_settings.core_selection_threshold) {
          break;
        }
      }
      if (logger) {
        logger->trace("  * Core selection: nkeep={}, weight={:.6f}", nkeep,
                      core_weight);
      }
      break;
    }
    default:
      throw std::runtime_error("Unknown CoreSelectionStrategy");
  }

  // Sort kept dets on alpha string
  if (wfn.size() > 1)
    reorder_ci_on_alpha(wfn.begin(), wfn.begin() + nkeep, X.data());

  // Save old (det → coeff) mapping before the search replaces wfn.
  // Only the kept core dets (first nkeep) plus their reordered X matter;
  // the search will return a superset.  We store *all* old dets so that
  // during refine (where most dets are kept) the warm-start is effective.
  using wfn_traits = wavefunction_traits<wfn_t<N>>;
  using wfn_comp = typename wfn_traits::spin_comparator;

  // Capture old wfn for warm-start before asci_search takes ownership
  std::vector<wfn_t<N>> old_wfn;
  std::vector<double> old_X;
  if (asci_settings.warm_start_davidson) {
    old_wfn = wfn;  // copy — wfn is about to be overwritten
    old_X = X;
  }

  // Perform the ASCI search
  wfn = asci_search(asci_settings, ndets_max, wfn.begin(), wfn.begin() + nkeep,
                    E0, X, norb, ham_gen.T(), ham_gen.G_red(), ham_gen.V_red(),
                    ham_gen.V(), ham_gen MACIS_MPI_CODE(, comm));

  std::sort(wfn.begin(), wfn.end(), wfn_comp{});

  // Rediagonalize — optionally warm-start Davidson from previous coefficients.
  // Only warm-start when the determinant overlap is high; during aggressive
  // growth the old eigenvector can point toward the wrong eigenstate in the
  // much larger space, causing Davidson to stall.
  std::vector<double> X_local;
  if (asci_settings.warm_start_davidson && !old_wfn.empty()) {
    // new wfn is already sorted by wfn_comp.  Iterate unsorted old dets
    // and binary-search each into the new wfn.  This avoids sorting the
    // old wfn (O(D log D) on large bitsets) and the associated permutation
    // temporaries.
    X_local.resize(wfn.size(), 0.0);
    size_t n_matched = 0;
    const size_t old_size = old_wfn.size();
    for (size_t i = 0; i < old_size; ++i) {
      auto it =
          std::lower_bound(wfn.begin(), wfn.end(), old_wfn[i], wfn_comp{});
      if (it != wfn.end() && *it == old_wfn[i]) {
        size_t new_idx = static_cast<size_t>(std::distance(wfn.begin(), it));
        X_local[new_idx] = old_X[i];
        ++n_matched;
      }
    }
    old_wfn.clear();
    old_X.clear();

    // Use the projected vector norm as the warm-start quality metric.
    // ||P·ψ_old||₂ measures how much of the old eigenvector's weight lives
    // in the new determinant space.  Near 1.0 → excellent guess; near 0.0
    // → most weight was on determinants that were dropped.
    double norm = blas::nrm2(X_local.size(), X_local.data(), 1);
    if (logger) {
      logger->info("  WARM_START: matched={}/{}, projected_norm={:.4f}",
                   n_matched, wfn.size(), norm);
    }
    if (norm < asci_settings.min_warm_start_overlap) {
      X_local.clear();  // triggers identity guess in selected_ci_diag
      if (logger)
        logger->info(
            "  WARM_START: norm {:.4f} < {:.4f} threshold, using diagonal "
            "guess",
            norm, asci_settings.min_warm_start_overlap);
    } else {
      blas::scal(X_local.size(), 1.0 / norm, X_local.data(), 1);
    }
  }

  double E = selected_ci_diag<index_t>(
      wfn.begin(), wfn.end(), ham_gen, mcscf_settings.ci_matel_tol,
      mcscf_settings.ci_max_subspace, mcscf_settings.ci_res_tol, X_local,
      h_cache, asci_settings.min_patch_overlap MACIS_MPI_CODE(, comm));

#ifdef MACIS_ENABLE_MPI
  auto world_size = comm_size(comm);
  auto world_rank = comm_rank(comm);
  if (world_size > 1) {
    // Broadcast X_local to X
    const size_t wfn_size = wfn.size();
    const size_t local_count = wfn_size / world_size;
    X.resize(wfn.size());

    MPI_Allgather(X_local.data(), local_count, MPI_DOUBLE, X.data(),
                  local_count, MPI_DOUBLE, comm);
    if (wfn_size % world_size) {
      const size_t nrem = wfn_size % world_size;
      auto* X_rem = X.data() + world_size * local_count;
      if (world_rank == world_size - 1) {
        const auto* X_loc_rem = X_local.data() + local_count;
        std::copy_n(X_loc_rem, nrem, X_rem);
      }
      MPI_Bcast(X_rem, nrem, MPI_DOUBLE, world_size - 1, comm);
    }
  } else {
    // Avoid copy
    X = std::move(X_local);
  }
#else
  X = std::move(X_local);  // Serial
#endif /* MACIS_ENABLE_MPI */

  return std::make_tuple(E, wfn, X);
}

}  // namespace macis

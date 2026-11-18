/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm>
#include <cmath>
#include <limits>
#include <macis/asci/iteration.hpp>

namespace macis {

/**
 * @brief Perform ASCI (Adaptive Sampling Configuration Interaction)
 * wavefunction refinement phase
 *
 * Reselects the determinant space and rediagonalizes until the energy changes
 * by less than refine_energy_tol. The energy can instead oscillate, typically
 * as the selection cycles between two determinant spaces: the energy steps
 * flip sign, each cancelling the previous one to within refine_energy_tol.
 * After two such steps in a row, refinement rediagonalizes the union of the
 * last two spaces, continues from it at the target size (which can lead out of
 * the cycle) and grants four more iterations, enough to confirm an oscillation
 * that re-forms. A later oscillation ends refinement if the energy has not
 * dropped by more than refine_energy_tol since the last union, or once the
 * union steps have been granted at least max_refine_iter iterations in total.
 * Once the energy has oscillated, the result is the lowest-energy iterate of
 * a search, never the union; a refinement that never oscillates returns its
 * last iterate.
 *
 * @tparam N Size of the wavefunction bitset representation
 * @tparam index_t Integer type for indexing operations (default: int32_t)
 *
 * @param[in] asci_settings ASCI algorithm parameters including refinement
 * tolerance
 * @param[in] mcscf_settings MCSCF parameters for CI diagonalization
 * @param[in] E0 Initial reference energy from growth phase
 * @param[in] wfn Final wavefunction determinants from growth phase
 * @param[in] X Initial CI coefficients corresponding to wavefunction
 * @param[in] ham_gen Hamiltonian generator containing integrals and methods
 * @param[in] norb Number of molecular orbitals
 * @param[in] comm MPI communicator for parallel execution (MPI builds only)
 * @param[in,out] execution Optional borrowed numerical executor; null selects
 * CPU
 *
 * @return Tuple containing:
 *   - Final refined energy
 *   - Refined determinants
 *   - Refined CI coefficients
 *
 * @throws std::runtime_error if the energy neither converges nor ends in a
 * cycle within the iteration budget, the space shrinks below ntdets_min, or
 * an MPI build needs a union step
 *
 * @see asci_iter, asci_grow
 */
template <size_t N, typename index_t = int32_t>
auto asci_refine(ASCISettings asci_settings, MCSCFSettings mcscf_settings,
                 double E0, std::vector<wfn_t<N>> wfn, std::vector<double> X,
                 HamiltonianGenerator<wfn_t<N>>& ham_gen,
                 size_t norb MACIS_MPI_CODE(, MPI_Comm comm),
                 ASCIExecutionContext<N>* execution = nullptr) {
  auto logger = spdlog::get("asci_refine");
#ifdef MACIS_ENABLE_MPI
  auto world_rank = comm_rank(comm);
#else
  int world_rank = 0;
#endif /* MACIS_ENABLE_MPI */
  if (!logger) {
    logger = world_rank ? spdlog::null_logger_mt("asci_refine")
                        : spdlog::stdout_color_mt("asci_refine");
    logger->flush_on(logger->level());
  }

  logger->info("[ASCI Refine Settings]:");
  logger->info(
      "  NTDETS = {:6}, NCDETS = {:6}, MAX_REFINE_ITER = {:4}, REFINE_TOL = "
      "{:.2e}",
      wfn.size(), asci_settings.ncdets_max, asci_settings.max_refine_iter,
      asci_settings.refine_energy_tol);

  constexpr const char* fmt_string =
      "iter = {:4}, E0 = {:20.12e}, dE = {:14.6e}";

  logger->info(fmt_string, 0, E0, 0.0);

  // Refinement Loop
  size_t ndets = wfn.size();
  bool converged = false;
  double prev_E_delta = 0.0;
  int oscillation_count = 0;
  size_t max_iter = asci_settings.max_refine_iter;
  // Iterations granted per union step: a cycle that re-forms after it takes
  // up to four to confirm (the first step after the union cannot count)
  constexpr size_t union_extension = 4;
  // Union steps are taken while fewer than the original budget have been
  // granted, which bounds the total
  size_t total_extensions = 0;
  const size_t max_extensions = asci_settings.max_refine_iter;
  std::vector<wfn_t<N>> prev_wfn;  // saved for union stabilization
  bool cycled = false, stopped = false;
  // Lowest-energy iterate at the target size, the result once the selection
  // has cycled, and its energy when the last union was taken
  double E_best = std::numeric_limits<double>::infinity();
  double E_best_at_union = std::numeric_limits<double>::infinity();
  size_t iter_best = 0;
  std::vector<wfn_t<N>> wfn_best;
  std::vector<double> X_best;
  // Incremental H_build cache — carried across refine iterations
  CachedHamiltonianState<wfn_t<N>, index_t> h_cache;

  for (size_t iter = 0; iter < max_iter; ++iter) {
    double E;
    std::tie(E, wfn, X) = asci_iter<N, index_t>(
        asci_settings, mcscf_settings, ndets, E0, std::move(wfn), std::move(X),
        ham_gen, norb, &h_cache MACIS_MPI_CODE(, comm), execution);

    // Check if wavefunction size changed
    if (wfn.size() != ndets) {
      logger->warn(
          "Wavefunction size changed from {} to {} during refinement iteration "
          "{}",
          ndets, wfn.size(), iter + 1);

      // Update target size for next iteration
      ndets = wfn.size();

      // If wavefunction became too small, stop refinement
      if (wfn.size() < asci_settings.ntdets_min) {
        logger->error(
            "Wavefunction shrunk below ntdets_min ({}), stopping refinement",
            asci_settings.ntdets_min);
        break;
      }
    }

    const auto E_delta = E - E0;
    logger->info(fmt_string, iter + 1, E, E_delta);

    if (std::abs(E_delta) < asci_settings.refine_energy_tol) {
      E0 = E;
      converged = true;
      break;
    }

    if (E < E_best) {
      E_best = E;
      iter_best = iter + 1;
      wfn_best = wfn;
      X_best = X;
    }

    // Detect oscillation: the step flips sign and cancels the previous one,
    // so the energy returns to that of two iterations ago
    if (iter > 0 && prev_E_delta * E_delta < 0 &&
        std::abs(prev_E_delta + E_delta) < asci_settings.refine_energy_tol) {
      oscillation_count++;

      if (oscillation_count >= 2 && !prev_wfn.empty()) {
        cycled = true;
        // A union step that has not led to a lower energy will not either
        const bool improved =
            E_best < E_best_at_union - asci_settings.refine_energy_tol;
        if (!improved || total_extensions >= max_extensions) {
          logger->warn(
              "  Oscillation detected — the selection cycles {}; stopping at "
              "the lowest-energy iterate (iter = {}, E0 = {:20.12e})",
              improved ? "and the union steps have used their iterations"
                       : "without lowering the energy since the last union",
              iter_best, E_best);
          E0 = E;
          stopped = true;
          break;
        }
        E_best_at_union = E_best;

        // Rediagonalize in the union of the two oscillating determinant sets
        // and continue from it at the target size.
        using wfn_comp =
            typename wavefunction_traits<wfn_t<N>>::spin_comparator;

        std::vector<wfn_t<N>> union_wfn;
        union_wfn.reserve(wfn.size() + prev_wfn.size());
        std::set_union(prev_wfn.begin(), prev_wfn.end(), wfn.begin(), wfn.end(),
                       std::back_inserter(union_wfn), wfn_comp{});

        logger->info(
            "  Oscillation detected — stabilizing via union of last two "
            "det sets ({} + {} → {} unique dets), searching {} dets next",
            prev_wfn.size(), wfn.size(), union_wfn.size(), ndets);

        // Rediagonalize in the enlarged space — keep the cache since the
        // union is a superset of the cached dets (high overlap → patched
        // build). Warm-start from current X mapped onto the union ordering.
        std::vector<double> X_union(union_wfn.size(), 0.0);
        {
          // wfn is sorted and is a subset of union_wfn (also sorted).
          // Binary-search each wfn det into union_wfn to map coefficients.
          for (size_t i = 0; i < wfn.size(); ++i) {
            auto it = std::lower_bound(union_wfn.begin(), union_wfn.end(),
                                       wfn[i], wfn_comp{});
            if (it != union_wfn.end() && *it == wfn[i]) {
              X_union[static_cast<size_t>(
                  std::distance(union_wfn.begin(), it))] = X[i];
            }
          }
        }
#ifdef MACIS_ENABLE_MPI
        throw std::runtime_error(
            "asci_refine: incremental Hamiltonian caching is not supported "
            "in MPI builds. Union-stabilization cannot "
            "proceed with h_cache under MPI.");
#endif
        double E_union;
        if (execution) {
          ASCIExecutionGuard<N> execution_guard(execution);
          execution->start_diagonalization(union_wfn.begin(), union_wfn.end(),
                                           ham_gen, mcscf_settings);
          E_union = execution->finish_diagonalization(X_union);
        } else {
          E_union = selected_ci_diag<index_t>(
              union_wfn.begin(), union_wfn.end(), ham_gen,
              mcscf_settings.ci_matel_tol, mcscf_settings.ci_max_subspace,
              mcscf_settings.ci_res_tol, X_union, &h_cache,
              asci_settings.min_patch_overlap MACIS_MPI_CODE(, comm));
        }

        // In MPI builds, X_union is the local portion — gather it.
#ifdef MACIS_ENABLE_MPI
        {
          auto ws = comm_size(comm);
          if (ws > 1) {
            const size_t n = union_wfn.size();
            const size_t lc = n / ws;
            std::vector<double> X_full(n);
            MPI_Allgather(X_union.data(), lc, MPI_DOUBLE, X_full.data(), lc,
                          MPI_DOUBLE, comm);
            if (n % ws) {
              auto wr = comm_rank(comm);
              auto* rem = X_full.data() + ws * lc;
              if (wr == ws - 1) std::copy_n(X_union.data() + lc, n % ws, rem);
              MPI_Bcast(rem, n % ws, MPI_DOUBLE, ws - 1, comm);
            }
            X_union = std::move(X_full);
          }
        }
#endif

        logger->info("  Union diag: E = {:20.12e}", E_union);

        max_iter += union_extension;
        total_extensions += union_extension;
        logger->info(
            "  Extending max_iter by {} → {} (total extended: {}; unions "
            "start below {})",
            union_extension, max_iter, total_extensions, max_extensions);

        // The next search keeps ndets determinants, from a core within the
        // union (asci_iter caps it): searching at the union's size would only
        // move the cycle to it.
        wfn = std::move(union_wfn);
        X = std::move(X_union);
        E0 = E_union;
        prev_wfn.clear();
        oscillation_count = 0;
        prev_E_delta = 0.0;
        continue;  // re-enter refinement loop for real convergence check
      }
    } else {
      oscillation_count = 0;
    }
    prev_E_delta = E_delta;
    prev_wfn = wfn;

    E0 = E;
  }  // Refinement loop

  if (!converged && !stopped) {
    std::string msg = "ASCI Refine did not converge";
    if (total_extensions > 0) {
      msg += " (oscillation detected, " + std::to_string(total_extensions) +
             " extra iterations granted). "
             "Consider using percentage core_selection_strategy, increasing "
             "ncdets_max, or loosening refine_energy_tol.";
    }
    throw std::runtime_error(msg);
  }
  if (cycled && E_best <= E0) {
    E0 = E_best;
    wfn = std::move(wfn_best);
    X = std::move(X_best);
  }
  if (converged)
    logger->info("ASCI Refine Converged!");
  else
    logger->info("ASCI Refine stopped at an oscillation");

  return std::make_tuple(E0, wfn, X);
}

}  // namespace macis

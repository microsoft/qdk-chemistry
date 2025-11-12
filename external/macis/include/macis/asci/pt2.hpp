/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <deque>
#include <fstream>
#include <macis/asci/determinant_contributions.hpp>
#include <macis/asci/determinant_sort.hpp>
#include <sstream>

#ifdef MACIS_ENABLE_MPI
namespace macis {

template <size_t N>
double asci_pt2_constraint(ASCISettings asci_settings,
                           wavefunction_iterator_t<N> cdets_begin,
                           wavefunction_iterator_t<N> cdets_end,
                           const double E_ASCI, const std::vector<double>& C,
                           size_t norb, const double* T_pq, const double* G_red,
                           const double* V_red, const double* G_pqrs,
                           const double* V_pqrs,
                           HamiltonianGenerator<wfn_t<N>>& ham_gen,
                           MPI_Comm comm) {
  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;
  using wfn_traits = wavefunction_traits<wfn_t<N>>;
  using spin_wfn_type = spin_wfn_t<wfn_t<N>>;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;
  using wfn_comp = typename wfn_traits::spin_comparator;
  if (!std::is_sorted(cdets_begin, cdets_end, wfn_comp{}))
    throw std::runtime_error("PT2 Only Works with Sorted Wfns");

  auto world_rank = comm_rank(comm);
  auto world_size = comm_size(comm);
  auto logger = spdlog::get("asci_pt2");
  if (!logger)
    logger = world_rank ? spdlog::null_logger_mt("asci_pt2")
                        : spdlog::stdout_color_mt("asci_pt2");

  const size_t ncdets = std::distance(cdets_begin, cdets_end);
  logger->info("[ASCI PT2 Settings]");
  logger->info("  * NDETS                  = {}", ncdets);
  logger->info("  * PT2_TOL                = {}", asci_settings.pt2_tol);
  logger->info("  * PT2_RESERVE_COUNT      = {}",
               asci_settings.pt2_reserve_count);
  logger->info("  * PT2_CONSTRAINT_LVL_MAX = {}",
               asci_settings.pt2_max_constraint_level);
  logger->info("  * PT2_CONSTRAINT_LVL_MIN = {}",
               asci_settings.pt2_min_constraint_level);
  logger->info("  * PT2_CNSTRNT_RFNE_FORCE = {}",
               asci_settings.pt2_constraint_refine_force);
  logger->info("  * PT2_PRUNE              = {}", asci_settings.pt2_prune);
  logger->info("  * PT2_PRECOMP_EPS        = {}",
               asci_settings.pt2_precompute_eps);
  logger->info("  * PT2_BIGCON_THRESH      = {}",
               asci_settings.pt2_bigcon_thresh);
  logger->info("  * NXTVAL_BCOUNT_THRESH   = {}",
               asci_settings.nxtval_bcount_thresh);
  logger->info("  * NXTVAL_BCOUNT_INC      = {}",
               asci_settings.nxtval_bcount_inc);
  logger->info("");

  // For each unique alpha, create a list of beta string and store metadata
  struct beta_coeff_data {
    spin_wfn_type beta_string;
    std::vector<uint8_t> occ_beta;
    std::vector<uint8_t> vir_beta;
    std::vector<double> orb_ens_alpha;
    std::vector<double> orb_ens_beta;
    double coeff;
    double h_diag;

    size_t mem() const {
      return sizeof(spin_wfn_type) +
             (occ_beta.capacity() + vir_beta.capacity()) * sizeof(uint8_t) +
             (2 + orb_ens_alpha.capacity() + orb_ens_beta.capacity()) *
                 sizeof(double);
    }

    beta_coeff_data(double c, size_t norb,
                    const std::vector<uint32_t>& occ_alpha, wfn_t<N> w,
                    const HamiltonianGenerator<wfn_t<N>>& ham_gen, bool pce,
                    bool pci) {
      coeff = c;

      beta_string = wfn_traits::beta_string(w);

      // Compute diagonal matrix element
      h_diag = ham_gen.matrix_element(w, w);

      // Compute occ/vir for beta string
      std::vector<uint32_t> o_32, v_32;
      if (pce or pci) {
        spin_wfn_traits::state_to_occ_vir(norb, beta_string, o_32, v_32);
        occ_beta.resize(o_32.size());
        std::copy(o_32.begin(), o_32.end(), occ_beta.begin());
        vir_beta.resize(v_32.size());
        std::copy(v_32.begin(), v_32.end(), vir_beta.begin());
      }

      // Precompute orbital energies
      if (pce) {
        orb_ens_alpha = ham_gen.single_orbital_ens(norb, occ_alpha, o_32);
        orb_ens_beta = ham_gen.single_orbital_ens(norb, o_32, occ_alpha);
      }
    }
  };

  auto uniq_alpha = get_unique_alpha(cdets_begin, cdets_end);
  const size_t nuniq_alpha = uniq_alpha.size();
  logger->info("  * NUNIQ_ALPHA = {}", nuniq_alpha);
  std::vector<size_t> uniq_alpha_ioff(nuniq_alpha);
  std::transform_exclusive_scan(
      uniq_alpha.begin(), uniq_alpha.end(), uniq_alpha_ioff.begin(), 0ul,
      std::plus<size_t>(), [](const auto& p) { return p.second; });

  using unique_alpha_data = std::vector<beta_coeff_data>;
  std::vector<unique_alpha_data> uad(nuniq_alpha);
  for (auto i = 0, iw = 0; i < nuniq_alpha; ++i) {
    std::vector<uint32_t> occ_alpha, vir_alpha;
    spin_wfn_traits::state_to_occ_vir(norb, uniq_alpha[i].first, occ_alpha,
                                      vir_alpha);

    const auto nbeta = uniq_alpha[i].second;
    uad[i].reserve(nbeta);
    for (auto j = 0; j < nbeta; ++j, ++iw) {
      const auto& w = *(cdets_begin + iw);
      uad[i].emplace_back(C[iw], norb, occ_alpha, w, ham_gen,
                          asci_settings.pt2_precompute_eps,
                          asci_settings.pt2_precompute_idx);
    }
  }

  if (world_rank == 0) {
    constexpr double gib = 1024 * 1024 * 1024;
    logger->info("MEM REQ DETS = {:.2e}", ncdets * sizeof(wfn_t<N>) / gib);
    logger->info("MEM REQ C    = {:.2e}", ncdets * sizeof(double) / gib);
    size_t mem_alpha = 0;
    for (auto i = 0ul; i < nuniq_alpha; ++i) {
      mem_alpha += sizeof(spin_wfn_type);
      for (auto j = 0ul; j < uad[i].size(); ++j) {
        mem_alpha += uad[i][j].mem();
      }
    }
    logger->info("MEM REQ ALPH = {:.2e}", mem_alpha / gib);
    logger->info(
        "MEM REQ CONT = {:.2e}",
        asci_settings.pt2_reserve_count * sizeof(asci_contrib<wfn_t<N>>) / gib);
  }
  MPI_Barrier(comm);

  const auto n_occ_alpha = spin_wfn_traits::count(uniq_alpha[0].first);
  const auto n_vir_alpha = norb - n_occ_alpha;
  const auto n_sing_alpha = n_occ_alpha * n_vir_alpha;
  const auto n_doub_alpha = (n_sing_alpha * (n_sing_alpha - norb + 1)) / 4;

  const auto n_occ_beta = cdets_begin->count() - n_occ_alpha;
  const auto n_vir_beta = norb - n_occ_beta;
  const auto n_sing_beta = n_occ_beta * n_vir_beta;
  const auto n_doub_beta = (n_sing_beta * (n_sing_beta - norb + 1)) / 4;

  logger->info("  * NS = {} ND = {}", n_sing_alpha, n_doub_alpha);

  auto gen_c_st = clock_type::now();
  // auto constraints = dist_constraint_general<wfn_t<N>>(
  //     5, norb, n_sing_beta, n_doub_beta, uniq_alpha, comm);
  auto constraints = gen_constraints_general<wfn_t<N>>(
      asci_settings.pt2_max_constraint_level, norb, n_sing_beta, n_doub_beta,
      uniq_alpha, world_size * omp_get_max_threads(),
      asci_settings.pt2_min_constraint_level,
      asci_settings.pt2_constraint_refine_force);
  auto gen_c_en = clock_type::now();
  duration_type gen_c_dur = gen_c_en - gen_c_st;
  logger->info("  * GEN_DUR = {:.2e} ms", gen_c_dur.count());
  // if(!world_rank) {
  //   std::ofstream c_file("constraint_work.txt");
  //   std::stringstream ss;
  //   for(auto [c,s] : constraints) {
  //     ss << c.C() << " " << s << std::endl;
  //   }
  //   auto str = ss.str();
  //   c_file.write(str.c_str(), str.size());
  // }
  // if(!world_rank) {
  //   std::ofstream c_file("unique_alpha.txt");
  //   std::stringstream ss;
  //   for(size_t i = 0; i < nuniq_alpha; ++i) {
  //     ss << uniq_alpha[i].first << " " << uniq_alpha[i].second << std::endl;
  //   }
  //   auto str = ss.str();
  //   c_file.write(str.c_str(), str.size());
  // }

  double EPT2 = 0.0;
  size_t NPT2 = 0;

  const size_t ncon_total = constraints.size();
  const size_t ncon_big = asci_settings.pt2_bigcon_thresh;
  const size_t ncon_small = ncon_total - ncon_big;

  // Global atomic task-id counter
  global_atomic<size_t> nxtval_big(comm, 0);
  global_atomic<size_t> nxtval_small(comm, ncon_big);
  const double h_el_tol = asci_settings.pt2_tol;

  auto pt2_st = clock_type::now();
  // Assign each "big" constraint to an MPI rank, thread over contributions
  {
    size_t ic = 0;
    while (ic < ncon_big) {
      // Atomically get the next task ID and increment for other
      // MPI ranks
      ic = nxtval_big.fetch_add(1);
      if (ic >= ncon_big) continue;
      if (asci_settings.pt2_print_progress)
        printf("[pt2_big rank %4d] %10lu / %10lu\n", world_rank, ic,
               ncon_total);
      const auto& con = constraints[ic].first;

      asci_contrib_container<wfn_t<N>> asci_pairs_con;
#pragma omp parallel
      {
        asci_contrib_container<wfn_t<N>> asci_pairs;
#pragma omp for schedule(dynamic)
        for (size_t i_alpha = 0; i_alpha < nuniq_alpha; ++i_alpha) {
          const size_t old_pair_size = asci_pairs.size();
          const auto& alpha_det = uniq_alpha[i_alpha].first;
          const auto ncon_alpha = constraint_histogram(alpha_det, 1, 1, con);
          if (!ncon_alpha) continue;
          const auto occ_alpha = bits_to_indices(alpha_det);
          const bool alpha_satisfies_con = satisfies_constraint(alpha_det, con);

          const auto& bcd = uad[i_alpha];
          const size_t nbeta = bcd.size();
          for (size_t j_beta = 0; j_beta < nbeta; ++j_beta) {
            const size_t iw = uniq_alpha_ioff[i_alpha] + j_beta;
            const auto w = *(cdets_begin + iw);
            const auto c = C[iw];
            const auto& beta_det = bcd[j_beta].beta_string;
            const auto h_diag = bcd[j_beta].h_diag;

// TODO: These copies are slow
#if 0
          const auto& occ_beta_8 = bcd[j_beta].occ_beta;
          const auto& vir_beta_8 = bcd[j_beta].vir_beta;
          std::vector<uint32_t> occ_beta(occ_beta_8.size()), vir_beta(vir_beta_8.size());
          std::copy(occ_beta_8.begin(), occ_beta_8.end(), occ_beta.begin());
          std::copy(vir_beta_8.begin(), vir_beta_8.end(), vir_beta.begin());
#else
            std::vector<uint32_t> occ_beta, vir_beta;
            spin_wfn_traits::state_to_occ_vir(norb, beta_det, occ_beta,
                                              vir_beta);
#endif /* 0 */

            std::vector<double> orb_ens_alpha, orb_ens_beta;
            if (asci_settings.pt2_precompute_eps) {
              orb_ens_alpha = bcd[j_beta].orb_ens_alpha;
              orb_ens_beta = bcd[j_beta].orb_ens_beta;
            } else {
              orb_ens_alpha =
                  ham_gen.single_orbital_ens(norb, occ_alpha, occ_beta);
              orb_ens_beta =
                  ham_gen.single_orbital_ens(norb, occ_beta, occ_alpha);
            }

            // AA excitations
            generate_constraint_singles_contributions_ss(
                c, w, con, occ_alpha, occ_beta, orb_ens_alpha.data(), T_pq,
                norb, G_red, norb, V_red, norb, h_el_tol, h_diag, E_ASCI,
                ham_gen, asci_pairs);

            // AAAA excitations
            generate_constraint_doubles_contributions_ss(
                c, w, con, occ_alpha, occ_beta, orb_ens_alpha.data(), G_pqrs,
                norb, h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs);

            // AABB excitations
            generate_constraint_doubles_contributions_os(
                c, w, con, occ_alpha, occ_beta, vir_beta, orb_ens_alpha.data(),
                orb_ens_beta.data(), V_pqrs, norb, h_el_tol, h_diag, E_ASCI,
                ham_gen, asci_pairs);

            if (alpha_satisfies_con) {
              // BB excitations
              append_singles_asci_contributions<Spin::Beta>(
                  c, w, beta_det, occ_beta, vir_beta, occ_alpha,
                  orb_ens_beta.data(), T_pq, norb, G_red, norb, V_red, norb,
                  h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs);

              // BBBB excitations
              append_ss_doubles_asci_contributions<Spin::Beta>(
                  c, w, beta_det, alpha_det, occ_beta, vir_beta, occ_alpha,
                  orb_ens_beta.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI,
                  ham_gen, asci_pairs);

              // No excitation (push inf to remove from list)
              asci_pairs.push_back(
                  {w, std::numeric_limits<double>::infinity(), 1.0});
            }
          }
#if 0
        if(asci_settings.pt2_prune and asci_pairs.size() > asci_settings.pt2_reserve_count and asci_pairs.size() != old_pair_size) {
        // Cleanup
        auto uit = stable_sort_and_accumulate_asci_pairs(asci_pairs.begin(),
                                                  asci_pairs.end());
        asci_pairs.erase(uit, asci_pairs.end());
        //uit = std::stable_partition(asci_pairs.begin(), asci_pairs.end(), [&](const auto& p){ return std::abs(p.pt2()) > h_el_tol; });
        //asci_pairs.erase(uit, asci_pairs.end());
          printf("[pt2_prune rank %4d tid:%4d] IC = %lu / %lu IA = %lu / %lu SZ = %lu\n", world_rank,
                 omp_get_thread_num(), ic, ncon_total, i_alpha,
                 nuniq_alpha, asci_pairs.size());
        }
#endif /* 0 */

        }  // Unique Alpha Loop

        // S&A Thread local pairs
        sort_and_accumulate_asci_pairs(asci_pairs);

// Insert
#pragma omp critical
        {
          if (asci_pairs_con.size()) {
            asci_pairs_con.reserve(asci_pairs.size() + asci_pairs_con.size());
            asci_pairs_con.insert(asci_pairs_con.end(), asci_pairs.begin(),
                                  asci_pairs.end());
          } else {
            asci_pairs_con = std::move(asci_pairs);
          }
        }

      }  // OpenMP

      double EPT2_local = 0.0;
      size_t NPT2_local = 0;
      size_t pair_size = 0;
      // Local S&A for each quad + update EPT2
      {
        auto uit = sort_and_accumulate_asci_pairs(asci_pairs_con.begin(),
                                                  asci_pairs_con.end());
        pair_size = std::distance(asci_pairs_con.begin(), uit);
        for (auto it = asci_pairs_con.begin(); it != uit; ++it) {
          if (!std::isinf(it->c_times_matel)) {
            EPT2_local += it->pt2();
            NPT2_local++;
          }
        }
        asci_pairs_con.clear();
        if (asci_settings.pt2_print_progress)
          printf("[pt2_big rank %4d] CAPACITY %lu SZ %lu\n", world_rank,
                 asci_pairs_con.capacity(), pair_size);
      }

      EPT2 += EPT2_local;
      NPT2 += NPT2_local;
    }  // Constraint "loop"
  }  // "Big constraints"

  // Parallelize over both MPI + threads for "small" constraints
#pragma omp parallel reduction(+ : EPT2) reduction(+ : NPT2)
  {
    // Process ASCI pair contributions for each constraint
    asci_contrib_container<wfn_t<N>> asci_pairs;
    // asci_pairs.reserve(asci_settings.pt2_reserve_count);
    size_t ic = 0;
    while (ic < ncon_total) {
      // Atomically get the next task ID and increment for other
      // MPI ranks and threads
      size_t ntake = ic < asci_settings.nxtval_bcount_thresh
                         ? 1
                         : asci_settings.nxtval_bcount_inc;
      ic = nxtval_small.fetch_add(ntake);

      // Loop over assigned tasks
      const size_t c_end = std::min(ncon_total, ic + ntake);
      for (; ic < c_end; ++ic) {
        const auto& con = constraints[ic].first;
        if (asci_settings.pt2_print_progress)
          printf("[pt2_small rank %4d tid:%4d] %10lu / %10lu\n", world_rank,
                 omp_get_thread_num(), ic, ncon_total);

        for (size_t i_alpha = 0; i_alpha < nuniq_alpha; ++i_alpha) {
          const size_t old_pair_size = asci_pairs.size();
          const auto& alpha_det = uniq_alpha[i_alpha].first;
          const auto ncon_alpha = constraint_histogram(alpha_det, 1, 1, con);
          if (!ncon_alpha) continue;
          const auto occ_alpha = bits_to_indices(alpha_det);
          const bool alpha_satisfies_con = satisfies_constraint(alpha_det, con);

          const auto& bcd = uad[i_alpha];
          const size_t nbeta = bcd.size();
          for (size_t j_beta = 0; j_beta < nbeta; ++j_beta) {
            const size_t iw = uniq_alpha_ioff[i_alpha] + j_beta;
            const auto w = *(cdets_begin + iw);
            const auto c = C[iw];
            const auto& beta_det = bcd[j_beta].beta_string;
            const auto h_diag = bcd[j_beta].h_diag;

// TODO: These copies are slow
#if 0
            const auto& occ_beta_8 = bcd[j_beta].occ_beta;
            const auto& vir_beta_8 = bcd[j_beta].vir_beta;
            std::vector<uint32_t> occ_beta(occ_beta_8.size()), vir_beta(vir_beta_8.size());
            std::copy(occ_beta_8.begin(), occ_beta_8.end(), occ_beta.begin());
            std::copy(vir_beta_8.begin(), vir_beta_8.end(), vir_beta.begin());
#else
            std::vector<uint32_t> occ_beta, vir_beta;
            spin_wfn_traits::state_to_occ_vir(norb, beta_det, occ_beta,
                                              vir_beta);
#endif /* 0 */

            std::vector<double> orb_ens_alpha, orb_ens_beta;
            if (asci_settings.pt2_precompute_eps) {
              orb_ens_alpha = bcd[j_beta].orb_ens_alpha;
              orb_ens_beta = bcd[j_beta].orb_ens_beta;
            } else {
              orb_ens_alpha =
                  ham_gen.single_orbital_ens(norb, occ_alpha, occ_beta);
              orb_ens_beta =
                  ham_gen.single_orbital_ens(norb, occ_beta, occ_alpha);
            }

            // AA excitations
            generate_constraint_singles_contributions_ss(
                c, w, con, occ_alpha, occ_beta, orb_ens_alpha.data(), T_pq,
                norb, G_red, norb, V_red, norb, h_el_tol, h_diag, E_ASCI,
                ham_gen, asci_pairs);

            // AAAA excitations
            generate_constraint_doubles_contributions_ss(
                c, w, con, occ_alpha, occ_beta, orb_ens_alpha.data(), G_pqrs,
                norb, h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs);

            // AABB excitations
            generate_constraint_doubles_contributions_os(
                c, w, con, occ_alpha, occ_beta, vir_beta, orb_ens_alpha.data(),
                orb_ens_beta.data(), V_pqrs, norb, h_el_tol, h_diag, E_ASCI,
                ham_gen, asci_pairs);

            if (alpha_satisfies_con) {
              // BB excitations
              append_singles_asci_contributions<Spin::Beta>(
                  c, w, beta_det, occ_beta, vir_beta, occ_alpha,
                  orb_ens_beta.data(), T_pq, norb, G_red, norb, V_red, norb,
                  h_el_tol, h_diag, E_ASCI, ham_gen, asci_pairs);

              // BBBB excitations
              append_ss_doubles_asci_contributions<Spin::Beta>(
                  c, w, beta_det, alpha_det, occ_beta, vir_beta, occ_alpha,
                  orb_ens_beta.data(), G_pqrs, norb, h_el_tol, h_diag, E_ASCI,
                  ham_gen, asci_pairs);

              // No excitation (push inf to remove from list)
              asci_pairs.push_back(
                  {w, std::numeric_limits<double>::infinity(), 1.0});
            }
          }
          if (asci_settings.pt2_prune and
              asci_pairs.size() > asci_settings.pt2_reserve_count and
              asci_pairs.size() != old_pair_size) {
            // Cleanup
            auto uit = stable_sort_and_accumulate_asci_pairs(asci_pairs.begin(),
                                                             asci_pairs.end());
            asci_pairs.erase(uit, asci_pairs.end());
            // uit = std::stable_partition(asci_pairs.begin(), asci_pairs.end(),
            // [&](const auto& p){ return std::abs(p.pt2()) > h_el_tol; });
            // asci_pairs.erase(uit, asci_pairs.end());
            if (asci_settings.pt2_print_progress)
              printf(
                  "[pt2_prune rank %4d tid:%4d] IC = %lu / %lu IA = %lu / %lu "
                  "SZ = %lu\n",
                  world_rank, omp_get_thread_num(), ic, ncon_total, i_alpha,
                  nuniq_alpha, asci_pairs.size());

            if (asci_pairs.size() > asci_settings.pt2_reserve_count) {
              printf("* WARNING: PRUNED SIZE LARGER THAN RESERVE COUNT\n");
            }
          }

        }  // Unique Alpha Loop

        double EPT2_local = 0.0;
        size_t NPT2_local = 0;
        // Local S&A for each quad + update EPT2
        {
          auto uit = sort_and_accumulate_asci_pairs(asci_pairs.begin(),
                                                    asci_pairs.end());
          for (auto it = asci_pairs.begin(); it != uit; ++it) {
            if (!std::isinf(it->c_times_matel)) {
              EPT2_local += it->pt2();
              NPT2_local++;
            }
          }
          asci_pairs.clear();
          // Deallocate
          if (asci_pairs.capacity() > asci_settings.pt2_reserve_count)
            asci_contrib_container<wfn_t<N>>().swap(asci_pairs);
        }

        EPT2 += EPT2_local;
        NPT2 += NPT2_local;
      }  // Loc constraint loop
    }  // Constraint Loop
  }  // OpenMP
  auto pt2_en = clock_type::now();

  EPT2 = allreduce(EPT2, MPI_SUM, comm);

  double local_pt2_dur = duration_type(pt2_en - pt2_st).count();
  if (world_size > 1) {
    double total_dur = allreduce(local_pt2_dur, MPI_SUM, comm);
    double min_dur = allreduce(local_pt2_dur, MPI_MIN, comm);
    double max_dur = allreduce(local_pt2_dur, MPI_MAX, comm);
    logger->info("* PT2_DUR MIN = {:.2e}, MAX = {:.2e}, AVG = {:.2e} ms",
                 min_dur, max_dur, total_dur / world_size);
  } else {
    logger->info("* PT2_DUR = ${:.2e} ms", local_pt2_dur);
  }

  NPT2 = allreduce(NPT2, MPI_SUM, comm);
  logger->info("* NPT2 = {}", NPT2);

  return EPT2;
}
}  // namespace macis
#endif /* MACIS_ENABLE_MPI */

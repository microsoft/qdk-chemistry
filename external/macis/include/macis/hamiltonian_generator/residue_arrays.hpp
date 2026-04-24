/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <macis/hamiltonian_generator/connection_build_utils.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>

namespace macis {

/**
 * @brief Hamiltonian generator using the Residue Arrays algorithm.
 *
 * Implements Algorithm 5 from Tubman et al., J. Chem. Theory Comput. 2020,
 * 16, 2139–2159 (arXiv:1807.00821).  Optimized with:
 *   - ffs-based occupied-orbital extraction (vs scanning all N bits)
 *   - ips4o parallel sort of residue array
 *   - parallel block scan with thread-local pair buffers
 *   - pre-computed spin decomposition cache (no redundant work in fill)
 *   - exact-size CSR construction (no compact/copy phase)
 *
 * Only used for diagonal (bra==ket) blocks; falls back to SortedDoubleLoop
 * for off-diagonal blocks.
 */
template <typename WfnType>
class ResidueArrayHamiltonianGenerator
    : public SortedDoubleLoopHamiltonianGenerator<WfnType> {
 public:
  using base_type = SortedDoubleLoopHamiltonianGenerator<WfnType>;
  using full_det_t = typename base_type::full_det_t;
  using full_det_iterator = typename base_type::full_det_iterator;
  using matrix_span_t = matrix_span<double>;
  using rank4_span_t = rank4_span<double>;
  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double, index_t>;

  template <typename... Args>
  ResidueArrayHamiltonianGenerator(Args&&... args)
      : base_type(std::forward<Args>(args)...) {}

  // ---- Pair-based RDM override (spin-dependent only) ----

  void form_rdms_spin_dep(full_det_iterator bra_begin,
                          full_det_iterator bra_end,
                          full_det_iterator ket_begin,
                          full_det_iterator ket_end, double* C,
                          matrix_span_t ordm_aa, matrix_span_t ordm_bb,
                          rank4_span_t trdm_aaaa, rank4_span_t trdm_bbbb,
                          rank4_span_t trdm_aabb) override {
    const bool is_symm = bra_begin == ket_begin && bra_end == ket_end;
    if (!is_symm)
      return base_type::form_rdms_spin_dep(bra_begin, bra_end, ket_begin,
                                           ket_end, C, ordm_aa, ordm_bb,
                                           trdm_aaaa, trdm_bbbb, trdm_aabb);
    using clock_type = std::chrono::high_resolution_clock;
    auto st = clock_type::now();
    auto [pairs, cache] = enumerate_connected_pairs_(bra_begin, bra_end);
    const size_t ndets = std::distance(bra_begin, bra_end);
    const size_t npairs = pairs.size();

#pragma omp parallel
    {
      std::vector<uint32_t> bra_oa, bra_ob;
      uint32_t last_bra = UINT32_MAX;

#pragma omp for schedule(static)
      for (size_t k = 0; k < npairs; ++k) {
        uint32_t i = detail::pair_lo(pairs[k]);
        uint32_t j = detail::pair_hi(pairs[k]);

        double val = C[i] * C[j];
        if (std::abs(val) < 1e-16) continue;

        if (i != last_bra) {
          bra_oa.assign(cache.occ_a(i), cache.occ_a(i) + cache.n_alpha_elec);
          bra_ob.assign(cache.occ_b(i), cache.occ_b(i) + cache.n_beta_elec);
          last_bra = i;
        }

        auto ex_a = cache.alpha[i] ^ cache.alpha[j];
        auto ex_b = cache.beta[i] ^ cache.beta[j];

        rdm_contributions_spin_dep<true>(cache.alpha[i], cache.alpha[j], ex_a,
                                         cache.beta[i], cache.beta[j], ex_b,
                                         bra_oa, bra_ob, val, ordm_aa, ordm_bb,
                                         trdm_aaaa, trdm_bbbb, trdm_aabb);
      }

#pragma omp for schedule(static)
      for (size_t i = 0; i < ndets; ++i) {
        double val = C[i] * C[i];
        if (std::abs(val) < 1e-16) continue;

        bra_oa.assign(cache.occ_a(i), cache.occ_a(i) + cache.n_alpha_elec);
        bra_ob.assign(cache.occ_b(i), cache.occ_b(i) + cache.n_beta_elec);
        rdm_contributions_diag_spin_dep(bra_oa, bra_ob, val, ordm_aa, ordm_bb,
                                        trdm_aaaa, trdm_bbbb, trdm_aabb);
      }
    }

    auto en = clock_type::now();
    auto logger = spdlog::get("rdm");
    if (logger)
      logger->info("  RDM_SD(RA): {:.2e}s  ndets={} pairs={}",
                   std::chrono::duration<double>(en - st).count(), ndets,
                   pairs.size());
  }

  // form_entropies: NOT overridden — SDL's unique-alpha grouping is faster
  // for entropies because only excitation ≤ 2 contributes, and SDL naturally
  // skips high-excitation pairs via alpha-string grouping.

 protected:
  template <typename index_t>
  sparse_matrix_type<index_t> make_csr_hamiltonian_block_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end, double H_thresh) {
    if ((bra_begin != ket_begin) || (bra_end != ket_end))
      return base_type::template make_csr_hamiltonian_block_<index_t>(
          bra_begin, bra_end, ket_begin, ket_end, H_thresh);
    return residue_array_build_<index_t>(bra_begin, bra_end, H_thresh);
  }

  sparse_matrix_type<int32_t> make_csr_hamiltonian_block_32bit_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end,
      double H_thresh) override {
    return make_csr_hamiltonian_block_<int32_t>(bra_begin, bra_end, ket_begin,
                                                ket_end, H_thresh);
  }
  sparse_matrix_type<int64_t> make_csr_hamiltonian_block_64bit_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end,
      double H_thresh) override {
    return make_csr_hamiltonian_block_<int64_t>(bra_begin, bra_end, ket_begin,
                                                ket_end, H_thresh);
  }

  // ---- Pair enumeration via residue arrays ----
  std::pair<std::vector<uint64_t>, detail::SpinCache<WfnType>>
  enumerate_connected_pairs_(full_det_iterator dets_begin,
                             full_det_iterator dets_end) {
    using wfn_traits = wavefunction_traits<WfnType>;

    const size_t ndets = std::distance(dets_begin, dets_end);
    detail::SpinCache<WfnType> cache;
    cache.build(&*dets_begin, ndets);

    const size_t n_elec = dets_begin->count();
    const size_t residues_per_det = n_elec * (n_elec - 1) / 2;
    const size_t total_residues = ndets * residues_per_det;

    struct ResiduePair {
      WfnType residue;
      uint32_t det_idx;
    };
    std::vector<ResiduePair> residue_arr(total_residues);

#pragma omp parallel
    {
      std::vector<uint32_t> occ;
#pragma omp for schedule(static)
      for (size_t d = 0; d < ndets; ++d) {
        const auto& det = *(dets_begin + d);
        bits_to_indices(det, occ);
        size_t offset = d * residues_per_det;
        size_t k = 0;
        for (size_t i = 0; i < occ.size(); ++i) {
          for (size_t j = i + 1; j < occ.size(); ++j) {
            WfnType res = det;
            res.reset(occ[i]);
            res.reset(occ[j]);
            residue_arr[offset + k] = {res, static_cast<uint32_t>(d)};
            ++k;
          }
        }
      }
    }

    using wfn_less = typename wfn_traits::wfn_comparator;
    auto cmp = [](const ResiduePair& a, const ResiduePair& b) {
      return wfn_less{}(a.residue, b.residue);
    };
#if MACIS_HAS_IPS4O && defined(_OPENMP)
    ips4o::parallel::sort(residue_arr.begin(), residue_arr.end(), cmp);
#else
    std::sort(residue_arr.begin(), residue_arr.end(), cmp);
#endif

    std::vector<size_t> block_starts;
    block_starts.reserve(total_residues / 4);
    block_starts.push_back(0);
    for (size_t k = 1; k < total_residues; ++k) {
      if (!(residue_arr[k].residue == residue_arr[k - 1].residue))
        block_starts.push_back(k);
    }
    block_starts.push_back(total_residues);
    const size_t n_blocks = block_starts.size() - 1;

    int nthreads = 1;
#ifdef _OPENMP
    nthreads = omp_get_max_threads();
#endif
    std::vector<std::vector<uint64_t>> thread_pairs(nthreads);

#pragma omp parallel
    {
      int tid = 0;
#ifdef _OPENMP
      tid = omp_get_thread_num();
#endif
      auto& my_pairs = thread_pairs[tid];

#pragma omp for schedule(dynamic, 64)
      for (size_t bi = 0; bi < n_blocks; ++bi) {
        size_t bst = block_starts[bi];
        size_t ben = block_starts[bi + 1];
        if (ben - bst < 2) continue;

        for (size_t a = bst; a < ben; ++a) {
          uint32_t di = residue_arr[a].det_idx;
          for (size_t b = a + 1; b < ben; ++b) {
            uint32_t dj = residue_arr[b].det_idx;
            if (di == dj) continue;
            uint32_t lo = di < dj ? di : dj;
            uint32_t hi = di < dj ? dj : di;
            my_pairs.push_back(detail::pack_pair(lo, hi));
          }
        }
      }
    }

    {
      std::vector<ResiduePair>().swap(residue_arr);
    }

    size_t total_raw = 0;
    for (auto& tp : thread_pairs) total_raw += tp.size();
    std::vector<uint64_t> all_pairs;
    all_pairs.reserve(total_raw);
    for (auto& tp : thread_pairs) {
      all_pairs.insert(all_pairs.end(), tp.begin(), tp.end());
      {
        std::vector<uint64_t>().swap(tp);
      }
    }
    detail::sort_unique_pairs(all_pairs);

    return {std::move(all_pairs), std::move(cache)};
  }

 private:
  template <typename index_t>
  sparse_matrix_type<index_t> residue_array_build_(full_det_iterator dets_begin,
                                                   full_det_iterator dets_end,
                                                   double H_thresh) {
    const size_t ndets = std::distance(dets_begin, dets_end);
    auto [all_pairs, cache] = enumerate_connected_pairs_(dets_begin, dets_end);
    return detail::pair_based_build_impl_<index_t>(ndets, all_pairs, cache,
                                                   *this, H_thresh);
  }
};

}  // namespace macis

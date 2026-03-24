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
  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double, index_t>;

  template <typename... Args>
  ResidueArrayHamiltonianGenerator(Args&&... args)
      : base_type(std::forward<Args>(args)...) {}

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

 private:
  template <typename index_t>
  sparse_matrix_type<index_t> residue_array_build_(
      full_det_iterator dets_begin, full_det_iterator dets_end,
      double H_thresh) {
    using wfn_traits = wavefunction_traits<WfnType>;
    using clock_type = std::chrono::high_resolution_clock;

    const size_t ndets = std::distance(dets_begin, dets_end);
    if (ndets == 0) return sparse_matrix_type<index_t>(0, 0, 0, 0);
    auto h_logger = spdlog::get("h_build");

    // ---- Pre-compute spin decomposition ----
    auto decomp_st = clock_type::now();
    detail::SpinCache<WfnType> cache;
    cache.build(&*dets_begin, ndets);
    auto decomp_en = clock_type::now();

    // ---- Generate residues ----
    auto gen_st = clock_type::now();
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
        bits_to_indices(det, occ);  // ffs-based, much faster than scanning N bits

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
    auto gen_en = clock_type::now();

    // ---- Sort residues ----
    auto sort_st = clock_type::now();
    using wfn_less = typename wfn_traits::wfn_comparator;
    auto cmp = [](const ResiduePair& a, const ResiduePair& b) {
      return wfn_less{}(a.residue, b.residue);
    };
#if MACIS_HAS_IPS4O && defined(_OPENMP)
    ips4o::parallel::sort(residue_arr.begin(), residue_arr.end(), cmp);
#else
    std::sort(residue_arr.begin(), residue_arr.end(), cmp);
#endif
    auto sort_en = clock_type::now();

    // ---- Find block boundaries + parallel pair enumeration ----
    auto scan_st = clock_type::now();

    // Identify where residue changes (block boundaries)
    std::vector<size_t> block_starts;
    block_starts.reserve(total_residues / 4);
    block_starts.push_back(0);
    for (size_t k = 1; k < total_residues; ++k) {
      if (!(residue_arr[k].residue == residue_arr[k - 1].residue))
        block_starts.push_back(k);
    }
    block_starts.push_back(total_residues);
    const size_t n_blocks = block_starts.size() - 1;

    // Each thread accumulates pairs into a local buffer
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
    auto scan_en = clock_type::now();

    // Free residue array
    { std::vector<ResiduePair>().swap(residue_arr); }

    // ---- Merge + sort + dedup ----
    auto dedup_st = clock_type::now();
    size_t total_raw = 0;
    for (auto& tp : thread_pairs) total_raw += tp.size();
    std::vector<uint64_t> all_pairs;
    all_pairs.reserve(total_raw);
    for (auto& tp : thread_pairs) {
      all_pairs.insert(all_pairs.end(), tp.begin(), tp.end());
      { std::vector<uint64_t>().swap(tp); }
    }
    detail::sort_unique_pairs(all_pairs);
    auto dedup_en = clock_type::now();

    // ---- Build CSR ----
    auto csr_st = clock_type::now();
    auto result = detail::build_csr_from_pairs<index_t>(
        ndets, all_pairs, cache, *this, H_thresh);
    auto csr_en = clock_type::now();

    if (h_logger) {
      auto dur = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
      };
      h_logger->info(
          "  H_BUILD(RA): decomp={:.2e}s gen={:.2e}s sort={:.2e}s "
          "scan={:.2e}s dedup={:.2e}s csr={:.2e}s "
          "pairs_raw={} pairs_uniq={} nnz={}",
          dur(decomp_st, decomp_en), dur(gen_st, gen_en),
          dur(sort_st, sort_en), dur(scan_st, scan_en),
          dur(dedup_st, dedup_en), dur(csr_st, csr_en), total_raw,
          all_pairs.size(), result.nnz());
    }
    return result;
  }
};

}  // namespace macis

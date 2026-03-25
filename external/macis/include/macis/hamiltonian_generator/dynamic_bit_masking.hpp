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

#include <array>

namespace macis {

/**
 * @brief Hamiltonian generator using the Dynamic Bit Masking algorithm.
 *
 * Implements Algorithm 6 from Tubman et al., J. Chem. Theory Comput. 2020,
 * 16, 2139–2159 (arXiv:1807.00821).  Optimized with:
 *   - parallel combo processing (each thread handles a subset of C(n,4)
 *     combinations independently with thread-local key+pair buffers)
 *   - non-discriminating orbital filtering (always-0/1 orbitals removed
 *     from masks, reducing key width — often enables uint64 keys)
 *   - uint64 fast-path when total key bits ≤ 64
 *   - pre-computed spin decomposition cache
 *   - configurable number of masks (default 10; fewer for larger problems)
 *
 * Only used for diagonal (bra==ket) blocks; falls back to SortedDoubleLoop
 * for off-diagonal blocks.
 */
template <typename WfnType>
class DynamicBitMaskHamiltonianGenerator
    : public SortedDoubleLoopHamiltonianGenerator<WfnType> {
 public:
  using base_type = SortedDoubleLoopHamiltonianGenerator<WfnType>;
  using full_det_t = typename base_type::full_det_t;
  using full_det_iterator = typename base_type::full_det_iterator;
  template <typename index_t>
  using sparse_matrix_type = sparsexx::csr_matrix<double, index_t>;

  /// @param n_masks Number of bit mask groups (default 10 → C(10,4)=210
  /// combos).  Use 8 for C(8,4)=70 combos at the cost of more false
  /// positives.
  template <typename... Args>
  DynamicBitMaskHamiltonianGenerator(Args&&... args)
      : base_type(std::forward<Args>(args)...) {}

  void set_num_masks(int n) { n_masks_ = n; }
  int num_masks() const { return n_masks_; }

  // form_rdms, form_rdms_spin_dep, form_entropies: NOT overridden.
  // DBM's pair enumeration (C(10,4)=210 combo passes) is too expensive to
  // be faster than SDL's unique-alpha grouping for RDMs.  Benchmarks showed
  // DBM RDM at 35s vs SDL's 29s at 1M dets.  Entropy uses SDL via
  // inheritance (excitation ≤ 2 only, alpha grouping is ideal).

 protected:
  template <typename index_t>
  sparse_matrix_type<index_t> make_csr_hamiltonian_block_(
      full_det_iterator bra_begin, full_det_iterator bra_end,
      full_det_iterator ket_begin, full_det_iterator ket_end, double H_thresh) {
    if ((bra_begin != ket_begin) || (bra_end != ket_end))
      return base_type::template make_csr_hamiltonian_block_<index_t>(
          bra_begin, bra_end, ket_begin, ket_end, H_thresh);
    return dynamic_bit_mask_build_<index_t>(bra_begin, bra_end, H_thresh);
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

  // ---- Pair enumeration via dynamic bit masking ----
  std::pair<std::vector<uint64_t>, detail::SpinCache<WfnType>>
  enumerate_connected_pairs_(full_det_iterator dets_begin,
                             full_det_iterator dets_end) {
    const size_t ndets = std::distance(dets_begin, dets_end);
    const WfnType* dets = &*dets_begin;

    detail::SpinCache<WfnType> cache;
    cache.build(dets, ndets);

    auto variable_orbs = compute_discriminating_orbitals(dets_begin, ndets);
    std::vector<std::vector<uint32_t>> mask_bits;
    std::vector<int> bits_per_mask;
    assign_masks(variable_orbs, n_masks_, mask_bits, bits_per_mask);

    int max_key_bits = 0;
    {
      auto combos_tmp = generate_combos(n_masks_, bits_per_mask);
      for (auto& c : combos_tmp)
        max_key_bits = std::max(max_key_bits, c.key_bits);
    }
    const bool use_u64 = (max_key_bits <= 64);

    std::vector<uint64_t> masked_values(ndets * n_masks_);
#pragma omp parallel for schedule(static)
    for (size_t d = 0; d < ndets; ++d) {
      for (int m = 0; m < n_masks_; ++m) {
        masked_values[d * n_masks_ + m] = apply_mask(dets[d], mask_bits[m]);
      }
    }

    auto combos = generate_combos(n_masks_, bits_per_mask);
    const size_t n_combos = combos.size();

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

#pragma omp for schedule(dynamic)
      for (size_t ci = 0; ci < n_combos; ++ci) {
        if (use_u64)
          process_combo_u64_(combos[ci], ndets, masked_values, n_masks_,
                             bits_per_mask, dets, my_pairs);
        else
          process_combo_128_(combos[ci], ndets, masked_values, n_masks_,
                             bits_per_mask, dets, my_pairs);
      }

      std::sort(my_pairs.begin(), my_pairs.end());
      my_pairs.erase(std::unique(my_pairs.begin(), my_pairs.end()),
                     my_pairs.end());
    }

    size_t total_raw = 0;
    for (auto& tp : thread_pairs) total_raw += tp.size();
    std::vector<uint64_t> all_pairs;
    all_pairs.reserve(total_raw);
    for (auto& tp : thread_pairs) {
      all_pairs.insert(all_pairs.end(), tp.begin(), tp.end());
      { std::vector<uint64_t>().swap(tp); }
    }
    detail::sort_unique_pairs(all_pairs);

    return {std::move(all_pairs), std::move(cache)};
  }

 private:
  int n_masks_ = 10;

  // Max masks we support (compile-time upper bound for arrays)
  static constexpr int MAX_MASKS = 16;

  // ---- Combo descriptor ----
  struct Combo {
    std::vector<int> keep;   // indices of masks to keep
    int key_bits;            // total packed bits in composite key
  };

  // ---- Generate all C(n_masks, 4) combinations ----
  static std::vector<Combo> generate_combos(
      int n_masks, const std::vector<int>& bits_per_mask) {
    std::vector<Combo> combos;
    for (int a = 0; a < n_masks; ++a)
      for (int b = a + 1; b < n_masks; ++b)
        for (int c = b + 1; c < n_masks; ++c)
          for (int d = c + 1; d < n_masks; ++d) {
            Combo co;
            int total_bits = 0;
            for (int m = 0; m < n_masks; ++m) {
              if (m != a && m != b && m != c && m != d) {
                co.keep.push_back(m);
                total_bits += bits_per_mask[m];
              }
            }
            co.key_bits = total_bits;
            combos.push_back(std::move(co));
          }
    return combos;
  }

  // ---- Compute orbital activity, filtering non-discriminating orbitals ----
  static std::vector<uint32_t> compute_discriminating_orbitals(
      full_det_iterator dets_begin, size_t ndets) {
    using wfn_traits = wavefunction_traits<WfnType>;
    const size_t norbs = wfn_traits::size();

    // Count occupation freq
    std::vector<size_t> occ_count(norbs, 0);
    for (size_t d = 0; d < ndets; ++d) {
      const auto& det = *(dets_begin + d);
      for (size_t b = 0; b < norbs; ++b) {
        if (det[b]) occ_count[b]++;
      }
    }

    // Keep only orbitals that are variable (not always 0 or always 1)
    std::vector<uint32_t> variable_orbs;
    std::vector<std::pair<double, uint32_t>> activity;
    for (size_t b = 0; b < norbs; ++b) {
      if (occ_count[b] > 0 && occ_count[b] < ndets) {
        double freq = static_cast<double>(occ_count[b]) / ndets;
        activity.push_back({std::abs(freq - 0.5), static_cast<uint32_t>(b)});
      }
    }
    // Sort by activity (closest to 50% first — most discriminating)
    std::sort(activity.begin(), activity.end());
    variable_orbs.reserve(activity.size());
    for (auto& [d, b] : activity) variable_orbs.push_back(b);
    return variable_orbs;
  }

  // ---- Assign variable orbitals to masks (round-robin by activity) ----
  static void assign_masks(
      const std::vector<uint32_t>& variable_orbs, int n_masks,
      std::vector<std::vector<uint32_t>>& mask_bits,
      std::vector<int>& bits_per_mask) {
    mask_bits.resize(n_masks);
    bits_per_mask.resize(n_masks, 0);
    for (auto& mb : mask_bits) mb.clear();

    for (size_t i = 0; i < variable_orbs.size(); ++i) {
      int m = static_cast<int>(i % n_masks);
      mask_bits[m].push_back(variable_orbs[i]);
    }
    for (int m = 0; m < n_masks; ++m) {
      bits_per_mask[m] = static_cast<int>(mask_bits[m].size());
    }
  }

  // ---- Extract packed masked value for a det under one mask ----
  static uint64_t apply_mask(const WfnType& det,
                             const std::vector<uint32_t>& mbits) {
    uint64_t result = 0;
    for (size_t i = 0; i < mbits.size(); ++i) {
      if (det[mbits[i]]) result |= (uint64_t(1) << i);
    }
    return result;
  }

  // ---- Compose a uint64 key from kept masks ----
  static uint64_t compose_key_u64(
      const uint64_t* mv, int n_masks_stride,
      const std::vector<int>& keep,
      const std::vector<int>& bits_per_mask) {
    uint64_t key = 0;
    int shift = 0;
    for (int ki = 0; ki < static_cast<int>(keep.size()); ++ki) {
      int m = keep[ki];
      key |= (mv[m] << shift);
      shift += bits_per_mask[m];
    }
    return key;
  }

  // ---- Compose a 128-bit key ----
  struct Key128 {
    uint64_t lo, hi;
    uint32_t idx;
    bool operator<(const Key128& o) const {
      return hi != o.hi ? hi < o.hi : lo < o.lo;
    }
    bool key_eq(const Key128& o) const { return lo == o.lo && hi == o.hi; }
  };

  static Key128 compose_key_128(
      const uint64_t* mv, int n_masks_stride,
      const std::vector<int>& keep,
      const std::vector<int>& bits_per_mask, uint32_t det_idx) {
    uint64_t lo = 0, hi = 0;
    int shift = 0;
    for (int ki = 0; ki < static_cast<int>(keep.size()); ++ki) {
      int m = keep[ki];
      uint64_t val = mv[m];
      int nbits = bits_per_mask[m];
      if (shift < 64) {
        lo |= (val << shift);
        if (shift + nbits > 64) hi |= (val >> (64 - shift));
      } else {
        hi |= (val << (shift - 64));
      }
      shift += nbits;
    }
    return {lo, hi, det_idx};
  }

  // ---- Process one combo (uint64 key path) ----
  void process_combo_u64_(
      const Combo& combo, size_t ndets,
      const std::vector<uint64_t>& masked_values, int n_masks_total,
      const std::vector<int>& bits_per_mask,
      const WfnType* dets,
      std::vector<uint64_t>& out_pairs) const {
    // Build (key, det_idx) pairs
    struct KeyIdx {
      uint64_t key;
      uint32_t idx;
      bool operator<(const KeyIdx& o) const { return key < o.key; }
    };
    std::vector<KeyIdx> keys(ndets);
    for (size_t d = 0; d < ndets; ++d) {
      keys[d] = {compose_key_u64(&masked_values[d * n_masks_total],
                                  n_masks_total, combo.keep, bits_per_mask),
                 static_cast<uint32_t>(d)};
    }

    std::sort(keys.begin(), keys.end());

    // Scan blocks
    size_t bs = 0;
    while (bs < ndets) {
      size_t be = bs + 1;
      while (be < ndets && keys[be].key == keys[bs].key) ++be;
      if (be - bs > 1) {
        for (size_t a = bs; a < be; ++a) {
          for (size_t b = a + 1; b < be; ++b) {
            uint32_t di = keys[a].idx, dj = keys[b].idx;
            if (di > dj) std::swap(di, dj);
            // Verify: XOR popcount ≤ 4
            auto ex = dets[di] ^ dets[dj];
            if (ex.count() <= 4) {
              out_pairs.push_back(detail::pack_pair(di, dj));
            }
          }
        }
      }
      bs = be;
    }
  }

  // ---- Process one combo (128-bit key path) ----
  void process_combo_128_(
      const Combo& combo, size_t ndets,
      const std::vector<uint64_t>& masked_values, int n_masks_total,
      const std::vector<int>& bits_per_mask,
      const WfnType* dets,
      std::vector<uint64_t>& out_pairs) const {
    std::vector<Key128> keys(ndets);
    for (size_t d = 0; d < ndets; ++d) {
      keys[d] = compose_key_128(&masked_values[d * n_masks_total],
                                n_masks_total, combo.keep, bits_per_mask,
                                static_cast<uint32_t>(d));
    }

    std::sort(keys.begin(), keys.end());

    size_t bs = 0;
    while (bs < ndets) {
      size_t be = bs + 1;
      while (be < ndets && keys[be].key_eq(keys[bs])) ++be;
      if (be - bs > 1) {
        for (size_t a = bs; a < be; ++a) {
          for (size_t b = a + 1; b < be; ++b) {
            uint32_t di = keys[a].idx, dj = keys[b].idx;
            if (di > dj) std::swap(di, dj);
            auto ex = dets[di] ^ dets[dj];
            if (ex.count() <= 4) {
              out_pairs.push_back(detail::pack_pair(di, dj));
            }
          }
        }
      }
      bs = be;
    }
  }

  template <typename index_t>
  sparse_matrix_type<index_t> dynamic_bit_mask_build_(
      full_det_iterator dets_begin, full_det_iterator dets_end,
      double H_thresh) {
    using clock_type = std::chrono::high_resolution_clock;

    const size_t ndets = std::distance(dets_begin, dets_end);
    if (ndets == 0) return sparse_matrix_type<index_t>(0, 0, 0, 0);
    auto h_logger = spdlog::get("h_build");

    auto enum_st = clock_type::now();
    auto [all_pairs, cache] = enumerate_connected_pairs_(dets_begin, dets_end);
    auto enum_en = clock_type::now();

    auto csr_st = clock_type::now();
    auto result = detail::build_csr_from_pairs<index_t>(
        ndets, all_pairs, cache, *this, H_thresh);
    auto csr_en = clock_type::now();

    if (h_logger) {
      auto dur = [](auto a, auto b) {
        return std::chrono::duration<double>(b - a).count();
      };
      h_logger->info(
          "  H_BUILD(DBM): enum={:.2e}s csr={:.2e}s "
          "pairs_uniq={} nnz={}",
          dur(enum_st, enum_en), dur(csr_st, csr_en), all_pairs.size(),
          result.nnz());
    }
    return result;
  }
};

}  // namespace macis

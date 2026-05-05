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
#include <cstring>
#include <macis/hamiltonian_generator.hpp>
#include <macis/types.hpp>
#include <macis/wfn/raw_bitset.hpp>
#include <numeric>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if __has_include(<ips4o.hpp>)
#include <ips4o.hpp>
#define MACIS_HAS_IPS4O 1
#else
#define MACIS_HAS_IPS4O 0
#endif

namespace macis {
namespace detail {

// -----------------------------------------------------------------------
// SpinCache: pre-computed alpha/beta spin strings and occupation arrays for
// all determinants.  Eliminates redundant decomposition in the matrix
// element computation inner loop.
// -----------------------------------------------------------------------
template <typename WfnType>
struct SpinCache {
  using wfn_traits = wavefunction_traits<WfnType>;
  using spin_wfn_type = typename wfn_traits::spin_wfn_type;
  using spin_wfn_traits = wavefunction_traits<spin_wfn_type>;

  std::vector<spin_wfn_type> alpha;
  std::vector<spin_wfn_type> beta;
  std::vector<uint32_t> occ_alpha_flat;  // [ndets * n_alpha_elec]
  std::vector<uint32_t> occ_beta_flat;   // [ndets * n_beta_elec]
  size_t n_alpha_elec = 0;
  size_t n_beta_elec = 0;
  size_t ndets = 0;

  void build(const WfnType* dets, size_t n) {
    ndets = n;
    alpha.resize(n);
    beta.resize(n);

    auto a0 = wfn_traits::alpha_string(dets[0]);
    auto b0 = wfn_traits::beta_string(dets[0]);
    n_alpha_elec = a0.count();
    n_beta_elec = b0.count();

    occ_alpha_flat.resize(n * n_alpha_elec);
    occ_beta_flat.resize(n * n_beta_elec);

#pragma omp parallel
    {
      std::vector<uint32_t> tmp;
#pragma omp for schedule(static)
      for (size_t i = 0; i < n; ++i) {
        alpha[i] = wfn_traits::alpha_string(dets[i]);
        beta[i] = wfn_traits::beta_string(dets[i]);
        spin_wfn_traits::state_to_occ(alpha[i], tmp);
        std::copy_n(tmp.data(), n_alpha_elec,
                    &occ_alpha_flat[i * n_alpha_elec]);
        spin_wfn_traits::state_to_occ(beta[i], tmp);
        std::copy_n(tmp.data(), n_beta_elec, &occ_beta_flat[i * n_beta_elec]);
      }
    }
  }

  const uint32_t* occ_a(size_t i) const {
    return &occ_alpha_flat[i * n_alpha_elec];
  }
  const uint32_t* occ_b(size_t i) const {
    return &occ_beta_flat[i * n_beta_elec];
  }
};

// -----------------------------------------------------------------------
// Pair packing: encode (lo, hi) with lo<hi into a single uint64_t so
// that natural uint64 ordering sorts by lo first, hi second.
// -----------------------------------------------------------------------
inline uint64_t pack_pair(uint32_t lo, uint32_t hi) {
  return (static_cast<uint64_t>(lo) << 32) | hi;
}
inline uint32_t pair_lo(uint64_t p) { return static_cast<uint32_t>(p >> 32); }
inline uint32_t pair_hi(uint64_t p) {
  return static_cast<uint32_t>(p & 0xFFFFFFFF);
}

// -----------------------------------------------------------------------
// sort_unique_pairs: parallel sort and deduplicate a flat pair array.
// -----------------------------------------------------------------------
inline void sort_unique_pairs(std::vector<uint64_t>& pairs) {
#if MACIS_HAS_IPS4O && defined(_OPENMP)
  ips4o::parallel::sort(pairs.begin(), pairs.end());
#else
  std::sort(pairs.begin(), pairs.end());
#endif
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
}

// -----------------------------------------------------------------------
// build_csr_from_pairs: construct a symmetric CSR Hamiltonian from a
// deduplicated list of upper-triangle connection pairs.
//
// Input:  dets[0..ndets), sorted unique packed pairs (lo < hi),
//         pre-built SpinCache, ham_gen reference, threshold.
// Output: complete symmetric CSR with sorted columns per row.
// -----------------------------------------------------------------------
template <typename index_t, typename WfnType>
sparsexx::csr_matrix<double, index_t> build_csr_from_pairs(
    size_t ndets, const std::vector<uint64_t>& pairs,
    const SpinCache<WfnType>& cache, HamiltonianGenerator<WfnType>& ham_gen,
    double H_thresh) {
  using csr_type = sparsexx::csr_matrix<double, index_t>;
  if (ndets == 0) return csr_type(0, 0, 0, 0);

  const size_t npairs = pairs.size();

  // ---- Compute matrix elements for all pairs (parallel) ----
  // Pairs are sorted by lo (row), so static scheduling gives each thread
  // a contiguous row range → bra spin decomposition is reused.
  std::vector<double> pair_vals(npairs);

#pragma omp parallel
  {
    std::vector<uint32_t> bra_oa, bra_ob;
    uint32_t last_bra = UINT32_MAX;

#pragma omp for schedule(static)
    for (size_t k = 0; k < npairs; ++k) {
      uint32_t i = pair_lo(pairs[k]);
      uint32_t j = pair_hi(pairs[k]);

      if (i != last_bra) {
        bra_oa.assign(cache.occ_a(i), cache.occ_a(i) + cache.n_alpha_elec);
        bra_ob.assign(cache.occ_b(i), cache.occ_b(i) + cache.n_beta_elec);
        last_bra = i;
      }

      auto ex_a = cache.alpha[i] ^ cache.alpha[j];
      auto ex_b = cache.beta[i] ^ cache.beta[j];
      pair_vals[k] = ham_gen.matrix_element(cache.alpha[i], cache.alpha[j],
                                            ex_a, cache.beta[i], cache.beta[j],
                                            ex_b, bra_oa, bra_ob);
    }
  }

  // ---- Count NNZ per row (exact after threshold) ----
  std::vector<index_t> row_nnz(ndets, 1);  // 1 for diagonal
  for (size_t k = 0; k < npairs; ++k) {
    if (std::abs(pair_vals[k]) < H_thresh) continue;
    row_nnz[pair_lo(pairs[k])]++;
    row_nnz[pair_hi(pairs[k])]++;
  }

  // ---- Build rowptr ----
  std::vector<index_t> rowptr(ndets + 1, 0);
  for (size_t i = 0; i < ndets; ++i) rowptr[i + 1] = rowptr[i] + row_nnz[i];
  const size_t total_nnz = static_cast<size_t>(rowptr[ndets]);

  std::vector<index_t> colind(total_nnz);
  std::vector<double> nzval(total_nnz);
  // row_off tracks the next write position for each row
  std::vector<index_t> row_off(rowptr.begin(), rowptr.end() - 1);

  // ---- Fill diagonals (parallel, no contention) ----
#pragma omp parallel
  {
    std::vector<uint32_t> oa, ob;
#pragma omp for schedule(static)
    for (size_t i = 0; i < ndets; ++i) {
      oa.assign(cache.occ_a(i), cache.occ_a(i) + cache.n_alpha_elec);
      ob.assign(cache.occ_b(i), cache.occ_b(i) + cache.n_beta_elec);
      index_t pos = row_off[i]++;
      colind[pos] = static_cast<index_t>(i);
      nzval[pos] = ham_gen.matrix_element_diag(oa, ob);
    }
  }

  // ---- Fill off-diagonals (parallel over pairs, atomics for transpose) ----
#pragma omp parallel for schedule(static)
  for (size_t k = 0; k < npairs; ++k) {
    double v = pair_vals[k];
    if (std::abs(v) < H_thresh) continue;
    uint32_t i = pair_lo(pairs[k]);
    uint32_t j = pair_hi(pairs[k]);

    {
      index_t pos;
#pragma omp atomic capture
      pos = row_off[i]++;
      colind[pos] = static_cast<index_t>(j);
      nzval[pos] = v;
    }
    {
      index_t pos;
#pragma omp atomic capture
      pos = row_off[j]++;
      colind[pos] = static_cast<index_t>(i);
      nzval[pos] = v;
    }
  }

  // ---- Sort columns within each row (parallel) ----
#pragma omp parallel
  {
    std::vector<size_t> perm;
    std::vector<index_t> tmp_ci;
    std::vector<double> tmp_nz;
#pragma omp for schedule(dynamic)
    for (size_t i = 0; i < ndets; ++i) {
      index_t rs = rowptr[i], re = rowptr[i + 1];
      size_t rlen = static_cast<size_t>(re - rs);
      if (rlen <= 1) continue;
      perm.resize(rlen);
      std::iota(perm.begin(), perm.end(), size_t(0));
      std::sort(perm.begin(), perm.end(), [&](size_t a, size_t b) {
        return colind[rs + a] < colind[rs + b];
      });
      tmp_ci.resize(rlen);
      tmp_nz.resize(rlen);
      for (size_t k = 0; k < rlen; ++k) {
        tmp_ci[k] = colind[rs + perm[k]];
        tmp_nz[k] = nzval[rs + perm[k]];
      }
      std::copy(tmp_ci.begin(), tmp_ci.end(), colind.begin() + rs);
      std::copy(tmp_nz.begin(), tmp_nz.end(), nzval.begin() + rs);
    }
  }

  return csr_type(ndets, ndets, std::move(rowptr), std::move(colind),
                  std::move(nzval));
}

// -----------------------------------------------------------------------
// pair_based_build_impl_: shared CSR construction from pre-enumerated pairs.
// The caller is responsible for calling enumerate_connected_pairs_ (which
// is protected) and for any logging.
// -----------------------------------------------------------------------
template <typename index_t, typename WfnType>
auto pair_based_build_impl_(size_t ndets, std::vector<uint64_t>& all_pairs,
                            SpinCache<WfnType>& cache,
                            HamiltonianGenerator<WfnType>& gen, double H_thresh)
    -> sparsexx::csr_matrix<double, index_t> {
  if (ndets == 0) return sparsexx::csr_matrix<double, index_t>(0, 0, 0, 0);
  return build_csr_from_pairs<index_t>(ndets, all_pairs, cache, gen, H_thresh);
}

}  // namespace detail
}  // namespace macis

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
#include <cassert>
#include <chrono>
#include <cstring>
#include <macis/csr_hamiltonian.hpp>
#include <macis/types.hpp>
#include <numeric>
#include <optional>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace macis {

/**
 * @brief Davidson operator backed by a masked old CSR + delta blocks.
 *
 * Implements operator_action(m, alpha, V, LDV, beta, AV, LDAV) for
 * the Davidson eigensolver without rebuilding the full CSR each
 * iteration.
 *
 * @tparam index_t  CSR index type
 */
template <typename index_t>
class PatchedSparseOperator {
 public:
  using csr_type = sparsexx::csr_matrix<double, index_t>;

  PatchedSparseOperator(const csr_type& old_H, size_t n_new,
                        std::vector<size_t> old_to_new, csr_type H_nn,
                        csr_type H_on, std::vector<size_t> kept_new_indices,
                        std::vector<size_t> added_new_indices,
                        std::vector<double> diagonal)
      : old_H_(old_H),
        n_new_(n_new),
        n_old_(old_H.m()),
        old_to_new_(std::move(old_to_new)),
        H_nn_(std::move(H_nn)),
        H_on_(std::move(H_on)),
        kept_new_(std::move(kept_new_indices)),
        added_new_(std::move(added_new_indices)),
        diag_(std::move(diagonal)) {}

  size_t dimension() const { return n_new_; }
  const std::vector<double>& diagonal() const { return diag_; }

  /// @brief AV = alpha * H * V + beta * AV
  void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                       double beta, double* AV, size_t LDAV) const {
    assert(m == 1);
    (void)LDV;
    (void)LDAV;

    if (beta == 0.0)
      std::memset(AV, 0, n_new_ * sizeof(double));
    else if (beta != 1.0)
      for (size_t i = 0; i < n_new_; ++i) AV[i] *= beta;

    // Block 1: Old-Old — CSR SpMV with index remap (parallelised per row)
    {
      const auto& rp = old_H_.rowptr();
      const auto& ci = old_H_.colind();
      const auto& nz = old_H_.nzval();

#pragma omp parallel for schedule(static)
      for (size_t i_old = 0; i_old < n_old_; ++i_old) {
        size_t i_new = old_to_new_[i_old];
        if (i_new == SIZE_MAX) continue;
        double sum = 0.0;
        for (index_t jj = rp[i_old]; jj < rp[i_old + 1]; ++jj) {
          size_t j_new = old_to_new_[static_cast<size_t>(ci[jj])];
          if (j_new == SIZE_MAX) continue;
          sum += nz[jj] * V[j_new];
        }
        AV[i_new] += alpha * sum;
      }
    }

    // Block 2: Added-Added
    if (H_nn_.m() > 0) {
      const size_t n_added = added_new_.size();
      const auto& rp = H_nn_.rowptr();
      const auto& ci = H_nn_.colind();
      const auto& nz = H_nn_.nzval();

#pragma omp parallel for schedule(static)
      for (size_t i_local = 0; i_local < n_added; ++i_local) {
        size_t i_global = added_new_[i_local];
        double sum = 0.0;
        for (index_t jj = rp[i_local]; jj < rp[i_local + 1]; ++jj) {
          sum += nz[jj] * V[added_new_[static_cast<size_t>(ci[jj])]];
        }
        AV[i_global] += alpha * sum;
      }
    }

    // Block 3: Kept-Added + transpose
    if (H_on_.m() > 0) {
      const size_t n_kept = kept_new_.size();
      const size_t n_added = added_new_.size();
      const auto& rp = H_on_.rowptr();
      const auto& ci = H_on_.colind();
      const auto& nz = H_on_.nzval();

      // Forward: AV[kept] += H_on * V[added]
#pragma omp parallel for schedule(static)
      for (size_t i_local = 0; i_local < n_kept; ++i_local) {
        size_t i_global = kept_new_[i_local];
        double sum = 0.0;
        for (index_t jj = rp[i_local]; jj < rp[i_local + 1]; ++jj) {
          sum += nz[jj] * V[added_new_[static_cast<size_t>(ci[jj])]];
        }
        AV[i_global] += alpha * sum;
      }

      // Transpose: AV[added] += H_on^T * V[kept]
#ifdef _OPENMP
      if (n_added > 256) {
        const int nthreads = omp_get_max_threads();
        // Lazily allocate / resize cached scratch vectors
        if (transpose_scratch_.size() != static_cast<size_t>(nthreads) ||
            transpose_scratch_n_added_ != n_added) {
          transpose_scratch_.assign(nthreads,
                                    std::vector<double>(n_added, 0.0));
          transpose_scratch_n_added_ = n_added;
        } else {
          for (auto& v : transpose_scratch_)
            std::memset(v.data(), 0, n_added * sizeof(double));
        }
#pragma omp parallel
        {
#ifdef _OPENMP
          auto& local = transpose_scratch_[omp_get_thread_num()];
#else
          auto& local = transpose_scratch_[0];
#endif
#pragma omp for schedule(static)
          for (size_t i_local = 0; i_local < n_kept; ++i_local) {
            double v_kept = alpha * V[kept_new_[i_local]];
            for (index_t jj = rp[i_local]; jj < rp[i_local + 1]; ++jj) {
              local[static_cast<size_t>(ci[jj])] += nz[jj] * v_kept;
            }
          }
        }
        for (int t = 0; t < nthreads; ++t)
          for (size_t j = 0; j < n_added; ++j)
            AV[added_new_[j]] += transpose_scratch_[t][j];
      } else
#endif
      {
        for (size_t i_local = 0; i_local < n_kept; ++i_local) {
          double v_kept = alpha * V[kept_new_[i_local]];
          for (index_t jj = rp[i_local]; jj < rp[i_local + 1]; ++jj) {
            AV[added_new_[static_cast<size_t>(ci[jj])]] += nz[jj] * v_kept;
          }
        }
      }
    }
  }

 private:
  const csr_type& old_H_;
  size_t n_new_, n_old_;
  std::vector<size_t> old_to_new_;
  csr_type H_nn_, H_on_;
  std::vector<size_t> kept_new_, added_new_;
  std::vector<double> diag_;

  // Cached scratch for the transpose SpMV in operator_action.
  // Resized lazily on first use or when n_added changes.
  mutable std::vector<std::vector<double>> transpose_scratch_;
  mutable size_t transpose_scratch_n_added_ = 0;
};

/**
 * @brief Cached state for incremental H_build between ASCI iterations.
 */
template <typename WfnType, typename index_t = int64_t>
struct CachedHamiltonianState {
  using csr_type = sparsexx::csr_matrix<double, index_t>;

  std::vector<WfnType> dets;
  csr_type H;
  bool valid = false;

  void store(std::vector<WfnType> d, csr_type h) {
    dets = std::move(d);
    H = std::move(h);
    valid = true;
  }

  void clear() {
    dets.clear();
    H = csr_type(0, 0, 0, 0);
    valid = false;
  }
};

/**
 * @brief Build a PatchedSparseOperator from cached state + new dets.
 *
 * @param min_overlap  Minimum kept/new ratio to use patched build.
 * @return The operator, or nullopt if overlap is too low.
 */
template <typename index_t, typename WfnType>
std::optional<PatchedSparseOperator<index_t>> build_patched_operator(
    const CachedHamiltonianState<WfnType, index_t>& cache,
    const std::vector<WfnType>& new_dets,
    HamiltonianGenerator<WfnType>& ham_gen, double h_el_tol,
    double min_overlap = 0.3) {
  using csr_type = sparsexx::csr_matrix<double, index_t>;
  using wfn_traits = wavefunction_traits<WfnType>;
  using spin_wfn_traits =
      wavefunction_traits<typename wfn_traits::spin_wfn_type>;
  using wfn_comp = typename wfn_traits::spin_comparator;
  using clock_type = std::chrono::high_resolution_clock;
  using dur_s = std::chrono::duration<double>;

  auto logger = spdlog::get("h_build_inc");
  const size_t n_new = new_dets.size();
  const size_t n_old = cache.dets.size();

  // Classify: merge-scan both sorted lists
  std::vector<size_t> old_to_new(n_old, SIZE_MAX);
  std::vector<size_t> kept_new, added_new;
  kept_new.reserve(n_new);
  added_new.reserve(n_new);

  {
    size_t io = 0, in = 0;
    while (io < n_old && in < n_new) {
      if (wfn_comp{}(cache.dets[io], new_dets[in])) {
        ++io;
      } else if (wfn_comp{}(new_dets[in], cache.dets[io])) {
        added_new.push_back(in);
        ++in;
      } else {
        old_to_new[io] = in;
        kept_new.push_back(in);
        ++io;
        ++in;
      }
    }
    while (in < n_new) {
      added_new.push_back(in);
      ++in;
    }
  }

  const size_t n_kept = kept_new.size();
  const size_t n_added = added_new.size();
  const double overlap_frac =
      n_new > 0 ? static_cast<double>(n_kept) / static_cast<double>(n_new)
                : 0.0;

  if (logger) {
    logger->info(
        "  PATCH: n_old={}, n_new={}, n_kept={}, n_added={}, overlap={:.1f}%",
        n_old, n_new, n_kept, n_added, 100.0 * overlap_frac);
  }

  if (overlap_frac < min_overlap) {
    if (logger)
      logger->info(
          "  PATCH: overlap {:.1f}% < {:.0f}% threshold, falling back to full "
          "build",
          100.0 * overlap_frac, 100.0 * min_overlap);
    return std::nullopt;
  }

  // Build H_nn: added × added
  auto nn_st = clock_type::now();
  std::vector<WfnType> added_dets(n_added);
  for (size_t i = 0; i < n_added; ++i) added_dets[i] = new_dets[added_new[i]];

  csr_type H_nn(0, 0, 0, 0);
  if (n_added > 0) {
    H_nn = make_csr_hamiltonian_block<index_t>(
        added_dets.begin(), added_dets.end(), added_dets.begin(),
        added_dets.end(), ham_gen, h_el_tol);
  }
  auto nn_en = clock_type::now();

  // Build H_on: kept × added (rectangular)
  auto on_st = clock_type::now();
  std::vector<WfnType> kept_dets(n_kept);
  for (size_t i = 0; i < n_kept; ++i) kept_dets[i] = new_dets[kept_new[i]];

  csr_type H_on(0, 0, 0, 0);
  if (n_kept > 0 && n_added > 0) {
    H_on = make_csr_hamiltonian_block<index_t>(
        kept_dets.begin(), kept_dets.end(), added_dets.begin(),
        added_dets.end(), ham_gen, h_el_tol);
  }
  auto on_en = clock_type::now();

  // Compute diagonal
  auto diag_st = clock_type::now();
  std::vector<double> diagonal(n_new, 0.0);

  {
    const auto& rp = cache.H.rowptr();
    const auto& ci = cache.H.colind();
    const auto& nz = cache.H.nzval();
    for (size_t io = 0; io < n_old; ++io) {
      size_t in = old_to_new[io];
      if (in == SIZE_MAX) continue;
      for (index_t jj = rp[io]; jj < rp[io + 1]; ++jj) {
        if (static_cast<size_t>(ci[jj]) == io) {
          diagonal[in] = nz[jj];
          break;
        }
      }
    }
  }

#pragma omp parallel
  {
    std::vector<uint32_t> occ_a, occ_b;
#pragma omp for schedule(static)
    for (size_t i = 0; i < n_added; ++i) {
      auto a = wfn_traits::alpha_string(added_dets[i]);
      auto b = wfn_traits::beta_string(added_dets[i]);
      spin_wfn_traits::state_to_occ(a, occ_a);
      spin_wfn_traits::state_to_occ(b, occ_b);
      diagonal[added_new[i]] = ham_gen.matrix_element_diag(occ_a, occ_b);
    }
  }
  auto diag_en = clock_type::now();

  if (logger) {
    logger->info(
        "  PATCH: nn_nnz={}, nn_dur={:.3e}s, on_nnz={}, on_dur={:.3e}s, "
        "diag_dur={:.3e}s",
        H_nn.nnz(), dur_s(nn_en - nn_st).count(), H_on.nnz(),
        dur_s(on_en - on_st).count(), dur_s(diag_en - diag_st).count());
  }

  return PatchedSparseOperator<index_t>(
      cache.H, n_new, std::move(old_to_new), std::move(H_nn), std::move(H_on),
      std::move(kept_new), std::move(added_new), std::move(diagonal));
}

}  // namespace macis

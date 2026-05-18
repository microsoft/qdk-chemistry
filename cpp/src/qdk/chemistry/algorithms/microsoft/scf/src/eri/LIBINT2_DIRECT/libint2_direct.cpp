// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "libint2_direct.h"

#include <qdk/chemistry/scf/core/types.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <qdk/chemistry/utils/logger.hpp>

#include "../schwarz.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <blas.hh>
#include <cassert>
#include <stdexcept>
#include <unordered_set>

#include "util/timer.h"

namespace qdk::chemistry::scf {

namespace libint2::direct {
using qdk::chemistry::scf::RowMajorMatrix;

const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

/**
 * @brief Compute shell block norms of a matrix for screening purposes
 *
 * Computes the infinity norm (maximum absolute value) for each shell block
 * of matrix A. This is used for integral screening to determine which
 * shell quartets can be safely neglected based on density matrix magnitudes.
 *
 * @param obs Libint2 orbital basis set
 * @param A Matrix data in row-major format
 * @param LDA Leading dimension of matrix A
 * @return Matrix of shell block norms for screening
 */
/**
 * @brief Compute shell block norms of a density matrix for screening purposes
 *
 * Computes the infinity norm (maximum absolute value) for each shell block
 * of matrix A. When multiple density matrices are present (UHF: ndm=2),
 * takes the elementwise maximum across all densities so that screening
 * is conservative for all spin channels.
 *
 * @param obs Libint2 orbital basis set
 * @param A Matrix data in row-major format (ndm consecutive NAO x NAO blocks)
 * @param LDA Leading dimension of each matrix block (= NAO)
 * @param ndm Number of density matrices (1 for RHF, 2 for UHF)
 * @return Matrix of shell block norms for screening
 */
RowMajorMatrix compute_shellblock_norm(const ::libint2::BasisSet& obs,
                                       const double* A, size_t LDA,
                                       size_t ndm = 1) {
  QDK_LOG_TRACE_ENTERING();

  auto shell2bf = obs.shell2bf();
  const size_t nsh = obs.size();
  const size_t nbf = obs.nbf();
  RowMajorMatrix shnrms(nsh, nsh);
  shnrms.setZero();

  for (size_t idm = 0; idm < ndm; ++idm) {
    Eigen::Map<const RowMajorMatrix> A_map(A + idm * nbf * LDA, nbf, LDA);
    // Exploit density symmetry: compute lower triangle, mirror to upper
    for (size_t ish = 0; ish < nsh; ++ish)
      for (size_t jsh = 0; jsh <= ish; ++jsh) {
        double block_norm = A_map
                                .block(shell2bf[ish], shell2bf[jsh],
                                       obs[ish].size(), obs[jsh].size())
                                .lpNorm<Eigen::Infinity>();
        shnrms(ish, jsh) = std::max(shnrms(ish, jsh), block_norm);
        shnrms(jsh, ish) = shnrms(ish, jsh);
      }
  }
  return shnrms;
}

/**
 * @brief CSR (Compressed Sparse Row) shell-pair list
 *
 * Cache-friendly replacement for the unordered_map + shared_ptr shell-pair
 * data structures. Stores shell-pair neighbor indices in CSR format and
 * ShellPair geometric data contiguously by value.
 *
 * For each shell s, the significant neighbors are stored at indices
 * [offsets[s], offsets[s+1]) in the neighbors and data arrays. Neighbors
 * within each row are sorted in ascending order, which is required for the
 * `if (s4 > s4_max) break;` early-exit optimization.
 */
struct ShellPairCSR {
  std::vector<size_t> offsets;           ///< size nsh+1; row boundaries
  std::vector<size_t> neighbors;         ///< size = total significant pairs
  std::vector<::libint2::ShellPair> data;///< same length as neighbors; by value

  /// Number of significant pairs for shell s
  size_t row_size(size_t s) const { return offsets[s + 1] - offsets[s]; }

  /// Check if shell pair (s, t) exists (binary search since neighbors are sorted)
  bool contains(size_t s, size_t t) const {
    auto begin = neighbors.begin() + offsets[s];
    auto end = neighbors.begin() + offsets[s + 1];
    return std::binary_search(begin, end, t);
  }

  /// Get ShellPair data for (s, t), or nullptr if not found
  const ::libint2::ShellPair* get_data(size_t s, size_t t) const {
    auto begin = neighbors.begin() + offsets[s];
    auto end = neighbors.begin() + offsets[s + 1];
    auto it = std::lower_bound(begin, end, t);
    if (it != end && *it == t)
      return &data[it - neighbors.begin()];
    return nullptr;
  }
};

/**
 * @brief Precompute significant shell pairs in CSR format
 *
 * Replaces compute_shellpairs() with a cache-friendly CSR layout.
 * ShellPair objects are stored by value (no shared_ptr indirection).
 * Neighbors within each row are sorted ascending (required by the
 * early-exit break in the ket-pair loop).
 */
ShellPairCSR compute_shellpairs_csr(const ::libint2::BasisSet& obs,
                                    double threshold) {
  QDK_LOG_TRACE_ENTERING();

  const auto ln_max_engine_precision = std::log(max_engine_precision);
  const size_t nsh = obs.size();
#ifdef _OPENMP
  const int nthreads = omp_get_max_threads();
#else
  const int nthreads = 1;
#endif
  std::vector<::libint2::Engine> engines(
      nthreads, ::libint2::Engine(::libint2::Operator::overlap, obs.max_nprim(),
                                  obs.max_l(), 0));
  for (auto& e : engines) e.set_precision(0.);

  // Phase 1: Determine significant pairs per shell (thread-local lists)
  std::vector<std::vector<size_t>> per_shell_neighbors(nsh);

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    const int thread_id = omp_get_thread_num();
#else
    const int thread_id = 0;
#endif
    auto& engine = engines[thread_id];
    const auto& buf = engine.results();
#ifdef _OPENMP
#pragma omp for schedule(dynamic)
#endif
    for (size_t s1 = 0; s1 < nsh; ++s1) {
      const auto n1 = obs[s1].size();
      for (size_t s2 = 0; s2 <= s1; ++s2) {
        const auto n2 = obs[s2].size();
        bool on_same_center = (obs[s1].O == obs[s2].O);
        bool significant = on_same_center;
        if (not on_same_center) {
          engine.compute(obs[s1], obs[s2]);
          Eigen::Map<const RowMajorMatrix> buf_map(buf[0], n1, n2);
          const auto norm = buf_map.norm();
          significant = (norm >= threshold);
        }
        if (significant) per_shell_neighbors[s1].emplace_back(s2);
      }
    }
  }

  // Phase 2: Build CSR offsets
  ShellPairCSR csr;
  csr.offsets.resize(nsh + 1);
  csr.offsets[0] = 0;
  for (size_t s = 0; s < nsh; ++s) {
    csr.offsets[s + 1] = csr.offsets[s] + per_shell_neighbors[s].size();
  }
  const size_t total_pairs = csr.offsets[nsh];

  // Phase 3: Fill neighbors array (preserving ascending order per row)
  csr.neighbors.resize(total_pairs);
  for (size_t s = 0; s < nsh; ++s) {
    assert(std::is_sorted(per_shell_neighbors[s].begin(),
                          per_shell_neighbors[s].end()));
    std::copy(per_shell_neighbors[s].begin(), per_shell_neighbors[s].end(),
              csr.neighbors.begin() + csr.offsets[s]);
  }

  // Phase 4: Construct ShellPair data by value (parallel, indexed by position)
  csr.data.resize(total_pairs);
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic)
#endif
  for (size_t s1 = 0; s1 < nsh; ++s1) {
    for (size_t idx = csr.offsets[s1]; idx < csr.offsets[s1 + 1]; ++idx) {
      const size_t s2 = csr.neighbors[idx];
      csr.data[idx] = ::libint2::ShellPair(obs[s1], obs[s2],
                                           ln_max_engine_precision,
                                           ::libint2::ScreeningMethod::Original);
    }
  }

  return csr;
}

/**
 * @brief ERI class for direct computation of electron repulsion integrals using
 * Libint2
 *
 * This class implements efficient direct evaluation of two-electron repulsion
 * integrals using the Libint2 library. It employs Schwarz screening for
 * computational efficiency and supports both restricted and unrestricted
 * Hartree-Fock and DFT calculations.
 *
 * The implementation uses shell-pair pre-screening to avoid computation of
 * negligible integrals, significantly reducing computational cost for large
 * basis sets. All integrals are computed on-the-fly without storage, making it
 * suitable for large-scale calculations with limited memory.
 *
 * @note This class requires Libint2 library for integral evaluation
 */
class ERI {
  size_t spin_density_factor_;     ///< Number of spin density matrices (1 or 2)
  bool use_thread_local_buffers_;  ///< Use thread-local buffers (true) or
                                   ///< atomic ops (false)
  double eri_threshold_;           ///< Integral screening threshold
  ::libint2::BasisSet obs_;        ///< Libint2 orbital basis set representation
  std::vector<size_t>
      shell2bf_;  ///< Mapping from shell index to first atomic orbital
  ShellPairCSR sp_csr_;            ///< CSR shell-pair list + data (by value)
  RowMajorMatrix K_schwarz_;       ///< Schwarz screening matrix for integral bounds

  // Persistent per-thread Libint2 engine pool (constructed once, reused across calls)
  std::vector<::libint2::Engine> engines_; ///< One engine per thread
  int nthreads_;                           ///< Number of threads engines_ is sized for

  // Per-shell POD arrays for fast inner-loop access (avoid obs_[s].size() calls)
  std::vector<size_t> shell_nbf_;       ///< shell_nbf_[s] = number of basis functions in shell s
  std::vector<size_t> shell_bf_offset_; ///< shell_bf_offset_[s] = first basis function index
  size_t max_shellblock_bf_;            ///< max over all s of shell_nbf_[s]
  size_t nsh_ = 0;                     ///< number of shells

  // Persistent per-thread TLS accumulation buffers in TILED layout.
  // Each shell-pair (sa,sb) owns a contiguous tile of nbf[sa]*nbf[sb]
  // doubles. Tiles are packed sequentially, indexed by tile_offset_.
  // This eliminates N-strided scattered writes — all contraction flushes
  // and reduction reads are unit-stride within each tile.
  std::vector<size_t> tile_offset_;  ///< tile_offset_[sa*nsh+sb] = element offset
  size_t tile_total_size_ = 0;      ///< total doubles per density matrix
  std::vector<std::vector<double>> J_tls_; ///< Per-thread J accumulation (tiled)
  std::vector<std::vector<double>> K_tls_; ///< Per-thread K accumulation (tiled)
  size_t tls_mat_size_ = 0;               ///< Current allocation size per buffer

  // Per-thread touched shell-pair tracking for sparse reduction + zeroing.
  // Bitmap for O(1) dedup + compact list for O(touched) iteration.
  // The list avoids scanning the full nsh² bitmap 3× per call (zeroing,
  // cost model, sparse reduction).
  std::vector<std::vector<uint8_t>> touched_J_; ///< Per-thread J touched bitmap
  std::vector<std::vector<uint8_t>> touched_K_; ///< Per-thread K touched bitmap
  std::vector<std::vector<uint32_t>> touched_J_list_; ///< Per-thread J touched index list
  std::vector<std::vector<uint32_t>> touched_K_list_; ///< Per-thread K touched index list
  bool first_build_jk_ = true; ///< True until first build_JK completes (skip sparse zeroing)

  // Screening diagnostic counters (atomic for thread safety)
  std::atomic<size_t> diag_considered_{0};
  std::atomic<size_t> diag_schwarz_{0};
  std::atomic<size_t> diag_density_{0};
  std::atomic<size_t> diag_computed_{0};
  std::atomic<size_t> diag_engine_skip_{0};
  std::atomic<size_t> diag_j_would_screen_{0};
  std::atomic<size_t> diag_k_would_screen_{0};
  std::atomic<size_t> diag_calls_{0};

  // Precomputed screening data
  std::vector<double> K_schwarz_rowmax_;  ///< max_Q K_schwarz_(s3,Q) per shell s3
  // Distance-dependent Schwarz: shell-pair Gaussian product centers
  std::vector<std::array<double, 3>> pair_center_;  ///< pair_center_[s1*nsh+s2]

  // Pre-LinK data structures (Kussmann & Ochsenfeld, JCP 138, 134114, 2013)
  // sig_bras_[P] = shells Q sorted descending by K_schwarz_(P,Q) — built once
  std::vector<std::vector<size_t>> sig_bras_;
  std::vector<std::vector<double>> sig_bras_schwarz_;  ///< parallel Schwarz values
  // sig_kets_[P] = shells R sorted descending by screening value — rebuilt per build_JK
  std::vector<std::vector<size_t>> sig_kets_;
  std::vector<std::vector<double>> sig_kets_screen_;  ///< parallel screening values

  // Precomputed cost-balanced bra-pair schedule (built once, reused across calls)
  struct BraPair {
    size_t s1, s2;
    size_t sp_idx;
  };
  std::vector<BraPair> bra_pairs_;  ///< Cost-balanced ordered bra-pair list
  size_t n_bra_pairs_ = 0;

 public:
  /**
   * @brief Construct Libint2 direct ERI engine
   *
   * Initializes the direct integral evaluation engine with the given basis set.
   * Precomputes shell pair lists and Schwarz bounds for efficient screening
   * during integral evaluation. All screening data is computed once during
   * construction and reused throughout the calculation.
   *
   * @param spin_density_factor Number of spin density matrices (1 or 2)
   * @param basis_set QDK basis set (converted to Libint2 format internally)
   * @param use_atomics Use atomic operations (true) or thread-local buffers
   *        (false)
   * @param eri_threshold ERI screening threshold for skipping negligible
   *        shell quartets during J/K builds and quarter transformations
   * @param shell_pair_threshold Overlap-based shell pair pre-screening
   *        threshold
   *
   * @note Construction involves significant overhead due to screening setup
   * @note Shell pair and Schwarz data is computed using OpenMP parallelization
   */
  ERI(size_t spin_density_factor, qdk::chemistry::scf::BasisSet& basis_set,
      bool use_atomics, double eri_threshold, double shell_pair_threshold)
      : spin_density_factor_(spin_density_factor),
        use_thread_local_buffers_(!use_atomics),
        eri_threshold_(eri_threshold),
        obs_(libint2_util::convert_to_libint_basisset(basis_set)) {
    QDK_LOG_TRACE_ENTERING();

    shell2bf_ = obs_.shell2bf();

    // Build per-shell POD arrays
    const size_t nsh = obs_.size();
    shell_nbf_.resize(nsh);
    shell_bf_offset_.resize(nsh);
    max_shellblock_bf_ = 0;
    for (size_t s = 0; s < nsh; ++s) {
      shell_nbf_[s] = obs_[s].size();
      shell_bf_offset_[s] = shell2bf_[s];
      max_shellblock_bf_ = std::max(max_shellblock_bf_, shell_nbf_[s]);
    }

    // Compute Shell Pairs in CSR format
    sp_csr_ = compute_shellpairs_csr(obs_, shell_pair_threshold);

    // Compute Schwarz Screening
    K_schwarz_ = RowMajorMatrix(obs_.size(), obs_.size());
    auto mpi = mpi_default_input();
    schwarz_integral(&basis_set, mpi, K_schwarz_.data(), true);

    // Construct persistent per-thread engine pool
#ifdef _OPENMP
    nthreads_ = omp_get_max_threads();
#else
    nthreads_ = 1;
#endif
    engines_.resize(nthreads_);
    engines_[0] = ::libint2::Engine(::libint2::Operator::coulomb,
                                    obs_.max_nprim(), obs_.max_l(), 0);
    engines_[0].set(::libint2::ScreeningMethod::Original);
    for (int i = 1; i < nthreads_; ++i) engines_[i] = engines_[0];

    // Build tile layout with cache-line alignment.
    // Each tile is padded to start at a 64-byte (8-double) boundary to
    // prevent false sharing between adjacent tiles during parallel reduction.
    constexpr size_t CACHE_LINE_DOUBLES = 8;  // 64 bytes / 8 bytes per double
    const size_t nbf = obs_.nbf();
    nsh_ = nsh;
    tile_offset_.resize(nsh * nsh);
    tile_total_size_ = 0;
    for (size_t sa = 0; sa < nsh; ++sa)
      for (size_t sb = 0; sb < nsh; ++sb) {
        tile_offset_[sa * nsh + sb] = tile_total_size_;
        size_t tile_size = shell_nbf_[sa] * shell_nbf_[sb];
        // Round up to cache line boundary
        tile_size = (tile_size + CACHE_LINE_DOUBLES - 1) & ~(CACHE_LINE_DOUBLES - 1);
        tile_total_size_ += tile_size;
      }
    tls_mat_size_ = spin_density_factor_ * tile_total_size_;

    // Pre-allocate per-thread TLS buffers with first-touch NUMA placement.
    J_tls_.resize(nthreads_);
    K_tls_.resize(nthreads_);
    touched_J_.resize(nthreads_);
    touched_K_.resize(nthreads_);
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      J_tls_[tid].resize(tls_mat_size_);
      K_tls_[tid].resize(tls_mat_size_);
      std::memset(J_tls_[tid].data(), 0, tls_mat_size_ * sizeof(double));
      std::memset(K_tls_[tid].data(), 0, tls_mat_size_ * sizeof(double));
      touched_J_[tid].assign(nsh * nsh, 0);
      touched_K_[tid].assign(nsh * nsh, 0);
    }
    touched_J_list_.resize(nthreads_);
    touched_K_list_.resize(nthreads_);
    for (int i = 0; i < nthreads_; ++i) {
      touched_J_list_[i].reserve(nsh * nsh / 4);
      touched_K_list_[i].reserve(nsh * nsh / 2);
    }

    // Precompute K_schwarz row maxima for coarse s3-row pre-screening
    K_schwarz_rowmax_.resize(nsh);
    for (size_t s3 = 0; s3 < nsh; ++s3) {
      double mx = 0.0;
      for (size_t s4 = 0; s4 < nsh; ++s4)
        mx = std::max(mx, K_schwarz_(s3, s4));
      K_schwarz_rowmax_[s3] = mx;
    }

    // Distance-dependent Schwarz: precompute shell-pair Gaussian product centers
    pair_center_.resize(nsh * nsh);
    for (size_t s1 = 0; s1 < nsh; ++s1) {
      const auto& sh1 = obs_[s1];
      double min_exp1 = sh1.alpha[0];
      for (const auto& a : sh1.alpha) min_exp1 = std::min(min_exp1, a);
      for (size_t s2 = 0; s2 <= s1; ++s2) {
        const auto& sh2 = obs_[s2];
        double min_exp2 = sh2.alpha[0];
        for (const auto& a : sh2.alpha) min_exp2 = std::min(min_exp2, a);
        double zeta = min_exp1 + min_exp2;
        std::array<double, 3> P;
        for (int d = 0; d < 3; ++d)
          P[d] = (min_exp1 * sh1.O[d] + min_exp2 * sh2.O[d]) / zeta;
        pair_center_[s1 * nsh + s2] = P;
        pair_center_[s2 * nsh + s1] = P;
      }
    }

    // Pre-LinK: build Schwarz-sorted bra lists (density-independent, built once)
    // sig_bras_[P] = shells Q sorted descending by K_schwarz_(P,Q)
    sig_bras_.resize(nsh);
    sig_bras_schwarz_.resize(nsh);
    for (size_t P = 0; P < nsh; ++P) {
      std::vector<std::pair<double, size_t>> pq_vals;
      for (size_t Q = 0; Q < nsh; ++Q) {
        double val = K_schwarz_(P, Q);
        if (val >= 1e-15) pq_vals.push_back({val, Q});
      }
      std::sort(pq_vals.begin(), pq_vals.end(),
                [](const auto& a, const auto& b) { return a.first > b.first; });
      sig_bras_[P].reserve(pq_vals.size());
      sig_bras_schwarz_[P].reserve(pq_vals.size());
      for (const auto& [val, Q] : pq_vals) {
        sig_bras_[P].push_back(Q);
        sig_bras_schwarz_[P].push_back(val);
      }
    }

    // Build cost-balanced bra-pair schedule (once, reused across calls).
    // Pure function of basis and thread count — no density dependence.
    {
      struct BraPairCost { size_t s1, s2, sp_idx, est; };
      std::vector<BraPairCost> raw;
      raw.reserve(sp_csr_.neighbors.size());
      for (size_t s1 = 0; s1 < nsh; ++s1) {
        size_t est_kets = 0;
        for (size_t s3 = 0; s3 <= s1; ++s3)
          est_kets += sp_csr_.row_size(s3);
        for (size_t idx = sp_csr_.offsets[s1]; idx < sp_csr_.offsets[s1 + 1]; ++idx)
          raw.push_back({s1, sp_csr_.neighbors[idx], idx, est_kets});
      }
      n_bra_pairs_ = raw.size();

      std::vector<size_t> order(n_bra_pairs_);
      std::iota(order.begin(), order.end(), 0);
      std::stable_sort(order.begin(), order.end(),
                       [&](size_t a, size_t b) { return raw[a].est > raw[b].est; });
      std::vector<std::vector<size_t>> bins(nthreads_);
      for (size_t r = 0; r < n_bra_pairs_; ++r)
        bins[r % nthreads_].push_back(order[r]);
      bra_pairs_.reserve(n_bra_pairs_);
      for (int t = 0; t < nthreads_; ++t)
        for (size_t idx : bins[t])
          bra_pairs_.push_back({raw[idx].s1, raw[idx].s2, raw[idx].sp_idx});
    }
  }

  /// Update the integral screening threshold for subsequent build_JK calls.
  void set_eri_threshold(double threshold) { eri_threshold_ = threshold; }

  /**
   * @brief Build Coulomb (J) and exchange (K) matrices using direct integral
   * evaluation
   *
   * Computes J and K matrices by direct evaluation of two-electron repulsion
   * integrals using the Libint2 library. Employs multiple levels of screening:
   * 1. Shell pair pre-screening based on overlap
   * 2. Schwarz inequality bounds: (μν|λσ) ≤ K(μν) × K(λσ)
   * 3. Density-based screening using shell block norms
   * 4. Libint2 internal screening based on primitive overlap
   *
   * The implementation uses OpenMP parallelization and atomic operations for
   * thread-safe accumulation of matrix elements.
   *
   * @param P Density matrix in AO basis (packed for unrestricted: [Pα, Pβ])
   * @param J Output Coulomb matrix (nullptr if not needed)
   * @param K Output exchange matrix (nullptr if not needed)
   * @param alpha Scaling factor for Hartree-Fock exchange
   * @param beta Scaling factor for DFT exchange
   * @param omega Range-separation parameter (not supported, must be ~0)
   *
   * @throws std::runtime_error If range-separated hybrid is requested
   *
   * @note Both J and K matrices are symmetrized at the end
   * @note K matrix is scaled by (alpha + beta)
   * @note Supports both restricted (num_density_matrices=1) and unrestricted
   * (num_density_matrices=2) calculations
   */
  void build_JK(const double* P, double* J, double* K, double alpha,
                double beta, double omega) {
    QDK_LOG_TRACE_ENTERING();

    AutoTimer t("ERI::build_JK");
    // Reset per-call screening counters
    diag_considered_ = 0; diag_schwarz_ = 0; diag_density_ = 0;
    diag_computed_ = 0; diag_engine_skip_ = 0;
    diag_j_would_screen_ = 0; diag_k_would_screen_ = 0;
    const size_t N = obs_.nbf();
    const size_t nsh = obs_.size();
    const size_t ndm = spin_density_factor_;
    const size_t mat_size = ndm * N * N;
    const bool is_rsx = std::abs(omega) > 1e-12;

    if (is_rsx) throw std::runtime_error("RSX + LIBINT2_DIRECT NYI");

    // Compute shell block norm of P (max over all density matrices for UHF)
    const auto P_shnrm =
        compute_shellblock_norm(obs_, P, N, ndm);
    const auto P_shmax = P_shnrm.maxCoeff();

    if (!std::isfinite(P_shmax)) {
      throw std::runtime_error("Density matrix contains NaN/Inf values.");
    }

    // Use the configured screening threshold directly. The caller
    // (scf_impl.cpp) may adjust eri_threshold_ via set_screening_threshold()
    // for adaptive precision as SCF converges.
    const double eff_threshold = eri_threshold_;

    // Pre-LinK: build density-weighted significant ket lists
    // sig_kets_[P] = shells R where ceiling[P]*ceiling[R]*D_max(P,R) >= threshold
    // Sorted descending by screening value for early-exit
    sig_kets_.resize(nsh);
    sig_kets_screen_.resize(nsh);
    if (K) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (size_t P = 0; P < nsh; ++P) {
        sig_kets_[P].clear();
        sig_kets_screen_[P].clear();
        std::vector<std::pair<double, size_t>> pr_vals;
        for (size_t R = 0; R < nsh; ++R) {
          double screen_val =
              K_schwarz_rowmax_[P] * K_schwarz_rowmax_[R] * P_shnrm(P, R);
          if (screen_val >= eff_threshold) {
            pr_vals.push_back({screen_val, R});
          }
        }
        std::sort(pr_vals.begin(), pr_vals.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });
        sig_kets_[P].reserve(pr_vals.size());
        sig_kets_screen_[P].reserve(pr_vals.size());
        for (const auto& [val, R] : pr_vals) {
          sig_kets_[P].push_back(R);
          sig_kets_screen_[P].push_back(val);
        }
      }
    }

    // Pre-LinK diagnostic: report sig_kets statistics
    if (K && std::getenv("QDK_FOCK_SCREEN_DIAG")) {
      size_t total_kets = 0, max_kets = 0;
      for (size_t P = 0; P < nsh; ++P) {
        total_kets += sig_kets_[P].size();
        max_kets = std::max(max_kets, sig_kets_[P].size());
      }
      double avg_kets = static_cast<double>(total_kets) / nsh;
      QDK_LOGGER().info(
          "preLinK: nsh={} sig_kets_avg={:.1f} sig_kets_max={} "
          "sig_kets_total={} (of {} = {:.1f}%)",
          nsh, avg_kets, max_kets, total_kets, nsh * nsh,
          100.0 * total_kets / (nsh * nsh));
    }

    double engine_precision;
    if (P_shmax <= 0.0) {
      engine_precision = std::numeric_limits<double>::epsilon();
    } else {
      engine_precision = std::numeric_limits<double>::epsilon() / P_shmax;
    }
    engine_precision = std::max(engine_precision, max_engine_precision);

    for (int i = 0; i < nthreads_; ++i)
      engines_[i].set_precision(engine_precision);

    // Parallel zero of output (NUMA-friendly, avoid serial memset)
    if (J) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (size_t i = 0; i < mat_size; ++i) J[i] = 0.0;
    }
    if (K) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
      for (size_t i = 0; i < mat_size; ++i) K[i] = 0.0;
    }

    // Accumulators for per-thread sparse work cost (computed in parallel)
    size_t j_sparse_total = 0;
    size_t k_sparse_total = 0;

    // Single parallel region: zero TLS buffers + compute + cost model
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const int tid = omp_get_thread_num();
#else
      const int tid = 0;
#endif
      // Sparse zeroing: contiguous tile memset (O(touched), no N-stride)
      auto& tj = touched_J_[tid];
      auto& tk = touched_K_[tid];
      if (!first_build_jk_) {
        {
          double* jd = J_tls_[tid].data();
          for (uint32_t idx : touched_J_list_[tid]) {
            const size_t sa = idx / nsh, sb = idx % nsh;
            const size_t tile_sz = shell_nbf_[sa] * shell_nbf_[sb];
            for (size_t idm = 0; idm < ndm; ++idm)
              std::memset(jd + idm * tile_total_size_ + tile_offset_[idx],
                          0, tile_sz * sizeof(double));
            tj[idx] = 0;
          }
        }
        {
          double* kd = K_tls_[tid].data();
          for (uint32_t idx : touched_K_list_[tid]) {
            const size_t sa = idx / nsh, sb = idx % nsh;
            const size_t tile_sz = shell_nbf_[sa] * shell_nbf_[sb];
            for (size_t idm = 0; idm < ndm; ++idm)
              std::memset(kd + idm * tile_total_size_ + tile_offset_[idx],
                          0, tile_sz * sizeof(double));
            tk[idx] = 0;
          }
        }
      }
      touched_J_list_[tid].clear();
      touched_K_list_[tid].clear();

#ifdef _OPENMP
#pragma omp barrier
#endif

      auto& engine = engines_[tid];
      const auto& buf = engine.results();
      double* J_thr = J ? J_tls_[tid].data() : nullptr;
      double* K_thr = K ? K_tls_[tid].data() : nullptr;

      // Per-thread screening counters
      size_t loc_considered = 0, loc_schwarz = 0, loc_density = 0;
      size_t loc_computed = 0, loc_engine_skip = 0;
      size_t loc_j_screen = 0, loc_k_screen = 0;

      // Pre-LinK: thread-local epoch-based dedup for ML_PQ construction
      // ml_epoch_ marks which (s3*nsh+s4) pairs were already added this bra pair
      std::vector<uint32_t> ml_epoch(nsh * nsh, 0);
      uint32_t ml_epoch_counter = 0;
      std::vector<std::pair<size_t, size_t>> ml_pairs;  // (s3, s4) candidates
      ml_pairs.reserve(256);

      // Dynamic scheduling over bra-pairs — each thread processes
      // contiguous bra-pair ranges, improving cache locality for bra-side
      // density rows and Schwarz data vs the old round-robin.
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
      for (size_t bp_idx = 0; bp_idx < n_bra_pairs_; ++bp_idx) {
        const auto& bp = bra_pairs_[bp_idx];
        const size_t s1 = bp.s1;
        const size_t s2 = bp.s2;
        const auto bf1_st = shell_bf_offset_[s1];
        const auto bf2_st = shell_bf_offset_[s2];
        const auto n1 = shell_nbf_[s1];
        const auto n2 = shell_nbf_[s2];
        const auto* sp12_data = &sp_csr_.data[bp.sp_idx];
        const auto P12_nrm = P_shnrm(s1, s2);

        const auto schwarz_pq = K_schwarz_(s1, s2);

        // ============================================================
        // Pre-LinK K-only path: build ML_PQ from sig_kets + sig_bras
        // with double-sorted early-exit, then iterate only ML_PQ.
        // ============================================================
        if (K_thr && !J_thr) {
          ++ml_epoch_counter;
          ml_pairs.clear();

          // Build ML_PQ from sig_kets[s1]: for each R in sig_kets[s1],
          // walk sig_bras[R] with early-exit to find significant (R,S) pairs
          for (size_t ri = 0; ri < sig_kets_[s1].size(); ++ri) {
            const size_t R = sig_kets_[s1][ri];
            bool any_sig = false;
            for (size_t si = 0; si < sig_bras_[R].size(); ++si) {
              const size_t S = sig_bras_[R][si];
              // Canonical ordering: (R,S) with R >= S
              size_t s3 = (R >= S) ? R : S;
              size_t s4 = (R >= S) ? S : R;
              // Canonical constraint: (s3,s4) <= (s1,s2)
              if (s3 > s1) continue;
              if (s3 == s1 && s4 > s2) continue;
              // Screen: D_max(s1,R) * K_schwarz(s1,s2) * K_schwarz(R,S) >= threshold
              double screen_val = P_shnrm(s1, R) * schwarz_pq *
                                  sig_bras_schwarz_[R][si];
              if (screen_val < eff_threshold) break;  // early exit: sig_bras sorted desc by K_schwarz(R,S)
              any_sig = true;
              // Dedup via epoch marker
              uint32_t pair_key = static_cast<uint32_t>(s3 * nsh + s4);
              if (ml_epoch[pair_key] != ml_epoch_counter) {
                ml_epoch[pair_key] = ml_epoch_counter;
                ml_pairs.push_back({s3, s4});
              }
            }
            // DISABLED: // DISABLED: if (!any_sig) break;  // early exit on R
          }

          // Also walk sig_kets[s2] to capture pairs significant through Q
          for (size_t ri = 0; ri < sig_kets_[s2].size(); ++ri) {
            const size_t R = sig_kets_[s2][ri];
            bool any_sig = false;
            for (size_t si = 0; si < sig_bras_[R].size(); ++si) {
              const size_t S = sig_bras_[R][si];
              size_t s3 = (R >= S) ? R : S;
              size_t s4 = (R >= S) ? S : R;
              if (s3 > s1) continue;
              if (s3 == s1 && s4 > s2) continue;
              double screen_val = P_shnrm(s2, R) * schwarz_pq *
                                  sig_bras_schwarz_[R][si];
              if (screen_val < eff_threshold) break;
              any_sig = true;
              uint32_t pair_key = static_cast<uint32_t>(s3 * nsh + s4);
              if (ml_epoch[pair_key] != ml_epoch_counter) {
                ml_epoch[pair_key] = ml_epoch_counter;
                ml_pairs.push_back({s3, s4});
              }
            }
            // DISABLED: if (!any_sig) break;
          }

          // Iterate ML_PQ and compute/contract K
          for (const auto& [s3, s4] : ml_pairs) {
            ++loc_considered;

            // Check shell pair significance (overlap prescreen)
            if (!sp_csr_.contains(s3, s4)) continue;

            const auto bf3_st = shell_bf_offset_[s3];
            const auto bf4_st = shell_bf_offset_[s4];
            const auto n3 = shell_nbf_[s3];
            const auto n4 = shell_nbf_[s4];

            // Exact K screening with all 4 cross-pair norms
            const auto P13_nrm = P_shnrm(s1, s3);
            const auto P14_nrm = P_shnrm(s1, s4);
            const auto P23_nrm = P_shnrm(s2, s3);
            const auto P24_nrm = P_shnrm(s2, s4);
            double Pk = std::max({P13_nrm, P14_nrm, P23_nrm, P24_nrm});
            const auto schwarz_bound = K_schwarz_(s1, s2) * K_schwarz_(s3, s4);
            if (Pk * schwarz_bound < eff_threshold) {
              ++loc_density;
              continue;
            }

            // Get shell pair data for s3,s4
            const auto* sp34_data = sp_csr_.get_data(s3, s4);
            if (!sp34_data) continue;

            // Mark touched K shell pairs
            {
              uint32_t idxs[] = {
                static_cast<uint32_t>(s1*nsh+s3), static_cast<uint32_t>(s1*nsh+s4),
                static_cast<uint32_t>(s2*nsh+s3), static_cast<uint32_t>(s2*nsh+s4)};
              for (auto idx : idxs)
                if (!tk[idx]) { tk[idx] = 1; touched_K_list_[tid].push_back(idx); }
            }

            auto s12_deg = (s1 == s2) ? 1 : 2;
            auto s34_deg = (s3 == s4) ? 1 : 2;
            auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

            engine.compute2<::libint2::Operator::coulomb,
                            ::libint2::BraKet::xx_xx, 0>(
                obs_[s1], obs_[s2], obs_[s3], obs_[s4], sp12_data, sp34_data);

            const auto buf_1234 = buf[0];
            if (buf_1234 == nullptr) { ++loc_engine_skip; continue; }
            ++loc_computed;

            // K-only contraction (tiled TLS)
            for (size_t idm = 0; idm < ndm; ++idm) {
              auto* K_cur = K_thr + idm * tile_total_size_;
              const auto* P_cur = P + idm * N * N;
              const size_t kt_13 = tile_offset_[s1 * nsh_ + s3];
              const size_t kt_14 = tile_offset_[s1 * nsh_ + s4];
              const size_t kt_23 = tile_offset_[s2 * nsh_ + s3];
              const size_t kt_24 = tile_offset_[s2 * nsh_ + s4];
              for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                for (size_t j = 0; j < n2; ++j) {
                  for (size_t k = 0; k < n3; ++k) {
                    double K_ik = 0.0, K_jk = 0.0;
                    const double P_ik = 0.25 * P_cur[(bf1_st + i) * N + (bf3_st + k)];
                    const double P_jk = 0.25 * P_cur[(bf2_st + j) * N + (bf3_st + k)];
                    for (size_t l = 0; l < n4; ++l, ++ijkl) {
                      const auto value = buf_1234[ijkl] * s1234_deg;
                      K_ik += 0.25 * P_cur[(bf2_st + j) * N + (bf4_st + l)] * value;
                      K_jk += 0.25 * P_cur[(bf1_st + i) * N + (bf4_st + l)] * value;
                      K_cur[kt_14 + i * n4 + l] += P_jk * value;
                      K_cur[kt_24 + j * n4 + l] += P_ik * value;
                    }
                    K_cur[kt_13 + i * n3 + k] += K_ik;
                    K_cur[kt_23 + j * n3 + k] += K_jk;
                  }
                }
              }
            }
          }  // end ML_PQ iteration
          continue;  // skip the standard s3/s4 loop below
        }  // end K-only pre-LinK path

        for (size_t s3 = 0; s3 <= s1; ++s3) {
          // Coarse s3-row pre-screen: skip entire row when even the best
          // ket pair can't produce a significant integral bound
          if (schwarz_pq * K_schwarz_rowmax_[s3] < eff_threshold) {
            ++loc_schwarz; ++loc_considered;
            continue;
          }

          const auto bf3_st = shell_bf_offset_[s3];
          const auto n3 = shell_nbf_[s3];
          const auto P13_nrm = P_shnrm(s1, s3);
          const auto P23_nrm = P_shnrm(s2, s3);

          const size_t sp34_begin = sp_csr_.offsets[s3];
          const size_t sp34_end = sp_csr_.offsets[s3 + 1];
          const size_t s4_max = (s1 == s3) ? s2 : s3;

          for (size_t sp34_idx = sp34_begin; sp34_idx < sp34_end; ++sp34_idx) {
            const size_t s4 = sp_csr_.neighbors[sp34_idx];
            if (s4 > s4_max) break;
            ++loc_considered;

            const auto* sp34_data = &sp_csr_.data[sp34_idx];

            // Distance-dependent Schwarz bound:
            // |(PQ|1/r|RS)| ≤ Q_PQ * Q_RS / sqrt(1 + R²_PQ)
            // Rigorous for large R (Coulomb decay), degenerates to
            // standard Schwarz at R=0.
            const auto schwarz_bound = K_schwarz_(s1, s2) * K_schwarz_(s3, s4);
            {
              const auto& Pc = pair_center_[s1 * nsh_ + s2];
              const auto& Qc = pair_center_[s3 * nsh_ + s4];
              double R2 = 0.0;
              for (int d = 0; d < 3; ++d) {
                double dx = Pc[d] - Qc[d];
                R2 += dx * dx;
              }
              if (schwarz_bound < eff_threshold * std::sqrt(1.0 + R2)) {
                ++loc_schwarz;
                continue;
              }
            }
            const auto P14_nrm = P_shnrm(s1, s4);
            const auto P24_nrm = P_shnrm(s2, s4);
            const auto P34_nrm = P_shnrm(s3, s4);

            double P_screen;
            if (J && K) {
              P_screen = std::max(std::max(P12_nrm, P34_nrm),
                                  std::max({P13_nrm, P14_nrm, P23_nrm, P24_nrm}));
            } else if (J) {
              P_screen = std::max(P12_nrm, P34_nrm);
            } else {
              P_screen = std::max({P13_nrm, P14_nrm, P23_nrm, P24_nrm});
            }
            // Track J-only and K-only screening
            double Pj = std::max(P12_nrm, P34_nrm);
            double Pk = std::max({P13_nrm, P14_nrm, P23_nrm, P24_nrm});
            if (Pj * schwarz_bound < eff_threshold) ++loc_j_screen;
            if (Pk * schwarz_bound < eff_threshold) ++loc_k_screen;
            if (P_screen * schwarz_bound < eff_threshold) { ++loc_density; continue; }

            const auto bf4_st = shell_bf_offset_[s4];
            const auto n4 = shell_nbf_[s4];

            // Mark touched shell pairs (bitmap for dedup, list for iteration)
            if (J_thr) {
              auto idx12 = static_cast<uint32_t>(s1 * nsh + s2);
              auto idx34 = static_cast<uint32_t>(s3 * nsh + s4);
              if (!tj[idx12]) { tj[idx12] = 1; touched_J_list_[tid].push_back(idx12); }
              if (!tj[idx34]) { tj[idx34] = 1; touched_J_list_[tid].push_back(idx34); }
            }
            // Mark touched shell pairs (bitmap for dedup, list for iteration)
            if (K_thr) {
              uint32_t idxs[] = {
                static_cast<uint32_t>(s1*nsh+s3), static_cast<uint32_t>(s1*nsh+s4),
                static_cast<uint32_t>(s2*nsh+s3), static_cast<uint32_t>(s2*nsh+s4)};
              for (auto idx : idxs)
                if (!tk[idx]) { tk[idx] = 1; touched_K_list_[tid].push_back(idx); }
            }

            auto s12_deg = (s1 == s2) ? 1 : 2;
            auto s34_deg = (s3 == s4) ? 1 : 2;
            auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
            auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

            engine.compute2<::libint2::Operator::coulomb,
                            ::libint2::BraKet::xx_xx, 0>(
                obs_[s1], obs_[s2], obs_[s3], obs_[s4], sp12_data, sp34_data);

            const auto buf_1234 = buf[0];
            if (buf_1234 == nullptr) { ++loc_engine_skip; continue; }
            ++loc_computed;

            // Stack-tile contraction: accumulate shell-block contributions
            // in small L1-hot tiles, then flush once to the N² TLS buffer.
            // When both J and K are requested (the common case), a single
            // pass over the integral buffer serves both — halving reads and
            // eliminating redundant value = buf[ijkl] * deg computation.
            constexpr size_t MAX_BF = 15;  // supports up to Cartesian g-shells
            constexpr size_t TILE_THRESHOLD = 4;  // min n3*n4 to justify tiling
            const bool use_tiles = (n1 <= MAX_BF && n2 <= MAX_BF &&
                                    n3 <= MAX_BF && n4 <= MAX_BF &&
                                    n3 * n4 >= TILE_THRESHOLD);

            // Merged J+K contraction (single pass over integral buffer)
            if (J_thr && K_thr) {
              for (size_t idm = 0; idm < ndm; ++idm) {
                auto* J_cur = J_thr + idm * tile_total_size_;
                auto* K_cur = K_thr + idm * tile_total_size_;
                const auto* P_cur = P + idm * N * N;
                // Tiled TLS base pointers for this quartet's shell pairs
                const size_t jt_12 = tile_offset_[s1 * nsh_ + s2];
                const size_t jt_34 = tile_offset_[s3 * nsh_ + s4];
                const size_t kt_13 = tile_offset_[s1 * nsh_ + s3];
                const size_t kt_14 = tile_offset_[s1 * nsh_ + s4];
                const size_t kt_23 = tile_offset_[s2 * nsh_ + s3];
                const size_t kt_24 = tile_offset_[s2 * nsh_ + s4];
                if (use_tiles) {
                  // Pre-load density tiles (from dense P — not tiled)
                  double P_34[MAX_BF * MAX_BF];
                  double P_13[MAX_BF * MAX_BF], P_23[MAX_BF * MAX_BF];
                  double P_14[MAX_BF * MAX_BF], P_24[MAX_BF * MAX_BF];
                  for (size_t k = 0; k < n3; ++k)
                    for (size_t l = 0; l < n4; ++l)
                      P_34[k * n4 + l] = P_cur[(bf3_st + k) * N + (bf4_st + l)];
                  for (size_t i = 0; i < n1; ++i)
                    for (size_t k = 0; k < n3; ++k)
                      P_13[i * n3 + k] = 0.25 * P_cur[(bf1_st + i) * N + (bf3_st + k)];
                  for (size_t j = 0; j < n2; ++j)
                    for (size_t k = 0; k < n3; ++k)
                      P_23[j * n3 + k] = 0.25 * P_cur[(bf2_st + j) * N + (bf3_st + k)];
                  for (size_t i = 0; i < n1; ++i)
                    for (size_t l = 0; l < n4; ++l)
                      P_14[i * n4 + l] = 0.25 * P_cur[(bf1_st + i) * N + (bf4_st + l)];
                  for (size_t j = 0; j < n2; ++j)
                    for (size_t l = 0; l < n4; ++l)
                      P_24[j * n4 + l] = 0.25 * P_cur[(bf2_st + j) * N + (bf4_st + l)];

                  // Accumulator tiles
                  double J_34[MAX_BF * MAX_BF];
                  double K_13[MAX_BF * MAX_BF], K_23[MAX_BF * MAX_BF];
                  double K_14[MAX_BF * MAX_BF], K_24[MAX_BF * MAX_BF];
                  std::memset(J_34, 0, n3 * n4 * sizeof(double));
                  std::memset(K_13, 0, n1 * n3 * sizeof(double));
                  std::memset(K_23, 0, n2 * n3 * sizeof(double));
                  std::memset(K_14, 0, n1 * n4 * sizeof(double));
                  std::memset(K_24, 0, n2 * n4 * sizeof(double));

                  for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                    for (size_t j = 0; j < n2; ++j) {
                      double J_ij = 0.0;
                      const double P_ij = P_cur[(bf1_st + i) * N + (bf2_st + j)];
                      for (size_t k = 0; k < n3; ++k) {
                        double K_ik = 0.0, K_jk = 0.0;
                        const double P_ik = P_13[i * n3 + k];
                        const double P_jk = P_23[j * n3 + k];
                        for (size_t l = 0; l < n4; ++l, ++ijkl) {
                          const auto value = buf_1234[ijkl] * s1234_deg;
                          J_ij += P_34[k * n4 + l] * value;
                          J_34[k * n4 + l] += P_ij * value;
                          K_ik += P_24[j * n4 + l] * value;
                          K_jk += P_14[i * n4 + l] * value;
                          K_14[i * n4 + l] += P_jk * value;
                          K_24[j * n4 + l] += P_ik * value;
                        }
                        K_13[i * n3 + k] += K_ik;
                        K_23[j * n3 + k] += K_jk;
                      }
                      // Flush J_ij to tiled TLS (contiguous)
                      J_cur[jt_12 + i * n2 + j] += J_ij;
                    }
                  }
                  // Flush stack tiles to tiled TLS (all contiguous writes)
                  for (size_t k = 0; k < n3; ++k)
                    for (size_t l = 0; l < n4; ++l)
                      J_cur[jt_34 + k * n4 + l] += J_34[k * n4 + l];
                  for (size_t i = 0; i < n1; ++i)
                    for (size_t k = 0; k < n3; ++k)
                      K_cur[kt_13 + i * n3 + k] += K_13[i * n3 + k];
                  for (size_t j = 0; j < n2; ++j)
                    for (size_t k = 0; k < n3; ++k)
                      K_cur[kt_23 + j * n3 + k] += K_23[j * n3 + k];
                  for (size_t i = 0; i < n1; ++i)
                    for (size_t l = 0; l < n4; ++l)
                      K_cur[kt_14 + i * n4 + l] += K_14[i * n4 + l];
                  for (size_t j = 0; j < n2; ++j)
                    for (size_t l = 0; l < n4; ++l)
                      K_cur[kt_24 + j * n4 + l] += K_24[j * n4 + l];
                } else {
                  // Direct fallback: merged J+K without stack tiles
                  for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                    for (size_t j = 0; j < n2; ++j) {
                      double J_ij = 0.0;
                      const double P_ij = P_cur[(bf1_st + i) * N + (bf2_st + j)];
                      for (size_t k = 0; k < n3; ++k) {
                        double K_ik = 0.0, K_jk = 0.0;
                        const double P_ik = 0.25 * P_cur[(bf1_st + i) * N + (bf3_st + k)];
                        const double P_jk = 0.25 * P_cur[(bf2_st + j) * N + (bf3_st + k)];
                        for (size_t l = 0; l < n4; ++l, ++ijkl) {
                          const auto value = buf_1234[ijkl] * s1234_deg;
                          J_ij += P_cur[(bf3_st + k) * N + (bf4_st + l)] * value;
                          J_cur[jt_34 + k * n4 + l] += P_ij * value;
                          K_ik += 0.25 * P_cur[(bf2_st + j) * N + (bf4_st + l)] * value;
                          K_jk += 0.25 * P_cur[(bf1_st + i) * N + (bf4_st + l)] * value;
                          K_cur[kt_14 + i * n4 + l] += P_jk * value;
                          K_cur[kt_24 + j * n4 + l] += P_ik * value;
                        }
                        K_cur[kt_13 + i * n3 + k] += K_ik;
                        K_cur[kt_23 + j * n3 + k] += K_jk;
                      }
                      J_cur[jt_12 + i * n2 + j] += J_ij;
                    }
                  }
                }
              }
            } else if (J_thr) {
              // J-only contraction (tiled TLS)
              for (size_t idm = 0; idm < ndm; ++idm) {
                auto* J_cur = J_thr + idm * tile_total_size_;
                const auto* P_cur = P + idm * N * N;
                const size_t jt_12 = tile_offset_[s1 * nsh_ + s2];
                const size_t jt_34 = tile_offset_[s3 * nsh_ + s4];
                for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                  for (size_t j = 0; j < n2; ++j) {
                    double J_ij = 0.0;
                    const double P_ij = P_cur[(bf1_st + i) * N + (bf2_st + j)];
                    for (size_t k = 0; k < n3; ++k)
                      for (size_t l = 0; l < n4; ++l, ++ijkl) {
                        const auto value = buf_1234[ijkl] * s1234_deg;
                        J_ij += P_cur[(bf3_st + k) * N + (bf4_st + l)] * value;
                        J_cur[jt_34 + k * n4 + l] += P_ij * value;
                      }
                    J_cur[jt_12 + i * n2 + j] += J_ij;
                  }
                }
              }
            } else if (K_thr) {
              // K-only contraction (tiled TLS)
              for (size_t idm = 0; idm < ndm; ++idm) {
                auto* K_cur = K_thr + idm * tile_total_size_;
                const auto* P_cur = P + idm * N * N;
                const size_t kt_13 = tile_offset_[s1 * nsh_ + s3];
                const size_t kt_14 = tile_offset_[s1 * nsh_ + s4];
                const size_t kt_23 = tile_offset_[s2 * nsh_ + s3];
                const size_t kt_24 = tile_offset_[s2 * nsh_ + s4];
                for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                  for (size_t j = 0; j < n2; ++j) {
                    for (size_t k = 0; k < n3; ++k) {
                      double K_ik = 0.0, K_jk = 0.0;
                      const double P_ik = 0.25 * P_cur[(bf1_st + i) * N + (bf3_st + k)];
                      const double P_jk = 0.25 * P_cur[(bf2_st + j) * N + (bf3_st + k)];
                      for (size_t l = 0; l < n4; ++l, ++ijkl) {
                        const auto value = buf_1234[ijkl] * s1234_deg;
                        K_ik += 0.25 * P_cur[(bf2_st + j) * N + (bf4_st + l)] * value;
                        K_jk += 0.25 * P_cur[(bf1_st + i) * N + (bf4_st + l)] * value;
                        K_cur[kt_14 + i * n4 + l] += P_jk * value;
                        K_cur[kt_24 + j * n4 + l] += P_ik * value;
                      }
                      K_cur[kt_13 + i * n3 + k] += K_ik;
                      K_cur[kt_23 + j * n3 + k] += K_jk;
                    }
                  }
                }
              }
            }

          }  // s4
        }  // s3
      }  // bra-pair (omp for)

      // Parallel cost model: iterate touched lists (O(touched) per thread)
      {
        size_t my_j_sparse = 0, my_k_sparse = 0;
        if (J) {
          for (uint32_t idx : touched_J_list_[tid]) {
            const size_t sa = idx / nsh, sb = idx % nsh;
            my_j_sparse += shell_nbf_[sa] * shell_nbf_[sb];
          }
        }
        if (K) {
          for (uint32_t idx : touched_K_list_[tid]) {
            const size_t sa = idx / nsh, sb = idx % nsh;
            my_k_sparse += shell_nbf_[sa] * shell_nbf_[sb];
          }
        }
#ifdef _OPENMP
#pragma omp atomic
#endif
        j_sparse_total += my_j_sparse * ndm;
#ifdef _OPENMP
#pragma omp atomic
#endif
        k_sparse_total += my_k_sparse * ndm;
      }
      // Accumulate screening counters into class atomics
      diag_considered_ += loc_considered;
      diag_schwarz_ += loc_schwarz;
      diag_density_ += loc_density;
      diag_computed_ += loc_computed;
      diag_engine_skip_ += loc_engine_skip;
      diag_j_would_screen_ += loc_j_screen;
      diag_k_would_screen_ += loc_k_screen;
    }  // parallel region

    ++diag_calls_;
    if (std::getenv("QDK_FOCK_SCREEN_DIAG")) {
      auto total = diag_considered_.load();
      auto sw = diag_schwarz_.load();
      auto ds = diag_density_.load();
      auto comp = diag_computed_.load();
      auto eskip = diag_engine_skip_.load();
      auto jsc = diag_j_would_screen_.load();
      auto ksc = diag_k_would_screen_.load();
      auto screened = sw + ds;
      QDK_LOGGER().info(
          "SCREEN call={} nsh={} eff_thresh={:.1e} P_shmax={:.1e} | "
          "total_quartets={} schwarz={:.1f}% density={:.1f}% "
          "computed={} ({:.1f}%) engine_skip={} | "
          "J_would_screen={:.1f}% K_would_screen={:.1f}%",
          diag_calls_.load(), nsh, eff_threshold, P_shmax, total,
          total > 0 ? 100.0 * sw / total : 0.0,
          total > 0 ? 100.0 * ds / total : 0.0,
          comp, total > 0 ? 100.0 * comp / total : 0.0, eskip,
          total > 0 ? 100.0 * jsc / total : 0.0,
          total > 0 ? 100.0 * ksc / total : 0.0);
    }

    // Fused tile-to-dense reduction: read contiguous tiles from tiled TLS,
    // sum across threads, write to dense N×N output. Source reads are
    // unit-stride (tile-contiguous); dest writes are N-strided but happen
    // only once per tile element per thread.
    if (J) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
      for (size_t sa = 0; sa < nsh; ++sa)
        for (size_t sb = 0; sb < nsh; ++sb) {
          const size_t na = shell_nbf_[sa], nb = shell_nbf_[sb];
          const size_t tile_off = tile_offset_[sa * nsh_ + sb];
          const size_t bf_a = shell_bf_offset_[sa];
          const size_t bf_b = shell_bf_offset_[sb];
          for (int t = 0; t < nthreads_; ++t) {
            if (!touched_J_[t][sa * nsh + sb]) continue;
            for (size_t idm = 0; idm < ndm; ++idm) {
              const double* src = J_tls_[t].data() + idm * tile_total_size_ + tile_off;
              double* dst_base = J + idm * N * N;
              for (size_t a = 0; a < na; ++a)
                for (size_t b = 0; b < nb; ++b)
                  dst_base[(bf_a + a) * N + (bf_b + b)] += src[a * nb + b];
            }
          }
        }
    }
    if (K) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) collapse(2)
#endif
      for (size_t sa = 0; sa < nsh; ++sa)
        for (size_t sb = 0; sb < nsh; ++sb) {
          const size_t na = shell_nbf_[sa], nb = shell_nbf_[sb];
          const size_t tile_off = tile_offset_[sa * nsh_ + sb];
          const size_t bf_a = shell_bf_offset_[sa];
          const size_t bf_b = shell_bf_offset_[sb];
          for (int t = 0; t < nthreads_; ++t) {
            if (!touched_K_[t][sa * nsh + sb]) continue;
            for (size_t idm = 0; idm < ndm; ++idm) {
              const double* src = K_tls_[t].data() + idm * tile_total_size_ + tile_off;
              double* dst_base = K + idm * N * N;
              for (size_t a = 0; a < na; ++a)
                for (size_t b = 0; b < nb; ++b)
                  dst_base[(bf_a + a) * N + (bf_b + b)] += src[a * nb + b];
            }
          }
        }
    }

    first_build_jk_ = false;

    // Symmetrize J (parallel)
    if (J)
      for (size_t idm = 0; idm < ndm; ++idm) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < N; ++i)
          for (size_t j = 0; j <= i; ++j) {
            auto v = J[idm * N * N + i * N + j];
            auto w = J[idm * N * N + j * N + i];
            v = 0.25 * (v + w);
            J[idm * N * N + i * N + j] = v;
            J[idm * N * N + j * N + i] = v;
          }
      }

    // Symmetrize K + scale (parallel)
    if (K)
      for (size_t idm = 0; idm < ndm; ++idm) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
        for (size_t i = 0; i < N; ++i)
          for (size_t j = 0; j <= i; ++j) {
            auto v = K[idm * N * N + i * N + j];
            auto w = K[idm * N * N + j * N + i];
            v = (alpha + beta) * 0.5 * (v + w);
            K[idm * N * N + i * N + j] = v;
            K[idm * N * N + j * N + i] = v;
          }
      }
  }

  /**
   * @brief Compute nuclear gradients for Coulomb and exchange energies
   *
   * Computes the contribution to nuclear gradients from the Coulomb and
   * exchange energies using direct evaluation of derivative integrals. This
   * involves computing nuclear derivatives of two-electron integrals and
   * contracting with density matrices to obtain energy gradient contributions:
   * ∂E_J/∂R and ∂E_K/∂R.
   *
   * @param P Density matrix in AO basis
   * @param dJ Output nuclear gradient contribution from Coulomb energy
   * @param dK Output nuclear gradient contribution from exchange energy
   * @param alpha Scaling factor for Hartree-Fock exchange
   * @param beta Scaling factor for DFT exchange
   * @param omega Range-separation parameter
   *
   * @throws std::runtime_error Always - energy gradients not yet implemented
   *
   * @note These are energy derivatives, not matrix element derivatives
   */
  void get_gradients(const double* P, double* dJ, double* dK, double alpha,
                     double beta, double omega) {
    QDK_LOG_TRACE_ENTERING();

    throw std::runtime_error("LIBINT2_DIRECT + Gradients Not Yet Implemented");
  }

  /**
   * @brief Perform quarter transformation of two-electron integrals
   *
   * Transforms the first index of the four-center integrals from AO to MO
   * basis: (pj|kl) = Σᵢ C(i,p) × (ij|kl)
   *
   * This is the first step in integral transformations for post-HF methods.
   * The implementation uses the same screening as build_JK but performs the
   * transformation during integral evaluation to minimize memory usage.
   *
   * @param nt Number of transformed orbitals (size of MO space)
   * @param C Transformation matrix C(AO,MO) in row-major format
   * @param out Output buffer for transformed integrals (pj|kl)
   *           Layout: out[p + nt*(l + num_atomic_orbitals*(j +
   * num_atomic_orbitals*k))] for (kj|lp)
   *
   * @note First index (i) is transformed to MO index (p): i → p
   * @note Output tensor has mixed AO/MO indices with p as fastest index
   * @note Uses same shell pair and Schwarz screening as build_JK
   * @note Result includes proper symmetrization for shell permutations
   */
  void quarter_trans(size_t nt, const double* C, double* out) {
    QDK_LOG_TRACE_ENTERING();

    AutoTimer t("ERI::quarter_trans");
    const size_t num_atomic_orbitals = obs_.nbf();
    const size_t nsh = obs_.size();

    // Clear output tensor: (ij|kp) with p as fast index
    const size_t out_size =
        nt * num_atomic_orbitals * num_atomic_orbitals * num_atomic_orbitals;
    std::memset(out, 0, out_size * sizeof(double));
    const size_t inner_size = nt * num_atomic_orbitals;

    // No shell block norm needed for quarter transformation
    // We'll use Schwarz screening directly

    // Setup required precision for libint2 engine
    const auto engine_precision = std::numeric_limits<double>::epsilon();

    // Update engine precision for quarter transformation
    for (int i = 0; i < nthreads_; ++i)
      engines_[i].set_precision(engine_precision);

    // Thread-local accumulation buffers for reproducibility
    std::vector<std::vector<double>> out_local(0);
    if (use_thread_local_buffers_) {
      out_local.resize(nthreads_);
      for (int t = 0; t < nthreads_; ++t) {
        out_local[t].resize(out_size, 0.0);
      }
    }

#ifdef _OPENMP
#pragma omp parallel
#endif
    {
#ifdef _OPENMP
      const auto thread_id = omp_get_thread_num();
#else
      const auto thread_id = 0;
#endif
      auto& engine = engines_[thread_id];
      const auto& buf = engine.results();

      // Get pointer to thread-local buffer
      double* out_thread = nullptr;
      if (use_thread_local_buffers_) {
        out_thread = out_local[thread_id].data();
      }

      for (size_t s1 = 0ul, s1234 = 0ul; s1 < nsh; ++s1) {
        const auto bf1_st = shell_bf_offset_[s1];
        const auto n1 = shell_nbf_[s1];

        const size_t sp12_begin = sp_csr_.offsets[s1];
        const size_t sp12_end = sp_csr_.offsets[s1 + 1];

        // Only loop over significant shell pairs (CSR row for s1)
        for (size_t sp12_idx = sp12_begin; sp12_idx < sp12_end; ++sp12_idx) {
          const size_t s2 = sp_csr_.neighbors[sp12_idx];
          const auto bf2_st = shell_bf_offset_[s2];
          const auto n2 = shell_nbf_[s2];
          const auto* sp12_data = &sp_csr_.data[sp12_idx];

          // Permutational Degeneracy
          auto s12_deg = (s1 == s2) ? 1 : 2;

          for (size_t s3 = 0; s3 <= s1; ++s3) {
            const auto bf3_st = shell_bf_offset_[s3];
            const auto n3 = shell_nbf_[s3];

            const size_t sp34_begin = sp_csr_.offsets[s3];
            const size_t sp34_end = sp_csr_.offsets[s3 + 1];
            const size_t s4_max = (s1 == s3) ? s2 : s3;

            // Only loop over significant shell pairs (CSR row for s3)
            for (size_t sp34_idx = sp34_begin; sp34_idx < sp34_end; ++sp34_idx) {
              const size_t s4 = sp_csr_.neighbors[sp34_idx];
              if (s4 > s4_max) break;

              const auto* sp34_data = &sp_csr_.data[sp34_idx];

              // Assign to threads
              if ((s1234++) % nthreads_ != thread_id) continue;
              if (K_schwarz_(s1, s2) * K_schwarz_(s3, s4) < eri_threshold_)
                continue;

              const auto bf4_st = shell_bf_offset_[s4];
              const auto n4 = shell_nbf_[s4];

              // Compute the integral shell quartet
              engine.compute2<::libint2::Operator::coulomb,
                              ::libint2::BraKet::xx_xx, 0>(
                  obs_[s1], obs_[s2], obs_[s3], obs_[s4], sp12_data, sp34_data);

              // Coarse integral screening
              const auto buf_1234 = buf[0];
              if (buf_1234 == nullptr) continue;

              auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1 : 2) : 2;
              auto s34_deg = (s3 == s4) ? 1 : 2;

              // Perform quarter transformation: (pn|lk) = sum_m C(p,m) *
              // (mn|lk) Here: m=s1i, n=s2j, l=s3k, k=s4l Output layout: p is
              // fast index, so out[p + nt*(k + num_atomic_orbitals*(j +
              // num_atomic_orbitals*i))]
              for (size_t i = 0, ijkl = 0; i < n1; ++i) {
                const size_t bf1 = bf1_st + i;
                for (size_t j = 0; j < n2; ++j) {
                  const size_t bf2 = bf2_st + j;
                  for (size_t k = 0; k < n3; ++k) {
                    const size_t bf3 = bf3_st + k;
                    for (size_t l = 0; l < n4; ++l, ++ijkl) {
                      const size_t bf4 = bf4_st + l;

                      const auto value = buf_1234[ijkl] * s12_34_deg;
                      // p and l are fast indices separately
                      for (size_t p = 0; p < nt; ++p) {
                        // (ij|kp) = \sum_l (ij|kl) * C(l,p)
                        const size_t idx1 =
                            (bf1 * num_atomic_orbitals + bf2) * inner_size +
                            bf3 * nt + p;
                        if (use_thread_local_buffers_) {
                          out_thread[idx1] += C[bf4 * nt + p] * value * s12_deg;
                        } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                          out[idx1] += C[bf4 * nt + p] * value * s12_deg;
                        }

                        // (ij|lp) = \sum_k (ij|lk) * C(k,p) = \sum_l (ij|kl) *
                        // C(k,p)
                        if (s3 != s4) {
                          const size_t idx2 =
                              (bf1 * num_atomic_orbitals + bf2) * inner_size +
                              bf4 * nt + p;
                          if (use_thread_local_buffers_) {
                            out_thread[idx2] +=
                                C[bf3 * nt + p] * value * s12_deg;
                          } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                            out[idx2] += C[bf3 * nt + p] * value * s12_deg;
                          }
                        }

                        // (kl|ip) = \sum_j (kl|ij) * C(j,p)
                        const size_t idx3 =
                            (bf3 * num_atomic_orbitals + bf4) * inner_size +
                            bf1 * nt + p;
                        if (use_thread_local_buffers_) {
                          out_thread[idx3] += C[bf2 * nt + p] * value * s34_deg;
                        } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                          out[idx3] += C[bf2 * nt + p] * value * s34_deg;
                        }

                        // (kl|jp) = \sum_i (kl|ji) * C(i,p)
                        if (s1 != s2) {
                          const size_t idx4 =
                              (bf3 * num_atomic_orbitals + bf4) * inner_size +
                              bf2 * nt + p;
                          if (use_thread_local_buffers_) {
                            out_thread[idx4] +=
                                C[bf1 * nt + p] * value * s34_deg;
                          } else {
#ifdef _OPENMP
#pragma omp atomic update relaxed
#endif
                            out[idx4] += C[bf1 * nt + p] * value * s34_deg;
                          }
                        }
                      }
                    }  // l (bf4)
                  }  // k (bf3)
                }  // j (bf2)
              }  // i (bf1)

            }  // s4
          }  // s3
        }  // s2
      }  // s1
    }

    // Deterministic reduction: combine thread-local buffers in order
    if (use_thread_local_buffers_) {
      for (int t = 0; t < nthreads_; ++t) {
        for (size_t i = 0; i < out_size; ++i) {
          out[i] += out_local[t][i];
        }
      }
    }

// Symmetrize the first and second index: (ij|kp) = 0.5 * ( (ji|kp) + (ij|kp) ),
// then cut half due to s12_34_deg
#ifdef _OPENMP
#pragma omp parallel for collapse(2)
#endif
    for (size_t i = 0; i < num_atomic_orbitals; ++i)
      for (size_t j = 0; j <= i; ++j) {
        for (size_t k = 0; k < num_atomic_orbitals; ++k) {
          for (size_t p = 0; p < nt; ++p) {
            const size_t idx_ij =
                (i * num_atomic_orbitals + j) * inner_size + k * nt + p;
            const size_t idx_ji =
                (j * num_atomic_orbitals + i) * inner_size + k * nt + p;
            const double a = out[idx_ij];
            const double b = out[idx_ji];
            const double avg =
                0.25 * (a + b);  // average and then cut half due to s12_34_deg
            out[idx_ij] = avg;
            out[idx_ji] = avg;
          }
        }
      }
  }

  /**
   * @brief Factory method to create Libint2 direct ERI instance
   *
   * Template factory method that forwards arguments to the ERI constructor.
   * Provides a clean interface for creating ERI instances with perfect
   * forwarding of constructor arguments.
   *
   * @tparam Args Parameter pack for constructor arguments
   * @param args Arguments to forward to ERI constructor
   * @return Unique pointer to new Libint2 direct ERI instance
   */
  template <typename... Args>
  static std::unique_ptr<ERI> make_libint2_direct_eri(Args&&... args) {
    QDK_LOG_TRACE_ENTERING();

    return std::make_unique<ERI>(std::forward<Args>(args)...);
  }
};

}  // namespace libint2::direct

LIBINT2_DIRECT::LIBINT2_DIRECT(SCFOrbitalType scf_orbital_type,
                               BasisSet& basis_set, ParallelConfig _mpi,
                               bool use_atomics, double eri_threshold,
                               double shell_pair_threshold)
    : ERI(scf_orbital_type, eri_threshold, basis_set, _mpi),
      eri_impl_(libint2::direct::ERI::make_libint2_direct_eri(
          scf_orbital_type == SCFOrbitalType::Restricted ? 1 : 2, basis_set,
          use_atomics, eri_threshold, shell_pair_threshold)) {
  QDK_LOG_TRACE_ENTERING();
  if (_mpi.world_size > 1) throw std::runtime_error("LIBINT2_DIRECT + MPI NYI");
}

LIBINT2_DIRECT::~LIBINT2_DIRECT() noexcept = default;

// Public interface implementation - delegates to internal ERI implementation
void LIBINT2_DIRECT::build_JK(const double* P, double* J, double* K,
                              double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();
  if (!eri_impl_) throw std::runtime_error("LIBINT2_DIRECT NOT INITIALIZED");
  eri_impl_->build_JK(P, J, K, alpha, beta, omega);
}

// Override implementation for ERI base class
void LIBINT2_DIRECT::build_JK_impl_(const double* P, double* J, double* K,
                                    double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();
  if (!eri_impl_) throw std::runtime_error("LIBINT2_DIRECT NOT INITIALIZED");
  eri_impl_->build_JK(P, J, K, alpha, beta, omega);
}

// Gradient computation interface
void LIBINT2_DIRECT::get_gradients(const double* P, double* dJ, double* dK,
                                   double alpha, double beta, double omega) {
  QDK_LOG_TRACE_ENTERING();
  if (!eri_impl_) throw std::runtime_error("LIBINT2_DIRECT NOT INITIALIZED");
  eri_impl_->get_gradients(P, dJ, dK, alpha, beta, omega);
}

// Quarter transformation interface
void LIBINT2_DIRECT::quarter_trans_impl(size_t nt, const double* C,
                                        double* out) {
  QDK_LOG_TRACE_ENTERING();
  if (!eri_impl_) throw std::runtime_error("LIBINT2_DIRECT NOT INITIALIZED");
  eri_impl_->quarter_trans(nt, C, out);
};

void LIBINT2_DIRECT::set_screening_threshold(double threshold) {
  ERI::set_screening_threshold(threshold);
  if (eri_impl_) eri_impl_->set_eri_threshold(threshold);
}

}  // namespace qdk::chemistry::scf

/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <algorithm>
#include <array>
#include <blas.hh>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <lapack.hh>
#include <limits>
#include <macis/fci/fci_hamiltonian.hpp>
#include <macis/solvers/davidson.hpp>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include "fci_kernels.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#if defined(__SSE2__) || defined(_M_X64)
#include <xmmintrin.h>
#endif
#if defined(__linux__)
#include <sys/mman.h>
#endif

namespace macis {

namespace {
// Beta strings per GEMM of the alpha-beta pass: as many as keep a task's
// products (row_group x npair x block elements) within about 512 KiB, i.e.
// L2 resident together with its sigma rows, capped at 256 (measured best
// for 16 orbitals; 64 doubled the gather time)
template <typename T>
size_t beta_block_size(size_t row_group, size_t npair) {
  const size_t fit = (size_t(1) << 19) / (row_group * npair * sizeof(T));
  return std::clamp<size_t>(fit & ~size_t(15), 32, 256);
}

// Columns of the same-spin tiles: whole cache lines per string, so that each
// loaded matrix element gets at least a cache line of FMAs; wider tiles (see
// tile_task_lines) give it more
using fci_detail::sign_bit;
using fci_detail::tile_width;

/// Widths, in cache lines, of the column tasks of a same-spin pass over
/// ntiles cache lines of columns on nt threads, widest first. Wider tiles
/// cost less per line (2 lines 0.6-0.7x, 4 lines 0.5-0.6x of a single line
/// on Cascade Lake) but leave fewer tasks: full rounds of nt tasks of
/// max_lines, then one round of at most nt tasks as even as possible
/// (a width of 3 split into 2 + 1)
std::vector<size_t> tile_task_lines(size_t ntiles, size_t nt,
                                    size_t max_lines) {
  const size_t full = ntiles / (max_lines * nt);
  std::vector<size_t> lines(full * nt, max_lines);
  const size_t rest = ntiles - full * nt * max_lines;
  for (size_t t = 0; t < nt; ++t) {
    size_t w = rest / nt + (t < rest % nt);
    if (w == 3) {
      lines.push_back(2);
      w = 1;
    }
    if (w > 0) lines.push_back(w);
  }
  std::sort(lines.begin() + full * nt, lines.end(), std::greater<>());
  return lines;
}

// Below this many determinants FCIPrecision::automatic stays in double
// precision: measured slower mixed at 8.5e5 (H12), faster at 2.9e6 (H13)
constexpr size_t mixed_precision_min_size = size_t(1) << 21;

// From this many determinants the P-space is also an exact block of the
// Davidson preconditioner: its eigendecomposition (about 25 ms) is then
// cheaper than the iteration it saves (57 ms for 1.2e7 determinants, 18 ms
// for 2.9e6)
constexpr size_t block_preconditioner_min_size = size_t(1) << 23;

#ifdef MACIS_FCI_PROFILE
// Development builds only: per-pass sigma timings on stderr
inline double profile_clock() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
#define MACIS_FCI_TIC(t) const double t = profile_clock()
#define MACIS_FCI_TOC(acc, t) acc += profile_clock() - t
#else
#define MACIS_FCI_TIC(t)
#define MACIS_FCI_TOC(acc, t)
#endif

/// In single precision, flushes denormal results to zero and reads denormal
/// operands as zero in the calling thread while alive (the MXCSR is per
/// thread, so every parallel region sets it). Denormals carry nothing at
/// single precision, but on x86 every operation producing or reading one
/// takes a microcode assist; they arise in converged vectors, as products of
/// vanishing (e.g. symmetry-forbidden) coefficients with integrals that are
/// zero up to rounding, and were measured to slow the single-precision GEMM
/// down fivefold.
template <typename T>
struct FlushDenormals {
#if defined(__SSE2__) || defined(_M_X64)
  unsigned int saved = 0;
  FlushDenormals() {
    if constexpr (std::is_same_v<T, float>) {
      saved = _mm_getcsr();
      _mm_setcsr(saved | 0x8040u);  // FTZ | DAZ
    }
  }
  ~FlushDenormals() {
    if constexpr (std::is_same_v<T, float>) _mm_setcsr(saved);
  }
#endif
};

/// Row of the same-spin pair matrix for the orbital pair p < r
inline size_t distinct_pair_index(size_t p, size_t r) {
  return r * (r - 1) / 2 + p;
}

/// v = n zeros. A large fresh buffer is reserved and advised for transparent
/// huge pages before it is zeroed: the zeroing is serial (std::vector), and
/// with 4 KiB pages its page faults dominated (0.86 s for 1.3 GB)
void assign_zeros(std::vector<double>& v, size_t n) {
  if (v.capacity() < n) {
    std::vector<double>().swap(v);
    v.reserve(n);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
    constexpr uintptr_t page = 4096;
    const auto begin = reinterpret_cast<uintptr_t>(v.data()) & ~(page - 1);
    const auto end = reinterpret_cast<uintptr_t>(v.data() + n);
    madvise(reinterpret_cast<void*>(begin), end - begin, MADV_HUGEPAGE);
#endif
  }
  v.assign(n, 0.0);
}

template <typename T>
T* cache_line_aligned(T* p) {
  return reinterpret_cast<T*>((reinterpret_cast<uintptr_t>(p) + 63) &
                              ~uintptr_t(63));
}

/// Threads for a parallel region doing `flops` of work: one per 2e6 flops
/// (about 0.1 ms), so that small spaces are not dominated by thread start-up
int threads_for(double flops) {
#ifdef _OPENMP
  const double max_threads = omp_get_max_threads();
#else
  const double max_threads = 1.0;
#endif
  return int(std::clamp(std::ceil(flops / 2e6), 1.0, max_threads));
}

/// While alive, OpenMP-threaded BLAS/LAPACK calls of the calling thread run
/// serially: for the 400 x 400 P-space matrix, 64 MKL threads took 14 ms
/// instead of 6 (syevr) and up to 170 ms instead of 23 (syevd of a spin-flip
/// sector matrix); threading only pays from about 1000 x 1000
struct SerialLapack {
#ifdef _OPENMP
  int saved = omp_get_max_threads();
  SerialLapack() { omp_set_num_threads(1); }
  ~SerialLapack() { omp_set_num_threads(saved); }
#endif
};

}  // namespace

struct FCIHamiltonian::Workspace {
  std::mutex mutex;
  std::vector<detail::davidson_buffer<char>> buffers;  ///< per thread
  std::vector<size_t> bytes;
};

/// Work arrays for the threads of one sigma call: the Hamiltonian's cached
/// ones unless a concurrent call holds them, else arrays of this call
class FCIHamiltonian::Scratch {
 public:
  explicit Scratch(Workspace& workspace)
      : lock_(workspace.mutex, std::try_to_lock),
        buffers_(lock_.owns_lock() ? workspace.buffers : local_buffers_),
        bytes_(lock_.owns_lock() ? workspace.bytes : local_bytes_) {
#ifdef _OPENMP
    const size_t nt = omp_get_max_threads();
#else
    const size_t nt = 1;
#endif
    if (buffers_.size() < nt) {
      buffers_.resize(nt);
      bytes_.resize(nt, 0);
    }
  }

  /// n elements of T on 64-byte aligned memory of the calling OpenMP thread,
  /// whose contents are not kept from one request to the next
  template <typename T>
  T* get(size_t n) {
#ifdef _OPENMP
    const size_t t = omp_get_thread_num();
#else
    const size_t t = 0;
#endif
    if (bytes_[t] < n * sizeof(T)) {
      buffers_[t] = detail::davidson_alloc<char>(n * sizeof(T));
      bytes_[t] = n * sizeof(T);
    }
    return reinterpret_cast<T*>(buffers_[t].get());
  }

 private:
  std::unique_lock<std::mutex> lock_;
  std::vector<detail::davidson_buffer<char>> local_buffers_;
  std::vector<size_t> local_bytes_;
  std::vector<detail::davidson_buffer<char>>& buffers_;
  std::vector<size_t>& bytes_;
};

FCIHamiltonian::FCIHamiltonian(size_t norb, size_t nalpha, size_t nbeta,
                               const double* T, size_t LDT, const double* V,
                               size_t LDV)
    : norb_(norb),
      npair_(norb * (norb + 1) / 2),
      workspace_(std::make_shared<Workspace>()) {
  alpha_ = std::make_shared<const FCIStringSpace>(norb, nalpha);
  beta_ = nbeta == nalpha ? alpha_
                          : std::make_shared<const FCIStringSpace>(norb, nbeta);

  const size_t LDV2 = LDV * LDV;
  const size_t LDV3 = LDV2 * LDV;
  auto v = [&](size_t p, size_t q, size_t r, size_t s) {
    return V[p + q * LDV + r * LDV2 + s * LDV3];
  };

  vpair_.resize(npair_ * npair_);
  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q <= p; ++q)
      for (size_t r = 0; r < norb; ++r)
        for (size_t s = 0; s <= r; ++s)
          vpair_[fci_pair_index(p, q) * npair_ + fci_pair_index(r, s)] =
              v(p, q, r, s);
  vpair_f_.assign(vpair_.begin(), vpair_.end());

  coulomb_.resize(norb * norb);
  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q < norb; ++q) coulomb_[p * norb + q] = v(p, p, q, q);

  h_alpha_ = make_same_spin_operator(*alpha_, T, LDT, V, LDV);
  h_beta_ = beta_ == alpha_ ? h_alpha_
                            : make_same_spin_operator(*beta_, T, LDT, V, LDV);

  // Compact beta links of the alpha-beta scatter (8 bytes per excitation
  // instead of 16) and the diagonal pairs of every beta string
  const size_t nbs = beta_->size();
  const size_t nexc = beta_->num_excitations();
  const size_t nocc = beta_->nelec();
  beta_link_target_.resize(nbs * nexc);
  beta_link_pair_.resize(nbs * nexc);
  beta_diag_pair_.resize(nbs * nocc);
#pragma omp parallel for schedule(static) num_threads(threads_for(nbs* nexc))
  for (size_t ib = 0; ib < nbs; ++ib) {
    const auto* ex = beta_->excitations(ib);
    for (size_t e = 0; e < nexc; ++e) {
      beta_link_target_[ib * nexc + e] = ex[e].target;
      beta_link_pair_[ib * nexc + e] =
          ex[e].pair | (ex[e].sign < 0 ? sign_bit : 0u);
    }
    const uint32_t* occ = beta_->occupied(ib);
    for (size_t q = 0; q < nocc; ++q)
      beta_diag_pair_[ib * nocc + q] = uint32_t(fci_pair_index(occ[q], occ[q]));
  }
}

std::shared_ptr<const FCIHamiltonian::SameSpinOperator>
FCIHamiltonian::make_same_spin_operator(const FCIStringSpace& space,
                                        const double* T, size_t LDT,
                                        const double* V, size_t LDV) {
  const size_t norb = space.norb();
  const size_t k = space.nelec();
  const size_t ns = space.size();
  const size_t LDV2 = LDV * LDV;
  const size_t LDV3 = LDV2 * LDV;
  auto h = [&](size_t p, size_t q) { return T[p + q * LDT]; };
  auto v = [&](size_t p, size_t q, size_t r, size_t s) {
    return V[p + q * LDV + r * LDV2 + s * LDV3];
  };

  auto op = std::make_shared<SameSpinOperator>();
  op->diagonal.resize(ns);
#pragma omp parallel for schedule(static) num_threads(threads_for(ns* k* k))
  for (size_t i = 0; i < ns; ++i) {
    const uint32_t* occ = space.occupied(i);
    double d = 0.0;
    for (size_t a = 0; a < k; ++a) {
      d += h(occ[a], occ[a]);
      for (size_t b = 0; b < a; ++b)
        d += v(occ[a], occ[a], occ[b], occ[b]) -
             v(occ[a], occ[b], occ[b], occ[a]);
    }
    op->diagonal[i] = d;
  }

  if (k == 0) return op;
  if (k == 1) {
    // One-electron strings are the orbitals themselves and H^ss = h
    op->clique_size = norb;
    op->pair_dim = norb;
    op->W.resize(norb * norb);
    op->string.resize(norb);
    op->pair.resize(norb);
    for (size_t p = 0; p < norb; ++p) {
      for (size_t q = 0; q < norb; ++q) op->W[p * norb + q] = h(p, q);
      op->string[p] = p;
      op->pair[p] = p;
    }
    op->W_f.assign(op->W.begin(), op->W.end());
    return op;
  }

  // W(pr, qs) = (pq|rs) - (ps|rq) + [h_pq d_rs - h_rq d_ps - h_ps d_rq
  // + h_rs d_pq] / (k - 1) for p < r, q < s
  const size_t npd = norb * (norb - 1) / 2;
  const double f = 1.0 / double(k - 1);
  op->pair_dim = npd;
  op->W.resize(npd * npd);
  for (size_t r = 1; r < norb; ++r)
    for (size_t p = 0; p < r; ++p)
      for (size_t s = 1; s < norb; ++s)
        for (size_t q = 0; q < s; ++q) {
          double w = v(p, q, r, s) - v(p, s, r, q);
          if (r == s) w += f * h(p, q);
          if (p == s) w -= f * h(r, q);
          if (r == q) w -= f * h(p, s);
          if (p == q) w += f * h(r, s);
          op->W[distinct_pair_index(p, r) * npd + distinct_pair_index(q, s)] =
              w;
        }
  op->W_f.assign(op->W.begin(), op->W.end());

  // Cliques of the (k-2)-electron strings K in lexicographic order; member
  // (q, s) is K + {q, s} with sign (-1)^(#K below q + #K below s), the sign
  // of a+_q a+_s |K>
  const size_t nk = k - 2;
  const size_t nfree = norb - nk;
  const size_t m = nfree * (nfree - 1) / 2;
  std::vector<uint32_t> cliques;
  std::vector<uint32_t> occ(nk);
  std::iota(occ.begin(), occ.end(), 0u);
  size_t ncl = 0;
  while (true) {
    cliques.insert(cliques.end(), occ.begin(), occ.end());
    ++ncl;
    size_t i = nk;
    while (i > 0 and occ[i - 1] == norb - nk + i - 1) --i;
    if (i == 0) break;
    ++occ[i - 1];
    for (size_t j = i; j < nk; ++j) occ[j] = occ[j - 1] + 1;
  }

  op->clique_size = m;
  op->string.resize(ncl * m);
  op->pair.resize(ncl * m);
#pragma omp parallel num_threads(threads_for(double(ncl) * m * k))
  {
    std::vector<uint32_t> member(k), free_orbs, below(norb);
#pragma omp for schedule(static)
    for (size_t K = 0; K < ncl; ++K) {
      const uint32_t* kocc = cliques.data() + K * nk;
      free_orbs.clear();
      for (uint32_t o = 0, a = 0; o < norb; ++o) {
        below[o] = a;
        if (a < nk and kocc[a] == o)
          ++a;
        else
          free_orbs.push_back(o);
      }
      size_t e = K * m;
      for (size_t i = 0; i < nfree; ++i)
        for (size_t j = i + 1; j < nfree; ++j, ++e) {
          const uint32_t add[2] = {free_orbs[i], free_orbs[j]};
          for (size_t x = 0, a = 0, b = 0; x < k; ++x)
            member[x] =
                b < 2 and (a == nk or add[b] < kocc[a]) ? add[b++] : kocc[a++];
          op->string[e] = space.address(member.data());
          op->pair[e] = uint32_t(distinct_pair_index(add[0], add[1])) |
                        ((below[add[0]] + below[add[1]]) & 1 ? sign_bit : 0u);
        }
    }
  }
  return op;
}

double FCIHamiltonian::same_spin_element(const SameSpinOperator& op,
                                         const FCIStringSpace& space, size_t I,
                                         size_t J) {
  if (I == J) return op.diagonal[I];
  const size_t k = space.nelec();
  const uint32_t* oi = space.occupied(I);
  const uint32_t* oj = space.occupied(J);
  if (k == 1) return op.W[oi[0] * op.pair_dim + oj[0]];

  // Orbitals only in I / only in J (at most two each for a nonzero element)
  uint32_t xi[2], xj[2];
  size_t ni = 0, nj = 0;
  for (size_t a = 0, b = 0; a < k or b < k;) {
    if (b == k or (a < k and oi[a] < oj[b])) {
      if (ni == 2) return 0.0;
      xi[ni++] = oi[a++];
    } else if (a == k or oj[b] < oi[a]) {
      if (nj == 2) return 0.0;
      xj[nj++] = oj[b++];
    } else {
      ++a;
      ++b;
    }
  }
  // Common orbitals below x
  auto below = [&](uint32_t x) {
    size_t c = 0;
    for (size_t a = 0; a < k; ++a)
      if (oi[a] < x and oi[a] != xi[0] and (ni < 2 or oi[a] != xi[1])) ++c;
    return c;
  };
  const size_t npd = op.pair_dim;

  if (ni == 2) {
    const size_t parity =
        below(xi[0]) + below(xi[1]) + below(xj[0]) + below(xj[1]);
    const double w = op.W[distinct_pair_index(xi[0], xi[1]) * npd +
                          distinct_pair_index(xj[0], xj[1])];
    return parity & 1 ? -w : w;
  }

  // Single difference x -> y: one clique per common orbital z (K = common - z)
  const uint32_t x = xi[0], y = xj[0];
  const size_t cxy = below(x) + below(y);
  double sum = 0.0;
  for (size_t a = 0; a < k; ++a) {
    const uint32_t z = oi[a];
    if (z == x) continue;
    const size_t pa =
        z < x ? distinct_pair_index(z, x) : distinct_pair_index(x, z);
    const size_t pb =
        z < y ? distinct_pair_index(z, y) : distinct_pair_index(y, z);
    const double w = op.W[pa * npd + pb];
    sum += (cxy + (z < x) + (z < y)) & 1 ? -w : w;
  }
  return sum;
}

void FCIHamiltonian::diagonal(double* D) const {
  const auto& sa = *alpha_;
  const auto& sb = *beta_;
  const size_t nb = sb.size();

#pragma omp parallel num_threads(threads_for(double(size()) * (sb.nelec() + 2)))
  {
    // Per-thread scratch on cache lines of its own (no false sharing)
    std::vector<double> coulomb_buffer(norb_ + 16);
    double* coulomb_row = cache_line_aligned(coulomb_buffer.data());
#pragma omp for schedule(static)
    for (size_t ia = 0; ia < sa.size(); ++ia) {
      std::fill_n(coulomb_row, norb_, 0.0);
      const uint32_t* occa = sa.occupied(ia);
      for (size_t a = 0; a < sa.nelec(); ++a)
        for (size_t q = 0; q < norb_; ++q)
          coulomb_row[q] += coulomb_[occa[a] * norb_ + q];

      for (size_t ib = 0; ib < nb; ++ib) {
        const uint32_t* occb = sb.occupied(ib);
        double d = h_alpha_->diagonal[ia] + h_beta_->diagonal[ib];
        for (size_t b = 0; b < sb.nelec(); ++b) d += coulomb_row[occb[b]];
        D[ia * nb + ib] = d;
      }
    }
  }
}

void FCIHamiltonian::set_simd(FCISimd level) {
  if (int(level) < int(FCISimd::generic) or
      int(level) > int(fci_simd_supported()))
    throw std::invalid_argument(
        "FCIHamiltonian::set_simd: kernel level not supported here");
  simd_ = level;
}

void FCIHamiltonian::sigma(const double* C, double* S, double alpha,
                           double beta, int spin_flip) const {
  sigma_impl<double>(C, S, alpha, beta, spin_flip);
}

void FCIHamiltonian::sigma(const float* C, float* S, float alpha, float beta,
                           int spin_flip) const {
  sigma_impl<float>(C, S, alpha, beta, spin_flip);
}

template <typename T>
void FCIHamiltonian::sigma_impl(const T* C, T* S, T alpha, T beta,
                                int spin_flip) const {
  if (spin_flip < -1 or spin_flip > 1)
    throw std::invalid_argument(
        "FCIHamiltonian::sigma: spin_flip must be -1, 0 or 1");
  if (spin_flip != 0 and alpha_ != beta_)
    throw std::invalid_argument(
        "FCIHamiltonian::sigma: spin-flip symmetry requires equal numbers of "
        "alpha and beta electrons");
  const auto& sa = *alpha_;
  const auto& sb = *beta_;
  const size_t na = sa.size();
  const size_t nb = sb.size();
  const size_t nexc_a = sa.num_excitations();
  const size_t nexc_b = sb.num_excitations();
  // Inner GEMM dimension: alpha excitations plus one merged diagonal column
  const size_t K = nexc_a + 1;
  const bool mixed = sa.nelec() > 0 and sb.nelec() > 0;
  const double flops = mixed ? 2.0 * npair_ * K * size() : double(size());
  const T* vpair;
  if constexpr (std::is_same_v<T, float>)
    vpair = vpair_f_.data();
  else
    vpair = vpair_.data();

  // In a sector, C = L + spin_flip * L^T with L the strict lower triangle of
  // C plus half its diagonal. The alpha-beta term is Q + spin_flip * Q^T with
  // Q its contraction of L, and the beta-beta term is spin_flip times the
  // transpose of the alpha-alpha term, so T = Q + (alpha-alpha term) is
  // formed in S and then S <- S + spin_flip * S^T, with beta * S split evenly
  // between S and its transpose
  const T s_scale = spin_flip ? T(0.5) * beta : beta;

#ifdef MACIS_FCI_PROFILE
  double prof_threads = 0, prof_gather = 0, prof_gemm = 0, prof_scatter = 0;
  MACIS_FCI_TIC(prof_t0);
#endif
  // Alpha strings per task: each beta link entry is read once for all of
  // them, and their sigma rows, products and integral columns stay L2
  // resident (about 0.6 MB)
  constexpr size_t row_group = sizeof(T) == 4 ? 4 : 2;
  const size_t beta_block = beta_block_size<T>(row_group, npair_);
  const auto beta_scatter = fci_detail::beta_scatter_kernel<T>(simd_);
  const size_t ngroups = (na + row_group - 1) / row_group;
  const size_t nocc_b = sb.nelec();
  Scratch scratch(*workspace_);
#pragma omp parallel num_threads(threads_for(spin_flip ? 0.5 * flops : flops))
  {
    [[maybe_unused]] FlushDenormals<T> flush;
    // Every element is written before it is read (gemm with beta = 0 does
    // not read Z)
    const size_t w_size = mixed ? row_group * npair_ * K : 0;
    const size_t x_size = mixed ? beta_block * K : 0;
    T* const W = scratch.get<T>(w_size + x_size +
                                (mixed ? row_group * beta_block * npair_ : 0));
    T* const X = W + w_size;
    T* const Z = X + x_size;
    // Per row of the group: column order of X and W and the row of C each
    // column gathers, on cache lines of their own
    std::vector<uint32_t> column_buffer(row_group * 2 * K + 32);
    uint32_t* const columns = cache_line_aligned(column_buffer.data());
#ifdef MACIS_FCI_PROFILE
    double t_gather = 0, t_gemm = 0, t_scatter = 0;
#endif

    // alpha-beta (and S <- beta * S): Z(rs, b) = sum_m W(rs, m) X(b, m) with
    // X gathering C over ia itself (for all diagonal E_pp) and the alpha
    // excitations of ia, and W the matching signed integral columns; Z is
    // then scattered through the beta excitations into row ia. In a sector
    // the work of a row grows with the addresses of its excitations, hence
    // the dynamic schedule (measured as fast as static without a sector)
#pragma omp for schedule(dynamic, 1)
    for (size_t g = 0; g < ngroups; ++g) {
      const size_t ia0 = g * row_group;
      const size_t nr = std::min(row_group, na - ia0);
      T* s[row_group];
      size_t kb[row_group];
      for (size_t r = 0; r < nr; ++r) {
        const size_t ia = ia0 + r;
        s[r] = S + ia * nb;
        if (beta == T(0))
          std::fill_n(s[r], nb, T(0));
        else if (s_scale != T(1))
          for (size_t ib = 0; ib < nb; ++ib) s[r][ib] *= s_scale;
        if (not mixed) continue;

        // Column 0 is ia, column l + 1 its l-th excitation; in a sector the
        // columns are ordered by decreasing source row, so that the rows of
        // L with entries in a beta block form a prefix
        uint32_t* column = columns + r * 2 * K;
        uint32_t* source = column + K;
        const uint32_t* occa = sa.occupied(ia);
        const auto* exa = sa.excitations(ia);
        auto source_row = [&](uint32_t col) {
          return col == 0 ? uint32_t(ia) : exa[col - 1].target;
        };
        std::iota(column, column + K, 0u);
        if (spin_flip)
          std::sort(column, column + K, [&](uint32_t x, uint32_t y) {
            return source_row(x) > source_row(y);
          });
        for (size_t m = 0; m < K; ++m) {
          T* w = W + (r * K + m) * npair_;
          source[m] = source_row(column[m]);
          if (column[m] == 0) {
            std::fill_n(w, npair_, T(0));
            for (size_t a = 0; a < sa.nelec(); ++a) {
              const T* v = vpair + fci_pair_index(occa[a], occa[a]) * npair_;
              for (size_t rs = 0; rs < npair_; ++rs) w[rs] += alpha * v[rs];
            }
          } else {
            const auto& ex = exa[column[m] - 1];
            const T* v = vpair + ex.pair * npair_;
            const T f = alpha * T(ex.sign);
            for (size_t rs = 0; rs < npair_; ++rs) w[rs] = f * v[rs];
          }
        }
        kb[r] = K;
      }
      if (not mixed) continue;

      for (size_t b0 = 0; b0 < nb; b0 += beta_block) {
        const size_t bl = std::min(beta_block, nb - b0);
        // Rows of the group with entries in this block (in a sector a row
        // has none past its diagonal)
        T* s_active[row_group];
        const T* z_active[row_group];
        int nactive = 0;
        for (size_t r = 0; r < nr; ++r) {
          const uint32_t* source = columns + r * 2 * K + K;
          if (spin_flip)
            while (kb[r] > 0 and source[kb[r] - 1] < b0) --kb[r];
          if (kb[r] == 0) continue;

          MACIS_FCI_TIC(tg);
          for (size_t m = 0; m < kb[r]; ++m) {
            const T* c = C + size_t(source[m]) * nb + b0;
            T* x = X + m * bl;
            // A row of L ends with its halved diagonal entry
            const size_t e = spin_flip ? source[m] - b0 : bl;
            if (e >= bl) {
              std::copy_n(c, bl, x);
            } else {
              std::copy_n(c, e, x);
              x[e] = T(0.5) * c[e];
              std::fill(x + e + 1, x + bl, T(0));
            }
          }
          MACIS_FCI_TOC(t_gather, tg);

          MACIS_FCI_TIC(tm);
          T* z = Z + nactive * beta_block * npair_;
          blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                     npair_, bl, kb[r], T(1), W + r * K * npair_, npair_, X, bl,
                     T(0), z, npair_);
          MACIS_FCI_TOC(t_gemm, tm);
          s_active[nactive] = s[r];
          z_active[nactive] = z;
          ++nactive;
        }
        // Columns only drop out of rows as the blocks advance
        if (nactive == 0) break;

        MACIS_FCI_TIC(ts);
        const uint32_t* lt = beta_link_target_.data();
        const uint32_t* lp = beta_link_pair_.data();
        const uint32_t* dp = beta_diag_pair_.data();
        beta_scatter(b0, bl, npair_, nexc_b, nocc_b, lt, lp, dp, nactive,
                     z_active, s_active);
        MACIS_FCI_TOC(t_scatter, ts);
      }
    }
#ifdef MACIS_FCI_PROFILE
#pragma omp critical
    {
      prof_threads += 1;
      prof_gather += t_gather;
      prof_gemm += t_gemm;
      prof_scatter += t_scatter;
    }
#endif
  }
#ifdef MACIS_FCI_PROFILE
  MACIS_FCI_TIC(prof_t1);
#endif

  same_spin_pass<T>(true, C, S, alpha, scratch);
#ifdef MACIS_FCI_PROFILE
  MACIS_FCI_TIC(prof_t2);
  double prof_t3 = prof_t2;
#endif
  if (spin_flip == 0) {
    same_spin_pass<T>(false, C, S, alpha, scratch);
#ifdef MACIS_FCI_PROFILE
    prof_t3 = profile_clock();
#endif
  } else {
    // S <- S + spin_flip * S^T over pairs of mirrored tiles; mirrored entries
    // come out exactly equal (opposite)
    constexpr size_t tile = 64;
    const size_t ntiles = (na + tile - 1) / tile;
#pragma omp parallel num_threads(threads_for(size()))
    {
      [[maybe_unused]] FlushDenormals<T> flush;
#pragma omp for schedule(dynamic)
      for (size_t ti = 0; ti < ntiles; ++ti)
        for (size_t tj = 0; tj <= ti; ++tj)
          for (size_t i = ti * tile; i < std::min(na, (ti + 1) * tile); ++i) {
            const size_t j1 = ti == tj ? i : std::min(na, (tj + 1) * tile);
            for (size_t j = tj * tile; j < j1; ++j) {
              const T a = S[i * na + j], b = S[j * na + i];
              S[i * na + j] = a + T(spin_flip) * b;
              S[j * na + i] = b + T(spin_flip) * a;
            }
            if (ti == tj) S[i * na + i] *= T(1 + spin_flip);
          }
    }
  }
#ifdef MACIS_FCI_PROFILE
  const double prof_t4 = profile_clock();
  const double nt = std::max(prof_threads, 1.0);
  std::fprintf(stderr,
               "PROFILE sigma<%s> total=%.4f ab=%.4f (gather %.4f gemm %.4f "
               "scatter %.4f) aa=%.4f bb=%.4f sym=%.4f\n",
               std::is_same_v<T, float> ? "float" : "double", prof_t4 - prof_t0,
               prof_t1 - prof_t0, prof_gather / nt, prof_gemm / nt,
               prof_scatter / nt, prof_t2 - prof_t1,
               spin_flip ? 0.0 : prof_t3 - prof_t2,
               spin_flip ? prof_t4 - prof_t2 : 0.0);
#endif
}

template <typename T>
void FCIHamiltonian::same_spin_pass(bool alpha_strings, const T* C, T* S,
                                    T alpha, Scratch& scratch) const {
  const SameSpinOperator& op = alpha_strings ? *h_alpha_ : *h_beta_;
  const size_t ncl = op.num_cliques();
  if (ncl == 0) return;
  const T* W;
  if constexpr (std::is_same_v<T, float>)
    W = op.W_f.data();
  else
    W = op.W.data();
  constexpr size_t TW = tile_width<T>;

  // Tiles of strings of the other spin (whole cache lines of TW, the last
  // one padded), with this spin's strings as rows: X(i, c) = C(i, t0 + c)
  // along alpha strings and C(t0 + c, i) along beta strings, and likewise
  // for S; RW is the row width of a tile
  const size_t nb = beta_->size();
  const size_t ns = alpha_strings ? alpha_->size() : nb;
  const size_t nother = alpha_strings ? nb : alpha_->size();
  const size_t ntiles = (nother + TW - 1) / TW;
  const size_t ldc = alpha_strings ? 1 : nb;  // between tile columns
  const size_t ldi = alpha_strings ? nb : 1;  // between strings
  const size_t m = op.clique_size;
  const size_t mp = fci_detail::clique_rows(m);
  const int nt = threads_for(2.0 * ncl * m * m * nother);

  auto gather = [&](size_t t0, size_t RW, size_t i0, size_t i1, T* X) {
    const size_t w = std::min(RW, nother - t0);
    const T* c0 = alpha_strings ? C + t0 : C + t0 * nb;
    for (size_t i = i0; i < i1; ++i)
      for (size_t c = 0; c < RW; ++c)
        X[i * RW + c] = c < w ? c0[i * ldi + c * ldc] : T(0);
  };
  // S(i, t0 + c) += alpha * value(i, c)
  auto scatter = [&](size_t t0, size_t RW, size_t i0, size_t i1, auto&& value) {
    const size_t w = std::min(RW, nother - t0);
    T* s0 = alpha_strings ? S + t0 : S + t0 * nb;
    for (size_t i = i0; i < i1; ++i)
      for (size_t c = 0; c < w; ++c)
        s0[i * ldi + c * ldc] += alpha * value(i, c);
  };

  // Per-thread work arrays with rows of RW elements on whole cache lines,
  // and pid after them (not shared with other threads: false sharing of pid
  // was measured to cost 1.6x)
  struct Work {
    T *X, *Y, *xk, *yk;
    uint32_t* pid;
    Work(Scratch& scratch, size_t nx, size_t ns, size_t mp, size_t RW) {
      X = scratch.get<T>((nx + ns + 2 * mp) * RW + mp);
      Y = X + nx * RW;
      xk = Y + ns * RW;
      yk = xk + mp * RW;
      pid = reinterpret_cast<uint32_t*>(yk + mp * RW);
    }
  };

  if (2 * ntiles >= size_t(nt)) {
    // Every task owns a tile. With fewer tiles than threads some threads
    // idle, which still beats sharing the cliques of each tile among them
    // (12x slower at 58 tiles for 64 threads: per-tile barriers and
    // reductions), so tiles are only shared when they are much scarcer
    const auto lines = tile_task_lines(ntiles, std::min<size_t>(nt, ntiles),
                                       fci_detail::max_tile_lines(simd_));
    const size_t max_lines = lines.front();
    std::vector<size_t> first(lines.size() + 1, 0);
    for (size_t t = 0; t < lines.size(); ++t)
      first[t + 1] = first[t] + lines[t] * TW;
    fci_detail::SameSpinTileKernel<T> kernels[3];
    for (size_t l = 1, k = 0; l <= max_lines; l *= 2, ++k)
      kernels[k] = fci_detail::same_spin_tile_kernel<T>(simd_, l);
#pragma omp parallel num_threads(std::min<size_t>(nt, lines.size()))
    {
      [[maybe_unused]] FlushDenormals<T> flush;
      Work w(scratch, ns, ns, mp, max_lines * TW);
#pragma omp for schedule(dynamic)
      for (size_t task = 0; task < lines.size(); ++task) {
        const size_t t0 = first[task], RW = lines[task] * TW;
        gather(t0, RW, 0, ns, w.X);
        std::fill_n(w.Y, ns * RW, T(0));
        kernels[lines[task] / 2](m, op.pair_dim, W, op.string.data(),
                                 op.pair.data(), 0, ncl, w.X, w.Y, w.xk, w.yk,
                                 w.pid);
        scatter(t0, RW, 0, ns,
                [&](size_t i, size_t c) { return w.Y[i * RW + c]; });
      }
    }
    return;
  }

  // Far fewer tiles than threads: the threads share the cliques of each tile
  // and their partial products are summed in thread order (reproducible, no
  // atomics)
  const auto kernel = fci_detail::same_spin_tile_kernel<T>(simd_, 1);
  std::vector<T> x_buffer(ns * TW + 64 / sizeof(T));
  T* X = cache_line_aligned(x_buffer.data());
  std::vector<const T*> partial(nt, nullptr);
#pragma omp parallel num_threads(nt)
  {
    [[maybe_unused]] FlushDenormals<T> flush;
#ifdef _OPENMP
    const int tid = omp_get_thread_num(), team = omp_get_num_threads();
#else
    const int tid = 0, team = 1;
#endif
    Work w(scratch, 0, ns, mp, TW);
    partial[tid] = w.Y;
    for (size_t tile = 0; tile < ntiles; ++tile) {
      const size_t t0 = tile * TW;
#pragma omp for schedule(static)
      for (size_t i = 0; i < ns; ++i) gather(t0, TW, i, i + 1, X);
      std::fill_n(w.Y, ns * TW, T(0));
      constexpr size_t clique_chunk = 16;
#pragma omp for schedule(dynamic)
      for (size_t k0 = 0; k0 < ncl; k0 += clique_chunk)
        kernel(m, op.pair_dim, W, op.string.data(), op.pair.data(), k0,
               std::min(k0 + clique_chunk, ncl), X, w.Y, w.xk, w.yk, w.pid);
#pragma omp for schedule(static)
      for (size_t i = 0; i < ns; ++i)
        scatter(t0, TW, i, i + 1, [&](size_t r, size_t c) {
          T sum = 0;
          for (int t = 0; t < team; ++t) sum += partial[t][r * TW + c];
          return sum;
        });
    }
  }
}

void FCIHamiltonian::operator_action(size_t m, double alpha, const double* V,
                                     size_t LDV, double beta, double* AV,
                                     size_t LDAV) const {
  for (size_t i = 0; i < m; ++i) sigma(V + i * LDV, AV + i * LDAV, alpha, beta);
}

void FCIHamiltonian::submatrix(const std::vector<size_t>& dets,
                               double* H_sub) const {
  const size_t nb = beta_->size();
  const size_t nd = dets.size();

  // Nonzero <I|E_pq|J> of one spin as (pair(p,q), value): all occupied
  // diagonal pairs when I == J, the single excitation linking I and J if any
  auto transition = [](const FCIStringSpace& space, size_t I, size_t J,
                       std::vector<std::pair<size_t, double>>& t) {
    t.clear();
    const uint32_t* oi = space.occupied(I);
    const uint32_t* oj = space.occupied(J);
    const size_t ne = space.nelec();
    if (I == J) {
      for (size_t a = 0; a < ne; ++a)
        t.emplace_back(fci_pair_index(oi[a], oi[a]), 1.0);
      return;
    }
    // p in I \ J is created, q in J \ I is annihilated
    size_t ndiff = 0, p = 0, q = 0;
    for (size_t a = 0, b = 0; a < ne or b < ne;) {
      if (b == ne or (a < ne and oi[a] < oj[b])) {
        p = oi[a++];
        ++ndiff;
      } else if (a == ne or oj[b] < oi[a]) {
        q = oj[b++];
      } else {
        ++a;
        ++b;
      }
    }
    if (ndiff != 1) return;
    size_t nbetween = 0;
    for (size_t b = 0; b < ne; ++b)
      if (oj[b] > std::min(p, q) and oj[b] < std::max(p, q)) ++nbetween;
    t.emplace_back(fci_pair_index(p, q), nbetween % 2 ? -1.0 : 1.0);
  };

  // H is symmetric: each element of the lower triangle is mirrored
#pragma omp parallel num_threads(threads_for(50.0 * nd * nd))
  {
    std::vector<std::pair<size_t, double>> ta, tb;
#pragma omp for schedule(dynamic)
    for (size_t j = 0; j < nd; ++j) {
      const size_t ja = dets[j] / nb, jb = dets[j] % nb;
      for (size_t i = j; i < nd; ++i) {
        const size_t ia = dets[i] / nb, ib = dets[i] % nb;
        double h = 0.0;
        if (ib == jb) h += same_spin_element(*h_alpha_, *alpha_, ia, ja);
        if (ia == ja) h += same_spin_element(*h_beta_, *beta_, ib, jb);
        transition(*alpha_, ia, ja, ta);
        if (not ta.empty()) {
          transition(*beta_, ib, jb, tb);
          for (auto [pq, xa] : ta)
            for (auto [rs, xb] : tb) h += xa * xb * vpair_[pq * npair_ + rs];
        }
        H_sub[i + j * nd] = H_sub[j + i * nd] = h;
      }
    }
  }
}

namespace {

/// Lowest eigenpair of a small dense symmetric m x m matrix A (column-major)
/// by Lanczos iterations with full reorthogonalization, from a fixed
/// pseudo-random start that overlaps every eigenvector; z gets the normalized
/// eigenvector, converged to a residual of about 1e-10 |A|. Single-threaded:
/// LAPACK's threaded dense solvers took 15-60 ms for the 400 x 400 P-space
/// matrix on 64 threads (6 ms on one), this about 2 ms.
double lanczos_lowest(size_t m, const double* A, double* z) {
  SerialLapack serial;
  const size_t kmax = std::min<size_t>(m, 200);
  std::vector<double> V((kmax + 1) * m), alpha, beta, T, w(kmax);
  uint64_t state = 0x9e3779b97f4a7c15ull;
  double nrm = 0.0;
  for (size_t i = 0; i < m; ++i) {
    // xorshift64*: reproducible on every platform
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    V[i] = double((state * 0x2545f4914f6cdd1dull) >> 11) * 0x1p-53 - 0.5;
    nrm += V[i] * V[i];
  }
  for (size_t i = 0; i < m; ++i) V[i] /= std::sqrt(nrm);

  // Frobenius norm, a bound on the spectral norm
  double a_norm = 0.0;
  for (size_t i = 0; i < m * m; ++i) a_norm += A[i] * A[i];
  a_norm = std::max(std::sqrt(a_norm), 1e-300);

  double theta = 0.0;
  size_t k = 0;
  // Lowest eigenpair (theta, w) of the k x k tridiagonal matrix so far
  auto ritz = [&] {
    T.assign(k * k, 0.0);
    for (size_t i = 0; i < k; ++i) {
      T[i + i * k] = alpha[i];
      if (i + 1 < k) T[i + 1 + i * k] = T[i + (i + 1) * k] = beta[i];
    }
    std::vector<double> evals(k);
    lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, k, T.data(), k,
                 evals.data());
    theta = evals[0];
    std::copy_n(T.data(), k, w.data());
  };
  for (k = 1; k <= kmax; ++k) {
    const double* v = V.data() + (k - 1) * m;
    double* r = V.data() + k * m;
    std::fill_n(r, m, 0.0);
    for (size_t c = 0; c < m; ++c) {
      const double* a = A + c * m;
      const double vc = v[c];
      for (size_t i = 0; i < m; ++i) r[i] += a[i] * vc;
    }
    double a_k = 0.0;
    for (size_t i = 0; i < m; ++i) a_k += v[i] * r[i];
    alpha.push_back(a_k);
    // Full reorthogonalization (twice is enough) instead of the three-term
    // recurrence alone, which loses orthogonality as Ritz values converge
    for (int pass = 0; pass < 2; ++pass)
      for (size_t j = 0; j < k; ++j) {
        const double* vj = V.data() + j * m;
        double d = 0.0;
        for (size_t i = 0; i < m; ++i) d += vj[i] * r[i];
        for (size_t i = 0; i < m; ++i) r[i] -= d * vj[i];
      }
    double b_k = 0.0;
    for (size_t i = 0; i < m; ++i) b_k += r[i] * r[i];
    b_k = std::sqrt(b_k);
    const bool invariant = b_k <= 1e-14 * a_norm;
    if (invariant or k == kmax or k % 8 == 0) {
      ritz();
      // |A z - theta z| = b_k |last component of the tridiagonal eigenvector|
      if (invariant or k == kmax or b_k * std::abs(w[k - 1]) <= 1e-10 * a_norm)
        break;
    }
    beta.push_back(b_k);
    for (size_t i = 0; i < m; ++i) r[i] /= b_k;
  }

  std::fill_n(z, m, 0.0);
  for (size_t j = 0; j < k; ++j)
    for (size_t i = 0; i < m; ++i) z[i] += w[j] * V[j * m + i];
  double zn = 0.0;
  for (size_t i = 0; i < m; ++i) zn += z[i] * z[i];
  for (size_t i = 0; i < m; ++i) z[i] /= std::sqrt(zn);
  return theta;
}

/// H restricted to the span of determinants (completed with their spin-flip
/// partners in a sector), in an orthonormal basis: the determinants P, or in
/// a sector one function per spin-flip pair, (|Ia,Ib> + spin_flip |Ib,Ia>) /
/// sqrt(2), and |Ia,Ia> for spin_flip = 1; function f has coefficient
/// coef[f][t] on determinant P[pos[f][t]]
struct SubspaceMatrix {
  static constexpr size_t unused = std::numeric_limits<size_t>::max();
  std::vector<size_t> P;
  std::vector<std::array<size_t, 2>> pos;
  std::vector<std::array<double, 2>> coef;
  std::vector<double> H;  ///< functions x functions, column-major
  size_t size() const { return pos.size(); }
};

SubspaceMatrix subspace_matrix(const FCIHamiltonian& H,
                               const std::vector<size_t>& dets, int spin_flip) {
  if (spin_flip < -1 or spin_flip > 1)
    throw std::invalid_argument(
        "fci_subspace_ground_state: spin_flip must be -1, 0 or 1");
  if (spin_flip != 0 and &H.alpha_strings() != &H.beta_strings())
    throw std::invalid_argument(
        "fci_subspace_ground_state: spin-flip symmetry requires equal numbers "
        "of alpha and beta electrons");
  const size_t nb = H.beta_strings().size();
  constexpr size_t unused = SubspaceMatrix::unused;

  SubspaceMatrix S;
  auto& P = S.P;
  P = dets;
  if (spin_flip != 0)
    for (size_t d : dets) P.push_back((d % nb) * nb + d / nb);
  std::sort(P.begin(), P.end());
  P.erase(std::unique(P.begin(), P.end()), P.end());
  const size_t np = P.size();

  for (size_t i = 0; i < np; ++i) {
    const size_t ia = P[i] / nb, ib = P[i] % nb;
    if (spin_flip == 0 or (ia == ib and spin_flip == 1)) {
      S.pos.push_back({i, unused});
      S.coef.push_back({1.0, 0.0});
    } else if (ia > ib) {
      const size_t j =
          std::lower_bound(P.begin(), P.end(), ib * nb + ia) - P.begin();
      S.pos.push_back({i, j});
      S.coef.push_back({std::sqrt(0.5), spin_flip * std::sqrt(0.5)});
    }
  }
  const size_t m = S.size();
  if (m == 0)
    throw std::invalid_argument(
        "fci_subspace_ground_state: no basis function in the spin-flip "
        "sector");

  std::vector<double> H_P(np * np);
  H.submatrix(P, H_P.data());
  if (spin_flip == 0) {
    S.H = std::move(H_P);
  } else {
    S.H.resize(m * m);
    const auto& pos = S.pos;
    const auto& coef = S.coef;
#pragma omp parallel for schedule(static) num_threads(threads_for(16.0 * m * m))
    for (size_t g = 0; g < m; ++g)
      for (size_t f = 0; f < m; ++f) {
        double h = 0.0;
        for (size_t t = 0; t < 2; ++t)
          for (size_t u = 0; u < 2; ++u)
            if (pos[f][t] != unused and pos[g][u] != unused)
              h += coef[f][t] * coef[g][u] * H_P[pos[f][t] + pos[g][u] * np];
        S.H[f + g * m] = h;
      }
  }
  return S;
}

/// The nonzero (determinant, coefficient) pairs of the vector with function
/// coefficients z
void expand(const SubspaceMatrix& S, const double* z,
            std::vector<std::pair<size_t, double>>& vector) {
  vector.clear();
  for (size_t f = 0; f < S.size(); ++f)
    for (size_t t = 0; t < 2; ++t)
      if (S.pos[f][t] != SubspaceMatrix::unused)
        vector.emplace_back(S.P[S.pos[f][t]], z[f] * S.coef[f][t]);
}

/// fci_subspace_ground_state with the eigenvector returned as its nonzero
/// (determinant, coefficient) pairs, from LAPACK or (iterative) Lanczos
double subspace_ground_state(const FCIHamiltonian& H,
                             const std::vector<size_t>& dets, int spin_flip,
                             bool iterative,
                             std::vector<std::pair<size_t, double>>& vector) {
  SubspaceMatrix S = subspace_matrix(H, dets, spin_flip);
  const size_t m = S.size();
  double E = 0.0;
  std::vector<double> Z(m);
  if (iterative) {
    E = lanczos_lowest(m, S.H.data(), Z.data());
  } else {
    std::vector<double> W(m);
    // lapackpp copies back 2*n support indices whatever the range
    std::vector<int64_t> support(2 * m);
    int64_t found = 0;
    lapack::syevr(lapack::Job::Vec, lapack::Range::Index, lapack::Uplo::Lower,
                  m, S.H.data(), m, 0.0, 0.0, 1, 1, 0.0, &found, W.data(),
                  Z.data(), m, support.data());
    E = W[0];
  }
  expand(S, Z.data(), vector);
  return E;
}

}  // namespace

double fci_subspace_ground_state(const FCIHamiltonian& H,
                                 const std::vector<size_t>& dets,
                                 std::vector<double>& C, int spin_flip) {
  std::vector<std::pair<size_t, double>> vector;
  const double E = subspace_ground_state(H, dets, spin_flip, false, vector);
  assign_zeros(C, H.size());
  for (auto [d, c] : vector) C[d] = c;
  return E;
}

double fci_ground_state(const FCIHamiltonian& H, size_t max_iter,
                        double res_tol, std::vector<double>& C, int spin_flip,
                        size_t max_subspace, FCIPrecision precision) {
  const bool guess = C.size() == H.size();
  if (not guess) assign_zeros(C, H.size());
  return fci_ground_state(H, max_iter, res_tol, C.data(), guess, spin_flip,
                          max_subspace, precision);
}

double fci_ground_state(const FCIHamiltonian& H, size_t max_iter,
                        double res_tol, double* C, bool guess, int spin_flip,
                        size_t max_subspace, FCIPrecision precision) {
  if (spin_flip < -1 or spin_flip > 1)
    throw std::invalid_argument(
        "fci_ground_state: spin_flip must be -1, 0 or 1");
  if (spin_flip != 0 and &H.alpha_strings() != &H.beta_strings())
    throw std::invalid_argument(
        "fci_ground_state: spin-flip symmetry requires equal numbers of alpha "
        "and beta electrons");

  auto logger = spdlog::get("ci_solver");
  if (!logger) {
    logger = spdlog::stdout_color_mt("ci_solver");
  }

  const size_t N = H.size();
  const size_t ns = H.alpha_strings().size();
  const bool mixed =
      precision == FCIPrecision::mixed or
      (precision == FCIPrecision::automatic and N >= mixed_precision_min_size);
  logger->info("[FCI Solver]:");
  logger->info(
      "  {} = {:6}, {} = {:.5e}, {} = {:4}, {} = {:4}, {} = {:2}, {} = {}",
      "NDETS", N, "RES_TOL", res_tol, "MAX_SUB", max_subspace, "MAX_ITER",
      max_iter, "SPIN_FLIP", spin_flip, "MIXED", mixed);

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;
  auto st = clock_type::now();

  // Left uninitialized so that its pages are first touched by the threads of
  // H.diagonal, with the partition the solver uses (NUMA)
  auto D = detail::davidson_alloc(N);
  H.diagonal(D.get());
  logger->info("  * diagonal: {:.1f} ms",
               duration_type(clock_type::now() - st).count());
  if (spin_flip != 0) {
    // An exactly symmetric diagonal and a guess exactly in the sector keep
    // every Davidson vector exactly in the sector
#pragma omp parallel for schedule(dynamic) num_threads(threads_for(N))
    for (size_t i = 0; i < ns; ++i)
      for (size_t j = 0; j < i; ++j) {
        const double d = 0.5 * (D[i * ns + j] + D[j * ns + i]);
        D[i * ns + j] = D[j * ns + i] = d;
      }
    if (guess)
      for (size_t i = 0; i < ns; ++i) {
        for (size_t j = 0; j < i; ++j) {
          const double c = 0.5 * (C[i * ns + j] + spin_flip * C[j * ns + i]);
          C[i * ns + j] = c;
          C[j * ns + i] = spin_flip * c;
        }
        if (spin_flip == -1) C[i * ns + i] = 0.0;
      }
  }

  // P-space: the determinants (sector functions, represented by their Ia >=
  // Ib determinant) with the lowest diagonal elements. Its lowest
  // eigenvector is the guess: unlike a single determinant, it couples to the
  // ground state even when the lowest determinant belongs to another
  // spatial-symmetry sector. In large spaces it is also an exact block of
  // the preconditioner, which saved one iteration (two for a stretched
  // H12) in every measured case for the cost of its eigendecomposition
  // (about 25 ms); larger P-spaces, or the largest components of the
  // converged vector, converged no faster or slower.
  const bool need_guess = not guess or blas::nrm2(N, C, 1) == 0.0;
  const bool use_block = N >= block_preconditioner_min_size;
  DavidsonBlockPreconditioner block;
  if (need_guess or use_block) {
    constexpr size_t pspace_size = 400;
    std::vector<std::pair<double, size_t>> lowest;
#pragma omp parallel
    {
      std::vector<std::pair<double, size_t>> heap;
      heap.reserve(pspace_size);
#pragma omp for schedule(static) nowait
      for (size_t i = 0; i < N; ++i) {
        if (spin_flip != 0 and
            (i / ns < i % ns or (i / ns == i % ns and spin_flip == -1)))
          continue;
        const std::pair<double, size_t> di(D[i], i);
        if (heap.size() < pspace_size) {
          heap.push_back(di);
          std::push_heap(heap.begin(), heap.end());
        } else if (di < heap.front()) {
          std::pop_heap(heap.begin(), heap.end());
          heap.back() = di;
          std::push_heap(heap.begin(), heap.end());
        }
      }
#pragma omp critical
      lowest.insert(lowest.end(), heap.begin(), heap.end());
    }
    std::sort(lowest.begin(), lowest.end());
    if (lowest.empty())
      throw std::invalid_argument(
          "fci_ground_state: the spin-flip sector has no determinants");
    const size_t np = std::min(lowest.size(), pspace_size);
    std::vector<size_t> pspace(np);
    for (size_t i = 0; i < np; ++i) pspace[i] = lowest[i].second;

    const auto t_guess = clock_type::now();
    std::vector<std::pair<size_t, double>> vector;
    if (use_block) {
      SubspaceMatrix S = subspace_matrix(H, pspace, spin_flip);
      const size_t m = S.size();
      std::vector<double> Z = S.H;
      block.lambda.resize(m);
      {
        SerialLapack serial;
        lapack::syevd(lapack::Job::Vec, lapack::Uplo::Lower, m, Z.data(), m,
                      block.lambda.data());
      }
      block.index.assign(S.P.begin(), S.P.end());
      const size_t nc = S.P.size();
      block.Z.assign(nc * m, 0.0);
      for (size_t a = 0; a < m; ++a)
        for (size_t f = 0; f < m; ++f)
          for (size_t t = 0; t < 2; ++t)
            if (S.pos[f][t] != SubspaceMatrix::unused)
              block.Z[S.pos[f][t] + a * nc] = S.coef[f][t] * Z[f + a * m];
      if (need_guess) expand(S, Z.data(), vector);
    } else {
      subspace_ground_state(H, pspace, spin_flip, true, vector);
    }
    if (need_guess) {
      // C is written (first touched) with the partition of the solver
      detail::davidson_partition(int64_t(N),
                                 [&](int64_t, int64_t i0, int64_t i1) {
                                   std::fill(C + i0, C + i1, 0.0);
                                 });
      for (auto [d, c] : vector) C[d] = c;
    }
    logger->info("  * P-space ({} determinants{}{}): {:.1f} ms", np,
                 need_guess ? ", guess" : "",
                 use_block ? ", preconditioner block" : "",
                 duration_type(clock_type::now() - t_guess).count());
  }
  if (not need_guess) logger->info("  * Will use passed vector as guess");
  const DavidsonBlockPreconditioner* block_ptr = use_block ? &block : nullptr;

  struct SectorOperator {
    const FCIHamiltonian& H;
    int spin_flip;
    void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                         double beta, double* AV, size_t LDAV) const {
      for (size_t i = 0; i < m; ++i)
        H.sigma(V + i * LDV, AV + i * LDAV, alpha, beta, spin_flip);
    }
  } op{H, spin_flip};

  auto finish = [&](size_t niter, double E) {
    logger->info("  {} = {:4}, {} = {:.6e} Eh, {} = {:.5e} ms", "DAV_NITER",
                 niter, "E0", E, "DAVIDSON_DUR",
                 duration_type(clock_type::now() - st).count());
    return E;
  };

  if (not mixed or N == 1) {
    auto [niter, E] = restarted_davidson(
        N, max_subspace, max_iter, op, D.get(), res_tol, C, 1e-12, true,
        nullptr, 0, static_cast<const SectorOperator*>(nullptr), 0.0,
        block_ptr);
    return finish(niter, E);
  }

  // Mixed precision. The Ritz vector is sum_i y_i v_i; the sigma error of a
  // correction vector v_i (i > 0) enters it scaled by y_i, which is as small
  // as the change of the Ritz vector since the start vector v_0. So after a
  // single-precision phase, the solve restarts from its vector with that
  // start vector in double precision and keeps the corrections in single
  // precision, and the final Ritz pair is accepted on its double-precision
  // residual.
  auto xf = detail::davidson_alloc<float>(N);
  auto sf = detail::davidson_alloc<float>(N);
  detail::davidson_partition(int64_t(N), [&](int64_t, int64_t i0, int64_t i1) {
    std::fill(xf.get() + i0, xf.get() + i1, 0.0f);
    std::fill(sf.get() + i0, sf.get() + i1, 0.0f);
  });
  struct SingleOperator {
    const FCIHamiltonian& H;
    int spin_flip;
    float* x;
    float* s;
    int64_t N;
    void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                         double beta, double* AV, size_t LDAV) const {
      for (size_t i = 0; i < m; ++i) {
        const double* v = V + i * LDV;
        double* av = AV + i * LDAV;
        // Values below the float range would become denormals (see
        // FlushDenormals)
        detail::davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
          constexpr double tiny = std::numeric_limits<float>::min();
          for (int64_t j = i0; j < i1; ++j)
            x[j] = std::abs(v[j]) < tiny ? 0.0f : float(v[j]);
        });
        H.sigma(x, s, 1.0f, 0.0f, spin_flip);
        detail::davidson_partition(N, [&](int64_t, int64_t i0, int64_t i1) {
          for (int64_t j = i0; j < i1; ++j)
            av[j] = alpha * double(s[j]) + (beta == 0.0 ? 0.0 : beta * av[j]);
        });
      }
    }
  } single{H, spin_flip, xf.get(), sf.get(), int64_t(N)};
  // Double precision for the first (start) vector, single for the rest
  struct AnchoredOperator {
    const SectorOperator& dp;
    const SingleOperator& sp;
    mutable bool anchored = false;
    void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                         double beta, double* AV, size_t LDAV) const {
      if (anchored) return sp.operator_action(m, alpha, V, LDV, beta, AV, LDAV);
      anchored = true;
      dp.operator_action(m, alpha, V, LDV, beta, AV, LDAV);
    }
  };

  // Single-precision sigma resolves residuals down to about 1e-7 |E| (H c ~
  // E c involves no cancellation); the switch keeps a wide margin above
  // that, with the lowest diagonal element as the energy scale
  double d_min = std::numeric_limits<double>::infinity();
#pragma omp parallel for schedule(static) reduction(min : d_min)
  for (size_t i = 0; i < N; ++i) d_min = std::min(d_min, D[i]);
  const double switch_tol =
      std::max(res_tol, std::max(1e-4, 5e-6 * std::abs(d_min)));

  // Single precision throughout, anchored in place (double-precision sigma
  // of the Ritz vector) at switch_tol, or earlier if single precision stops
  // the residual from falling; afterwards a stagnating solve stops and is
  // re-anchored below
  constexpr int64_t stagnation_window = 6;
  double res = 0.0;
  auto [niter, E] = restarted_davidson(
      N, max_subspace, max_iter, single, D.get(), res_tol, C, 1e-12, false,
      &res, stagnation_window, &op, switch_tol, block_ptr);
  logger->info("  * mixed precision: {} iterations, |r| = {:.3e}", niter, res);

  auto S = detail::davidson_alloc(N);
  constexpr int max_rounds = 4;
  for (int round = 0; round < max_rounds; ++round) {
    if (round > 0) {
      if (niter >= max_iter) break;
      AnchoredOperator anchored{op, single};
      auto [it, E2] = restarted_davidson(
          N, max_subspace, max_iter - niter, anchored, D.get(), res_tol, C,
          1e-12, false, &res, stagnation_window,
          static_cast<const AnchoredOperator*>(nullptr), 0.0, block_ptr);
      niter += it;
      logger->info("  * anchored: {} iterations, |r| = {:.3e}", it, res);
    }
    // Ritz pair (C is normalized) and its residual in double precision
    op.operator_action(1, 1.0, C, N, 0.0, S.get(), N);
    double e = 0.0, r2 = 0.0;
    detail::davidson_reduce(
        int64_t(N), 1, &e, [&](int64_t i0, int64_t i1, double* acc) {
          acc[0] += detail::davidson_sum(
              i0, i1, [&](int64_t i) { return C[i] * S[i]; });
        });
    detail::davidson_reduce(
        int64_t(N), 1, &r2, [&](int64_t i0, int64_t i1, double* acc) {
          acc[0] += detail::davidson_sum(i0, i1, [&](int64_t i) {
            const double r = S[i] - e * C[i];
            return r * r;
          });
        });
    logger->info("  * double-precision residual {:.3e}", std::sqrt(r2));
    if (std::sqrt(r2) < res_tol) return finish(niter, e);
  }

  logger->info("  * continuing in double precision");
  auto [it, E3] = restarted_davidson(
      N, max_subspace, max_iter > niter ? max_iter - niter : 1, op, D.get(),
      res_tol, C, 1e-12, true, nullptr, 0,
      static_cast<const SectorOperator*>(nullptr), 0.0, block_ptr);
  return finish(niter + it, E3);
}

}  // namespace macis

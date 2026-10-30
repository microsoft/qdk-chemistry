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
#include <lapack.hh>
#include <limits>
#include <macis/fci/fci_hamiltonian.hpp>
#include <macis/solvers/davidson.hpp>
#include <numeric>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace macis {

namespace {
// Beta strings contracted per DGEMM in the alpha-beta pass; keeps the
// per-thread gather/product buffers L2 resident for typical active spaces.
constexpr size_t beta_block = 128;

// Columns of the same-spin tiles: one cache line of doubles per string keeps
// a thread's input and output tiles cache resident (2 x 0.8 MB for 8e/16o)
// while still giving each loaded matrix element eight FMAs
constexpr size_t tile_width = 8;

// Clique members store their row of W with the sign in the top bit
constexpr uint32_t sign_bit = 0x80000000u;

/// +1 or -1 from the sign bit of a clique member, without a branch (member
/// signs are effectively random, so branches would mispredict half the time)
inline double member_sign(uint32_t pair) { return 1.0 - 2.0 * (pair >> 31); }

/// Row of the same-spin pair matrix for the orbital pair p < r
inline size_t distinct_pair_index(size_t p, size_t r) {
  return r * (r - 1) / 2 + p;
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

/*
 * Y += H^ss X over the cliques [k0, k1) of a same-spin operator; X and Y are
 * row-major (strings x tile_width) tiles. Each clique gathers its signed
 * members of X, multiplies them by its block of W four output members at a
 * time (register blocking) and scatters the signed results into Y. xk/yk hold
 * clique_size rounded up to four rows of tile_width, pid that many indices.
 */
void same_spin_tile(size_t m, size_t pair_dim, const double* W,
                    const uint32_t* string, const uint32_t* pair, size_t k0,
                    size_t k1, const double* __restrict X, double* __restrict Y,
                    double* __restrict xk, double* __restrict yk,
                    uint32_t* __restrict pid) {
  constexpr size_t TW = tile_width;
  const size_t mp = (m + 3) & ~size_t(3);
  for (size_t K = k0; K < k1; ++K) {
    const uint32_t* str = string + K * m;
    const uint32_t* pr = pair + K * m;
    for (size_t b = 0; b < m; ++b) {
      const double s = member_sign(pr[b]);
      const double* __restrict x = X + size_t(str[b]) * TW;
      double* __restrict xo = xk + b * TW;
      for (size_t c = 0; c < TW; ++c) xo[c] = s * x[c];
      pid[b] = pr[b] & ~sign_bit;
    }
    // Padding rows reuse a valid row of W; their results are never scattered
    for (size_t b = m; b < mp; ++b) pid[b] = pid[0];

    for (size_t a = 0; a < mp; a += 4) {
      const double* w0 = W + size_t(pid[a]) * pair_dim;
      const double* w1 = W + size_t(pid[a + 1]) * pair_dim;
      const double* w2 = W + size_t(pid[a + 2]) * pair_dim;
      const double* w3 = W + size_t(pid[a + 3]) * pair_dim;
      double acc0[TW] = {}, acc1[TW] = {}, acc2[TW] = {}, acc3[TW] = {};
      for (size_t b = 0; b < m; ++b) {
        const size_t pb = pid[b];
        const double h0 = w0[pb], h1 = w1[pb], h2 = w2[pb], h3 = w3[pb];
        const double* __restrict x = xk + b * TW;
        for (size_t c = 0; c < TW; ++c) {
          acc0[c] += h0 * x[c];
          acc1[c] += h1 * x[c];
          acc2[c] += h2 * x[c];
          acc3[c] += h3 * x[c];
        }
      }
      double* __restrict y = yk + a * TW;
      for (size_t c = 0; c < TW; ++c) {
        y[c] = acc0[c];
        y[TW + c] = acc1[c];
        y[2 * TW + c] = acc2[c];
        y[3 * TW + c] = acc3[c];
      }
    }

    for (size_t a = 0; a < m; ++a) {
      const double s = member_sign(pr[a]);
      double* __restrict yo = Y + size_t(str[a]) * TW;
      const double* __restrict yi = yk + a * TW;
      for (size_t c = 0; c < TW; ++c) yo[c] += s * yi[c];
    }
  }
}
}  // namespace

FCIHamiltonian::FCIHamiltonian(size_t norb, size_t nalpha, size_t nbeta,
                               const double* T, size_t LDT, const double* V,
                               size_t LDV)
    : norb_(norb), npair_(norb * (norb + 1) / 2) {
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

  coulomb_.resize(norb * norb);
  for (size_t p = 0; p < norb; ++p)
    for (size_t q = 0; q < norb; ++q) coulomb_[p * norb + q] = v(p, p, q, q);

  h_alpha_ = make_same_spin_operator(*alpha_, T, LDT, V, LDV);
  h_beta_ = beta_ == alpha_ ? h_alpha_
                            : make_same_spin_operator(*beta_, T, LDT, V, LDV);
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

void FCIHamiltonian::sigma(const double* C, double* S, double alpha,
                           double beta, int spin_flip) const {
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
  // Inner DGEMM dimension: alpha excitations plus one merged diagonal column
  const size_t K = nexc_a + 1;
  const bool mixed = sa.nelec() > 0 and sb.nelec() > 0;
  const double flops = mixed ? 2.0 * npair_ * K * size() : double(size());

  // In a sector, C = L + spin_flip * L^T with L the strict lower triangle of
  // C plus half its diagonal. The alpha-beta term is Q + spin_flip * Q^T with
  // Q its contraction of L, and the beta-beta term is spin_flip times the
  // transpose of the alpha-alpha term, so T = Q + (alpha-alpha term) is
  // formed in S and then S <- S + spin_flip * S^T, with beta * S split evenly
  // between S and its transpose
  const double s_scale = spin_flip ? 0.5 * beta : beta;

#pragma omp parallel num_threads(threads_for(spin_flip ? 0.5 * flops : flops))
  {
    std::vector<double> W(mixed ? npair_ * K : 0);
    std::vector<double> X(mixed ? beta_block * K : 0);
    std::vector<double> Z(mixed ? beta_block * npair_ : 0);
    // Column order of X and W and the row of C each column gathers, on cache
    // lines of their own
    std::vector<uint32_t> column_buffer(2 * K + 32);
    uint32_t* column = cache_line_aligned(column_buffer.data());
    uint32_t* source = column + K;

    // alpha-beta (and S <- beta * S): Z(rs, b) = sum_m W(rs, m) X(b, m) with
    // X gathering C over ia itself (for all diagonal E_pp) and the alpha
    // excitations of ia, and W the matching signed integral columns; Z is
    // then scattered through the beta excitations into row ia. In a sector
    // the work of a row grows with the addresses of its excitations, hence
    // the dynamic schedule (measured as fast as static without a sector)
#pragma omp for schedule(dynamic, 4)
    for (size_t ia = 0; ia < na; ++ia) {
      double* s = S + ia * nb;
      if (beta == 0.0)
        std::fill_n(s, nb, 0.0);
      else if (s_scale != 1.0)
        for (size_t ib = 0; ib < nb; ++ib) s[ib] *= s_scale;
      if (not mixed) continue;

      // Column 0 is ia, column l + 1 its l-th excitation; in a sector the
      // columns are ordered by decreasing source row, so that the rows of L
      // with entries in a beta block form a prefix
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
        double* w = W.data() + m * npair_;
        source[m] = source_row(column[m]);
        if (column[m] == 0) {
          std::fill_n(w, npair_, 0.0);
          for (size_t a = 0; a < sa.nelec(); ++a) {
            const double* v =
                vpair_.data() + fci_pair_index(occa[a], occa[a]) * npair_;
            for (size_t rs = 0; rs < npair_; ++rs) w[rs] += alpha * v[rs];
          }
        } else {
          const auto& ex = exa[column[m] - 1];
          const double* v = vpair_.data() + ex.pair * npair_;
          const double f = alpha * ex.sign;
          for (size_t rs = 0; rs < npair_; ++rs) w[rs] = f * v[rs];
        }
      }

      size_t kb = K;
      for (size_t b0 = 0; b0 < nb; b0 += beta_block) {
        const size_t bl = std::min(beta_block, nb - b0);
        if (spin_flip) {
          while (kb > 0 and source[kb - 1] < b0) --kb;
          if (kb == 0) break;
        }
        for (size_t m = 0; m < kb; ++m) {
          const double* c = C + size_t(source[m]) * nb + b0;
          double* x = X.data() + m * bl;
          // A row of L ends with its halved diagonal entry
          const size_t e = spin_flip ? source[m] - b0 : bl;
          if (e >= bl) {
            std::copy_n(c, bl, x);
          } else {
            std::copy_n(c, e, x);
            x[e] = 0.5 * c[e];
            std::fill(x + e + 1, x + bl, 0.0);
          }
        }

        blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
                   npair_, bl, kb, 1.0, W.data(), npair_, X.data(), bl, 0.0,
                   Z.data(), npair_);

        for (size_t b = 0; b < bl; ++b) {
          const size_t jb = b0 + b;
          const double* z = Z.data() + b * npair_;
          const auto* exb = sb.excitations(jb);
          for (size_t e = 0; e < nexc_b; ++e)
            s[exb[e].target] += exb[e].sign * z[exb[e].pair];
          const uint32_t* occb = sb.occupied(jb);
          double d = 0.0;
          for (size_t q = 0; q < sb.nelec(); ++q)
            d += z[fci_pair_index(occb[q], occb[q])];
          s[jb] += d;
        }
      }
    }
  }

  same_spin_pass(true, C, S, alpha);
  if (spin_flip == 0) {
    same_spin_pass(false, C, S, alpha);
    return;
  }

  // S <- S + spin_flip * S^T over pairs of mirrored tiles; mirrored entries
  // come out exactly equal (opposite)
  constexpr size_t tile = 64;
  const size_t ntiles = (na + tile - 1) / tile;
#pragma omp parallel for schedule(dynamic) num_threads(threads_for(size()))
  for (size_t ti = 0; ti < ntiles; ++ti)
    for (size_t tj = 0; tj <= ti; ++tj)
      for (size_t i = ti * tile; i < std::min(na, (ti + 1) * tile); ++i) {
        const size_t j1 = ti == tj ? i : std::min(na, (tj + 1) * tile);
        for (size_t j = tj * tile; j < j1; ++j) {
          const double a = S[i * na + j], b = S[j * na + i];
          S[i * na + j] = a + spin_flip * b;
          S[j * na + i] = b + spin_flip * a;
        }
        if (ti == tj) S[i * na + i] *= 1 + spin_flip;
      }
}

void FCIHamiltonian::same_spin_pass(bool alpha_strings, const double* C,
                                    double* S, double alpha) const {
  const SameSpinOperator& op = alpha_strings ? *h_alpha_ : *h_beta_;
  const size_t ncl = op.num_cliques();
  if (ncl == 0) return;

  // Tiles of tile_width strings of the other spin, with this spin's strings
  // as rows: X(i, c) = C(i, t0 + c) along alpha strings and C(t0 + c, i)
  // along beta strings, and likewise for S
  const size_t nb = beta_->size();
  const size_t ns = alpha_strings ? alpha_->size() : nb;
  const size_t nother = alpha_strings ? nb : alpha_->size();
  const size_t ntiles = (nother + tile_width - 1) / tile_width;
  const size_t ldc = alpha_strings ? 1 : nb;  // between tile columns
  const size_t ldi = alpha_strings ? nb : 1;  // between strings
  const size_t m = op.clique_size;
  const size_t mp = (m + 3) & ~size_t(3);
  const int nt = threads_for(2.0 * ncl * m * m * nother);

  auto gather = [&](size_t t0, size_t i0, size_t i1, double* X) {
    const size_t w = std::min(tile_width, nother - t0);
    const double* c0 = alpha_strings ? C + t0 : C + t0 * nb;
    for (size_t i = i0; i < i1; ++i)
      for (size_t c = 0; c < tile_width; ++c)
        X[i * tile_width + c] = c < w ? c0[i * ldi + c * ldc] : 0.0;
  };
  // S(i, t0 + c) += alpha * value(i, c)
  auto scatter = [&](size_t t0, size_t i0, size_t i1, auto&& value) {
    const size_t w = std::min(tile_width, nother - t0);
    double* s0 = alpha_strings ? S + t0 : S + t0 * nb;
    for (size_t i = i0; i < i1; ++i)
      for (size_t c = 0; c < w; ++c)
        s0[i * ldi + c * ldc] += alpha * value(i, c);
  };
  auto apply = [&](size_t k0, size_t k1, const double* X, double* Y, double* xk,
                   double* yk, uint32_t* pid) {
    same_spin_tile(m, op.pair_dim, op.W.data(), op.string.data(),
                   op.pair.data(), k0, k1, X, Y, xk, yk, pid);
  };

  // Per-thread work arrays with rows of tile_width doubles on whole cache
  // lines; pid gets cache lines of its own, as small heap blocks of
  // different threads may share lines (false sharing measured to cost 1.6x)
  struct Work {
    std::vector<double> buffer;
    std::vector<uint32_t> pid_buffer;
    double *X, *Y, *xk, *yk;
    uint32_t* pid;
    Work(size_t nx, size_t ns, size_t mp)
        : buffer((nx + ns + 2 * mp) * tile_width + 8), pid_buffer(mp + 32) {
      X = cache_line_aligned(buffer.data());
      Y = X + nx * tile_width;
      xk = Y + ns * tile_width;
      yk = xk + mp * tile_width;
      pid = cache_line_aligned(pid_buffer.data());
    }
  };

  if (ntiles >= size_t(nt)) {
    // Every task owns a tile
#pragma omp parallel num_threads(nt)
    {
      Work w(ns, ns, mp);
#pragma omp for schedule(dynamic)
      for (size_t tile = 0; tile < ntiles; ++tile) {
        const size_t t0 = tile * tile_width;
        gather(t0, 0, ns, w.X);
        std::fill_n(w.Y, ns * tile_width, 0.0);
        apply(0, ncl, w.X, w.Y, w.xk, w.yk, w.pid);
        scatter(t0, 0, ns,
                [&](size_t i, size_t c) { return w.Y[i * tile_width + c]; });
      }
    }
    return;
  }

  // Fewer tiles than threads: the threads share the cliques of each tile and
  // their partial products are summed in thread order (reproducible, no
  // atomics)
  std::vector<double> x_buffer(ns * tile_width + 8);
  double* X = cache_line_aligned(x_buffer.data());
  std::vector<const double*> partial(nt, nullptr);
#pragma omp parallel num_threads(nt)
  {
#ifdef _OPENMP
    const int tid = omp_get_thread_num(), team = omp_get_num_threads();
#else
    const int tid = 0, team = 1;
#endif
    Work w(0, ns, mp);
    partial[tid] = w.Y;
    for (size_t tile = 0; tile < ntiles; ++tile) {
      const size_t t0 = tile * tile_width;
#pragma omp for schedule(static)
      for (size_t i = 0; i < ns; ++i) gather(t0, i, i + 1, X);
      std::fill_n(w.Y, ns * tile_width, 0.0);
      constexpr size_t clique_chunk = 16;
#pragma omp for schedule(dynamic)
      for (size_t k0 = 0; k0 < ncl; k0 += clique_chunk)
        apply(k0, std::min(k0 + clique_chunk, ncl), X, w.Y, w.xk, w.yk, w.pid);
#pragma omp for schedule(static)
      for (size_t i = 0; i < ns; ++i)
        scatter(t0, i, i + 1, [&](size_t r, size_t c) {
          double sum = 0.0;
          for (int t = 0; t < team; ++t) sum += partial[t][r * tile_width + c];
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

double fci_subspace_ground_state(const FCIHamiltonian& H,
                                 const std::vector<size_t>& dets,
                                 std::vector<double>& C, int spin_flip) {
  if (spin_flip < -1 or spin_flip > 1)
    throw std::invalid_argument(
        "fci_subspace_ground_state: spin_flip must be -1, 0 or 1");
  if (spin_flip != 0 and &H.alpha_strings() != &H.beta_strings())
    throw std::invalid_argument(
        "fci_subspace_ground_state: spin-flip symmetry requires equal numbers "
        "of alpha and beta electrons");
  const size_t nb = H.beta_strings().size();

  std::vector<size_t> P(dets);
  if (spin_flip != 0)
    for (size_t d : dets) P.push_back((d % nb) * nb + d / nb);
  std::sort(P.begin(), P.end());
  P.erase(std::unique(P.begin(), P.end()), P.end());
  const size_t np = P.size();

  // Orthonormal basis: the determinants of P, or in a sector one function per
  // spin-flip pair, (|Ia,Ib> + spin_flip |Ib,Ia>) / sqrt(2), and |Ia,Ia> for
  // spin_flip = 1; function f has coefficient coef[f][t] on P[pos[f][t]]
  constexpr size_t unused = std::numeric_limits<size_t>::max();
  std::vector<std::array<size_t, 2>> pos;
  std::vector<std::array<double, 2>> coef;
  for (size_t i = 0; i < np; ++i) {
    const size_t ia = P[i] / nb, ib = P[i] % nb;
    if (spin_flip == 0 or (ia == ib and spin_flip == 1)) {
      pos.push_back({i, unused});
      coef.push_back({1.0, 0.0});
    } else if (ia > ib) {
      const size_t j =
          std::lower_bound(P.begin(), P.end(), ib * nb + ia) - P.begin();
      pos.push_back({i, j});
      coef.push_back({std::sqrt(0.5), spin_flip * std::sqrt(0.5)});
    }
  }
  const size_t m = pos.size();
  if (m == 0)
    throw std::invalid_argument(
        "fci_subspace_ground_state: no basis function in the spin-flip "
        "sector");

  std::vector<double> H_P(np * np);
  H.submatrix(P, H_P.data());
  std::vector<double> H_m;
  if (spin_flip == 0) {
    H_m = std::move(H_P);
  } else {
    H_m.resize(m * m);
#pragma omp parallel for schedule(static) num_threads(threads_for(16.0 * m * m))
    for (size_t g = 0; g < m; ++g)
      for (size_t f = 0; f < m; ++f) {
        double h = 0.0;
        for (size_t t = 0; t < 2; ++t)
          for (size_t u = 0; u < 2; ++u)
            if (pos[f][t] != unused and pos[g][u] != unused)
              h += coef[f][t] * coef[g][u] * H_P[pos[f][t] + pos[g][u] * np];
        H_m[f + g * m] = h;
      }
  }

  std::vector<double> W(m), Z(m);
  // lapackpp copies back 2*n support indices whatever the range
  std::vector<int64_t> support(2 * m);
  int64_t found = 0;
  lapack::syevr(lapack::Job::Vec, lapack::Range::Index, lapack::Uplo::Lower, m,
                H_m.data(), m, 0.0, 0.0, 1, 1, 0.0, &found, W.data(), Z.data(),
                m, support.data());

  C.assign(H.size(), 0.0);
  for (size_t f = 0; f < m; ++f)
    for (size_t t = 0; t < 2; ++t)
      if (pos[f][t] != unused) C[P[pos[f][t]]] = Z[f] * coef[f][t];
  return W[0];
}

double fci_ground_state(const FCIHamiltonian& H, size_t max_iter,
                        double res_tol, std::vector<double>& C, int spin_flip,
                        size_t max_subspace) {
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
  logger->info("[FCI Solver]:");
  logger->info("  {} = {:6}, {} = {:.5e}, {} = {:4}, {} = {:4}, {} = {:2}",
               "NDETS", N, "RES_TOL", res_tol, "MAX_SUB", max_subspace,
               "MAX_ITER", max_iter, "SPIN_FLIP", spin_flip);

  using clock_type = std::chrono::high_resolution_clock;
  using duration_type = std::chrono::duration<double, std::milli>;
  auto st = clock_type::now();

  // Left uninitialized so that its pages are first touched by the threads of
  // H.diagonal, with the partition the solver uses (NUMA)
  auto D = detail::davidson_alloc(N);
  H.diagonal(D.get());
  if (spin_flip != 0) {
    // An exactly symmetric diagonal and a guess exactly in the sector keep
    // every Davidson vector exactly in the sector
#pragma omp parallel for schedule(dynamic) num_threads(threads_for(N))
    for (size_t i = 0; i < ns; ++i)
      for (size_t j = 0; j < i; ++j) {
        const double d = 0.5 * (D[i * ns + j] + D[j * ns + i]);
        D[i * ns + j] = D[j * ns + i] = d;
      }
    if (C.size() == N)
      for (size_t i = 0; i < ns; ++i) {
        for (size_t j = 0; j < i; ++j) {
          const double c = 0.5 * (C[i * ns + j] + spin_flip * C[j * ns + i]);
          C[i * ns + j] = c;
          C[j * ns + i] = spin_flip * c;
        }
        if (spin_flip == -1) C[i * ns + i] = 0.0;
      }
  }

  if (C.size() != N or blas::nrm2(N, C.data(), 1) == 0.0) {
    // Lowest eigenvector of H within the P-space of the determinants (sector
    // functions, represented by their Ia >= Ib determinant) with the lowest
    // diagonal elements: unlike a single determinant, it couples to the
    // ground state even when the lowest determinant belongs to another
    // spatial-symmetry sector
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
    fci_subspace_ground_state(H, pspace, C, spin_flip);
  } else {
    logger->info("  * Will use passed vector as guess");
  }

  struct SectorOperator {
    const FCIHamiltonian& H;
    int spin_flip;
    void operator_action(size_t m, double alpha, const double* V, size_t LDV,
                         double beta, double* AV, size_t LDAV) const {
      for (size_t i = 0; i < m; ++i)
        H.sigma(V + i * LDV, AV + i * LDAV, alpha, beta, spin_flip);
    }
  } op{H, spin_flip};
  auto [niter, E] = restarted_davidson(N, max_subspace, max_iter, op, D.get(),
                                       res_tol, C.data());

  logger->info("  {} = {:4}, {} = {:.6e} Eh, {} = {:.5e} ms", "DAV_NITER",
               niter, "E0", E, "DAVIDSON_DUR",
               duration_type(clock_type::now() - st).count());
  return E;
}

}  // namespace macis

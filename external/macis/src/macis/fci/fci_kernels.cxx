/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include "fci_kernels.hpp"

#include <stdexcept>

// x86-64 kernels for instruction sets beyond the build's baseline, selected at
// run time. They use target attributes rather than per-file flags, so no
// inline function shared with other translation units is ever compiled for
// an instruction set the CPU may lack.
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#define MACIS_FCI_X86_SIMD 1
#include <immintrin.h>
#define MACIS_AVX2 __attribute__((target("avx2,fma")))
#define MACIS_AVX512 __attribute__((target("avx512f")))
#else
#define MACIS_FCI_X86_SIMD 0
#endif

namespace macis {

FCISimd fci_simd_supported() {
#if MACIS_FCI_X86_SIMD
  static const FCISimd level = [] {
    __builtin_cpu_init();
    if (__builtin_cpu_supports("avx512f")) return FCISimd::avx512;
    if (__builtin_cpu_supports("avx2") and __builtin_cpu_supports("fma"))
      return FCISimd::avx2;
    return FCISimd::generic;
  }();
  return level;
#else
  return FCISimd::generic;
#endif
}

namespace fci_detail {
namespace {

/// +1 or -1 from the sign bit of a clique member, without a branch (member
/// signs are effectively random, so branches would mispredict half the time)
template <typename T>
inline T member_sign(uint32_t pair) {
  return T(1) - T(2) * T(pair >> 31);
}

// Compiler-vectorized kernel: four output members at a time
template <typename T>
void same_spin_tile_generic(size_t m, size_t pair_dim, const T* W,
                            const uint32_t* string, const uint32_t* pair,
                            size_t k0, size_t k1, const T* __restrict X,
                            T* __restrict Y, T* __restrict xk, T* __restrict yk,
                            uint32_t* __restrict pid) {
  constexpr size_t TW = tile_width<T>;
  const size_t mp = (m + 3) & ~size_t(3);
  for (size_t K = k0; K < k1; ++K) {
    const uint32_t* str = string + K * m;
    const uint32_t* pr = pair + K * m;
    for (size_t b = 0; b < m; ++b) {
      const T s = member_sign<T>(pr[b]);
      const T* __restrict x = X + size_t(str[b]) * TW;
      T* __restrict xo = xk + b * TW;
      for (size_t c = 0; c < TW; ++c) xo[c] = s * x[c];
      pid[b] = pr[b] & ~sign_bit;
    }
    // Padding rows reuse a valid row of W; their results are never scattered
    for (size_t b = m; b < mp; ++b) pid[b] = pid[0];

    for (size_t a = 0; a < mp; a += 4) {
      const T* w0 = W + size_t(pid[a]) * pair_dim;
      const T* w1 = W + size_t(pid[a + 1]) * pair_dim;
      const T* w2 = W + size_t(pid[a + 2]) * pair_dim;
      const T* w3 = W + size_t(pid[a + 3]) * pair_dim;
      T acc0[TW] = {}, acc1[TW] = {}, acc2[TW] = {}, acc3[TW] = {};
      for (size_t b = 0; b < m; ++b) {
        const size_t pb = pid[b];
        const T h0 = w0[pb], h1 = w1[pb], h2 = w2[pb], h3 = w3[pb];
        const T* __restrict x = xk + b * TW;
        for (size_t c = 0; c < TW; ++c) {
          acc0[c] += h0 * x[c];
          acc1[c] += h1 * x[c];
          acc2[c] += h2 * x[c];
          acc3[c] += h3 * x[c];
        }
      }
      T* __restrict y = yk + a * TW;
      for (size_t c = 0; c < TW; ++c) {
        y[c] = acc0[c];
        y[TW + c] = acc1[c];
        y[2 * TW + c] = acc2[c];
        y[3 * TW + c] = acc3[c];
      }
    }

    for (size_t a = 0; a < m; ++a) {
      const T s = member_sign<T>(pr[a]);
      T* __restrict yo = Y + size_t(str[a]) * TW;
      const T* __restrict yi = yk + a * TW;
      for (size_t c = 0; c < TW; ++c) yo[c] += s * yi[c];
    }
  }
}

#if MACIS_FCI_X86_SIMD

// AVX-512: a tile row is one register. R output members are accumulated in
// R registers (R = 8 keeps both FMA pipes busy despite the FMA latency); each
// matrix element is one broadcast load, and the finished rows are added to Y
// directly.
MACIS_AVX512 inline __m512d v512_load(const double* p) {
  return _mm512_load_pd(p);
}
MACIS_AVX512 inline __m512 v512_load(const float* p) {
  return _mm512_load_ps(p);
}
MACIS_AVX512 inline void v512_store(double* p, __m512d v) {
  _mm512_store_pd(p, v);
}
MACIS_AVX512 inline void v512_store(float* p, __m512 v) {
  _mm512_store_ps(p, v);
}
MACIS_AVX512 inline __m512d v512_set1(double x) { return _mm512_set1_pd(x); }
MACIS_AVX512 inline __m512 v512_set1(float x) { return _mm512_set1_ps(x); }
template <typename T>
using v512_t = decltype(v512_set1(T(0)));
MACIS_AVX512 inline __m512d v512_fma(__m512d a, __m512d b, __m512d c) {
  return _mm512_fmadd_pd(a, b, c);
}
MACIS_AVX512 inline __m512 v512_fma(__m512 a, __m512 b, __m512 c) {
  return _mm512_fmadd_ps(a, b, c);
}
MACIS_AVX512 inline __m512d v512_mul(__m512d a, __m512d b) {
  return _mm512_mul_pd(a, b);
}
MACIS_AVX512 inline __m512 v512_mul(__m512 a, __m512 b) {
  return _mm512_mul_ps(a, b);
}

template <typename T, int R, int V>
MACIS_AVX512 inline void avx512_rows(size_t a, size_t m, size_t pair_dim,
                                     const T* W, const uint32_t* str,
                                     const uint32_t* pr, const uint32_t* pid,
                                     const T* xk, T* Y) {
  constexpr size_t L = tile_width<T>;
  constexpr size_t RW = V * L;
  using VT = v512_t<T>;
  const T* w[R];
  VT acc[R][V];
  for (int r = 0; r < R; ++r) {
    w[r] = W + size_t(pid[a + r]) * pair_dim;
    for (int v = 0; v < V; ++v) acc[r][v] = v512_set1(T(0));
  }
  for (size_t b = 0; b < m; ++b) {
    const size_t pb = pid[b];
    VT x[V];
    for (int v = 0; v < V; ++v) x[v] = v512_load(xk + b * RW + v * L);
    for (int r = 0; r < R; ++r) {
      const VT h = v512_set1(w[r][pb]);
      for (int v = 0; v < V; ++v) acc[r][v] = v512_fma(h, x[v], acc[r][v]);
    }
  }
  // Rows past m pad the last block and are dropped
  for (int r = 0; r < R; ++r) {
    if (a + r >= m) break;
    T* y = Y + size_t(str[a + r]) * RW;
    const VT s = v512_set1(member_sign<T>(pr[a + r]));
    for (int v = 0; v < V; ++v)
      v512_store(y + v * L, v512_fma(s, acc[r][v], v512_load(y + v * L)));
  }
}

// Tile rows of V registers: each broadcast matrix element feeds V FMAs, which
// is what limits one-register rows (per cache line 1.5-1.8x faster with V =
// 2 and 1.7-2x with V = 4 on Cascade Lake)
template <typename T, int V>
MACIS_AVX512 void same_spin_tile_avx512(size_t m, size_t pair_dim, const T* W,
                                        const uint32_t* string,
                                        const uint32_t* pair, size_t k0,
                                        size_t k1, const T* X, T* Y, T* xk, T*,
                                        uint32_t* pid) {
  constexpr size_t RW = V * tile_width<T>;
  // Blocks of R output members: at least eight independent FMA chains (R V
  // accumulators) while the R row pointers of W still fit in registers, R
  // being 9 or 8 (6 or 5 for V = 4), whichever pads m less. The last block
  // is padded with copies of a valid row rather than run short, as a short
  // block is latency bound and costs as much as a full one.
  constexpr size_t R1 = V == 4 ? 6 : 9;
  const size_t R =
      m % R1 == 0 or (m + R1 - 1) / R1 * R1 < (m + R1 - 2) / (R1 - 1) * (R1 - 1)
          ? R1
          : R1 - 1;
  const size_t mp = (m + R - 1) / R * R;
  for (size_t K = k0; K < k1; ++K) {
    const uint32_t* str = string + K * m;
    const uint32_t* pr = pair + K * m;
    for (size_t b = 0; b < m; ++b) {
      const v512_t<T> s = v512_set1(member_sign<T>(pr[b]));
      const T* x = X + size_t(str[b]) * RW;
      for (int v = 0; v < V; ++v)
        v512_store(xk + b * RW + v * tile_width<T>,
                   v512_mul(s, v512_load(x + v * tile_width<T>)));
      pid[b] = pr[b] & ~sign_bit;
    }
    for (size_t b = m; b < mp; ++b) pid[b] = pid[0];
    for (size_t a = 0; a < m; a += R) {
      if (R == R1)
        avx512_rows<T, R1, V>(a, m, pair_dim, W, str, pr, pid, xk, Y);
      else
        avx512_rows<T, R1 - 1, V>(a, m, pair_dim, W, str, pr, pid, xk, Y);
    }
  }
}

// AVX2: a tile row is two registers; four output members (eight
// accumulators) at a time
MACIS_AVX2 inline __m256d v256_load(const double* p) {
  return _mm256_load_pd(p);
}
MACIS_AVX2 inline __m256 v256_load(const float* p) { return _mm256_load_ps(p); }
MACIS_AVX2 inline void v256_store(double* p, __m256d v) {
  _mm256_store_pd(p, v);
}
MACIS_AVX2 inline void v256_store(float* p, __m256 v) { _mm256_store_ps(p, v); }
MACIS_AVX2 inline __m256d v256_set1(double x) { return _mm256_set1_pd(x); }
MACIS_AVX2 inline __m256 v256_set1(float x) { return _mm256_set1_ps(x); }
template <typename T>
using v256_t = decltype(v256_set1(T(0)));
MACIS_AVX2 inline __m256d v256_fma(__m256d a, __m256d b, __m256d c) {
  return _mm256_fmadd_pd(a, b, c);
}
MACIS_AVX2 inline __m256 v256_fma(__m256 a, __m256 b, __m256 c) {
  return _mm256_fmadd_ps(a, b, c);
}
MACIS_AVX2 inline __m256d v256_mul(__m256d a, __m256d b) {
  return _mm256_mul_pd(a, b);
}
MACIS_AVX2 inline __m256 v256_mul(__m256 a, __m256 b) {
  return _mm256_mul_ps(a, b);
}

template <typename T, int R>
MACIS_AVX2 inline void avx2_rows(size_t a, size_t m, size_t pair_dim,
                                 const T* W, const uint32_t* str,
                                 const uint32_t* pr, const uint32_t* pid,
                                 const T* xk, T* Y) {
  constexpr size_t TW = tile_width<T>;
  constexpr size_t H = TW / 2;
  using V = v256_t<T>;
  const T* w[R];
  V acc[R][2];
  for (int r = 0; r < R; ++r) {
    w[r] = W + size_t(pid[a + r]) * pair_dim;
    acc[r][0] = acc[r][1] = v256_set1(T(0));
  }
  for (size_t b = 0; b < m; ++b) {
    const size_t pb = pid[b];
    const V x0 = v256_load(xk + b * TW);
    const V x1 = v256_load(xk + b * TW + H);
    for (int r = 0; r < R; ++r) {
      const V h = v256_set1(w[r][pb]);
      acc[r][0] = v256_fma(h, x0, acc[r][0]);
      acc[r][1] = v256_fma(h, x1, acc[r][1]);
    }
  }
  for (int r = 0; r < R; ++r) {
    if (a + r >= m) break;
    T* y = Y + size_t(str[a + r]) * TW;
    const V s = v256_set1(member_sign<T>(pr[a + r]));
    v256_store(y, v256_fma(s, acc[r][0], v256_load(y)));
    v256_store(y + H, v256_fma(s, acc[r][1], v256_load(y + H)));
  }
}

template <typename T>
MACIS_AVX2 void same_spin_tile_avx2(size_t m, size_t pair_dim, const T* W,
                                    const uint32_t* string,
                                    const uint32_t* pair, size_t k0, size_t k1,
                                    const T* X, T* Y, T* xk, T*,
                                    uint32_t* pid) {
  constexpr size_t TW = tile_width<T>;
  constexpr size_t H = TW / 2;
  // Padded blocks of four, as for AVX-512
  const size_t mp = (m + 3) & ~size_t(3);
  for (size_t K = k0; K < k1; ++K) {
    const uint32_t* str = string + K * m;
    const uint32_t* pr = pair + K * m;
    for (size_t b = 0; b < m; ++b) {
      const v256_t<T> s = v256_set1(member_sign<T>(pr[b]));
      const T* x = X + size_t(str[b]) * TW;
      v256_store(xk + b * TW, v256_mul(s, v256_load(x)));
      v256_store(xk + b * TW + H, v256_mul(s, v256_load(x + H)));
      pid[b] = pr[b] & ~sign_bit;
    }
    for (size_t b = m; b < mp; ++b) pid[b] = pid[0];
    for (size_t a = 0; a < m; a += 4)
      avx2_rows<T, 4>(a, m, pair_dim, W, str, pr, pid, xk, Y);
  }
}

#endif  // MACIS_FCI_X86_SIMD

// Alpha-beta scatter, compiler-vectorized: each link entry is read once for
// all NR rows
template <typename T, int NR>
void beta_scatter_rows(size_t b0, size_t bl, size_t npair, size_t nexc,
                       size_t nocc, const uint32_t* link_target,
                       const uint32_t* link_pair, const uint32_t* diag_pair,
                       const T* const* z, T* const* s) {
  for (size_t b = 0; b < bl; ++b) {
    const size_t jb = b0 + b;
    const uint32_t* lt = link_target + jb * nexc;
    const uint32_t* lp = link_pair + jb * nexc;
    const T* zb[NR];
    for (int r = 0; r < NR; ++r) zb[r] = z[r] + b * npair;
    for (size_t e = 0; e < nexc; ++e) {
      const uint32_t t = lt[e], ps = lp[e];
      const uint32_t p = ps & ~sign_bit;
      const T sg = member_sign<T>(ps);
      for (int r = 0; r < NR; ++r) s[r][t] += sg * zb[r][p];
    }
    const uint32_t* dp = diag_pair + jb * nocc;
    for (int r = 0; r < NR; ++r) {
      T d = 0;
      for (size_t q = 0; q < nocc; ++q) d += zb[r][dp[q]];
      s[r][jb] += d;
    }
  }
}

template <typename T>
void beta_scatter_generic(size_t b0, size_t bl, size_t npair, size_t nexc,
                          size_t nocc, const uint32_t* link_target,
                          const uint32_t* link_pair, const uint32_t* diag_pair,
                          int nrows, const T* const* z, T* const* s) {
  switch (nrows) {
    case 1:
      beta_scatter_rows<T, 1>(b0, bl, npair, nexc, nocc, link_target, link_pair,
                              diag_pair, z, s);
      break;
    case 2:
      beta_scatter_rows<T, 2>(b0, bl, npair, nexc, nocc, link_target, link_pair,
                              diag_pair, z, s);
      break;
    case 3:
      beta_scatter_rows<T, 3>(b0, bl, npair, nexc, nocc, link_target, link_pair,
                              diag_pair, z, s);
      break;
    default:
      beta_scatter_rows<T, 4>(b0, bl, npair, nexc, nocc, link_target, link_pair,
                              diag_pair, z, s);
  }
}

}  // namespace

size_t max_tile_lines(FCISimd simd) {
#if MACIS_FCI_X86_SIMD
  if (simd == FCISimd::avx512) return 4;
#endif
  return 1;
}

template <typename T>
SameSpinTileKernel<T> same_spin_tile_kernel(FCISimd simd, size_t lines) {
  if (lines > max_tile_lines(simd) or (lines & (lines - 1)) != 0)
    throw std::invalid_argument("same_spin_tile_kernel: unsupported width");
#if MACIS_FCI_X86_SIMD
  if (simd == FCISimd::avx512) {
    if (lines == 4) return &same_spin_tile_avx512<T, 4>;
    if (lines == 2) return &same_spin_tile_avx512<T, 2>;
    return &same_spin_tile_avx512<T, 1>;
  }
  if (simd == FCISimd::avx2) return &same_spin_tile_avx2<T>;
#endif
  return &same_spin_tile_generic<T>;
}

template SameSpinTileKernel<float> same_spin_tile_kernel<float>(FCISimd,
                                                                size_t);
template SameSpinTileKernel<double> same_spin_tile_kernel<double>(FCISimd,
                                                                  size_t);

// The scatter has no SIMD variants: AVX-512 gather/scatter over a string's
// links (conflict free) measured 2.2x (float) / 2.7x (double) slower than
// the scalar multi-row loop on Cascade Lake
template <typename T>
BetaScatterKernel<T> beta_scatter_kernel(FCISimd) {
  return &beta_scatter_generic<T>;
}

template BetaScatterKernel<float> beta_scatter_kernel<float>(FCISimd);
template BetaScatterKernel<double> beta_scatter_kernel<double>(FCISimd);

}  // namespace fci_detail
}  // namespace macis

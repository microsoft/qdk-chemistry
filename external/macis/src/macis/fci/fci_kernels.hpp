/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

// Internal: hand-vectorized kernels of the direct FCI Hamiltonian
#pragma once
#include <cstddef>
#include <cstdint>
#include <macis/fci/fci_hamiltonian.hpp>

namespace macis::fci_detail {

/// Elements of one cache line, the unit of same-spin tile rows: exactly one
/// AVX-512 register (8 doubles or 16 floats)
template <typename T>
inline constexpr size_t tile_width = 64 / sizeof(T);

/// Clique members store their row of W with the sign in the top bit
inline constexpr uint32_t sign_bit = 0x80000000u;

/**
 * Y += H^ss X over the cliques [k0, k1) of a same-spin operator, for X and
 * Y row-major tiles (strings x lines * tile_width<T>, the lines the kernel
 * was selected for) on 64-byte aligned rows. Each clique gathers its signed
 * member rows of X into xk, multiplies them by its block of W (rows and
 * columns pid) and adds the signed results to Y. xk and yk hold
 * clique_rows(m) rows of the tile width, pid as many indices.
 */
template <typename T>
using SameSpinTileKernel = void (*)(size_t m, size_t pair_dim, const T* W,
                                    const uint32_t* string,
                                    const uint32_t* pair, size_t k0, size_t k1,
                                    const T* X, T* Y, T* xk, T* yk,
                                    uint32_t* pid);

/// Rows of the per-clique work arrays for cliques of m members (room for
/// padding m to a multiple of any register block of up to 9 rows)
inline constexpr size_t clique_rows(size_t m) {
  return (m + 8 + 15) & ~size_t(15);
}

/// Widest same-spin tile rows, in cache lines, of an instruction-set level
size_t max_tile_lines(FCISimd simd);

/// Same-spin kernel for an instruction-set level (which must be supported)
/// and tile rows of 1, 2 or 4 cache lines (at most max_tile_lines(simd))
template <typename T>
SameSpinTileKernel<T> same_spin_tile_kernel(FCISimd simd, size_t lines);

/**
 * Alpha-beta scatter of a beta block for nrows (1 to 4) alpha strings:
 * s[r](target) += sign * z[r](pair) over the off-diagonal beta excitations of
 * every string jb of the block [b0, b0 + bl), and s[r](jb) += the sum of
 * z[r] over the diagonal pairs of jb. Links of string jb are
 * link_target/link_pair[jb * nexc + e] (pair index with the sign in bit
 * 31), its diagonal pairs diag_pair[jb * nocc + q]; z[r] holds npair
 * products per string of the block.
 */
template <typename T>
using BetaScatterKernel = void (*)(size_t b0, size_t bl, size_t npair,
                                   size_t nexc, size_t nocc,
                                   const uint32_t* link_target,
                                   const uint32_t* link_pair,
                                   const uint32_t* diag_pair, int nrows,
                                   const T* const* z, T* const* s);

/// Alpha-beta scatter kernel for an instruction-set level
template <typename T>
BetaScatterKernel<T> beta_scatter_kernel(FCISimd simd);

}  // namespace macis::fci_detail

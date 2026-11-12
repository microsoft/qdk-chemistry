/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

// Internal: performance kernels of the direct FCI Hamiltonian.
//
// Instruction sets. All code written for a specific instruction set is in
// fci_kernels.cxx: the same-spin clique kernels for AVX2+FMA and AVX-512F
// (x86-64, GCC or Clang), compiled through target attributes and selected at
// run time from fci_simd_supported(), so builds for a baseline ISA (QDK's
// default -march=x86-64, the x86-64-v3 wheels) still use them where the CPU
// has them. Each has a portable C++ fallback (FCISimd::generic, the only
// level of other compilers and architectures, e.g. AArch64); the alpha-beta
// scatter is portable C++ only (compiler-vectorized: sigma built for
// -march=x86-64 runs within 2 % of -march=native on Cascade Lake). Tests run
// every level the CPU supports (FCIHamiltonian::set_simd). A new instruction
// set (NEON or SVE, AVX10, ...) needs an FCISimd value, its detection in
// fci_simd_supported(), and a SameSpinTileKernel for each tile width up to
// its max_tile_lines.
//
// Contracts the kernels rely on: tile rows of tile_width elements (one
// 64-byte cache line) on 64-byte aligned memory (detail::davidson_alloc),
// loaded with aligned instructions; clique work arrays of clique_rows(m)
// rows (padding for the AVX-512 register blocks).
//
// Tuning. Constants that only affect speed were measured on one machine (a
// Xeon Platinum 8280M Cascade Lake VM: AVX-512 with two FMA ports, 1 MiB L2
// per core, 4 NUMA nodes of 16 cores, MKL); their comments say "on Cascade
// Lake" and are the places to revisit for another microarchitecture.
#pragma once
#include <cstddef>
#include <cstdint>
#include <macis/fci/fci_hamiltonian.hpp>

namespace macis::fci_detail {

/// Elements of one cache line, the unit of same-spin tile rows: one AVX-512
/// register (8 doubles or 16 floats), two AVX2 registers
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
 * Alpha strings per alpha-beta task: each beta link entry is read once for
 * all of them, and they share the cache lines of their interleaved sigma
 * rows and of the gathered rows of C (4 measured best for both precisions
 * on Cascade Lake: 2 for double cost 5 % more)
 */
inline constexpr size_t row_group = 4;

/**
 * Alpha-beta scatter of a beta block into the sigma rows of a group of
 * row_group alpha strings, stored interleaved (element t of row r at
 * si[t * row_group + r]): row r gets s_r(target) += sign * z_r(pair) over
 * the off-diagonal beta excitations of every string jb of the block [b0, b0
 * + bl), and s_r(jb) += the sum of z_r over the diagonal pairs of jb. Links
 * of string jb are link_target/link_pair[jb * nexc + e], of positive sign
 * for e < link_npos[jb]; its diagonal pairs are diag_pair[jb * nocc + q].
 * The nrows rows lane[0:nrows) (increasing) take part, z[a] holding npair
 * products per string of the block for row lane[a].
 */
template <typename T>
using BetaScatterKernel = void (*)(size_t b0, size_t bl, size_t npair,
                                   size_t nexc, size_t nocc,
                                   const uint32_t* link_target,
                                   const uint32_t* link_pair,
                                   const uint32_t* link_npos,
                                   const uint32_t* diag_pair, int nrows,
                                   const int* lane, const T* const* z, T* si);

/// Alpha-beta scatter kernel for an instruction-set level
template <typename T>
BetaScatterKernel<T> beta_scatter_kernel(FCISimd simd);

}  // namespace macis::fci_detail

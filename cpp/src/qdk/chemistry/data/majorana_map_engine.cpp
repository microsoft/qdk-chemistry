// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Core>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <map>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/sparse.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <qdk/chemistry/utils/hash_context.hpp>
#include <set>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "two_body_symmetry.hpp"

namespace qdk::chemistry::data {
namespace detail {

// Majorana decomposition coefficients for the excitation operator a†_p a_q:
//   a†_p a_q = (1/4) Σ_{a,b} coeff[a][b] · γ_{2p+a} · γ_{2q+b}
//   coeff[a][b] = (-i)^a · (i)^b
constexpr std::complex<double> excitation_coeff[2][2] = {
    {{1.0, 0.0}, {0.0, 1.0}},
    {{0.0, -1.0}, {1.0, 0.0}},
};

// ─── Bitpacked Pauli representation ────────────────────────────────
//
// Represents a Pauli string as a pair of bitmasks (x_bits, z_bits)
// using the symplectic encoding:
//   I: x=0, z=0    X: x=1, z=0    Z: x=0, z=1    Y: x=1, z=1
//
// Each uint64_t covers 64 qubits. Templated on NW (number of uint64
// words) to keep everything on the stack. The engine dispatches to
// NW in {1, ..., 16} at runtime (covers up to 1024 qubits).

template <std::size_t NW>
struct PackedPauliWord {
  std::array<std::uint64_t, NW> x{};
  std::array<std::uint64_t, NW> z{};

  bool operator==(const PackedPauliWord& other) const = default;
};

template <std::size_t NW>
struct PackedPauliWordHash {
  std::size_t operator()(const PackedPauliWord<NW>& w) const noexcept {
    utils::HashContext ctx;
    hash_value(ctx, "packed_pauli_word");
    hash_value(ctx, static_cast<uint64_t>(NW));
    for (std::size_t i = 0; i < NW; ++i) {
      hash_value(ctx, static_cast<uint64_t>(w.x[i]));
    }
    for (std::size_t i = 0; i < NW; ++i) {
      hash_value(ctx, static_cast<uint64_t>(w.z[i]));
    }
    return ctx.hash_code();
  }
};

template <std::size_t NW>
PackedPauliWord<NW> sparse_to_packed(const SparsePauliWord& word) {
  PackedPauliWord<NW> pw{};
  for (const auto& [qubit, op] : word) {
    std::size_t wi = qubit / 64;
    std::uint64_t bit = std::uint64_t(1) << (qubit % 64);
    if (wi < NW) {
      if (op == 1 || op == 2) pw.x[wi] |= bit;
      if (op == 2 || op == 3) pw.z[wi] |= bit;
    }
  }
  return pw;
}

template <std::size_t NW>
SparsePauliWord packed_to_sparse(const PackedPauliWord<NW>& pw) {
  SparsePauliWord word;
  for (std::size_t wi = 0; wi < NW; ++wi) {
    std::uint64_t x = pw.x[wi];
    std::uint64_t z = pw.z[wi];
    std::uint64_t active = x | z;
    while (active) {
      int bit = std::countr_zero(active);
      std::uint64_t mask = std::uint64_t(1) << bit;
      std::uint64_t qubit = wi * 64 + bit;
      bool has_x = (x & mask) != 0;
      bool has_z = (z & mask) != 0;
      std::uint8_t op = has_x ? (has_z ? 2 : 1) : 3;
      word.emplace_back(qubit, op);
      active &= ~mask;
    }
  }
  return word;
}

template <std::size_t NW>
std::pair<int, PackedPauliWord<NW>> multiply_packed(
    const PackedPauliWord<NW>& p1, const PackedPauliWord<NW>& p2) {
  PackedPauliWord<NW> result;
  int phase_exp = 0;
  for (std::size_t i = 0; i < NW; ++i) {
    std::uint64_t x1 = p1.x[i], z1 = p1.z[i];
    std::uint64_t x2 = p2.x[i], z2 = p2.z[i];
    result.x[i] = x1 ^ x2;
    result.z[i] = z1 ^ z2;
    std::uint64_t nx1 = ~x1, nz1 = ~z1, nx2 = ~x2, nz2 = ~z2;
    std::uint64_t cyc =
        (x1 & nz1 & x2 & z2) | (x1 & z1 & nx2 & z2) | (nx1 & z1 & x2 & nz2);
    std::uint64_t anti =
        (x1 & z1 & x2 & nz2) | (nx1 & z1 & x2 & z2) | (x1 & nz1 & nx2 & z2);
    phase_exp += std::popcount(cyc);
    phase_exp -= std::popcount(anti);
  }
  // Pauli multiplication phases cycle mod 4:
  // 0: +1, 1: +i, 2: −1, 3: −i.
  return {phase_exp & 3, result};
}

/// Apply a phase index (0=+1, 1=+i, 2=-1, 3=-i) to a complex scale factor.
inline std::complex<double> apply_phase(int phase_idx,
                                        std::complex<double> scale) {
  switch (phase_idx & 3) {
    case 0:
      return scale;
    case 1:
      return {-scale.imag(), scale.real()};
    case 2:
      return {-scale.real(), -scale.imag()};
    case 3:
      return {scale.imag(), -scale.real()};
  }
  return scale;  // unreachable
}

template <std::size_t NW>
class PackedAccumulator {
 public:
  void accumulate(const PackedPauliWord<NW>& word, std::complex<double> coeff) {
    terms_[word] += coeff;
  }

  void accumulate_product(const PackedPauliWord<NW>& w1,
                          const PackedPauliWord<NW>& w2,
                          std::complex<double> scale) {
    auto [phase_idx, word] = multiply_packed(w1, w2);
    terms_[word] += apply_phase(phase_idx, scale);
  }

  std::vector<std::pair<std::complex<double>, SparsePauliWord>> get_terms(
      double threshold) const {
    std::vector<std::pair<std::complex<double>, SparsePauliWord>> result;
    result.reserve(terms_.size());
    for (const auto& [pw, coeff] : terms_) {
      if (std::abs(coeff) >= threshold) {
        result.emplace_back(coeff, packed_to_sparse(pw));
      }
    }
    return result;
  }

 private:
  std::unordered_map<PackedPauliWord<NW>, std::complex<double>,
                     PackedPauliWordHash<NW>>
      terms_;
};

// ─── Engine implementation ─────────────────────────────────────────
//
// The standard non-relativistic electronic Hamiltonian is spin-free: it
// contains no spin operators.  The Majorana decomposition of excitation
// operators E_pq = a†_p a_q into bilinear products of γ's works for
// any spin-free Hamiltonian regardless of the orbital basis.
//
// The spin_symmetric flag declares equality of the spin channels:
//
//   spin_symmetric = true   (restricted orbitals)
//     The caller guarantees that all spin channels share the same integrals
//     (h_alpha == h_beta, eri_aaaa == eri_bbbb == eri_aabb).  Eightfold
//     integrals use spin-summed E_pq = E^α_pq + E^β_pq, roughly halving the
//     two-body work; fourfold integrals use spin-summed symmetric and
//     antisymmetric excitation pairs E_pq ± E_qp.  The δ_{qr} two-body
//     contraction is folded into the one-body integrals.
//
//   spin_symmetric = false  (unrestricted orbitals)
//     Each spin channel (αα, ββ, αβ, βα) is handled independently
//     with its own integrals.  This is needed for unrestricted orbitals,
//     where h^α ≠ h^β and the ERI channels differ — even though the
//     underlying Hamiltonian is still spin-free.  αβ and βα are each
//     other's exchange partners, so αβ alone needs only reversal
//     (pq|rs) = (qp|sr) for the excitation-pair split, which also requires
//     fourfold αα and ββ; other inputs loop over every channel in full.

// ─── ERI providers ─────────────────────────────────────────────────
//
// The two-body engine reads each electron-repulsion integral through a
// small accessor object instead of a raw dense pointer.  This lets the
// *same* mapping arithmetic consume integrals from different storage
// formats without ever materializing a dense N^4 tensor:
//
//   * DenseEriProvider     — flat row-major N^4 array (canonical storage)
//   * CholeskyEriProvider  — three-center factors L; the aux index is
//                            contracted one (pq|·) row at a time
//   * SparseEriProvider    — literal sparse (p,q,r,s) -> value entries,
//                            absent positions are zero
//
// Index convention matches the dense layout used throughout the engine:
//   eri(p,q,r,s) -> ((p*N + q)*N + r)*N + s   (chemist notation (pq|rs)).

// Providers expose two access granularities:
//   * aaaa(p,q,r,s)   — scalar access; used only by the one-body delta
//                       fold, which touches O(N^3) integrals.
//   * row_xxxx(p,q)   — the entire (pq|·) row of N^2 integrals; used by
//                       the two-body mapping loops, which fix (p,q) in
//                       their outer loops and sweep (r,s) in the inner
//                       ones.  Returning whole rows lets each provider
//                       amortize its access cost across the inner sweep.
struct DenseEriProvider {
  // Dense storage is traversed with the standard symmetric quad loop.
  static constexpr bool kSparse = false;
  const double* aaaa_;
  const double* aabb_;
  const double* bbbb_;
  std::size_t n_;
  std::size_t idx4(std::size_t p, std::size_t q, std::size_t r,
                   std::size_t s) const {
    return ((p * n_ + q) * n_ + r) * n_ + s;
  }
  double aaaa(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return aaaa_[idx4(p, q, r, s)];
  }
  double bbbb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return bbbb_[idx4(p, q, r, s)];
  }
  double aabb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return aabb_[idx4(p, q, r, s)];
  }
  // Rows are contiguous in the dense layout, so row access is zero-copy
  // and the dense path reads exactly the same memory as before.
  const double* row_aaaa(std::size_t p, std::size_t q) const {
    return aaaa_ + (p * n_ + q) * n_ * n_;
  }
  const double* row_aabb(std::size_t p, std::size_t q) const {
    return aabb_ + (p * n_ + q) * n_ * n_;
  }
  const double* row_bbbb(std::size_t p, std::size_t q) const {
    return bbbb_ + (p * n_ + q) * n_ * n_;
  }
};

// Three-center (Cholesky/density-fitted) provider.  The factor matrices
// are column-major Eigen storage of shape [N^2 x naux] with the orbital
// pair index in row-major order, so element (pair, Q) lives at
// L[pair + Q * N^2].
//
// The aux index is contracted in integral space, one orbital pair at a
// time: for a fixed (pq), the matrix-vector product L2 * l1.row(pq)^T
// yields the entire (pq|·) row, which the mapping loops then consume as
// plain scalars.  The contraction streams sequentially through the factor
// columns (vectorizable axpy updates) instead of issuing a strided
// O(naux) dot product per integral. Row evaluation uses one N^2 row;
// the separate factor-symmetry QR validation also needs scratch memory.
struct CholeskyEriProvider {
  static constexpr bool kSparse = false;
  const double* La_;
  const double* Lb_;
  std::size_t n_;
  std::size_t naux_;
  // Scratch row for the aux contraction; the engine is single-threaded
  // and each row is fully consumed before the next one is requested.
  mutable std::vector<double> row_buf_{};

  // Scalar access for the one-body delta fold only (O(N^3) accesses, far
  // below the cost of the two-body row contractions).
  double dot(const double* l1, std::size_t pair1, const double* l2,
             std::size_t pair2) const {
    const std::size_t n2 = n_ * n_;
    double acc = 0.0;
    for (std::size_t Q = 0; Q < naux_; ++Q) {
      acc += l1[pair1 + Q * n2] * l2[pair2 + Q * n2];
    }
    return acc;
  }
  double aaaa(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return dot(La_, p * n_ + q, La_, r * n_ + s);
  }
  double bbbb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return dot(Lb_, p * n_ + q, Lb_, r * n_ + s);
  }
  double aabb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return dot(La_, p * n_ + q, Lb_, r * n_ + s);
  }

  // row[rs] = sum_Q l1[pq + Q*N^2] * l2[rs + Q*N^2] for all rs at once.
  // The per-element accumulation order over Q matches the scalar dot, so
  // results are identical to the element-wise reconstruction.
  const double* build_row(const double* l1, std::size_t pair1,
                          const double* l2) const {
    const std::size_t n2 = n_ * n_;
    row_buf_.assign(n2, 0.0);
    double* out = row_buf_.data();
    for (std::size_t Q = 0; Q < naux_; ++Q) {
      const double w = l1[pair1 + Q * n2];
      if (w == 0.0) continue;
      const double* col = l2 + Q * n2;
      for (std::size_t rs = 0; rs < n2; ++rs) {
        out[rs] += w * col[rs];
      }
    }
    return out;
  }
  const double* row_aaaa(std::size_t p, std::size_t q) const {
    return build_row(La_, p * n_ + q, La_);
  }
  const double* row_aabb(std::size_t p, std::size_t q) const {
    return build_row(La_, p * n_ + q, Lb_);
  }
  const double* row_bbbb(std::size_t p, std::size_t q) const {
    return build_row(Lb_, p * n_ + q, Lb_);
  }
};

// Sparse provider for restricted model Hamiltonians.  All spin channels
// share the same integrals (SparseHamiltonianContainer is always
// restricted), so a single channel backs the aaaa/aabb/bbbb accessors.
//
// Ingestion reads the entries literally, after expanding deprecated single
// eightfold representatives:
//   * entries_ — exactly one canonical (p<=q, r<=s, (p,q)<=(r,s))
//                position per eightfold class with a stored member, with
//                its literal value and that of (p,q,s,r), sorted
//                lexicographically so accumulation order is deterministic
//                for any input ordering.
//   * map_     — every stored position with its literal value; used by
//                scalar lookups and the general fallback.
struct SparseEriProvider {
  // Enables the dedicated non-zero-only two-body loop in majorana_map_impl.
  static constexpr bool kSparse = true;
  struct Entry {
    std::size_t p, q, r, s;
    double value;
    double ket_swapped_value = 0.0;
  };
  // Literal flattened position -> value.
  std::unordered_map<std::uint64_t, double> map_;
  // Canonical class positions, sorted lexicographically.
  std::vector<Entry> entries_;
  std::size_t n_ = 0;
  // Scratch row for the general fallback when screening breaks symmetry.
  mutable std::vector<double> row_buf_{};
  const std::vector<Entry>& entries() const { return entries_; }
  std::uint64_t key(std::size_t p, std::size_t q, std::size_t r,
                    std::size_t s) const {
    return static_cast<std::uint64_t>(((p * n_ + q) * n_ + r) * n_ + s);
  }
  double aaaa(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    auto it = map_.find(key(p, q, r, s));
    return it == map_.end() ? 0.0 : it->second;
  }
  double bbbb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return aaaa(p, q, r, s);
  }
  double aabb(std::size_t p, std::size_t q, std::size_t r,
              std::size_t s) const {
    return aaaa(p, q, r, s);
  }
  // Build the (pq|·) row from map_ for the general fallback.
  const double* build_row(std::size_t p, std::size_t q) const {
    row_buf_.assign(n_ * n_, 0.0);
    const std::uint64_t n2 = static_cast<std::uint64_t>(n_ * n_);
    const std::uint64_t pair = static_cast<std::uint64_t>(p * n_ + q);
    for (const auto& [k, v] : map_) {
      if (k / n2 == pair) {
        row_buf_[static_cast<std::size_t>(k % n2)] = v;
      }
    }
    return row_buf_.data();
  }
  const double* row_aaaa(std::size_t p, std::size_t q) const {
    return build_row(p, q);
  }
  const double* row_aabb(std::size_t p, std::size_t q) const {
    return build_row(p, q);
  }
  const double* row_bbbb(std::size_t p, std::size_t q) const {
    return build_row(p, q);
  }
};

template <class EriProvider>
bool pair_screening_preserves_symmetry(const EriProvider& provider,
                                       std::size_t extent, bool restricted,
                                       double integral_threshold) {
  if (integral_threshold <= 0.0) return true;
  auto retained = [integral_threshold](double value) {
    return std::abs(value) >= integral_threshold;
  };
  if constexpr (EriProvider::kSparse) {
    for (const auto& entry : provider.entries()) {
      const std::size_t left[2][2] = {{entry.p, entry.q}, {entry.q, entry.p}};
      const std::size_t right[2][2] = {{entry.r, entry.s}, {entry.s, entry.r}};
      for (const auto& first : left)
        for (const auto& second : right) {
          const auto value =
              retained(provider.aaaa(first[0], first[1], second[0], second[1]));
          if (value != retained(provider.aaaa(second[0], second[1], first[0],
                                              first[1])) ||
              value != retained(provider.aaaa(first[1], first[0], second[1],
                                              second[0])))
            return false;
        }
    }
    return true;
  }
  auto check_channel = [&](auto row_accessor, auto scalar_accessor,
                           bool exchange) {
    for (std::size_t bra = 0; bra < extent; ++bra)
      for (std::size_t ket = 0; ket < extent; ++ket) {
        const double* row = (provider.*row_accessor)(bra, ket);
        for (std::size_t other_bra = 0; other_bra < extent; ++other_bra)
          for (std::size_t other_ket = 0; other_ket < extent; ++other_ket) {
            const bool value = retained(row[other_bra * extent + other_ket]);
            if (value != retained((provider.*scalar_accessor)(
                             ket, bra, other_ket, other_bra)) ||
                (exchange && value != retained((provider.*scalar_accessor)(
                                          other_bra, other_ket, bra, ket))))
              return false;
          }
      }
    return true;
  };
  return check_channel(&EriProvider::row_aaaa, &EriProvider::aaaa, true) &&
         (restricted ||
          (check_channel(&EriProvider::row_bbbb, &EriProvider::bbbb, true) &&
           check_channel(&EriProvider::row_aabb, &EriProvider::aabb, false)));
}

template <std::size_t NW, class EriProvider>
MajoranaMapResult majorana_map_impl(const MajoranaMapping& mapping,
                                    double core_energy, const double* h1_alpha,
                                    const double* h1_beta,
                                    const EriProvider& eri_provider,
                                    std::size_t n_spatial, bool spin_symmetric,
                                    bool pair_split, double threshold,
                                    double integral_threshold) {
  if (pair_split &&
      !pair_screening_preserves_symmetry(eri_provider, n_spatial,
                                         spin_symmetric, integral_threshold)) {
    pair_split = false;
    spin_symmetric = false;
  }
  const std::size_t n_modes = 2 * n_spatial;

  PackedAccumulator<NW> acc;

  // ─── Pair product cache ───────────────────────────────────────────
  //
  // Precompute γ_j · γ_k = (-i) · bilinear(j, k) for each within-spin-block
  // pair. Using bilinear() as the access primitive makes this work for both
  // Majorana-atomic and bilinear-only mappings.

  struct PackedPairProduct {
    std::complex<double> coeff;
    PackedPauliWord<NW> word;
  };
  const std::size_t maj_per_spin = 2 * n_spatial;
  const std::size_t alpha_offset = 0;
  const std::size_t beta_offset = maj_per_spin;

  auto build_pair_cache =
      [&](std::size_t global_offset) -> std::vector<PackedPairProduct> {
    std::vector<PackedPairProduct> cache(maj_per_spin * maj_per_spin);
    for (std::size_t i = 0; i < maj_per_spin; ++i) {
      cache[i * maj_per_spin + i] = {{1.0, 0.0}, PackedPauliWord<NW>{}};
      for (std::size_t j = 0; j < maj_per_spin; ++j) {
        if (i == j) continue;
        auto [bl_coeff, bl_word] =
            mapping.bilinear(i + global_offset, j + global_offset);
        cache[i * maj_per_spin + j] = {
            std::complex<double>(0.0, -1.0) * bl_coeff,
            sparse_to_packed<NW>(bl_word)};
      }
    }
    return cache;
  };

  const std::size_t num_spin_species =
      (mapping.num_modes() < 2 * n_spatial) ? 1 : 2;
  const bool use_spin_symmetric =
      spin_symmetric && (num_spin_species == 2) && !pair_split;

  auto ppair_alpha = build_pair_cache(alpha_offset);
  std::vector<PackedPairProduct> ppair_beta;
  if (num_spin_species == 2) {
    ppair_beta = build_pair_cache(beta_offset);
  }

  auto alpha_pair = [&](std::size_t i,
                        std::size_t j) -> const PackedPairProduct& {
    return ppair_alpha[i * maj_per_spin + j];
  };
  auto beta_pair = [&](std::size_t i,
                       std::size_t j) -> const PackedPairProduct& {
    return ppair_beta[i * maj_per_spin + j];
  };

  auto mode_alpha = [](std::size_t p) -> std::size_t { return p; };
  auto mode_beta = [n_spatial](std::size_t p) -> std::size_t {
    return p + n_spatial;
  };

  auto accumulate_epq = [&](std::size_t mode_p, std::size_t mode_q,
                            double h_pq) {
    bool is_alpha = mode_p < n_spatial;
    auto& cache = is_alpha ? ppair_alpha : ppair_beta;
    std::size_t bp = is_alpha ? mode_p : (mode_p - n_spatial);
    std::size_t bq = is_alpha ? mode_q : (mode_q - n_spatial);
    for (int a = 0; a < 2; ++a) {
      for (int b = 0; b < 2; ++b) {
        const auto& [coeff, word] =
            cache[(2 * bp + a) * maj_per_spin + (2 * bq + b)];
        acc.accumulate(word, coeff * h_pq * 0.25 * excitation_coeff[a][b]);
      }
    }
  };

  // ─── Core energy ──────────────────────────────────────────────────
  if (std::abs(core_energy) > integral_threshold) {
    PackedPauliWord<NW> identity{};
    acc.accumulate(identity, std::complex<double>(core_energy, 0.0));
  }

  // ─── One-body terms ──────────────────────────────────────────────
  std::vector<double> h1_eff_alpha(n_spatial * n_spatial);
  std::vector<double> h1_eff_beta(n_spatial * n_spatial);
  for (std::size_t p = 0; p < n_spatial; ++p) {
    for (std::size_t s = 0; s < n_spatial; ++s) {
      double h_a = h1_alpha[p * n_spatial + s];
      double h_b = use_spin_symmetric ? h_a : h1_beta[p * n_spatial + s];
      if (use_spin_symmetric) {
        double delta_corr = 0.0;
        for (std::size_t q = 0; q < n_spatial; ++q) {
          delta_corr += eri_provider.aaaa(p, q, q, s);
        }
        h_a -= 0.5 * delta_corr;
        h_b = h_a;
      }
      h1_eff_alpha[p * n_spatial + s] = h_a;
      h1_eff_beta[p * n_spatial + s] = h_b;
    }
  }

  for (std::size_t p = 0; p < n_spatial; ++p) {
    for (std::size_t q = 0; q < n_spatial; ++q) {
      double h_pq_a = h1_eff_alpha[p * n_spatial + q];
      if (std::abs(h_pq_a) > integral_threshold) {
        accumulate_epq(mode_alpha(p), mode_alpha(q), h_pq_a);
      }
      if (num_spin_species == 2) {
        double h_pq_b = h1_eff_beta[p * n_spatial + q];
        if (std::abs(h_pq_b) > integral_threshold) {
          accumulate_epq(mode_beta(p), mode_beta(q), h_pq_b);
        }
      }
    }
  }

  // ─── Two-body terms ───────────────────────────────────────────────

  if (pair_split) {
    using Terms =
        std::vector<std::pair<std::complex<double>, PackedPauliWord<NW>>>;
    struct ExcitationPair {
      std::size_t bra, ket;
      Terms symmetric, antisymmetric;
    };
    auto make_pairs = [&](bool spin_sum, std::size_t spin) {
      std::vector<ExcitationPair> pairs;
      pairs.reserve(n_spatial * (n_spatial + 1) / 2);
      for (std::size_t bra = 0; bra < n_spatial; ++bra) {
        for (std::size_t ket = bra; ket < n_spatial; ++ket) {
          std::unordered_map<PackedPauliWord<NW>, std::complex<double>,
                             PackedPauliWordHash<NW>>
              symmetric, antisymmetric;
          for (std::size_t channel = spin_sum ? 0 : spin;
               channel < (spin_sum ? num_spin_species : spin + 1); ++channel) {
            const auto& cache = channel == 0 ? ppair_alpha : ppair_beta;
            for (std::size_t first = 0; first < 2; ++first) {
              for (std::size_t second = 0; second < 2; ++second) {
                const auto& forward =
                    cache[(2 * bra + first) * maj_per_spin + 2 * ket + second];
                const auto coefficient = 0.25 * excitation_coeff[first][second];
                symmetric[forward.word] += coefficient * forward.coeff;
                if (bra != ket) {
                  const auto& reverse = cache[(2 * ket + first) * maj_per_spin +
                                              2 * bra + second];
                  symmetric[reverse.word] += coefficient * reverse.coeff;
                  antisymmetric[forward.word] += coefficient * forward.coeff;
                  antisymmetric[reverse.word] -= coefficient * reverse.coeff;
                }
              }
            }
          }
          ExcitationPair pair{bra, ket, {}, {}};
          for (const auto& [word, coefficient] : symmetric) {
            if (std::abs(coefficient) > 1e-15)
              pair.symmetric.emplace_back(coefficient, word);
          }
          for (const auto& [word, coefficient] : antisymmetric) {
            if (std::abs(coefficient) > 1e-15)
              pair.antisymmetric.emplace_back(coefficient, word);
          }
          pairs.push_back(std::move(pair));
        }
      }
      return pairs;
    };
    auto product = [&](const Terms& left, const Terms& right, double scale,
                       bool both_orders) {
      if (scale == 0.0) return;
      for (const auto& [left_coefficient, left_word] : left) {
        for (const auto& [right_coefficient, right_word] : right) {
          acc.accumulate_product(left_word, right_word,
                                 scale * left_coefficient * right_coefficient);
          if (both_orders)
            acc.accumulate_product(
                right_word, left_word,
                scale * right_coefficient * left_coefficient);
        }
      }
    };
    auto screened = [integral_threshold](double value) {
      return std::abs(value) < integral_threshold ? 0.0 : value;
    };
    auto same_spin = [&](const auto& pairs, bool beta) {
      auto process = [&](std::size_t left, std::size_t right, double direct,
                         double ket_swapped) {
        const auto& first = pairs[left];
        const auto& second = pairs[right];
        direct = screened(direct);
        ket_swapped = screened(ket_swapped);
        product(first.symmetric, second.symmetric,
                0.25 * (direct + ket_swapped), left != right);
        product(first.antisymmetric, second.antisymmetric,
                0.25 * (direct - ket_swapped), left != right);
      };
      if constexpr (EriProvider::kSparse) {
        std::vector<std::size_t> pair_indices(n_spatial * n_spatial);
        for (std::size_t index = 0; index < pairs.size(); ++index)
          pair_indices[pairs[index].bra * n_spatial + pairs[index].ket] = index;
        for (const auto& entry : eri_provider.entries())
          process(pair_indices[entry.p * n_spatial + entry.q],
                  pair_indices[entry.r * n_spatial + entry.s], entry.value,
                  entry.ket_swapped_value);
        return;
      }
      for (std::size_t left = 0; left < pairs.size(); ++left) {
        const auto& first = pairs[left];
        const double* row = beta ? eri_provider.row_bbbb(first.bra, first.ket)
                                 : eri_provider.row_aaaa(first.bra, first.ket);
        for (std::size_t right = left; right < pairs.size(); ++right) {
          const auto& second = pairs[right];
          process(left, right, row[second.bra * n_spatial + second.ket],
                  row[second.ket * n_spatial + second.bra]);
        }
      }
    };
    for (std::size_t bra = 0; bra < n_spatial; ++bra) {
      for (std::size_t ket = 0; ket < n_spatial; ++ket) {
        for (std::size_t contracted = 0; contracted < n_spatial; ++contracted) {
          const double alpha =
              screened(eri_provider.aaaa(bra, contracted, contracted, ket));
          if (alpha != 0.0)
            accumulate_epq(mode_alpha(bra), mode_alpha(ket), -0.5 * alpha);
          if (num_spin_species == 2) {
            const double beta =
                screened(eri_provider.bbbb(bra, contracted, contracted, ket));
            if (beta != 0.0)
              accumulate_epq(mode_beta(bra), mode_beta(ket), -0.5 * beta);
          }
        }
      }
    }
    if (spin_symmetric && num_spin_species == 2) {
      same_spin(make_pairs(true, 0), false);
    } else {
      const auto alpha = make_pairs(false, 0);
      same_spin(alpha, false);
      if (num_spin_species == 2) {
        const auto beta = make_pairs(false, 1);
        same_spin(beta, true);
        for (const auto& first : alpha) {
          const double* row = eri_provider.row_aabb(first.bra, first.ket);
          for (const auto& second : beta) {
            const double direct =
                screened(row[second.bra * n_spatial + second.ket]);
            const double ket_swapped =
                screened(row[second.ket * n_spatial + second.bra]);
            product(first.symmetric, second.symmetric,
                    0.5 * (direct + ket_swapped), false);
            product(first.antisymmetric, second.antisymmetric,
                    0.5 * (direct - ket_swapped), false);
          }
        }
      }
    }
  } else if (use_spin_symmetric) {
    struct SpinSummedE {
      std::vector<std::pair<std::complex<double>, PackedPauliWord<NW>>> terms;
    };
    std::vector<SpinSummedE> ss_e(n_spatial * n_spatial);

    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        auto& sse = ss_e[p * n_spatial + q];
        for (int a = 0; a < 2; ++a) {
          for (int b = 0; b < 2; ++b) {
            const auto& [coeff_a, word_a] = alpha_pair(2 * p + a, 2 * q + b);
            sse.terms.emplace_back(coeff_a * 0.25 * excitation_coeff[a][b],
                                   word_a);
            const auto& [coeff_b, word_b] = beta_pair(2 * p + a, 2 * q + b);
            sse.terms.emplace_back(coeff_b * 0.25 * excitation_coeff[a][b],
                                   word_b);
          }
        }
      }
    }

    struct SymmetrizedE {
      std::vector<std::pair<std::complex<double>, PackedPauliWord<NW>>> terms;
    };
    std::vector<SymmetrizedE> sym_e;
    std::vector<std::size_t> sym_map(n_spatial * n_spatial);
    {
      std::size_t idx = 0;
      sym_e.resize(n_spatial * (n_spatial + 1) / 2);
      for (std::size_t p = 0; p < n_spatial; ++p) {
        for (std::size_t q = p; q < n_spatial; ++q) {
          sym_map[p * n_spatial + q] = idx;
          if (p != q) sym_map[q * n_spatial + p] = idx;

          std::unordered_map<PackedPauliWord<NW>, std::complex<double>,
                             PackedPauliWordHash<NW>>
              merged;
          for (const auto& [c, w] : ss_e[p * n_spatial + q].terms) {
            merged[w] += c;
          }
          if (p != q) {
            for (const auto& [c, w] : ss_e[q * n_spatial + p].terms) {
              merged[w] += c;
            }
          }
          auto& se = sym_e[idx];
          for (auto& [w, c] : merged) {
            if (std::abs(c) > 1e-15) {
              se.terms.emplace_back(c, std::move(w));
            }
          }
          ++idx;
        }
      }
    }

    // Accumulate the contribution of one canonical (pq|rs) pair.  Both the
    // dense quad loop and the sparse non-zero loop funnel through this lambda
    // so the arithmetic, ordering, and thresholding are byte-for-byte
    // identical between paths.  Pre-condition: pq_idx <= rs_idx.
    auto process_pair = [&](std::size_t pq_idx, std::size_t rs_idx,
                            double eri) {
      if (std::abs(eri) < integral_threshold) return;
      const auto& s_pq = sym_e[pq_idx];
      const auto& s_rs = sym_e[rs_idx];
      double half_eri = 0.5 * eri;
      if (pq_idx == rs_idx) {
        for (const auto& [c1, w1] : s_pq.terms) {
          for (const auto& [c2, w2] : s_rs.terms) {
            acc.accumulate_product(w1, w2, half_eri * c1 * c2);
          }
        }
      } else {
        for (const auto& [c1, w1] : s_pq.terms) {
          for (const auto& [c2, w2] : s_rs.terms) {
            acc.accumulate_product(w1, w2, half_eri * c1 * c2);
            acc.accumulate_product(w2, w1, half_eri * c2 * c1);
          }
        }
      }
    };

    if constexpr (EriProvider::kSparse) {
      // Fast path: visit only the stored non-zero entries.  Entries are
      // canonicalized (p<=q, r<=s, (p,q)<=(r,s)) at ingestion; enforce the
      // same pq_idx <= rs_idx ordering the dense quad loop uses before
      // calling process_pair.
      for (const auto& e : eri_provider.entries()) {
        std::size_t pq_idx = sym_map[e.p * n_spatial + e.q];
        std::size_t rs_idx = sym_map[e.r * n_spatial + e.s];
        if (pq_idx > rs_idx) std::swap(pq_idx, rs_idx);
        process_pair(pq_idx, rs_idx, e.value);
      }
    } else {
      for (std::size_t p = 0; p < n_spatial; ++p) {
        for (std::size_t q = p; q < n_spatial; ++q) {
          std::size_t pq_idx = sym_map[p * n_spatial + q];
          // One (pq|·) row per outer pair: zero-copy for dense, a single
          // vectorized aux contraction for Cholesky.
          const double* row = eri_provider.row_aaaa(p, q);
          for (std::size_t r = 0; r < n_spatial; ++r) {
            for (std::size_t s = r; s < n_spatial; ++s) {
              std::size_t rs_idx = sym_map[r * n_spatial + s];
              if (pq_idx > rs_idx) continue;
              process_pair(pq_idx, rs_idx, row[r * n_spatial + s]);
            }
          }
        }
      }
    }

  } else {
    auto accumulate_two_body_product =
        [&](std::size_t mode_p, std::size_t mode_q, std::size_t mode_r,
            std::size_t mode_s, double eri) {
          bool pq_is_alpha = mode_p < n_spatial;
          bool rs_is_alpha = mode_r < n_spatial;
          auto& cache_pq = pq_is_alpha ? ppair_alpha : ppair_beta;
          auto& cache_rs = rs_is_alpha ? ppair_alpha : ppair_beta;
          std::size_t bp = pq_is_alpha ? mode_p : (mode_p - n_spatial);
          std::size_t bq = pq_is_alpha ? mode_q : (mode_q - n_spatial);
          std::size_t br = rs_is_alpha ? mode_r : (mode_r - n_spatial);
          std::size_t bs = rs_is_alpha ? mode_s : (mode_s - n_spatial);

          double half_eri = 0.5 * eri;
          for (int a = 0; a < 2; ++a) {
            for (int b = 0; b < 2; ++b) {
              const auto& [coeff1, w1] =
                  cache_pq[(2 * bp + a) * maj_per_spin + (2 * bq + b)];
              for (int c = 0; c < 2; ++c) {
                for (int d = 0; d < 2; ++d) {
                  const auto& [coeff2, w2] =
                      cache_rs[(2 * br + c) * maj_per_spin + (2 * bs + d)];
                  std::complex<double> scale = coeff1 * coeff2 * half_eri *
                                               0.0625 * excitation_coeff[a][b] *
                                               excitation_coeff[c][d];
                  acc.accumulate_product(w1, w2, scale);
                }
              }
            }
          }
        };

    // αα channel
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        const double* row = eri_provider.row_aaaa(p, q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = row[r * n_spatial + s];
            if (std::abs(eri) < integral_threshold) continue;
            accumulate_two_body_product(mode_alpha(p), mode_alpha(q),
                                        mode_alpha(r), mode_alpha(s), eri);
            if (q == r) {
              accumulate_epq(mode_alpha(p), mode_alpha(s), -0.5 * eri);
            }
          }
        }
      }
    }

    // ββ channel
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        const double* row = eri_provider.row_bbbb(p, q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = row[r * n_spatial + s];
            if (std::abs(eri) < integral_threshold) continue;
            accumulate_two_body_product(mode_beta(p), mode_beta(q),
                                        mode_beta(r), mode_beta(s), eri);
            if (q == r) {
              accumulate_epq(mode_beta(p), mode_beta(s), -0.5 * eri);
            }
          }
        }
      }
    }

    // αβ + βα cross-spin channels, related by Coulomb symmetry (pq|rs)=(rs|pq)
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        const double* row = eri_provider.row_aabb(p, q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double eri = row[r * n_spatial + s];
            if (std::abs(eri) < integral_threshold) continue;
            accumulate_two_body_product(mode_alpha(p), mode_alpha(q),
                                        mode_beta(r), mode_beta(s), eri);
            accumulate_two_body_product(mode_beta(r), mode_beta(s),
                                        mode_alpha(p), mode_alpha(q), eri);
          }
        }
      }
    }
  }

  if (!mapping.stabilizers().empty()) {
    double h1_sum = 0.0;
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        h1_sum += std::abs(h1_alpha[p * n_spatial + q]);
        if (!spin_symmetric) {
          h1_sum += std::abs(h1_beta[p * n_spatial + q]);
        } else {
          h1_sum += std::abs(h1_alpha[p * n_spatial + q]);
        }
      }
    }

    // If stabilizers are included in the mapping, our total Hamiltonian becomes
    // H_total = H_physical + H_aux. We add H_aux = λ · Σ_i (I − S_i S_{i+1})
    // (sum of products of adjacent stabilizers per spin channel) to penalize
    // unphysical codespace sectors using local terms.
    double aux_ham_coefficient = 1.0 + h1_sum;

    std::size_t num_stabs = mapping.stabilizers().size();
    std::size_t half_stabs = num_stabs / 2;

    struct PackedStab {
      std::complex<double> coeff;
      PackedPauliWord<NW> word;
    };
    std::vector<PackedStab> packed_stabs;
    packed_stabs.reserve(num_stabs);
    for (const auto& [coeff, word] : mapping.stabilizers()) {
      packed_stabs.push_back({coeff, sparse_to_packed<NW>(word)});
    }

    std::size_t num_penalty_terms = 0;
    if (half_stabs >= 2) {
      num_penalty_terms = 2 * (half_stabs - 1);
    } else {
      num_penalty_terms = num_stabs;
    }

    PackedPauliWord<NW> identity{};
    acc.accumulate(identity,
                   std::complex<double>(aux_ham_coefficient * num_stabs, 0.0));

    // For each spin sector, we accumulate the single boundary link stabilizer
    // individually to lift the Z_2 gauge degeneracy, followed by the products
    // of adjacent link stabilizers (paired via Minimum Spanning Tree to
    // minimize their Pauli weight) to form local loop-plaquette stabilizers.
    if (half_stabs >= 1) {
      auto get_packed_weight = [](const PackedPauliWord<NW>& pw) -> int {
        int w = 0;
        for (std::size_t k = 0; k < NW; ++k) {
          w += std::popcount(pw.x[k] | pw.z[k]);
        }
        return w;
      };

      auto accumulate_product = [&](std::size_t i, std::size_t j) {
        const auto& s1 = packed_stabs[i];
        const auto& s2 = packed_stabs[j];
        auto [phase_idx, word] = multiply_packed(s1.word, s2.word);
        std::complex<double> prod_coeff =
            apply_phase(phase_idx, s1.coeff * s2.coeff);
        acc.accumulate(word, -aux_ham_coefficient * prod_coeff);
      };

      auto build_mst_and_accumulate = [&](std::size_t start_idx,
                                          std::size_t count) {
        if (count == 0) return;

        // Find the lightest stabilizer as root
        std::size_t root = 0;
        int root_weight = get_packed_weight(packed_stabs[start_idx].word);
        for (std::size_t i = 1; i < count; ++i) {
          int w = get_packed_weight(packed_stabs[start_idx + i].word);
          if (w < root_weight) {
            root = i;
            root_weight = w;
          }
        }

        // Accumulate the root boundary link stabilizer
        acc.accumulate(
            packed_stabs[start_idx + root].word,
            -aux_ham_coefficient * packed_stabs[start_idx + root].coeff);

        if (count <= 1) return;

        // Prim's algorithm to find the minimum spanning tree of stabilizer
        // pairings starting from the chosen root
        std::vector<bool> in_mst(count, false);
        std::vector<int> min_weight(count, 1e9);
        std::vector<std::size_t> parent(count, root);

        min_weight[root] = 0;

        for (std::size_t step = 0; step < count; ++step) {
          std::size_t u = 0;
          int min_val = 1e9;
          for (std::size_t i = 0; i < count; ++i) {
            if (!in_mst[i] && min_weight[i] < min_val) {
              min_val = min_weight[i];
              u = i;
            }
          }

          in_mst[u] = true;

          for (std::size_t v = 0; v < count; ++v) {
            if (!in_mst[v]) {
              auto [phase_idx, word] =
                  multiply_packed(packed_stabs[start_idx + u].word,
                                  packed_stabs[start_idx + v].word);
              int w = get_packed_weight(word);
              if (w < min_weight[v]) {
                min_weight[v] = w;
                parent[v] = u;
              }
            }
          }
        }

        // Accumulate products for the selected MST edges
        for (std::size_t v = 0; v < count; ++v) {
          if (v != root) {
            accumulate_product(start_idx + parent[v], start_idx + v);
          }
        }
      };

      // Construct MST for α spin Majoranas
      build_mst_and_accumulate(0, half_stabs);

      // Construct MST for β spin Majoranas
      build_mst_and_accumulate(half_stabs, half_stabs);

    } else {
      // Fallback to individual stabilizers if too few to form products
      for (const auto& stab : packed_stabs) {
        acc.accumulate(stab.word, -aux_ham_coefficient * stab.coeff);
      }
    }
  }

  // ─── Extract results as sparse Pauli words ──────────────────────
  auto terms = acc.get_terms(threshold);

  MajoranaMapResult result;
  result.words.reserve(terms.size());
  result.coefficients.reserve(terms.size());

  for (auto& [coeff, word] : terms) {
    result.words.push_back(std::move(word));
    result.coefficients.push_back(coeff);
  }

  return result;
}

constexpr std::size_t max_nw = 16;

// Runtime dispatch over the number of 64-bit words needed for the qubit
// count, templated on the ERI provider so each storage format reuses the
// identical mapping implementation.
template <class EriProvider, std::size_t... Is>
MajoranaMapResult dispatch_by_words(
    std::index_sequence<Is...>, std::size_t num_words,
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const EriProvider& eri_provider,
    std::size_t n_spatial, bool spin_symmetric, bool pair_split,
    double threshold, double integral_threshold) {
  using Fn = MajoranaMapResult (*)(
      const MajoranaMapping&, double, const double*, const double*,
      const EriProvider&, std::size_t, bool, bool, double, double);
  static const std::array<Fn, sizeof...(Is)> table = {
      {&majorana_map_impl<Is + 1, EriProvider>...}};
  return table[num_words - 1](mapping, core_energy, h1_alpha, h1_beta,
                              eri_provider, n_spatial, spin_symmetric,
                              pair_split, threshold, integral_threshold);
}

template <class EriProvider>
MajoranaMapResult run_with_provider(const MajoranaMapping& mapping,
                                    double core_energy, const double* h1_alpha,
                                    const double* h1_beta,
                                    const EriProvider& eri_provider,
                                    std::size_t n_spatial, bool spin_symmetric,
                                    bool pair_split, double threshold,
                                    double integral_threshold) {
  const std::size_t num_qubits = mapping.num_qubits();
  if (num_qubits == 0) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian: mapping has zero qubits; the encoding "
        "must produce at least one qubit.");
  }
  const std::size_t num_words = (num_qubits + 63) / 64;

  if (num_words > max_nw) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian: num_qubits=" + std::to_string(num_qubits) +
        " exceeds the maximum of " + std::to_string(max_nw * 64) + " qubits.");
  }

  return dispatch_by_words<EriProvider>(
      std::make_index_sequence<max_nw>{}, num_words, mapping, core_energy,
      h1_alpha, h1_beta, eri_provider, n_spatial, spin_symmetric, pair_split,
      threshold, integral_threshold);
}

}  // namespace detail

MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool spin_symmetric,
    double threshold, double integral_threshold) {
  detail::DenseEriProvider provider{eri_aaaa, eri_aabb, eri_bbbb, n_spatial};
  bool pair_split = false;
  if (spin_symmetric) {
    pair_split = detail::require_fourfold_two_body(
                     eri_aaaa, n_spatial, "majorana_map_hamiltonian") ==
                 detail::TwoBodySymmetry::FourFold;
  } else {
    auto supports_split = [n_spatial](const double* tensor) {
      const auto symmetry = detail::classify_two_body_symmetry(
          tensor, n_spatial, detail::two_body_symmetry_tolerance);
      return symmetry == detail::TwoBodySymmetry::FourFold ||
             symmetry == detail::TwoBodySymmetry::EightFold;
    };
    pair_split = supports_split(eri_aaaa) && supports_split(eri_bbbb) &&
                 detail::two_body_has_reversal_symmetry(
                     eri_aabb, n_spatial, detail::two_body_symmetry_tolerance);
  }
  return detail::run_with_provider(mapping, core_energy, h1_alpha, h1_beta,
                                   provider, n_spatial, spin_symmetric,
                                   pair_split, threshold, integral_threshold);
}

MajoranaMapResult majorana_map_hamiltonian_cholesky(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* three_center_aa,
    const double* three_center_bb, std::size_t n_spatial, std::size_t naux,
    bool spin_symmetric, double threshold, double integral_threshold) {
  const auto alpha_symmetry = detail::cholesky_two_body_symmetry(
      three_center_aa, n_spatial, naux, detail::two_body_symmetry_tolerance);
  if (spin_symmetric &&
      alpha_symmetry == detail::TwoBodySymmetry::NonHermitian) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian_cholesky: pair-symmetric and "
        "pair-antisymmetric "
        "factor components have a nonzero cross product, so the reconstructed "
        "two-body integrals define a non-Hermitian operator.");
  }
  const bool pair_split =
      spin_symmetric
          ? alpha_symmetry == detail::TwoBodySymmetry::FourFold
          : alpha_symmetry != detail::TwoBodySymmetry::NonHermitian &&
                detail::cholesky_two_body_symmetry(
                    three_center_bb, n_spatial, naux,
                    detail::two_body_symmetry_tolerance) !=
                    detail::TwoBodySymmetry::NonHermitian &&
                detail::cholesky_mixed_spin_has_reversal_symmetry(
                    three_center_aa, three_center_bb, n_spatial, naux,
                    detail::two_body_symmetry_tolerance);
  detail::CholeskyEriProvider provider{three_center_aa, three_center_bb,
                                       n_spatial, naux};
  return detail::run_with_provider(mapping, core_energy, h1_alpha, h1_beta,
                                   provider, n_spatial, spin_symmetric,
                                   pair_split, threshold, integral_threshold);
}

MajoranaMapResult majorana_map_hamiltonian_sparse(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const int* two_body_indices,
    const double* two_body_values, std::size_t num_entries,
    std::size_t n_spatial, bool spin_symmetric, double threshold,
    double integral_threshold) {
  // The sparse fast path is spin-summed: SparseHamiltonianContainer is
  // restricted-only, so all spin channels share the same integrals and the
  // unrestricted branch would only redo identical work per channel.  Callers
  // with unrestricted orbitals must use the dense entry point instead.
  if (!spin_symmetric) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian_sparse: the sparse fast path requires "
        "spin_symmetric=true (restricted orbitals); use "
        "majorana_map_hamiltonian with dense integrals for unrestricted "
        "Hamiltonians.");
  }

  detail::SparseEriProvider provider;
  provider.n_ = n_spatial;

  detail::SparseTwoBodyTensor tensor;
  for (std::size_t e = 0; e < num_entries; ++e) {
    // Validate before the int -> size_t cast: a negative index would wrap
    // to a huge value and read out of bounds downstream (e.g. in sym_map).
    for (std::size_t component = 0; component < 4; ++component) {
      const int idx = two_body_indices[4 * e + component];
      if (idx < 0 || static_cast<std::size_t>(idx) >= n_spatial) {
        throw std::invalid_argument(
            "majorana_map_hamiltonian_sparse: two-body index " +
            std::to_string(idx) + " of entry " + std::to_string(e) +
            " is outside [0, " + std::to_string(n_spatial) + ").");
      }
    }
    const detail::TwoBodyKey k{
        static_cast<std::size_t>(two_body_indices[4 * e + 0]),
        static_cast<std::size_t>(two_body_indices[4 * e + 1]),
        static_cast<std::size_t>(two_body_indices[4 * e + 2]),
        static_cast<std::size_t>(two_body_indices[4 * e + 3])};
    const double v = two_body_values[e];
    if (!std::isfinite(v)) {
      throw std::invalid_argument(
          "majorana_map_hamiltonian_sparse: two-body entry " +
          std::to_string(e) + " is not finite.");
    }
    // Duplicates must agree bitwise: a tolerance would keep whichever of two
    // genuinely different values came first, making the result order-dependent.
    auto [it, inserted] = tensor.try_emplace(k, v);
    if (!inserted && it->second != v) {
      throw std::invalid_argument(
          "majorana_map_hamiltonian_sparse: conflicting duplicate values "
          "for two-body entry " +
          detail::two_body_key_string(k) + ".");
    }
  }

  detail::expand_legacy_eightfold_representatives(
      tensor, "majorana_map_hamiltonian_sparse");

  const auto classification = detail::classify_sparse_two_body_symmetry(
      tensor, detail::two_body_symmetry_tolerance);
  const auto& [p, q, r, s] = classification.witness;
  if (classification.symmetry == detail::TwoBodySymmetry::NonHermitian) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian_sparse: at " +
        detail::two_body_key_string(classification.witness) +
        ", (pq|rs) + (rs|pq) != (qp|sr) + (sr|qp) with absent entries read as "
        "zero, so the two-body integrals define a non-Hermitian operator, "
        "which this engine cannot map. Store the missing symmetry partners, "
        "or expand symmetry representatives with "
        "SparseHamiltonianContainer.expand_symmetry_representatives.");
  }
  if (classification.symmetry == detail::TwoBodySymmetry::NotBraKetSymmetric) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian_sparse: the two-body integrals are "
        "Hermitian but not stored with (pq|rs) = (rs|pq): " +
        detail::two_body_key_string(classification.witness) + " and " +
        detail::two_body_key_string({r, s, p, q}) +
        " differ, with absent entries read as zero. That swap leaves the "
        "operator unchanged, so store the bra-ket average ((pq|rs) + (rs|pq)) "
        "/ 2 at both positions.");
  }
  const bool pair_split =
      classification.symmetry == detail::TwoBodySymmetry::FourFold;

  provider.map_.reserve(tensor.size());
  std::set<detail::TwoBodyKey> canonical_keys;
  for (const auto& [key, value] : tensor) {
    provider.map_[provider.key(key[0], key[1], key[2], key[3])] = value;
    canonical_keys.insert(detail::eightfold_canonical(key));
  }
  // One canonical (p<=q, r<=s, (p,q)<=(r,s)) entry per class, in
  // lexicographic order, so accumulation is independent of input order.
  provider.entries_.reserve(canonical_keys.size());
  for (const auto& key : canonical_keys) {
    provider.entries_.push_back(
        {key[0], key[1], key[2], key[3],
         provider.aaaa(key[0], key[1], key[2], key[3]),
         provider.aaaa(key[0], key[1], key[3], key[2])});
  }

  return detail::run_with_provider(mapping, core_energy, h1_alpha, h1_beta,
                                   provider, n_spatial, spin_symmetric,
                                   pair_split, threshold, integral_threshold);
}

MajoranaMapResult majorana_map_hamiltonian(const MajoranaMapping& mapping,
                                           const Hamiltonian& hamiltonian,
                                           bool spin_symmetric,
                                           double threshold,
                                           double integral_threshold) {
  constexpr double core_energy = 0.0;

  auto one_body = hamiltonian.get_one_body_integrals();
  const Eigen::MatrixXd& h1a = std::get<0>(one_body);
  const Eigen::MatrixXd& h1b = std::get<1>(one_body);
  const std::size_t n = static_cast<std::size_t>(h1a.rows());

  using RowMajorMatrix =
      Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
  const RowMajorMatrix h1a_flat = h1a;
  const double* h1b_ptr = nullptr;
  RowMajorMatrix h1b_flat;
  if (spin_symmetric) {
    h1b_ptr = h1a_flat.data();
  } else {
    h1b_flat = h1b;
    h1b_ptr = h1b_flat.data();
  }

  if (hamiltonian.has_container_type<CholeskyHamiltonianContainer>()) {
    const auto& container =
        hamiltonian.get_container<CholeskyHamiltonianContainer>();
    const auto three_center = container.get_three_center_integrals();
    const Eigen::MatrixXd& three_center_aa = three_center.first;
    const Eigen::MatrixXd& three_center_bb = three_center.second;
    const std::size_t naux = static_cast<std::size_t>(three_center_aa.cols());
    return majorana_map_hamiltonian_cholesky(
        mapping, core_energy, h1a_flat.data(), h1b_ptr, three_center_aa.data(),
        three_center_bb.data(), n, naux, spin_symmetric, threshold,
        integral_threshold);
  }

  if (hamiltonian.has_container_type<SparseHamiltonianContainer>()) {
    const auto& container =
        hamiltonian.get_container<SparseHamiltonianContainer>();
    const auto& two_body_map = container.sparse_two_body_integrals();
    static_assert(
        std::is_same_v<SparseHamiltonianContainer::TwoBodyIndex,
                       std::tuple<int, int, int, int>>,
        "sparse two-body indices must be int to match the engine API");
    std::vector<int> indices;
    std::vector<double> values;
    indices.reserve(two_body_map.size() * 4);
    values.reserve(two_body_map.size());
    for (const auto& [idx, val] : two_body_map) {
      const auto& [p, q, r, s] = idx;
      indices.push_back(p);
      indices.push_back(q);
      indices.push_back(r);
      indices.push_back(s);
      values.push_back(val);
    }
    return majorana_map_hamiltonian_sparse(
        mapping, core_energy, h1a_flat.data(), h1b_ptr, indices.data(),
        values.data(), values.size(), n, spin_symmetric, threshold,
        integral_threshold);
  }

  const auto two_body = hamiltonian.get_two_body_integrals();
  const Eigen::VectorXd& aaaa = std::get<0>(two_body);
  const Eigen::VectorXd& aabb = std::get<1>(two_body);
  const Eigen::VectorXd& bbbb = std::get<2>(two_body);
  return majorana_map_hamiltonian(
      mapping, core_energy, h1a_flat.data(), h1b_ptr, aaaa.data(), aabb.data(),
      bbbb.data(), n, spin_symmetric, threshold, integral_threshold);
}

}  // namespace qdk::chemistry::data

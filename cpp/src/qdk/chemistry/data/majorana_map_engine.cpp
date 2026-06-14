// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <array>
#include <bit>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/cholesky.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/sparse.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <qdk/chemistry/utils/hash_context.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

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

// ─── EriProvider Architectures ──────────────────────────────────────

class DenseEriProvider {
 public:
  static constexpr bool kSparse = false;
  DenseEriProvider(const double* aaaa_ptr, const double* aabb_ptr,
                   const double* bbbb_ptr, size_t n_spatial)
      : aaaa_ptr_(aaaa_ptr),
        aabb_ptr_(aabb_ptr),
        bbbb_ptr_(bbbb_ptr),
        n_spatial_(n_spatial) {}

  void build_row_aaaa(size_t ij) { row_offset_ = ij * n_spatial_ * n_spatial_; }
  void build_row_aabb(size_t ij) { row_offset_ = ij * n_spatial_ * n_spatial_; }
  void build_row_bbbb(size_t ij) { row_offset_ = ij * n_spatial_ * n_spatial_; }

  double aaaa(size_t kl) const { return aaaa_ptr_[row_offset_ + kl]; }
  double aabb(size_t kl) const { return aabb_ptr_[row_offset_ + kl]; }
  double bbbb(size_t kl) const { return bbbb_ptr_[row_offset_ + kl]; }

  double get_element(size_t p, size_t q, size_t r, size_t s,
                     SpinChannel channel) const {
    size_t idx = ((p * n_spatial_ + q) * n_spatial_ + r) * n_spatial_ + s;
    if (channel == SpinChannel::aaaa) return aaaa_ptr_[idx];
    if (channel == SpinChannel::aabb) return aabb_ptr_[idx];
    if (channel == SpinChannel::bbbb) return bbbb_ptr_[idx];
    return 0.0;
  }

 private:
  const double* aaaa_ptr_;
  const double* aabb_ptr_;
  const double* bbbb_ptr_;
  size_t n_spatial_;
  size_t row_offset_ = 0;
};

class CholeskyEriProvider {
 public:
  static constexpr bool kSparse = false;
  CholeskyEriProvider(const Eigen::MatrixXd& L_alpha,
                      const Eigen::MatrixXd& L_beta, size_t n_spatial,
                      bool restricted)
      : L_alpha_(L_alpha),
        L_beta_(L_beta),
        n_spatial_(n_spatial),
        norb2_(n_spatial * n_spatial),
        restricted_(restricted) {
    aaaa_row_.resize(norb2_);
    if (!restricted_) {
      aabb_row_.resize(norb2_);
      bbbb_row_.resize(norb2_);
    }
  }

  void build_row_aaaa(size_t ij) {
    if (ij == cached_aaaa_ij_) return;
    build_row_impl(ij, aaaa_row_.data(), L_alpha_, L_alpha_);
    cached_aaaa_ij_ = ij;
  }

  void build_row_aabb(size_t ij) {
    if (restricted_) return;
    if (ij == cached_aabb_ij_) return;
    build_row_impl(ij, aabb_row_.data(), L_alpha_, L_beta_);
    cached_aabb_ij_ = ij;
  }

  void build_row_bbbb(size_t ij) {
    if (restricted_) return;
    if (ij == cached_bbbb_ij_) return;
    build_row_impl(ij, bbbb_row_.data(), L_beta_, L_beta_);
    cached_bbbb_ij_ = ij;
  }

  double aaaa(size_t kl) const { return aaaa_row_[kl]; }
  double aabb(size_t kl) const {
    return restricted_ ? aaaa_row_[kl] : aabb_row_[kl];
  }
  double bbbb(size_t kl) const {
    return restricted_ ? aaaa_row_[kl] : bbbb_row_[kl];
  }

  double get_element(size_t p, size_t q, size_t r, size_t s,
                     SpinChannel channel) const {
    size_t ij = p * n_spatial_ + q;
    size_t kl = r * n_spatial_ + s;
    if (restricted_) {
      return L_alpha_.row(ij).dot(L_alpha_.row(kl));
    }
    if (channel == SpinChannel::aaaa) {
      return L_alpha_.row(ij).dot(L_alpha_.row(kl));
    } else if (channel == SpinChannel::bbbb) {
      return L_beta_.row(ij).dot(L_beta_.row(kl));
    } else if (channel == SpinChannel::aabb) {
      return L_alpha_.row(ij).dot(L_beta_.row(kl));
    }
    return 0.0;
  }

 private:
  void build_row_impl(size_t ij, double* out_buffer,
                      const Eigen::MatrixXd& L_left,
                      const Eigen::MatrixXd& L_right) {
    Eigen::Map<Eigen::VectorXd> out_view(out_buffer, norb2_);
    out_view.setZero();
    size_t naux = L_right.cols();
    for (size_t Q = 0; Q < naux; ++Q) {
      double w = L_left(ij, Q);
      if (std::abs(w) < 1e-15) continue;
      Eigen::Map<const Eigen::VectorXd> col_view(L_right.data() + Q * norb2_,
                                                 norb2_);
      out_view += w * col_view;
    }
  }

  const Eigen::MatrixXd& L_alpha_;
  const Eigen::MatrixXd& L_beta_;
  size_t n_spatial_;
  size_t norb2_;
  bool restricted_;
  std::vector<double> aaaa_row_;
  std::vector<double> aabb_row_;
  std::vector<double> bbbb_row_;
  size_t cached_aaaa_ij_ = -1;
  size_t cached_aabb_ij_ = -1;
  size_t cached_bbbb_ij_ = -1;
};

class SparseEriProvider {
 public:
  static constexpr bool kSparse = true;

  struct Entry {
    unsigned p, q, r, s;
    double value;

    bool operator<(const Entry& other) const {
      if (p != other.p) return p < other.p;
      if (q != other.q) return q < other.q;
      if (r != other.r) return r < other.r;
      return s < other.s;
    }
  };

  SparseEriProvider(const SymmetryBlockedSparseMap<4>* two_body,
                    size_t n_spatial)
      : n_spatial_(n_spatial) {
    if (two_body && two_body->num_blocks() > 0) {
      const auto& block = two_body->block(
          {axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()});

      // Sort and deduplicate
      std::map<Entry, double> sorted_entries;
      for (const auto& [idx, val] : block) {
        Entry e = canonicalize(idx[0], idx[1], idx[2], idx[3]);
        sorted_entries[e] = val;
      }

      entries_.reserve(sorted_entries.size());
      map_.reserve(sorted_entries.size());
      for (const auto& [e, val] : sorted_entries) {
        Entry full_entry = e;
        full_entry.value = val;
        entries_.push_back(full_entry);
        uint64_t key = pack_key(e.p, e.q, e.r, e.s);
        map_[key] = val;
      }
    }
  }

  const std::vector<Entry>& entries() const { return entries_; }

  void build_row_aaaa(size_t ij) {
    p_ = ij / n_spatial_;
    q_ = ij % n_spatial_;
  }
  void build_row_aabb(size_t ij) { build_row_aaaa(ij); }
  void build_row_bbbb(size_t ij) { build_row_aaaa(ij); }

  double aaaa(size_t kl) const {
    unsigned r = kl / n_spatial_;
    unsigned s = kl % n_spatial_;
    uint64_t key = canonical_key(p_, q_, r, s);
    auto it = map_.find(key);
    return it != map_.end() ? it->second : 0.0;
  }

  double aabb(size_t kl) const { return aaaa(kl); }
  double bbbb(size_t kl) const { return aaaa(kl); }

  double get_element(size_t p, size_t q, size_t r, size_t s,
                     SpinChannel) const {
    uint64_t key = canonical_key(p, q, r, s);
    auto it = map_.find(key);
    return it != map_.end() ? it->second : 0.0;
  }

 private:
  static inline Entry canonicalize(unsigned p, unsigned q, unsigned r,
                                   unsigned s) {
    unsigned p_c = p;
    unsigned q_c = q;
    if (p_c < q_c) std::swap(p_c, q_c);
    unsigned r_c = r;
    unsigned s_c = s;
    if (r_c < s_c) std::swap(r_c, s_c);
    if (p_c < r_c || (p_c == r_c && q_c < s_c)) {
      std::swap(p_c, r_c);
      std::swap(q_c, s_c);
    }
    return Entry{p_c, q_c, r_c, s_c, 0.0};
  }

  static inline uint64_t canonical_key(unsigned p, unsigned q, unsigned r,
                                       unsigned s) {
    unsigned p_c = p;
    unsigned q_c = q;
    if (p_c < q_c) std::swap(p_c, q_c);
    unsigned r_c = r;
    unsigned s_c = s;
    if (r_c < s_c) std::swap(r_c, s_c);
    if (p_c < r_c || (p_c == r_c && q_c < s_c)) {
      std::swap(p_c, r_c);
      std::swap(q_c, s_c);
    }
    return pack_key(p_c, q_c, r_c, s_c);
  }

  static inline uint64_t pack_key(unsigned p, unsigned q, unsigned r,
                                  unsigned s) {
    return (static_cast<uint64_t>(p) << 48) | (static_cast<uint64_t>(q) << 32) |
           (static_cast<uint64_t>(r) << 16) | (static_cast<uint64_t>(s));
  }

  size_t n_spatial_;
  unsigned p_ = 0;
  unsigned q_ = 0;
  std::vector<Entry> entries_;
  std::unordered_map<uint64_t, double> map_;
};

// ─── Engine implementation ─────────────────────────────────────────
//
// The standard non-relativistic electronic Hamiltonian is spin-free: it
// contains no spin operators.  The Majorana decomposition of excitation
// operators E_pq = a†_p a_q into bilinear products of γ's works for
// any spin-free Hamiltonian regardless of the orbital basis.
//
// The spin_symmetric flag selects one of two evaluation strategies:
//
//   spin_symmetric = true   (restricted orbitals)
//     All spin channels share the same integrals (h_alpha == h_beta,
//     eri_aaaa == eri_bbbb == eri_aabb).  The engine precomputes
//     spin-summed E_pq = E^α_pq + E^β_pq and exploits 8-fold ERI
//     symmetry, roughly halving the two-body work.  The δ_{qr}
//     two-body contraction is folded into the one-body integrals.
//
//   spin_symmetric = false  (unrestricted orbitals)
//     Each spin channel (αα, ββ, αβ, βα) is handled independently
//     with its own integrals.  This is needed for unrestricted orbitals,
//     where h^α ≠ h^β and the ERI channels differ — even though the
//     underlying Hamiltonian is still spin-free.  The αβ and βα
//     cross-spin channels are related by Coulomb symmetry (pq|rs) =
//     (rs|pq) and are handled in a single merged loop.

template <std::size_t NW, typename TEriProvider>
MajoranaMapResult majorana_map_impl(const MajoranaMapping& mapping,
                                    double core_energy, const double* h1_alpha,
                                    const double* h1_beta, TEriProvider& eri,
                                    std::size_t n_spatial, bool spin_symmetric,
                                    double threshold,
                                    double integral_threshold) {
  if (mapping.num_modes() != 2 * n_spatial) {
    throw std::invalid_argument(
        "majorana_map_hamiltonian: mapping num_modes (" +
        std::to_string(mapping.num_modes()) + ") must match 2 * n_spatial (" +
        std::to_string(2 * n_spatial) + ")");
  }

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

  auto ppair_alpha = build_pair_cache(alpha_offset);
  auto ppair_beta = build_pair_cache(beta_offset);

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
      double h_b = spin_symmetric ? h_a : h1_beta[p * n_spatial + s];
      if (spin_symmetric) {
        double delta_corr = 0.0;
        for (std::size_t q = 0; q < n_spatial; ++q) {
          delta_corr += eri.get_element(p, q, q, s, SpinChannel::aaaa);
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
      double h_pq_b = h1_eff_beta[p * n_spatial + q];
      if (std::abs(h_pq_b) > integral_threshold) {
        accumulate_epq(mode_beta(p), mode_beta(q), h_pq_b);
      }
    }
  }

  // ─── Two-body terms ───────────────────────────────────────────────

  if (spin_symmetric) {
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

    if constexpr (TEriProvider::kSparse) {
      for (const auto& e : eri.entries()) {
        std::size_t pq_idx = sym_map[e.p * n_spatial + e.q];
        std::size_t rs_idx = sym_map[e.r * n_spatial + e.s];
        std::size_t idx1 = pq_idx;
        std::size_t idx2 = rs_idx;
        if (idx1 > idx2) std::swap(idx1, idx2);

        double val = e.value;
        if (std::abs(val) < integral_threshold) continue;

        const auto& s_pq = sym_e[idx1];
        const auto& s_rs = sym_e[idx2];

        if (idx1 == idx2) {
          double half_eri = 0.5 * val;
          for (const auto& [c1, w1] : s_pq.terms) {
            for (const auto& [c2, w2] : s_rs.terms) {
              acc.accumulate_product(w1, w2, half_eri * c1 * c2);
            }
          }
        } else {
          double half_eri = 0.5 * val;
          for (const auto& [c1, w1] : s_pq.terms) {
            for (const auto& [c2, w2] : s_rs.terms) {
              acc.accumulate_product(w1, w2, half_eri * c1 * c2);
              acc.accumulate_product(w2, w1, half_eri * c2 * c1);
            }
          }
        }
      }
    } else {
      for (std::size_t p = 0; p < n_spatial; ++p) {
        for (std::size_t q = p; q < n_spatial; ++q) {
          std::size_t pq_idx = sym_map[p * n_spatial + q];
          const auto& s_pq = sym_e[pq_idx];

          eri.build_row_aaaa(p * n_spatial + q);

          for (std::size_t r = 0; r < n_spatial; ++r) {
            for (std::size_t s = r; s < n_spatial; ++s) {
              std::size_t rs_idx = sym_map[r * n_spatial + s];
              if (pq_idx > rs_idx) continue;

              double val = eri.aaaa(r * n_spatial + s);
              if (std::abs(val) < integral_threshold) continue;

              const auto& s_rs = sym_e[rs_idx];

              if (pq_idx == rs_idx) {
                double half_eri = 0.5 * val;
                for (const auto& [c1, w1] : s_pq.terms) {
                  for (const auto& [c2, w2] : s_rs.terms) {
                    acc.accumulate_product(w1, w2, half_eri * c1 * c2);
                  }
                }
              } else {
                double half_eri = 0.5 * val;
                for (const auto& [c1, w1] : s_pq.terms) {
                  for (const auto& [c2, w2] : s_rs.terms) {
                    acc.accumulate_product(w1, w2, half_eri * c1 * c2);
                    acc.accumulate_product(w2, w1, half_eri * c2 * c1);
                  }
                }
              }
            }
          }
        }
      }
    }

  } else {
    auto accumulate_two_body_product =
        [&](std::size_t mode_p, std::size_t mode_q, std::size_t mode_r,
            std::size_t mode_s, double eri_val) {
          bool pq_is_alpha = mode_p < n_spatial;
          bool rs_is_alpha = mode_r < n_spatial;
          auto& cache_pq = pq_is_alpha ? ppair_alpha : ppair_beta;
          auto& cache_rs = rs_is_alpha ? ppair_alpha : ppair_beta;
          std::size_t bp = pq_is_alpha ? mode_p : (mode_p - n_spatial);
          std::size_t bq = pq_is_alpha ? mode_q : (mode_q - n_spatial);
          std::size_t br = rs_is_alpha ? mode_r : (mode_r - n_spatial);
          std::size_t bs = rs_is_alpha ? mode_s : (mode_s - n_spatial);

          double half_eri = 0.5 * eri_val;
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
        eri.build_row_aaaa(p * n_spatial + q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double val = eri.aaaa(r * n_spatial + s);
            if (std::abs(val) < integral_threshold) continue;
            accumulate_two_body_product(mode_alpha(p), mode_alpha(q),
                                        mode_alpha(r), mode_alpha(s), val);
            if (q == r) {
              accumulate_epq(mode_alpha(p), mode_alpha(s), -0.5 * val);
            }
          }
        }
      }
    }

    // ββ channel
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        eri.build_row_bbbb(p * n_spatial + q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double val = eri.bbbb(r * n_spatial + s);
            if (std::abs(val) < integral_threshold) continue;
            accumulate_two_body_product(mode_beta(p), mode_beta(q),
                                        mode_beta(r), mode_beta(s), val);
            if (q == r) {
              accumulate_epq(mode_beta(p), mode_beta(s), -0.5 * val);
            }
          }
        }
      }
    }

    // αβ + βα cross-spin channels, related by Coulomb symmetry (pq|rs)=(rs|pq)
    for (std::size_t p = 0; p < n_spatial; ++p) {
      for (std::size_t q = 0; q < n_spatial; ++q) {
        eri.build_row_aabb(p * n_spatial + q);
        for (std::size_t r = 0; r < n_spatial; ++r) {
          for (std::size_t s = 0; s < n_spatial; ++s) {
            double val = eri.aabb(r * n_spatial + s);
            if (std::abs(val) < integral_threshold) continue;
            accumulate_two_body_product(mode_alpha(p), mode_alpha(q),
                                        mode_beta(r), mode_beta(s), val);
            accumulate_two_body_product(mode_beta(r), mode_beta(s),
                                        mode_alpha(p), mode_alpha(q), val);
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
    // H_total = H_physical + H_aux. We add H_aux = λ · Σ_i (I − S_i) to
    // penalize unphysical codespace sectors.
    double aux_ham_coefficient = 1.0 + h1_sum;

    PackedPauliWord<NW> identity{};
    acc.accumulate(
        identity, std::complex<double>(
                      aux_ham_coefficient * mapping.stabilizers().size(), 0.0));

    for (const auto& [coeff, word] : mapping.stabilizers()) {
      acc.accumulate(sparse_to_packed<NW>(word), -aux_ham_coefficient * coeff);
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

// Dispatch table: function pointer per NW, covering 1..16 (up to 1024 qubits).
template <typename TEriProvider>
struct Dispatcher {
  using Fn = MajoranaMapResult (*)(const MajoranaMapping&, double,
                                   const double*, const double*, TEriProvider&,
                                   std::size_t, bool, double, double);

  template <std::size_t NW>
  static MajoranaMapResult run(const MajoranaMapping& mapping,
                               double core_energy, const double* h1_alpha,
                               const double* h1_beta, TEriProvider& ERI,
                               std::size_t n_spatial, bool spin_symmetric,
                               double threshold, double integral_threshold) {
    return majorana_map_impl<NW>(mapping, core_energy, h1_alpha, h1_beta, ERI,
                                 n_spatial, spin_symmetric, threshold,
                                 integral_threshold);
  }

  template <std::size_t... Is>
  static constexpr std::array<Fn, sizeof...(Is)> make_table(
      std::index_sequence<Is...>) {
    return {{&run<Is + 1>...}};
  }
};

constexpr std::size_t max_nw = 16;

template <typename TEriProvider>
MajoranaMapResult majorana_map_dispatch(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, TEriProvider& ERI, std::size_t n_spatial,
    bool spin_symmetric, double threshold, double integral_threshold) {
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

  static constexpr auto table =
      Dispatcher<TEriProvider>::make_table(std::make_index_sequence<max_nw>{});
  return table[num_words - 1](mapping, core_energy, h1_alpha, h1_beta, ERI,
                              n_spatial, spin_symmetric, threshold,
                              integral_threshold);
}

}  // namespace detail

MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool spin_symmetric,
    double threshold, double integral_threshold) {
  detail::DenseEriProvider dense_eri(eri_aaaa, eri_aabb, eri_bbbb, n_spatial);
  return detail::majorana_map_dispatch(mapping, core_energy, h1_alpha, h1_beta,
                                       dense_eri, n_spatial, spin_symmetric,
                                       threshold, integral_threshold);
}

MajoranaMapResult majorana_map_hamiltonian(const MajoranaMapping& mapping,
                                           const Hamiltonian& hamiltonian,
                                           double threshold,
                                           double integral_threshold) {
  double core_energy = 0.0;  // Core energy constant is excluded from mapped
                             // operator to match dense path contract
  auto [h1_alpha, h1_beta] = hamiltonian.get_one_body_integrals();
  const double* h1_alpha_ptr = h1_alpha.data();
  const double* h1_beta_ptr = h1_beta.data();
  size_t n_spatial = h1_alpha.rows();
  bool spin_symmetric = hamiltonian.is_restricted();

  std::string container_type = hamiltonian.get_container_type();
  if (container_type == "cholesky") {
    const auto& cholesky_container =
        hamiltonian.get_container<CholeskyHamiltonianContainer>();
    auto [L_alpha, L_beta] = cholesky_container.get_three_center_integrals();
    detail::CholeskyEriProvider cholesky_eri(L_alpha, L_beta, n_spatial,
                                             spin_symmetric);
    return detail::majorana_map_dispatch(
        mapping, core_energy, h1_alpha_ptr, h1_beta_ptr, cholesky_eri,
        n_spatial, spin_symmetric, threshold, integral_threshold);
  } else if (container_type == "sparse") {
    const auto& sparse_container =
        hamiltonian.get_container<SparseHamiltonianContainer>();
    const SymmetryBlockedSparseMap<4>* two_body = nullptr;
    if (sparse_container.has_two_body_integrals()) {
      two_body = &sparse_container.two_body_integrals_sparse();
    }
    detail::SparseEriProvider sparse_eri(two_body, n_spatial);
    return detail::majorana_map_dispatch(
        mapping, core_energy, h1_alpha_ptr, h1_beta_ptr, sparse_eri, n_spatial,
        spin_symmetric, threshold, integral_threshold);
  } else {
    // Fallback: dense path
    auto [eri_aaaa, eri_aabb, eri_bbbb] = hamiltonian.get_two_body_integrals();
    detail::DenseEriProvider dense_eri(eri_aaaa.data(), eri_aabb.data(),
                                       eri_bbbb.data(), n_spatial);
    return detail::majorana_map_dispatch(
        mapping, core_energy, h1_alpha_ptr, h1_beta_ptr, dense_eri, n_spatial,
        spin_symmetric, threshold, integral_threshold);
  }
}

}  // namespace qdk::chemistry::data

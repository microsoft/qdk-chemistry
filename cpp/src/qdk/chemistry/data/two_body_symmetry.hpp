// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/QR>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data::detail {

/// Symmetry violations up to this fraction of the largest integral in the
/// tensor are treated as round-off; larger ones as genuinely reduced symmetry.
inline constexpr double two_body_symmetry_tolerance = 1e-5;

using TwoBodyKey = std::array<std::size_t, 4>;

inline double max_abs(const double* values, std::size_t count) {
  double scale = 0.0;
  for (std::size_t i = 0; i < count; ++i) {
    scale = std::max(scale, std::abs(values[i]));
  }
  return scale;
}

/// Permutation symmetry of a restricted or same-spin two-body tensor, in
/// increasing strength.
///
/// The three generators have different standing.  a+_p a+_r a_s a_q is
/// invariant under (p,q) <-> (r,s), so the operator depends only on the
/// bra-ket average gbar_pqrs = (g_pqrs + g_rspq) / 2: bra-ket symmetry is a
/// gauge choice, not a property of the physics.  Given that average,
/// (pq|rs) = (qp|sr) is exactly Hermiticity of the operator, and the
/// independent bra swap (pq|rs) = (qp|rs) additionally requires real orbitals.
///
/// Genuine electron-repulsion integrals over real orbitals therefore always
/// reach `EightFold`.  `FourFold` is the general Hermitian two-body operator,
/// which is what downfolding and effective-Hamiltonian constructions produce.
enum class TwoBodySymmetry {
  NonHermitian,        ///< gbar_pqrs != gbar_qpsr: no Hermitian operator
  NotBraKetSymmetric,  ///< g_pqrs != g_rspq: Hermitian, but g != gbar
  FourFold,            ///< (pq|rs) = (qp|sr) = (rs|pq) = (sr|qp)
  EightFold,           ///< additionally (pq|rs) = (qp|rs) = (pq|sr)
};

/// Strongest symmetry that holds at `key`, reading partner positions via `at`.
/// Results order like the enum, so a tensor's symmetry is the minimum over
/// its positions.
template <class At>
TwoBodySymmetry two_body_symmetry_at(const TwoBodyKey& key, double value,
                                     const At& at, double tolerance) {
  const auto [p, q, r, s] = key;
  auto equal = [tolerance](double lhs, double rhs) {
    return std::abs(lhs - rhs) <= tolerance;
  };
  const double swapped = at({r, s, p, q});
  // Hermiticity and the 8-fold test apply to the bra-ket average, since that
  // is all the operator sees.
  const double mean = 0.5 * (value + swapped);
  if (!equal(mean, 0.5 * (at({q, p, s, r}) + at({s, r, q, p})))) {
    return TwoBodySymmetry::NonHermitian;
  }
  // Both evaluation strategies read the stored tensor directly in the
  // cross-spin channel, so a Hermitian operator in the wrong gauge still has
  // to be symmetrized by the caller before it can be mapped.
  if (!equal(value, swapped)) return TwoBodySymmetry::NotBraKetSymmetric;
  if (!equal(mean, 0.5 * (at({q, p, r, s}) + at({r, s, q, p})))) {
    return TwoBodySymmetry::FourFold;
  }
  return TwoBodySymmetry::EightFold;
}

/// Classify `integrals`, a row-major (pq|rs) tensor of extent `n` per index.
inline TwoBodySymmetry classify_two_body_symmetry(const double* integrals,
                                                  std::size_t n,
                                                  double relative_tolerance) {
  const double tolerance =
      relative_tolerance * max_abs(integrals, n * n * n * n);
  auto at = [integrals, n](const TwoBodyKey& k) {
    return integrals[((k[0] * n + k[1]) * n + k[2]) * n + k[3]];
  };

  auto symmetry = TwoBodySymmetry::EightFold;
  for (std::size_t p = 0; p < n; ++p) {
    for (std::size_t q = 0; q < n; ++q) {
      for (std::size_t r = 0; r < n; ++r) {
        for (std::size_t s = 0; s < n; ++s) {
          const TwoBodyKey key{p, q, r, s};
          symmetry = std::min(
              symmetry, two_body_symmetry_at(key, at(key), at, tolerance));
          if (symmetry == TwoBodySymmetry::NonHermitian) return symmetry;
        }
      }
    }
  }
  return symmetry;
}

/// Whether (pq|rs) = (qp|sr) for every element, the Hermiticity condition of
/// a mixed-spin block that has no exchange partner within itself.
inline bool two_body_has_reversal_symmetry(const double* integrals,
                                           std::size_t n,
                                           double relative_tolerance) {
  const double tolerance =
      relative_tolerance * max_abs(integrals, n * n * n * n);
  for (std::size_t bra = 0; bra < n; ++bra)
    for (std::size_t ket = 0; ket < n; ++ket)
      for (std::size_t other_bra = 0; other_bra < n; ++other_bra)
        for (std::size_t other_ket = 0; other_ket < n; ++other_ket)
          if (std::abs(
                  integrals[((bra * n + ket) * n + other_bra) * n + other_ket] -
                  integrals[((ket * n + bra) * n + other_ket) * n +
                            other_bra]) > tolerance)
            return false;
  return true;
}

inline void require_finite_two_body(const double* integrals, std::size_t count,
                                    const std::string& consumer) {
  if (!std::all_of(integrals, integrals + count,
                   [](double value) { return std::isfinite(value); })) {
    throw std::invalid_argument(
        consumer + ": the two-body integrals contain a non-finite value.");
  }
}

/// Rejects `NonHermitian` and `NotBraKetSymmetric`, for consumers that
/// silently assume (pq|rs) = (rs|pq) = (qp|sr).
inline void throw_unless_fourfold(TwoBodySymmetry symmetry,
                                  const std::string& consumer) {
  if (symmetry == TwoBodySymmetry::NonHermitian) {
    throw std::invalid_argument(
        consumer +
        ": (pq|rs) + (rs|pq) != (qp|sr) + (sr|qp), so the two-body integrals "
        "define a non-Hermitian operator, which is not supported.");
  }
  if (symmetry == TwoBodySymmetry::NotBraKetSymmetric) {
    throw std::invalid_argument(
        consumer +
        ": the two-body integrals are Hermitian but not stored with "
        "(pq|rs) = (rs|pq), which this method assumes. That swap leaves the "
        "operator unchanged, so replace the tensor by its bra-ket average "
        "((pq|rs) + (rs|pq)) / 2.");
  }
}

/// Rejects a restricted or same-spin tensor outside the fourfold group, and
/// returns its classification (`FourFold` or `EightFold`).
inline TwoBodySymmetry require_fourfold_two_body(const double* integrals,
                                                 std::size_t n,
                                                 const std::string& consumer) {
  require_finite_two_body(integrals, n * n * n * n, consumer);
  const auto symmetry =
      classify_two_body_symmetry(integrals, n, two_body_symmetry_tolerance);
  throw_unless_fourfold(symmetry, consumer);
  return symmetry;
}

/// Three-center factors split into their parts symmetric and antisymmetric in
/// the orbital pair, one row per pair p <= q in row-major order.
///
/// Factors are column-major `[n^2 x naux]` with the pair index in row-major
/// order, so element (pq, Q) lives at `factors[p * n + q + Q * n^2]`.  With
/// u = {p, q}, L_pq = S_u + A_u and L_qp = S_u - A_u; A_u is zero for p = q.
struct CholeskyPairParts {
  Eigen::MatrixXd symmetric;
  Eigen::MatrixXd antisymmetric;
};

inline CholeskyPairParts cholesky_pair_parts(const double* factors,
                                             std::size_t n, std::size_t naux,
                                             double scale) {
  const std::size_t n2 = n * n;
  const auto pairs = static_cast<Eigen::Index>(n * (n + 1) / 2);
  const auto columns = static_cast<Eigen::Index>(naux);
  CholeskyPairParts parts{Eigen::MatrixXd(pairs, columns),
                          Eigen::MatrixXd(pairs, columns)};
  for (std::size_t aux = 0; aux < naux; ++aux) {
    const auto column = static_cast<Eigen::Index>(aux);
    Eigen::Index row = 0;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p; q < n; ++q, ++row) {
        const double forward = factors[p * n + q + aux * n2] / scale;
        const double reverse = factors[q * n + p + aux * n2] / scale;
        parts.symmetric(row, column) = 0.5 * (forward + reverse);
        parts.antisymmetric(row, column) = 0.5 * (forward - reverse);
      }
    }
  }
  return parts;
}

/// Whether (pq|rs) = sum_Q left^Q_pq right^Q_rs satisfies (pq|rs) = (qp|sr)
/// element by element to within `relative_tolerance` times its largest
/// element, as the dense checks require.  For a same-spin block (`left ==
/// right`) the reconstruction is bra-ket symmetric, so this is its Hermiticity
/// and its fourfold symmetry.
///
/// In the notation of `CholeskyPairParts`, (pq|rs) - (qp|sr) =
/// 2(±S^l_u.A^r_v ± A^l_u.S^r_v), whose largest magnitude is
/// 2 max_{u,v} (|S^l_u.A^r_v| + |A^l_u.S^r_v|).  Two sufficient bounds are
/// tried before that exact maximum: a Cauchy-Schwarz bound from one pass over
/// the factors in O(n^2) memory, which settles factors that are
/// pair-symmetric to round-off without copying them, and a Frobenius bound
/// from thin QR factors of copied S and A in O(n^2 naux^2).  The exact
/// maximum is formed in row blocks, O(n^4 naux), only when both fail.  The
/// bounds compare against the largest diagonal element |(pq|pq)|: that is the
/// largest element of a same-spin block, and a lower bound for a mixed one.
inline bool cholesky_has_reversal_symmetry(const double* left,
                                           const double* right, std::size_t n,
                                           std::size_t naux,
                                           double relative_tolerance) {
  const bool same_spin = left == right;
  const std::size_t count = n * n * naux;
  const double left_scale = max_abs(left, count);
  const double right_scale = same_spin ? left_scale : max_abs(right, count);
  if (left_scale == 0.0 || right_scale == 0.0) return true;

  // Squared norms of S_u and A_u over the auxiliary index, per side, and the
  // diagonal elements (pq|pq) and (qp|qp), in units of the two scales.
  const std::size_t n2 = n * n;
  const std::size_t pair_count = n * (n + 1) / 2;
  std::vector<double> left_symmetric_sq(pair_count),
      left_antisymmetric_sq(pair_count);
  std::vector<double> right_symmetric_sq(same_spin ? 0 : pair_count);
  std::vector<double> right_antisymmetric_sq(same_spin ? 0 : pair_count);
  std::vector<double> forward_diagonal(pair_count),
      reverse_diagonal(pair_count);
  for (std::size_t aux = 0; aux < naux; ++aux) {
    std::size_t pair = 0;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p; q < n; ++q, ++pair) {
        const std::size_t forward = p * n + q + aux * n2;
        const std::size_t reverse = q * n + p + aux * n2;
        const double left_forward = left[forward] / left_scale;
        const double left_reverse = left[reverse] / left_scale;
        const double symmetric = 0.5 * (left_forward + left_reverse);
        const double antisymmetric = 0.5 * (left_forward - left_reverse);
        left_symmetric_sq[pair] += symmetric * symmetric;
        left_antisymmetric_sq[pair] += antisymmetric * antisymmetric;
        if (same_spin) {
          forward_diagonal[pair] += left_forward * left_forward;
          reverse_diagonal[pair] += left_reverse * left_reverse;
          continue;
        }
        const double right_forward = right[forward] / right_scale;
        const double right_reverse = right[reverse] / right_scale;
        const double right_sym = 0.5 * (right_forward + right_reverse);
        const double right_anti = 0.5 * (right_forward - right_reverse);
        right_symmetric_sq[pair] += right_sym * right_sym;
        right_antisymmetric_sq[pair] += right_anti * right_anti;
        forward_diagonal[pair] += left_forward * right_forward;
        reverse_diagonal[pair] += left_reverse * right_reverse;
      }
    }
  }
  auto max_norm = [](const std::vector<double>& squared_norms) {
    return std::sqrt(
        *std::max_element(squared_norms.begin(), squared_norms.end()));
  };
  auto max_abs_of = [](const std::vector<double>& values) {
    return max_abs(values.data(), values.size());
  };
  const double diagonal_max =
      std::max(max_abs_of(forward_diagonal), max_abs_of(reverse_diagonal));
  const double bound = relative_tolerance * diagonal_max;
  const auto& right_symmetric_sq_or_left =
      same_spin ? left_symmetric_sq : right_symmetric_sq;
  const auto& right_antisymmetric_sq_or_left =
      same_spin ? left_antisymmetric_sq : right_antisymmetric_sq;
  if (2.0 * (max_norm(left_symmetric_sq) *
                 max_norm(right_antisymmetric_sq_or_left) +
             max_norm(left_antisymmetric_sq) *
                 max_norm(right_symmetric_sq_or_left)) <=
      bound) {
    return true;
  }

  const auto l = cholesky_pair_parts(left, n, naux, left_scale);
  CholeskyPairParts right_parts;
  if (!same_spin)
    right_parts = cholesky_pair_parts(right, n, naux, right_scale);
  const auto& r = same_spin ? l : right_parts;

  // ||X Y^T||_F = ||R_X R_Y^T||_F for thin QR factors X = Q_X R_X.
  auto reduced_r = [](const Eigen::MatrixXd& matrix) -> Eigen::MatrixXd {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(matrix);
    const Eigen::Index rows = std::min(matrix.rows(), matrix.cols());
    return qr.matrixQR().topRows(rows).triangularView<Eigen::Upper>();
  };
  auto product_norm = [](const Eigen::MatrixXd& first,
                         const Eigen::MatrixXd& second) {
    Eigen::MatrixXd product(first.rows(), second.rows());
    product.noalias() = first * second.transpose();
    return product.stableNorm();
  };
  const Eigen::MatrixXd left_symmetric = reduced_r(l.symmetric);
  const Eigen::MatrixXd left_antisymmetric = reduced_r(l.antisymmetric);
  const Eigen::MatrixXd right_symmetric =
      same_spin ? left_symmetric : reduced_r(r.symmetric);
  const Eigen::MatrixXd right_antisymmetric =
      same_spin ? left_antisymmetric : reduced_r(r.antisymmetric);
  if (2.0 * (product_norm(left_symmetric, right_antisymmetric) +
             product_norm(left_antisymmetric, right_symmetric)) <=
      bound) {
    return true;
  }

  // Exact maxima.  Element (u, v) of the reconstruction is a ± b ± c ± d with
  // a = S.S, b = S.A, c = A.S, d = A.A, signs set by the pair orientations.
  const Eigen::Index pairs = l.symmetric.rows();
  const Eigen::Index block =
      std::clamp<Eigen::Index>((Eigen::Index{1} << 20) / pairs, 1, pairs);
  double max_difference = 0.0;
  double max_value = diagonal_max;
  for (Eigen::Index start = 0; start < pairs; start += block) {
    const Eigen::Index rows = std::min(block, pairs - start);
    Eigen::MatrixXd symmetric_antisymmetric(rows, pairs);
    symmetric_antisymmetric.noalias() =
        l.symmetric.middleRows(start, rows) * r.antisymmetric.transpose();
    Eigen::MatrixXd antisymmetric_symmetric(rows, pairs);
    antisymmetric_symmetric.noalias() =
        l.antisymmetric.middleRows(start, rows) * r.symmetric.transpose();
    max_difference =
        std::max(max_difference, (symmetric_antisymmetric.cwiseAbs() +
                                  antisymmetric_symmetric.cwiseAbs())
                                     .maxCoeff());
    if (same_spin) continue;
    Eigen::MatrixXd symmetric_symmetric(rows, pairs);
    symmetric_symmetric.noalias() =
        l.symmetric.middleRows(start, rows) * r.symmetric.transpose();
    Eigen::MatrixXd antisymmetric_antisymmetric(rows, pairs);
    antisymmetric_antisymmetric.noalias() =
        l.antisymmetric.middleRows(start, rows) * r.antisymmetric.transpose();
    max_value = std::max(
        {max_value,
         ((symmetric_symmetric + antisymmetric_antisymmetric).cwiseAbs() +
          (symmetric_antisymmetric + antisymmetric_symmetric).cwiseAbs())
             .maxCoeff(),
         ((symmetric_symmetric - antisymmetric_antisymmetric).cwiseAbs() +
          (symmetric_antisymmetric - antisymmetric_symmetric).cwiseAbs())
             .maxCoeff()});
  }
  return 2.0 * max_difference <= relative_tolerance * max_value;
}

/// Representative shared by all eight permutations of (p,q,r,s); also the
/// lexicographic minimum of the class.
inline TwoBodyKey eightfold_canonical(TwoBodyKey k) {
  if (k[0] > k[1]) std::swap(k[0], k[1]);
  if (k[2] > k[3]) std::swap(k[2], k[3]);
  if (k[0] > k[2] || (k[0] == k[2] && k[1] > k[3])) {
    std::swap(k[0], k[2]);
    std::swap(k[1], k[3]);
  }
  return k;
}

/// Number of *distinct* index tuples in the permutation class of a canonical
/// key: repeated indices make some of the eight permutations coincide.
inline std::size_t eightfold_class_size(const TwoBodyKey& canonical) {
  const std::size_t bra = canonical[0] == canonical[1] ? 1 : 2;
  const std::size_t ket = canonical[2] == canonical[3] ? 1 : 2;
  const std::size_t swap =
      canonical[0] == canonical[2] && canonical[1] == canonical[3] ? 1 : 2;
  return bra * ket * swap;
}

/// Images of (pq|rs) under exchange and reversal; may repeat positions.
inline std::array<TwoBodyKey, 4> fourfold_orbit(const TwoBodyKey& k) {
  return {{k,
           {k[2], k[3], k[0], k[1]},
           {k[1], k[0], k[3], k[2]},
           {k[3], k[2], k[1], k[0]}}};
}

/// Images of (pq|rs) under the eightfold group; may repeat positions.
inline std::array<TwoBodyKey, 8> eightfold_orbit(const TwoBodyKey& k) {
  const auto direct = fourfold_orbit(k);
  const auto swapped = fourfold_orbit({k[1], k[0], k[2], k[3]});
  std::array<TwoBodyKey, 8> orbit;
  std::copy(direct.begin(), direct.end(), orbit.begin());
  std::copy(swapped.begin(), swapped.end(), orbit.begin() + 4);
  return orbit;
}

/// Literal sparse two-body tensor: positions that are absent are zero.
using SparseTwoBodyTensor = std::map<TwoBodyKey, double>;

inline std::string two_body_key_string(const TwoBodyKey& key) {
  return "(" + std::to_string(key[0]) + ", " + std::to_string(key[1]) + ", " +
         std::to_string(key[2]) + ", " + std::to_string(key[3]) + ")";
}

/// Sparse classification, with a stored position that violates the
/// strongest symmetry the result denies.
struct SparseTwoBodySymmetry {
  TwoBodySymmetry symmetry;
  TwoBodyKey witness{};
};

/// Sparse counterpart of @ref classify_two_body_symmetry, reading absent
/// positions as zero.  Every comparison involving a nonzero value is made at
/// a stored position, so visiting only the stored entries is sufficient.
inline SparseTwoBodySymmetry classify_sparse_two_body_symmetry(
    const SparseTwoBodyTensor& tensor, double relative_tolerance) {
  double scale = 0.0;
  for (const auto& [key, value] : tensor) {
    scale = std::max(scale, std::abs(value));
  }
  const double tolerance = relative_tolerance * scale;
  auto at = [&tensor](const TwoBodyKey& key) {
    const auto it = tensor.find(key);
    return it == tensor.end() ? 0.0 : it->second;
  };

  SparseTwoBodySymmetry result{TwoBodySymmetry::EightFold};
  for (const auto& [key, value] : tensor) {
    const auto symmetry = two_body_symmetry_at(key, value, at, tolerance);
    if (symmetry < result.symmetry) {
      result = {symmetry, key};
      if (symmetry == TwoBodySymmetry::NonHermitian) break;
    }
  }
  return result;
}

/// Nonzero entries that are the only stored member of a larger eightfold
/// class: the deprecated sparse shorthand in which one record stands for its
/// whole class.  Read literally, such a record is never fourfold symmetric
/// (only (ii|ii) is its own fourfold orbit), so it would be rejected.
inline std::vector<TwoBodyKey> lone_eightfold_representatives(
    const SparseTwoBodyTensor& tensor) {
  std::map<TwoBodyKey, std::pair<TwoBodyKey, std::size_t>> classes;
  for (const auto& [key, value] : tensor) {
    ++classes.try_emplace(eightfold_canonical(key), key, 0)
          .first->second.second;
  }
  std::vector<TwoBodyKey> lone;
  for (const auto& [canonical, stored] : classes) {
    const auto& [key, count] = stored;
    if (count == 1 && eightfold_class_size(canonical) > 1 &&
        tensor.at(key) != 0.0) {
      lone.push_back(key);
    }
  }
  return lone;
}

/// Expands every lone eightfold representative over its class.
inline void expand_legacy_eightfold_representatives(
    SparseTwoBodyTensor& tensor) {
  for (const auto& key : lone_eightfold_representatives(tensor)) {
    const double value = tensor.at(key);
    for (const auto& position : eightfold_orbit(key)) {
      tensor.emplace(position, value);
    }
  }
}

}  // namespace qdk::chemistry::data::detail

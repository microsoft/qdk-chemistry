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

/// Relative to the largest integral in the tensor.  Rounding in a four-index
/// transformation grows with the orbital coefficients: about 1e-13 for
/// ordinary molecules, but up to 2e-6 measured in nearly linearly dependent
/// bases (an H8 chain in cc-pVTZ).  Genuinely reduced symmetry, as in
/// downfolded Hamiltonians, appears near 1e-3 and above.  Qiskit Nature uses
/// the same relative tolerance when it infers this symmetry.
inline constexpr double two_body_symmetry_tolerance = 1e-5;

/// Relative residual up to which restricted mapping takes the 8-fold path,
/// which drops any part without 8-fold symmetry.  Ordinary round-off is at
/// most 9e-12 (cc-pVQZ); the 4-fold path is exact for 8-fold input too, so
/// anything above this takes it instead of being dropped.
inline constexpr double eightfold_path_tolerance = 1e-10;

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

/// Whether every three-center factor is symmetric in its orbital pair.
///
/// Factors are column-major `[n^2 x naux]` with the pair index in row-major
/// order, so element (pq, Q) lives at `factors[p * n + q + Q * n^2]`.  The
/// reconstruction sum_Q L^Q_pq L^Q_rs is bra-ket symmetric by construction but
/// carries the full eightfold symmetry only when each L^Q is symmetric.
inline bool cholesky_factors_are_pair_symmetric(const double* factors,
                                                std::size_t n, std::size_t naux,
                                                double relative_tolerance) {
  const std::size_t n2 = n * n;
  const double tolerance = relative_tolerance * max_abs(factors, n2 * naux);

  for (std::size_t aux = 0; aux < naux; ++aux) {
    const double* factor = factors + aux * n2;
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = p + 1; q < n; ++q) {
        if (std::abs(factor[p * n + q] - factor[q * n + p]) > tolerance) {
          return false;
        }
      }
    }
  }
  return true;
}

struct CholeskyFactorSymmetryQR {
  Eigen::MatrixXd symmetric;
  Eigen::MatrixXd antisymmetric;
};

inline CholeskyFactorSymmetryQR cholesky_factor_symmetry_qr(
    const double* factors, std::size_t n, std::size_t naux) {
  const std::size_t n2 = n * n;
  const Eigen::Index rows = static_cast<Eigen::Index>(n2);
  const Eigen::Index columns = static_cast<Eigen::Index>(naux);
  const double scale = max_abs(factors, n2 * naux);
  if (scale == 0.0) {
    return {Eigen::MatrixXd::Zero(std::min(rows, columns), columns),
            Eigen::MatrixXd::Zero(std::min(rows, columns), columns)};
  }

  Eigen::MatrixXd symmetric(rows, columns);
  Eigen::MatrixXd antisymmetric(rows, columns);
  for (std::size_t aux = 0; aux < naux; ++aux) {
    for (std::size_t p = 0; p < n; ++p) {
      for (std::size_t q = 0; q < n; ++q) {
        const double forward = factors[p * n + q + aux * n2] / scale;
        const double reverse = factors[q * n + p + aux * n2] / scale;
        symmetric(p * n + q, aux) = 0.5 * (forward + reverse);
        antisymmetric(p * n + q, aux) = 0.5 * (forward - reverse);
      }
    }
  }
  auto reduced_qr = [](const Eigen::MatrixXd& matrix) -> Eigen::MatrixXd {
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(matrix);
    const Eigen::Index reduced_rows = std::min(matrix.rows(), matrix.cols());
    return qr.matrixQR().topRows(reduced_rows).triangularView<Eigen::Upper>();
  };
  return {reduced_qr(symmetric), reduced_qr(antisymmetric)};
}

inline double cholesky_reduced_product_norm(const Eigen::MatrixXd& left,
                                            const Eigen::MatrixXd& right) {
  Eigen::MatrixXd product(left.rows(), right.rows());
  product.noalias() = left * right.transpose();
  return product.stableNorm();
}

inline TwoBodySymmetry cholesky_two_body_symmetry(const double* factors,
                                                  std::size_t n,
                                                  std::size_t naux,
                                                  double relative_tolerance) {
  if (n == 0 || naux == 0 ||
      cholesky_factors_are_pair_symmetric(factors, n, naux,
                                          relative_tolerance)) {
    return TwoBodySymmetry::EightFold;
  }
  const auto reduced = cholesky_factor_symmetry_qr(factors, n, naux);
  const double mixed_norm =
      cholesky_reduced_product_norm(reduced.symmetric, reduced.antisymmetric);
  const double scale = std::hypot(
      cholesky_reduced_product_norm(reduced.symmetric, reduced.symmetric),
      cholesky_reduced_product_norm(reduced.antisymmetric,
                                    reduced.antisymmetric),
      std::sqrt(2.0) * mixed_norm);
  return mixed_norm <= relative_tolerance * scale
             ? TwoBodySymmetry::FourFold
             : TwoBodySymmetry::NonHermitian;
}

inline bool cholesky_mixed_spin_has_reversal_symmetry(
    const double* alpha, const double* beta, std::size_t n, std::size_t naux,
    double relative_tolerance) {
  if (n == 0 || naux == 0 ||
      (cholesky_factors_are_pair_symmetric(alpha, n, naux,
                                           relative_tolerance) &&
       cholesky_factors_are_pair_symmetric(beta, n, naux,
                                           relative_tolerance))) {
    return true;
  }
  const auto alpha_reduced = cholesky_factor_symmetry_qr(alpha, n, naux);
  const auto beta_reduced = cholesky_factor_symmetry_qr(beta, n, naux);
  const double symmetric_norm = cholesky_reduced_product_norm(
      alpha_reduced.symmetric, beta_reduced.symmetric);
  const double antisymmetric_norm = cholesky_reduced_product_norm(
      alpha_reduced.antisymmetric, beta_reduced.antisymmetric);
  const double symmetric_antisymmetric_norm = cholesky_reduced_product_norm(
      alpha_reduced.symmetric, beta_reduced.antisymmetric);
  const double antisymmetric_symmetric_norm = cholesky_reduced_product_norm(
      alpha_reduced.antisymmetric, beta_reduced.symmetric);
  const double scale = std::hypot(
      std::hypot(symmetric_norm, antisymmetric_norm),
      std::hypot(symmetric_antisymmetric_norm, antisymmetric_symmetric_norm));
  const double tolerance = relative_tolerance * scale;
  return symmetric_antisymmetric_norm <= tolerance &&
         antisymmetric_symmetric_norm <= tolerance;
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

/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <macis/sd_operations.hpp>
#include <macis/types.hpp>
#include <macis/wfn/raw_bitset.hpp>
#include <vector>

namespace macis {
namespace detail {

/**
 * @brief Compute the four eigenvalues of a real symmetric 4x4 matrix
 *        using Ferrari's analytical method.
 *
 * @param mat  Flat array of 16 doubles in row-major order.
 *             Only the 10 unique entries of the symmetric matrix are read.
 * @return     Sorted array of the four eigenvalues in ascending order.
 */
inline std::array<double, 4> eigenvalues_4x4(const double* mat) noexcept {
  // ---- Stage 1: extract unique entries of the symmetric matrix ----
  double m00 = mat[0], m01 = mat[1], m02 = mat[2], m03 = mat[3];
  double m11 = mat[5], m12 = mat[6], m13 = mat[7];
  double m22 = mat[10], m23 = mat[11];
  double m33 = mat[15];

  // Squares of off-diagonal entries (used repeatedly)
  double sq01 = m01 * m01, sq02 = m02 * m02, sq03 = m03 * m03;
  double sq12 = m12 * m12, sq13 = m13 * m13, sq23 = m23 * m23;

  // Power-sum traces s_k = tr(A^k)
  double s1 = m00 + m11 + m22 + m33;
  // s2 = tr(A^2) = sum_ij A_ij^2  (A is symmetric)
  double s2 = m00 * m00 + m11 * m11 + m22 * m22 + m33 * m33 +
              2.0 * (sq01 + sq02 + sq03 + sq12 + sq13 + sq23);

  // Compute A^2 (symmetric) — only upper triangle needed.
  double a2_00 = m00 * m00 + sq01 + sq02 + sq03;
  double a2_01 = m00 * m01 + m01 * m11 + m02 * m12 + m03 * m13;
  double a2_02 = m00 * m02 + m01 * m12 + m02 * m22 + m03 * m23;
  double a2_03 = m00 * m03 + m01 * m13 + m02 * m23 + m03 * m33;
  double a2_11 = sq01 + m11 * m11 + sq12 + sq13;
  double a2_12 = m01 * m02 + m11 * m12 + m12 * m22 + m13 * m23;
  double a2_13 = m01 * m03 + m11 * m13 + m12 * m23 + m13 * m33;
  double a2_22 = sq02 + sq12 + m22 * m22 + sq23;
  double a2_23 = m02 * m03 + m12 * m13 + m22 * m23 + m23 * m33;
  double a2_33 = sq03 + sq13 + sq23 + m33 * m33;

  // s3 = tr(A^3) = sum_ij (A^2)_ij * A_ji = sum_ij (A^2)_ij * A_ij
  double s3 = a2_00 * m00 + a2_11 * m11 + a2_22 * m22 + a2_33 * m33 +
              2.0 * (a2_01 * m01 + a2_02 * m02 + a2_03 * m03 + a2_12 * m12 +
                     a2_13 * m13 + a2_23 * m23);
  // s4 = tr(A^4) = ||A^2||_F^2
  double s4 = a2_00 * a2_00 + a2_11 * a2_11 + a2_22 * a2_22 + a2_33 * a2_33 +
              2.0 * (a2_01 * a2_01 + a2_02 * a2_02 + a2_03 * a2_03 +
                     a2_12 * a2_12 + a2_13 * a2_13 + a2_23 * a2_23);

  // ---- Stage 2: elementary symmetric polynomials (Newton's identities) ----
  // c_k = e_k(lambda_1,...,lambda_4), the k-th elementary symmetric
  // polynomial of the eigenvalues.  The characteristic polynomial is
  //     t^4 - c1 t^3 + c2 t^2 - c3 t + c4 = 0.
  double c1 = s1;
  double c2 = (s1 * s1 - s2) / 2.0;
  double c3 = (s1 * s1 * s1 - 3.0 * s1 * s2 + 2.0 * s3) / 6.0;
  double c4 = (s1 * s1 * s1 * s1 - 6.0 * s1 * s1 * s2 + 3.0 * s2 * s2 +
               8.0 * s1 * s3 - 6.0 * s4) /
              24.0;

  // ---- Stage 3: depress the quartic ----
  // Substitute t = x + c1/4 to eliminate the cubic term, giving
  //     x^4 + dp x^2 + dq x + dr = 0.
  double shift = c1 / 4.0;
  double dp = c2 - 3.0 * c1 * c1 / 8.0;
  double dq = -c1 * c1 * c1 / 8.0 + c1 * c2 / 2.0 - c3;
  double dr = -3.0 * c1 * c1 * c1 * c1 / 256.0 + c1 * c1 * c2 / 16.0 -
              c1 * c3 / 4.0 + c4;

  std::array<double, 4> roots;

  // Biquadratic when dq ~ 0 relative to the other depressed coefficients
  double scale = std::max({std::abs(dp), std::abs(dr), 1.0});
  if (std::abs(dq) < 1e-14 * scale) {
    // ---- Special case: biquadratic (q ~ 0) ----
    // x^4 + dp x^2 + dr = 0  =>  quadratic in u = x^2:
    //     u = (-dp +/- sqrt(dp^2 - 4 dr)) / 2
    // then  x = +/- sqrt(u).
    double disc = dp * dp - 4.0 * dr;
    double sqrt_disc = std::sqrt(std::max(disc, 0.0));
    double u1 = (-dp + sqrt_disc) / 2.0;
    double u2 = (-dp - sqrt_disc) / 2.0;
    double su1 = std::sqrt(std::max(u1, 0.0));
    double su2 = std::sqrt(std::max(u2, 0.0));
    roots = {-su1 + shift, su1 + shift, -su2 + shift, su2 + shift};
  } else {
    // ---- Stage 4: solve the resolvent cubic ----
    // The resolvent cubic for Ferrari's method is
    //     m^3 + dp m^2 + ((dp^2 - 4 dr) / 4) m - dq^2 / 8 = 0.
    // Depress it via m = u - dp/3 to obtain  u^3 + P u + Q = 0.
    double rc_b = (dp * dp - 4.0 * dr) / 4.0;
    double rc_c = -dq * dq / 8.0;
    double P = rc_b - dp * dp / 3.0;
    double Q = 2.0 * dp * dp * dp / 27.0 - dp * rc_b / 3.0 + rc_c;

    // Discriminant of the depressed cubic:
    //     Delta = Q^2/4 + P^3/27.
    double m0;
    double cubic_disc = Q * Q / 4.0 + P * P * P / 27.0;
    if (cubic_disc <= 0) {
      // Three real roots — use the trigonometric (Viete) form:
      //     u_k = 2 sqrt(-P/3) cos(1/3 arccos(-Q / (2 (-P/3)^{3/2})) - 2kpi/3)
      // We pick k=0 (the largest root).
      double mag = std::sqrt(-P / 3.0);
      double arg = std::clamp(-Q / (2.0 * mag * mag * mag), -1.0, 1.0);
      m0 = 2.0 * mag * std::cos(std::acos(arg) / 3.0);
    } else {
      // One real root — Cardano's formula:
      //     u = cbrt(-Q/2 + sqrt(Delta)) + cbrt(-Q/2 - sqrt(Delta))
      double sd = std::sqrt(cubic_disc);
      m0 = std::cbrt(-Q / 2.0 + sd) + std::cbrt(-Q / 2.0 - sd);
    }
    // Un-depress: m0 = u - dp/3
    m0 -= dp / 3.0;
    // Numerical guard: m0 must be non-negative for real factorization
    m0 = std::max(m0, 0.0);

    // ---- Stage 5: factor the depressed quartic (Ferrari) ----
    // With S = sqrt(2 m0), the depressed quartic factors as
    //   (x^2 - S x + (alpha + dq/(2S))) (x^2 + S x + (alpha - dq/(2S))) = 0
    // where alpha = dp/2 + m0.
    double S = std::sqrt(2.0 * m0);
    double hqs = dq / (2.0 * S);  // dq / (2S)
    double alpha = dp / 2.0 + m0;

    // Discriminants of the two quadratics:
    //   x^2 - S x + (alpha + hqs) = 0  =>  disc1 = S^2 - 4(alpha + hqs)
    //   x^2 + S x + (alpha - hqs) = 0  =>  disc2 = S^2 - 4(alpha - hqs)
    double d1 = S * S - 4.0 * (alpha + hqs);
    double d2 = S * S - 4.0 * (alpha - hqs);
    double sd1 = std::sqrt(std::max(d1, 0.0));
    double sd2 = std::sqrt(std::max(d2, 0.0));

    // Solve both quadratics and shift back by c1/4
    roots = {(S - sd1) / 2.0 + shift, (S + sd1) / 2.0 + shift,
             (-S - sd2) / 2.0 + shift, (-S + sd2) / 2.0 + shift};
  }

  std::sort(roots.begin(), roots.end());
  return roots;
}

}  // namespace detail

/**
 * @brief Storage for orbital reduced density matrix (RDM) intermediates.
 *
 * Collects the 1-, 2-, 3-, and 4-body RDM intermediates needed to compute
 * single-orbital and two-orbital von Neumann entropies and mutual information.
 * Matrices are stored as flat vectors wrapped in a Matrix helper that provides
 * a 2D span view. Diagonal elements are kept in separate vectors for
 * efficient diagonal-contribution accumulation.
 */
class OrbitalRDMIntermediates {
 public:
  struct Matrix {
    std::vector<double> data;
    matrix_span<double> span;

    Matrix(size_t n) : data(n * n, 0.0), span(data.data(), n, n) {}

    operator matrix_span<double>() { return span; }
    operator matrix_span<const double>() const { return span; }
    double& operator()(size_t i, size_t j) { return span(i, j); }
    const double& operator()(size_t i, size_t j) const { return span(i, j); }

    Matrix(const Matrix&) = delete;
    Matrix& operator=(const Matrix&) = delete;
  };

  /**
   * @brief Copy diagonal vector entries into their corresponding matrix
   *        diagonal positions.
   */
  void update_diagonal() {
    if (!need_s2) return;
    for (size_t i = 0; i < norb; ++i) {
      a_ij(i, i) = a_ii[i];
      b_ij(i, i) = b_ii[i];
      ab_ijjj(i, i) = ab_iiii[i];
      ab_jijj(i, i) = ab_iiii[i];
      ab_jjij(i, i) = ab_iiii[i];
      ab_jjji(i, i) = ab_iiii[i];
      ab_ijij(i, i) = ab_iiii[i];
      ab_ijji(i, i) = ab_iiii[i];
      ab_iijj(i, i) = ab_iiii[i];
    }
  }

  // Number of orbitals
  size_t norb;
  // Whether two-orbital (s2) intermediates are allocated
  bool need_s2;
  // one rdm
  std::vector<double> a_ii;
  std::vector<double> b_ii;
  Matrix a_ij;
  Matrix b_ij;
  // two rdm
  std::vector<double> ab_iiii;
  Matrix aa_iijj;
  Matrix bb_iijj;
  Matrix ab_iijj;
  Matrix ab_ijjj;
  Matrix ab_jijj;
  Matrix ab_jjij;
  Matrix ab_jjji;
  Matrix ab_ijij;
  Matrix ab_ijji;
  // three rdm
  Matrix aab_iijjjj;
  Matrix abb_jjiijj;
  Matrix aab_iijjii;
  Matrix abb_iiiijj;
  Matrix abb_ijiijj;
  Matrix aab_iijjij;
  // four rdm
  Matrix aabb_iijjiijj;

  /**
   * @brief Construct intermediates storage for a given number of orbitals.
   *
   * @param norb Number of spatial orbitals.
   */
  OrbitalRDMIntermediates(size_t norb, bool need_s2 = true)
      : norb(norb),
        need_s2(need_s2),
        a_ii(norb, 0.0),
        b_ii(norb, 0.0),
        a_ij(need_s2 ? norb : 0),
        b_ij(need_s2 ? norb : 0),
        ab_iiii(norb, 0.0),
        aa_iijj(need_s2 ? norb : 0),
        bb_iijj(need_s2 ? norb : 0),
        ab_iijj(need_s2 ? norb : 0),
        ab_ijjj(need_s2 ? norb : 0),
        ab_jijj(need_s2 ? norb : 0),
        ab_jjij(need_s2 ? norb : 0),
        ab_jjji(need_s2 ? norb : 0),
        ab_ijij(need_s2 ? norb : 0),
        ab_ijji(need_s2 ? norb : 0),
        aab_iijjjj(need_s2 ? norb : 0),
        abb_jjiijj(need_s2 ? norb : 0),
        aab_iijjii(need_s2 ? norb : 0),
        abb_iiiijj(need_s2 ? norb : 0),
        abb_ijiijj(need_s2 ? norb : 0),
        aab_iijjij(need_s2 ? norb : 0),
        aabb_iijjiijj(need_s2 ? norb : 0) {}
};

/**
 * @brief Accumulate diagonal contributions needed for single-orbital entropies
 *        only.
 *
 * This is a lightweight version of orbital_rdm_contrib_diag that only
 * accumulates the three diagonal vectors (a_ii, b_ii, ab_iiii) needed by
 * build_s1_entropy, skipping all two-orbital matrix intermediates.
 *
 * @tparam T          Scalar type.
 * @tparam IndexType  Container of orbital indices.
 * @param occ_alpha   Occupied alpha orbital indices.
 * @param occ_beta    Occupied beta orbital indices.
 * @param val         Coefficient product value to accumulate.
 * @param[in,out] a_ii    1-body alpha diagonal vector.
 * @param[in,out] b_ii    1-body beta diagonal vector.
 * @param[in,out] ab_iiii Mixed alpha-beta same-orbital diagonal vector.
 */
template <typename T, typename IndexType>
inline void orbital_rdm_contrib_diag_s1(const IndexType& occ_alpha,
                                        const IndexType& occ_beta, T val,
                                        std::vector<T>& a_ii,
                                        std::vector<T>& b_ii,
                                        std::vector<T>& ab_iiii) {
  for (auto p : occ_alpha) {
#pragma omp atomic
    a_ii[p] += val;
  }
  for (auto p : occ_beta) {
#pragma omp atomic
    b_ii[p] += val;
  }
  for (auto q : occ_beta) {
    for (auto p : occ_alpha) {
      if (p == q) {
#pragma omp atomic
        ab_iiii[q] += val;
      }
    }
  }
}

/**
 * @brief Accumulate diagonal (same-determinant) contributions to orbital RDM
 *        intermediates.
 *
 * @tparam T          Scalar type.
 * @tparam IndexType  Container of orbital indices.
 * @param occ_alpha       Occupied alpha orbital indices.
 * @param occ_beta        Occupied beta orbital indices.
 * @param val             Coefficient product value to accumulate.
 * @param[in,out] a_ii    1-body alpha diagonal vector.
 * @param[in,out] b_ii    1-body beta diagonal vector.
 * @param[in,out] ab_iiii Mixed alpha-beta same-orbital diagonal vector.
 * @param[in,out] aa_iijj Alpha-alpha 2-body matrix.
 * @param[in,out] bb_iijj Beta-beta 2-body matrix.
 * @param[in,out] ab_iijj Mixed alpha-beta 2-body matrix.
 * @param[in,out] aab_iijjij Alpha-alpha-beta 3-body matrix.
 * @param[in,out] abb_jjiijj Beta-beta-alpha 3-body matrix.
 * @param[in,out] aab_iijjii Alpha-alpha-beta 3-body matrix.
 * @param[in,out] abb_iiiijj Beta-beta-alpha 3-body matrix.
 * @param[in,out] aabb_iijjiijj Alpha-alpha-beta-beta 4-body matrix.
 */
template <typename T, typename IndexType>
inline void orbital_rdm_contrib_diag(
    const IndexType& occ_alpha, const IndexType& occ_beta, T val,
    std::vector<T>& a_ii, std::vector<T>& b_ii, std::vector<T>& ab_iiii,
    matrix_span<T> aa_iijj, matrix_span<T> bb_iijj, matrix_span<T> ab_iijj,
    matrix_span<T> aab_iijjjj, matrix_span<T> abb_jjiijj,
    matrix_span<T> aab_iijjii, matrix_span<T> abb_iiiijj,
    matrix_span<T> aabb_iijjiijj) {
  size_t norb = a_ii.size();
  // look up for 3 and 4 body
  std::vector<bool> is_alpha(norb, false);
  std::vector<bool> is_beta(norb, false);
  for (auto p : occ_alpha) {
    is_alpha[p] = true;
  }
  for (auto p : occ_beta) {
    is_beta[p] = true;
  }

  for (auto p : occ_alpha) {
#pragma omp atomic
    a_ii[p] += val;
  }
  for (auto p : occ_beta) {
#pragma omp atomic
    b_ii[p] += val;
  }

  for (auto q : occ_alpha) {
    for (auto p : occ_alpha) {
      if (p == q) {
        continue;
      }
#pragma omp atomic
      aa_iijj(p, q) += val;
      if (is_beta[p]) {
#pragma omp atomic
        aab_iijjii(p, q) += val;
        if (is_beta[q]) {
#pragma omp atomic
          aabb_iijjiijj(p, q) += val;
        }
      }
      if (is_beta[q]) {
#pragma omp atomic
        aab_iijjjj(p, q) += val;
      }
    }
  }

  for (auto q : occ_beta) {
    for (auto p : occ_beta) {
      if (p == q) {
        continue;
      }
#pragma omp atomic
      bb_iijj(p, q) += val;
      if (is_alpha[p]) {
#pragma omp atomic
        abb_jjiijj(q, p) += val;
      }
      if (is_alpha[q]) {
#pragma omp atomic
        abb_iiiijj(q, p) += val;
      }
    }
  }

  for (auto q : occ_beta) {
    for (auto p : occ_alpha) {
      if (p == q) {
#pragma omp atomic
        ab_iiii[q] += val;
      } else {
#pragma omp atomic
        ab_iijj(p, q) += val;
      }
    }
  }
}

/**
 * @brief Accumulate single-excitation contributions to orbital RDM
 *        intermediates.
 *
 * @tparam transpose  If false, treat as alpha excitation; if true, beta.
 * @tparam T          Scalar type.
 * @tparam N          Wavefunction bitset size.
 * @tparam IndexType  Container of orbital indices.
 * @param bra         Bra determinant (same-spin part).
 * @param ket         Ket determinant (same-spin part).
 * @param ex          Excitation bitmask (bra XOR ket, same-spin).
 * @param bra_occ_ss  Occupied orbitals of the same spin in the bra.
 * @param bra_occ_os  Occupied orbitals of the opposite spin in the bra.
 * @param val         Coefficient product value to accumulate.
 * @param[in,out] x_ij        1-body off-diagonal matrix (alpha or beta).
 * @param[in,out] ab_ijjj     2-body mixed-spin intermediate.
 * @param[in,out] ab_jjij     2-body mixed-spin intermediate.
 * @param[in,out] ab_jijj     2-body mixed-spin intermediate.
 * @param[in,out] ab_jjji     2-body mixed-spin intermediate.
 * @param[in,out] aab_iijjij   3-body intermediate.
 * @param[in,out] abb_ijiijj   3-body intermediate.
 */
template <bool transpose, typename T, size_t N, typename IndexType>
inline void orbital_rdm_contrib_2(
    wfn_t<N> bra, wfn_t<N> ket, wfn_t<N> ex, const IndexType& bra_occ_ss,
    const IndexType& bra_occ_os, T val, matrix_span<T> x_ij,
    matrix_span<T> ab_ijjj, matrix_span<T> ab_jjij, matrix_span<T> ab_jijj,
    matrix_span<T> ab_jjji, matrix_span<T> aab_iijjij,
    matrix_span<T> abb_ijiijj) {
  auto [o1, v1, sign] = single_excitation_sign_indices(bra, ket, ex);
  auto signed_val = sign * val;

#pragma omp atomic
  x_ij(v1, o1) += signed_val;
#pragma omp atomic
  x_ij(o1, v1) += signed_val;

  bool o1_in_occ = false;
  bool v1_in_occ = false;
  for (auto p : bra_occ_os) {
    if (o1 == p) {
      o1_in_occ = true;
      if constexpr (transpose) {  // beta
                                  // p, p, v1, o1
#pragma omp atomic
        ab_jjij(v1, o1) += signed_val;
        // p, p, v1, o1 -> Symmetry
#pragma omp atomic
        ab_jjji(v1, o1) += signed_val;
      } else {
        // v1, o1, p, p
#pragma omp atomic
        ab_ijjj(v1, o1) += signed_val;
        // v1, o1, p, p -> Symmetry
#pragma omp atomic
        ab_jijj(v1, o1) += signed_val;
      }
    } else if (v1 == p) {
      v1_in_occ = true;
      if constexpr (transpose) {
        // p, p, o1, v1
#pragma omp atomic
        ab_jjij(o1, v1) += signed_val;
        // p, p, o1, v1 -> Symmetry
#pragma omp atomic
        ab_jjji(o1, v1) += signed_val;
      } else {
        // o1, v1, p, p
#pragma omp atomic
        ab_ijjj(o1, v1) += signed_val;
        // o1, v1, p, p -> Symmetry
#pragma omp atomic
        ab_jijj(o1, v1) += signed_val;
      }
    }
  }
  if (o1_in_occ && v1_in_occ) {
    if constexpr (transpose) {
      // p, p, q, q, o1, v1
#pragma omp atomic
      aab_iijjij(o1, v1) += signed_val;
      aab_iijjij(v1, o1) += signed_val;
    } else {
      // o1, v1, p, p, q, q
#pragma omp atomic
      abb_ijiijj(v1, o1) += signed_val;
      abb_ijiijj(o1, v1) += signed_val;
    }
  }
}

/**
 * @brief Accumulate mixed alpha-beta double single-excitation contributions.
 *
 * @tparam T   Scalar type.
 * @tparam N   Wavefunction bitset size.
 * @param bra_alpha  Alpha bra determinant.
 * @param ket_alpha  Alpha ket determinant.
 * @param ex_alpha   Alpha excitation bitmask.
 * @param bra_beta   Beta bra determinant.
 * @param ket_beta   Beta ket determinant.
 * @param ex_beta    Beta excitation bitmask.
 * @param val        Coefficient product value to accumulate.
 * @param[in,out] ab_ijij  Mixed-spin same-index excitation intermediate.
 * @param[in,out] ab_ijji  Mixed-spin cross-index excitation intermediate.
 */
template <typename T, size_t N>
inline void orbital_rdm_contrib_22(wfn_t<N> bra_alpha, wfn_t<N> ket_alpha,
                                   wfn_t<N> ex_alpha, wfn_t<N> bra_beta,
                                   wfn_t<N> ket_beta, wfn_t<N> ex_beta, T val,
                                   matrix_span<T> ab_ijij,
                                   matrix_span<T> ab_ijji) {
  auto [o2, v2, sign_b] =
      single_excitation_sign_indices(bra_alpha, ket_alpha, ex_alpha);
  auto [o1, v1, sign_a] =
      single_excitation_sign_indices(bra_beta, ket_beta, ex_beta);
  auto signed_val = sign_a * sign_b * val;

  if (o1 == o2 && v1 == v2) {
#pragma omp atomic
    ab_ijij(v1, o1) += signed_val;
#pragma omp atomic
    ab_ijij(o1, v1) += signed_val;
  } else if (o1 == v2 && v1 == o2) {
#pragma omp atomic
    ab_ijji(v1, o1) += signed_val;
#pragma omp atomic
    ab_ijji(o1, v1) += signed_val;
  }
}

/**
 * @brief Compute single-orbital entropies from RDM intermediates.
 *
 * @param intermediates  Precomputed orbital RDM intermediates.
 * @param[out] s1_entropy  Vector of length norb to receive the entropies.
 */
inline void build_s1_entropy(const OrbitalRDMIntermediates& intermediates,
                             std::vector<double>& s1_entropy) {
  const size_t norb = intermediates.norb;
  const auto& a_ii = intermediates.a_ii;
  const auto& b_ii = intermediates.b_ii;
  const auto& ab_iiii = intermediates.ab_iiii;

  double val;
  for (size_t i = 0; i < norb; ++i) {
    s1_entropy[i] = 0.0;

    val = 1 - a_ii[i] - b_ii[i] + ab_iiii[i];
    if (val > std::numeric_limits<double>::epsilon()) {
      s1_entropy[i] += -val * std::log(val);
    }
    val = a_ii[i] - ab_iiii[i];
    if (val > std::numeric_limits<double>::epsilon()) {
      s1_entropy[i] += -val * std::log(val);
    }
    val = b_ii[i] - ab_iiii[i];
    if (val > std::numeric_limits<double>::epsilon()) {
      s1_entropy[i] += -val * std::log(val);
    }
    val = ab_iiii[i];
    if (val > std::numeric_limits<double>::epsilon()) {
      s1_entropy[i] += -val * std::log(val);
    }
  }
}

/**
 * @brief Build the diagonal of the 4x4 single-orbital reduced density matrix.
 *
 * @param intermediates  Precomputed orbital RDM intermediates.
 * @param i              Orbital index.
 * @return               Array of four diagonal elements (eigenvalues).
 */
inline std::array<double, 4> build_1ordm_block(
    const OrbitalRDMIntermediates& intermediates, size_t i) {
  const auto& a_ii = intermediates.a_ii;
  const auto& b_ii = intermediates.b_ii;
  const auto& ab_iiii = intermediates.ab_iiii;

  std::array<double, 4> diag;
  diag[0] = 1 - a_ii[i] - b_ii[i] + ab_iiii[i];
  diag[1] = a_ii[i] - ab_iiii[i];
  diag[2] = b_ii[i] - ab_iiii[i];
  diag[3] = ab_iiii[i];
  return diag;
}

/**
 * @brief Build the single-orbital reduced density matrix diagonals for all
 *        orbitals.
 *
 * @param intermediates  Precomputed orbital RDM intermediates.
 * @param[out] ordm      Matrix span of shape (norb, 4) to receive the
 *                       1-ORDM diagonals.
 */
inline void build_1ordm(const OrbitalRDMIntermediates& intermediates,
                        matrix_span<double> ordm) {
  const size_t norb = intermediates.norb;
  for (size_t i = 0; i < norb; ++i) {
    auto diag = build_1ordm_block(intermediates, i);
    for (size_t p = 0; p < 4; ++p) {
      ordm(i, p) = diag[p];
    }
  }
}

/**
 * @brief Compute two-orbital entropies from RDM intermediates.
 *
 * @param intermediates  Precomputed orbital RDM intermediates.
 * @param[out] s2_entropy  norb x norb matrix span to receive the entropies.
 */
inline void build_s2_entropy(const OrbitalRDMIntermediates& intermediates,
                             matrix_span<double> s2_entropy) {
  const size_t norb = intermediates.norb;
  auto& a_ii = intermediates.a_ii;
  auto& b_ii = intermediates.b_ii;
  auto& a_ij = intermediates.a_ij;
  auto& b_ij = intermediates.b_ij;
  auto& ab_iiii = intermediates.ab_iiii;
  auto& aa_iijj = intermediates.aa_iijj;
  auto& bb_iijj = intermediates.bb_iijj;
  auto& ab_iijj = intermediates.ab_iijj;
  auto& ab_ijjj = intermediates.ab_ijjj;
  auto& ab_jijj = intermediates.ab_jijj;
  auto& ab_jjij = intermediates.ab_jjij;
  auto& ab_jjji = intermediates.ab_jjji;
  auto& ab_ijij = intermediates.ab_ijij;
  auto& ab_ijji = intermediates.ab_ijji;
  auto& aab_iijjjj = intermediates.aab_iijjjj;
  auto& abb_jjiijj = intermediates.abb_jjiijj;
  auto& aab_iijjii = intermediates.aab_iijjii;
  auto& abb_iiiijj = intermediates.abb_iiiijj;
  auto& abb_ijiijj = intermediates.abb_ijiijj;
  auto& aab_iijjij = intermediates.aab_iijjij;
  auto& aabb_iijjiijj = intermediates.aabb_iijjiijj;

  // Analytical eigenvalues of a symmetric 2x2 matrix [[a, b], [b, d]]
  auto eigenvalues_2x2 = [](double a, double b, double d) {
    double half_sum = 0.5 * (a + d);
    double half_diff = 0.5 * (a - d);
    double w = std::sqrt(half_diff * half_diff + b * b);
    return std::array<double, 2>{half_sum - w, half_sum + w};
  };

  double val, val1, val2;
  std::vector<double> block_4x4(16, 0.0);
  for (size_t i = 0; i < norb; ++i) {
    for (size_t j = i; j < norb; ++j) {
      s2_entropy(i, j) = 0.0;

      auto add_entropy = [&](double v) {
        if (v > std::numeric_limits<double>::epsilon()) {
          s2_entropy(i, j) -= v * std::log(v);
        }
      };

      // 0, 0
      val = 1 - a_ii[i] - b_ii[i] - a_ii[j] - b_ii[j] + ab_iiii[i] +
            ab_iiii[j] + aa_iijj(i, j) + ab_iijj(i, j) + ab_iijj(j, i) +
            bb_iijj(i, j) - aab_iijjjj(i, j) - abb_jjiijj(i, j) -
            aab_iijjii(i, j) - abb_iiiijj(i, j) + aabb_iijjiijj(i, j);
      add_entropy(val);

      // 1, 1
      val = a_ii[j] - ab_iijj(j, i) - aa_iijj(i, j) - ab_iijj(j, j) +
            aab_iijjjj(i, j) + aab_iijjii(i, j) + abb_jjiijj(i, j) -
            aabb_iijjiijj(i, j);
      // 1, 2 = 2, 1
      val1 = a_ij(i, j) - ab_jijj(j, i) - ab_ijjj(i, j) + abb_ijiijj(i, j);
      // 2, 2
      val2 = a_ii[i] - ab_iijj(i, j) - aa_iijj(i, j) - ab_iijj(i, i) +
             aab_iijjjj(i, j) + aab_iijjii(i, j) + abb_iiiijj(i, j) -
             aabb_iijjiijj(i, j);
      auto eigs_2x2 = eigenvalues_2x2(val, val1, val2);
      for (auto v : eigs_2x2) add_entropy(v);

      // 3, 3
      val = b_ii[j] - ab_iijj(i, j) - bb_iijj(i, j) - ab_iijj(j, j) +
            abb_iiiijj(i, j) + aab_iijjjj(i, j) + abb_jjiijj(i, j) -
            aabb_iijjiijj(i, j);
      // 3, 4 = 4, 3
      val1 = b_ij(i, j) - ab_jjij(j, i) - ab_jjji(i, j) + aab_iijjij(i, j);
      // 4, 4
      val2 = b_ii[i] - ab_iijj(j, i) - bb_iijj(i, j) - ab_iijj(i, i) +
             abb_jjiijj(i, j) + aab_iijjii(i, j) + abb_iiiijj(i, j) -
             aabb_iijjiijj(i, j);
      auto eigs_2x2_beta = eigenvalues_2x2(val, val1, val2);
      for (auto v : eigs_2x2_beta) add_entropy(v);

      // 5, 5
      val = aa_iijj(i, j) - aab_iijjii(i, j) - aab_iijjjj(i, j) +
            aabb_iijjiijj(i, j);
      add_entropy(val);

      // 6, 6
      val = bb_iijj(i, j) - abb_iiiijj(i, j) - abb_jjiijj(i, j) +
            aabb_iijjiijj(i, j);
      add_entropy(val);

      // 7, 7
      block_4x4[0] = ab_iijj(j, j) - aab_iijjjj(i, j) - abb_jjiijj(i, j) +
                     aabb_iijjiijj(i, j);

      // 7, 8
      block_4x4[1] = ab_ijjj(i, j) - abb_ijiijj(i, j);
      block_4x4[4] = block_4x4[1];  // 8, 7

      // 7, 9
      block_4x4[2] = -ab_jjij(i, j) + aab_iijjij(i, j);
      block_4x4[8] = block_4x4[2];  // 9, 7

      // 7, 10
      block_4x4[3] = ab_ijij(i, j);
      block_4x4[12] = block_4x4[3];  // 10, 7

      // 8, 8
      block_4x4[5] = ab_iijj(i, j) - abb_iiiijj(i, j) - aab_iijjjj(i, j) +
                     aabb_iijjiijj(i, j);

      // 8, 9
      block_4x4[6] = -ab_ijji(j, i);
      block_4x4[9] = block_4x4[6];  // 9, 8

      // 8, 10
      block_4x4[7] = ab_jjji(j, i) - aab_iijjij(i, j);
      block_4x4[13] = block_4x4[7];  // 10, 8

      // 9, 9
      block_4x4[10] = ab_iijj(j, i) - aab_iijjii(i, j) - abb_jjiijj(i, j) +
                      aabb_iijjiijj(i, j);

      // 9, 10
      block_4x4[11] = -ab_jijj(j, i) + abb_ijiijj(i, j);
      block_4x4[14] = block_4x4[11];  // 10, 9

      // 10, 10
      block_4x4[15] = ab_iiii[i] - aab_iijjii(i, j) - abb_iiiijj(i, j) +
                      aabb_iijjiijj(i, j);
      // compute eigenvalues of the 4x4 block 7-10, 7-10
      {
        auto eigs_4x4 = detail::eigenvalues_4x4(block_4x4.data());
        for (auto ev : eigs_4x4) add_entropy(ev);
      }

      // 11, 11
      val = aab_iijjjj(i, j) - aabb_iijjiijj(i, j);
      // 11, 12 = 12, 11
      val1 = -aab_iijjij(i, j);
      // 12, 12
      val2 = aab_iijjii(i, j) - aabb_iijjiijj(i, j);
      auto eigs_2x2_c = eigenvalues_2x2(val, val1, val2);
      for (auto v : eigs_2x2_c) add_entropy(v);

      // 13, 13
      val = abb_jjiijj(i, j) - aabb_iijjiijj(i, j);
      // 13, 14 = 14, 13
      val1 = -abb_ijiijj(i, j);
      // 14, 14
      val2 = abb_iiiijj(i, j) - aabb_iijjiijj(i, j);
      auto eigs_2x2_d = eigenvalues_2x2(val, val1, val2);
      for (auto v : eigs_2x2_d) add_entropy(v);

      // 15, 15
      val = aabb_iijjiijj(i, j);
      add_entropy(val);

      // symmetry
      s2_entropy(j, i) = s2_entropy(i, j);
    }
  }
}

/**
 * @brief Build the 16x16 two-orbital reduced density matrix block for a pair
 *        of orbitals.
 *
 * @param intermediates  Precomputed orbital RDM intermediates.
 * @param i              First orbital index.
 * @param j              Second orbital index.
 * @param[out] tordm     16x16 matrix span to receive the 2-ORDM block
 *                        (must be zero-initialized on entry).
 */
inline void build_2ordm_block(const OrbitalRDMIntermediates& intermediates,
                              size_t i, size_t j, matrix_span<double> tordm) {
  auto& a_ii = intermediates.a_ii;
  auto& b_ii = intermediates.b_ii;
  auto& a_ij = intermediates.a_ij;
  auto& b_ij = intermediates.b_ij;
  auto& ab_iiii = intermediates.ab_iiii;
  auto& aa_iijj = intermediates.aa_iijj;
  auto& bb_iijj = intermediates.bb_iijj;
  auto& ab_iijj = intermediates.ab_iijj;
  auto& ab_ijjj = intermediates.ab_ijjj;
  auto& ab_jijj = intermediates.ab_jijj;
  auto& ab_jjij = intermediates.ab_jjij;
  auto& ab_jjji = intermediates.ab_jjji;
  auto& ab_ijij = intermediates.ab_ijij;
  auto& ab_ijji = intermediates.ab_ijji;
  auto& aab_iijjjj = intermediates.aab_iijjjj;
  auto& abb_jjiijj = intermediates.abb_jjiijj;
  auto& aab_iijjii = intermediates.aab_iijjii;
  auto& abb_iiiijj = intermediates.abb_iiiijj;
  auto& abb_ijiijj = intermediates.abb_ijiijj;
  auto& aab_iijjij = intermediates.aab_iijjij;
  auto& aabb_iijjiijj = intermediates.aabb_iijjiijj;

  tordm(0, 0) = 1 - a_ii[i] - b_ii[i] - a_ii[j] - b_ii[j] + ab_iiii[i] +
                ab_iiii[j] + aa_iijj(i, j) + ab_iijj(i, j) + ab_iijj(j, i) +
                bb_iijj(i, j) - aab_iijjjj(i, j) - abb_jjiijj(i, j) -
                aab_iijjii(i, j) - abb_iiiijj(i, j) + aabb_iijjiijj(i, j);

  tordm(1, 1) = a_ii[j] - ab_iijj(j, i) - aa_iijj(i, j) - ab_iijj(j, j) +
                aab_iijjjj(i, j) + aab_iijjii(i, j) + abb_jjiijj(i, j) -
                aabb_iijjiijj(i, j);

  tordm(1, 2) = a_ij(i, j) - ab_jijj(j, i) - ab_ijjj(i, j) + abb_ijiijj(i, j);
  tordm(2, 1) = tordm(1, 2);

  tordm(2, 2) = a_ii[i] - ab_iijj(i, j) - aa_iijj(i, j) - ab_iijj(i, i) +
                aab_iijjjj(i, j) + aab_iijjii(i, j) + abb_iiiijj(i, j) -
                aabb_iijjiijj(i, j);

  tordm(3, 3) = b_ii[j] - ab_iijj(i, j) - bb_iijj(i, j) - ab_iijj(j, j) +
                abb_iiiijj(i, j) + aab_iijjjj(i, j) + abb_jjiijj(i, j) -
                aabb_iijjiijj(i, j);

  tordm(3, 4) = b_ij(i, j) - ab_jjij(j, i) - ab_jjji(i, j) + aab_iijjij(i, j);
  tordm(4, 3) = tordm(3, 4);

  tordm(4, 4) = b_ii[i] - ab_iijj(j, i) - bb_iijj(i, j) - ab_iijj(i, i) +
                abb_jjiijj(i, j) + aab_iijjii(i, j) + abb_iiiijj(i, j) -
                aabb_iijjiijj(i, j);

  tordm(5, 5) =
      aa_iijj(i, j) - aab_iijjii(i, j) - aab_iijjjj(i, j) + aabb_iijjiijj(i, j);

  tordm(6, 6) =
      bb_iijj(i, j) - abb_iiiijj(i, j) - abb_jjiijj(i, j) + aabb_iijjiijj(i, j);

  tordm(7, 7) =
      ab_iijj(j, j) - aab_iijjjj(i, j) - abb_jjiijj(i, j) + aabb_iijjiijj(i, j);

  tordm(7, 8) = ab_ijjj(i, j) - abb_ijiijj(i, j);
  tordm(8, 7) = tordm(7, 8);

  tordm(7, 9) = -ab_jjij(i, j) + aab_iijjij(i, j);
  tordm(9, 7) = tordm(7, 9);

  tordm(7, 10) = ab_ijij(i, j);
  tordm(10, 7) = tordm(7, 10);

  tordm(8, 8) =
      ab_iijj(i, j) - abb_iiiijj(i, j) - aab_iijjjj(i, j) + aabb_iijjiijj(i, j);

  tordm(8, 9) = -ab_ijji(j, i);
  tordm(9, 8) = tordm(8, 9);

  tordm(8, 10) = ab_jjji(j, i) - aab_iijjij(i, j);
  tordm(10, 8) = tordm(8, 10);

  tordm(9, 9) =
      ab_iijj(j, i) - aab_iijjii(i, j) - abb_jjiijj(i, j) + aabb_iijjiijj(i, j);

  tordm(9, 10) = -ab_jijj(j, i) + abb_ijiijj(i, j);
  tordm(10, 9) = tordm(9, 10);

  tordm(10, 10) =
      ab_iiii[i] - aab_iijjii(i, j) - abb_iiiijj(i, j) + aabb_iijjiijj(i, j);

  tordm(11, 11) = aab_iijjjj(i, j) - aabb_iijjiijj(i, j);

  tordm(11, 12) = -aab_iijjij(i, j);
  tordm(12, 11) = tordm(11, 12);

  tordm(12, 12) = aab_iijjii(i, j) - aabb_iijjiijj(i, j);

  tordm(13, 13) = abb_jjiijj(i, j) - aabb_iijjiijj(i, j);

  tordm(13, 14) = -abb_ijiijj(i, j);
  tordm(14, 13) = tordm(13, 14);

  tordm(14, 14) = abb_iiiijj(i, j) - aabb_iijjiijj(i, j);

  tordm(15, 15) = aabb_iijjiijj(i, j);
}

/**
 * @brief Build the full two-orbital reduced density matrix for all orbital
 *        pairs.
 *
 * @param intermediates  Precomputed orbital RDM intermediates.
 * @param[out] tordm     Rank-4 span of shape (norb, norb, 16, 16) to receive
 *                        the 2-ORDM blocks (must be zero-initialized).
 */
inline void build_2ordm(const OrbitalRDMIntermediates& intermediates,
                        rank4_span<double> tordm) {
  const size_t norb = intermediates.norb;
  // Flat storage for a single 16x16 block
  std::vector<double> block(16 * 16, 0.0);
  matrix_span<double> block_span(block.data(), 16, 16);

  for (size_t i = 0; i < norb; ++i) {
    for (size_t j = i; j < norb; ++j) {
      // Zero the block
      std::fill(block.begin(), block.end(), 0.0);

      // Fill it for this orbital pair
      build_2ordm_block(intermediates, i, j, block_span);

      // Copy into the 4D tensor
      for (size_t p = 0; p < 16; ++p) {
        for (size_t q = 0; q < 16; ++q) {
          tordm(i, j, p, q) = block_span(p, q);
          tordm(j, i, p, q) = block_span(p, q);  // symmetry
        }
      }
    }
  }
}

/**
 * @brief Compute mutual information between all pairs of orbitals.
 *
 * @param s1_entropy   Single-orbital entropies.
 * @param s2_entropy   Two-orbital entropies.
 * @param[out] mutual_info  norb x norb matrix span to receive the result.
 */
inline void build_mutual_information(const std::vector<double>& s1_entropy,
                                     const matrix_span<double>& s2_entropy,
                                     matrix_span<double>& mutual_info) {
  size_t norb = s1_entropy.size();
  for (size_t i = 0; i < norb; ++i) {
    for (size_t j = i + 1; j < norb; ++j) {
      mutual_info(i, j) = s1_entropy[i] + s1_entropy[j] - s2_entropy(i, j);
      mutual_info(j, i) = mutual_info(i, j);
    }
  }
}

/**
 * @brief Dispatch a bra-ket pair to the appropriate RDM intermediate
 *        accumulator.
 *
 * @tparam T          Scalar type.
 * @tparam N          Wavefunction bitset size.
 * @tparam IndexType  Container of orbital indices.
 * @param bra_alpha       Alpha part of the bra determinant.
 * @param ket_alpha       Alpha part of the ket determinant.
 * @param ex_alpha        Alpha excitation bitmask (bra XOR ket).
 * @param bra_beta        Beta part of the bra determinant.
 * @param ket_beta        Beta part of the ket determinant.
 * @param ex_beta         Beta excitation bitmask (bra XOR ket).
 * @param bra_occ_alpha   Occupied alpha orbitals in the bra.
 * @param bra_occ_beta    Occupied beta orbitals in the bra.
 * @param val             Coefficient product value to accumulate.
 * @param[in,out] intermediates  RDM intermediate storage to update.
 */
template <typename T, size_t N, typename IndexType>
inline void eval_ordm_intermediates(wfn_t<N> bra_alpha, wfn_t<N> ket_alpha,
                                    wfn_t<N> ex_alpha, wfn_t<N> bra_beta,
                                    wfn_t<N> ket_beta, wfn_t<N> ex_beta,
                                    const IndexType& bra_occ_alpha,
                                    const IndexType& bra_occ_beta, T val,
                                    OrbitalRDMIntermediates& intermediates) {
  using wfn_traits = wavefunction_traits<wfn_t<N>>;
  const uint32_t ex_alpha_count = wfn_traits::count(ex_alpha);
  const uint32_t ex_beta_count = wfn_traits::count(ex_beta);

  if (ex_alpha_count > 2 || ex_beta_count > 2) return;

  // Fast path: when only s1 is needed, accumulate diagonal vectors only
  if (!intermediates.need_s2) {
    if (ex_alpha_count == 0 && ex_beta_count == 0) {
      orbital_rdm_contrib_diag_s1(bra_occ_alpha, bra_occ_beta, val,
                                  intermediates.a_ii, intermediates.b_ii,
                                  intermediates.ab_iiii);
    }
    return;
  }

  if (ex_alpha_count == 0 && ex_beta_count == 0) {
    // Diagonal contribution
    orbital_rdm_contrib_diag(
        bra_occ_alpha, bra_occ_beta, val, intermediates.a_ii,
        intermediates.b_ii, intermediates.ab_iiii, intermediates.aa_iijj.span,
        intermediates.bb_iijj.span, intermediates.ab_iijj.span,
        intermediates.aab_iijjjj.span, intermediates.abb_jjiijj.span,
        intermediates.aab_iijjii.span, intermediates.abb_iiiijj.span,
        intermediates.aabb_iijjiijj.span);
  } else if (ex_alpha_count == 2 && ex_beta_count == 0) {
    // Single alpha excitation
    orbital_rdm_contrib_2<false>(
        bra_alpha, ket_alpha, ex_alpha, bra_occ_alpha, bra_occ_beta, val,
        intermediates.a_ij.span, intermediates.ab_ijjj.span,
        intermediates.ab_jjij.span, intermediates.ab_jijj.span,
        intermediates.ab_jjji.span, intermediates.aab_iijjij.span,
        intermediates.abb_ijiijj.span);
  } else if (ex_alpha_count == 0 && ex_beta_count == 2) {
    // Single beta excitation
    orbital_rdm_contrib_2<true>(
        bra_beta, ket_beta, ex_beta, bra_occ_beta, bra_occ_alpha, val,
        intermediates.b_ij.span, intermediates.ab_ijjj.span,
        intermediates.ab_jjij.span, intermediates.ab_jijj.span,
        intermediates.ab_jjji.span, intermediates.aab_iijjij.span,
        intermediates.abb_ijiijj.span);
  } else if (ex_alpha_count == 2 && ex_beta_count == 2) {
    // Mixed alpha-beta single excitations
    orbital_rdm_contrib_22(bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta,
                           ex_beta, val, intermediates.ab_ijij.span,
                           intermediates.ab_ijji.span);
  }
}

}  // namespace macis

/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#include <algorithm>
#include <blas.hh>
#include <macis/fci/fci_rdms.hpp>
#include <stdexcept>
#include <vector>

namespace macis {

namespace {

// Column strings contracted per DGEMM
constexpr size_t col_block = 128;

/*
 * Accumulates, over all rows I of the row-major CI matrix C (rows: strings of
 * `rows`, columns: strings of `cols`), with E the excitations of the row spin
 * and F those of the column spin:
 *   gamma(p,q)       += <E_pq>
 *   gamma2(p,q,r,s)  += <E_pq E_rs>
 *   mixed(p,q,r,s)   += <F_pq E_rs>            (when mixed is non-null)
 * in column-major order p + n*(q + n*(r + n*s)).
 *
 * Per row I, column l of X holds (E_xy C)(I, :) for the l-th single
 * excitation of I (x destroyed, y created) and column 0 holds C(I, :), which
 * equals (E_pp C)(I, :) for every occupied p. Then <E_pq E_rs> collects
 * X(:,l1).X(:,l2) for labels (q,p) and (r,s), and <F_pq E_rs> collects
 * D(:,qp).X(:,l) with D(:,xy) = (F_xy C)(I, :).
 */
void accumulate_rdms(const FCIStringSpace& rows, const FCIStringSpace& cols,
                     const double* C, std::vector<double>& gamma,
                     std::vector<double>& gamma2, std::vector<double>* mixed) {
  const size_t n = rows.norb();
  const size_t n2 = n * n;
  const size_t n4 = n2 * n2;
  const size_t nc = cols.size();
  const size_t nexc = rows.num_excitations();
  const size_t nexc_c = cols.num_excitations();
  const size_t K = nexc + 1;
  auto idx4 = [n](size_t p, size_t q, size_t r, size_t s) {
    return p + n * (q + n * (r + n * s));
  };

#pragma omp parallel
  {
    std::vector<double> g1(n2, 0.0), g2(n4, 0.0), gm(mixed ? n4 : 0, 0.0);
    std::vector<double> X(col_block * K), G(K * K);
    std::vector<double> D(mixed ? col_block * n2 : 0), GM(mixed ? n2 * K : 0);
    // (x,y) labels of the X columns; column 0 stands for all occupied (p,p)
    std::vector<std::pair<uint32_t, uint32_t>> diag_labels;

#pragma omp for schedule(dynamic)
    for (size_t I = 0; I < rows.size(); ++I) {
      const uint32_t* occ = rows.occupied(I);
      const auto* ex = rows.excitations(I);
      const double* c = C + I * nc;
      std::fill(G.begin(), G.end(), 0.0);
      std::fill(GM.begin(), GM.end(), 0.0);

      for (size_t b0 = 0; b0 < nc; b0 += col_block) {
        const size_t bl = std::min(col_block, nc - b0);
        std::copy_n(c + b0, bl, X.data());
        for (size_t l = 0; l < nexc; ++l) {
          const double* cj = C + size_t(ex[l].target) * nc + b0;
          double* x = X.data() + (l + 1) * bl;
          for (size_t b = 0; b < bl; ++b) x[b] = ex[l].sign * cj[b];
        }
        blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                   K, K, bl, 1.0, X.data(), bl, X.data(), bl, 1.0, G.data(), K);

        if (mixed) {
          std::fill_n(D.data(), bl * n2, 0.0);
          for (size_t b = 0; b < bl; ++b) {
            const size_t J = b0 + b;
            const auto* exc = cols.excitations(J);
            for (size_t e = 0; e < nexc_c; ++e)
              D[b + (exc[e].destroy + n * exc[e].create) * bl] +=
                  exc[e].sign * c[exc[e].target];
            const uint32_t* occc = cols.occupied(J);
            for (size_t q = 0; q < cols.nelec(); ++q)
              D[b + occc[q] * (n + 1) * bl] += c[J];
          }
          blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
                     n2, K, bl, 1.0, D.data(), bl, X.data(), bl, 1.0, GM.data(),
                     n2);
        }
      }

      diag_labels.clear();
      for (size_t a = 0; a < rows.nelec(); ++a)
        diag_labels.emplace_back(occ[a], occ[a]);
      auto for_labels = [&](size_t l, auto&& f) {
        if (l == 0)
          for (auto [x, y] : diag_labels) f(x, y);
        else
          f(ex[l - 1].destroy, ex[l - 1].create);
      };

      for (size_t l = 0; l < K; ++l)
        for_labels(l, [&](size_t x, size_t y) { g1[x + n * y] += G[l * K]; });

      for (size_t l2 = 0; l2 < K; ++l2)
        for_labels(l2, [&](size_t r, size_t s) {
          for (size_t l1 = 0; l1 < K; ++l1) {
            const double g = G[l1 + l2 * K];
            for_labels(l1,
                       [&](size_t q, size_t p) { g2[idx4(p, q, r, s)] += g; });
          }
          if (mixed)
            for (size_t q = 0; q < n; ++q)
              for (size_t p = 0; p < n; ++p)
                gm[idx4(p, q, r, s)] += GM[q + n * p + l2 * n2];
        });
    }

#pragma omp critical
    {
      for (size_t i = 0; i < n2; ++i) gamma[i] += g1[i];
      for (size_t i = 0; i < n4; ++i) gamma2[i] += g2[i];
      if (mixed)
        for (size_t i = 0; i < n4; ++i) (*mixed)[i] += gm[i];
    }
  }
}

}  // namespace

void fci_rdms_spin_dep(const FCIStringSpace& alpha, const FCIStringSpace& beta,
                       const double* C, matrix_span<double> ordm_aa,
                       matrix_span<double> ordm_bb,
                       rank4_span<double> trdm_aaaa,
                       rank4_span<double> trdm_bbbb,
                       rank4_span<double> trdm_aabb, int spin_flip) {
  if (spin_flip != 0 and
      (alpha.norb() != beta.norb() or alpha.nelec() != beta.nelec()))
    throw std::invalid_argument(
        "fci_rdms_spin_dep: spin-flip symmetry requires equal alpha and beta "
        "strings");
  const size_t n = alpha.norb();
  const size_t n2 = n * n;
  const size_t n4 = n2 * n2;
  const size_t na = alpha.size();
  const size_t nb = beta.size();
  const bool need_alpha = ordm_aa.data_handle() or trdm_aaaa.data_handle();
  const bool need_beta = ordm_bb.data_handle() or trdm_bbbb.data_handle();
  const bool need_mixed = trdm_aabb.data_handle() != nullptr;

  auto write_same_spin =
      [&](const std::vector<double>& g, const std::vector<double>& G,
          matrix_span<double> ordm, rank4_span<double> trdm) {
        if (ordm.data_handle())
          for (size_t q = 0; q < n; ++q)
            for (size_t p = 0; p < n; ++p) ordm(p, q) = g[p + n * q];
        // a+_p a+_r a_s a_q = E_pq E_rs - delta_qr E_ps
        if (trdm.data_handle())
          for (size_t s = 0; s < n; ++s)
            for (size_t r = 0; r < n; ++r)
              for (size_t q = 0; q < n; ++q)
                for (size_t p = 0; p < n; ++p)
                  trdm(p, q, r, s) = 0.5 * (G[p + n * (q + n * (r + n * s))] -
                                            (q == r ? g[p + n * s] : 0.0));
      };

  if (need_alpha or need_mixed or (spin_flip != 0 and need_beta)) {
    std::vector<double> g(n2, 0.0), G(n4, 0.0), mixed(need_mixed ? n4 : 0, 0.0);
    accumulate_rdms(alpha, beta, C, g, G, need_mixed ? &mixed : nullptr);
    write_same_spin(g, G, ordm_aa, trdm_aaaa);
    if (spin_flip != 0) write_same_spin(g, G, ordm_bb, trdm_bbbb);
    if (need_mixed)
      for (size_t s = 0; s < n; ++s)
        for (size_t r = 0; r < n; ++r)
          for (size_t q = 0; q < n; ++q)
            for (size_t p = 0; p < n; ++p)
              trdm_aabb(p, q, r, s) =
                  0.5 * mixed[p + n * (q + n * (r + n * s))];
  }

  if (need_beta and spin_flip == 0) {
    // Beta-major copy so that beta excitations become row operations
    constexpr size_t tile = 32;
    std::vector<double> CT(na * nb);
#pragma omp parallel for collapse(2) schedule(static)
    for (size_t i0 = 0; i0 < na; i0 += tile)
      for (size_t j0 = 0; j0 < nb; j0 += tile)
        for (size_t i = i0; i < std::min(i0 + tile, na); ++i)
          for (size_t j = j0; j < std::min(j0 + tile, nb); ++j)
            CT[j * na + i] = C[i * nb + j];

    std::vector<double> g(n2, 0.0), G(n4, 0.0);
    accumulate_rdms(beta, alpha, CT.data(), g, G, nullptr);
    write_same_spin(g, G, ordm_bb, trdm_bbbb);
  }
}

}  // namespace macis

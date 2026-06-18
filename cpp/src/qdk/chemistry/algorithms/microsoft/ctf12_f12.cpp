// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ctf12_f12.hpp"

#include <qdk/chemistry/scf/util/geminal_eri.h>

#include <cmath>
#include <memory>
#include <qdk/chemistry/utils/logger.hpp>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft::ctf12 {

namespace {

namespace gem = qdk::chemistry::scf::geminal;

// Row-major 4-index accessor over an MO integral buffer of shape [*,n2,n3,n4].
struct Tensor4 {
  const double* data;
  std::size_t n2, n3, n4;
  double operator()(std::size_t a, std::size_t b, std::size_t c,
                    std::size_t d) const {
    return data[((a * n2 + b) * n3 + c) * n4 + d];
  }
};

// Molecular-orbital integral context in the RI-MO basis
// Cri_mo = [occupied | OBS-virtual | CABS]. Holds everything required to build
// the diagonal V/X/B intermediates and the dressed transcorrelated Hamiltonian.
struct Workspace {
  std::size_t n_ao_obs, n_ri, n_mo, n_occ, n_core, n_val, n_vir, n_cabs, n_cv,
      n_rimo;
  double gamma, pref, g2, inv_g2;

  Eigen::MatrixXd c_obs;     // OBS MO over OBS AO [n_ao_obs, n_mo]
  Eigen::MatrixXd c_val;     // valence MO over OBS AO
  Eigen::MatrixXd c_obs_ri;  // OBS MO over RI AO [n_ri, n_mo]
  Eigen::MatrixXd c_cabs;    // CABS over RI AO [n_ri, n_cabs]
  Eigen::MatrixXd c_ri_mo;   // RI-MO over RI AO [n_ri, n_rimo]
  Eigen::VectorXd eps;       // orbital energies [n_mo]

  Eigen::MatrixXd h_ri;      // one-electron H over RI AO
  Eigen::MatrixXd f_rimo;    // Fock over RI-MO
  Eigen::MatrixXd k_rimo;    // exchange operator over RI-MO (Psi4 "k")
  Eigen::MatrixXd fpk_rimo;  // h + 2J = f_rimo + k_rimo

  // MO integral buffers (kept alive; accessed through Tensor4 views).
  std::unique_ptr<double[]> gfr;  // <PQ|f12|ij>   [n_rimo,n_val,n_rimo,n_val]
  std::unique_ptr<double[]> cf;   // <ij|g|PQ>     [n_val,n_rimo,n_val,n_rimo]
  std::unique_ptr<double[]> fg;   // <ij|f12 g|kl> [n_val,n_val,n_val,n_val]
  std::unique_ptr<double[]> e2;   // <ij|e^-2gr|kl>[n_val,n_val,n_val,n_val]
  std::unique_ptr<double[]> s2;   // <ij|e^-2gr|PQ>[n_val,n_rimo,n_val,n_rimo]

  std::vector<double> g_sp;  // SP-coupled geminal G^{PQ}_{ij}

  Tensor4 GFR() const { return {gfr.get(), n_val, n_rimo, n_val}; }
  Tensor4 CF() const { return {cf.get(), n_rimo, n_val, n_rimo}; }
  Tensor4 FG() const { return {fg.get(), n_val, n_val, n_val}; }
  Tensor4 E2() const { return {e2.get(), n_val, n_val, n_val}; }
  Tensor4 S2() const { return {s2.get(), n_rimo, n_val, n_rimo}; }

  std::size_t gfi(std::size_t p, std::size_t i, std::size_t q,
                  std::size_t j) const {
    return ((p * n_val + i) * n_rimo + q) * n_val + j;
  }
  // SP-coupled geminal G^{PQ}_{ij} (P,Q in RI-MO, i,j valence).
  double G(std::size_t p, std::size_t i, std::size_t q, std::size_t j) const {
    return g_sp[gfi(p, i, q, j)];
  }
  // Raw geminal r^{ij}_{PQ} = <ij|f12|PQ>.
  double rg(std::size_t i, std::size_t j, std::size_t p, std::size_t q) const {
    return pref * GFR()(p, i, q, j);
  }
  // <ij|g|PQ>, valence i,j and RI-MO P,Q.
  double coul(std::size_t i, std::size_t j, std::size_t p,
              std::size_t q) const {
    return CF()(i, p, j, q);
  }
  // <ij|f12^2|PQ> via e^{-2 gamma r}; valence i,j (active-occ index occ_idx),
  // RI-MO P,Q.
  double occ_idx(std::size_t k) const { return n_core + k; }
};

Workspace build_workspace(const F12HartreeFockInput& in) {
  Workspace w;
  w.n_ao_obs = static_cast<std::size_t>(in.obs.nbf());
  w.n_ri = static_cast<std::size_t>(in.cabs_ri_basis.nbf());
  w.n_mo = static_cast<std::size_t>(in.mo_coefficients.cols());
  w.n_occ = in.n_occupied;
  w.n_core = in.n_core;
  w.n_val = w.n_occ - w.n_core;
  w.n_vir = w.n_mo - w.n_occ;
  w.n_cabs = static_cast<std::size_t>(in.cabs_coefficients.cols());
  w.n_cv = w.n_vir + w.n_cabs;
  w.n_rimo = w.n_occ + w.n_cv;
  w.gamma = in.gamma;
  w.pref = -1.0 / in.gamma;
  w.g2 = in.gamma * in.gamma;
  w.inv_g2 = 1.0 / w.g2;

  const auto& obs = in.obs;
  const auto& ri = in.cabs_ri_basis;
  const std::size_t nao = w.n_ao_obs, nri = w.n_ri;

  w.c_obs = in.mo_coefficients;
  w.c_val = in.mo_coefficients.middleCols(static_cast<int>(w.n_core),
                                          static_cast<int>(w.n_val));
  w.eps = in.orbital_energies;
  w.c_cabs = in.cabs_coefficients;

  Eigen::MatrixXd c_occ =
      in.mo_coefficients.leftCols(static_cast<int>(w.n_occ));

  // Complete-virtual coefficients over RI AO: [OBS-virtual padded | CABS].
  Eigen::MatrixXd c_cv = Eigen::MatrixXd::Zero(nri, w.n_cv);
  c_cv.topLeftCorner(nao, w.n_vir) =
      in.mo_coefficients.rightCols(static_cast<int>(w.n_vir));
  c_cv.rightCols(static_cast<int>(w.n_cabs)) = w.c_cabs;

  Eigen::MatrixXd c_occ_ri = Eigen::MatrixXd::Zero(nri, w.n_occ);
  c_occ_ri.topRows(nao) = c_occ;
  w.c_obs_ri = Eigen::MatrixXd::Zero(nri, w.n_mo);
  w.c_obs_ri.topRows(nao) = in.mo_coefficients;
  w.c_ri_mo = Eigen::MatrixXd(nri, w.n_rimo);
  w.c_ri_mo.leftCols(static_cast<int>(w.n_occ)) = c_occ_ri;
  w.c_ri_mo.rightCols(static_cast<int>(w.n_cv)) = c_cv;

  // Fock over RI AO: f = h + 2J - K with closed-shell OBS density.
  w.h_ri = gem::kinetic_matrix(ri) + gem::nuclear_matrix(ri, in.nuclei);
  Eigen::MatrixXd d_obs = 2.0 * c_occ * c_occ.transpose();
  auto cj = gem::four_center_coulomb(ri, ri, obs, obs);
  auto ck = gem::four_center_coulomb(ri, obs, ri, obs);
  Eigen::MatrixXd jmat = Eigen::MatrixXd::Zero(nri, nri);
  Eigen::MatrixXd kmat = Eigen::MatrixXd::Zero(nri, nri);
  for (std::size_t mu = 0; mu < nri; ++mu)
    for (std::size_t nu = 0; nu < nri; ++nu) {
      double jv = 0.0, kv = 0.0;
      for (std::size_t la = 0; la < nao; ++la)
        for (std::size_t si = 0; si < nao; ++si) {
          const double d = d_obs(static_cast<int>(la), static_cast<int>(si));
          jv += cj[((mu * nri + nu) * nao + la) * nao + si] * d;
          kv += ck[((mu * nao + la) * nri + nu) * nao + si] * d;
        }
      jmat(static_cast<int>(mu), static_cast<int>(nu)) = jv;
      kmat(static_cast<int>(mu), static_cast<int>(nu)) = kv;
    }
  Eigen::MatrixXd f_ri = w.h_ri + jmat - 0.5 * kmat;
  w.f_rimo = w.c_ri_mo.transpose() * f_ri * w.c_ri_mo;
  w.k_rimo = 0.5 * (w.c_ri_mo.transpose() * kmat * w.c_ri_mo);
  w.fpk_rimo = w.f_rimo + w.k_rimo;

  // Geminal <PQ|f12|ij> over RI-MO x valence and its SP-coupled form.
  auto stg_ao = gem::stg_geminal_eri(::libint2::Operator::stg, in.gamma, ri,
                                     obs, ri, obs);
  w.gfr = gem::mo_transform_4index(stg_ao.get(), nri, nao, nri, nao, w.c_ri_mo,
                                   w.c_val, w.c_ri_mo, w.c_val);
  w.g_sp.assign(w.n_rimo * w.n_val * w.n_rimo * w.n_val, 0.0);
  Tensor4 gfrv = w.GFR();
  for (std::size_t p = 0; p < w.n_rimo; ++p)
    for (std::size_t q = 0; q < w.n_rimo; ++q)
      for (std::size_t i = 0; i < w.n_val; ++i)
        for (std::size_t j = 0; j < w.n_val; ++j)
          w.g_sp[w.gfi(p, i, q, j)] =
              w.pref * (0.375 * gfrv(p, i, q, j) + 0.125 * gfrv(p, j, q, i));

  // Coulomb <ij|g|PQ> over valence x RI-MO.
  auto cf_ao = gem::four_center_coulomb(obs, ri, obs, ri);
  w.cf = gem::mo_transform_4index(cf_ao.get(), nao, nri, nao, nri, w.c_val,
                                  w.c_ri_mo, w.c_val, w.c_ri_mo);

  // f12*g and e^{-2 gamma r} over valence^4.
  auto fg_ao = gem::stg_geminal_eri(::libint2::Operator::stg_x_coulomb,
                                    in.gamma, obs, obs, obs, obs);
  w.fg = gem::mo_transform_4index(fg_ao.get(), nao, nao, nao, nao, w.c_val,
                                  w.c_val, w.c_val, w.c_val);
  auto e2_ao = gem::stg_geminal_eri(::libint2::Operator::stg, 2.0 * in.gamma,
                                    obs, obs, obs, obs);
  w.e2 = gem::mo_transform_4index(e2_ao.get(), nao, nao, nao, nao, w.c_val,
                                  w.c_val, w.c_val, w.c_val);
  // <ij|e^{-2 gamma r}|PQ> over valence x RI-MO (for the B metric term).
  auto s2_ao = gem::stg_geminal_eri(::libint2::Operator::stg, 2.0 * in.gamma,
                                    obs, ri, obs, ri);
  w.s2 = gem::mo_transform_4index(s2_ao.get(), nao, nri, nao, nri, w.c_val,
                                  w.c_ri_mo, w.c_val, w.c_ri_mo);
  return w;
}

// Diagonal F12 intermediates over the valence space, SP (cusp) coupled.
struct VXB {
  std::vector<double> v, x, b;  // [n_val^4]
};

std::size_t vidx(std::size_t n, std::size_t i, std::size_t j, std::size_t k,
                 std::size_t l) {
  return ((i * n + j) * n + k) * n + l;
}

VXB compute_vxb(const Workspace& w) {
  const std::size_t nv = w.n_val, nocc = w.n_occ, nrimo = w.n_rimo;
  const std::size_t nobs = w.n_mo;
  const std::size_t sz = nv * nv * nv * nv;
  auto P = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    return vidx(nv, i, j, k, l);
  };
  Tensor4 FG = w.FG(), E2 = w.E2(), S2 = w.S2();

  // Psi4-form V and X (direct - occ-CABS - OBS-OBS projectors).
  std::vector<double> vp(sz, 0.0), xp(sz, 0.0);
  for (std::size_t i = 0; i < nv; ++i)
    for (std::size_t j = 0; j < nv; ++j)
      for (std::size_t k = 0; k < nv; ++k)
        for (std::size_t l = 0; l < nv; ++l) {
          double vv = w.pref * FG(i, k, j, l);
          double xx = E2(i, k, j, l) * w.inv_g2;
          for (std::size_t m = 0; m < nocc; ++m)
            for (std::size_t y = nobs; y < nrimo; ++y) {
              vv -= w.coul(i, j, m, y) * w.rg(k, l, m, y);
              vv -= w.coul(j, i, m, y) * w.rg(l, k, m, y);
              xx -= w.rg(i, j, m, y) * w.rg(k, l, m, y);
              xx -= w.rg(j, i, m, y) * w.rg(l, k, m, y);
            }
          for (std::size_t p = 0; p < nobs; ++p)
            for (std::size_t q = 0; q < nobs; ++q) {
              vv -= w.coul(i, j, p, q) * w.rg(k, l, p, q);
              xx -= w.rg(i, j, p, q) * w.rg(k, l, p, q);
            }
          vp[P(i, j, k, l)] = vv;
          xp[P(i, j, k, l)] = xx;
        }

  // 8-term approximation-C B (Kedzuch/MPQC form), Hermitized.
  auto occ_idx = [&](std::size_t k) { return w.n_core + k; };
  auto r2g = [&](std::size_t k, std::size_t l, std::size_t m, std::size_t I) {
    return w.inv_g2 * S2(k, occ_idx(m), l, I);
  };
  std::vector<double> b8(sz, 0.0);
  for (std::size_t k = 0; k < nv; ++k)
    for (std::size_t l = 0; l < nv; ++l)
      for (std::size_t m = 0; m < nv; ++m)
        for (std::size_t n = 0; n < nv; ++n) {
          double t1 = E2(k, m, l, n);
          double t2 = 0, t3 = 0, t4 = 0, t5 = 0, t6 = 0, t7 = 0, t8 = 0;
          for (std::size_t I = 0; I < nrimo; ++I)
            t2 += r2g(k, l, m, I) * w.fpk_rimo(static_cast<int>(occ_idx(n)),
                                               static_cast<int>(I)) +
                  r2g(l, k, n, I) * w.fpk_rimo(static_cast<int>(occ_idx(m)),
                                               static_cast<int>(I));
          for (std::size_t Pp = 0; Pp < nrimo; ++Pp)
            for (std::size_t A = 0; A < nrimo; ++A) {
              double rkC = 0, rlC = 0;
              for (std::size_t Cc = 0; Cc < nrimo; ++Cc) {
                rkC += w.rg(k, l, Pp, Cc) *
                       w.k_rimo(static_cast<int>(Cc), static_cast<int>(A));
                rlC += w.rg(l, k, Pp, Cc) *
                       w.k_rimo(static_cast<int>(Cc), static_cast<int>(A));
              }
              t3 -= rkC * w.rg(m, n, Pp, A) + rlC * w.rg(n, m, Pp, A);
            }
          for (std::size_t jp = 0; jp < nocc; ++jp)
            for (std::size_t A = 0; A < nrimo; ++A) {
              double rkC = 0, rlC = 0;
              for (std::size_t Cc = 0; Cc < nrimo; ++Cc) {
                rkC += w.rg(k, l, jp, Cc) *
                       w.f_rimo(static_cast<int>(Cc), static_cast<int>(A));
                rlC += w.rg(l, k, jp, Cc) *
                       w.f_rimo(static_cast<int>(Cc), static_cast<int>(A));
              }
              t4 -= rkC * w.rg(m, n, jp, A) + rlC * w.rg(n, m, jp, A);
            }
          for (std::size_t x = nobs; x < nrimo; ++x)
            for (std::size_t ip = 0; ip < nocc; ++ip) {
              double rkj = 0, rlj = 0;
              for (std::size_t jp = 0; jp < nocc; ++jp) {
                rkj += w.rg(k, l, x, jp) *
                       w.f_rimo(static_cast<int>(jp), static_cast<int>(ip));
                rlj += w.rg(l, k, x, jp) *
                       w.f_rimo(static_cast<int>(jp), static_cast<int>(ip));
              }
              t5 += rkj * w.rg(m, n, x, ip) + rlj * w.rg(n, m, x, ip);
            }
          for (std::size_t b = nocc; b < nobs; ++b)
            for (std::size_t pp = 0; pp < nobs; ++pp) {
              double rkr = 0, rlr = 0;
              for (std::size_t rr = 0; rr < nobs; ++rr) {
                rkr += w.rg(k, l, b, rr) *
                       w.f_rimo(static_cast<int>(rr), static_cast<int>(pp));
                rlr += w.rg(l, k, b, rr) *
                       w.f_rimo(static_cast<int>(rr), static_cast<int>(pp));
              }
              t6 -= rkr * w.rg(m, n, b, pp) + rlr * w.rg(n, m, b, pp);
            }
          for (std::size_t x = nobs; x < nrimo; ++x)
            for (std::size_t jp = 0; jp < nocc; ++jp) {
              double rkI = 0, rlI = 0;
              for (std::size_t I = 0; I < nrimo; ++I) {
                rkI += w.rg(k, l, x, I) *
                       w.f_rimo(static_cast<int>(jp), static_cast<int>(I));
                rlI += w.rg(l, k, x, I) *
                       w.f_rimo(static_cast<int>(jp), static_cast<int>(I));
              }
              t7 -= 2.0 * (rkI * w.rg(m, n, x, jp) + rlI * w.rg(n, m, x, jp));
            }
          for (std::size_t b = nocc; b < nobs; ++b)
            for (std::size_t x = nobs; x < nrimo; ++x) {
              double rkr = 0, rlr = 0;
              for (std::size_t rr = 0; rr < nobs; ++rr) {
                rkr += w.rg(k, l, b, rr) *
                       w.f_rimo(static_cast<int>(rr), static_cast<int>(x));
                rlr += w.rg(l, k, b, rr) *
                       w.f_rimo(static_cast<int>(rr), static_cast<int>(x));
              }
              t8 -= 2.0 * (rkr * w.rg(m, n, b, x) + rlr * w.rg(n, m, b, x));
            }
          b8[P(k, l, m, n)] = t1 + t2 + t3 + t4 + t5 + t6 + t7 + t8;
        }
  std::vector<double> b8s(sz, 0.0);
  for (std::size_t k = 0; k < nv; ++k)
    for (std::size_t l = 0; l < nv; ++l)
      for (std::size_t m = 0; m < nv; ++m)
        for (std::size_t n = 0; n < nv; ++n)
          b8s[P(k, l, m, n)] = 0.5 * (b8[P(k, l, m, n)] + b8[P(m, n, k, l)]);

  // SP (cusp) coupling: V on the geminal pair, X and B on both pairs.
  VXB out;
  out.v.assign(sz, 0.0);
  out.x.assign(sz, 0.0);
  out.b.assign(sz, 0.0);
  for (std::size_t i = 0; i < nv; ++i)
    for (std::size_t j = 0; j < nv; ++j)
      for (std::size_t k = 0; k < nv; ++k)
        for (std::size_t l = 0; l < nv; ++l) {
          out.v[P(i, j, k, l)] =
              0.375 * vp[P(i, j, k, l)] + 0.125 * vp[P(i, j, l, k)];
          out.x[P(i, j, k, l)] =
              (9.0 * xp[P(i, j, k, l)] + 3.0 * xp[P(i, j, l, k)] +
               3.0 * xp[P(j, i, k, l)] + 1.0 * xp[P(j, i, l, k)]) /
              64.0;
          out.b[P(i, j, k, l)] =
              (9.0 * b8s[P(i, j, k, l)] + 3.0 * b8s[P(i, j, l, k)] +
               3.0 * b8s[P(j, i, k, l)] + 1.0 * b8s[P(j, i, l, k)]) /
              64.0;
        }
  return out;
}

}  // namespace

F12Intermediates build_intermediates(const F12HartreeFockInput& in) {
  QDK_LOG_TRACE_ENTERING();
  Workspace w = build_workspace(in);
  VXB vxb = compute_vxb(w);
  F12Intermediates out;
  out.n_val = w.n_val;
  out.v = std::move(vxb.v);
  out.x = std::move(vxb.x);
  out.b = std::move(vxb.b);
  out.valence_energies = in.orbital_energies.segment(
      static_cast<int>(in.n_core), static_cast<int>(w.n_val));
  return out;
}

double f12_hf_energy(const F12Intermediates& im) {
  QDK_LOG_TRACE_ENTERING();
  const std::size_t n = im.n_val;
  auto idx = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    return vidx(n, i, j, k, l);
  };
  double energy = 0.0;
  for (std::size_t i = 0; i < n; ++i)
    for (std::size_t j = 0; j < n; ++j) {
      const double eij = im.valence_energies(static_cast<Eigen::Index>(i)) +
                         im.valence_energies(static_cast<Eigen::Index>(j));
      const double v_term = 2.0 * im.v[idx(i, j, i, j)] - im.v[idx(i, j, j, i)];
      const double x_term = 2.0 * im.x[idx(i, j, i, j)] - im.x[idx(i, j, j, i)];
      const double b_term = 2.0 * im.b[idx(i, j, i, j)] - im.b[idx(i, j, j, i)];
      energy += 2.0 * v_term - eij * x_term + b_term;
    }
  return energy;
}

double mp2_energy(const F12HartreeFockInput& in) {
  QDK_LOG_TRACE_ENTERING();
  const auto& obs = in.obs;
  const std::size_t nao = static_cast<std::size_t>(obs.nbf());
  const std::size_t nmo = static_cast<std::size_t>(in.mo_coefficients.cols());
  const std::size_t nocc = in.n_occupied, nc = in.n_core;

  auto cc_ao = gem::four_center_coulomb(obs, obs, obs, obs);
  auto cc_mo = gem::mo_transform_4index(cc_ao.get(), nao, nao, nao, nao,
                                        in.mo_coefficients, in.mo_coefficients,
                                        in.mo_coefficients, in.mo_coefficients);
  Tensor4 chem{cc_mo.get(), nmo, nmo,
               nmo};  // chemist (pq|rs); <pq|rs> = (pr|qs)

  // Orbital energies self-consistent with these integrals (diagonal Fock),
  // rather than the input energies which come from a different SCF backend.
  Eigen::MatrixXd h_ao =
      gem::kinetic_matrix(obs) + gem::nuclear_matrix(obs, in.nuclei);
  Eigen::MatrixXd hmo =
      in.mo_coefficients.transpose() * h_ao * in.mo_coefficients;
  Eigen::VectorXd eps(nmo);
  for (std::size_t p = 0; p < nmo; ++p) {
    double f = hmo(static_cast<int>(p), static_cast<int>(p));
    for (std::size_t i = 0; i < nocc; ++i)
      f += 2.0 * chem(p, p, i, i) - chem(p, i, i, p);
    eps(static_cast<int>(p)) = f;
  }

  double energy = 0.0;
  for (std::size_t i = nc; i < nocc; ++i)
    for (std::size_t j = nc; j < nocc; ++j)
      for (std::size_t a = nocc; a < nmo; ++a)
        for (std::size_t b = nocc; b < nmo; ++b) {
          const double iajb = chem(i, a, j, b);  // <ij|ab>
          const double ibja = chem(i, b, j, a);  // <ij|ba>
          energy += iajb * (2.0 * iajb - ibja) /
                    (eps(static_cast<int>(i)) + eps(static_cast<int>(j)) -
                     eps(static_cast<int>(a)) - eps(static_cast<int>(b)));
        }
  return energy;
}

namespace {

struct DressedResult {
  double e_hf;
  double e_f12hf;
  std::vector<double> gbar;     // dressed <pq|rs> in the original MO basis
  Eigen::MatrixXd c_relaxed;    // original-MO -> F12-HF-relaxed-MO rotation
  Eigen::VectorXd eps_relaxed;  // dressed-Fock eigenvalues
  std::size_t nbf;
  std::size_t nocc;
  std::size_t ncore;
};

DressedResult run_f12_hf(const F12HartreeFockInput& in) {
  QDK_LOG_TRACE_ENTERING();
  Workspace w = build_workspace(in);
  VXB vxb = compute_vxb(w);

  const std::size_t nbf = w.n_mo, nv = w.n_val, nocc = w.n_occ, nc = w.n_core;
  const std::size_t nvir = w.n_vir, ncabs = w.n_cabs, nrimo = w.n_rimo;
  const auto& obs = in.obs;
  const auto& ri = in.cabs_ri_basis;
  const std::size_t nao = w.n_ao_obs, nri = w.n_ri;
  auto P = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    return vidx(nv, i, j, k, l);
  };

  // Bare one-electron and Coulomb integrals over the orbital basis.
  Eigen::MatrixXd hmo = w.c_obs_ri.transpose() * w.h_ri * w.c_obs_ri;
  auto cc_ao = gem::four_center_coulomb(obs, obs, obs, obs);
  auto cc_mo = gem::mo_transform_4index(cc_ao.get(), nao, nao, nao, nao,
                                        w.c_obs, w.c_obs, w.c_obs, w.c_obs);
  Tensor4 CHEM{cc_mo.get(), nbf, nbf, nbf};
  auto gpidx = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    return ((p * nbf + q) * nbf + r) * nbf + s;
  };
  std::vector<double> gp(nbf * nbf * nbf * nbf, 0.0);  // <pq|rs>
  for (std::size_t p = 0; p < nbf; ++p)
    for (std::size_t q = 0; q < nbf; ++q)
      for (std::size_t r = 0; r < nbf; ++r)
        for (std::size_t s = 0; s < nbf; ++s)
          gp[gpidx(p, q, r, s)] = CHEM(p, r, q, s);

  struct ScfOut {
    double e;
    Eigen::MatrixXd c;
    Eigen::VectorXd eps;
  };
  auto run_scf = [&](const Eigen::MatrixXd& hh,
                     const std::vector<double>& gg) -> ScfOut {
    Eigen::MatrixXd cmo = Eigen::MatrixXd::Identity(nbf, nbf);
    Eigen::MatrixXd d_old = Eigen::MatrixXd::Zero(nbf, nbf);
    Eigen::VectorXd eps = Eigen::VectorXd::Zero(nbf);
    double e_old = 0.0;
    for (int iter = 0; iter < 200; ++iter) {
      Eigen::MatrixXd d = Eigen::MatrixXd::Zero(nbf, nbf);
      for (std::size_t i = 0; i < nocc; ++i)
        d += 2.0 * cmo.col(static_cast<int>(i)) *
             cmo.col(static_cast<int>(i)).transpose();
      const double dd_change = (d - d_old).cwiseAbs().maxCoeff();
      d_old = d;
      Eigen::MatrixXd f = hh;
      for (std::size_t p = 0; p < nbf; ++p)
        for (std::size_t q = 0; q < nbf; ++q) {
          double jv = 0.0, kv = 0.0;
          for (std::size_t r = 0; r < nbf; ++r)
            for (std::size_t s = 0; s < nbf; ++s) {
              const double dd = d(static_cast<int>(r), static_cast<int>(s));
              jv += gg[gpidx(p, r, q, s)] * dd;
              kv += gg[gpidx(p, r, s, q)] * dd;
            }
          f(static_cast<int>(p), static_cast<int>(q)) += jv - 0.5 * kv;
        }
      f = 0.5 * (f + f.transpose()).eval();
      double e = 0.0;
      for (std::size_t p = 0; p < nbf; ++p)
        for (std::size_t q = 0; q < nbf; ++q)
          e += 0.5 * d(static_cast<int>(q), static_cast<int>(p)) *
               (hh(static_cast<int>(p), static_cast<int>(q)) +
                f(static_cast<int>(p), static_cast<int>(q)));
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(f);
      cmo = es.eigenvectors();
      eps = es.eigenvalues();
      if (iter > 0 && std::abs(e - e_old) < 1e-12 && dd_change < 1e-11)
        return {e, cmo, eps};
      e_old = e;
    }
    return {e_old, cmo, eps};
  };

  // ---- Dressed two-body correction C2bar (paper Eq. 20-22) over OBS^4 ----
  auto c2idx = [&](std::size_t a, std::size_t b, std::size_t c, std::size_t d) {
    return ((a * nbf + b) * nbf + c) * nbf + d;
  };
  std::vector<double> c2(nbf * nbf * nbf * nbf, 0.0);

  Eigen::MatrixXd h_gc = w.c_obs_ri.transpose() * w.h_ri * w.c_cabs;
  auto cfg_ao = gem::four_center_coulomb(obs, ri, obs, ri);
  auto cfg_mo = gem::mo_transform_4index(
      cfg_ao.get(), nao, nri, nao, nri, w.c_obs, w.c_ri_mo, w.c_obs, w.c_ri_mo);
  Tensor4 CFG{cfg_mo.get(), nrimo, nbf, nrimo};  // <pr|g|MN>
  auto fgg_ao = gem::stg_geminal_eri(::libint2::Operator::stg_x_coulomb,
                                     in.gamma, obs, obs, obs, obs);
  auto fgg_mo = gem::mo_transform_4index(fgg_ao.get(), nao, nao, nao, nao,
                                         w.c_obs, w.c_val, w.c_obs, w.c_val);
  Tensor4 FGG{fgg_mo.get(), nv, nbf, nv};  // <pr|f12 g|ij>

  // Generalized V^{pr}_{ij} (Coulomb on pr, geminal on ij), SP-coupled on ij.
  auto vgen = [&](std::size_t p, std::size_t r, std::size_t i, std::size_t j) {
    double v = w.pref * FGG(p, i, r, j);
    for (std::size_t m = 0; m < nocc; ++m)
      for (std::size_t x = nbf; x < nrimo; ++x) {
        v -= CFG(p, m, r, x) * w.rg(i, j, m, x);
        v -= CFG(r, m, p, x) * w.rg(j, i, m, x);
      }
    for (std::size_t s = 0; s < nbf; ++s)
      for (std::size_t u = 0; u < nbf; ++u)
        v -= CFG(p, s, r, u) * w.rg(i, j, s, u);
    return v;
  };

  // T1 (h.G), T2 (V).
  for (std::size_t i = 0; i < nv; ++i)
    for (std::size_t j = 0; j < nv; ++j) {
      for (std::size_t p = 0; p < nbf; ++p)
        for (std::size_t r = 0; r < nbf; ++r) {
          double vsp = 0.375 * vgen(p, r, i, j) + 0.125 * vgen(p, r, j, i);
          c2[c2idx(p, r, nc + i, nc + j)] += 2.0 * vsp;
        }
      for (std::size_t p = 0; p < nbf; ++p)
        for (std::size_t b = nocc; b < nbf; ++b) {
          double hg = 0.0;
          for (std::size_t x = nbf; x < nrimo; ++x)
            hg += h_gc(static_cast<int>(p), static_cast<int>(x - nbf)) *
                  w.G(x, i, b, j);
          c2[c2idx(p, b, nc + i, nc + j)] += 4.0 * hg;
        }
    }
  // T3 (-2 eps_k X, canonical) and T4 (B).
  for (std::size_t k = 0; k < nv; ++k)
    for (std::size_t l = 0; l < nv; ++l)
      for (std::size_t i = 0; i < nv; ++i)
        for (std::size_t j = 0; j < nv; ++j)
          c2[c2idx(nc + k, nc + l, nc + i, nc + j)] +=
              -2.0 * w.eps(static_cast<int>(nc + k)) * vxb.x[P(k, l, i, j)] +
              vxb.b[P(k, l, i, j)];

  // ---- One-body correction C1bar (paper Eq. 17-19) over OBS^2 ----
  Eigen::MatrixXd c1 = Eigen::MatrixXd::Zero(nbf, nbf);

  // SP-coupled geminal G^{b a'}_{ij} (vir b, CABS a').
  auto Gvc = [&](std::size_t iv, std::size_t jv, std::size_t bv,
                 std::size_t A) { return w.G(nocc + bv, iv, nbf + A, jv); };
  // U^{prs}_{ijb} = sum_A <pr|g|(a' in s)>... = sum_A CFG[r][s][p][A]
  // G(j,i,b,A).
  auto uval = [&](std::size_t p, std::size_t r, std::size_t s, std::size_t iv,
                  std::size_t jv, std::size_t bv) {
    double u = 0.0;
    for (std::size_t A = 0; A < ncabs; ++A)
      u += CFG(r, s, p, nbf + A) * Gvc(jv, iv, bv, A);
    return u;
  };

  // c1'bar (Eq 18) and c2'bar (Eq 21): U-based terms.
  for (std::size_t av = 0; av < nvir; ++av)
    for (std::size_t q = 0; q < nbf; ++q) {
      double v = 0.0;
      for (std::size_t i = 0; i < nv; ++i)
        for (std::size_t j = 0; j < nv; ++j)
          for (std::size_t A = 0; A < ncabs; ++A)
            v += (4.0 * CFG(nc + j, q, nc + i, nbf + A) -
                  2.0 * CFG(nc + i, q, nc + j, nbf + A)) *
                 Gvc(j, i, av, A);
      c1(nocc + av, q) += v;
    }
  for (std::size_t av = 0; av < nvir; ++av)
    for (std::size_t j = 0; j < nv; ++j) {
      double v = 0.0;
      for (std::size_t i = 0; i < nv; ++i)
        for (std::size_t m = 0; m < nocc; ++m)
          v += 2.0 * (4.0 * uval(nc + i, m, m, i, j, av) -
                      2.0 * uval(m, nc + i, m, i, j, av)) -
               (4.0 * uval(nc + i, m, m, j, i, av) -
                2.0 * uval(m, nc + i, m, j, i, av));
      c1(nocc + av, nc + j) -= v;
    }
  for (std::size_t av = 0; av < nvir; ++av)
    for (std::size_t r = 0; r < nbf; ++r)
      for (std::size_t j = 0; j < nv; ++j)
        for (std::size_t s = 0; s < nbf; ++s) {
          double v = 0.0;
          for (std::size_t i = 0; i < nv; ++i)
            v += 2.0 * (4.0 * uval(nc + i, r, s, i, j, av) -
                        2.0 * uval(nc + i, r, s, j, i, av) -
                        2.0 * uval(r, nc + i, s, i, j, av));
          c2[c2idx(nocc + av, r, nc + j, s)] += v;
        }
  for (std::size_t j = 0; j < nv; ++j)
    for (std::size_t i = 0; i < nv; ++i)
      for (std::size_t av = 0; av < nvir; ++av)
        for (std::size_t p = 0; p < nbf; ++p) {
          double v = 0.0;
          for (std::size_t t = 0; t < nocc; ++t)
            v += 2.0 * (4.0 * uval(p, t, t, i, j, av) -
                        2.0 * uval(t, p, t, i, j, av));
          c2[c2idx(nc + j, nc + i, nocc + av, p)] += v;
        }
  for (std::size_t i = 0; i < nv; ++i)
    for (std::size_t s = 0; s < nbf; ++s)
      for (std::size_t p = 0; p < nbf; ++p)
        for (std::size_t av = 0; av < nvir; ++av) {
          double v = 0.0;
          for (std::size_t j = 0; j < nv; ++j)
            v += uval(p, nc + j, s, i, j, av);
          c2[c2idx(nc + i, s, p, nocc + av)] += -4.0 * v;
        }

  // S^{klb}_{ija} (Eq 27), canonical-Fock form, then c1''/c2'' (Eq 19, 22).
  auto Foo = [&](std::size_t mv, std::size_t iv) {
    return w.f_rimo(static_cast<int>(nc + mv), static_cast<int>(nc + iv));
  };
  auto Fuu = [&](std::size_t cv, std::size_t av) {
    return w.f_rimo(static_cast<int>(nocc + cv), static_cast<int>(nocc + av));
  };
  auto Fcc = [&](std::size_t A, std::size_t B) {
    return w.f_rimo(static_cast<int>(nbf + A), static_cast<int>(nbf + B));
  };
  auto bidx = [&](std::size_t i, std::size_t j, std::size_t a, std::size_t A) {
    return ((i * nv + j) * nvir + a) * ncabs + A;
  };
  std::vector<double> brk(nv * nv * nvir * ncabs, 0.0);
  for (std::size_t i = 0; i < nv; ++i)
    for (std::size_t j = 0; j < nv; ++j)
      for (std::size_t av = 0; av < nvir; ++av)
        for (std::size_t A = 0; A < ncabs; ++A) {
          double b1 = 0.0, b2 = 0.0, b3 = 0.0;
          for (std::size_t cv = 0; cv < nvir; ++cv)
            b1 += Gvc(j, i, cv, A) * Fuu(cv, av);
          for (std::size_t mv = 0; mv < nv; ++mv) {
            b2 += Gvc(j, mv, av, A) * Foo(mv, i);
            b3 += Gvc(mv, i, av, A) * Foo(mv, j);
          }
          brk[bidx(i, j, av, A)] = b1 - b2 - b3;
        }
  auto sidx = [&](std::size_t k, std::size_t l, std::size_t b, std::size_t i,
                  std::size_t j, std::size_t a) {
    return ((((k * nv + l) * nvir + b) * nv + i) * nv + j) * nvir + a;
  };
  std::vector<double> s1(nv * nv * nvir * nv * nv * nvir, 0.0);
  for (std::size_t k = 0; k < nv; ++k)
    for (std::size_t l = 0; l < nv; ++l)
      for (std::size_t bv = 0; bv < nvir; ++bv)
        for (std::size_t i = 0; i < nv; ++i)
          for (std::size_t j = 0; j < nv; ++j)
            for (std::size_t av = 0; av < nvir; ++av) {
              double v = 0.0;
              for (std::size_t A = 0; A < ncabs; ++A)
                v += Gvc(l, k, bv, A) * brk[bidx(i, j, av, A)];
              s1[sidx(k, l, bv, i, j, av)] = 0.25 * v;
            }
  std::vector<double> sm(s1.size(), 0.0);
  for (std::size_t k = 0; k < nv; ++k)
    for (std::size_t l = 0; l < nv; ++l)
      for (std::size_t bv = 0; bv < nvir; ++bv)
        for (std::size_t i = 0; i < nv; ++i)
          for (std::size_t j = 0; j < nv; ++j)
            for (std::size_t av = 0; av < nvir; ++av) {
              double v =
                  s1[sidx(k, l, bv, i, j, av)] + s1[sidx(i, j, av, k, l, bv)];
              for (std::size_t A = 0; A < ncabs; ++A)
                for (std::size_t B = 0; B < ncabs; ++B)
                  v += 0.5 * Gvc(l, k, bv, A) * Gvc(j, i, av, B) * Fcc(B, A);
              // Remove the spurious 1/2 prefactor in paper Eq. 27 (Comment on
              // J. Chem. Phys. 136, 084107; MPQC S_prefactor_error=false).
              sm[sidx(k, l, bv, i, j, av)] = 2.0 * v;
            }
  for (std::size_t av = 0; av < nvir; ++av)
    for (std::size_t bv = 0; bv < nvir; ++bv) {
      double v = 0.0;
      for (std::size_t i = 0; i < nv; ++i)
        for (std::size_t j = 0; j < nv; ++j)
          v +=
              2.0 * sm[sidx(i, j, bv, i, j, av)] - sm[sidx(j, i, bv, i, j, av)];
      c1(nocc + av, nocc + bv) += v;
    }
  for (std::size_t l = 0; l < nv; ++l)
    for (std::size_t av = 0; av < nvir; ++av)
      for (std::size_t bv = 0; bv < nvir; ++bv)
        for (std::size_t j = 0; j < nv; ++j) {
          double t1 = 0.0, t3 = 0.0;
          for (std::size_t i = 0; i < nv; ++i) {
            t1 += sm[sidx(i, l, bv, i, j, av)];
            t3 += sm[sidx(l, i, bv, i, j, av)];
          }
          c2[c2idx(nc + l, nocc + av, nocc + bv, nc + j)] +=
              4.0 * t1 - 2.0 * t3;
        }
  for (std::size_t l = 0; l < nv; ++l)
    for (std::size_t av = 0; av < nvir; ++av)
      for (std::size_t bv = 0; bv < nvir; ++bv)
        for (std::size_t i = 0; i < nv; ++i) {
          double t2 = 0.0;
          for (std::size_t j = 0; j < nv; ++j)
            t2 += sm[sidx(j, l, bv, i, j, av)];
          c2[c2idx(nc + l, nocc + av, nocc + bv, nc + i)] += -2.0 * t2;
        }
  for (std::size_t k = 0; k < nv; ++k)
    for (std::size_t av = 0; av < nvir; ++av)
      for (std::size_t i = 0; i < nv; ++i)
        for (std::size_t bv = 0; bv < nvir; ++bv) {
          double t4 = 0.0;
          for (std::size_t j = 0; j < nv; ++j)
            t4 += sm[sidx(k, j, bv, i, j, av)];
          c2[c2idx(nc + k, nocc + av, nc + i, nocc + bv)] += -2.0 * t4;
        }

  // Dressed one- and two-body Hamiltonian (paper Eq. 15-16).
  Eigen::MatrixXd hbar = hmo + 0.5 * (c1 + c1.transpose());
  std::vector<double> gbar = gp;
  for (std::size_t p = 0; p < nbf; ++p)
    for (std::size_t r = 0; r < nbf; ++r)
      for (std::size_t q = 0; q < nbf; ++q)
        for (std::size_t s = 0; s < nbf; ++s)
          gbar[gpidx(p, r, q, s)] +=
              0.25 * (c2[c2idx(p, r, q, s)] + c2[c2idx(r, p, s, q)] +
                      c2[c2idx(q, s, p, r)] + c2[c2idx(s, q, r, p)]);

  const ScfOut hf = run_scf(hmo, gp);
  const ScfOut f12 = run_scf(hbar, gbar);
  return {hf.e, f12.e, std::move(gbar), f12.c, f12.eps, nbf, nocc, nc};
}

}  // namespace

double f12_hf_scf_energy(const F12HartreeFockInput& in) {
  QDK_LOG_TRACE_ENTERING();
  const DressedResult r = run_f12_hf(in);
  return r.e_f12hf - r.e_hf;
}

double f12_mp2_energy(const F12HartreeFockInput& in) {
  QDK_LOG_TRACE_ENTERING();
  const DressedResult r = run_f12_hf(in);
  const std::size_t nbf = r.nbf, nocc = r.nocc, nc = r.ncore;
  const std::size_t nvo = nocc - nc, nvir = nbf - nocc;
  const Eigen::MatrixXd& c = r.c_relaxed;
  const Eigen::VectorXd& eps = r.eps_relaxed;
  const std::vector<double>& g = r.gbar;
  auto gidx = [&](std::size_t p, std::size_t q, std::size_t rr, std::size_t s) {
    return ((p * nbf + q) * nbf + rr) * nbf + s;
  };

  // Transform the dressed <pq|rs> from the original to the F12-HF-relaxed MO
  // basis, restricted to the (occ occ | vir vir) block needed for MP2.
  std::vector<double> t1(nvo * nbf * nbf * nbf, 0.0);
  for (std::size_t i = 0; i < nvo; ++i)
    for (std::size_t q = 0; q < nbf; ++q)
      for (std::size_t rr = 0; rr < nbf; ++rr)
        for (std::size_t s = 0; s < nbf; ++s) {
          double v = 0.0;
          for (std::size_t p = 0; p < nbf; ++p)
            v += c(static_cast<int>(p), static_cast<int>(nc + i)) *
                 g[gidx(p, q, rr, s)];
          t1[((i * nbf + q) * nbf + rr) * nbf + s] = v;
        }
  std::vector<double> t2(nvo * nvo * nbf * nbf, 0.0);
  for (std::size_t i = 0; i < nvo; ++i)
    for (std::size_t j = 0; j < nvo; ++j)
      for (std::size_t rr = 0; rr < nbf; ++rr)
        for (std::size_t s = 0; s < nbf; ++s) {
          double v = 0.0;
          for (std::size_t q = 0; q < nbf; ++q)
            v += c(static_cast<int>(q), static_cast<int>(nc + j)) *
                 t1[((i * nbf + q) * nbf + rr) * nbf + s];
          t2[((i * nvo + j) * nbf + rr) * nbf + s] = v;
        }
  std::vector<double> t3(nvo * nvo * nvir * nbf, 0.0);
  for (std::size_t i = 0; i < nvo; ++i)
    for (std::size_t j = 0; j < nvo; ++j)
      for (std::size_t a = 0; a < nvir; ++a)
        for (std::size_t s = 0; s < nbf; ++s) {
          double v = 0.0;
          for (std::size_t rr = 0; rr < nbf; ++rr)
            v += c(static_cast<int>(rr), static_cast<int>(nocc + a)) *
                 t2[((i * nvo + j) * nbf + rr) * nbf + s];
          t3[((i * nvo + j) * nvir + a) * nbf + s] = v;
        }
  std::vector<double> t4(nvo * nvo * nvir * nvir, 0.0);
  for (std::size_t i = 0; i < nvo; ++i)
    for (std::size_t j = 0; j < nvo; ++j)
      for (std::size_t a = 0; a < nvir; ++a)
        for (std::size_t b = 0; b < nvir; ++b) {
          double v = 0.0;
          for (std::size_t s = 0; s < nbf; ++s)
            v += c(static_cast<int>(s), static_cast<int>(nocc + b)) *
                 t3[((i * nvo + j) * nvir + a) * nbf + s];
          t4[((i * nvo + j) * nvir + a) * nvir + b] = v;
        }

  double energy = 0.0;
  for (std::size_t i = 0; i < nvo; ++i)
    for (std::size_t j = 0; j < nvo; ++j)
      for (std::size_t a = 0; a < nvir; ++a)
        for (std::size_t b = 0; b < nvir; ++b) {
          const double iajb = t4[((i * nvo + j) * nvir + a) * nvir + b];
          const double ibja = t4[((i * nvo + j) * nvir + b) * nvir + a];
          energy +=
              iajb * (2.0 * iajb - ibja) /
              (eps(static_cast<int>(nc + i)) + eps(static_cast<int>(nc + j)) -
               eps(static_cast<int>(nocc + a)) -
               eps(static_cast<int>(nocc + b)));
        }
  // Total F12-MP2 correlation relative to the bare HF reference: the F12-HF
  // relaxation plus the residual MP2 over the dressed Hamiltonian.
  return (r.e_f12hf - r.e_hf) + energy;
}

double mp2_f12_correction(const F12HartreeFockInput& in) {
  QDK_LOG_TRACE_ENTERING();
  Workspace w = build_workspace(in);
  VXB vxb = compute_vxb(w);
  const std::size_t nv = w.n_val, nocc = w.n_occ, nc = w.n_core, nbf = w.n_mo;
  const std::size_t nvir = w.n_vir, ncabs = w.n_cabs, nao = w.n_ao_obs;
  auto P = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    return vidx(nv, i, j, k, l);
  };
  // Orbital energies self-consistent with the integrals (diagonal Fock).
  auto eocc = [&](std::size_t i) {
    return w.f_rimo(static_cast<int>(nc + i), static_cast<int>(nc + i));
  };
  auto evir = [&](std::size_t a) {
    return w.f_rimo(static_cast<int>(nocc + a), static_cast<int>(nocc + a));
  };

  // No-coupling part V[2Cbar] - X[CCbar] + B[CCbar] (= first-order F12-HF).
  double e_nc = 0.0;
  for (std::size_t i = 0; i < nv; ++i)
    for (std::size_t j = 0; j < nv; ++j) {
      const double eij = eocc(i) + eocc(j);
      const double vt = 2 * vxb.v[P(i, j, i, j)] - vxb.v[P(i, j, j, i)];
      const double xt = 2 * vxb.x[P(i, j, i, j)] - vxb.x[P(i, j, j, i)];
      const double bt = 2 * vxb.b[P(i, j, i, j)] - vxb.b[P(i, j, j, i)];
      e_nc += 2 * vt - eij * xt + bt;
    }

  // Geminal-conventional coupling C_ij^{ab} (raw geminal, vir-CABS Fock).
  auto Cidx = [&](std::size_t i, std::size_t j, std::size_t a, std::size_t b) {
    return ((i * nv + j) * nvir + a) * nvir + b;
  };
  std::vector<double> cc(nv * nv * nvir * nvir, 0.0);
  for (std::size_t i = 0; i < nv; ++i)
    for (std::size_t j = 0; j < nv; ++j)
      for (std::size_t a = 0; a < nvir; ++a)
        for (std::size_t b = 0; b < nvir; ++b) {
          double c = 0.0;
          for (std::size_t x = 0; x < ncabs; ++x)
            c += w.f_rimo(static_cast<int>(nocc + a),
                          static_cast<int>(nbf + x)) *
                     w.rg(i, j, nbf + x, nocc + b) +
                 w.f_rimo(static_cast<int>(nocc + b),
                          static_cast<int>(nbf + x)) *
                     w.rg(i, j, nocc + a, nbf + x);
          cc[Cidx(i, j, a, b)] = c;
        }

  // Conventional <ij|ab> over OBS valence-occ x virtual (restricted transform).
  Eigen::MatrixXd c_vir = w.c_obs.rightCols(static_cast<int>(nvir));
  auto cc_ao = gem::four_center_coulomb(in.obs, in.obs, in.obs, in.obs);
  auto ovov = gem::mo_transform_4index(cc_ao.get(), nao, nao, nao, nao, w.c_val,
                                       c_vir, w.c_val, c_vir);
  Tensor4 OVOV{ovov.get(), nvir, nv, nvir};  // (i a | j b) chemist

  // Conventional amplitudes t2 and Cbar = C / (e_i + e_j - e_a - e_b).
  std::vector<double> t2(cc.size(), 0.0), cbar(cc.size(), 0.0);
  for (std::size_t i = 0; i < nv; ++i)
    for (std::size_t j = 0; j < nv; ++j)
      for (std::size_t a = 0; a < nvir; ++a)
        for (std::size_t b = 0; b < nvir; ++b) {
          const double den = eocc(i) + eocc(j) - evir(a) - evir(b);
          t2[Cidx(i, j, a, b)] = OVOV(i, a, j, b) / den;  // <ij|ab> / den
          cbar[Cidx(i, j, a, b)] = cc[Cidx(i, j, a, b)] / den;
        }

  // Coupling energy: E_CT (C with conventional doubles) + E_CC (C squared).
  constexpr double cbar_d = 5.0 / 8.0, cbar_x = -1.0 / 8.0;
  constexpr double ccbar_d = 14.0 / 64.0, ccbar_x = 2.0 / 64.0;
  double e_ct = 0.0, e_cc = 0.0;
  for (std::size_t i = 0; i < nv; ++i)
    for (std::size_t j = 0; j < nv; ++j) {
      double ct_d = 0, ct_x = 0, cc_d = 0, cc_x = 0;
      for (std::size_t a = 0; a < nvir; ++a)
        for (std::size_t b = 0; b < nvir; ++b) {
          const double cij = cc[Cidx(i, j, a, b)];
          ct_d += cij * t2[Cidx(i, j, a, b)];
          ct_x += cij * t2[Cidx(j, i, a, b)];
          cc_d += cij * cbar[Cidx(i, j, a, b)];
          cc_x += cij * cbar[Cidx(j, i, a, b)];
        }
      e_ct += 2 * (cbar_d * ct_d + cbar_x * ct_x);
      e_cc += ccbar_d * cc_d + ccbar_x * cc_x;
    }
  return e_nc + e_ct + e_cc;
}

}  // namespace qdk::chemistry::algorithms::microsoft::ctf12

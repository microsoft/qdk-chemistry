// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/scf/util/libint2_util.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace qdk::chemistry::scf::libint2_util {

libint2::Shell convert_to_libint_shell(const Shell& o, bool pure) {
  libint2::Shell sh;
  sh.O = o.O;
  sh.contr.resize(1);
  sh.contr[0].l = o.angular_momentum;
  sh.contr[0].pure = (pure && o.angular_momentum >= 2);

  for (uint64_t i = 0; i < o.contraction; i++) {
    sh.alpha.push_back(o.exponents[i]);
    sh.contr[0].coeff.push_back(o.coefficients[i]);
  }
  return libint2::Shell(sh.alpha, sh.contr, sh.O, false);
}

libint2::BasisSet convert_to_libint_basisset(const BasisSet& o) {
  std::vector<libint2::Shell> shells;
  for (auto& sh : o.shells) {
    shells.push_back(convert_to_libint_shell(sh, o.pure));
  }
  return libint2::BasisSet(shells);
}

std::unique_ptr<double[]> debug_eri(BasisMode basis_mode,
                                    const libint2::BasisSet& obs, double omega,
                                    size_t i_lo, size_t i_hi) {
  const size_t num_atomic_orbitals = obs.nbf();
  const size_t num_atomic_orbitals2 = num_atomic_orbitals * num_atomic_orbitals;
  const size_t num_atomic_orbitals3 =
      num_atomic_orbitals2 * num_atomic_orbitals;
  const size_t eri_sz = num_atomic_orbitals3 * (i_hi - i_lo);

  auto h_eri = std::make_unique<double[]>(eri_sz);
  auto* h_eri_ptr = h_eri.get();
  std::fill_n(h_eri_ptr, eri_sz, 0.0);

  const size_t nshells = obs.size();
  auto shell2bf = obs.shell2bf();

  // Find i-shells that need to be computed
  size_t ish_st = 0;
  for (auto i = 0; i < nshells; ++i) {
    if (shell2bf[i] <= i_lo) {
      ish_st = i;
    }
  }

  size_t ish_en = nshells;
  for (auto i = 0; i < nshells; ++i) {
    if (shell2bf[i] > i_hi) {
      ish_en = i;
      break;
    }
  }

  bool is_erf = std::abs(omega) > 1e-12;

  libint2::Engine engine;

  if (is_erf) {
    engine = libint2::Engine(libint2::Operator::erf_coulomb, obs.max_nprim(),
                             obs.max_l(), 0);
    engine.set_params(omega);
  } else {
    engine = libint2::Engine(libint2::Operator::coulomb, obs.max_nprim(),
                             obs.max_l(), 0);
  }

  for (size_t i = ish_st; i < ish_en; ++i)
    for (size_t j = 0; j < nshells; ++j)
      for (size_t k = 0; k < nshells; ++k)
        for (size_t l = 0; l < nshells; ++l) {
          if (is_erf)
            engine.compute2<libint2::Operator::erf_coulomb,
                            libint2::BraKet::xx_xx, 0>(obs[i], obs[j], obs[k],
                                                       obs[l]);
          else
            engine.compute2<libint2::Operator::coulomb, libint2::BraKet::xx_xx,
                            0>(obs[i], obs[j], obs[k], obs[l]);

          auto data = engine.results()[0];
          if (data) {
            size_t i_st = shell2bf[i];
            size_t j_st = shell2bf[j];
            size_t k_st = shell2bf[k];
            size_t l_st = shell2bf[l];

            const size_t ni = obs[i].size();
            const size_t nj = obs[j].size();
            const size_t nk = obs[k].size();
            const size_t nl = obs[l].size();

            auto* h_eri_loc = h_eri_ptr + (i_st - i_lo) * num_atomic_orbitals3 +
                              j_st * num_atomic_orbitals2 +
                              k_st * num_atomic_orbitals + l_st;
            for (size_t ii = 0, int_ijkl = 0; ii < ni; ++ii)
              for (size_t jj = 0; jj < nj; ++jj)
                for (size_t kk = 0; kk < nk; ++kk)
                  for (size_t ll = 0; ll < nl; ++ll, int_ijkl++) {
                    if (i_st + ii >= i_lo and i_st + ii < i_hi) {
                      h_eri_loc[ii * num_atomic_orbitals3 +
                                jj * num_atomic_orbitals2 +
                                kk * num_atomic_orbitals + ll] = data[int_ijkl];
                    }
                  }
          }
        }

  return h_eri;
}

std::unique_ptr<double[]> opt_eri(BasisMode basis_mode,
                                  const libint2::BasisSet& obs, double omega,
                                  size_t i_lo, size_t i_hi) {
  const size_t num_atomic_orbitals = obs.nbf();
  const size_t num_atomic_orbitals2 = num_atomic_orbitals * num_atomic_orbitals;
  const size_t num_atomic_orbitals3 =
      num_atomic_orbitals2 * num_atomic_orbitals;
  const size_t eri_sz = num_atomic_orbitals3 * (i_hi - i_lo);

  auto h_eri = std::make_unique<double[]>(eri_sz);
  auto* h_eri_ptr = h_eri.get();
  std::fill_n(h_eri_ptr, eri_sz, 0.0);

  const size_t nshells = obs.size();
  auto shell2bf = obs.shell2bf();

  bool is_erf = std::abs(omega) > 1e-12;

  libint2::Engine base_engine;

  if (is_erf) {
    base_engine = libint2::Engine(libint2::Operator::erf_coulomb,
                                  obs.max_nprim(), obs.max_l(), 0);
    base_engine.set_params(omega);
  } else {
    base_engine = libint2::Engine(libint2::Operator::coulomb, obs.max_nprim(),
                                  obs.max_l(), 0);
  }

#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  std::vector<libint2::Engine> engines(nthreads, base_engine);

  auto range_intersect = [](int x_st, int x_en, int y_st, int y_en) {
    return x_st <= (y_en - 1) and y_st <= (x_en - 1);
  };

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    auto& engine = engines[thread_id];
    auto& buf = engine.results();
    for (size_t i = 0, ijkl = 0; i < nshells; ++i) {
      const size_t ni = obs[i].size();
      const size_t i_st = shell2bf[i];
      for (size_t j = 0; j <= i; ++j) {
        const size_t nj = obs[j].size();
        const size_t j_st = shell2bf[j];
        for (size_t k = 0; k <= i; ++k) {
          const size_t nk = obs[k].size();
          const size_t k_st = shell2bf[k];
          const size_t l_max = (i == k) ? j : k;
          for (size_t l = 0; l <= l_max; ++l) {
            const size_t nl = obs[l].size();
            const size_t l_st = shell2bf[l];

            const bool is_loc = range_intersect(i_st, i_st + ni, i_lo, i_hi) or
                                range_intersect(j_st, j_st + nj, i_lo, i_hi) or
                                range_intersect(k_st, k_st + nk, i_lo, i_hi) or
                                range_intersect(l_st, l_st + nl, i_lo, i_hi);

            if (!is_loc) continue;
            ijkl++;  // Only increment the counter for local integrals

            if (ijkl % nthreads == thread_id) {
              if (is_erf)
                engine.compute2<libint2::Operator::erf_coulomb,
                                libint2::BraKet::xx_xx, 0>(obs[i], obs[j],
                                                           obs[k], obs[l]);
              else
                engine.compute2<libint2::Operator::coulomb,
                                libint2::BraKet::xx_xx, 0>(obs[i], obs[j],
                                                           obs[k], obs[l]);

              auto data = buf[0];
              if (data) {
                auto* h_eri_ijkl = h_eri_ptr +
                                   (i_st - i_lo) * num_atomic_orbitals3 +
                                   j_st * num_atomic_orbitals2 +
                                   k_st * num_atomic_orbitals + l_st;
                auto* h_eri_ijlk = h_eri_ptr +
                                   (i_st - i_lo) * num_atomic_orbitals3 +
                                   j_st * num_atomic_orbitals2 +
                                   l_st * num_atomic_orbitals + k_st;
                auto* h_eri_jikl = h_eri_ptr +
                                   (j_st - i_lo) * num_atomic_orbitals3 +
                                   i_st * num_atomic_orbitals2 +
                                   k_st * num_atomic_orbitals + l_st;
                auto* h_eri_jilk = h_eri_ptr +
                                   (j_st - i_lo) * num_atomic_orbitals3 +
                                   i_st * num_atomic_orbitals2 +
                                   l_st * num_atomic_orbitals + k_st;

                auto* h_eri_klij = h_eri_ptr +
                                   (k_st - i_lo) * num_atomic_orbitals3 +
                                   l_st * num_atomic_orbitals2 +
                                   i_st * num_atomic_orbitals + j_st;
                auto* h_eri_klji = h_eri_ptr +
                                   (k_st - i_lo) * num_atomic_orbitals3 +
                                   l_st * num_atomic_orbitals2 +
                                   j_st * num_atomic_orbitals + i_st;
                auto* h_eri_lkij = h_eri_ptr +
                                   (l_st - i_lo) * num_atomic_orbitals3 +
                                   k_st * num_atomic_orbitals2 +
                                   i_st * num_atomic_orbitals + j_st;
                auto* h_eri_lkji = h_eri_ptr +
                                   (l_st - i_lo) * num_atomic_orbitals3 +
                                   k_st * num_atomic_orbitals2 +
                                   j_st * num_atomic_orbitals + i_st;

                for (size_t ii = 0; ii < ni; ++ii)
                  for (size_t jj = 0; jj < nj; ++jj)
                    for (size_t kk = 0; kk < nk; ++kk)
                      for (size_t ll = 0; ll < nl; ++ll) {
                        const auto integral = *data++;

                        if (i_st + ii >= i_lo and i_st + ii < i_hi) {
                          h_eri_ijkl[ii * num_atomic_orbitals3 +
                                     jj * num_atomic_orbitals2 +
                                     kk * num_atomic_orbitals + ll] =
                              integral;  // (ij|kl)
                          h_eri_ijlk[ii * num_atomic_orbitals3 +
                                     jj * num_atomic_orbitals2 +
                                     ll * num_atomic_orbitals + kk] =
                              integral;  // (ij|lk)
                        }

                        if (j_st + jj >= i_lo and j_st + jj < i_hi) {
                          h_eri_jikl[jj * num_atomic_orbitals3 +
                                     ii * num_atomic_orbitals2 +
                                     kk * num_atomic_orbitals + ll] =
                              integral;  // (ji|kl)
                          h_eri_jilk[jj * num_atomic_orbitals3 +
                                     ii * num_atomic_orbitals2 +
                                     ll * num_atomic_orbitals + kk] =
                              integral;  // (ji|lk)
                        }

                        if (k_st + kk >= i_lo and k_st + kk < i_hi) {
                          h_eri_klij[kk * num_atomic_orbitals3 +
                                     ll * num_atomic_orbitals2 +
                                     ii * num_atomic_orbitals + jj] =
                              integral;  // (kl|ij)
                          h_eri_klji[kk * num_atomic_orbitals3 +
                                     ll * num_atomic_orbitals2 +
                                     jj * num_atomic_orbitals + ii] =
                              integral;  // (kl|ji)
                        }

                        if (l_st + ll >= i_lo and l_st + ll < i_hi) {
                          h_eri_lkij[ll * num_atomic_orbitals3 +
                                     kk * num_atomic_orbitals2 +
                                     ii * num_atomic_orbitals + jj] =
                              integral;  // (lk|ij)
                          h_eri_lkji[ll * num_atomic_orbitals3 +
                                     kk * num_atomic_orbitals2 +
                                     jj * num_atomic_orbitals + ii] =
                              integral;  // (lk|ji)
                        }
                      }
              }
            }
          }
        }
      }
    }

  }  // OpenMP Context

  return h_eri;
}

std::unique_ptr<double[]> eri_df(BasisMode basis_mode,
                                 const libint2::BasisSet& obs,
                                 const libint2::BasisSet& abs, size_t i_lo,
                                 size_t i_hi) {
  const size_t num_atomic_orbitals = obs.nbf();
  const size_t naux = abs.nbf();
  const size_t num_atomic_orbitals2 = num_atomic_orbitals * num_atomic_orbitals;
  const size_t eri_sz = num_atomic_orbitals2 * (i_hi - i_lo);

  auto h_eri = std::make_unique<double[]>(eri_sz);
  auto* h_eri_ptr = h_eri.get();
  std::fill_n(h_eri_ptr, eri_sz, 0.0);

  const size_t nshells_obs = obs.size();
  const size_t nshells_abs = abs.size();
  auto shell2bf_obs = obs.shell2bf();
  auto shell2bf_abs = abs.shell2bf();

  // Find aux i-shells that need to be computed
  size_t ish_st = 0;
  for (auto i = 0; i < nshells_abs; ++i) {
    if (shell2bf_abs[i] <= i_lo) {
      ish_st = i;
    }
  }

  size_t ish_en = nshells_abs;
  for (auto i = 0; i < nshells_abs; ++i) {
    if (shell2bf_abs[i] > i_hi) {
      ish_en = i;
      break;
    }
  }

  libint2::Engine base_engine(libint2::Operator::coulomb,
                              std::max(abs.max_nprim(), obs.max_nprim()),
                              std::max(abs.max_l(), obs.max_l()), 0);
  base_engine.set(libint2::BraKet::xs_xx);

#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  std::vector<libint2::Engine> engines(nthreads, base_engine);
  const auto& unitshell = libint2::Shell::unit();

#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    auto& engine = engines[thread_id];
    auto& buf = engine.results();

    for (size_t i = ish_st, ipq = 0; i < ish_en; ++i) {
      const size_t i_st = shell2bf_abs[i];
      const size_t ni = abs[i].size();
      for (size_t p = 0; p < nshells_obs; ++p) {
        const size_t p_st = shell2bf_obs[p];
        const size_t np = obs[p].size();
        for (size_t q = p; q < nshells_obs; ++q, ipq++) {
          if (ipq % nthreads != thread_id) continue;

          const size_t q_st = shell2bf_obs[q];
          const size_t nq = obs[q].size();
          engine
              .compute2<libint2::Operator::coulomb, libint2::BraKet::xs_xx, 0>(
                  abs[i], unitshell, obs[p], obs[q]);
          auto data = engine.results()[0];
          if (data) {
            auto* h_eri_loc_pq = h_eri_ptr +
                                 (i_st - i_lo) * num_atomic_orbitals2 +
                                 p_st * num_atomic_orbitals + q_st;
            auto* h_eri_loc_qp = h_eri_ptr +
                                 (i_st - i_lo) * num_atomic_orbitals2 +
                                 q_st * num_atomic_orbitals + p_st;
            for (size_t ii = 0, int_ipq = 0; ii < ni; ++ii)
              for (size_t pp = 0; pp < np; ++pp)
                for (size_t qq = 0; qq < nq; ++qq, int_ipq++) {
                  if (i_st + ii >= i_lo and i_st + ii < i_hi) {
                    h_eri_loc_pq[ii * num_atomic_orbitals2 +
                                 pp * num_atomic_orbitals + qq] = data[int_ipq];
                    h_eri_loc_qp[ii * num_atomic_orbitals2 +
                                 qq * num_atomic_orbitals + pp] = data[int_ipq];
                  }
                }
          }
        }
      }
    }  // Loop over local AUX shells

  }  // OpenMP Context

  return h_eri;
}

std::unique_ptr<double[]> metric_df(BasisMode basis_mode,
                                    const libint2::BasisSet& abs) {
  const size_t naux = abs.nbf();
  const size_t met_sz = naux * naux;

  auto h_metric = std::make_unique<double[]>(met_sz);
  auto* h_metric_ptr = h_metric.get();
  std::fill_n(h_metric_ptr, met_sz, 0.0);

  const size_t nshells = abs.size();
  auto shell2bf = abs.shell2bf();

  libint2::Engine base_engine(libint2::Operator::coulomb, abs.max_nprim(),
                              abs.max_l(), 0);
  base_engine.set(libint2::BraKet::xs_xs);

#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  std::vector<libint2::Engine> engines(nthreads, base_engine);
  const auto& unitshell = libint2::Shell::unit();
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    auto& engine = engines[thread_id];
    auto& buf = engine.results();

    for (auto i = 0, ij = 0; i < nshells; ++i) {
      const size_t i_st = shell2bf[i];
      const size_t ni = abs[i].size();
      for (auto j = i; j < nshells; ++j, ++ij) {
        if (ij % nthreads != thread_id) continue;

        const size_t j_st = shell2bf[j];
        const size_t nj = abs[j].size();

        engine.compute2<libint2::Operator::coulomb, libint2::BraKet::xs_xs, 0>(
            abs[i], unitshell, abs[j], unitshell);
        auto data = engine.results()[0];
        if (data) {
          auto* h_metric_loc_ij = h_metric_ptr + i_st * naux + j_st;
          auto* h_metric_loc_ji = h_metric_ptr + j_st * naux + i_st;
          for (size_t ii = 0, int_ij = 0; ii < ni; ++ii)
            for (size_t jj = 0; jj < nj; ++jj, int_ij++) {
              h_metric_loc_ij[ii * naux + jj] = data[int_ij];
              h_metric_loc_ji[jj * naux + ii] = data[int_ij];
            }
        }
      }
    }

  }  // OpenMP Context

  return h_metric;
}

void eri_df_grad(double* dJ, const double* P, const double* X,
                 BasisMode basis_mode, const libint2::BasisSet& obs,
                 const libint2::BasisSet& abs,
                 const std::vector<int>& obs_sh2atom,
                 const std::vector<int>& abs_sh2atom, size_t n_atoms,
                 ParallelConfig mpi) {
  const size_t num_atomic_orbitals = obs.nbf();
  const size_t naux = abs.nbf();
  const size_t num_atomic_orbitals2 = num_atomic_orbitals * num_atomic_orbitals;
  const size_t nshells_obs = obs.size();
  const size_t nshells_abs = abs.size();
  auto shell2bf_obs = obs.shell2bf();
  auto shell2bf_abs = abs.shell2bf();

  libint2::Engine base_engine(libint2::Operator::coulomb,
                              std::max(abs.max_nprim(), obs.max_nprim()),
                              std::max(abs.max_l(), obs.max_l()), 1);
  base_engine.set(libint2::BraKet::xs_xx);

  const auto& unitshell = libint2::Shell::unit();
#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  int total_threads = mpi.world_size * nthreads;
  std::vector<libint2::Engine> engines(nthreads, base_engine);
#ifdef _OPENMP
#pragma omp parallel reduction(+ : dJ[ : 3 * n_atoms])
#endif
  {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    int world_thread_id = mpi.world_rank * nthreads + thread_id;
    auto& engine = engines[thread_id];

    size_t shell_atoms[3];

    for (size_t i = 0, ipq = 0; i < nshells_abs; ++i) {
      const size_t i_st = shell2bf_abs[i];
      const size_t ni = abs[i].size();
      shell_atoms[0] = abs_sh2atom[i];
      for (size_t p = 0; p < nshells_obs; ++p) {
        const size_t p_st = shell2bf_obs[p];
        const size_t np = obs[p].size();
        shell_atoms[1] = obs_sh2atom[p];
        for (size_t q = p; q < nshells_obs; ++q, ipq++) {
          if (ipq % total_threads != world_thread_id) continue;

          const size_t q_st = shell2bf_obs[q];
          const size_t nq = obs[q].size();
          shell_atoms[2] = obs_sh2atom[q];
          engine
              .compute2<libint2::Operator::coulomb, libint2::BraKet::xs_xx, 1>(
                  abs[i], unitshell, obs[p], obs[q]);
          auto& buf = engine.results();
          if (buf[0] == nullptr)
            continue;  // if all integrals screened out, skip to next quartet

          // Form part 1 of dJ^x: \Sum_{pqI} P(p,q) * (pq|I)^x * Y(I)
          for (auto d = 0; d != 9; ++d) {
            const int a = d / 3;
            const int xyz = d % 3;
            auto coord = shell_atoms[a] + xyz * n_atoms;
            double dJ_coord = 0.0;
            auto shset = buf[d];
            for (size_t ii = i_st, int_ipq = 0; ii < i_st + ni; ++ii)
              for (size_t pp = p_st; pp < p_st + np; ++pp)
                for (size_t qq = q_st; qq < q_st + nq; ++qq, int_ipq++)
                  dJ_coord +=
                      P[pp * num_atomic_orbitals + qq] * shset[int_ipq] * X[ii];
            if (q > p) dJ_coord *= 2.0;  // use symmetry of (I|pq) D(p,q)
            dJ[coord] += dJ_coord;
          }
        }
      }
    }  // Loop over local AUX shells

  }  // OpenMP Context
}

void metric_df_grad(double* dJ, const double* X, BasisMode basis_mode,
                    const libint2::BasisSet& abs,
                    const std::vector<int>& abs_sh2atom, size_t n_atoms,
                    ParallelConfig mpi) {
  const size_t naux = abs.nbf();
  const size_t nshells = abs.size();
  auto shell2bf = abs.shell2bf();

  libint2::Engine base_engine(libint2::Operator::coulomb, abs.max_nprim(),
                              abs.max_l(), 1);
  base_engine.set(libint2::BraKet::xs_xs);

#ifdef _OPENMP
  int nthreads = omp_get_max_threads();
#else
  int nthreads = 1;
#endif
  int total_threads = mpi.world_size * nthreads;
  std::vector<libint2::Engine> engines(nthreads, base_engine);
  const auto& unitshell = libint2::Shell::unit();
#ifdef _OPENMP
#pragma omp parallel reduction(+ : dJ[ : 3 * n_atoms])
#endif
  {
#ifdef _OPENMP
    int thread_id = omp_get_thread_num();
#else
    int thread_id = 0;
#endif
    int world_thread_id = mpi.world_rank * nthreads + thread_id;
    auto& engine = engines[thread_id];
    auto& buf = engine.results();

    for (size_t i = 0, ij = 0; i < nshells; ++i) {
      const size_t i_st = shell2bf[i];
      const size_t ni = abs[i].size();
      for (auto j = i; j < nshells; ++j, ++ij) {
        if (ij % total_threads != world_thread_id) continue;

        const size_t j_st = shell2bf[j];
        const size_t nj = abs[j].size();

        engine.compute2<libint2::Operator::coulomb, libint2::BraKet::xs_xs, 1>(
            abs[i], unitshell, abs[j], unitshell);
        if (buf[0] == nullptr)
          continue;  // if all integrals screened out, skip to next quartet
        const auto fact = -0.5;
        // Form part 2 of dJ(p,q): -1/2 \Sum_{IJ}(I|J)^x * Y(I) * Y(J)
        for (auto d = 0; d != 6; ++d) {
          const int atom_idx = d < 3 ? abs_sh2atom[i] : abs_sh2atom[j];
          const int xyz = d % 3;
          auto coord = atom_idx + xyz * n_atoms;
          double dJ_coord = 0.0;
          auto shset = buf[d];
          for (size_t ii = i_st, int_ij = 0; ii < i_st + ni; ++ii)
            for (size_t jj = j_st; jj < j_st + nj; ++jj, int_ij++)
              dJ_coord += X[jj] * shset[int_ij] * X[ii];
          if (j > i) dJ_coord *= 2.0;  // use symmetry of (I|J)
          dJ[coord] += fact * dJ_coord;
        }
      }
    }

  }  // OpenMP Context
}

}  // namespace qdk::chemistry::scf::libint2_util

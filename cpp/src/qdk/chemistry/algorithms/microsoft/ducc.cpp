// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ducc.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <memory>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/amplitude_container.hpp>
#include <qdk/chemistry/utils/eri_notation.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

namespace {

// real (VectorXd) alternative of a container VectorVariant (CCSD amps are real)
template <typename Variant>
const Eigen::VectorXd& real_vec(const Variant& v) {
  return std::get<Eigen::VectorXd>(v);
}

}  // namespace

SpinOrbitalData extract_spinorbital_data(
    const data::Hamiltonian& hamiltonian,
    const data::Wavefunction& wavefunction) {
  const std::size_t nmo =
      hamiltonian.get_orbitals()->get_num_molecular_orbitals();
  const auto n_elec = wavefunction.get_total_num_electrons();
  const std::size_t nocc_a = n_elec.first, nocc_b = n_elec.second;
  const std::size_t nvir_a = nmo - nocc_a, nvir_b = nmo - nocc_b;
  const std::size_t nso = 2 * nmo;
  const std::size_t nocc_so = nocc_a + nocc_b;
  const std::size_t nvir_so = nso - nocc_so;
  const bool restricted = hamiltonian.is_restricted();

  // ── full-space spin-blocked MO integrals (chemist (pq|rs)); the accessors
  // are
  //    container-agnostic: CanonicalFourCenter returns the aaaa/aabb/bbbb
  //    blocks, Cholesky lazily reconstructs them; restricted returns duplicated
  //    blocks. ──
  const auto h1 = hamiltonian.get_one_body_integrals();  // (MatrixXd a, b)
  const auto eri =
      hamiltonian.get_two_body_integrals();  // (VectorXd aa, ab, bb)
  const Eigen::MatrixXd& h1a = std::get<0>(h1);
  const Eigen::MatrixXd& h1b = std::get<1>(h1);
  const Eigen::VectorXd& eri_aa = std::get<0>(eri);
  const Eigen::VectorXd& eri_ab = std::get<1>(eri);
  const Eigen::VectorXd& eri_bb = std::get<2>(eri);
  const double core = hamiltonian.get_core_energy();

  auto e4 = [nmo](const Eigen::VectorXd& e, std::size_t a, std::size_t b,
                  std::size_t c, std::size_t d) -> double {
    return e[((a * nmo + b) * nmo + c) * nmo + d];
  };

  // ── occ-first blocked SO layout [a-occ, b-occ, a-vir, b-vir] ──
  std::vector<std::size_t> spat(nso);
  std::vector<int> spin(nso);
  {
    std::size_t k = 0;
    for (std::size_t p = 0; p < nocc_a; ++p) {
      spat[k] = p;
      spin[k++] = 0;
    }
    for (std::size_t p = 0; p < nocc_b; ++p) {
      spat[k] = p;
      spin[k++] = 1;
    }
    for (std::size_t p = nocc_a; p < nmo; ++p) {
      spat[k] = p;
      spin[k++] = 0;
    }
    for (std::size_t p = nocc_b; p < nmo; ++p) {
      spat[k] = p;
      spin[k++] = 1;
    }
  }

  // chemist (PQ|RS) spin-orbital value: nonzero only for spin_P==spin_Q and
  // spin_R==spin_S; (bb|aa) is the (aa|bb) block transposed.
  auto eri_so = [&](std::size_t P, std::size_t Q, std::size_t R,
                    std::size_t S) -> double {
    if (spin[P] != spin[Q] || spin[R] != spin[S]) return 0.0;
    const std::size_t p = spat[P], q = spat[Q], r = spat[R], s = spat[S];
    if (spin[P] == 0 && spin[R] == 0) return e4(eri_aa, p, q, r, s);
    if (spin[P] == 1 && spin[R] == 1) return e4(eri_bb, p, q, r, s);
    if (spin[P] == 0 && spin[R] == 1) return e4(eri_ab, p, q, r, s);
    return e4(eri_ab, r, s, p, q);  // (bb|aa)
  };

  auto vidx = [nso](std::size_t P, std::size_t Q, std::size_t R,
                    std::size_t S) {
    return ((P * nso + Q) * nso + R) * nso + S;
  };

  // ── chemist (PQ|RS); then V = <PQ||RS> via
  // utils::chemist_to_antisymmetrized.
  //    V is the antisymmetrized tensor used by the correlation machinery (Fock
  //    build, scalar, the gamma->chi re-normal-ordering). ──
  std::vector<double> chemist(nso * nso * nso * nso, 0.0);
  for (std::size_t P = 0; P < nso; ++P)
    for (std::size_t Q = 0; Q < nso; ++Q)
      for (std::size_t R = 0; R < nso; ++R)
        for (std::size_t S = 0; S < nso; ++S)
          chemist[vidx(P, Q, R, S)] = eri_so(P, Q, R, S);
  std::vector<double> V = utils::chemist_to_antisymmetrized(chemist, nso);

  // ── h1_so ──
  std::vector<double> h1_so(nso * nso, 0.0);
  for (std::size_t P = 0; P < nso; ++P)
    for (std::size_t Q = 0; Q < nso; ++Q)
      if (spin[P] == spin[Q])
        h1_so[P * nso + Q] =
            (spin[P] == 0) ? h1a(spat[P], spat[Q]) : h1b(spat[P], spat[Q]);

  // ── F = h1_so + sum_occ <P m || Q m>;  E0 = core + sum h1_mm + 1/2 sum
  // <mn||mn> ──
  std::vector<double> F = h1_so;
  for (std::size_t P = 0; P < nso; ++P)
    for (std::size_t Q = 0; Q < nso; ++Q)
      for (std::size_t m = 0; m < nocc_so; ++m)
        F[P * nso + Q] += V[vidx(P, m, Q, m)];
  double scalar = core;
  for (std::size_t m = 0; m < nocc_so; ++m) scalar += h1_so[m * nso + m];
  for (std::size_t m = 0; m < nocc_so; ++m)
    for (std::size_t n = 0; n < nocc_so; ++n)
      scalar += 0.5 * V[vidx(m, n, m, n)];

  // ── CC amplitudes (full-space, spin-separated) ──
  const auto& amp = wavefunction.get_container<data::AmplitudeContainer>();
  const auto t1p = amp.get_t1_amplitudes();  // (a, b)
  const auto t2t = amp.get_t2_amplitudes();  // (ab, aa, bb)
  const Eigen::VectorXd& t1a = real_vec(std::get<0>(t1p));
  const Eigen::VectorXd& t1b = real_vec(std::get<1>(t1p));
  const Eigen::VectorXd& t2ab = real_vec(std::get<0>(t2t));
  Eigen::VectorXd t2aa =
      real_vec(std::get<1>(t2t));  // copies (may antisymmetrize)
  Eigen::VectorXd t2bb = real_vec(std::get<2>(t2t));

  auto ab4 = [&](std::size_t i, std::size_t j, std::size_t a, std::size_t b) {
    return ((i * nocc_b + j) * nvir_a + a) * nvir_b + b;
  };
  auto aa4 = [&](std::size_t i, std::size_t j, std::size_t a, std::size_t b) {
    return ((i * nocc_a + j) * nvir_a + a) * nvir_a + b;
  };
  auto bb4 = [&](std::size_t i, std::size_t j, std::size_t a, std::size_t b) {
    return ((i * nocc_b + j) * nvir_b + a) * nvir_b + b;
  };

  // Restrictedness is a single system property shared by the Hamiltonian and
  // the amplitudes; guard against a mismatch. A restricted reference stores the
  // same-spin T2 as the RAW spatial amplitude (== t2ab) and is antisymmetrized
  // here; an unrestricted reference is already antisymmetric.
  const bool amps_restricted =
      t2aa.size() == t2ab.size() && (t2aa.array() == t2ab.array()).all();
  if (restricted != amps_restricted)
    throw std::runtime_error(
        std::string("ducc: the Hamiltonian is ") +
        (restricted ? "restricted" : "unrestricted") +
        " but the wavefunction amplitudes are " +
        (amps_restricted ? "restricted" : "unrestricted") +
        "; the Hamiltonian and wavefunction must share the same "
        "restricted/unrestricted character.");
  if (restricted) {
    Eigen::VectorXd anti(t2ab.size());
    for (std::size_t i = 0; i < nocc_a; ++i)
      for (std::size_t j = 0; j < nocc_a; ++j)
        for (std::size_t a = 0; a < nvir_a; ++a)
          for (std::size_t b = 0; b < nvir_a; ++b)
            anti[aa4(i, j, a, b)] =
                t2ab[ab4(i, j, a, b)] - t2ab[ab4(i, j, b, a)];
    t2aa = anti;
    t2bb = anti;
  }

  // ── map spin-separated amplitudes to blocked spin-orbital T1[a,i],
  // T2[a,b,i,j] ──
  std::vector<double> T1(nvir_so * nocc_so, 0.0);
  for (std::size_t I = 0; I < nocc_so; ++I) {
    const int si = (I < nocc_a) ? 0 : 1;
    const std::size_t pi = (I < nocc_a) ? I : (I - nocc_a);
    for (std::size_t A = 0; A < nvir_so; ++A) {
      const int sa = (A < nvir_a) ? 0 : 1;
      const std::size_t la = (A < nvir_a) ? A : (A - nvir_a);
      if (si != sa) continue;
      T1[A * nocc_so + I] =
          (si == 0) ? t1a[pi * nvir_a + la] : t1b[pi * nvir_b + la];
    }
  }

  std::vector<double> T2(nvir_so * nvir_so * nocc_so * nocc_so, 0.0);
  auto t2idx = [&](std::size_t a, std::size_t b, std::size_t i, std::size_t j) {
    return ((a * nvir_so + b) * nocc_so + i) * nocc_so + j;
  };
  for (std::size_t I = 0; I < nocc_so; ++I) {
    const int si = (I < nocc_a) ? 0 : 1;
    const std::size_t pi = (I < nocc_a) ? I : (I - nocc_a);
    for (std::size_t J = 0; J < nocc_so; ++J) {
      const int sj = (J < nocc_a) ? 0 : 1;
      const std::size_t pj = (J < nocc_a) ? J : (J - nocc_a);
      for (std::size_t A = 0; A < nvir_so; ++A) {
        const int sa = (A < nvir_a) ? 0 : 1;
        const std::size_t la = (A < nvir_a) ? A : (A - nvir_a);
        for (std::size_t B = 0; B < nvir_so; ++B) {
          const int sb = (B < nvir_a) ? 0 : 1;
          const std::size_t lb = (B < nvir_a) ? B : (B - nvir_a);
          if (si + sj != sa + sb) continue;  // spin conservation
          double val = 0.0;
          if (si == 0 && sj == 0 && sa == 0 && sb == 0)
            val = t2aa[aa4(pi, pj, la, lb)];
          else if (si == 1 && sj == 1 && sa == 1 && sb == 1)
            val = t2bb[bb4(pi, pj, la, lb)];
          else if (si == 0 && sj == 1 && sa == 0 && sb == 1)
            val = t2ab[ab4(pi, pj, la, lb)];
          else if (si == 1 && sj == 0 && sa == 1 && sb == 0)
            val = t2ab[ab4(pj, pi, lb, la)];
          else if (si == 0 && sj == 1 && sa == 1 && sb == 0)
            val = -t2ab[ab4(pi, pj, lb, la)];
          else if (si == 1 && sj == 0 && sa == 0 && sb == 1)
            val = -t2ab[ab4(pj, pi, la, lb)];
          T2[t2idx(A, B, I, J)] = val;
        }
      }
    }
  }

  SpinOrbitalData out;
  out.nso = nso;
  out.nocc_so = nocc_so;
  out.nvir_so = nvir_so;
  out.nocc_a = nocc_a;
  out.nocc_b = nocc_b;
  out.scalar = scalar;
  out.F = std::move(F);
  out.V = std::move(V);
  out.T1 = std::move(T1);
  out.T2 = std::move(T2);
  return out;
}

std::shared_ptr<data::Hamiltonian> assemble_active_hamiltonian(
    const SpinOrbitalData& dressed,
    const std::vector<std::size_t>& active_a_spatial,
    const std::vector<std::size_t>& active_b_spatial, bool restricted,
    std::shared_ptr<data::Orbitals> active_orbitals) {
  const std::size_t nso = dressed.nso;
  const std::size_t nocc_so = dressed.nocc_so;
  const std::size_t nocc_a = dressed.nocc_a;
  const std::size_t nocc_b = dressed.nocc_b;
  const std::size_t nmo = nso / 2;
  const std::size_t nvir_a = nmo - nocc_a;

  // active spatial MO (per spin) -> occ-first blocked spin-orbital index
  auto so_alpha = [&](std::size_t p) {
    return (p < nocc_a) ? p : (nocc_so + (p - nocc_a));
  };
  auto so_beta = [&](std::size_t p) {
    return (p < nocc_b) ? (nocc_a + p) : (nocc_so + nvir_a + (p - nocc_b));
  };

  // active spin-orbital list, sorted ascending (spin: 0=alpha, 1=beta)
  std::vector<std::pair<std::size_t, int>> act;
  act.reserve(active_a_spatial.size() + active_b_spatial.size());
  for (std::size_t p : active_a_spatial) act.emplace_back(so_alpha(p), 0);
  for (std::size_t p : active_b_spatial) act.emplace_back(so_beta(p), 1);
  std::sort(act.begin(), act.end());

  const std::size_t nact = act.size();
  std::vector<std::size_t> active_so(nact);
  std::vector<std::size_t> a_local, b_local, aol;  // positions within active_so
  for (std::size_t i = 0; i < nact; ++i) {
    active_so[i] = act[i].first;
    (act[i].second == 0 ? a_local : b_local).push_back(i);
    if (active_so[i] < nocc_so) aol.push_back(i);  // active-occupied
  }

  const std::vector<double>& F = dressed.F;  // fbar [nso, nso]
  const std::vector<double>& V = dressed.V;  // vbar <PQ||RS> [nso^4]
  // gamma = F/V restricted to the active space (indices into active_so)
  auto g1 = [&](std::size_t i, std::size_t j) {
    return F[active_so[i] * nso + active_so[j]];
  };
  auto g2 = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    return V[((active_so[i] * nso + active_so[j]) * nso + active_so[k]) * nso +
             active_so[l]];
  };

  // gamma -> chi (Fermi vacuum -> physical vacuum), Bauman et al. Eq.
  // (30)-(31):
  //   chi_1[p,q] = gamma_1[p,q] - sum_{m in aol} gamma_2[p,m,q,m]
  //   chi_2      = gamma_2 (unchanged; accessed via g2 below)
  std::vector<double> chi1(nact * nact, 0.0);
  for (std::size_t i = 0; i < nact; ++i)
    for (std::size_t j = 0; j < nact; ++j) {
      double v = g1(i, j);
      for (std::size_t m : aol) v -= g2(i, m, j, m);
      chi1[i * nact + j] = v;
    }

  // physical-vacuum scalar (downfolded core energy), Bauman et al. Eq. (32):
  //   C = scalar - sum_M chi_1[M,M] - 1/2 sum_MN chi_2[M,N,M,N]
  double core = dressed.scalar;
  for (std::size_t m : aol) core -= chi1[m * nact + m];
  for (std::size_t m : aol)
    for (std::size_t n : aol) core -= 0.5 * g2(m, n, m, n);

  const std::size_t na = a_local.size();
  const std::size_t nb = b_local.size();
  if (na != nb)
    throw std::runtime_error(
        "ducc: the DUCC active space must have equal alpha/beta active orbital "
        "counts (got " +
        std::to_string(na) + " alpha and " + std::to_string(nb) + " beta).");

  auto chi1_block = [&](const std::vector<std::size_t>& rows,
                        const std::vector<std::size_t>& cols) {
    Eigen::MatrixXd M(rows.size(), cols.size());
    for (std::size_t i = 0; i < rows.size(); ++i)
      for (std::size_t j = 0; j < cols.size(); ++j)
        M(i, j) = chi1[rows[i] * nact + cols[j]];
    return M;
  };
  // Two-body output in qdk's chemist convention. The same-spin block stores the
  // chemist representative g = 1/2 chi_2^(0,2,1,3)
  // (utils::antisymmetrized_to_chemist), whose antisymmetrization recovers the
  // physical <PQ||RS>; the qdk / MACIS solvers and QPE assume only 4-fold
  // symmetry and re-antisymmetrize the same-spin block. Per spin block the qdk
  // packing is same-spin g, opposite-spin 2g (matching wicked_ducc). At
  // ducc_level 0 this reproduces the CASCI energy of the input restricted to
  // the active space; unlike raw chemist it is computable at any BCH level from
  // the antisymmetrized two-body alone.
  std::vector<double> chi2(nact * nact * nact * nact);
  for (std::size_t i = 0; i < nact; ++i)
    for (std::size_t j = 0; j < nact; ++j)
      for (std::size_t k = 0; k < nact; ++k)
        for (std::size_t l = 0; l < nact; ++l)
          chi2[((i * nact + j) * nact + k) * nact + l] = g2(i, j, k, l);
  const std::vector<double> g = utils::antisymmetrized_to_chemist(chi2, nact);
  auto g_block = [&](const std::vector<std::size_t>& R0,
                     const std::vector<std::size_t>& R1,
                     const std::vector<std::size_t>& R2,
                     const std::vector<std::size_t>& R3, double factor) {
    Eigen::VectorXd out(static_cast<Eigen::Index>(R0.size() * R1.size() *
                                                  R2.size() * R3.size()));
    Eigen::Index idx = 0;
    for (std::size_t i : R0)
      for (std::size_t j : R1)
        for (std::size_t k : R2)
          for (std::size_t l : R3)
            out[idx++] = factor * g[((i * nact + j) * nact + k) * nact + l];
    return out;
  };

  std::unique_ptr<data::HamiltonianContainer> container;
  // The output carries the real active-space @p active_orbitals (matching the
  // hamiltonian_constructor CAS convention), so the effective Hamiltonian
  // retains the input orbital type/coefficients/basis rather than abstract
  // model orbitals. An empty (0x0) inactive Fock signals "no inactive-Fock
  // metadata", as hamiltonian_constructor does for the all-active case; the
  // downfolded inactive contribution is already folded into chi1 and the core
  // scalar.
  const Eigen::MatrixXd no_inactive_fock = Eigen::MatrixXd::Zero(0, 0);
  if (restricted) {
    // single spatial chemist block (alpha-beta Coulomb): v = 2 g[a,a,b,b]
    Eigen::MatrixXd h1 = chi1_block(a_local, a_local);
    Eigen::VectorXd v = g_block(a_local, a_local, b_local, b_local, 2.0);
    container = std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
        h1, v, active_orbitals, core, no_inactive_fock);
  } else {
    Eigen::MatrixXd h1_a = chi1_block(a_local, a_local);
    Eigen::MatrixXd h1_b = chi1_block(b_local, b_local);
    Eigen::VectorXd v_aaaa = g_block(a_local, a_local, a_local, a_local, 1.0);
    Eigen::VectorXd v_bbbb = g_block(b_local, b_local, b_local, b_local, 1.0);
    Eigen::VectorXd v_aabb = g_block(a_local, a_local, b_local, b_local, 2.0);
    container = std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
        h1_a, h1_b, v_aaaa, v_aabb, v_bbbb, active_orbitals, core,
        no_inactive_fock, no_inactive_fock);
  }
  return std::make_shared<data::Hamiltonian>(std::move(container));
}

std::shared_ptr<data::Hamiltonian> DuccSolver::_run_impl(
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<data::Wavefunction> wavefunction,
    std::shared_ptr<data::Orbitals> active_orbitals) const {
  const int64_t ducc_level = _settings->get<int64_t>("ducc_level");

  // DUCC dresses and downfolds a FULL-space Hamiltonian; the active subspace is
  // designated separately via the active_orbitals argument. A Hamiltonian that
  // carries an inactive Fock matrix is an already-reduced (frozen-core /
  // active-space) Hamiltonian: its one-/two-body integrals span only the active
  // orbitals while get_num_molecular_orbitals() still reports the full count,
  // so its integrals cannot be consumed here (extract_spinorbital_data would
  // index the active-sized integrals with full-space orbital indices). The
  // downfolded inactive contribution is instead reproduced by DUCC's own chi1 +
  // core scalar, so the input must be the untruncated full-space Hamiltonian.
  // Reject a pre-reduced input up front with a clear diagnostic.
  if (hamiltonian->has_inactive_fock_matrix())
    throw std::runtime_error(
        "ducc: the input Hamiltonian carries an inactive Fock matrix, i.e. it "
        "is "
        "already a reduced (frozen-core / active-space) Hamiltonian. DUCC "
        "requires the full-space Hamiltonian whose integrals span all "
        "molecular "
        "orbitals; select the active space via the active_orbitals argument "
        "rather than pre-reducing the Hamiltonian.");

  // Step 2a: spin-orbital F/V/T1/T2 from the full-space Hamiltonian +
  // amplitudes.
  const SpinOrbitalData data =
      extract_spinorbital_data(*hamiltonian, *wavefunction);

  // The active-space orbitals must be a subset of the full-space wavefunction
  // orbitals: the same single-particle basis, with active-space indices
  // selecting the active subset. Verify the shared basis (matching MO
  // coefficients), then read the active indices (no active space => all MOs).
  const auto wf_orbitals = wavefunction->get_orbitals();
  const auto& active_c = active_orbitals->get_coefficients_alpha();
  const auto& wf_c = wf_orbitals->get_coefficients_alpha();
  if (active_c.rows() != wf_c.rows() || active_c.cols() != wf_c.cols() ||
      !active_c.isApprox(wf_c))
    throw std::runtime_error(
        "ducc: the active-space orbitals must be a subset of the wavefunction "
        "orbitals (their molecular-orbital coefficients must match); pass an "
        "active-space Orbitals derived from the wavefunction's orbitals.");

  std::vector<std::size_t> active_a, active_b;
  if (active_orbitals->has_active_space()) {
    const auto [aa, ab] = active_orbitals->get_active_space_indices();
    active_a.assign(aa.begin(), aa.end());
    active_b.assign(ab.begin(), ab.end());
  } else {
    const std::size_t nmo = data.nso / 2;
    active_a.resize(nmo);
    active_b.resize(nmo);
    for (std::size_t p = 0; p < nmo; ++p) active_a[p] = active_b[p] = p;
  }

  if (ducc_level > 0) {
    // The BCH dressing (levels 1-2) builds bar{H} = e^{-sigma} H e^{sigma} with
    // sigma = T_ext - T_ext^dagger via a truncated Baker-Campbell-Hausdorff
    // expansion, evaluated symbolically + numerically with SeQuant/BTAS. That
    // backend is ported in Phase 2b; until then only the bare level 0 is
    // served.
    throw std::runtime_error(
        "ducc: ducc_level " + std::to_string(ducc_level) +
        " requires the BCH transformation, whose SeQuant/BTAS backend is not "
        "yet available in this build. ducc_level 0 (the bare active-space "
        "Hamiltonian) is fully supported.");
  }

  // Step 2c: level 0 has no BCH dressing, so the bare spin-orbital F/V are used
  // directly; the result is the input Hamiltonian restricted to the active
  // space.
  return assemble_active_hamiltonian(
      data, active_a, active_b, hamiltonian->is_restricted(), active_orbitals);
}

}  // namespace qdk::chemistry::algorithms::microsoft

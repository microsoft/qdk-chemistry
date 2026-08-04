// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ducc.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <memory>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/amplitude_container.hpp>
#include <qdk/chemistry/utils/eri_notation.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <variant>
#include <vector>

// ── BTAS backend (level>0 BCH dressing; DUCC step 2b) ──
// The symbolic Baker-Campbell-Hausdorff transform and partial-Wick contraction
// are performed OFFLINE by the SeQuant-based code generator
// (ducc/export_ducc_btas.cpp), whose output is checked in as
// ducc_equations.inc. This translation unit therefore needs no symbolic algebra
// library at all: it only supplies tensor blocks to the generated code and
// executes it with BTAS.
#include <btas/btas.h>
#include <btas/tensor.h>

#include <map>
#include <numeric>

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

// Duplicate a restricted Orbitals into an equal-spin unrestricted Orbitals
// (alpha = beta), preserving any active-space designation. A spin-blocked
// (unrestricted-type) effective Hamiltonian requires unrestricted orbitals --
// the Hamiltonian enforces that its container and orbitals agree on
// restrictedness -- and a closed-shell system is faithfully represented with
// identical alpha/beta channels.
std::shared_ptr<data::Orbitals> as_unrestricted(const data::Orbitals& r) {
  const Eigen::MatrixXd& c = r.get_coefficients_alpha();
  std::optional<Eigen::VectorXd> e;
  if (r.has_energies()) e = r.get_energies_alpha();
  std::optional<Eigen::MatrixXd> s = r.get_overlap_matrix();
  // The index sets carry through unchanged; they are null when unset.
  return std::make_shared<data::Orbitals>(c, c, e, e, s, r.get_basis_set(),
                                          r.active_indices(),
                                          r.inactive_indices());
}

std::shared_ptr<data::Hamiltonian> assemble_active_hamiltonian(
    const SpinOrbitalData& dressed,
    const std::vector<std::size_t>& active_a_spatial,
    const std::vector<std::size_t>& active_b_spatial, bool restricted,
    int ducc_level, std::shared_ptr<data::Orbitals> active_orbitals) {
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

  const std::vector<double>& F = dressed.F;  // fbar
  const std::vector<double>& V = dressed.V;  // vbar <PQ||RS>
  // dressed.F/V are either full-space [nso] or already restricted to the active
  // space [nact] in this same ascending active_so order (what the generated
  // DUCC equations produce). The two are distinguished by size; when
  // nact == nso the active list is the identity permutation, so the two
  // readings coincide and the choice is immaterial.
  const bool compact = F.size() == nact * nact;
  if (!compact && F.size() != nso * nso)
    throw std::runtime_error(
        "ducc: dressed one-body has neither full-space (nso^2) nor "
        "active-space (nact^2) extent.");
  // gamma = F/V restricted to the active space (indices into active_so)
  auto g1 = [&](std::size_t i, std::size_t j) {
    return compact ? F[i * nact + j] : F[active_so[i] * nso + active_so[j]];
  };
  auto g2 = [&](std::size_t i, std::size_t j, std::size_t k, std::size_t l) {
    if (compact) return V[((i * nact + j) * nact + k) * nact + l];
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
  // packing is same-spin g, opposite-spin 2g (matching the wicked-based DUCC
  // reference). At
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
  // Output container form. A restricted level-0 result has full 8-fold two-body
  // symmetry and is stored as a single-block restricted container (matching the
  // hamiltonian_constructor input format). The BCH dressing (level > 0) lowers
  // the two-body to the 4-fold-symmetric 1/2-antisymmetrized representative,
  // which a single restricted block cannot convey to consumers that assume
  // 8-fold symmetry -- notably the qubit mapper's restricted fast path, which
  // spin-sums the excitations and exploits the (pq|rs)=(qp|rs) symmetry the
  // dressed integrals lack. A dressed result is therefore emitted in
  // spin-blocked (aaaa/bbbb/aabb) form -- with identical alpha/beta blocks for
  // a restricted system -- so every consumer reads it through its unrestricted
  // path (each spin channel independently, assuming only the (pq|rs)=(rs|pq)
  // Coulomb symmetry the dressed integrals retain). A restricted level>0 output
  // is thus an unrestricted-type container consumed by the qubit mapper (or
  // PySCF direct_uhf), not the restricted-only MACIS solver.
  if (restricted && ducc_level == 0) {
    // single spatial chemist block (alpha-beta Coulomb): v = 2 g[a,a,b,b]
    Eigen::MatrixXd h1 = chi1_block(a_local, a_local);
    Eigen::VectorXd v = g_block(a_local, a_local, b_local, b_local, 2.0);
    container = std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
        h1, v, active_orbitals, core, no_inactive_fock);
  } else {
    // A restricted level>0 result is emitted here in spin-blocked form, so its
    // orbitals must also be unrestricted to satisfy the container's
    // restrictedness-consistency check (a closed-shell system is represented
    // faithfully with identical alpha/beta channels). An unrestricted input
    // already carries unrestricted orbitals.
    std::shared_ptr<data::Orbitals> out_orbitals =
        restricted ? as_unrestricted(*active_orbitals) : active_orbitals;
    Eigen::MatrixXd h1_a = chi1_block(a_local, a_local);
    Eigen::MatrixXd h1_b = chi1_block(b_local, b_local);
    Eigen::VectorXd v_aaaa = g_block(a_local, a_local, a_local, a_local, 1.0);
    Eigen::VectorXd v_bbbb = g_block(b_local, b_local, b_local, b_local, 1.0);
    Eigen::VectorXd v_aabb = g_block(a_local, a_local, b_local, b_local, 2.0);
    container = std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
        h1_a, h1_b, v_aaaa, v_aabb, v_bbbb, out_orbitals, core,
        no_inactive_fock, no_inactive_fock);
  }
  return std::make_shared<data::Hamiltonian>(std::move(container));
}

namespace {

// ─────────────────────────────────────────────────────────────────────────
// DUCC BCH dressing (level > 0), evaluated by GENERATED BTAS code.
//
// All symbolic work -- the unitary transform bar{H} = e^{-sigma} H e^{sigma}
// with sigma = T - T^dagger, its DUCC F-split BCH truncation, the partial Wick
// contraction, the truncation to <=2-body and the contraction-order
// optimization -- is done OFFLINE by the SeQuant-based code generator
// ducc/export_ducc_btas.cpp. Its output is checked in as ducc_equations.inc,
// which defines run_all_L0 / run_all_L1 / run_all_L2: these pull tensor blocks
// from the TensorProvider below, contract them with BTAS, and push the dressed
// blocks back. This translation unit therefore links no symbolic-algebra
// library and its per-run cost is BTAS contractions only.
//
// The generated equations write ACTIVE-sized output blocks directly (their free
// legs carry active extents), so the effective Hamiltonian is never
// materialized over the full spin-orbital space.
// ─────────────────────────────────────────────────────────────────────────
using BTensorD = btas::Tensor<double>;

/// Placement of one index space inside the compact active output.
struct Rng {
  std::size_t off, ext;
};

/// Supplies -- and accumulates -- the tensor blocks referenced by the generated
/// DUCC equations, slicing the shared full-space spin-orbital F/V/T1/T2.
///
/// Index conventions. Leg space letters are 'o' (occupied), 'v' (virtual) and
/// 'p' (complete). Occupied space-local indices coincide with spin-orbital
/// indices; a virtual space-local index @c a is spin-orbital @c nocc+a. The
/// per-leg @c mask marks legs the generator restricted to the active space
/// ('A') versus legs kept at full extent ('-').
class TensorProvider {
 public:
  /// @param data Full-space spin-orbital data (occ-first blocked layout).
  /// @param active_occ Active occupied spin-orbital indices, ascending.
  /// @param active_virt Active virtual indices (spin-orbital - nocc),
  /// ascending.
  TensorProvider(const SpinOrbitalData& data,
                 std::vector<std::size_t> active_occ,
                 std::vector<std::size_t> active_virt)
      : m_nocc(data.nocc_so),
        m_nvirt(data.nvir_so),
        m_nmo(data.nso),
        m_act_occ(std::move(active_occ)),
        m_act_virt(std::move(active_virt)),
        m_nao(m_act_occ.size()),
        m_nav(m_act_virt.size()),
        m_nact(m_nao + m_nav),
        m_F(data.F),
        m_V(data.V),
        m_T1(data.T1),
        m_T2(data.T2) {}

  /// @param mask per-leg active flags: 'A' restrict to active, '-' keep full.
  BTensorD get(const std::string& label, const std::string& tags,
               const std::string& mask) {
    // The mask belongs in the cache key: the same leaf is requested with
    // different active-slice patterns by terms with different free legs.
    const std::string key = label + "_" + tags + "#" + mask;
    if (auto it = m_store.find(key); it != m_store.end()) return it->second;
    BTensorD t = make(label, tags, mask);
    m_store.emplace(key, t);
    return t;
  }

  void put(const std::string& label, const std::string& tags,
           const std::string& mask, const BTensorD& value) {
    m_store[label + "_" + tags + "#" + mask] = value;
  }

  double get_scalar(const std::string& label) { return m_scalars[label]; }
  void put_scalar(const std::string& label, double value) {
    m_scalars[label] = value;
  }

  /// Gather the accumulated Fbar/Vbar blocks into the compact active output and
  /// add the bare H = F + 1/4 V over the active block.
  ///
  /// The partial Wick returns only the vacuum expectation value of the lone
  /// bare operator, but bar{H} = H + commutators contains H with coefficient 1,
  /// so H is added back here. The 1/4 becomes 1 under the antisymmetrizer
  /// applied downstream.
  void assemble(BTensorD& Fbar, BTensorD& Vbar) const {
    const std::size_t nt = m_nact;
    Fbar = BTensorD{btas::Range{nt, nt}};
    Vbar = BTensorD{btas::Range{nt, nt, nt, nt}};
    Fbar.fill(0.0);
    Vbar.fill(0.0);

    for (const auto& [key, block] : m_store) {
      const auto us = key.find('_');
      const auto hash = key.find('#');
      const std::string label = key.substr(0, us);
      if (label != "Fbar" && label != "Vbar") continue;
      const std::string tags = key.substr(us + 1, hash - us - 1);

      std::vector<Rng> r;
      for (char c : tags) r.push_back(active_range(c));

      if (label == "Fbar") {
        for (std::size_t a = 0; a < r[0].ext; ++a)
          for (std::size_t b = 0; b < r[1].ext; ++b)
            Fbar(r[0].off + a, r[1].off + b) += block(a, b);
      } else {
        for (std::size_t a = 0; a < r[0].ext; ++a)
          for (std::size_t b = 0; b < r[1].ext; ++b)
            for (std::size_t c = 0; c < r[2].ext; ++c)
              for (std::size_t d = 0; d < r[3].ext; ++d)
                Vbar(r[0].off + a, r[1].off + b, r[2].off + c, r[3].off + d) +=
                    block(a, b, c, d);
      }
    }

    const auto a2so = active_gather('p');  // compact position -> spin-orbital
    const std::size_t n = m_nmo;
    for (std::size_t p = 0; p < nt; ++p)
      for (std::size_t q = 0; q < nt; ++q)
        Fbar(p, q) += m_F[a2so[p] * n + a2so[q]];
    for (std::size_t p = 0; p < nt; ++p)
      for (std::size_t q = 0; q < nt; ++q)
        for (std::size_t r = 0; r < nt; ++r)
          for (std::size_t s = 0; s < nt; ++s)
            Vbar(p, q, r, s) +=
                0.25 *
                m_V[((a2so[p] * n + a2so[q]) * n + a2so[r]) * n + a2so[s]];
  }

 private:
  std::size_t full_extent(char c) const {
    if (c == 'o') return m_nocc;
    if (c == 'v') return m_nvirt;
    return m_nmo;  // 'p'
  }

  /// Space-local indices kept for an active leg. For 'p' the active occupied
  /// and active virtual pieces are concatenated, which is ascending in the
  /// spin-orbital index because every occupied index precedes every virtual
  /// one -- the same ordering assemble_active_hamiltonian's sorted active
  /// spin-orbital list has, so the compact layouts agree by construction.
  std::vector<std::size_t> active_gather(char c) const {
    if (c == 'o') return m_act_occ;
    if (c == 'v') return m_act_virt;
    std::vector<std::size_t> g = m_act_occ;
    for (std::size_t a : m_act_virt) g.push_back(m_nocc + a);
    return g;
  }

  /// Placement of a space within the compact active output.
  Rng active_range(char c) const {
    if (c == 'o') return {0, m_nao};
    if (c == 'v') return {m_nao, m_nav};
    return {0, m_nact};  // 'p'
  }

  /// Space-local index -> spin-orbital index.
  std::size_t to_so(char c, std::size_t local) const {
    return c == 'v' ? m_nocc + local : local;
  }

  /// Whether a space-local index lies inside the active space.
  bool in_active(char c, std::size_t local) const {
    if (c == 'o')
      return std::binary_search(m_act_occ.begin(), m_act_occ.end(), local);
    if (c == 'v')
      return std::binary_search(m_act_virt.begin(), m_act_virt.end(), local);
    return false;
  }

  BTensorD make(const std::string& label, const std::string& tags,
                const std::string& mask) const {
    const std::size_t rank = tags.size();

    // per-leg map: output index -> space-local source index
    std::vector<std::vector<std::size_t>> src(rank);
    std::vector<std::size_t> ext(rank);
    for (std::size_t k = 0; k < rank; ++k) {
      if (k < mask.size() && mask[k] == 'A') {
        src[k] = active_gather(tags[k]);
      } else {
        src[k].resize(full_extent(tags[k]));
        std::iota(src[k].begin(), src[k].end(), std::size_t{0});
      }
      ext[k] = src[k].size();
    }

    BTensorD out{btas::Range{ext}};
    out.fill(0.0);
    if (label == "Fbar" || label == "Vbar") return out;  // zeroed accumulator

    const std::size_t n = m_nmo;
    std::vector<std::size_t> idx(rank, 0);
    for (std::size_t flat = 0; flat < out.size(); ++flat) {
      if (label == "f") {
        out.data()[flat] = m_F[to_so(tags[0], src[0][idx[0]]) * n +
                               to_so(tags[1], src[1][idx[1]])];
      } else if (label == "g") {
        out.data()[flat] = m_V[((to_so(tags[0], src[0][idx[0]]) * n +
                                 to_so(tags[1], src[1][idx[1]])) *
                                    n +
                                to_so(tags[2], src[2][idx[2]])) *
                                   n +
                               to_so(tags[3], src[3][idx[3]])];
      } else if (label == "t" || label == "t_") {
        out.data()[flat] = amplitude(tags, src, idx);
      } else {
        throw std::runtime_error("ducc: unsupported leaf tensor '" + label +
                                 "'");
      }
      for (std::size_t k = rank; k-- > 0;) {
        if (++idx[k] < ext[k]) break;
        idx[k] = 0;
      }
    }
    return out;
  }

  /// Amplitude blocks in space-local coordinates: "vo" = T1[a,i], "ov" = its
  /// adjoint, "vvoo" = T2[a,b,i,j], "oovv" = its adjoint.
  ///
  /// T_ext: DUCC's cluster operator carries only EXTERNAL excitations, so an
  /// amplitude all of whose indices are active is structurally zero. With every
  /// orbital active this kills T entirely, sigma = 0, and bar{H} collapses to
  /// H -- the level > 0 result then reduces to the bare level-0 one.
  double amplitude(const std::string& tags,
                   const std::vector<std::vector<std::size_t>>& src,
                   const std::vector<std::size_t>& idx) const {
    const std::size_t no = m_nocc, nv = m_nvirt;
    std::vector<std::size_t> loc(tags.size());
    bool all_active = true;
    for (std::size_t k = 0; k < tags.size(); ++k) {
      loc[k] = src[k][idx[k]];
      all_active = all_active && in_active(tags[k], loc[k]);
    }
    if (all_active) return 0.0;

    if (tags == "vo") return m_T1[loc[0] * no + loc[1]];
    if (tags == "ov") return m_T1[loc[1] * no + loc[0]];
    if (tags == "vvoo")
      return m_T2[((loc[0] * nv + loc[1]) * no + loc[2]) * no + loc[3]];
    if (tags == "oovv")
      return m_T2[((loc[2] * nv + loc[3]) * no + loc[0]) * no + loc[1]];
    throw std::runtime_error("ducc: unsupported cluster block '" + tags + "'");
  }

  std::size_t m_nocc, m_nvirt, m_nmo;
  std::vector<std::size_t> m_act_occ, m_act_virt;
  std::size_t m_nao, m_nav, m_nact;
  const std::vector<double>&m_F, &m_V, &m_T1, &m_T2;  // row-major, not owned
  std::map<std::string, BTensorD> m_store;
  std::map<std::string, double> m_scalars;
};

// Generated DUCC equations (run_all_L0 / run_all_L1 / run_all_L2) plus the
// qdk_btas permute/scal/dot adapters they use. It re-includes <btas/btas.h>,
// <btas/tensor.h>, <cmath> and <vector>; those are all included at the top of
// this file, so their include guards make the re-inclusion a no-op and nothing
// is actually declared at this (anonymous-namespace) scope but the equations.
#include "ducc_equations.inc"

/// A[V]_pqrs = V_pqrs - V_qprs - V_pqsr + V_qpsr.
///
/// Turns the canonical generator the Wick expansion produces into the physical
/// antisymmetric <PQ||RS>. Unnormalized (coefficient 1), matching the
/// wicked-based DUCC reference's active-Hamiltonian assembly; applied to an
/// already
/// antisymmetric tensor it yields 4x that tensor, so the bare 1/4 V of
/// TensorProvider::assemble recovers exactly V.
///
/// Applied to the ASSEMBLED tensor, not per block: A permutes bra and ket
/// indices and the DUCC blocks have mixed index spaces (oovp, opvp, ...), so a
/// per-block A would move elements into a different block. Fbar needs no
/// counterpart -- at rank 2 the antisymmetrizer is the identity.
BTensorD antisymmetrize_2body(const BTensorD& V) {
  BTensorD out{V.range()};
  out.fill(0.0);
  qdk_btas::accumulate(1.0, V, {'p', 'q', 'r', 's'}, out, {'p', 'q', 'r', 's'});
  qdk_btas::accumulate(-1.0, V, {'q', 'p', 'r', 's'}, out,
                       {'p', 'q', 'r', 's'});
  qdk_btas::accumulate(-1.0, V, {'p', 'q', 's', 'r'}, out,
                       {'p', 'q', 'r', 's'});
  qdk_btas::accumulate(1.0, V, {'q', 'p', 's', 'r'}, out, {'p', 'q', 'r', 's'});
  return out;
}

/// BCH-dress the spin-orbital F/V/scalar for DUCC level > 0.
///
/// sigma is built from the EXTERNAL cluster amplitudes only (see
/// TensorProvider::amplitude), so the transform folds external correlation into
/// the active space. Returns a copy of @p data whose scalar/F/V are the
/// level-@p level DUCC effective values restricted to the ACTIVE space: F is
/// [nact, nact] and V is [nact^4], indexed by the ascending active
/// spin-orbital order that assemble_active_hamiltonian also uses.
SpinOrbitalData ducc_bch_dress(const SpinOrbitalData& data, int level,
                               const std::vector<std::size_t>& active_so) {
  const std::size_t nocc = data.nocc_so, nvirt = data.nvir_so, nmo = data.nso;
  std::vector<std::size_t> act_occ, act_virt;
  for (std::size_t so : active_so) {
    if (so < nocc)
      act_occ.push_back(so);
    else
      act_virt.push_back(so - nocc);
  }
  std::sort(act_occ.begin(), act_occ.end());
  std::sort(act_virt.begin(), act_virt.end());
  const std::size_t nao = act_occ.size(), nav = act_virt.size();
  const std::size_t nact = nao + nav;

  TensorProvider provider(data, std::move(act_occ), std::move(act_virt));
  switch (level) {
    case 1:
      run_all_L1(provider, nocc, nvirt, nmo, nao, nav, nact);
      break;
    case 2:
      run_all_L2(provider, nocc, nvirt, nmo, nao, nav, nact);
      break;
    default:
      throw std::runtime_error(
          "ducc: ducc_level " + std::to_string(level) +
          " is not supported; the generated DUCC equations cover levels 0-2.");
  }

  BTensorD Fbar, Vbar;
  provider.assemble(Fbar, Vbar);
  Vbar = antisymmetrize_2body(Vbar);

  // Pack into the extraction convention: F = Fock, V = antisymmetrized
  // <PQ||RS>, scalar = reference energy + BCH scalar correction -- all over the
  // active space.
  SpinOrbitalData dressed = data;
  dressed.scalar = data.scalar + provider.get_scalar("E0");
  dressed.F = std::vector<double>(Fbar.begin(), Fbar.end());
  dressed.V = std::vector<double>(Vbar.begin(), Vbar.end());
  return dressed;
}

}  // namespace

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
    const auto active_ai = active_orbitals->active_indices();
    active_a = data::spin_channel_indices(active_ai, data::axes::alpha());
    active_b = data::spin_channel_indices(active_ai, data::axes::beta());
  } else {
    const std::size_t nmo = data.nso / 2;
    active_a.resize(nmo);
    active_b.resize(nmo);
    for (std::size_t p = 0; p < nmo; ++p) active_a[p] = active_b[p] = p;
  }

  // Step 2b: BCH dressing for levels 1-2 -- bar{H} = e^{-sigma} H e^{sigma}
  // with sigma = T_ext - T_ext^dagger (external cluster amplitudes only),
  // evaluated by the generated BTAS equations. Level 0 skips this and uses the
  // bare spin-orbital data directly.
  SpinOrbitalData effective = data;
  if (ducc_level > 0) {
    // Active spin-orbitals (occ-first blocked layout) that drive the external
    // (T_ext) amplitude restriction, via the same spatial-MO -> spin-orbital
    // mapping used by assemble_active_hamiltonian.
    const std::size_t nocc_a = data.nocc_a, nocc_b = data.nocc_b;
    const std::size_t nocc_so = data.nocc_so, nmo = data.nso / 2;
    const std::size_t nvir_a = nmo - nocc_a;
    auto so_alpha = [&](std::size_t p) {
      return (p < nocc_a) ? p : (nocc_so + (p - nocc_a));
    };
    auto so_beta = [&](std::size_t p) {
      return (p < nocc_b) ? (nocc_a + p) : (nocc_so + nvir_a + (p - nocc_b));
    };
    std::vector<std::size_t> active_so;
    active_so.reserve(active_a.size() + active_b.size());
    for (std::size_t p : active_a) active_so.push_back(so_alpha(p));
    for (std::size_t p : active_b) active_so.push_back(so_beta(p));
    effective = ducc_bch_dress(data, static_cast<int>(ducc_level), active_so);
  }

  // Step 2c: assemble the effective active-space Hamiltonian (gamma->chi
  // downfold + active restriction). For level 0 `effective` is the bare
  // extraction, so the result reproduces the input Hamiltonian restricted to
  // the active space (CASCI(input) == FCI(output)).
  return assemble_active_hamiltonian(
      effective, active_a, active_b, hamiltonian->is_restricted(),
      static_cast<int>(ducc_level), active_orbitals);
}

}  // namespace qdk::chemistry::algorithms::microsoft

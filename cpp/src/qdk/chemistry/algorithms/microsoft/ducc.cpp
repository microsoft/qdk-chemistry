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

// ── SeQuant + BTAS backend (level>0 BCH dressing; DUCC step 2b) ──
// SeQuant supplies the symbolic unitary Baker-Campbell-Hausdorff transform and
// partial-Wick contraction; the BTAS backend evaluates the resulting tensor
// network. Always linked (SeQuant::SeQuant carries SEQUANT_HAS_BTAS=1), so this
// is compiled unconditionally.
#include <btas/btas.h>
#include <btas/tensor.h>

#include <SeQuant/core/binary_node.hpp>
#include <SeQuant/core/context.hpp>
#include <SeQuant/core/eval/backends/btas/eval_expr.hpp>
#include <SeQuant/core/eval/backends/btas/result.hpp>
#include <SeQuant/core/eval/eval.hpp>
#include <SeQuant/core/expr.hpp>
#include <SeQuant/core/expressions/tensor.hpp>
#include <SeQuant/core/op.hpp>
#include <SeQuant/core/optimize/optimize.hpp>
#include <SeQuant/core/runtime.hpp>
#include <SeQuant/core/tensor_canonicalizer.hpp>
#include <SeQuant/core/wick.hpp>
#include <SeQuant/domain/mbpt/context.hpp>
#include <SeQuant/domain/mbpt/convention.hpp>
#include <SeQuant/domain/mbpt/op.hpp>
#include <SeQuant/domain/mbpt/utils.hpp>
#include <SeQuant/domain/mbpt/vac_av.hpp>
#include <map>
#include <mutex>
#include <numeric>
#include <range/v3/range/conversion.hpp>
#include <set>

// SeQuant's op-level partial-Wick expectation-value driver. It is not declared
// in a public header, but is an exported (external-linkage) symbol of
// libSeQuant-mbpt; declare it here to drive a rank-uncapped partial Wick
// (full_contractions=false) that the local post-filter then truncates to
// <=2-body. Signature matches upstream SeQuant (no max_ops rank-cap parameter).
namespace sequant::mbpt::op {
ExprPtr expectation_value_impl(ExprPtr expr,
                               const OpConnections<std::wstring>& connect,
                               const OpConnections<std::wstring>& avoid,
                               bool use_topology, bool screen, bool skip_clone,
                               bool full_contractions);
}  // namespace sequant::mbpt::op

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
  std::optional<data::Orbitals::UnrestrictedCASIndices> cas;
  if (r.has_active_space()) {
    const auto& [aa, ab] = r.get_active_space_indices();
    const auto& [ia, ib] = r.get_inactive_space_indices();
    cas = std::make_tuple(aa, ab, ia, ib);
  }
  return std::make_shared<data::Orbitals>(c, c, e, e, s, r.get_basis_set(),
                                          cas);
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
// SeQuant + BTAS DUCC machinery (level>0 BCH dressing). Ported from the
// validated standalone prototype (ducc/ducc_sequant.cpp), fed from the
// in-memory spin-orbital data rather than files, and emitting the full
// spin-orbital effective Hamiltonian (assemble_active_hamiltonian performs the
// active-space restriction, so no active-space slicing is needed here).
// ─────────────────────────────────────────────────────────────────────────
using namespace sequant;
using BTensorD = btas::Tensor<double>;

// One-time SeQuant global setup: spaces (2-space spin-orbital occ/virt,
// wicked-equivalent; active/frozen/external distinctions are imposed
// numerically, not symbolically, so hole/particle stay base-directional and
// [H,T] survives), Fermi (SingleProduct) vacuum, canonicalizer, mbpt registry.
void ensure_sequant_setup() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    static sequant::detail::OpIdRegistrar op_id_registrar;
    auto isr = mbpt::make_min_sr_spaces(mbpt::SpinConvention::None);
    set_default_context(Context({.index_space_registry_shared_ptr = isr,
                                 .vacuum = Vacuum::SingleProduct}));
    TensorCanonicalizer::register_instance(
        std::make_shared<DefaultTensorCanonicalizer>());
    mbpt::set_default_mbpt_context(mbpt::Context::Options{
        .op_registry_ptr = mbpt::make_minimal_registry()});
  });
}

// residual FNOperators appearing in a Wick-result term
std::vector<const FNOperator*> collect_fnops(const ExprPtr& term) {
  std::vector<const FNOperator*> ops;
  auto visit = [&ops](const ExprPtr& e) {
    if (e.is<FNOperator>()) ops.push_back(&e.as<FNOperator>());
  };
  if (term.is<Product>()) {
    for (const auto& f : term.as<Product>().factors()) visit(f);
  } else {
    visit(term);
  }
  return ops;
}

// Keep only terms whose single residual normal operator is <= body_cap-body
// (rank-0 scalars kept). Applied BEFORE simplify(), this reproduces a
// rank-capped partial Wick (wicked's contract(...,0,4) / a patched
// WickTheorem::max_ops) with no SeQuant source change.
ExprPtr keep_up_to_body(const ExprPtr& nh, std::size_t body_cap) {
  auto keep = [&](const ExprPtr& t) {
    auto ops = collect_fnops(t);
    if (ops.empty()) return true;  // scalar (rank 0)
    if (ops.size() > 1)
      return false;  // >1 residual operator (should not occur)
    return ops.front()->ncreators() <= body_cap;
  };
  if (!nh) return nh;
  if (!nh.is<Sum>()) return keep(nh) ? nh : ex<Constant>(rational{0});
  std::vector<ExprPtr> kept;
  for (const auto& t : nh.as<Sum>().summands())
    if (keep(t)) kept.push_back(t);
  return ex<Sum>(kept.begin(), kept.end());
}

// coefficient of a term = the term with its residual FNOperator removed
ExprPtr coeff_of(const ExprPtr& term) {
  if (!term.is<Product>()) return term;
  const auto& p = term.as<Product>();
  auto out = ex<Product>(p.scalar(), ExprPtrList{});
  auto& op = out->as<Product>();
  for (const auto& f : p.factors())
    if (!f.is<FNOperator>()) op.append(1, f, Product::Flatten::No);
  return out;
}

// [A,B] = A*B - B*A (operator commutator)
inline ExprPtr commutator(const ExprPtr& A, const ExprPtr& B) {
  return A * B - B * A;
}

// DUCC F-split BCH (matches wicked_ducc._wicked_bch): the Fock part sits one
// commutator order higher than the two-body part (the DUCC truncation).
//   L1: H + [H,s] + 1/2 [[F,s],s]
//   L2: H + [H,s] + 1/2 [[H,s],s] + 1/6 [[[F,s],s],s]
// H = mbpt::H() (f + 1/4 g over the complete space), F = mbpt::F(),
// s = sigma = T - T^dagger.
ExprPtr ducc_fsplit_bch(std::size_t level, const ExprPtr& H, const ExprPtr& F,
                        const ExprPtr& sigma) {
  if (level == 0) return H;
  auto c1 = commutator(H, sigma);
  if (level == 1) {
    auto c2F = commutator(commutator(F, sigma), sigma);
    return H + c1 + ex<Constant>(rational{1, 2}) * c2F;
  }
  if (level == 2) {
    auto c2H = commutator(commutator(H, sigma), sigma);
    auto c3F = commutator(commutator(commutator(F, sigma), sigma), sigma);
    return H + c1 + ex<Constant>(rational{1, 2}) * c2H +
           ex<Constant>(rational{1, 6}) * c3F;
  }
  throw std::runtime_error("ducc: F-split BCH level must be 0, 1, or 2");
}

// full-MO range of an index: occ (base "i") -> [0,nocc); virt (base "a") ->
// [nocc,nmo); complete (base "p") -> [0,nmo).
struct SpaceRange {
  std::size_t off, ext;
};
inline SpaceRange space_range(const Index& idx, std::size_t nocc,
                              std::size_t nvirt) {
  const auto bk = idx.space().base_key();
  if (bk == L"i") return {0, nocc};
  if (bk == L"a") return {nocc, nvirt};
  if (bk == L"p") return {0, nocc + nvirt};
  throw std::runtime_error("ducc: unsupported index space in Wick residual");
}
inline char space_letter(const Index& idx) {
  const auto bk = idx.space().base_key();
  if (bk == L"i") return 'o';
  if (bk == L"a") return 'v';
  if (bk == L"p") return 'p';
  return '?';
}

// BTAS leaf yielder backed by the in-memory spin-orbital F/V/T1/T2. Slices the
// shared full tensors to whatever o/v/p leg block a leaf tensor requests
// (treating "p" as the full range avoids symbolic p-leg resolution).
class DataYielder {
  std::size_t nocc_, nvirt_, nmo_;
  const std::vector<double>&F_, &V_, &T1_, &T2_;  // row-major, not owned
  mutable std::map<std::string, ResultPtr> cache_;
  // b1' active-output restriction: when active_, each residual term's free
  // (output) legs are sliced to the active block so BTAS evaluates only it
  // (contracted/internal legs stay full). The active spin-orbital indices are
  // split into occupied (< nocc_) and virtual (>= nocc_) partitions.
  bool active_ = false;
  std::vector<std::size_t> active_occ_so_, active_virt_so_;
  mutable std::set<std::wstring>
      free_labels_;  // current term's free-leg labels

  std::string key_of(const Tensor& t) const {
    std::string k;
    for (wchar_t wc : std::wstring(t.label())) k += static_cast<char>(wc);
    k += '_';
    for (auto&& idx : t.const_braket_indices()) k += space_letter(idx);
    return k;
  }

  // fresh BTensorD block for leaf tensor `t` by slicing the full data
  BTensorD make_block(const Tensor& t, const std::string& key) const {
    std::vector<Index> idx(t.const_braket_indices().begin(),
                           t.const_braket_indices().end());
    std::vector<SpaceRange> sr;
    std::vector<std::size_t> ext;
    for (auto&& i : idx) {
      auto r = space_range(i, nocc_, nvirt_);
      sr.push_back(r);
      ext.push_back(r.ext);
    }
    const std::string lbl(key.begin(), key.begin() + key.find('_'));
    BTensorD out{btas::Range{ext}};

    if (lbl == "f") {  // Fock F[nmo,nmo]
      for (std::size_t a = 0; a < ext[0]; ++a)
        for (std::size_t b = 0; b < ext[1]; ++b)
          out(a, b) = F_[(sr[0].off + a) * nmo_ + (sr[1].off + b)];
    } else if (lbl == "g") {  // antisymmetrized two-body V[nmo,nmo,nmo,nmo]
      for (std::size_t a = 0; a < ext[0]; ++a)
        for (std::size_t b = 0; b < ext[1]; ++b)
          for (std::size_t c = 0; c < ext[2]; ++c)
            for (std::size_t d = 0; d < ext[3]; ++d)
              out(a, b, c, d) =
                  V_[(((sr[0].off + a) * nmo_ + (sr[1].off + b)) * nmo_ +
                      (sr[2].off + c)) *
                         nmo_ +
                     (sr[3].off + d)];
    } else if (lbl == "t") {  // cluster amplitudes / adjoint blocks
      std::string sp;
      for (auto&& i : idx) sp += space_letter(i);
      if (sp == "vo") {  // T1[a,i]
        for (std::size_t a = 0; a < nvirt_; ++a)
          for (std::size_t i = 0; i < nocc_; ++i)
            out(a, i) = T1_[a * nocc_ + i];
      } else if (sp == "ov") {  // T1^dagger[i,a]
        for (std::size_t i = 0; i < nocc_; ++i)
          for (std::size_t a = 0; a < nvirt_; ++a)
            out(i, a) = T1_[a * nocc_ + i];
      } else if (sp == "vvoo") {  // T2[a,b,i,j]
        std::copy(T2_.begin(), T2_.end(), out.begin());
      } else if (sp == "oovv") {  // T2^dagger[i,j,a,b] = T2[a,b,i,j]
        for (std::size_t i = 0; i < nocc_; ++i)
          for (std::size_t j = 0; j < nocc_; ++j)
            for (std::size_t a = 0; a < nvirt_; ++a)
              for (std::size_t b = 0; b < nvirt_; ++b)
                out(i, j, a, b) =
                    T2_[((a * nvirt_ + b) * nocc_ + i) * nocc_ + j];
      } else {
        throw std::runtime_error("ducc: unsupported cluster block '" + sp +
                                 "'");
      }
    } else {
      throw std::runtime_error("ducc: unsupported leaf tensor '" + lbl + "'");
    }
    return out;
  }

  // leaf-local indices to KEEP for a free leg of space letter `sl` (occ
  // leaf-local == SO index; virt leaf-local == SO - nocc_; complete leaf-local
  // == SO index).
  std::vector<std::size_t> active_gather(char sl) const {
    std::vector<std::size_t> g;
    if (sl == 'o') {
      g = active_occ_so_;
    } else if (sl == 'v') {
      g.reserve(active_virt_so_.size());
      for (std::size_t so : active_virt_so_) g.push_back(so - nocc_);
    } else if (sl == 'p') {
      g = active_occ_so_;
      g.insert(g.end(), active_virt_so_.begin(), active_virt_so_.end());
    }
    return g;
  }

  bool is_free_active(const Index& i) const {
    return active_ && free_labels_.count(std::wstring(i.full_label())) > 0;
  }

  // Gather-slice a full leaf to active on its free legs (contracted legs stay
  // full). A cheap leaf-sized copy so the downstream BTAS contraction runs at
  // active extents on the free legs.
  BTensorD gather_slice(const BTensorD& full,
                        const std::vector<Index>& idx) const {
    const std::size_t n = idx.size();
    std::vector<std::vector<std::size_t>> g(n);
    std::vector<std::size_t> oext(n), fext(n);
    for (std::size_t k = 0; k < n; ++k) {
      fext[k] = full.extent(k);
      if (is_free_active(idx[k])) {
        g[k] = active_gather(space_letter(idx[k]));
      } else {
        g[k].resize(fext[k]);
        std::iota(g[k].begin(), g[k].end(), std::size_t{0});
      }
      oext[k] = g[k].size();
    }
    BTensorD out{btas::Range{oext}};
    std::vector<std::size_t> fstr(n, 1);
    for (std::size_t k = n; k-- > 1;) fstr[k - 1] = fstr[k] * fext[k];
    std::vector<std::size_t> oi(n, 0);
    const std::size_t osize = out.size();
    for (std::size_t flat = 0; flat < osize; ++flat) {
      std::size_t foff = 0;
      for (std::size_t k = 0; k < n; ++k) foff += g[k][oi[k]] * fstr[k];
      out.data()[flat] = full.data()[foff];
      for (std::size_t k = n; k-- > 0;) {
        if (++oi[k] < oext[k]) break;
        oi[k] = 0;
      }
    }
    return out;
  }

 public:
  DataYielder(std::size_t nocc, std::size_t nvirt, const std::vector<double>& F,
              const std::vector<double>& V, const std::vector<double>& T1,
              const std::vector<double>& T2)
      : nocc_(nocc),
        nvirt_(nvirt),
        nmo_(nocc + nvirt),
        F_(F),
        V_(V),
        T1_(T1),
        T2_(T2) {}

  std::size_t nmo() const { return nmo_; }
  std::size_t nocc() const { return nocc_; }
  std::size_t nvirt() const { return nvirt_; }

  // b1' active-output controls. `active_occ_so` / `active_virt_so` are the
  // active spin-orbital indices in the occupied ([0,nocc_)) and virtual
  // ([nocc_,nmo_)) blocks respectively.
  void set_active(std::vector<std::size_t> active_occ_so,
                  std::vector<std::size_t> active_virt_so) {
    active_ = true;
    active_occ_so_ = std::move(active_occ_so);
    active_virt_so_ = std::move(active_virt_so);
  }
  bool active() const { return active_; }

  // Scatter map (evaluated position -> full spin-orbital index) for a free leg
  // of space letter `sl`; matches active_gather's ordering.
  std::vector<std::size_t> active_scatter(char sl) const {
    if (sl == 'o') return active_occ_so_;
    if (sl == 'v') return active_virt_so_;
    std::vector<std::size_t> g =
        active_occ_so_;  // 'p': active-occ ++ active-virt
    g.insert(g.end(), active_virt_so_.begin(), active_virt_so_.end());
    return g;
  }

  // Set the current term's free (residual-operator) leg labels.
  template <typename Range>
  void set_free_active(const Range& fi) const {
    free_labels_.clear();
    for (auto&& i : fi) free_labels_.insert(std::wstring(i.full_label()));
  }

  // Add the bare Hamiltonian H = F + 1/4 V over the full MO range.
  // expectation_value_impl returns only the VEV of the lone bare operator
  // (dropping its 1-/2-body parts), but Hbar = H + commutators contains H with
  // coefficient 1, so it is added back explicitly.
  void add_bare_H(BTensorD& Fbar, BTensorD& Vbar) const {
    const std::size_t n = nmo_;
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q) Fbar(p, q) += F_[p * n + q];
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t r = 0; r < n; ++r)
          for (std::size_t s = 0; s < n; ++s)
            Vbar(p, q, r, s) += 0.25 * V_[((p * n + q) * n + r) * n + s];
  }

  ResultPtr operator()(const Tensor& t) const {
    const std::string base = key_of(t);
    std::vector<Index> idx(t.const_braket_indices().begin(),
                           t.const_braket_indices().end());
    // The cache key carries the per-leg active-slice pattern so a leaf reused
    // across terms with different free-leg sets is not aliased.
    std::string ck = base;
    bool sliced = false;
    if (active_) {
      ck += '#';
      for (auto&& i : idx) {
        const bool fa = is_free_active(i);
        ck += fa ? 'A' : '-';
        sliced = sliced || fa;
      }
    }
    if (auto it = cache_.find(ck); it != cache_.end()) return it->second;
    BTensorD blk = make_block(t, base);
    if (sliced) blk = gather_slice(blk, idx);
    auto res = eval_result<ResultTensorBTAS<BTensorD>>(std::move(blk));
    cache_.emplace(ck, res);
    return res;
  }

  ResultPtr operator()(sequant::meta::can_evaluate auto const& node) const {
    if (node->result_type() == ResultType::Tensor)
      return (*this)(node->expr()->template as<Tensor>());
    return eval_result<ResultScalar<double>>(
        node->as_constant().template value<double>());
  }
};

// Sum the fully-contracted (scalar) terms of a Wick result via BTAS eval.
double eval_scalar_sum(const ExprPtr& nh, const DataYielder& yield) {
  SEQUANT_PRAGMA_IGNORE_DEPRECATED_BEGIN
  auto eval_term = [&](const ExprPtr& term) -> double {
    if (!term) return 0.0;
    if (term.is<Constant>()) return term.as<Constant>().value<double>();
    auto node = binarize<EvalExprBTAS>(sequant::optimize(term));
    return evaluate(node, container::svector<long>{}, yield)->get<double>();
  };
  auto is_scalar = [](const ExprPtr& term) {
    return collect_fnops(term).empty();
  };
  double total = 0.0;
  if (nh.is<Sum>()) {
    for (const auto& t : nh.as<Sum>().summands())
      if (is_scalar(t)) total += eval_term(t);
  } else if (nh && is_scalar(nh)) {
    total += eval_term(nh);
  }
  SEQUANT_PRAGMA_IGNORE_DEPRECATED_END
  return total;
}

// Evaluate each residual term's coefficient with the residual operator's legs
// as target indices, accumulating the 1-body (Fbar) and 2-body (Vbar) effective
// Hamiltonian over the full MO range.
void assemble_effective_H(const ExprPtr& nh, const DataYielder& yield,
                          BTensorD& Fbar, BTensorD& Vbar) {
  const std::size_t nocc = yield.nocc(), nvirt = yield.nvirt();
  SEQUANT_PRAGMA_IGNORE_DEPRECATED_BEGIN
  auto process = [&](const ExprPtr& term) {
    auto ops = collect_fnops(term);
    if (ops.size() != 1) return;  // scalar or (unexpected) multi-operator
    const auto& o = *ops.front();
    const std::size_t r = o.ncreators();
    if (r < 1 || r > 2) return;  // DUCC keeps <=2-body
    container::svector<Index> fi;
    for (auto&& c : o.creators()) fi.push_back(c.index());
    for (auto&& a : o.annihilators()) fi.push_back(a.index());
    // b1': restrict this term's free (output) legs to the active block so the
    // BTAS contraction runs at active extents; the internal sums stay full.
    if (yield.active()) yield.set_free_active(fi);
    auto node = binarize<EvalExprBTAS>(sequant::optimize(coeff_of(term)));
    container::svector<long> target =
        EvalExprBTAS::index_hash(fi) | ranges::to<container::svector<long>>;
    auto blk = evaluate(node, target, yield)->get<BTensorD>();
    // Map each free leg's evaluated positions to full spin-orbital indices: the
    // active spin-orbital list under b1', else the full space range.
    std::vector<std::vector<std::size_t>> amap;
    amap.reserve(fi.size());
    for (auto&& idx : fi) {
      if (yield.active()) {
        amap.push_back(yield.active_scatter(space_letter(idx)));
      } else {
        const auto s = space_range(idx, nocc, nvirt);
        std::vector<std::size_t> m(s.ext);
        std::iota(m.begin(), m.end(), s.off);
        amap.push_back(std::move(m));
      }
    }
    if (r == 1) {
      for (std::size_t a = 0; a < amap[0].size(); ++a)
        for (std::size_t b = 0; b < amap[1].size(); ++b)
          Fbar(amap[0][a], amap[1][b]) += blk(a, b);
    } else {
      for (std::size_t a = 0; a < amap[0].size(); ++a)
        for (std::size_t b = 0; b < amap[1].size(); ++b)
          for (std::size_t c = 0; c < amap[2].size(); ++c)
            for (std::size_t d = 0; d < amap[3].size(); ++d)
              Vbar(amap[0][a], amap[1][b], amap[2][c], amap[3][d]) +=
                  blk(a, b, c, d);
    }
  };
  if (nh.is<Sum>())
    for (const auto& t : nh.as<Sum>().summands()) process(t);
  else if (nh)
    process(nh);
  SEQUANT_PRAGMA_IGNORE_DEPRECATED_END
}

// physical antisymmetric two-body <PQ||RS> from the raw-canonical Vbar:
//   A[V]_pqrs = V_pqrs - V_qprs - V_pqsr + V_qpsr
// (applied to an already-antisymmetric tensor it yields 4x that tensor, so the
// bare 1/4 V contribution recovers exactly V; the residual raw-canonical blocks
// become their physical antisymmetrization). Matches the reference
// wicked_ducc_common.assemble_active_hamiltonian.
std::vector<double> asym4(const std::vector<double>& V, std::size_t n) {
  std::vector<double> A(V.size());
  auto id = [n](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    return ((p * n + q) * n + r) * n + s;
  };
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t r = 0; r < n; ++r)
        for (std::size_t s = 0; s < n; ++s)
          A[id(p, q, r, s)] = V[id(p, q, r, s)] - V[id(q, p, r, s)] -
                              V[id(p, q, s, r)] + V[id(q, p, s, r)];
  return A;
}

// BCH-dress the spin-orbital F/V/scalar for DUCC level>0. sigma is built from
// the EXTERNAL cluster amplitudes only (the all-active T block is zeroed), so
// the transform folds external correlation into the active space. Returns a
// copy of @p data with F/V/scalar replaced by the level-@p level DUCC effective
// values over the full spin-orbital space (assemble_active_hamiltonian then
// restricts to the active space).
SpinOrbitalData ducc_bch_dress(const SpinOrbitalData& data, int level,
                               const std::vector<std::size_t>& active_so) {
  ensure_sequant_setup();
  const std::size_t nso = data.nso, nocc = data.nocc_so, nvirt = data.nvir_so;

  // T_ext: zero the all-active block of T1/T2 (occ position i -> spin-orbital
  // i; virtual position a -> spin-orbital nocc + a).
  std::vector<double> T1 = data.T1, T2 = data.T2;
  std::vector<char> is_active(nso, 0);
  for (std::size_t so : active_so)
    if (so < nso) is_active[so] = 1;
  for (std::size_t a = 0; a < nvirt; ++a)
    for (std::size_t i = 0; i < nocc; ++i)
      if (is_active[nocc + a] && is_active[i]) T1[a * nocc + i] = 0.0;
  for (std::size_t a = 0; a < nvirt; ++a)
    for (std::size_t b = 0; b < nvirt; ++b)
      for (std::size_t i = 0; i < nocc; ++i)
        for (std::size_t j = 0; j < nocc; ++j)
          if (is_active[nocc + a] && is_active[nocc + b] && is_active[i] &&
              is_active[j])
            T2[((a * nvirt + b) * nocc + i) * nocc + j] = 0.0;

  // Symbolic DUCC F-split BCH over the whole spin-orbital space.
  auto H = mbpt::H();
  auto Top = mbpt::T(2);
  auto sigma = Top - adjoint(Top);
  auto hbar =
      ducc_fsplit_bch(static_cast<std::size_t>(level), H, mbpt::F(), sigma);

  // Partial Wick (full_contractions=false; unscreened -- DUCC connectedness
  // comes from the commutator structure, not Wick screening), truncated to
  // <=2-body by the post-filter, then simplified.
  auto nh = mbpt::op::expectation_value_impl(
      hbar, /*connect=*/{}, /*avoid=*/{}, /*use_topology=*/true,
      /*screen=*/false, /*skip_clone=*/false, /*full_contractions=*/false);
  nh = keep_up_to_body(nh, 2);
  simplify(nh);

  // BTAS numeric evaluation. b1' active-output optimization: each residual
  // term's free (output) legs are sliced to the active block so the BTAS
  // contractions run at active extents (real FLOP reduction), while the
  // internal sums stay full (external correlation, incl. frozen core). The
  // active block is scattered into the full-space Fbar/Vbar -- all that
  // assemble_active_hamiltonian reads.
  DataYielder yield(nocc, nvirt, data.F, data.V, T1, T2);
  std::vector<std::size_t> active_occ_so, active_virt_so;
  for (std::size_t so : active_so) {
    if (so < nocc)
      active_occ_so.push_back(so);
    else
      active_virt_so.push_back(so);
  }
  yield.set_active(std::move(active_occ_so), std::move(active_virt_so));
  const double scalar_bch = eval_scalar_sum(nh, yield);
  BTensorD Fbar{btas::Range{nso, nso}};
  BTensorD Vbar{btas::Range{nso, nso, nso, nso}};
  Fbar.fill(0.0);
  Vbar.fill(0.0);
  assemble_effective_H(nh, yield, Fbar, Vbar);
  yield.add_bare_H(Fbar, Vbar);

  // Pack into the extraction convention: F = Fock, V = antisymmetrized
  // <PQ||RS>, scalar = reference energy + BCH scalar correction.
  SpinOrbitalData dressed = data;
  dressed.scalar = data.scalar + scalar_bch;
  dressed.F.assign(Fbar.begin(), Fbar.end());
  dressed.V = asym4(std::vector<double>(Vbar.begin(), Vbar.end()), nso);
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
    const auto [aa, ab] = active_orbitals->get_active_space_indices();
    active_a.assign(aa.begin(), aa.end());
    active_b.assign(ab.begin(), ab.end());
  } else {
    const std::size_t nmo = data.nso / 2;
    active_a.resize(nmo);
    active_b.resize(nmo);
    for (std::size_t p = 0; p < nmo; ++p) active_a[p] = active_b[p] = p;
  }

  // Step 2b: BCH dressing for levels 1-2 -- bar{H} = e^{-sigma} H e^{sigma}
  // with sigma = T_ext - T_ext^dagger (external cluster amplitudes only),
  // evaluated symbolically + numerically with SeQuant/BTAS. Level 0 skips this
  // and uses the bare spin-orbital data directly.
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

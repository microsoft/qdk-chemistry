// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Constructor-level tests for the second-order Schrieffer-Wolff downfold
// (`swpt2`), exercised through the `EffectiveHamiltonianConstructor` factory.

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <map>
#include <numeric>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <set>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2.hpp"
#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/swpt2_kernel.hpp"
#include "testing_utilities_swpt2.hpp"
#include "ut_common.hpp"

namespace {
using qdk::chemistry::algorithms::ActiveSpaceSelectorFactory;
using qdk::chemistry::algorithms::DynamicalCorrelationCalculatorFactory;
using qdk::chemistry::algorithms::EffectiveHamiltonianConstructorFactory;
using qdk::chemistry::algorithms::HamiltonianConstructorFactory;
using qdk::chemistry::algorithms::LocalizerFactory;
using qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory;
using qdk::chemistry::algorithms::ScfSolverFactory;

namespace sw = qdk::chemistry::algorithms::microsoft::swpt2;

// A direct full-CI ground-state solver on the interleaved spin-orbital tensors.
// It independently verifies the emitted effective Hamiltonian below.
struct Ladder {
  std::uint64_t mask;
  int sign;
  bool ok;
};
Ladder apply_ladder(std::uint64_t mask, int orb, bool creation) {
  const std::uint64_t bit = std::uint64_t{1} << orb;
  const bool occupied = (mask & bit) != 0;
  if (creation == occupied) return {0, 0, false};  // Pauli
  const int below = std::popcount(mask & (bit - 1));
  return {mask ^ bit, (below & 1) ? -1 : 1, true};
}
Ladder apply_one(std::uint64_t mask, int P, int Q) {
  const Ladder a = apply_ladder(mask, Q, false);
  if (!a.ok) return a;
  const Ladder b = apply_ladder(a.mask, P, true);
  if (!b.ok) return b;
  return {b.mask, a.sign * b.sign, true};
}
Ladder apply_two(std::uint64_t mask, int P, int Q, int R, int S) {
  Ladder r = apply_ladder(mask, S, false);
  if (!r.ok) return r;
  int sign = r.sign;
  r = apply_ladder(r.mask, R, false);
  if (!r.ok) return r;
  sign *= r.sign;
  r = apply_ladder(r.mask, Q, true);
  if (!r.ok) return r;
  sign *= r.sign;
  r = apply_ladder(r.mask, P, true);
  if (!r.ok) return r;
  return {r.mask, sign * r.sign, true};
}

// Lowest eigenvalue of  e0 + sum_PQ f_PQ a^dag_P a_Q
//                          + (1/4) sum_PQRS v_PQRS a^dag_P a^dag_Q a_R a_S,
// restricted to `orbs` with `na` alpha (even index) / `nb` beta (odd)
// electrons.
double fci_ground_energy(double e0, const Eigen::MatrixXd& f,
                         const Eigen::VectorXd& v, int n_so,
                         const std::vector<int>& orbs, int na, int nb) {
  const int m = static_cast<int>(orbs.size());
  std::vector<std::uint64_t> basis;
  for (std::uint64_t sub = 0; sub < (std::uint64_t{1} << m); ++sub) {
    std::uint64_t mask = 0;
    int ca = 0, cb = 0;
    for (int k = 0; k < m; ++k) {
      if (sub & (std::uint64_t{1} << k)) {
        const int so = orbs[k];
        mask |= std::uint64_t{1} << so;
        ((so % 2 == 0) ? ca : cb)++;  // even index = alpha, odd = beta
      }
    }
    if (ca == na && cb == nb) basis.push_back(mask);
  }
  const int D = static_cast<int>(basis.size());
  std::map<std::uint64_t, int> index;
  for (int i = 0; i < D; ++i) index[basis[i]] = i;

  Eigen::MatrixXd Hmat = Eigen::MatrixXd::Zero(D, D);
  for (int col = 0; col < D; ++col) {
    const std::uint64_t ket = basis[col];
    Hmat(col, col) += e0;
    for (int P : orbs)
      for (int Q : orbs) {
        const double fpq = f(P, Q);
        if (fpq == 0.0) continue;
        const Ladder r = apply_one(ket, P, Q);
        if (!r.ok) continue;
        const auto it = index.find(r.mask);
        if (it != index.end()) Hmat(it->second, col) += fpq * r.sign;
      }
    for (int P : orbs)
      for (int Q : orbs)
        for (int R : orbs)
          for (int S : orbs) {
            const double vpqrs = v(sw::idx4(P, Q, R, S, n_so));
            if (vpqrs == 0.0) continue;
            const Ladder r = apply_two(ket, P, Q, R, S);
            if (!r.ok) continue;
            const auto it = index.find(r.mask);
            if (it != index.end())
              Hmat(it->second, col) += 0.25 * vpqrs * r.sign;
          }
  }
  Hmat = 0.5 * (Hmat + Hmat.transpose()).eval();  // H is Hermitian by build
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(Hmat);
  return es.eigenvalues()(0);
}

TEST(SchriefferWolffPT2Test, FactoryRegistration) {
  const auto available = EffectiveHamiltonianConstructorFactory::available();
  EXPECT_EQ(available, std::vector<std::string>{"qdk_swpt2"});

  auto ctor = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  ASSERT_NE(ctor, nullptr);
  EXPECT_EQ(ctor->name(), "qdk_swpt2");
  EXPECT_EQ(ctor->type_name(), "effective_hamiltonian_constructor");

  EXPECT_EQ(EffectiveHamiltonianConstructorFactory::create()->name(),
            "qdk_swpt2");
}

TEST(SchriefferWolffPT2Test, SettingsKnobs) {
  auto ctor = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  auto& settings = ctor->settings();
  // denominator knobs: sigma^2 regularization is on by default at this layer
  EXPECT_DOUBLE_EQ(settings.get<double>("regularizer_sigma2"), 1.0);
  EXPECT_TRUE(settings.get<bool>("semicanonicalize"));
  EXPECT_DOUBLE_EQ(settings.get<double>("max_folded_occupation_deviation"),
                   0.5);
  EXPECT_ANY_THROW(settings.set("regularizer_sigma2", -1.0));

  auto invalid = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  EXPECT_THROW(invalid->run(nullptr, nullptr,
                            testing::restricted_index_set(
                                1, std::vector<std::size_t>{0})),
               std::invalid_argument);
}

// End-to-end smoke test: SCF -> window Hamiltonian over W -> CAS reference over
// P -> swpt2 downfold -> CAS on the emitted effective Hamiltonian. Confirms the
// full data-layer pipeline runs and produces a sensible, below-HF effective
// Hamiltonian that actually differs from the bare active-space one.
//
// Two things are asserted quantitatively: (1) the data-layer emission+
// consumption path is faithful -- MACIS on the emitted effective Hamiltonian
// reproduces an independent full-CI on the *same* integrals to machine
// precision (so no 8-fold symmetrization / container change is needed); and
// (2) with regularization on by default (regularizer_sigma2=1.0) the downfold
// recovers
// correlation below the bare active-space CAS, whereas an explicitly *bare*
// (unregularized) run does not -- this particular window splits two
// near-degenerate virtuals (active LUMO orbital 5, folded external orbital 6)
// and the resulting small pp-ladder denominator makes bare second-order PT
// overshoot -- a classic intruder tamed by the flow regularizer, not a bug in
// the container/solver or the denominators.
TEST(SchriefferWolffPT2Test, DownfoldRunsEndToEndWater) {
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();

  const std::vector<size_t> core = {0, 1, 2};
  const std::vector<size_t> active = {3, 4, 5};
  const std::vector<size_t> window = {3, 4, 5, 6};
  const auto P_set = testing::restricted_index_set(
      orbitals->get_num_molecular_orbitals(), active);

  auto ham = HamiltonianConstructorFactory::create();
  auto p_orbitals = testing::with_active_space(orbitals, active, core);
  auto H_P = ham->run(p_orbitals);
  auto cas_ref = MultiConfigurationCalculatorFactory::create("macis_cas");
  cas_ref->settings().set("calculate_one_rdm", true);
  auto [E_bare, reference] = cas_ref->run(H_P, 2, 2);

  // window Hamiltonian over W = P + one virtual to fold
  auto w_orbitals = testing::with_active_space(orbitals, window, core);
  auto H_window = ham->run(w_orbitals);

  // Every reference-active orbital must be represented in the window. A
  // missing trailing orbital used to be silently dropped by the index map.
  auto incomplete_w_orbitals =
      testing::with_active_space(orbitals, {3, 4, 6}, core);
  auto H_incomplete = ham->run(incomplete_w_orbitals);
  auto swpt2_invalid =
      EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  EXPECT_THROW(swpt2_invalid->run(reference, H_incomplete, P_set),
               std::invalid_argument);

  // downfold the window onto P, then CAS on the effective Hamiltonian
  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  auto H_eff = swpt2->run(reference, H_window, P_set);
  ASSERT_NE(H_eff, nullptr);
  auto cas_sw = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_sw, wfn_sw] = cas_sw->run(H_eff, 2, 2);

  EXPECT_TRUE(std::isfinite(E_sw));
  EXPECT_LT(E_sw, E_hf);  // captures correlation below Hartree-Fock
  EXPECT_LT(E_sw,
            E_bare);  // regularized-by-default downfold recovers below CAS

  // the downfold actually dressed the two-body interaction (nonzero effect)
  const auto [ge, ge_ab, ge_bb] = H_eff->get_two_body_integrals();
  const auto [gb, gb_ab, gb_bb] = H_P->get_two_body_integrals();
  EXPECT_GT((ge - gb).norm(), 1e-6);

  // The emitted effective two-body is only 4-fold symmetric (the bra-swap
  // (pq|rs)=(qp|rs) is broken by the downfold). This asserts that the
  // data-layer consumption path is faithful anyway: MACIS on H_eff must
  // reproduce, to machine precision, an independent full-CI evaluation of the
  // *same* emitted integrals. It confirms CanonicalFourCenter stores the dense
  // n^4 block and that MACIS relies only on particle-exchange symmetry
  // (pq|rs)=(rs|pq), which the effective tensor preserves -- so no 8-fold
  // symmetrization is required.
  {
    const auto [T_a, T_b] = H_eff->get_one_body_integrals();
    const int m = static_cast<int>(T_a.rows());
    const auto H_so = testing::build_spin_orbital_tensors(
        T_a, T_a, ge, ge, ge, H_eff->get_core_energy(), m);
    std::vector<int> orbs(2 * m);
    for (int i = 0; i < 2 * m; ++i) orbs[i] = i;
    const double E_kernel_fci = fci_ground_energy(
        H_so.core_energy, H_so.one_body, H_so.two_body, 2 * m, orbs, 2, 2);
    EXPECT_NEAR(E_sw, E_kernel_fci, 1e-9);
  }

  // Contrast: disabling regularization (bare second-order PT) hits the
  // small-denominator near-intruder and fails to recover. The active space
  // keeps the active LUMO (orbital 5) while the folded external orbital 6 is a
  // *near-degenerate* virtual (eps_5=+0.47, eps_6=+0.59), so the pp-ladder
  // dressing of the active (5,5) pair has denominator 2(eps_6-eps_5) ~ 0.24 Eh
  // and bare PT2 overshoots. The default sigma^2 regularizer is what tames it.
  // (Root cause is the partition splitting two near-degenerate virtuals across
  // active/external -- NOT the container/solver, and NOT the MP denominators,
  // which are exact for this canonical closed-shell reference.)
  auto swpt2_bare = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  swpt2_bare->settings().set("regularizer_sigma2", 0.0);
  auto H_eff_bare = swpt2_bare->run(reference, H_window, P_set);
  auto cas_bare = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_sw_bare, wfn_bare] = cas_bare->run(H_eff_bare, 2, 2);
  EXPECT_GT(E_sw_bare, E_sw);  // regularization lowers the energy vs bare PT2
}

// The MP2 limit, where the downfold has a closed-form answer to compare
// against. Put every occupied orbital in the active space, leave the inactive
// set empty, fold all the virtuals, and switch the regularizer off. Nothing in
// H_BD can change the virtual occupation, so <HF|H_BD|HF> is E_HF; and
// Brillouin kills the singles channel of <HF| 1/2 [S, H_OD] |HF>, leaving
// exactly the doubles sum that defines MP2. Rank-3 terms of the commutator do
// contribute at ten electrons, but the reference density here is the idempotent
// HF one, for which the fold onto two-body is exact on the reference
// determinant. The active space comes out completely filled, so its CI space is
// a single determinant and the emitted Hamiltonian's ground state *is* that
// expectation value.
//
// This is the one test in the suite that checks the downfold against physics
// rather than against a second expansion of the same commutator: the reference
// number comes from the repository's MP2 calculator, which shares only the SCF
// and integral layers with this code path.
//
// Bare denominators log a large-amplitude warning on this window, and MP2 does
// not, which is consistent rather than contradictory: the offending channel is
// semi-internal (eps_5 + eps_1 - 2 eps_4 = 0.043 Eh here), and MP2 has no such
// channel -- its own oo->vv denominators bottom out at 1.7 Eh. That channel is
// blocked on the reference determinant, so it cannot move the number compared
// here, but it is a real feature of the emitted operator. The agreement below
// therefore constrains the doubles channel and the fold; it says nothing about
// the semi-internal ones, which the symbolic term table covers instead.
TEST(SchriefferWolffPT2Test, ReproducesMp2WhenEveryOccupiedOrbitalIsActive) {
  namespace qcd = qdk::chemistry::data;
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();
  const size_t norb = orbitals->get_num_molecular_orbitals();
  const auto [na, nb] = wfn_hf->get_active_num_electrons();
  ASSERT_EQ(na, nb) << "the closed-shell argument above assumes a singlet";

  std::vector<size_t> occupied(na), window(norb);
  std::iota(occupied.begin(), occupied.end(), size_t{0});
  std::iota(window.begin(), window.end(), size_t{0});
  ASSERT_LT(occupied.size(), window.size()) << "nothing would be folded";

  // Single-determinant HF reference over the occupied block, no inactive space.
  auto ref_orbs = testing::with_active_space(orbitals, occupied, {});
  auto reference = std::make_shared<qcd::Wavefunction>(
      std::make_unique<qcd::StateVectorContainer>(
          qcd::Configuration::from_spin_half_string(std::string(na, '2')),
          ref_orbs));

  auto ham = HamiltonianConstructorFactory::create();
  auto H_window = ham->run(testing::with_active_space(orbitals, window, {}));
  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  swpt2->settings().set("regularizer_sigma2", 0.0);  // bare MP denominators
  auto H_eff = swpt2->run(reference, H_window,
                          testing::restricted_index_set(norb, occupied));
  ASSERT_NE(H_eff, nullptr);

  const auto [T_a, T_b] = H_eff->get_one_body_integrals();
  const auto [g_eff, g_eff_ab, g_eff_bb] = H_eff->get_two_body_integrals();
  const int m = static_cast<int>(T_a.rows());
  const auto H_so = testing::build_spin_orbital_tensors(
      T_a, T_a, g_eff, g_eff, g_eff, H_eff->get_core_energy(), m);
  std::vector<int> orbs(2 * m);
  std::iota(orbs.begin(), orbs.end(), 0);
  const double E_downfold =
      fci_ground_energy(H_so.core_energy, H_so.one_body, H_so.two_body, 2 * m,
                        orbs, static_cast<int>(na), static_cast<int>(nb));

  auto H_full = ham->run(orbitals);
  auto ansatz = std::make_shared<qcd::Ansatz>(*H_full, *wfn_hf);
  auto [E_mp2, mp2_ket, mp2_bra] =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator")
          ->run(ansatz);

  EXPECT_LT(E_mp2, E_hf - 1e-3) << "the correlation being matched is not noise";
  EXPECT_NEAR(E_downfold, E_mp2, 1e-9);

  // The identity above is not free: at ten active electrons the rank-3 part of
  // the commutator has a large expectation in the reference, and only the fold
  // onto the reference density recovers it. Dropping it moves the answer by
  // several Hartree, so the agreement above is a real constraint on the fold.
  auto unfolded = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  unfolded->settings().set("regularizer_sigma2", 0.0);
  unfolded->settings().set("fold_above_two_body", false);
  auto H_unfolded = unfolded->run(
      reference, H_window, testing::restricted_index_set(norb, occupied));
  const auto [U_a, U_b] = H_unfolded->get_one_body_integrals();
  const auto [g_u, g_u_ab, g_u_bb] = H_unfolded->get_two_body_integrals();
  const auto U_so = testing::build_spin_orbital_tensors(
      U_a, U_a, g_u, g_u, g_u, H_unfolded->get_core_energy(), m);
  const double E_unfolded =
      fci_ground_energy(U_so.core_energy, U_so.one_body, U_so.two_body, 2 * m,
                        orbs, static_cast<int>(na), static_cast<int>(nb));
  EXPECT_GT(std::abs(E_unfolded - E_mp2), 1.0);
}

// A mean-field HF reference (single determinant, no active 1-RDM) must work:
// the downfold reads active orbital occupations directly from the reference
// determinant. Mirrors the SCF -> active_space_selector -> downfold user flow.
TEST(SchriefferWolffPT2Test, AcceptsMeanFieldHfReference) {
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();

  // HF reference with a small active space; no 1-RDM is computed for it.
  auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
  selector->settings().set("num_active_electrons", 2);
  selector->settings().set("num_active_orbitals", 2);
  auto reference = selector->run(wfn_hf);
  ASSERT_FALSE(reference->has_active_one_rdm());
  const auto [na, nb] = reference->get_active_num_electrons();

  // window = active P + the lowest orbital outside P and the frozen core, so a
  // real external orbital is folded while the window stays small (cheap).
  namespace qcd = qdk::chemistry::data;
  auto ref_orbs = reference->get_orbitals();
  const auto P =
      qcd::spin_channel_indices(ref_orbs->active_indices(), qcd::axes::alpha());
  const auto core = qcd::spin_channel_indices(ref_orbs->inactive_indices(),
                                              qcd::axes::alpha());
  const size_t norb = orbitals->get_num_molecular_orbitals();
  std::set<size_t> used(P.begin(), P.end());
  used.insert(core.begin(), core.end());
  std::vector<size_t> window(P.begin(), P.end());
  for (size_t i = 0; i < norb; ++i)
    if (!used.count(i)) {
      window.push_back(i);
      break;
    }
  std::sort(window.begin(), window.end());
  const std::vector<size_t> core_vec(core.begin(), core.end());

  auto ham = HamiltonianConstructorFactory::create();
  auto H_window =
      ham->run(testing::with_active_space(orbitals, window, core_vec));
  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  auto H_eff =
      swpt2->run(reference, H_window, testing::restricted_index_set(norb, P));
  ASSERT_NE(H_eff, nullptr);

  auto cas_sw = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_sw, wfn_sw] = cas_sw->run(H_eff, static_cast<unsigned int>(na),
                                    static_cast<unsigned int>(nb));
  EXPECT_TRUE(std::isfinite(E_sw));

  Eigen::MatrixXd legacy_density(2, 2);
  legacy_density << 1.0, 0.2, 0.2, 1.0;
  Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(1);
  qcd::Wavefunction::DeterminantVector determinants{
      qcd::Configuration::from_spin_half_string("20")};
  auto legacy_reference = std::make_shared<qcd::Wavefunction>(
      std::make_unique<qcd::StateVectorContainer>(
          coefficients, determinants, ref_orbs,
          std::optional<qcd::StateVectorContainer::MatrixVariant>{
              legacy_density},
          std::nullopt));
  ASSERT_TRUE(legacy_reference->has_one_rdm_spin_traced());
  ASSERT_FALSE(legacy_reference->has_active_one_rdm());
  EXPECT_NE(swpt2->run(legacy_reference, H_window,
                       testing::restricted_index_set(norb, P)),
            nullptr);
}

// The kept space P can be given explicitly via the `active_indices` setting,
// independently of the reference wavefunction's active space, as long as every
// folded orbital is closed-shell in the reference.
TEST(SchriefferWolffPT2Test, CustomActiveSpaceOverridesReference) {
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();

  const std::vector<size_t> core = {0, 1, 2};
  const std::vector<size_t> active = {3, 4};  // reference active space
  const std::vector<size_t> window = {3, 4, 5, 6};

  auto ham = HamiltonianConstructorFactory::create();
  auto cas_ref = MultiConfigurationCalculatorFactory::create("macis_cas");
  cas_ref->settings().set("calculate_one_rdm", true);
  auto [E_bare, reference] = cas_ref->run(
      ham->run(testing::with_active_space(orbitals, active, core)), 1, 1);
  auto H_window = ham->run(testing::with_active_space(orbitals, window, core));

  namespace qcd = qdk::chemistry::data;

  // (1) Custom P that differs from the reference active space {3,4}: also keep
  // the empty external orbital 5, so the folded rest {6} stays closed-shell.
  // The result must emit over exactly P and be consumable by MACIS.
  const std::vector<std::size_t> P_custom = {3, 4, 5};
  const auto p_indices_custom = testing::restricted_index_set(
      orbitals->get_num_molecular_orbitals(), P_custom);
  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  auto H_eff = swpt2->run(reference, H_window, p_indices_custom);
  ASSERT_NE(H_eff, nullptr);

  const auto [T_a, T_b] = H_eff->get_one_body_integrals();
  EXPECT_EQ(T_a.rows(), static_cast<Eigen::Index>(P_custom.size()));
  // The output contract requires the emitted active set to be the caller's
  // p_indices, in every spin channel and not just the alpha one.
  const auto emitted = H_eff->get_orbitals()->active_indices();
  for (const auto& spin : {qcd::axes::alpha(), qcd::axes::beta()})
    EXPECT_EQ(qcd::spin_channel_indices(emitted, spin),
              qcd::spin_channel_indices(p_indices_custom, spin));
  const auto emitted_active =
      qcd::spin_channel_indices(emitted, qcd::axes::alpha());
  EXPECT_EQ(
      std::vector<std::size_t>(emitted_active.begin(), emitted_active.end()),
      P_custom);

  auto cas = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_sw, wfn_sw] = cas->run(H_eff, 1, 1);
  EXPECT_TRUE(std::isfinite(E_sw));

  // (2) Custom P = the whole window => empty external => the downfold is the
  // identity, so the emitted operator must reproduce the bare window
  // Hamiltonian (validates the custom-P emission path end to end).
  const std::vector<std::size_t> P_full = {3, 4, 5, 6};
  auto swpt2_full = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  auto H_identity =
      swpt2_full->run(reference, H_window,
                      testing::restricted_index_set(
                          orbitals->get_num_molecular_orbitals(), P_full));
  const auto [ge, ge_ab, ge_bb] = H_identity->get_two_body_integrals();
  const auto [gw, gw_ab, gw_bb] = H_window->get_two_body_integrals();
  EXPECT_LT((ge - gw).norm(), 1e-9);
  EXPECT_NEAR(H_identity->get_core_energy(), H_window->get_core_energy(), 1e-9);

  // (3) Window that spans the reference core: the core orbitals {0,1,2} appear
  // both as folded window orbitals and in the reference inactive set, so the
  // emitted inactive index set must be deduplicated (strictly increasing).
  auto H_all =
      ham->run(testing::with_active_space(orbitals, {0, 1, 2, 3, 4, 5, 6}, {}));
  auto swpt2_core = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  auto H_eff_core = swpt2_core->run(
      reference, H_all,
      testing::restricted_index_set(orbitals->get_num_molecular_orbitals(),
                                    std::vector<std::size_t>{3, 4}));
  ASSERT_NE(H_eff_core, nullptr);
  const auto core_active = qcd::spin_channel_indices(
      H_eff_core->get_orbitals()->active_indices(), qcd::axes::alpha());
  EXPECT_EQ(std::vector<std::size_t>(core_active.begin(), core_active.end()),
            (std::vector<std::size_t>{3, 4}));
  const auto core_inactive = qcd::spin_channel_indices(
      H_eff_core->get_orbitals()->inactive_indices(), qcd::axes::alpha());
  EXPECT_EQ(
      std::vector<std::size_t>(core_inactive.begin(), core_inactive.end()),
      (std::vector<std::size_t>{0, 1, 2}));

  // (4) A spin-dependent P is rejected rather than silently reduced to its
  // alpha channel.
  auto swpt2_spin = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  EXPECT_THROW(swpt2_spin->run(reference, H_window,
                               testing::unrestricted_index_set(
                                   orbitals->get_num_molecular_orbitals(),
                                   {3, 4, 5}, {3, 4, 6})),
               std::invalid_argument);
}

// Folding an orbital whose correlated natural occupation is not exactly 2 or 0.
// The occupation is rounded to the nearer integer; the total electron count is
// still exact because the active space receives whatever the folded orbitals do
// not take. `max_folded_occupation_deviation` bounds how much rounding error is
// accepted, and setting it to zero restores the strict integer-occupation rule.
TEST(SchriefferWolffPT2Test, FoldsFractionallyOccupiedExternalOrbital) {
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();
  const auto num_mo = orbitals->get_num_molecular_orbitals();

  auto ham = HamiltonianConstructorFactory::create();
  auto cas_ref = MultiConfigurationCalculatorFactory::create("macis_cas");
  cas_ref->settings().set("calculate_one_rdm", true);
  // HOMO/LUMO CAS(2,2): natural occupations near 2 and 0 but not exactly, so
  // orbital 4 is a fractionally occupied orbital available to fold.
  auto [E_bare, reference] = cas_ref->run(
      ham->run(testing::with_active_space(orbitals, {4, 5}, {0, 1, 2, 3})), 1,
      1);

  // Window {3,4,5,6} holds 2 + n_4 + n_5 + 0 = 4 electrons. Folding the
  // fractionally occupied orbital 4 as doubly occupied and 6 as empty leaves
  // exactly 2 electrons for P = {3, 5}.
  auto H_window =
      ham->run(testing::with_active_space(orbitals, {3, 4, 5, 6}, {0, 1, 2}));
  const auto p_indices =
      testing::restricted_index_set(num_mo, std::vector<std::size_t>{3, 5});

  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  auto H_eff = swpt2->run(reference, H_window, p_indices);
  ASSERT_NE(H_eff, nullptr);

  namespace qcd2 = qdk::chemistry::data;
  const auto emitted_active = qcd2::spin_channel_indices(
      H_eff->get_orbitals()->active_indices(), qcd2::axes::alpha());
  EXPECT_EQ(
      std::vector<std::size_t>(emitted_active.begin(), emitted_active.end()),
      (std::vector<std::size_t>{3, 5}));
  // reference core {0,1,2} plus the folded doubly-occupied orbital {4}
  const auto emitted_inactive = qcd2::spin_channel_indices(
      H_eff->get_orbitals()->inactive_indices(), qcd2::axes::alpha());
  EXPECT_EQ(std::vector<std::size_t>(emitted_inactive.begin(),
                                     emitted_inactive.end()),
            (std::vector<std::size_t>{0, 1, 2, 4}));

  // The derived active electron count (2) is the one the solver must be given.
  auto cas = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_sw, wfn_sw] = cas->run(H_eff, 1, 1);
  EXPECT_TRUE(std::isfinite(E_sw));

  // Demanding exactly integer folded occupations rejects the same partition.
  auto strict = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  strict->settings().set("max_folded_occupation_deviation", 0.0);
  EXPECT_THROW(strict->run(reference, H_window, p_indices),
               std::invalid_argument);
}

TEST(SchriefferWolffPT2Test, AcceptsRestrictedOpenShellHfReference) {
  auto hydroxyl = testing::create_oh_structure();
  auto scf = ScfSolverFactory::create();
  scf->settings().set("enable_gdm", false);
  scf->settings().set("method", std::string("hf"));
  scf->settings().set("scf_type", std::string("restricted"));
  auto [E_rohf, wfn_rohf] = scf->run(hydroxyl, 0, 2, "sto-3g");
  ASSERT_TRUE(wfn_rohf->get_orbitals()->is_restricted());

  auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
  selector->settings().set("num_active_electrons", 1);
  selector->settings().set("num_active_orbitals", 1);
  auto reference = selector->run(wfn_rohf);
  ASSERT_FALSE(reference->has_active_one_rdm());
  const auto [occ_a, occ_b] = reference->get_active_orbital_occupations();
  ASSERT_EQ(occ_a.size(), 1);
  ASSERT_EQ(occ_b.size(), 1);
  EXPECT_DOUBLE_EQ(occ_a(0) + occ_b(0), 1.0);

  namespace qcd = qdk::chemistry::data;
  const auto orbitals = wfn_rohf->get_orbitals();
  const auto ref_orbitals = reference->get_orbitals();
  const auto active = qcd::spin_channel_indices(ref_orbitals->active_indices(),
                                                qcd::axes::alpha());
  const auto inactive = qcd::spin_channel_indices(
      ref_orbitals->inactive_indices(), qcd::axes::alpha());
  std::set<size_t> used(active.begin(), active.end());
  used.insert(inactive.begin(), inactive.end());
  std::vector<size_t> window(active.begin(), active.end());
  for (size_t i = 0; i < orbitals->get_num_molecular_orbitals(); ++i)
    if (!used.count(i)) {
      window.push_back(i);
      break;
    }
  std::sort(window.begin(), window.end());

  auto ham = HamiltonianConstructorFactory::create();
  auto H_window = ham->run(testing::with_active_space(
      orbitals, window, std::vector<size_t>(inactive.begin(), inactive.end())));
  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  auto H_eff = swpt2->run(reference, H_window,
                          testing::restricted_index_set(
                              orbitals->get_num_molecular_orbitals(), active));
  ASSERT_NE(H_eff, nullptr);

  auto cas = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_sw, wfn_sw] = cas->run(H_eff, 1, 0);
  EXPECT_TRUE(std::isfinite(E_sw));
}

TEST(SchriefferWolffPT2Test,
     AcceptsCorrelatedCasOnRestrictedOpenShellOrbitals) {
  auto hydroxyl = testing::create_oh_structure();
  auto scf = ScfSolverFactory::create();
  scf->settings().set("enable_gdm", false);
  scf->settings().set("method", std::string("hf"));
  scf->settings().set("scf_type", std::string("restricted"));
  auto [E_rohf, wfn_rohf] = scf->run(hydroxyl, 0, 2, "sto-3g");
  const auto orbitals = wfn_rohf->get_orbitals();
  ASSERT_TRUE(orbitals->is_restricted());

  auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
  selector->settings().set("num_active_electrons", 3);
  selector->settings().set("num_active_orbitals", 2);
  auto selected = selector->run(wfn_rohf);
  const auto [na, nb] = selected->get_active_num_electrons();
  ASSERT_EQ(na + nb, 3);

  auto ham = HamiltonianConstructorFactory::create();
  auto H_P = ham->run(selected->get_orbitals());
  auto cas_reference = MultiConfigurationCalculatorFactory::create("macis_cas");
  cas_reference->settings().set("calculate_one_rdm", true);
  auto [E_cas, reference] = cas_reference->run(H_P, na, nb);
  ASSERT_TRUE(reference->has_active_one_rdm());

  namespace qcd = qdk::chemistry::data;
  const auto active = qcd::spin_channel_indices(
      selected->get_orbitals()->active_indices(), qcd::axes::alpha());
  const auto inactive = qcd::spin_channel_indices(
      selected->get_orbitals()->inactive_indices(), qcd::axes::alpha());
  std::set<size_t> used(active.begin(), active.end());
  used.insert(inactive.begin(), inactive.end());
  std::vector<size_t> window(active.begin(), active.end());
  for (size_t orbital = 0; orbital < orbitals->get_num_molecular_orbitals();
       ++orbital)
    if (!used.count(orbital)) {
      window.push_back(orbital);
      break;
    }
  ASSERT_GT(window.size(), active.size());
  std::sort(window.begin(), window.end());

  auto H_window = ham->run(testing::with_active_space(
      orbitals, window, std::vector<size_t>(inactive.begin(), inactive.end())));
  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  auto H_eff = swpt2->run(reference, H_window,
                          testing::restricted_index_set(
                              orbitals->get_num_molecular_orbitals(), active));
  ASSERT_NE(H_eff, nullptr);

  auto cas_effective = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_sw, wfn_sw] = cas_effective->run(H_eff, na, nb);
  EXPECT_TRUE(std::isfinite(E_sw));

  // The non-semicanonical path must also handle a correlated (off-diagonal
  // 1-RDM) reference: its denominators now come from the full-density
  // generalized Fock rather than a diagonal-occupation Fock that would silently
  // drop the off-diagonal 1-RDM.
  auto swpt2_no_semicanon =
      EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  swpt2_no_semicanon->settings().set("semicanonicalize", false);
  auto H_eff_no_semicanon = swpt2_no_semicanon->run(
      reference, H_window,
      testing::restricted_index_set(orbitals->get_num_molecular_orbitals(),
                                    active));
  ASSERT_NE(H_eff_no_semicanon, nullptr);
  auto [E_sw_no_semicanon, wfn_sw_no_semicanon] =
      MultiConfigurationCalculatorFactory::create("macis_cas")
          ->run(H_eff_no_semicanon, na, nb);
  EXPECT_TRUE(std::isfinite(E_sw_no_semicanon));
}

TEST(SchriefferWolffPT2Test, RejectsUnrestrictedHfReference) {
  auto oxygen = testing::create_o2_structure();
  auto scf = ScfSolverFactory::create();
  auto [energy, reference] = scf->run(oxygen, 0, 3, "sto-3g");
  ASSERT_TRUE(reference->get_orbitals()->is_unrestricted());
  auto [restricted_energy, restricted_reference] =
      scf->run(oxygen, 0, 1, "sto-3g");
  ASSERT_TRUE(restricted_reference->get_orbitals()->is_restricted());

  auto ham = HamiltonianConstructorFactory::create();
  auto window_hamiltonian = ham->run(restricted_reference->get_orbitals());
  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");

  try {
    swpt2->run(
        reference, window_hamiltonian,
        testing::restricted_index_set(
            restricted_reference->get_orbitals()->get_num_molecular_orbitals(),
            std::vector<std::size_t>{0}));
    FAIL() << "Expected an unrestricted-reference error";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(std::string(error.what())
                  .find("does not support unrestricted reference orbitals"),
              std::string::npos);
  }
}

// Natural orbitals are the basis in which the reference 1-RDM is diagonal, so
// they are both a supported input and the one basis where folding a
// fractionally occupied orbital drops no off-diagonal density.
//
// Stretched LiH supplies two same-symmetry active sigma orbitals, giving a
// non-diagonal CAS 1-RDM in the canonical HF basis (~0.24 here) and a
// nontrivial natural-orbital rotation. Stretching also drives the occupations
// strongly fractional (1.86/0.14), the regime covered by the fold's rounding
// guard.
//
// The rotation to natural orbitals lies entirely inside P, so
// semicanonicalization must remove it and the downfolded spectrum must match.
TEST(SchriefferWolffPT2Test, AcceptsNaturalOrbitalReference) {
  namespace qcd = qdk::chemistry::data;
  std::vector<Eigen::Vector3d> coords = {{0, 0, 0}, {0, 0, 3.0}};
  for (auto& c : coords) c *= qdk::chemistry::constants::angstrom_to_bohr;
  auto stretched_lih = std::make_shared<qcd::Structure>(
      coords, std::vector<qcd::Element>{qcd::Element::Li, qcd::Element::H});

  auto scf = ScfSolverFactory::create();
  scf->settings().set("enable_gdm", false);
  auto [E_hf, wfn_hf] = scf->run(stretched_lih, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();
  ASSERT_TRUE(orbitals->is_restricted());

  const std::vector<std::size_t> P = {1, 2};
  const std::vector<std::size_t> core = {0};
  const std::vector<std::size_t> window = {1, 2, 3};
  const int norb = static_cast<int>(orbitals->get_num_molecular_orbitals());

  auto ham = HamiltonianConstructorFactory::create();
  auto cas_ref = MultiConfigurationCalculatorFactory::create("macis_cas");
  cas_ref->settings().set("calculate_one_rdm", true);
  auto [E_cas, reference] = cas_ref->run(
      ham->run(testing::with_active_space(orbitals, P, core)), 1, 1);
  ASSERT_TRUE(reference->has_active_one_rdm());
  {
    const auto* rdm = std::get_if<Eigen::MatrixXd>(
        &reference->get_active_one_rdm_spin_traced());
    ASSERT_NE(rdm, nullptr);
    ASSERT_GT(std::abs((*rdm)(0, 1)), 0.1)
        << "canonical basis is already natural; the test would be vacuous";
  }

  // swpt2 reads the (now diagonal) active 1-RDM; the aufbau determinant
  // occupations the localizer also attaches are integers and must not be used.
  auto localizer = LocalizerFactory::create("qdk_natural_orbitals");
  auto natural = localizer->run(reference, P, P);
  ASSERT_TRUE(natural->has_active_one_rdm())
      << "qdk_natural_orbitals must expose the rotated active 1-RDM; without "
         "it the downfold silently falls back to determinant occupations";
  const auto& natural_rdm_variant = natural->get_active_one_rdm_spin_traced();
  const auto* natural_rdm = std::get_if<Eigen::MatrixXd>(&natural_rdm_variant);
  ASSERT_NE(natural_rdm, nullptr);
  ASSERT_EQ(natural_rdm->rows(), 2);
  EXPECT_NEAR((*natural_rdm)(0, 1), 0.0, 1e-10);
  EXPECT_NEAR((*natural_rdm)(1, 0), 0.0, 1e-10);
  // Strongly correlated: neither natural orbital is close to 2 or 0.
  EXPECT_GT((*natural_rdm)(0, 0), 1.5);
  EXPECT_LT((*natural_rdm)(0, 0), 1.95);
  EXPECT_GT((*natural_rdm)(1, 1), 0.05);

  const auto natural_orbitals = natural->get_orbitals();
  ASSERT_TRUE(natural_orbitals->is_restricted());
  const auto& canonical_coeffs =
      orbitals->coefficients()->block({qcd::axes::alpha(), qcd::axes::alpha()});
  const auto& natural_coeffs = natural_orbitals->coefficients()->block(
      {qcd::axes::alpha(), qcd::axes::alpha()});
  ASSERT_GT((natural_coeffs - canonical_coeffs).norm(), 1e-6)
      << "natural-orbital rotation is trivial; the test would be vacuous";

  const auto P_set = testing::restricted_index_set(norb, P);
  auto cas = MultiConfigurationCalculatorFactory::create("macis_cas");

  auto H_natural =
      ham->run(testing::with_active_space(natural_orbitals, window, core));
  auto H_eff_natural =
      EffectiveHamiltonianConstructorFactory::create("qdk_swpt2")
          ->run(natural, H_natural, P_set);
  ASSERT_NE(H_eff_natural, nullptr);
  auto [E_natural, wfn_natural] = cas->run(H_eff_natural, 1, 1);
  EXPECT_TRUE(std::isfinite(E_natural));

  auto H_canonical =
      ham->run(testing::with_active_space(orbitals, window, core));
  auto H_eff_canonical =
      EffectiveHamiltonianConstructorFactory::create("qdk_swpt2")
          ->run(reference, H_canonical, P_set);
  auto [E_canonical, wfn_canonical] = cas->run(H_eff_canonical, 1, 1);

  // The two bases differ only by a rotation within P, so semicanonicalization
  // maps them onto the same internal basis.
  EXPECT_NEAR(E_natural, E_canonical, 1e-8);
}

// Localizing strictly inside the folded virtual block exercises a large
// semicanonical rotation while leaving the reference determinant and P
// untouched, so the emitted operator must be numerically equivalent to the
// canonical-basis result.
TEST(SchriefferWolffPT2Test, AcceptsLocalizedOrbitalReference) {
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();

  const std::vector<std::size_t> P = {3, 4};
  const std::vector<std::size_t> core = {0, 1, 2};
  const std::vector<std::size_t> window = {3, 4, 5, 6};
  const std::vector<std::size_t> virtuals = {5, 6};
  const int norb = static_cast<int>(orbitals->get_num_molecular_orbitals());

  // A rotation among empty virtual orbitals leaves both the reference
  // determinant and P untouched, so any change in the emitted operator would
  // be a basis artifact that semicanonicalization failed to remove.
  auto localizer = LocalizerFactory::create("qdk_pipek_mezey");
  auto localized = localizer->run(wfn_hf, virtuals, virtuals);
  const auto localized_orbitals = localized->get_orbitals();
  ASSERT_TRUE(localized_orbitals->is_restricted());

  namespace qcd = qdk::chemistry::data;
  const auto& canonical_coeffs =
      orbitals->coefficients()->block({qcd::axes::alpha(), qcd::axes::alpha()});
  const auto& localized_coeffs = localized_orbitals->coefficients()->block(
      {qcd::axes::alpha(), qcd::axes::alpha()});
  ASSERT_GT((localized_coeffs - canonical_coeffs).norm(), 1e-6)
      << "localization is trivial; the test would be vacuous";
  // Only the folded virtuals moved, so P sees an identical one-particle basis.
  for (std::size_t p : P)
    EXPECT_NEAR((localized_coeffs.col(p) - canonical_coeffs.col(p))
                    .cwiseAbs()
                    .maxCoeff(),
                0.0, 1e-10);

  auto ham = HamiltonianConstructorFactory::create();
  const auto P_set = testing::restricted_index_set(norb, P);
  const auto downfold = [&](const std::shared_ptr<qcd::Orbitals>& basis) {
    auto cas_ref = MultiConfigurationCalculatorFactory::create("macis_cas");
    cas_ref->settings().set("calculate_one_rdm", true);
    auto [E_cas, reference] = cas_ref->run(
        ham->run(testing::with_active_space(basis, P, core)), 2, 2);
    auto H_window = ham->run(testing::with_active_space(basis, window, core));
    return EffectiveHamiltonianConstructorFactory::create("qdk_swpt2")
        ->run(reference, H_window, P_set);
  };

  auto H_eff_localized = downfold(localized_orbitals);
  auto H_eff_canonical = downfold(orbitals);
  ASSERT_NE(H_eff_localized, nullptr);

  EXPECT_NEAR(H_eff_localized->get_core_energy(),
              H_eff_canonical->get_core_energy(), 1e-9);
  const auto [h_localized, h_localized_b] =
      H_eff_localized->get_one_body_integrals();
  const auto [h_canonical, h_canonical_b] =
      H_eff_canonical->get_one_body_integrals();
  ASSERT_EQ(h_localized.rows(), h_canonical.rows());
  EXPECT_NEAR((h_localized - h_canonical).cwiseAbs().maxCoeff(), 0.0, 1e-9);
  const auto& g_localized =
      std::get<0>(H_eff_localized->get_two_body_integrals());
  const auto& g_canonical =
      std::get<0>(H_eff_canonical->get_two_body_integrals());
  ASSERT_EQ(g_localized.size(), g_canonical.size());
  EXPECT_NEAR((g_localized - g_canonical).cwiseAbs().maxCoeff(), 0.0, 1e-9);
}

// An amplitude-based reference (MP2/CC) carries neither an active 1-RDM nor
// determinant occupations, so the downfold has no reference density to fold
// against and must say so rather than guess one.
TEST(SchriefferWolffPT2Test, RejectsReferenceWithoutDensity) {
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();

  auto ham = HamiltonianConstructorFactory::create();
  auto H_hf = ham->run(orbitals);
  auto ansatz = std::make_shared<qdk::chemistry::data::Ansatz>(*H_hf, *wfn_hf);
  auto [E_mp2, mp2_reference, bra] =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator")
          ->run(ansatz);
  ASSERT_NE(mp2_reference, nullptr);
  ASSERT_FALSE(mp2_reference->has_active_one_rdm());

  const int norb = static_cast<int>(orbitals->get_num_molecular_orbitals());
  auto H_window = ham->run(mp2_reference->get_orbitals());
  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("qdk_swpt2");
  EXPECT_THROW(swpt2->run(mp2_reference, H_window,
                          testing::restricted_index_set(
                              norb, std::vector<std::size_t>{4, 5})),
               std::runtime_error);
}
}  // namespace

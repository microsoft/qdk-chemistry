// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Constructor-level tests for the second-order Schrieffer-Wolff downfold
// (`swpt2`), exercised through the `EffectiveHamiltonianConstructor` factory.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <set>
#include <vector>

#include "swpt2_test_support.hpp"
#include "ut_common.hpp"

namespace {
using qdk::chemistry::algorithms::ActiveSpaceSelectorFactory;
using qdk::chemistry::algorithms::EffectiveHamiltonianConstructorFactory;
using qdk::chemistry::algorithms::HamiltonianConstructorFactory;
using qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory;
using qdk::chemistry::algorithms::ScfSolverFactory;

TEST(SchriefferWolffPT2, FactoryRegistration) {
  const auto available = EffectiveHamiltonianConstructorFactory::available();
  EXPECT_NE(std::find(available.begin(), available.end(), "swpt2"),
            available.end());

  auto ctor = EffectiveHamiltonianConstructorFactory::create("swpt2");
  ASSERT_NE(ctor, nullptr);
  EXPECT_EQ(ctor->name(), "swpt2");
  EXPECT_EQ(ctor->type_name(), "effective_hamiltonian_constructor");

  // aliases and the default name resolve to the same implementation
  EXPECT_EQ(EffectiveHamiltonianConstructorFactory::create("sw")->name(),
            "swpt2");
  EXPECT_EQ(EffectiveHamiltonianConstructorFactory::create("schrieffer_wolff")
                ->name(),
            "swpt2");
  EXPECT_EQ(EffectiveHamiltonianConstructorFactory::create()->name(), "swpt2");
}

TEST(SchriefferWolffPT2, SettingsKnobs) {
  auto ctor = EffectiveHamiltonianConstructorFactory::create("swpt2");
  auto& settings = ctor->settings();
  // denominator knobs: flow regularization is on by default at this layer
  EXPECT_DOUBLE_EQ(settings.get<double>("denom_floor"), 1e-8);
  EXPECT_DOUBLE_EQ(settings.get<double>("denom_shift"), 0.0);
  EXPECT_DOUBLE_EQ(settings.get<double>("denom_flow"), 1.0);
  EXPECT_DOUBLE_EQ(settings.get<double>("intruder_warn_amplitude"), 1.0);
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
// (2) with regularization on by default (denom_flow=1.0) the downfold recovers
// correlation below the bare active-space CAS, whereas an explicitly *bare*
// (unregularized) run does not -- this particular window splits two
// near-degenerate virtuals (active LUMO orbital 5, folded external orbital 6)
// and the resulting small pp-ladder denominator makes bare second-order PT
// overshoot -- a classic intruder tamed by the flow regularizer, not a bug in
// the container/solver or the denominators.
TEST(SchriefferWolffPT2, DownfoldRunsEndToEndWater) {
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();

  const std::vector<size_t> core = {0, 1, 2};
  const std::vector<size_t> active = {3, 4, 5};
  const std::vector<size_t> window = {3, 4, 5, 6};

  auto ham = HamiltonianConstructorFactory::create();

  // reference: CAS over P (produces the active 1-RDM the downfold needs)
  auto p_orbitals = testing::with_active_space(orbitals, active, core);
  auto H_P = ham->run(p_orbitals);
  auto cas_ref = MultiConfigurationCalculatorFactory::create("macis_cas");
  cas_ref->settings().set("calculate_one_rdm", true);
  auto [E_bare, reference] = cas_ref->run(H_P, 2, 2);

  // window Hamiltonian over W = P + one virtual to fold
  auto w_orbitals = testing::with_active_space(orbitals, window, core);
  auto H_window = ham->run(w_orbitals);

  // downfold the window onto P, then CAS on the effective Hamiltonian
  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("swpt2");
  auto H_eff = swpt2->run(reference, H_window);
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
    namespace sw = qdk::chemistry::algorithms::microsoft::swpt2;
    const auto [T_a, T_b] = H_eff->get_one_body_integrals();
    const int m = static_cast<int>(T_a.rows());
    const auto H_so =
        sw::build_tensors(T_a, T_a, ge, ge, ge, H_eff->get_core_energy(), m);
    std::vector<int> orbs(2 * m);
    for (int i = 0; i < 2 * m; ++i) orbs[i] = i;
    const double E_kernel_fci = swpt2_test::fci_ground_energy(
        H_so.e0, H_so.f, H_so.v, 2 * m, orbs, 2, 2);
    EXPECT_NEAR(E_sw, E_kernel_fci, 1e-9);
  }

  // Contrast: disabling regularization (bare second-order PT) hits the
  // small-denominator near-intruder and fails to recover. The active space
  // keeps the active LUMO (orbital 5) while the folded external orbital 6 is a
  // *near-degenerate* virtual (eps_5=+0.47, eps_6=+0.59), so the pp-ladder
  // dressing of the active (5,5) pair has denominator 2(eps_6-eps_5) ~ 0.24 Eh
  // and bare PT2 overshoots. The default flow regularizer is what tames it.
  // (Root cause is the partition splitting two near-degenerate virtuals across
  // active/external -- NOT the container/solver, and NOT the MP denominators,
  // which are exact for this canonical closed-shell reference.)
  auto swpt2_bare = EffectiveHamiltonianConstructorFactory::create("swpt2");
  swpt2_bare->settings().set("denom_flow", -1.0);
  auto H_eff_bare = swpt2_bare->run(reference, H_window);
  auto cas_bare = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_sw_bare, wfn_bare] = cas_bare->run(H_eff_bare, 2, 2);
  EXPECT_GT(E_sw_bare, E_sw);  // regularization lowers the energy vs bare PT2
}

// A mean-field HF reference (single determinant, no active 1-RDM) must work:
// the downfold reads active orbital occupations directly from the reference
// determinant. Mirrors the SCF -> active_space_selector -> downfold user flow.
TEST(SchriefferWolffPT2, AcceptsMeanFieldHfReference) {
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
  auto swpt2 = EffectiveHamiltonianConstructorFactory::create("swpt2");
  auto H_eff = swpt2->run(reference, H_window);
  ASSERT_NE(H_eff, nullptr);

  auto cas_sw = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_sw, wfn_sw] = cas_sw->run(H_eff, static_cast<unsigned int>(na),
                                    static_cast<unsigned int>(nb));
  EXPECT_TRUE(std::isfinite(E_sw));
}
}  // namespace

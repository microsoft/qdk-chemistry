// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

// Constructor-level tests for the single-determinant DSRG-PT2 downfold
// (`qdk_dsrg_pt2`), exercised through `EffectiveHamiltonianConstructorFactory`.

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
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <set>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/effective_hamiltonian/dsrg_pt2_kernel.hpp"
#include "ut_common.hpp"

namespace {
using qdk::chemistry::algorithms::ActiveSpaceSelectorFactory;
using qdk::chemistry::algorithms::DynamicalCorrelationCalculatorFactory;
using qdk::chemistry::algorithms::EffectiveHamiltonianConstructorFactory;
using qdk::chemistry::algorithms::HamiltonianConstructorFactory;
using qdk::chemistry::algorithms::LocalizerFactory;
using qdk::chemistry::algorithms::MultiConfigurationCalculatorFactory;
using qdk::chemistry::algorithms::ScfSolverFactory;

namespace dsrg = qdk::chemistry::algorithms::microsoft::dsrg_pt2;

struct SpinOrbitalTensors {
  double core_energy;
  Eigen::MatrixXd one_body;
  Eigen::VectorXd two_body;
};

SpinOrbitalTensors build_spin_orbital_tensors(
    const Eigen::MatrixXd& alpha, const Eigen::MatrixXd& beta,
    const Eigen::VectorXd& alpha_integrals,
    const Eigen::VectorXd& mixed_integrals,
    const Eigen::VectorXd& beta_integrals, double core_energy, int norb) {
  const int spin_count = 2 * norb;
  SpinOrbitalTensors result{
      core_energy, Eigen::MatrixXd::Zero(spin_count, spin_count),
      Eigen::VectorXd::Zero(spin_count * spin_count * spin_count * spin_count)};
  for (int row = 0; row < norb; ++row)
    for (int col = 0; col < norb; ++col) {
      result.one_body(2 * row, 2 * col) = alpha(row, col);
      result.one_body(2 * row + 1, 2 * col + 1) = beta(row, col);
    }
  std::vector<double> ordered(result.two_body.size(), 0.0);
  for (int first = 0; first < norb; ++first)
    for (int second = 0; second < norb; ++second)
      for (int third = 0; third < norb; ++third)
        for (int fourth = 0; fourth < norb; ++fourth) {
          const auto integral = dsrg::idx4(first, second, third, fourth, norb);
          ordered[dsrg::idx4(2 * first, 2 * third, 2 * fourth, 2 * second,
                             spin_count)] += 0.5 * alpha_integrals(integral);
          ordered[dsrg::idx4(2 * first + 1, 2 * third + 1, 2 * fourth + 1,
                             2 * second + 1, spin_count)] +=
              0.5 * beta_integrals(integral);
          ordered[dsrg::idx4(2 * first, 2 * third + 1, 2 * fourth + 1,
                             2 * second, spin_count)] +=
              mixed_integrals(integral);
        }
  for (int first = 0; first < spin_count; ++first)
    for (int second = 0; second < spin_count; ++second)
      for (int third = 0; third < spin_count; ++third)
        for (int fourth = 0; fourth < spin_count; ++fourth)
          result.two_body(
              dsrg::idx4(first, second, third, fourth, spin_count)) =
              ordered[dsrg::idx4(first, second, third, fourth, spin_count)] -
              ordered[dsrg::idx4(second, first, third, fourth, spin_count)] -
              ordered[dsrg::idx4(first, second, fourth, third, spin_count)] +
              ordered[dsrg::idx4(second, first, fourth, third, spin_count)];
  return result;
}

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
            const double vpqrs = v(dsrg::idx4(P, Q, R, S, n_so));
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

// Lowest energy of `hamiltonian` among determinants of `spin_orbitals` (alpha
// 2p, beta 2p+1), or of all spin orbitals if none are given.
double ground_energy(const qdk::chemistry::data::Hamiltonian& hamiltonian,
                     int na, int nb, std::vector<int> spin_orbitals = {}) {
  const auto [alpha, beta] = hamiltonian.get_one_body_integrals();
  const auto [aaaa, aabb, bbbb] = hamiltonian.get_two_body_integrals();
  const int norb = static_cast<int>(alpha.rows());
  const auto tensors = build_spin_orbital_tensors(
      alpha, beta, aaaa, aabb, bbbb, hamiltonian.get_core_energy(), norb);
  if (spin_orbitals.empty()) {
    spin_orbitals.resize(2 * norb);
    std::iota(spin_orbitals.begin(), spin_orbitals.end(), 0);
  }
  return fci_ground_energy(tensors.core_energy, tensors.one_body,
                           tensors.two_body, 2 * norb, spin_orbitals, na, nb);
}

TEST(DsrgPT2Test, FactoryRegistration) {
  const auto available = EffectiveHamiltonianConstructorFactory::available();
  EXPECT_NE(std::find(available.begin(), available.end(), "qdk_dsrg_pt2"),
            available.end());

  auto ctor = EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  ASSERT_NE(ctor, nullptr);
  EXPECT_EQ(ctor->name(), "qdk_dsrg_pt2");
  EXPECT_EQ(ctor->type_name(), "effective_hamiltonian_constructor");

  EXPECT_EQ(EffectiveHamiltonianConstructorFactory::create()->name(),
            "qdk_dsrg_pt2");
}

TEST(DsrgPT2Test, SettingsKnobs) {
  auto ctor = EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  auto& settings = ctor->settings();
  EXPECT_DOUBLE_EQ(settings.get<double>("flow_parameter"), 0.5);
  EXPECT_ANY_THROW(settings.set("flow_parameter", -1.0));

  auto invalid = EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  EXPECT_THROW(invalid->run(nullptr, nullptr,
                            testing::restricted_index_set(
                                1, std::vector<std::size_t>{0})),
               std::invalid_argument);
}

TEST(DsrgPT2Test, PreservesModelOrbitals) {
  namespace qcd = qdk::chemistry::data;
  constexpr int norb = 6;
  const auto window_indices =
      testing::restricted_index_set(norb, {0, 1, 2, 3, 4, 5});
  const auto empty_indices = testing::restricted_index_set(norb, {});
  const auto p_indices = testing::restricted_index_set(norb, {1, 2, 3, 4});
  const auto inactive_indices = testing::restricted_index_set(norb, {0});
  const auto model_orbitals =
      std::make_shared<qcd::ModelOrbitals>(window_indices, empty_indices);
  const auto coefficient_orbitals = std::make_shared<qcd::Orbitals>(
      Eigen::MatrixXd::Identity(norb, norb), std::nullopt, std::nullopt,
      testing::create_random_basis_set(norb), window_indices, empty_indices);

  Eigen::MatrixXd one_body = Eigen::MatrixXd::Zero(norb, norb);
  Eigen::VectorXd two_body = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  for (int i = 0; i < norb; ++i) {
    two_body(dsrg::idx4(i, i, i, i, norb)) = 4.0;
    if (i + 1 < norb) one_body(i, i + 1) = one_body(i + 1, i) = -1.0;
  }
  const auto downfold = [&](const std::shared_ptr<qcd::Orbitals>& orbitals) {
    auto reference = std::make_shared<qcd::Wavefunction>(
        std::make_unique<qcd::StateVectorContainer>(
            qcd::Configuration::from_spin_half_string("222000"), orbitals));
    auto window = std::make_shared<qcd::Hamiltonian>(
        std::make_unique<qcd::CanonicalFourCenterHamiltonianContainer>(
            one_body, two_body, orbitals, 0.0, Eigen::MatrixXd::Zero(0, 0)));
    return EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2")
        ->run(reference, window, p_indices);
  };
  const auto expected = downfold(coefficient_orbitals);
  const auto result = downfold(model_orbitals);
  const auto restored = qcd::Hamiltonian::from_json(result->to_json());
  const auto expect_same_indices =
      [](const std::shared_ptr<const qcd::SymmetryBlockedIndexSet>& actual,
         const std::shared_ptr<const qcd::SymmetryBlockedIndexSet>& target) {
        EXPECT_EQ(*actual->symmetries(), *target->symmetries());
        EXPECT_EQ(actual->extents(), target->extents());
        for (const auto& spin : {qcd::axes::alpha(), qcd::axes::beta()})
          EXPECT_EQ(qcd::spin_channel_indices(actual, spin),
                    qcd::spin_channel_indices(target, spin));
      };
  for (const auto& output : {result, restored}) {
    const auto orbitals = output->get_orbitals();
    ASSERT_NE(std::dynamic_pointer_cast<qcd::ModelOrbitals>(orbitals), nullptr);
    EXPECT_TRUE(orbitals->is_restricted());
    EXPECT_EQ(orbitals->get_num_molecular_orbitals(), norb);
    expect_same_indices(orbitals->active_indices(), p_indices);
    expect_same_indices(orbitals->inactive_indices(), inactive_indices);
    EXPECT_NEAR(output->get_core_energy(), expected->get_core_energy(), 1e-12);
    EXPECT_NEAR((std::get<0>(output->get_one_body_integrals()) -
                 std::get<0>(expected->get_one_body_integrals()))
                    .cwiseAbs()
                    .maxCoeff(),
                0.0, 1e-12);
    EXPECT_NEAR((std::get<0>(output->get_two_body_integrals()) -
                 std::get<0>(expected->get_two_body_integrals()))
                    .cwiseAbs()
                    .maxCoeff(),
                0.0, 1e-12);
  }
}

// SCF -> window Hamiltonian over W -> single-determinant reference over P ->
// DSRG-PT2 -> CAS on H_eff. Also checks that flow_parameter = 0 returns the
// bare core-folded operator over P.
TEST(DsrgPT2Test, DownfoldRunsEndToEndWater) {
  namespace qcd = qdk::chemistry::data;
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
  auto reference = std::make_shared<qcd::Wavefunction>(
      std::make_unique<qcd::StateVectorContainer>(
          qcd::Configuration::from_spin_half_string("220"), p_orbitals));
  auto cas_bare_ref = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_bare, wfn_bare_ref] = cas_bare_ref->run(H_P, 2, 2);

  // window Hamiltonian over W = P + one virtual to fold
  auto w_orbitals = testing::with_active_space(orbitals, window, core);
  auto H_window = ham->run(w_orbitals);

  // Every reference-active orbital must lie in the window.
  auto incomplete_w_orbitals =
      testing::with_active_space(orbitals, {3, 4, 6}, core);
  auto H_incomplete = ham->run(incomplete_w_orbitals);
  auto dsrg_invalid =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  EXPECT_THROW(dsrg_invalid->run(reference, H_incomplete, P_set),
               std::invalid_argument);

  // downfold the window onto P, then CAS on the effective Hamiltonian
  auto dsrg_ctor =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  auto H_eff = dsrg_ctor->run(reference, H_window, P_set);
  ASSERT_NE(H_eff, nullptr);
  auto cas_dsrg = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_dsrg, wfn_dsrg] = cas_dsrg->run(H_eff, 2, 2);

  EXPECT_TRUE(std::isfinite(E_dsrg));
  EXPECT_LT(E_dsrg, E_hf);
  EXPECT_LT(E_dsrg, E_bare);

  // the downfold actually dressed the two-body interaction (nonzero effect)
  const auto [ge, ge_ab, ge_bb] = H_eff->get_two_body_integrals();
  const auto [gb, gb_ab, gb_bb] = H_P->get_two_body_integrals();
  EXPECT_GT((ge - gb).norm(), 1e-6);

  // MACIS must reproduce a full CI of the same emitted integrals, which are
  // only 4-fold symmetric, so the container must hand it the dense block.
  EXPECT_NEAR(E_dsrg, ground_energy(*H_eff, 2, 2), 1e-9);

  // flow_parameter = 0 disables the DSRG transformation, returning the bare P
  // operator.
  auto dsrg_zero =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  dsrg_zero->settings().set("flow_parameter", 0.0);
  auto H_eff_zero = dsrg_zero->run(reference, H_window, P_set);
  EXPECT_NEAR(H_eff_zero->get_core_energy(), H_P->get_core_energy(), 1e-12);
  EXPECT_NEAR((std::get<0>(H_eff_zero->get_one_body_integrals()) -
               std::get<0>(H_P->get_one_body_integrals()))
                  .cwiseAbs()
                  .maxCoeff(),
              0.0, 1e-12);
  EXPECT_NEAR((std::get<0>(H_eff_zero->get_two_body_integrals()) - gb)
                  .cwiseAbs()
                  .maxCoeff(),
              0.0, 1e-12);
}

// MP2 limit: with every occupied orbital active, no inactive space, all
// virtuals folded and large flow s -> infinity, the reference determinant's
// energy in H_eff approaches E_HF + E_MP2.
TEST(DsrgPT2Test, ReproducesMp2WhenEveryOccupiedOrbitalIsActive) {
  namespace qcd = qdk::chemistry::data;
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  // MP2 uses the SCF orbital energies, so agreement tracks SCF convergence.
  scf->settings().set("convergence_threshold", 1e-9);
  scf->settings().set("enable_gdm", false);
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
  auto dsrg_ctor =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  dsrg_ctor->settings().set("flow_parameter", 1e6);
  auto H_eff = dsrg_ctor->run(reference, H_window,
                              testing::restricted_index_set(norb, occupied));
  ASSERT_NE(H_eff, nullptr);
  EXPECT_TRUE(H_eff->is_restricted());
  const int n_alpha = static_cast<int>(na), n_beta = static_cast<int>(nb);
  const double E_downfold = ground_energy(*H_eff, n_alpha, n_beta);

  auto H_full = ham->run(orbitals);
  auto ansatz = std::make_shared<qcd::Ansatz>(*H_full, *wfn_hf);
  auto [E_mp2, mp2_ket, mp2_bra] =
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator")
          ->run(ansatz);

  EXPECT_LT(E_mp2, E_hf - 1e-3) << "the correlation being matched is not noise";
  EXPECT_NEAR(E_downfold, E_mp2, 1e-9);
}

// The window Hamiltonian's own folded core must be exactly the reference core
// orbitals outside W.
TEST(DsrgPT2Test, RejectsWindowCoreThatIsNotTheReferenceCore) {
  namespace qcd = qdk::chemistry::data;
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();
  const std::size_t norb = orbitals->get_num_molecular_orbitals();

  const std::vector<std::size_t> core = {0, 1, 2};
  const std::vector<std::size_t> active = {3, 4};
  const std::vector<std::size_t> window = {3, 4, 5};

  auto ham = HamiltonianConstructorFactory::create();
  auto reference = std::make_shared<qcd::Wavefunction>(
      std::make_unique<qcd::StateVectorContainer>(
          qcd::Configuration::from_spin_half_string("22"),
          testing::with_active_space(orbitals, active, core)));

  // Window core drops orbital 2, so it no longer matches the reference core.
  auto H_window = ham->run(
      testing::with_active_space(orbitals, window, {core[0], core[1]}));
  EXPECT_THROW(EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2")
                   ->run(reference, H_window,
                         testing::restricted_index_set(norb, active)),
               std::invalid_argument);
}

// A mean-field HF reference (single determinant, no stored 1-RDM) works out of
// the box, while a single-determinant container carrying an inconsistent
// correlated 1-RDM is rejected.
TEST(DsrgPT2Test, AcceptsMeanFieldHfReference) {
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();

  auto selector = ActiveSpaceSelectorFactory::create("qdk_valence");
  selector->settings().set("num_active_electrons", 2);
  selector->settings().set("num_active_orbitals", 2);
  auto reference = selector->run(wfn_hf);
  const auto [na, nb] = reference->get_active_num_electrons();

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
  auto dsrg_ctor =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  auto H_eff = dsrg_ctor->run(reference, H_window,
                              testing::restricted_index_set(norb, P));
  ASSERT_NE(H_eff, nullptr);

  auto cas_dsrg = MultiConfigurationCalculatorFactory::create("macis_cas");
  auto [E_dsrg, wfn_dsrg] = cas_dsrg->run(H_eff, static_cast<unsigned int>(na),
                                          static_cast<unsigned int>(nb));
  EXPECT_TRUE(std::isfinite(E_dsrg));

  // A stored 1-RDM that does not match the single determinant is rejected.
  Eigen::MatrixXd correlated_density(2, 2);
  correlated_density << 1.0, 0.2, 0.2, 1.0;
  Eigen::VectorXd coefficients = Eigen::VectorXd::Ones(1);
  qcd::Wavefunction::DeterminantVector determinants{
      qcd::Configuration::from_spin_half_string("20")};
  auto correlated_reference = std::make_shared<qcd::Wavefunction>(
      std::make_unique<qcd::StateVectorContainer>(
          coefficients, determinants, ref_orbs,
          std::optional<qcd::StateVectorContainer::MatrixVariant>{
              correlated_density},
          std::nullopt));
  EXPECT_THROW(dsrg_ctor->run(correlated_reference, H_window,
                              testing::restricted_index_set(norb, P)),
               std::invalid_argument);
}

// The kept space P can differ from the reference wavefunction's active space,
// embedding reference core/virtual orbitals into P and folding closed-shell
// reference active orbitals outside P.
TEST(DsrgPT2Test, CustomActiveSpaceOverridesReference) {
  namespace qcd = qdk::chemistry::data;
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();

  const std::vector<size_t> core = {0, 1, 2};
  const std::vector<size_t> active = {3, 4};  // reference active space
  const std::vector<size_t> window = {3, 4, 5, 6};

  auto ham = HamiltonianConstructorFactory::create();
  auto reference = std::make_shared<qcd::Wavefunction>(
      std::make_unique<qcd::StateVectorContainer>(
          qcd::Configuration::from_spin_half_string("22"),
          testing::with_active_space(orbitals, active, core)));
  auto H_window = ham->run(testing::with_active_space(orbitals, window, core));

  // (1) Custom P that embeds empty external orbital 5 while folding {6}.
  const std::vector<std::size_t> P_custom = {3, 4, 5};
  const auto p_indices_custom = testing::restricted_index_set(
      orbitals->get_num_molecular_orbitals(), P_custom);
  auto dsrg_ctor =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  auto H_eff = dsrg_ctor->run(reference, H_window, p_indices_custom);
  ASSERT_NE(H_eff, nullptr);

  const auto [T_a, T_b] = H_eff->get_one_body_integrals();
  EXPECT_EQ(T_a.rows(), static_cast<Eigen::Index>(P_custom.size()));
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
  auto [E_dsrg, wfn_dsrg] = cas->run(H_eff, 2, 2);
  EXPECT_TRUE(std::isfinite(E_dsrg));

  // (2) Custom P = the whole window => empty external => identity.
  const std::vector<std::size_t> P_full = {3, 4, 5, 6};
  auto dsrg_full =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  auto H_identity =
      dsrg_full->run(reference, H_window,
                     testing::restricted_index_set(
                         orbitals->get_num_molecular_orbitals(), P_full));
  const auto [ge, ge_ab, ge_bb] = H_identity->get_two_body_integrals();
  const auto [gw, gw_ab, gw_bb] = H_window->get_two_body_integrals();
  EXPECT_LT((ge - gw).norm(), 1e-9);
  EXPECT_NEAR(H_identity->get_core_energy(), H_window->get_core_energy(), 1e-9);

  // (3) Window spanning the reference core: {0,1,2} are folded inside W and
  // still come back labelled inactive. Also matches downfolding the full-window
  // SCF wavefunction directly onto {3, 4}.
  auto H_all =
      ham->run(testing::with_active_space(orbitals, {0, 1, 2, 3, 4, 5, 6}, {}));
  auto dsrg_core =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  const auto p_34 = testing::restricted_index_set(
      orbitals->get_num_molecular_orbitals(), std::vector<std::size_t>{3, 4});
  auto H_eff_core = dsrg_core->run(reference, H_all, p_34);
  auto H_eff_from_hf =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2")
          ->run(wfn_hf, H_all, p_34);
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
  EXPECT_NEAR(H_eff_core->get_core_energy(), H_eff_from_hf->get_core_energy(),
              1e-12);

  // (4) A spin-dependent P is rejected rather than silently reduced to alpha.
  auto dsrg_spin =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  EXPECT_THROW(dsrg_spin->run(reference, H_window,
                              testing::unrestricted_index_set(
                                  orbitals->get_num_molecular_orbitals(),
                                  {3, 4, 5}, {3, 4, 6})),
               std::invalid_argument);
}

TEST(DsrgPT2Test, RejectsCorrelatedCasReference) {
  auto water = testing::create_water_structure();
  auto scf = ScfSolverFactory::create();
  auto [E_hf, wfn_hf] = scf->run(water, 0, 1, "sto-3g");
  const auto orbitals = wfn_hf->get_orbitals();
  const auto num_mo = orbitals->get_num_molecular_orbitals();

  auto ham = HamiltonianConstructorFactory::create();
  auto cas_ref = MultiConfigurationCalculatorFactory::create("macis_cas");
  cas_ref->settings().set("calculate_one_rdm", true);
  auto [E_bare, reference] = cas_ref->run(
      ham->run(testing::with_active_space(orbitals, {4, 5}, {0, 1, 2, 3})), 1,
      1);

  auto H_window =
      ham->run(testing::with_active_space(orbitals, {3, 4, 5, 6}, {0, 1, 2}));
  const auto p_indices =
      testing::restricted_index_set(num_mo, std::vector<std::size_t>{4, 5});

  auto dsrg_ctor =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  EXPECT_THROW(dsrg_ctor->run(reference, H_window, p_indices),
               std::invalid_argument);
}

// Restricted open-shell determinants have nonzero spin-averaged cumulants in
// DSRG-MRPT2 and are therefore rejected.
TEST(DsrgPT2Test, RejectsRestrictedOpenShellHfReference) {
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
  auto dsrg_ctor =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  EXPECT_THROW(
      dsrg_ctor->run(reference, H_window,
                     testing::restricted_index_set(
                         orbitals->get_num_molecular_orbitals(), active)),
      std::invalid_argument);
}

// For a UHF doublet with every occupied orbital but the folded core kept, P
// holds a single empty spin orbital, so no double excitation fits inside it
// and the reference determinant's energy in H_eff approaches UMP2 at large s.
// Rotating each spin's orbitals differently within its occupied active orbitals
// and folded virtuals must not change that; folding the singly occupied orbital
// is rejected.
TEST(DsrgPT2Test, ReproducesUmp2ForUnrestrictedDoublet) {
  namespace qcd = qdk::chemistry::data;
  auto scf = ScfSolverFactory::create();
  // UMP2 uses the SCF orbital energies, so agreement tracks SCF convergence.
  scf->settings().set("convergence_threshold", 1e-9);
  scf->settings().set("enable_gdm", false);
  auto [E_uhf, wfn_uhf] =
      scf->run(testing::create_water_structure(), 1, 2, "sto-3g");
  const auto orbitals = wfn_uhf->get_orbitals();
  ASSERT_TRUE(orbitals->is_unrestricted());
  const int norb = static_cast<int>(orbitals->get_num_molecular_orbitals());
  const auto [na, nb] = wfn_uhf->get_active_num_electrons();
  ASSERT_EQ(na, nb + 1);
  ASSERT_GE(norb, static_cast<int>(na) + 2);
  std::vector<size_t> kept(na - 1);
  std::iota(kept.begin(), kept.end(), size_t{1});
  const int kept_beta = static_cast<int>(nb) - 1;

  auto ham = HamiltonianConstructorFactory::create();
  auto H_full = ham->run(orbitals);
  auto ansatz = std::make_shared<qcd::Ansatz>(*H_full, *wfn_uhf);
  const double E_ump2 = std::get<0>(
      DynamicalCorrelationCalculatorFactory::create("qdk_mp2_calculator")
          ->run(ansatz));
  EXPECT_LT(E_ump2, E_uhf - 1e-3) << "the correlation being matched is noise";

  // Folding the singly occupied orbital (index na - 1) must be rejected.
  const std::vector<size_t> missing_singly_occupied{1, 2, 3};
  EXPECT_THROW(
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2")
          ->run(wfn_uhf, H_full,
                testing::unrestricted_index_set(norb, missing_singly_occupied,
                                                missing_singly_occupied)),
      std::invalid_argument);

  const auto reference_energy =
      [&](const std::shared_ptr<qcd::Wavefunction>& reference,
          const std::shared_ptr<qcd::Hamiltonian>& window) {
        auto dsrg_ctor =
            EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
        dsrg_ctor->settings().set("flow_parameter", 1e6);
        const auto H_eff =
            dsrg_ctor->run(reference, window,
                           testing::unrestricted_index_set(norb, kept, kept));
        EXPECT_TRUE(H_eff->is_unrestricted());
        const auto [T_a, T_b] = H_eff->get_one_body_integrals();
        EXPECT_GT((T_a - T_b).cwiseAbs().maxCoeff(), 1e-3);
        const int m = static_cast<int>(T_a.rows());
        std::vector<int> occupied;
        for (int orbital = 0; orbital < m; ++orbital) {
          occupied.push_back(2 * orbital);
          if (orbital < kept_beta) occupied.push_back(2 * orbital + 1);
        }
        return ground_energy(*H_eff, m, kept_beta, occupied);
      };
  EXPECT_NEAR(reference_energy(wfn_uhf, H_full), E_ump2, 1e-9);

  const auto givens = [norb](int first, int second, double angle) {
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(norb, norb);
    rotation(first, first) = rotation(second, second) = std::cos(angle);
    rotation(first, second) = -std::sin(angle);
    rotation(second, first) = std::sin(angle);
    return rotation;
  };
  const int first_virtual = static_cast<int>(na);
  const Eigen::MatrixXd alpha = orbitals->coefficients()->block(
                                    {qcd::axes::alpha(), qcd::axes::alpha()}) *
                                givens(1, 3, 0.3) *
                                givens(first_virtual, first_virtual + 1, 0.5);
  const Eigen::MatrixXd beta =
      orbitals->coefficients()->block({qcd::axes::beta(), qcd::axes::beta()}) *
      givens(1, 2, -0.2) * givens(first_virtual, first_virtual + 1, 0.25);
  std::vector<size_t> all(norb);
  std::iota(all.begin(), all.end(), size_t{0});
  auto rotated = std::make_shared<qcd::Orbitals>(
      alpha, beta, std::nullopt, std::nullopt, orbitals->get_overlap_matrix(),
      orbitals->get_basis_set(),
      testing::unrestricted_index_set(norb, all, all),
      testing::unrestricted_index_set(norb, {}, {}));
  // The rotations keep occupied and empty orbitals apart, so the per-spin
  // 1-RDMs stay diagonal.
  const auto [occupation_alpha, occupation_beta] =
      wfn_uhf->get_active_orbital_occupations();
  qcd::Wavefunction::DeterminantVector determinants{
      qcd::Configuration::from_spin_half_string(
          std::string(nb, '2') + 'u' +
          std::string(static_cast<size_t>(norb) - na, '0'))};
  auto reference = std::make_shared<qcd::Wavefunction>(
      std::make_unique<qcd::StateVectorContainer>(
          Eigen::VectorXd(Eigen::VectorXd::Ones(1)), determinants, rotated,
          std::nullopt,
          std::optional<qcd::StateVectorContainer::MatrixVariant>{
              Eigen::MatrixXd(occupation_alpha.asDiagonal())},
          std::optional<qcd::StateVectorContainer::MatrixVariant>{
              Eigen::MatrixXd(occupation_beta.asDiagonal())},
          std::nullopt, std::nullopt, std::nullopt, std::nullopt));
  ASSERT_TRUE(reference->has_one_rdm_spin_dependent());
  EXPECT_NEAR(reference_energy(reference, ham->run(rotated)), E_ump2, 1e-9);
}

// The reference active space places both spins' densities, so differing alpha
// and beta active orbitals are rejected rather than misread.
TEST(DsrgPT2Test, RejectsSpinDependentReferenceActiveSpace) {
  namespace qcd = qdk::chemistry::data;
  const auto orbitals =
      std::get<1>(ScfSolverFactory::create()->run(
                      testing::create_water_structure(), 1, 2, "sto-3g"))
          ->get_orbitals();
  const size_t norb = orbitals->get_num_molecular_orbitals();
  ASSERT_EQ(norb, 7);
  const Eigen::MatrixXd alpha =
      orbitals->coefficients()->block({qcd::axes::alpha(), qcd::axes::alpha()});
  const Eigen::MatrixXd beta =
      orbitals->coefficients()->block({qcd::axes::beta(), qcd::axes::beta()});
  const auto with_active = [&](const std::vector<size_t>& active_alpha,
                               const std::vector<size_t>& active_beta) {
    return std::make_shared<qcd::Orbitals>(
        alpha, beta, std::nullopt, std::nullopt, orbitals->get_overlap_matrix(),
        orbitals->get_basis_set(),
        testing::unrestricted_index_set(norb, active_alpha, active_beta),
        testing::unrestricted_index_set(norb, {}, {}));
  };
  auto window = HamiltonianConstructorFactory::create()->run(
      with_active({0, 1, 2, 3, 4, 5, 6}, {0, 1, 2, 3, 4, 5, 6}));
  Eigen::VectorXd occupation_alpha(6), occupation_beta(6);
  occupation_alpha << 1, 1, 1, 1, 1, 0;
  occupation_beta << 1, 1, 1, 1, 0, 1;
  qcd::Wavefunction::DeterminantVector determinants{
      qcd::Configuration::from_spin_half_string("2222ud")};
  auto reference = std::make_shared<qcd::Wavefunction>(
      std::make_unique<qcd::StateVectorContainer>(
          Eigen::VectorXd(Eigen::VectorXd::Ones(1)), determinants,
          with_active({0, 1, 2, 3, 4, 5}, {0, 1, 2, 3, 4, 6}), std::nullopt,
          std::optional<qcd::StateVectorContainer::MatrixVariant>{
              Eigen::MatrixXd(occupation_alpha.asDiagonal())},
          std::optional<qcd::StateVectorContainer::MatrixVariant>{
              Eigen::MatrixXd(occupation_beta.asDiagonal())},
          std::nullopt, std::nullopt, std::nullopt, std::nullopt));

  const std::vector<size_t> kept{3, 4, 5, 6};
  try {
    EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2")
        ->run(reference, window,
              testing::unrestricted_index_set(norb, kept, kept));
    FAIL() << "Expected a spin-dependent active space error";
  } catch (const std::invalid_argument& error) {
    EXPECT_NE(
        std::string(error.what()).find("same orbitals in both spin channels"),
        std::string::npos);
  }
}

// Localizing strictly inside the folded virtual block exercises a large
// semicanonical rotation on a single-determinant reference.
TEST(DsrgPT2Test, AcceptsLocalizedOrbitalReference) {
  namespace qcd = qdk::chemistry::data;
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
    auto reference = std::make_shared<qcd::Wavefunction>(
        std::make_unique<qcd::StateVectorContainer>(
            qcd::Configuration::from_spin_half_string("22"),
            testing::with_active_space(basis, P, core)));
    auto H_window = ham->run(testing::with_active_space(basis, window, core));
    return EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2")
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

// An amplitude-based reference (MP2/CC) is not a StateVectorContainer and must
// be rejected cleanly.
TEST(DsrgPT2Test, RejectsNonStateVectorReference) {
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

  const int norb = static_cast<int>(orbitals->get_num_molecular_orbitals());
  auto H_window = ham->run(mp2_reference->get_orbitals());
  auto dsrg_ctor =
      EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2");
  EXPECT_THROW(dsrg_ctor->run(mp2_reference, H_window,
                              testing::restricted_index_set(
                                  norb, std::vector<std::size_t>{4, 5})),
               std::invalid_argument);
}

TEST(DsrgPT2Test, AcceptsEmptyReferenceActiveSpaceWithZeroSizedRdm) {
  namespace qcd = qdk::chemistry::data;
  constexpr std::size_t norb = 4;
  Eigen::MatrixXd h1 = Eigen::MatrixXd::Zero(norb, norb);
  h1.diagonal() << -2.0, -1.0, 1.0, 2.0;
  h1(0, 2) = h1(2, 0) = 0.1;
  const Eigen::VectorXd g2 = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  const Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);

  for (const bool unrestricted : {false, true}) {
    const auto idx = [&](const std::vector<std::size_t>& orbitals) {
      return unrestricted
                 ? testing::unrestricted_index_set(norb, orbitals, orbitals)
                 : testing::restricted_index_set(norb, orbitals);
    };
    auto win_orbs = std::make_shared<qcd::ModelOrbitals>(idx({0, 1, 2, 3}));
    auto ref_orbs = std::make_shared<qcd::ModelOrbitals>(idx({}), idx({0, 1}));
    auto window = unrestricted
                      ? std::make_shared<qcd::Hamiltonian>(
                            std::make_unique<
                                qcd::CanonicalFourCenterHamiltonianContainer>(
                                h1, h1, g2, g2, g2, win_orbs, 0.0, empty_fock,
                                empty_fock))
                      : std::make_shared<qcd::Hamiltonian>(
                            std::make_unique<
                                qcd::CanonicalFourCenterHamiltonianContainer>(
                                h1, g2, win_orbs, 0.0, empty_fock));
    const qcd::Wavefunction::DeterminantVector dets{
        qcd::Configuration::from_spin_half_string("")};
    const Eigen::MatrixXd zero_rdm(0, 0);
    const Eigen::VectorXd coeffs(Eigen::VectorXd::Ones(1));
    std::unique_ptr<qcd::StateVectorContainer> container;
    if (unrestricted) {
      container = std::make_unique<qcd::StateVectorContainer>(
          coeffs, dets, ref_orbs, std::nullopt,
          std::optional<qcd::StateVectorContainer::MatrixVariant>{zero_rdm},
          std::optional<qcd::StateVectorContainer::MatrixVariant>{zero_rdm},
          std::nullopt, std::nullopt, std::nullopt, std::nullopt);
    } else {
      container = std::make_unique<qcd::StateVectorContainer>(
          coeffs, dets, ref_orbs,
          std::optional<qcd::StateVectorContainer::MatrixVariant>{zero_rdm},
          std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
          std::nullopt);
    }
    auto reference = std::make_shared<qcd::Wavefunction>(std::move(container));

    auto h_eff = EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2")
                     ->run(reference, window, idx({1, 2}));
    ASSERT_NE(h_eff, nullptr);
    EXPECT_TRUE(std::isfinite(h_eff->get_core_energy()));
    auto h_virt_only =
        EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2")
            ->run(reference, window, idx({0, 1, 2}));
    EXPECT_FALSE(h_virt_only->get_orbitals()->has_inactive_space());
  }
}

TEST(DsrgPT2Test, RejectsMismatchedStoredRdmsRegardlessOfSpinRestriction) {
  namespace qcd = qdk::chemistry::data;
  constexpr std::size_t norb = 4;
  Eigen::MatrixXd h1 = Eigen::MatrixXd::Zero(norb, norb);
  h1.diagonal() << -2.0, -1.0, 1.0, 2.0;
  const Eigen::VectorXd g2 = Eigen::VectorXd::Zero(norb * norb * norb * norb);
  const Eigen::MatrixXd empty_fock = Eigen::MatrixXd::Zero(0, 0);
  const qcd::Wavefunction::DeterminantVector dets{
      qcd::Configuration::from_spin_half_string("20")};

  // Unrestricted reference carrying only a correlated spin-traced 1-RDM.
  {
    auto u_win = std::make_shared<qcd::ModelOrbitals>(
        testing::unrestricted_index_set(norb, {0, 1, 2, 3}, {0, 1, 2, 3}));
    auto u_ref = std::make_shared<qcd::ModelOrbitals>(
        testing::unrestricted_index_set(norb, {1, 2}, {1, 2}),
        testing::unrestricted_index_set(norb, {0}, {0}));
    auto window = std::make_shared<qcd::Hamiltonian>(
        std::make_unique<qcd::CanonicalFourCenterHamiltonianContainer>(
            h1, h1, g2, g2, g2, u_win, 0.0, empty_fock, empty_fock));
    Eigen::Matrix2d bad_traced;
    bad_traced << 1.5, 0.4, 0.4, 0.5;
    auto reference = std::make_shared<qcd::Wavefunction>(
        std::make_unique<qcd::StateVectorContainer>(
            Eigen::VectorXd(Eigen::VectorXd::Ones(1)), dets, u_ref,
            std::optional<qcd::StateVectorContainer::MatrixVariant>{
                Eigen::MatrixXd(bad_traced)},
            std::nullopt, std::nullopt, std::nullopt, std::nullopt,
            std::nullopt, std::nullopt));
    EXPECT_THROW(
        EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2")
            ->run(reference, window,
                  testing::unrestricted_index_set(norb, {1, 2}, {1, 2})),
        std::invalid_argument);
  }

  // Restricted reference carrying a matching spin-traced 1-RDM plus mismatched
  // spin-dependent 1-RDMs.
  {
    auto r_win = std::make_shared<qcd::ModelOrbitals>(
        testing::restricted_index_set(norb, {0, 1, 2, 3}));
    auto r_ref = std::make_shared<qcd::ModelOrbitals>(
        testing::restricted_index_set(norb, {1, 2}),
        testing::restricted_index_set(norb, {0}));
    auto window = std::make_shared<qcd::Hamiltonian>(
        std::make_unique<qcd::CanonicalFourCenterHamiltonianContainer>(
            h1, g2, r_win, 0.0, empty_fock));
    Eigen::MatrixXd good_traced = Eigen::MatrixXd::Zero(2, 2);
    good_traced(0, 0) = 2.0;
    Eigen::MatrixXd bad_a = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd bad_b = Eigen::MatrixXd::Zero(2, 2);
    bad_a(0, 0) = 1.5;
    bad_b(0, 0) = 0.5;
    auto reference = std::make_shared<qcd::Wavefunction>(
        std::make_unique<qcd::StateVectorContainer>(
            Eigen::VectorXd(Eigen::VectorXd::Ones(1)), dets, r_ref,
            std::optional<qcd::StateVectorContainer::MatrixVariant>{
                good_traced},
            std::optional<qcd::StateVectorContainer::MatrixVariant>{bad_a},
            std::optional<qcd::StateVectorContainer::MatrixVariant>{bad_b},
            std::nullopt, std::nullopt, std::nullopt, std::nullopt));
    EXPECT_THROW(EffectiveHamiltonianConstructorFactory::create("qdk_dsrg_pt2")
                     ->run(reference, window,
                           testing::restricted_index_set(norb, {1, 2})),
                 std::invalid_argument);
  }
}
}  // namespace

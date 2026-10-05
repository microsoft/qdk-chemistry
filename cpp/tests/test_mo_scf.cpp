// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <limits>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mo_scf.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/utils/orbital_rotation.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

namespace {

std::shared_ptr<Hamiltonian> model_hamiltonian(
    double core_energy = 0.4, std::shared_ptr<Orbitals> orbitals = nullptr,
    const Eigen::MatrixXd& inactive_fock = Eigen::MatrixXd{}) {
  Eigen::MatrixXd h(3, 3);
  h << -1.3, 0.25, -0.1, 0.25, -0.7, 0.15, -0.1, 0.15, 0.3;
  std::array<Eigen::Matrix3d, 2> factors;
  factors[0] << 0.6, 0.04, -0.03, 0.04, 0.5, 0.02, -0.03, 0.02, 0.4;
  factors[1] << 0.1, -0.05, 0.02, -0.05, 0.2, 0.03, 0.02, 0.03, 0.15;
  Eigen::VectorXd g(81);
  for (int p = 0; p < 3; ++p)
    for (int q = 0; q < 3; ++q)
      for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s)
          g(((p * 3 + q) * 3 + r) * 3 + s) =
              factors[0](p, q) * factors[0](r, s) +
              factors[1](p, q) * factors[1](r, s);
  if (!orbitals) {
    orbitals = std::make_shared<ModelOrbitals>(
        3, std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial()));
  }
  return std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          h, g, orbitals, core_energy, inactive_fock));
}

std::array<Eigen::MatrixXd, 2> reference_focks(const Hamiltonian& h,
                                               unsigned na, unsigned nb) {
  const auto& [ha, hb] = h.get_one_body_integrals();
  std::array<Eigen::MatrixXd, 2> f{ha, hb};
  const int n = ha.rows();
  for (int p = 0; p < n; ++p) {
    for (int q = 0; q < n; ++q) {
      for (unsigned i = 0; i < na; ++i) {
        f[0](p, q) += h.get_two_body_element(p, q, i, i, SpinChannel::aaaa) -
                      h.get_two_body_element(p, i, i, q, SpinChannel::aaaa);
        f[1](p, q) += h.get_two_body_element(i, i, p, q, SpinChannel::aabb);
      }
      for (unsigned i = 0; i < nb; ++i) {
        f[1](p, q) += h.get_two_body_element(p, q, i, i, SpinChannel::bbbb) -
                      h.get_two_body_element(p, i, i, q, SpinChannel::bbbb);
        f[0](p, q) += h.get_two_body_element(p, q, i, i, SpinChannel::aabb);
      }
    }
  }
  return f;
}

double reference_energy(const Hamiltonian& h, unsigned na, unsigned nb) {
  auto f = reference_focks(h, na, nb);
  const auto& [ha, hb] = h.get_one_body_integrals();
  double energy = h.get_core_energy();
  for (unsigned i = 0; i < na; ++i) energy += 0.5 * (ha(i, i) + f[0](i, i));
  for (unsigned i = 0; i < nb; ++i) energy += 0.5 * (hb(i, i) + f[1](i, i));
  return energy;
}

std::unique_ptr<MoScfSolver> make_solver(const std::string& algorithm,
                                         const std::string& scf_type) {
  auto solver = MoScfSolverFactory::create();
  solver->settings().set("scf_algorithm", algorithm);
  solver->settings().set("scf_type", scf_type);
  solver->settings().set("convergence_threshold", 1e-8);
  solver->settings().set("max_iterations", 300);
  solver->settings().set("gdm_max_diis_iteration", 2);
  return solver;
}

class MoScfTest
    : public ::testing::TestWithParam<std::tuple<std::string, std::string>> {};

TEST_P(MoScfTest, OptimizesAndTransformsIntegrals) {
  const auto& [algorithm, reference] = GetParam();
  const unsigned na = reference == "rhf" ? 1 : 2;
  const unsigned nb = 1;
  auto input = model_hamiltonian();
  const auto original_hash = input->content_hash();
  auto solver = make_solver(algorithm,
                            reference == "uhf" ? "unrestricted" : "restricted");
  auto [energy, ansatz] = solver->run(input, na, nb);
  auto h = ansatz->get_hamiltonian();
  const auto orbitals = ansatz->get_orbitals();
  EXPECT_EQ(input->content_hash(), original_hash);
  EXPECT_EQ(orbitals->is_unrestricted(), reference == "uhf");
  EXPECT_EQ(ansatz->get_wavefunction()->size(), 1);
  EXPECT_NEAR(energy, reference_energy(*h, na, nb),
              testing::scf_energy_tolerance);
  EXPECT_NEAR(energy, ansatz->calculate_energy(),
              testing::scf_energy_tolerance);
  const double reference_energy_pyscf = reference == "rhf" ? -2.0799593408671955
                                        : reference == "rohf"
                                            ? -2.0787773791389976
                                            : -2.0787780784704446;
  EXPECT_NEAR(energy, reference_energy_pyscf, testing::scf_energy_tolerance);
  EXPECT_LT(energy, reference_energy(*input, na, nb) - 1e-4);
  EXPECT_DOUBLE_EQ(h->get_core_energy(), 0.4);

  const auto& ca =
      orbitals->coefficients()->block({axes::alpha(), axes::alpha()});
  const auto& cb =
      orbitals->coefficients()->block({axes::beta(), axes::beta()});
  EXPECT_LT((ca.transpose() * ca - Eigen::Matrix3d::Identity()).norm(), 1e-10);
  EXPECT_LT((cb.transpose() * cb - Eigen::Matrix3d::Identity()).norm(), 1e-10);
  const auto& [h0, unused] = input->get_one_body_integrals();
  const auto& [ha, hb] = h->get_one_body_integrals();
  EXPECT_LT((ha - ca.transpose() * h0 * ca).norm(), 1e-12);
  EXPECT_LT((hb - cb.transpose() * h0 * cb).norm(), 1e-12);

  // Direct small-tensor contraction is independent of the production MOERI
  // path.
  for (int p = 0; p < 3; ++p)
    for (int q = 0; q < 3; ++q)
      for (int r = 0; r < 3; ++r)
        for (int s = 0; s < 3; ++s) {
          double expected = 0.0;
          for (int i = 0; i < 3; ++i)
            for (int j = 0; j < 3; ++j)
              for (int k = 0; k < 3; ++k)
                for (int l = 0; l < 3; ++l)
                  expected += ca(i, p) * ca(j, q) * cb(k, r) * cb(l, s) *
                              input->get_two_body_element(i, j, k, l);
          EXPECT_NEAR(h->get_two_body_element(p, q, r, s, SpinChannel::aabb),
                      expected, 1e-12);
        }

  const auto f = reference_focks(*h, na, nb);
  if (reference == "rohf") {
    EXPECT_LT(std::abs(f[1](0, 1)), 1e-7);
    EXPECT_LT(std::abs(f[0](1, 2)), 1e-7);
    EXPECT_LT(std::abs(f[0](0, 2) + f[1](0, 2)), 1e-7);
  } else {
    EXPECT_LT(f[0].block(0, na, na, 3 - na).cwiseAbs().maxCoeff(), 1e-7);
    EXPECT_LT(f[1].block(0, nb, nb, 3 - nb).cwiseAbs().maxCoeff(), 1e-7);
  }
}

INSTANTIATE_TEST_SUITE_P(
    NativeAlgorithms, MoScfTest,
    ::testing::Combine(::testing::Values("diis", "gdm", "diis_gdm"),
                       ::testing::Values("rhf", "rohf", "uhf")));

TEST(MoScf, EmptyAndFullSpinChannels) {
  for (const auto& algorithm : {"diis", "gdm", "diis_gdm"}) {
    for (const auto& counts :
         {std::pair{0u, 0u}, {1u, 0u}, {0u, 1u}, {3u, 0u}, {3u, 3u}}) {
      SCOPED_TRACE(algorithm);
      SCOPED_TRACE(::testing::PrintToString(counts));
      auto solver = make_solver(algorithm, "unrestricted");
      auto [energy, ansatz] =
          solver->run(model_hamiltonian(), counts.first, counts.second);
      EXPECT_NEAR(energy,
                  reference_energy(*ansatz->get_hamiltonian(), counts.first,
                                   counts.second),
                  testing::scf_energy_tolerance);
      const auto& c = ansatz->get_orbitals()->coefficients()->block(
          {axes::beta(), axes::beta()});
      EXPECT_EQ(c.rows(), 3);
      EXPECT_EQ(c.cols(), 3);
      EXPECT_LT((c.transpose() * c - Eigen::Matrix3d::Identity()).norm(),
                1e-10);
    }
  }
}

TEST(MoScf, RohfEmptySubspaces) {
  for (const auto& algorithm : {"diis", "gdm", "diis_gdm"}) {
    for (const auto& counts :
         {std::pair{1u, 0u}, {2u, 0u}, {3u, 1u}, {3u, 2u}}) {
      SCOPED_TRACE(algorithm);
      SCOPED_TRACE(::testing::PrintToString(counts));
      auto solver = make_solver(algorithm, "restricted");
      auto [energy, ansatz] =
          solver->run(model_hamiltonian(), counts.first, counts.second);
      const auto hamiltonian = ansatz->get_hamiltonian();
      EXPECT_NEAR(energy,
                  reference_energy(*hamiltonian, counts.first, counts.second),
                  testing::scf_energy_tolerance);
      EXPECT_TRUE(ansatz->get_orbitals()->is_restricted());
      const auto& c = ansatz->get_orbitals()->coefficients()->block(
          {axes::alpha(), axes::alpha()});
      EXPECT_LT((c.transpose() * c - Eigen::Matrix3d::Identity()).norm(),
                1e-10);
      Eigen::Matrix3d da = Eigen::Matrix3d::Zero();
      Eigen::Matrix3d db = Eigen::Matrix3d::Zero();
      da.diagonal().head(counts.first).setOnes();
      db.diagonal().head(counts.second).setOnes();
      const auto f = reference_focks(*hamiltonian, counts.first, counts.second);
      const Eigen::Matrix3d gradient =
          f[0] * da - da * f[0] + f[1] * db - db * f[1];
      EXPECT_LT(gradient.cwiseAbs().maxCoeff(), 1e-7);
    }
  }
}

TEST(MoScf, LevelShiftDoesNotChangePhysicalOrbitalEnergies) {
  Eigen::Matrix2d h;
  h << -1.0, 0.2, 0.2, 0.5;
  auto orbitals = std::make_shared<ModelOrbitals>(2, testing::spin_symmetry());
  auto input = std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          h, Eigen::VectorXd::Zero(16), orbitals, 0.0, Eigen::MatrixXd{}));
  const Eigen::Vector2d expected(-0.25 - std::hypot(0.75, 0.2),
                                 -0.25 + std::hypot(0.75, 0.2));
  for (const auto& scf_type : {"restricted", "unrestricted"}) {
    for (const auto& algorithm : {"diis", "gdm", "diis_gdm"}) {
      SCOPED_TRACE(scf_type);
      SCOPED_TRACE(algorithm);
      auto solver = make_solver(algorithm, scf_type);
      solver->settings().set("level_shift", 0.5);
      auto [energy, ansatz] = solver->run(input, 1, 1);
      EXPECT_NEAR(energy, 2.0 * expected(0), testing::scf_energy_tolerance);
      const auto f = reference_focks(*ansatz->get_hamiltonian(), 1, 1);
      for (int spin = 0; spin < 2; ++spin) {
        const auto label = spin == 0 ? axes::alpha() : axes::beta();
        const auto& energies =
            ansatz->get_orbitals()->energies()->block({label});
        EXPECT_LT((energies - expected).cwiseAbs().maxCoeff(), 1e-10);
        EXPECT_LT((energies - f[spin].diagonal()).cwiseAbs().maxCoeff(), 1e-10);
      }
    }
  }
}

TEST(MoScf, PreservesInactiveAndExternalOrbitals) {
  Eigen::MatrixXd c = Eigen::MatrixXd::Identity(6, 5);
  Eigen::VectorXd energies = Eigen::VectorXd::LinSpaced(5, -2.0, 2.0);
  auto orbitals =
      std::make_shared<Orbitals>(c, energies, Eigen::MatrixXd::Identity(6, 6),
                                 testing::create_random_basis_set(6),
                                 testing::restricted_index_set(5, {1, 3, 4}),
                                 testing::restricted_index_set(5, {0}));
  Eigen::MatrixXd inactive_fock = Eigen::MatrixXd::Identity(5, 5);
  inactive_fock(0, 3) = inactive_fock(3, 0) = 0.2;
  auto input = model_hamiltonian(-2.0, orbitals, inactive_fock);
  auto solver = make_solver("gdm", "unrestricted");
  auto [energy, ansatz] = solver->run(input, 2, 1);
  EXPECT_NEAR(energy, ansatz->calculate_energy(),
              testing::scf_energy_tolerance);
  const auto out = ansatz->get_orbitals();
  EXPECT_FALSE(out->has_energies());
  EXPECT_EQ(spin_channel_indices(out->active_indices(), axes::alpha()),
            (std::vector<size_t>{1, 3, 4}));
  const auto& [fa, fb] = ansatz->get_hamiltonian()->get_inactive_fock_matrix();
  for (int spin_index = 0; spin_index < 2; ++spin_index) {
    const auto spin = spin_index == 0 ? axes::alpha() : axes::beta();
    const auto& rotated = out->coefficients()->block({spin, spin});
    EXPECT_EQ(rotated.rows(), 6);
    EXPECT_TRUE(rotated.col(0).isApprox(c.col(0)));
    EXPECT_TRUE(rotated.col(2).isApprox(c.col(2)));
    const Eigen::MatrixXd rotation = c.transpose() * rotated;
    const auto& fock = spin_index == 0 ? fa : fb;
    EXPECT_LT((fock - rotation.transpose() * inactive_fock * rotation).norm(),
              1e-12);
  }
}

TEST(MoScf, SameSolutionAsMolecularScf) {
  auto scf = ScfSolverFactory::create();
  scf->settings().set("convergence_threshold", 1e-8);
  auto [ao_energy, wfn] =
      scf->run(testing::create_water_structure(), 0, 1, "sto-3g");
  auto h = HamiltonianConstructorFactory::create()->run(wfn->get_orbitals());
  for (const auto& algorithm : {"diis", "gdm", "diis_gdm"}) {
    auto solver = make_solver(algorithm, "restricted");
    auto [energy, ansatz] = solver->run(h, 5, 5);
    EXPECT_NEAR(energy, ao_energy, testing::scf_energy_tolerance);
    EXPECT_NEAR(energy, ansatz->calculate_energy(),
                testing::scf_energy_tolerance);
  }
  for (bool use_initial_orbitals : {false, true}) {
    auto pure_gdm = ScfSolverFactory::create();
    pure_gdm->settings().set("scf_algorithm", "gdm");
    pure_gdm->settings().set("convergence_threshold", 1e-7);
    pure_gdm->settings().set("max_iterations", 300);
    const auto rotated_guess = qdk::chemistry::utils::rotate_orbitals(
        wfn->get_orbitals(), Eigen::VectorXd::Constant(10, 0.05), 5, 5);
    BasisOrGuessType guess = use_initial_orbitals
                                 ? BasisOrGuessType(rotated_guess)
                                 : BasisOrGuessType(std::string("sto-3g"));
    auto [energy, optimized] =
        pure_gdm->run(testing::create_water_structure(), 0, 1, guess);
    EXPECT_NEAR(energy, ao_energy, testing::scf_energy_tolerance);
    const auto& c = optimized->get_orbitals()->coefficients()->block(
        {axes::alpha(), axes::alpha()});
    const auto& s = optimized->get_orbitals()->get_overlap_matrix();
    EXPECT_LT(
        (c.transpose() * s * c - Eigen::MatrixXd::Identity(c.cols(), c.cols()))
            .norm(),
        1e-9);
  }
}

TEST(MoScf, CanonicalizationPreservesStationaryDeterminant) {
  Eigen::Matrix2d h = Eigen::Vector2d(1.0, -1.0).asDiagonal();
  auto orbitals = std::make_shared<ModelOrbitals>(2, testing::spin_symmetry());
  auto input = std::make_shared<Hamiltonian>(
      std::make_unique<CanonicalFourCenterHamiltonianContainer>(
          h, Eigen::VectorXd::Zero(16), orbitals, 0.0, Eigen::MatrixXd{}));
  for (const auto& algorithm : {"diis", "gdm", "diis_gdm"}) {
    SCOPED_TRACE(algorithm);
    auto solver = make_solver(algorithm, "restricted");
    solver->settings().set("level_shift", 2.0);
    auto [energy, ansatz] = solver->run(input, 1, 1);
    EXPECT_NEAR(energy, 2.0, testing::scf_energy_tolerance);
    EXPECT_NEAR(ansatz->calculate_energy(), energy,
                testing::scf_energy_tolerance);
    const auto& e = ansatz->get_orbitals()->energies()->block({axes::alpha()});
    EXPECT_NEAR(e(0), 1.0, testing::numerical_zero_tolerance);
    EXPECT_NEAR(e(1), -1.0, testing::numerical_zero_tolerance);
  }
}

TEST(MoScf, GdmHandlesReducedOverlapRank) {
  const auto structure = testing::create_water_structure();
  const auto basis = BasisSet::from_basis_name("sto-3g", structure);
  auto shells = basis->get_shells();
  shells.push_back(shells.back());
  auto redundant = std::make_shared<BasisSet>("custom", shells, structure);
  auto [reference_energy_ao, reference_wfn] =
      ScfSolverFactory::create()->run(structure, 0, 1, "sto-3g");
  const auto perturbed = qdk::chemistry::utils::rotate_orbitals(
      reference_wfn->get_orbitals(), Eigen::VectorXd::Constant(10, 0.05), 5, 5);
  Eigen::MatrixXd initial = Eigen::MatrixXd::Zero(8, 7);
  initial.topRows(7) =
      perturbed->coefficients()->block({axes::alpha(), axes::alpha()});
  auto guess = std::make_shared<Orbitals>(initial, std::nullopt, std::nullopt,
                                          redundant);
  for (const auto& algorithm : {"diis", "gdm", "diis_gdm"}) {
    auto solver = ScfSolverFactory::create();
    solver->settings().set("scf_algorithm", algorithm);
    solver->settings().set("convergence_threshold", 1e-7);
    solver->settings().set("max_iterations", 300);
    auto [energy, wfn] = solver->run(structure, 0, 1, guess);
    EXPECT_NEAR(energy, reference_energy_ao, testing::scf_energy_tolerance);
    auto orbitals = wfn->get_orbitals();
    EXPECT_EQ(orbitals->get_num_molecular_orbitals(), 7);
    EXPECT_EQ(orbitals->get_num_atomic_orbitals(), 8);
    const auto& c =
        orbitals->coefficients()->block({axes::alpha(), axes::alpha()});
    const auto& s = orbitals->get_overlap_matrix();
    EXPECT_LT((c.transpose() * s * c - Eigen::MatrixXd::Identity(7, 7)).norm(),
              1e-9);
    auto h = HamiltonianConstructorFactory::create()->run(orbitals);
    EXPECT_NEAR(energy, reference_energy(*h, 5, 5),
                testing::scf_energy_tolerance);
  }
}

TEST(MoScf, FrozenSpaceWithoutInactiveFockRoundTrips) {
  auto orbitals = std::make_shared<ModelOrbitals>(
      testing::restricted_index_set(4, {1, 2, 3}),
      testing::restricted_index_set(4, {0}));
  auto input = model_hamiltonian(-2.0, orbitals);
  auto [energy, ansatz] = make_solver("diis", "restricted")->run(input, 1, 1);
  auto restored = Ansatz::from_json(ansatz->to_json());
  EXPECT_EQ(restored->content_hash(), ansatz->content_hash());
  EXPECT_FALSE(restored->get_orbitals()->has_energies());
  EXPECT_FALSE(restored->get_hamiltonian()->has_inactive_fock_matrix());
  EXPECT_NEAR(restored->calculate_energy(), energy,
              testing::scf_energy_tolerance);
}

TEST(MoScf, ValidationAndNonconvergence) {
  auto input = model_hamiltonian();
  EXPECT_THROW(make_solver("diis", "restricted")->run(nullptr, 1, 1),
               std::invalid_argument);
  EXPECT_THROW(make_solver("diis", "restricted")->run(input, 4, 1),
               std::invalid_argument);
  EXPECT_THROW(make_solver("diis", "restricted")->run(input, 1, 2),
               std::invalid_argument);
  auto dft = make_solver("diis", "restricted");
  dft->settings().set("method", "pbe");
  EXPECT_THROW(dft->run(input, 1, 1), std::invalid_argument);
  auto unconverged = make_solver("diis", "restricted");
  unconverged->settings().set("max_iterations", 1);
  EXPECT_THROW(unconverged->run(input, 1, 1), std::runtime_error);
  auto invalid_threshold = make_solver("diis", "restricted");
  invalid_threshold->settings().set("convergence_threshold",
                                    std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(invalid_threshold->run(input, 1, 1), std::invalid_argument);
  auto [energy, ansatz] = make_solver("diis", "unrestricted")->run(input, 2, 1);
  EXPECT_THROW(
      make_solver("diis", "unrestricted")->run(ansatz->get_hamiltonian(), 2, 1),
      std::invalid_argument);
}

TEST(MoScf, RejectsInvalidIntegralData) {
  const auto input = model_hamiltonian();
  const auto& [h, hb] = input->get_one_body_integrals();
  const auto& [g, gab, gbb] = input->get_two_body_integrals();
  for (const auto& invalid_data : {"one_body", "two_body", "nan", "core"}) {
    SCOPED_TRACE(invalid_data);
    Eigen::MatrixXd one_body = h;
    Eigen::VectorXd two_body = g;
    double core = 0.4;
    if (std::string(invalid_data) == "one_body") one_body(0, 1) += 0.1;
    if (std::string(invalid_data) == "two_body") two_body(1) += 0.1;
    if (std::string(invalid_data) == "nan")
      two_body(0) = std::numeric_limits<double>::quiet_NaN();
    if (std::string(invalid_data) == "core")
      core = std::numeric_limits<double>::infinity();
    auto invalid = std::make_shared<Hamiltonian>(
        std::make_unique<CanonicalFourCenterHamiltonianContainer>(
            one_body, two_body, input->get_orbitals(), core,
            Eigen::MatrixXd{}));
    EXPECT_THROW(make_solver("diis", "restricted")->run(invalid, 1, 1),
                 std::invalid_argument);
  }
}

}  // namespace

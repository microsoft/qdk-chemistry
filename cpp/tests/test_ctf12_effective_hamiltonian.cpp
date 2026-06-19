// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <libint2.hpp>
#include <memory>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/algorithms/f12_scf.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <string>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/ctf12_f12.hpp"
#include "qdk/chemistry/algorithms/microsoft/ctf12_support.hpp"
#include "test_config.h"

using namespace qdk::chemistry;
namespace ctf12 = qdk::chemistry::algorithms::microsoft::ctf12;

namespace {

// Cross-checks the dressed CT-F12 Hamiltonian emitted by the
// CtF12HamiltonianConstructor against the validated transcorrelated F12-MP2
// energy (Comment on J. Chem. Phys. 136, 084107, Table I): the generic
// MP2Calculator run over the emitted Hamiltonian (relaxed F12-HF basis) must
// reproduce the MP2 residual of f12_mp2_energy. The two share identical
// orbitals and integrals, so they agree to machine precision (only the
// floating-point summation order differs); the determinant energy reproduces
// the F12-HF reference energy.
void run_neon_effective_hamiltonian_mp2(const std::string& obs_name,
                                        const std::string& cabs_name,
                                        double tol) {
  scf::QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  ::libint2::initialize();

  constexpr double gamma = 1.5;
  constexpr std::int64_t frozen_core = 1;

  Eigen::MatrixXd coords = Eigen::MatrixXd::Zero(1, 3);
  auto structure =
      std::make_shared<data::Structure>(coords, std::vector<std::string>{"Ne"});

  auto scf_solver = algorithms::ScfSolverFactory::create("qdk");
  const auto reference = scf_solver->run(structure, 0, 1, obs_name).second;

  auto constructor =
      algorithms::EffectiveHamiltonianConstructorFactory::create("qdk_ct_f12");
  constructor->settings().set("gamma", gamma);
  constructor->settings().set("frozen_core", frozen_core);
  constructor->settings().set("cabs_basis", cabs_name);
  constructor->settings().set("orbital_basis", std::string("relaxed"));
  auto dressed_hamiltonian = constructor->run(reference);

  // Closed-shell HF determinant over the frozen-core active space of the
  // relaxed F12-HF orbitals carried by the emitted Hamiltonian.
  auto orbitals = dressed_hamiltonian->get_orbitals();
  const std::size_t n_active =
      orbitals->get_active_space_indices().first.size();
  const std::size_t n_occupied = reference->get_total_num_electrons().first;
  const std::size_t n_active_occupied =
      n_occupied - static_cast<std::size_t>(frozen_core);

  std::string config_str(n_active, '0');
  for (std::size_t i = 0; i < n_active_occupied; ++i) config_str[i] = '2';
  auto det = data::Configuration::from_spin_half_string(config_str);
  auto container =
      std::make_unique<data::StateVectorContainer>(det, orbitals, "electrons");
  auto relaxed_reference =
      std::make_shared<data::Wavefunction>(std::move(container));

  auto ansatz =
      std::make_shared<data::Ansatz>(*dressed_hamiltonian, *relaxed_reference);
  auto mp2 = algorithms::DynamicalCorrelationCalculatorFactory::create(
      "qdk_mp2_calculator");
  auto [mp2_total_energy, ket, bra] = mp2->run(ansatz);

  const double reference_energy = ansatz->calculate_energy();
  const double e_corr = mp2_total_energy - reference_energy;

  // Reference values from the validated transcorrelated energy routines, which
  // share the same orbitals and integrals as the emitted Hamiltonian.
  const ctf12::F12HartreeFockInput input = ctf12::f12_input_from_wavefunction(
      *reference, gamma, cabs_name, static_cast<std::size_t>(frozen_core));
  const double expected_residual =
      ctf12::f12_mp2_energy(input) - ctf12::f12_hf_scf_energy(input);

  EXPECT_NEAR(e_corr, expected_residual, tol)
      << obs_name << ": MP2 over dressed Hamiltonian " << e_corr
      << " vs F12-MP2 residual " << expected_residual;

  // The emitted Hamiltonian's reference energy is the self-consistent F12-HF
  // energy of the dressed mean field (nuclear repulsion vanishes for an atom).
  const ctf12::DressedHamiltonian dressed =
      ctf12::build_dressed_hamiltonian(input, /*relax_orbitals=*/true);
  const double nuclear_repulsion =
      structure->calculate_nuclear_repulsion_energy();
  EXPECT_NEAR(reference_energy, dressed.e_f12hf + nuclear_repulsion, 1e-11)
      << obs_name << ": F12-HF reference energy " << reference_energy;

  ::libint2::finalize();
}

}  // namespace

TEST(CtF12EffectiveHamiltonian, NeonAugCcPvdzMp2) {
  run_neon_effective_hamiltonian_mp2("aug-cc-pvdz", "aug-cc-pvdz-optri", 1e-12);
}

namespace {

// Cross-checks the full F12HartreeFockSolver -> CtF12HamiltonianConstructor ->
// MP2Calculator pipeline: the relaxed reference emitted by the F12-HF SCF
// module paired with the dressed Hamiltonian must reproduce the validated
// F12-MP2 energy to machine precision, exercising the SCF module end-to-end.
void run_neon_f12_scf_module_mp2(const std::string& obs_name,
                                 const std::string& cabs_name, double tol) {
  scf::QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  ::libint2::initialize();

  constexpr double gamma = 1.5;
  constexpr std::int64_t frozen_core = 1;

  Eigen::MatrixXd coords = Eigen::MatrixXd::Zero(1, 3);
  auto structure =
      std::make_shared<data::Structure>(coords, std::vector<std::string>{"Ne"});

  auto scf_solver = algorithms::ScfSolverFactory::create("qdk");
  const auto reference = scf_solver->run(structure, 0, 1, obs_name).second;

  // Relaxed F12-HF reference from the SCF module.
  auto f12_scf = algorithms::F12HartreeFockSolverFactory::create("qdk_ct_f12");
  f12_scf->settings().set("gamma", gamma);
  f12_scf->settings().set("frozen_core", frozen_core);
  f12_scf->settings().set("cabs_basis", cabs_name);
  auto relaxed_reference = f12_scf->run(reference);

  // Dressed Hamiltonian in the same relaxed basis.
  auto constructor =
      algorithms::EffectiveHamiltonianConstructorFactory::create("qdk_ct_f12");
  constructor->settings().set("gamma", gamma);
  constructor->settings().set("frozen_core", frozen_core);
  constructor->settings().set("cabs_basis", cabs_name);
  constructor->settings().set("orbital_basis", std::string("relaxed"));
  auto dressed_hamiltonian = constructor->run(reference);

  auto ansatz =
      std::make_shared<data::Ansatz>(*dressed_hamiltonian, *relaxed_reference);
  auto mp2 = algorithms::DynamicalCorrelationCalculatorFactory::create(
      "qdk_mp2_calculator");
  auto [mp2_total_energy, ket, bra] = mp2->run(ansatz);
  const double e_corr = mp2_total_energy - ansatz->calculate_energy();

  const ctf12::F12HartreeFockInput input = ctf12::f12_input_from_wavefunction(
      *reference, gamma, cabs_name, static_cast<std::size_t>(frozen_core));
  const double expected_residual =
      ctf12::f12_mp2_energy(input) - ctf12::f12_hf_scf_energy(input);

  EXPECT_NEAR(e_corr, expected_residual, tol)
      << obs_name << ": SCF-module pipeline MP2 " << e_corr << " vs F12-MP2 "
      << expected_residual;

  ::libint2::finalize();
}

}  // namespace

TEST(CtF12EffectiveHamiltonian, NeonAugCcPvdzScfModuleMp2) {
  run_neon_f12_scf_module_mp2("aug-cc-pvdz", "aug-cc-pvdz-optri", 1e-12);
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <libint2.hpp>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/auxiliary_basis.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/ctf12_f12.hpp"
#include "qdk/chemistry/algorithms/microsoft/ctf12_hamiltonian.hpp"
#include "test_config.h"
#include "ut_common.hpp"

using namespace qdk::chemistry;
namespace ctf12 = qdk::chemistry::algorithms::microsoft::ctf12;

namespace {

// Bare molecular Hamiltonian over the whole orbital window W.
std::shared_ptr<data::Hamiltonian> bare_hamiltonian(
    const std::shared_ptr<data::Wavefunction>& reference) {
  return algorithms::HamiltonianConstructorFactory::create()->run(
      reference->get_orbitals());
}

// Target P-space spanning every molecular orbital above the frozen core.
std::shared_ptr<const data::SymmetryBlockedIndexSet> valence_p_space(
    const std::shared_ptr<data::Wavefunction>& reference,
    std::size_t frozen_core) {
  const std::size_t n_mo =
      reference->get_orbitals()->get_num_molecular_orbitals();
  std::vector<std::size_t> indices;
  for (std::size_t i = frozen_core; i < n_mo; ++i) indices.push_back(i);
  return testing::restricted_index_set(n_mo, indices);
}

// CT-F12's external space: the named OptRI/CABS auxiliary basis.
std::shared_ptr<const data::AuxiliaryBasisCollection> cabs_bases(
    const std::shared_ptr<data::Wavefunction>& reference,
    const std::string& cabs_name) {
  auto structure = reference->get_orbitals()->get_basis_set()->get_structure();
  return std::make_shared<data::AuxiliaryBasisCollection>(
      data::AuxiliaryBasisCollection::Map{
          {data::AuxiliaryBasisRole::CABS,
           data::AuxiliaryBasis::from_basis_name(cabs_name,
                                                 std::move(structure))}});
}

// Cross-checks the full CtF12ScfSolver -> CtF12HamiltonianConstructor ->
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

  // Relaxed F12-HF reference from the CT-F12 SCF solver.
  auto f12_scf = algorithms::ScfSolverFactory::create("qdk_ct_f12");
  f12_scf->settings().set("gamma", gamma);
  f12_scf->settings().set("frozen_core", frozen_core);
  f12_scf->settings().set("cabs_basis", cabs_name);
  const auto relaxed_reference = f12_scf->run(structure, 0, 1, obs_name).second;

  // Dressed Hamiltonian in the same relaxed basis.
  auto constructor =
      algorithms::EffectiveHamiltonianConstructorFactory::create("qdk_ct_f12");
  constructor->settings().set("gamma", gamma);
  constructor->settings().set("frozen_core", frozen_core);
  constructor->settings().set("orbital_basis", std::string("relaxed"));
  auto dressed_hamiltonian = constructor->run(
      reference, bare_hamiltonian(reference),
      valence_p_space(reference, static_cast<std::size_t>(frozen_core)),
      cabs_bases(reference, cabs_name));

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

TEST(CtF12ScfSolver, NeonAugCcPvdzScfModuleMp2) {
  run_neon_f12_scf_module_mp2("aug-cc-pvdz", "aug-cc-pvdz-optri", 1e-12);
}

TEST(CtF12ScfSolver, N2AugCcPvdzEnergyUsesCanonicalTotalPlusF12Correction) {
  scf::QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  ::libint2::initialize();

  constexpr double gamma = 1.0;
  constexpr std::int64_t frozen_core = 2;
  const std::string obs_name = "aug-cc-pvdz";
  const std::string cabs_name = "aug-cc-pvdz-optri";

  auto structure = testing::create_stretched_n2_structure(1.2);
  ASSERT_GT(structure->calculate_nuclear_repulsion_energy(), 0.0);

  auto scf_solver = algorithms::ScfSolverFactory::create("qdk");
  const auto [e_hf, reference] = scf_solver->run(structure, 0, 1, obs_name);

  const ctf12::F12HartreeFockInput input = ctf12::f12_input_from_wavefunction(
      *reference, gamma, cabs_name, static_cast<std::size_t>(frozen_core));
  const double f12_correction = ctf12::f12_hf_scf_energy(input);
  const double expected_total = e_hf + f12_correction;

  auto f12_scf = algorithms::ScfSolverFactory::create("qdk_ct_f12");
  f12_scf->settings().set("gamma", gamma);
  f12_scf->settings().set("frozen_core", frozen_core);
  f12_scf->settings().set("cabs_basis", cabs_name);
  const auto [actual_total, relaxed_reference] =
      f12_scf->run(structure, 0, 1, obs_name);

  std::cout << std::setprecision(16) << "N2 " << obs_name
            << " HF total energy: " << e_hf << '\n'
            << "N2 " << obs_name << " F12-HF total energy: " << actual_total
            << '\n'
            << "N2 " << obs_name << " F12-HF correction: " << f12_correction
            << '\n';

  EXPECT_NEAR(actual_total, expected_total, 1e-8)
      << "N2 F12-SCF total energy should be canonical total plus the "
         "self-consistent F12-HF correction\n"
      << "HF total energy: " << e_hf << '\n'
      << "F12-HF total energy: " << actual_total << '\n'
      << "Expected F12-HF total energy: " << expected_total << '\n'
      << "F12-HF correction: " << f12_correction;
  ASSERT_NE(relaxed_reference, nullptr);

  ::libint2::finalize();
}

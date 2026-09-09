// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/util/cabs.h>
#include <qdk/chemistry/scf/util/geminal_eri.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <libint2.hpp>
#include <limits>
#include <memory>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/auxiliary_basis.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <stdexcept>
#include <string>
#include <tuple>
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

// Target P-space spanning the molecular orbitals [first, last).
std::shared_ptr<const data::SymmetryBlockedIndexSet> p_space(
    const std::shared_ptr<data::Wavefunction>& reference, std::size_t first,
    std::size_t last) {
  const std::size_t n_mo =
      reference->get_orbitals()->get_num_molecular_orbitals();
  std::vector<std::size_t> indices;
  for (std::size_t i = first; i < std::min(last, n_mo); ++i)
    indices.push_back(i);
  return testing::restricted_index_set(n_mo, indices);
}

// Target P-space spanning every molecular orbital above the frozen core.
std::shared_ptr<const data::SymmetryBlockedIndexSet> valence_p_space(
    const std::shared_ptr<data::Wavefunction>& reference,
    std::size_t frozen_core) {
  return p_space(reference, frozen_core,
                 std::numeric_limits<std::size_t>::max());
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

std::shared_ptr<data::Hamiltonian> restrict_dressed_virtual_space(
    const std::shared_ptr<data::Hamiltonian>& hamiltonian,
    const std::vector<std::size_t>& selected_active_indices) {
  if (!hamiltonian || !hamiltonian->is_restricted()) {
    throw std::invalid_argument(
        "CT-F12 active-space test requires a restricted Hamiltonian");
  }

  const auto orbitals = hamiltonian->get_orbitals();
  const auto full_active_alpha = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::alpha());
  const auto full_active_beta = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::beta());
  const auto inactive_alpha = data::spin_channel_indices(
      orbitals->inactive_indices(), data::axes::alpha());
  const auto inactive_beta = data::spin_channel_indices(
      orbitals->inactive_indices(), data::axes::beta());
  if (full_active_alpha != full_active_beta ||
      inactive_alpha != inactive_beta) {
    throw std::invalid_argument(
        "CT-F12 active-space test requires matching spin spaces");
  }

  std::vector<std::size_t> source_positions;
  source_positions.reserve(selected_active_indices.size());
  for (const std::size_t index : selected_active_indices) {
    const auto position =
        std::find(full_active_alpha.begin(), full_active_alpha.end(), index);
    if (position == full_active_alpha.end()) {
      throw std::invalid_argument(
          "Selected orbital is not active in the dressed Hamiltonian");
    }
    source_positions.push_back(static_cast<std::size_t>(
        std::distance(full_active_alpha.begin(), position)));
  }

  const std::size_t n_active = selected_active_indices.size();
  Eigen::MatrixXd one_body(n_active, n_active);
  Eigen::VectorXd two_body(
      static_cast<Eigen::Index>(n_active * n_active * n_active * n_active));
  for (std::size_t p = 0; p < n_active; ++p)
    for (std::size_t q = 0; q < n_active; ++q) {
      one_body(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q)) =
          hamiltonian->get_one_body_element(source_positions[p],
                                            source_positions[q]);
      for (std::size_t r = 0; r < n_active; ++r)
        for (std::size_t s = 0; s < n_active; ++s)
          two_body(static_cast<Eigen::Index>(
              ((p * n_active + q) * n_active + r) * n_active + s)) =
              hamiltonian->get_two_body_element(
                  source_positions[p], source_positions[q], source_positions[r],
                  source_positions[s]);
    }

  const auto active_orbitals = testing::with_active_space(
      orbitals, selected_active_indices, inactive_alpha);
  const Eigen::MatrixXd inactive_fock =
      hamiltonian->get_inactive_fock_matrix().first;
  return std::make_shared<data::Hamiltonian>(
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          one_body, two_body, active_orbitals, hamiltonian->get_core_energy(),
          inactive_fock, hamiltonian->get_type()));
}

std::shared_ptr<data::Wavefunction> closed_shell_determinant(
    const std::shared_ptr<data::Hamiltonian>& hamiltonian,
    std::size_t n_active_occupied) {
  const auto orbitals = hamiltonian->get_orbitals();
  const auto active_alpha = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::alpha());
  const auto active_beta = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::beta());
  if (active_alpha != active_beta || n_active_occupied > active_alpha.size()) {
    throw std::invalid_argument(
        "CT-F12 determinant requires matching closed-shell active spaces");
  }

  std::string occupations(active_alpha.size(), '0');
  for (std::size_t i = 0; i < n_active_occupied; ++i) occupations[i] = '2';
  auto determinant = data::Configuration::from_spin_half_string(occupations);
  auto container = std::make_unique<data::StateVectorContainer>(
      determinant, orbitals, "electrons");
  return std::make_shared<data::Wavefunction>(std::move(container));
}

double closed_shell_determinant_energy(
    const std::shared_ptr<data::Hamiltonian>& hamiltonian,
    std::size_t n_active_occupied) {
  const auto wavefunction =
      closed_shell_determinant(hamiltonian, n_active_occupied);
  return data::Ansatz(*hamiltonian, *wavefunction).calculate_energy();
}

std::unique_ptr<algorithms::MultiConfigurationCalculator> make_asci() {
  return algorithms::MultiConfigurationCalculatorFactory::create("macis_asci");
}

std::shared_ptr<data::Hamiltonian> make_ctf12_hamiltonian(
    const std::shared_ptr<data::Wavefunction>& reference, double gamma,
    std::size_t frozen_core, const std::string& cabs_basis,
    bool symmetrize_two_body) {
  auto constructor =
      algorithms::EffectiveHamiltonianConstructorFactory::create("qdk_ct_f12");
  constructor->settings().set("gamma", gamma);
  constructor->settings().set("frozen_core",
                              static_cast<std::int64_t>(frozen_core));
  constructor->settings().set("orbital_basis", std::string("relaxed"));
  constructor->settings().set("symmetrize_two_body", symmetrize_two_body);
  return constructor->run(reference, bare_hamiltonian(reference),
                          valence_p_space(reference, frozen_core),
                          cabs_bases(reference, cabs_basis));
}

double mp2_energy(const std::shared_ptr<data::Hamiltonian>& hamiltonian,
                  std::size_t n_active_occupied) {
  const auto reference =
      closed_shell_determinant(hamiltonian, n_active_occupied);
  auto ansatz = std::make_shared<data::Ansatz>(*hamiltonian, *reference);
  auto mp2 = algorithms::DynamicalCorrelationCalculatorFactory::create(
      "qdk_mp2_calculator");
  return std::get<0>(mp2->run(ansatz));
}

}  // namespace

TEST(CtF12ActiveSpace, NeonAugCcPvdzAsci) {
  scf::QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  struct LibintGuard {
    LibintGuard() { ::libint2::initialize(); }
    ~LibintGuard() { ::libint2::finalize(); }
  } libint_guard;

  Eigen::MatrixXd coords = Eigen::MatrixXd::Zero(1, 3);
  auto structure =
      std::make_shared<data::Structure>(coords, std::vector<std::string>{"Ne"});
  auto scf_solver = algorithms::ScfSolverFactory::create("qdk");
  const auto [hf_energy, reference] =
      scf_solver->run(structure, 0, 1, "aug-cc-pvdz");

  const ctf12::F12HartreeFockInput f12_input =
      ctf12::f12_input_from_wavefunction(*reference, 1.5, "aug-cc-pvdz-optri",
                                         1);
  const double f12_hf_energy = hf_energy + ctf12::f12_hf_scf_energy(f12_input);
  const double f12_mp2_energy = hf_energy + ctf12::f12_mp2_energy(f12_input);

  auto constructor =
      algorithms::EffectiveHamiltonianConstructorFactory::create("qdk_ct_f12");
  constructor->settings().set("gamma", 1.5);
  constructor->settings().set("frozen_core", std::int64_t{1});
  constructor->settings().set("orbital_basis", std::string("relaxed"));
  constructor->settings().set("symmetrize_two_body", false);
  const auto dressed_hamiltonian = constructor->run(
      reference, bare_hamiltonian(reference), valence_p_space(reference, 1),
      cabs_bases(reference, "aug-cc-pvdz-optri"));

  const auto full_valence_indices = data::spin_channel_indices(
      dressed_hamiltonian->get_orbitals()->active_indices(),
      data::axes::alpha());
  ASSERT_GE(full_valence_indices.size(), 5u);
  const std::vector<std::size_t> selected_active_indices(
      full_valence_indices.begin(), full_valence_indices.begin() + 5);
  const auto active_hamiltonian = restrict_dressed_virtual_space(
      dressed_hamiltonian, selected_active_indices);

  EXPECT_EQ(data::spin_channel_indices(
                active_hamiltonian->get_orbitals()->active_indices(),
                data::axes::alpha()),
            selected_active_indices);
  EXPECT_EQ(std::get<0>(active_hamiltonian->get_one_body_integrals()).rows(),
            5);
  EXPECT_EQ(std::get<0>(active_hamiltonian->get_two_body_integrals()).size(),
            5 * 5 * 5 * 5);
  EXPECT_DOUBLE_EQ(active_hamiltonian->get_core_energy(),
                   dressed_hamiltonian->get_core_energy());
  EXPECT_DOUBLE_EQ(active_hamiltonian->get_one_body_element(0, 4),
                   dressed_hamiltonian->get_one_body_element(0, 4));
  EXPECT_DOUBLE_EQ(active_hamiltonian->get_two_body_element(0, 1, 3, 4),
                   dressed_hamiltonian->get_two_body_element(0, 1, 3, 4));

  const auto [n_alpha, n_beta] = reference->get_total_num_electrons();
  const std::size_t n_inactive =
      data::spin_channel_indices(
          active_hamiltonian->get_orbitals()->inactive_indices(),
          data::axes::alpha())
          .size();
  ASSERT_GE(n_alpha, n_inactive);
  ASSERT_GE(n_beta, n_inactive);
  const auto n_active_alpha = static_cast<unsigned int>(n_alpha - n_inactive);
  const auto n_active_beta = static_cast<unsigned int>(n_beta - n_inactive);
  EXPECT_EQ(n_alpha, 5u);
  EXPECT_EQ(n_beta, 5u);
  EXPECT_EQ(n_inactive, 1u);
  EXPECT_EQ(n_active_alpha, 4u);
  EXPECT_EQ(n_active_beta, 4u);
  EXPECT_EQ(2 * (n_active_alpha + n_inactive), 10u);
  EXPECT_EQ(f12_input.n_occupied, n_alpha);
  EXPECT_EQ(f12_input.n_core, n_inactive);
  EXPECT_EQ(f12_input.n_occupied - f12_input.n_core, n_active_alpha);

  const double native_full_hf_energy =
      closed_shell_determinant_energy(dressed_hamiltonian, n_active_alpha);
  const double native_active_hf_energy =
      closed_shell_determinant_energy(active_hamiltonian, n_active_alpha);
  EXPECT_NEAR(native_active_hf_energy, native_full_hf_energy, 1e-12);

  const auto native_reference =
      closed_shell_determinant(dressed_hamiltonian, n_active_alpha);
  auto native_ansatz =
      std::make_shared<data::Ansatz>(*dressed_hamiltonian, *native_reference);
  EXPECT_NEAR(native_ansatz->calculate_energy(), native_full_hf_energy, 1e-12);
  auto mp2 = algorithms::DynamicalCorrelationCalculatorFactory::create(
      "qdk_mp2_calculator");
  const auto native_mp2_result = mp2->run(native_ansatz);
  const double native_mp2_energy = std::get<0>(native_mp2_result);
  EXPECT_TRUE(std::isfinite(native_mp2_energy));
  ASSERT_NE(std::get<1>(native_mp2_result), nullptr);

  auto active_space_asci = make_asci();
  const auto [active_space_asci_energy, active_space_wavefunction] =
      active_space_asci->run(active_hamiltonian, n_active_alpha, n_active_beta);
  EXPECT_TRUE(std::isfinite(active_space_asci_energy));
  EXPECT_LE(active_space_asci_energy, native_active_hf_energy + 1e-10);
  ASSERT_NE(active_space_wavefunction, nullptr);
  EXPECT_GT(active_space_wavefunction->size(), 0u);
  EXPECT_EQ(data::spin_channel_indices(
                active_space_wavefunction->get_orbitals()->active_indices(),
                data::axes::alpha()),
            selected_active_indices);
  EXPECT_EQ(active_space_wavefunction->get_active_num_electrons().first, 4u);
  EXPECT_EQ(active_space_wavefunction->get_active_num_electrons().second, 4u);
  EXPECT_EQ(active_space_wavefunction->get_total_num_electrons().first, 5u);
  EXPECT_EQ(active_space_wavefunction->get_total_num_electrons().second, 5u);

  auto full_valence_asci = make_asci();
  const auto [full_valence_asci_energy, full_valence_wavefunction] =
      full_valence_asci->run(dressed_hamiltonian, n_active_alpha,
                             n_active_beta);
  EXPECT_TRUE(std::isfinite(full_valence_asci_energy));
  EXPECT_LE(full_valence_asci_energy, native_full_hf_energy + 1e-10);
  ASSERT_NE(full_valence_wavefunction, nullptr);
  EXPECT_GT(full_valence_wavefunction->size(), 0u);
  EXPECT_EQ(data::spin_channel_indices(
                full_valence_wavefunction->get_orbitals()->active_indices(),
                data::axes::alpha()),
            full_valence_indices);
  EXPECT_EQ(full_valence_wavefunction->get_active_num_electrons().first, 4u);
  EXPECT_EQ(full_valence_wavefunction->get_active_num_electrons().second, 4u);
  EXPECT_EQ(full_valence_wavefunction->get_total_num_electrons().first, 5u);
  EXPECT_EQ(full_valence_wavefunction->get_total_num_electrons().second, 5u);

  std::cout << std::setprecision(16) << "HF energy: " << hf_energy
            << " Hartree\n"
            << "Native F12-HF energy (4-fold tensor): " << f12_hf_energy
            << " Hartree\n"
            << "Native F12-MP2 energy (4-fold tensor): " << f12_mp2_energy
            << " Hartree\n"
            << "Electron partition for MP2/ASCI: 10 total = 2 frozen-core + "
               "8 correlated\n"
            << "Native CT-F12 HF determinant (full frozen-core valence "
               "orbital space): "
            << native_full_hf_energy << " Hartree\n"
            << "Native CT-F12 MP2 energy (4-fold tensor): " << native_mp2_energy
            << " Hartree\n"
            << "Native CT-F12 HF determinant (CAS(8e,5o), frozen "
               "1s^2): "
            << native_active_hf_energy << " Hartree\n"
            << "Native F12-ASCI (CAS(8e,5o), frozen 1s^2) energy: "
            << active_space_asci_energy << " Hartree\n"
            << "Native F12-ASCI (full frozen-core valence space, "
            << full_valence_indices.size()
            << " orbitals, 8 correlated electrons) energy: "
            << full_valence_asci_energy << " Hartree\n";
}

TEST(CtF12AbsoluteEnergy, StretchedN2InternalAndEmissionBoundaries) {
  scf::QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  struct LibintGuard {
    LibintGuard() { ::libint2::initialize(); }
    ~LibintGuard() { ::libint2::finalize(); }
  } libint_guard;

  constexpr double gamma = 1.0;
  constexpr std::size_t frozen_core = 2;
  const auto structure = testing::create_stretched_n2_structure(2.0);
  auto scf_solver = algorithms::ScfSolverFactory::create("qdk");
  const auto [hf_energy, reference] =
      scf_solver->run(structure, 0, 1, "cc-pvdz-f12");
  const ctf12::F12HartreeFockInput input = ctf12::f12_input_from_wavefunction(
      *reference, gamma, "cc-pvdz-f12-optri", frozen_core);

  const Eigen::MatrixXd overlap = scf::cabs::ao_overlap(input.obs, input.obs);
  const Eigen::MatrixXd mo_overlap =
      input.mo_coefficients.transpose() * overlap * input.mo_coefficients;
  EXPECT_LT((mo_overlap -
             Eigen::MatrixXd::Identity(mo_overlap.rows(), mo_overlap.cols()))
                .cwiseAbs()
                .maxCoeff(),
            1e-8);

  const Eigen::MatrixXd h_ao =
      scf::geminal::kinetic_matrix(input.obs) +
      scf::geminal::nuclear_matrix(input.obs, input.nuclei);
  const Eigen::MatrixXd h_mo =
      input.mo_coefficients.transpose() * h_ao * input.mo_coefficients;
  const std::size_t n_ao = static_cast<std::size_t>(input.obs.nbf());
  const std::size_t n_mo =
      static_cast<std::size_t>(input.mo_coefficients.cols());
  auto eri_ao = scf::geminal::four_center_coulomb(input.obs, input.obs,
                                                  input.obs, input.obs);
  auto eri_mo = scf::geminal::mo_transform_4index(
      eri_ao.get(), n_ao, n_ao, n_ao, n_ao, input.mo_coefficients,
      input.mo_coefficients, input.mo_coefficients, input.mo_coefficients);
  auto chemist_index = [n_mo](std::size_t p, std::size_t q, std::size_t r,
                              std::size_t s) {
    return ((p * n_mo + q) * n_mo + r) * n_mo + s;
  };

  const double nuclear_repulsion =
      structure->calculate_nuclear_repulsion_energy();
  double original_determinant_energy = nuclear_repulsion;
  for (std::size_t i = 0; i < input.n_occupied; ++i) {
    original_determinant_energy += 2.0 * h_mo(i, i);
    for (std::size_t j = 0; j < input.n_occupied; ++j) {
      original_determinant_energy += 2.0 * eri_mo[chemist_index(i, i, j, j)] -
                                     eri_mo[chemist_index(i, j, j, i)];
    }
  }
  EXPECT_NEAR(original_determinant_energy, hf_energy, 1e-8);

  Eigen::MatrixXd reference_fock = h_mo;
  for (std::size_t p = 0; p < n_mo; ++p)
    for (std::size_t q = 0; q < n_mo; ++q)
      for (std::size_t i = 0; i < input.n_occupied; ++i)
        reference_fock(p, q) += 2.0 * eri_mo[chemist_index(p, q, i, i)] -
                                eri_mo[chemist_index(p, i, i, q)];

  const Eigen::MatrixXd off_diagonal =
      reference_fock - reference_fock.diagonal().asDiagonal().toDenseMatrix();
  const double max_fock_off_diagonal = off_diagonal.cwiseAbs().maxCoeff();
  const double max_orbital_energy_residual =
      (reference_fock.diagonal() - input.orbital_energies)
          .cwiseAbs()
          .maxCoeff();
  EXPECT_LT(max_fock_off_diagonal, 2e-6)
      << "Reference Fock matrix is not diagonal in the input MO basis";
  EXPECT_LT(max_orbital_energy_residual, 2e-6)
      << "Regenerated Fock diagonal does not match input orbital energies";

  // Only the bare-reference side is checked here; that the emitted Hamiltonian
  // reproduces the F12-HF energy is covered by
  // StretchedN2HamiltonianMatchesHfAndMp2Apis.
  const ctf12::F12HartreeFockResult f12 = ctf12::run_f12_hf(input);

  EXPECT_NEAR(f12.e_hf + nuclear_repulsion, hf_energy, 1e-8);
  EXPECT_NEAR(f12.e_hf + nuclear_repulsion, original_determinant_energy, 1e-8);
}

TEST(CtF12AbsoluteEnergy, StretchedN2HamiltonianMatchesHfAndMp2Apis) {
  scf::QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  struct LibintGuard {
    LibintGuard() { ::libint2::initialize(); }
    ~LibintGuard() { ::libint2::finalize(); }
  } libint_guard;

  constexpr double gamma = 1.0;
  constexpr std::size_t frozen_core = 2;
  const std::string orbital_basis = "cc-pvdz-f12";
  const std::string cabs_basis = "cc-pvdz-f12-optri";
  const auto structure = testing::create_stretched_n2_structure(2.0);

  auto scf_solver = algorithms::ScfSolverFactory::create("qdk");
  const auto [hf_energy, reference] =
      scf_solver->run(structure, 0, 1, orbital_basis);
  const ctf12::F12HartreeFockInput input = ctf12::f12_input_from_wavefunction(
      *reference, gamma, cabs_basis, frozen_core);
  const double expected_f12_hf_energy =
      hf_energy + ctf12::f12_hf_scf_energy(input);
  const double expected_f12_mp2_energy =
      hf_energy + ctf12::f12_mp2_energy(input);

  const auto hamiltonian =
      make_ctf12_hamiltonian(reference, gamma, frozen_core, cabs_basis,
                             /*symmetrize_two_body=*/false);
  const auto [n_alpha, n_beta] = reference->get_total_num_electrons();
  ASSERT_EQ(n_alpha, n_beta);
  ASSERT_GE(n_alpha, frozen_core);
  const std::size_t n_active_occupied = n_alpha - frozen_core;
  const double emitted_f12_hf_energy =
      closed_shell_determinant_energy(hamiltonian, n_active_occupied);
  const double emitted_f12_mp2_energy =
      mp2_energy(hamiltonian, n_active_occupied);

  std::cout << std::setprecision(16) << "Canonical HF total: " << hf_energy
            << " Hartree\n"
            << "CT-F12 HF API total: " << expected_f12_hf_energy << " Hartree\n"
            << "Emitted-H determinant: " << emitted_f12_hf_energy
            << " Hartree\n"
            << "CT-F12 MP2 API total: " << expected_f12_mp2_energy
            << " Hartree\n"
            << "MP2 over emitted H: " << emitted_f12_mp2_energy << " Hartree\n";

  EXPECT_NEAR(emitted_f12_hf_energy, expected_f12_hf_energy, 1e-8);
  EXPECT_NEAR(emitted_f12_mp2_energy, expected_f12_mp2_energy, 1e-8);
}

TEST(CtF12ActiveSpace, StretchedN2CcPvdzAsci) {
  scf::QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  struct LibintGuard {
    LibintGuard() { ::libint2::initialize(); }
    ~LibintGuard() { ::libint2::finalize(); }
  } libint_guard;

  constexpr double bond_length_angstrom = 2.0;
  constexpr double gamma = 1.0;
  constexpr std::size_t frozen_core = 2;
  constexpr std::size_t cas_orbitals = 8;
  const std::string orbital_basis = "cc-pvdz-f12";
  const std::string cabs_basis = "cc-pvdz-f12-optri";

  const auto structure =
      testing::create_stretched_n2_structure(bond_length_angstrom);
  auto scf_solver = algorithms::ScfSolverFactory::create("qdk");
  const auto [hf_energy, reference] =
      scf_solver->run(structure, 0, 1, orbital_basis);
  const auto [n_alpha, n_beta] = reference->get_total_num_electrons();
  ASSERT_EQ(n_alpha, 7u);
  ASSERT_EQ(n_beta, 7u);
  const auto n_active_alpha = static_cast<unsigned int>(n_alpha - frozen_core);
  const auto n_active_beta = static_cast<unsigned int>(n_beta - frozen_core);
  ASSERT_EQ(n_active_alpha, 5u);
  ASSERT_EQ(n_active_beta, 5u);

  const auto native_hamiltonian =
      make_ctf12_hamiltonian(reference, gamma, frozen_core, cabs_basis,
                             /*symmetrize_two_body=*/false);
  const auto symmetrized_hamiltonian =
      make_ctf12_hamiltonian(reference, gamma, frozen_core, cabs_basis,
                             /*symmetrize_two_body=*/true);

  const double native_f12_hf_energy =
      closed_shell_determinant_energy(native_hamiltonian, n_active_alpha);
  const double native_f12_mp2_energy =
      mp2_energy(native_hamiltonian, n_active_alpha);
  const double symmetrized_f12_hf_energy =
      closed_shell_determinant_energy(symmetrized_hamiltonian, n_active_alpha);
  const double symmetrized_f12_mp2_energy =
      mp2_energy(symmetrized_hamiltonian, n_active_alpha);

  const auto full_valence_indices = data::spin_channel_indices(
      native_hamiltonian->get_orbitals()->active_indices(),
      data::axes::alpha());
  ASSERT_GE(full_valence_indices.size(), cas_orbitals);
  const std::vector<std::size_t> cas_indices(
      full_valence_indices.begin(),
      full_valence_indices.begin() + cas_orbitals);
  const auto cas_hamiltonian =
      restrict_dressed_virtual_space(native_hamiltonian, cas_indices);
  const double cas_hf_energy =
      closed_shell_determinant_energy(cas_hamiltonian, n_active_alpha);
  EXPECT_NEAR(cas_hf_energy, native_f12_hf_energy, 1e-12);

  auto cas_asci = make_asci();
  const auto [cas_asci_energy, cas_wavefunction] =
      cas_asci->run(cas_hamiltonian, n_active_alpha, n_active_beta);
  ASSERT_NE(cas_wavefunction, nullptr);
  EXPECT_TRUE(std::isfinite(cas_asci_energy));
  EXPECT_LE(cas_asci_energy, cas_hf_energy + 1e-10);
  EXPECT_EQ(cas_wavefunction->get_active_num_electrons().first, 5u);
  EXPECT_EQ(cas_wavefunction->get_active_num_electrons().second, 5u);
  EXPECT_EQ(cas_wavefunction->get_total_num_electrons().first, 7u);
  EXPECT_EQ(cas_wavefunction->get_total_num_electrons().second, 7u);

  auto full_valence_asci = make_asci();
  const auto [full_valence_asci_energy, full_valence_wavefunction] =
      full_valence_asci->run(native_hamiltonian, n_active_alpha, n_active_beta);
  ASSERT_NE(full_valence_wavefunction, nullptr);
  EXPECT_TRUE(std::isfinite(full_valence_asci_energy));
  EXPECT_LE(full_valence_asci_energy, native_f12_hf_energy + 1e-10);
  EXPECT_EQ(full_valence_wavefunction->get_active_num_electrons().first, 5u);
  EXPECT_EQ(full_valence_wavefunction->get_active_num_electrons().second, 5u);
  EXPECT_EQ(full_valence_wavefunction->get_total_num_electrons().first, 7u);
  EXPECT_EQ(full_valence_wavefunction->get_total_num_electrons().second, 7u);

  std::cout << std::setprecision(16)
            << "N2 bond length: " << bond_length_angstrom << " Angstrom\n"
            << "OBS: " << orbital_basis << ", CABS: " << cabs_basis << '\n'
            << "Electron partition: 14 total = 4 frozen-core + 10 "
               "correlated\n"
            << "HF energy: " << hf_energy << " Hartree\n"
            << "Native F12-HF energy (4-fold tensor): " << native_f12_hf_energy
            << " Hartree\n"
            << "Native F12-MP2 energy (4-fold tensor): "
            << native_f12_mp2_energy << " Hartree\n"
            << "Symmetrized CT-F12 HF determinant (8-fold tensor): "
            << symmetrized_f12_hf_energy << " Hartree\n"
            << "Symmetrized CT-F12 MP2 energy (8-fold tensor): "
            << symmetrized_f12_mp2_energy << " Hartree\n"
            << "Native F12-ASCI (CAS(10e,8o)): " << cas_asci_energy
            << " Hartree\n"
            << "Native F12-ASCI (full frozen-core valence space, "
            << full_valence_indices.size()
            << " orbitals, 10 correlated electrons): "
            << full_valence_asci_energy << " Hartree\n";
}

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
  constructor->settings().set("orbital_basis", std::string("relaxed"));
  auto dressed_hamiltonian = constructor->run(
      reference, bare_hamiltonian(reference),
      valence_p_space(reference, static_cast<std::size_t>(frozen_core)),
      cabs_bases(reference, cabs_name));

  // Closed-shell HF determinant over the frozen-core active space of the
  // relaxed F12-HF orbitals carried by the emitted Hamiltonian.
  auto orbitals = dressed_hamiltonian->get_orbitals();
  const std::size_t n_active =
      data::spin_channel_indices(orbitals->active_indices(),
                                 data::axes::alpha())
          .size();
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
  const ctf12::F12HartreeFockResult f12 = ctf12::run_f12_hf(input);
  const double nuclear_repulsion =
      structure->calculate_nuclear_repulsion_energy();
  EXPECT_NEAR(reference_energy, f12.e_f12hf + nuclear_repulsion, 1e-11)
      << obs_name << ": F12-HF reference energy " << reference_energy;

  ::libint2::finalize();
}

}  // namespace

TEST(CtF12EffectiveHamiltonian, NeonAugCcPvdzMp2) {
  run_neon_effective_hamiltonian_mp2("aug-cc-pvdz", "aug-cc-pvdz-optri", 1e-12);
}

TEST(CtF12EffectiveHamiltonian, NeonPIndicesDefinePostDressingActiveSpace) {
  scf::QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  struct LibintGuard {
    LibintGuard() { ::libint2::initialize(); }
    ~LibintGuard() { ::libint2::finalize(); }
  } libint_guard;

  Eigen::MatrixXd coords = Eigen::MatrixXd::Zero(1, 3);
  auto structure =
      std::make_shared<data::Structure>(coords, std::vector<std::string>{"Ne"});
  auto scf_solver = algorithms::ScfSolverFactory::create("qdk");
  const auto reference = scf_solver->run(structure, 0, 1, "aug-cc-pvdz").second;

  auto constructor =
      algorithms::EffectiveHamiltonianConstructorFactory::create("qdk_ct_f12");
  constructor->settings().set("gamma", 1.5);
  constructor->settings().set("frozen_core", std::int64_t{1});
  const auto dressed_hamiltonian = constructor->run(
      reference, bare_hamiltonian(reference), p_space(reference, 1, 6),
      cabs_bases(reference, "aug-cc-pvdz-optri"));

  const auto orbitals = dressed_hamiltonian->get_orbitals();
  const auto active_alpha = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::alpha());
  const auto active_beta = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::beta());
  const auto inactive_alpha = data::spin_channel_indices(
      orbitals->inactive_indices(), data::axes::alpha());
  const auto inactive_beta = data::spin_channel_indices(
      orbitals->inactive_indices(), data::axes::beta());
  const std::vector<std::size_t> expected_active{1, 2, 3, 4, 5};
  const std::vector<std::size_t> expected_inactive{0};

  EXPECT_EQ(active_alpha, expected_active);
  EXPECT_EQ(active_beta, expected_active);
  EXPECT_EQ(inactive_alpha, expected_inactive);
  EXPECT_EQ(inactive_beta, expected_inactive);
  EXPECT_GT(orbitals->get_num_molecular_orbitals(), 6u);
  EXPECT_EQ(std::get<0>(dressed_hamiltonian->get_one_body_integrals()).rows(),
            5);
  EXPECT_EQ(std::get<0>(dressed_hamiltonian->get_two_body_integrals()).size(),
            5 * 5 * 5 * 5);

  std::string occupations(expected_active.size(), '0');
  for (std::size_t i = 0; i < 4; ++i) occupations[i] = '2';
  auto determinant = data::Configuration::from_spin_half_string(occupations);
  auto container = std::make_unique<data::StateVectorContainer>(
      determinant, orbitals, "electrons");
  auto wavefunction =
      std::make_shared<data::Wavefunction>(std::move(container));
  const double determinant_energy =
      data::Ansatz(*dressed_hamiltonian, *wavefunction).calculate_energy();

  const ctf12::F12HartreeFockInput input = ctf12::f12_input_from_wavefunction(
      *reference, 1.5, "aug-cc-pvdz-optri", 1);
  const ctf12::F12HartreeFockResult f12 = ctf12::run_f12_hf(input);
  EXPECT_NEAR(determinant_energy, f12.e_f12hf, 1e-11);

  auto asci =
      algorithms::MultiConfigurationCalculatorFactory::create("macis_asci");
  asci->settings().set("ntdets_max", std::int64_t{1});
  asci->settings().set("ntdets_min", std::int64_t{1});
  asci->settings().set("ncdets_max", std::int64_t{1});
  asci->settings().set("max_refine_iter", std::int64_t{0});
  const auto [asci_energy, asci_wavefunction] =
      asci->run(dressed_hamiltonian, 4, 4);
  EXPECT_NEAR(asci_energy, determinant_energy, 1e-11);
  ASSERT_NE(asci_wavefunction, nullptr);
  EXPECT_EQ(asci_wavefunction->size(), 1u);
}

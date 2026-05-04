// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;
using namespace qdk::chemistry::algorithms;

// ---------------------------------------------------------------------------
// Helper: run SCF → Hamiltonian → CAS/SCI with RDMs
// ---------------------------------------------------------------------------

// Run SCF + CAS-FCI with RDMs and return the wavefunction
static std::shared_ptr<Wavefunction> run_cas_with_rdms(
    std::shared_ptr<Structure> structure, int charge, int multiplicity,
    const std::string& basis, int nalpha, int nbeta,
    std::shared_ptr<Orbitals> custom_orbitals = nullptr) {
  std::shared_ptr<Orbitals> orbitals;
  if (custom_orbitals) {
    orbitals = custom_orbitals;
  } else {
    auto scf_solver = ScfSolverFactory::create();
    scf_solver->settings().set("scf_type", std::string("restricted"));
    scf_solver->settings().set("method", "hf");
    scf_solver->settings().set("enable_gdm", false);
    auto [E_scf, wfn_scf] =
        scf_solver->run(structure, charge, multiplicity, basis);
    orbitals = wfn_scf->get_orbitals();
  }

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create();  // macis_cas
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  auto [E, wfn] = mc->run(H, nalpha, nbeta);
  return wfn;
}

// Run SCF + SCI (ASCI) with RDMs and return the wavefunction
static std::shared_ptr<Wavefunction> run_sci_with_rdms(
    std::shared_ptr<Structure> structure, int charge, int multiplicity,
    const std::string& basis, int nalpha, int nbeta, int ntdets_max = 128,
    std::shared_ptr<Orbitals> custom_orbitals = nullptr) {
  std::shared_ptr<Orbitals> orbitals;
  if (custom_orbitals) {
    orbitals = custom_orbitals;
  } else {
    auto scf_solver = ScfSolverFactory::create();
    scf_solver->settings().set("scf_type", std::string("restricted"));
    scf_solver->settings().set("method", "hf");
    scf_solver->settings().set("enable_gdm", false);
    auto [E_scf, wfn_scf] =
        scf_solver->run(structure, charge, multiplicity, basis);
    orbitals = wfn_scf->get_orbitals();
  }

  auto ham_gen = HamiltonianConstructorFactory::create();
  auto H = ham_gen->run(orbitals);

  auto mc = MultiConfigurationCalculatorFactory::create("macis_asci");
  mc->settings().set("calculate_one_rdm", true);
  mc->settings().set("calculate_two_rdm", true);
  mc->settings().set("ntdets_max", ntdets_max);
  mc->settings().set("max_refine_iter", 0);
  mc->settings().set("grow_factor", 2);
  mc->settings().set("core_selection_strategy", std::string("fixed"));
  auto [E, wfn] = mc->run(H, nalpha, nbeta);
  return wfn;
}

// ---------------------------------------------------------------------------
// Tests using SCF + CAS (full CI) with RDMs from MACIS
// ---------------------------------------------------------------------------

// H atom: single electron, doublet, <S^2> = 0.75
TEST(SSquaredCAS, H_Doublet) {
  auto wfn = run_cas_with_rdms(testing::create_hydrogen_structure(), 0, 2,
                               "sto-3g", 1, 0);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.75, testing::rdm_tolerance);
}

// H2+ (charge=+1): single electron, doublet, <S^2> = 0.75
TEST(SSquaredCAS, H2Plus_Doublet) {
  // H2 at 1.4 Bohr
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {0., 0., 1.4}};
  std::vector<std::string> symbols = {"H", "H"};
  auto h2 = std::make_shared<Structure>(coords, symbols);
  auto wfn = run_cas_with_rdms(h2, 1, 2, "sto-3g", 1, 0);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.75, testing::rdm_tolerance);
}

// H2 (neutral): singlet, <S^2> = 0
TEST(SSquaredCAS, H2_Singlet) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {0., 0., 1.4}};
  std::vector<std::string> symbols = {"H", "H"};
  auto h2 = std::make_shared<Structure>(coords, symbols);
  auto wfn = run_cas_with_rdms(h2, 0, 1, "sto-3g", 1, 1);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.0, testing::rdm_tolerance);
}

// H2O: closed-shell singlet, <S^2> = 0
// Full CI in sto-3g (7 MOs, 5α+5β) — no active space truncation
TEST(SSquaredCAS, Water_Singlet) {
  auto water = testing::create_water_structure();
  auto wfn = run_cas_with_rdms(water, 0, 1, "sto-3g", 5, 5);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.0, testing::rdm_tolerance);
}

// Li atom: doublet, <S^2> = 0.75
TEST(SSquaredCAS, Li_Doublet) {
  auto li = testing::create_li_structure();
  auto wfn = run_cas_with_rdms(li, 0, 2, "sto-3g", 2, 1);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.75, testing::rdm_tolerance);
}

// O2: triplet ground state, <S^2> = 2.0
// CAS(4,4) with 6 frozen (core + doubly-occupied valence)
TEST(SSquaredCAS, O2_Triplet) {
  auto o2 = testing::create_o2_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("restricted"));
  scf_solver->settings().set("method", "hf");
  scf_solver->settings().set("enable_gdm", false);
  auto [E_scf, wfn_scf] = scf_solver->run(o2, 0, 3, "sto-3g");
  auto orbitals = testing::with_active_space(
      wfn_scf->get_orbitals(), std::vector<size_t>{6, 7, 8, 9},
      std::vector<size_t>{0, 1, 2, 3, 4, 5});
  auto wfn = run_cas_with_rdms(o2, 0, 3, "sto-3g", 3, 1, orbitals);
  EXPECT_NEAR(wfn->compute_s_squared(), 2.0, testing::rdm_tolerance);
}

// N atom: quartet S=3/2, <S^2> = 3.75
// CAS in 2s2p shell (4 orbitals, freeze 1s)
TEST(SSquaredCAS, N_Quartet) {
  auto n_atom = testing::create_nitrogen_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("restricted"));
  scf_solver->settings().set("method", "hf");
  scf_solver->settings().set("enable_gdm", false);
  auto [E_scf, wfn_scf] = scf_solver->run(n_atom, 0, 4, "sto-3g");
  auto orbitals = testing::with_active_space(wfn_scf->get_orbitals(),
                                             std::vector<size_t>{1, 2, 3, 4},
                                             std::vector<size_t>{0});
  auto wfn = run_cas_with_rdms(n_atom, 0, 4, "sto-3g", 4, 1, orbitals);
  EXPECT_NEAR(wfn->compute_s_squared(), 3.75, testing::rdm_tolerance);
}

// Stretched H2 via CAS: singlet ground state, <S²> = 0
// At large separation, the CAS(1,1) wavefunction is a strongly correlated
// singlet: (1/√2)(|20⟩ − |02⟩), which is a spin eigenstate unlike broken-
// symmetry UHF.
TEST(SSquaredCAS, StretchedH2_Singlet) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {0., 0., 5.0}};
  std::vector<std::string> symbols = {"H", "H"};
  auto h2 = std::make_shared<Structure>(coords, symbols);
  auto wfn = run_cas_with_rdms(h2, 0, 1, "sto-3g", 1, 1);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.0, testing::rdm_tolerance);
}

// Stretched H2 triplet via CAS: <S²> = 2.0
TEST(SSquaredCAS, StretchedH2_Triplet) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {0., 0., 5.0}};
  std::vector<std::string> symbols = {"H", "H"};
  auto h2 = std::make_shared<Structure>(coords, symbols);
  auto wfn = run_cas_with_rdms(h2, 0, 3, "sto-3g", 2, 0);
  EXPECT_NEAR(wfn->compute_s_squared(), 2.0, testing::rdm_tolerance);
}

// ---------------------------------------------------------------------------
// Tests using SCF + SCI (ASCI) with RDMs from MACIS
// ---------------------------------------------------------------------------

// H atom via SCI: doublet, <S^2> = 0.75
TEST(SSquaredSCI, H_Doublet) {
  auto wfn = run_sci_with_rdms(testing::create_hydrogen_structure(), 0, 2,
                               "sto-3g", 1, 0, 1);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.75, testing::rdm_tolerance);
}

// H2 via SCI: singlet, <S^2> = 0
TEST(SSquaredSCI, H2_Singlet) {
  std::vector<Eigen::Vector3d> coords = {{0., 0., 0.}, {0., 0., 1.4}};
  std::vector<std::string> symbols = {"H", "H"};
  auto h2 = std::make_shared<Structure>(coords, symbols);
  auto wfn = run_sci_with_rdms(h2, 0, 1, "sto-3g", 1, 1, 4);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.0, testing::rdm_tolerance);
}

// Li atom via SCI: doublet, <S^2> = 0.75
TEST(SSquaredSCI, Li_Doublet) {
  auto li = testing::create_li_structure();
  auto wfn = run_sci_with_rdms(li, 0, 2, "sto-3g", 2, 1, 50);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.75, testing::rdm_tolerance);
}

// Water via SCI: singlet, <S^2> = 0
// Full orbital space in sto-3g
TEST(SSquaredSCI, Water_Singlet) {
  auto water = testing::create_water_structure();
  auto wfn = run_sci_with_rdms(water, 0, 1, "sto-3g", 5, 5, 128);
  EXPECT_NEAR(wfn->compute_s_squared(), 0.0, testing::rdm_tolerance);
}

// N atom via SCI: quartet, <S^2> = 3.75
TEST(SSquaredSCI, N_Quartet) {
  auto n_atom = testing::create_nitrogen_structure();
  auto scf_solver = ScfSolverFactory::create();
  scf_solver->settings().set("scf_type", std::string("restricted"));
  scf_solver->settings().set("method", "hf");
  scf_solver->settings().set("enable_gdm", false);
  auto [E_scf, wfn_scf] = scf_solver->run(n_atom, 0, 4, "sto-3g");
  auto orbitals = testing::with_active_space(wfn_scf->get_orbitals(),
                                             std::vector<size_t>{1, 2, 3, 4},
                                             std::vector<size_t>{0});
  auto wfn = run_sci_with_rdms(n_atom, 0, 4, "sto-3g", 4, 1, 20, orbitals);
  EXPECT_NEAR(wfn->compute_s_squared(), 3.75, testing::rdm_tolerance);
}

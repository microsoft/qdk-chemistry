// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/util/cabs.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <Eigen/Dense>
#include <cmath>
#include <memory>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <string>
#include <vector>

#include "qdk/chemistry/algorithms/microsoft/ctf12_f12.hpp"
#include "qdk/chemistry/algorithms/microsoft/utils.hpp"
#include "test_config.h"

using namespace qdk::chemistry;
namespace ctf12 = qdk::chemistry::algorithms::microsoft::ctf12;

namespace {

ctf12::F12HartreeFockInput neon_input(const std::string& obs_name,
                                      const std::string& cabs_name) {
  Eigen::MatrixXd coords = Eigen::MatrixXd::Zero(1, 3);
  auto structure =
      std::make_shared<data::Structure>(coords, std::vector<std::string>{"Ne"});

  auto solver = algorithms::ScfSolverFactory::create("qdk");
  auto [e_hf, wfn] = solver->run(structure, 0, 1, obs_name);
  auto orbitals = wfn->get_orbitals();

  auto obs_scf =
      utils::microsoft::convert_basis_set_from_qdk(*orbitals->get_basis_set());
  auto mol = obs_scf->mol;
  auto obs_libint = scf::libint2_util::convert_to_libint_basisset(*obs_scf);
  auto aux_scf = scf::BasisSet::from_database_json(mol, cabs_name,
                                                   scf::BasisMode::PSI4, true);
  auto aux_libint = scf::libint2_util::convert_to_libint_basisset(*aux_scf);
  auto cabs = scf::cabs::build_cabs(obs_libint, aux_libint);

  ctf12::F12HartreeFockInput input;
  input.obs = obs_libint;
  input.mo_coefficients = orbitals->get_coefficients_alpha();
  input.orbital_energies = orbitals->get_energies_alpha();
  input.n_occupied = 5;  // Ne: 1s 2s 2p
  input.n_core = 1;      // frozen 1s (formulation a)
  input.cabs_ri_basis = cabs.ri_basis;
  input.cabs_coefficients = cabs.cabs_coeff;
  input.gamma = 1.5;
  for (std::size_t a = 0; a < mol->n_atoms; ++a)
    input.nuclei.emplace_back(static_cast<double>(mol->atomic_charges[a]),
                              mol->coords[a]);
  return input;
}

// Self-consistent F12-HF energy correction for the neon atom, compared to the
// canonical transcorrelated reference values of Masteran et al. (Comment on
// J. Chem. Phys. 136, 084107), Table I. When @p first_order is finite, the
// inexpensive first-order "standard" estimate is checked against it as well.
void run_neon_f12_hf(const std::string& obs_name, const std::string& cabs_name,
                     double reference, double tol,
                     double first_order = std::nan("")) {
  scf::QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  ::libint2::initialize();

  auto input = neon_input(obs_name, cabs_name);
  for (std::size_t i = 1; i < input.n_occupied; ++i)
    ASSERT_LE(input.orbital_energies(static_cast<int>(i - 1)),
              input.orbital_energies(static_cast<int>(i)));

  const double e_f12 = ctf12::f12_hf_scf_energy(input);
  EXPECT_NEAR(e_f12, reference, tol)
      << obs_name << ": F12-HF correction " << e_f12 << " vs " << reference;

  if (std::isfinite(first_order)) {
    const double e_first =
        ctf12::f12_hf_energy(ctf12::build_intermediates(input));
    EXPECT_NEAR(e_first, first_order, 1e-7)
        << obs_name << ": first-order F12-HF " << e_first << " vs "
        << first_order;
  }

  ::libint2::finalize();
}

// Reproduces the MP2-level rows of the Comment Table I for the neon atom:
// conventional MP2, transcorrelated F12-MP2, the F12 correction to MP2
// (MP2-)F12c, and the total MP2-F12 correlation energy.
void run_neon_mp2(const std::string& obs_name, const std::string& cabs_name,
                  double mp2_ref, double f12_mp2_ref, double f12c_ref,
                  double mp2_f12_ref, double tol) {
  scf::QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  ::libint2::initialize();
  const auto input = neon_input(obs_name, cabs_name);
  const double mp2 = ctf12::mp2_energy(input);
  const double f12c = ctf12::mp2_f12_correction(input);
  EXPECT_NEAR(mp2, mp2_ref, tol) << obs_name << " MP2 " << mp2;
  EXPECT_NEAR(f12c, f12c_ref, tol) << obs_name << " (MP2-)F12c " << f12c;
  EXPECT_NEAR(mp2 + f12c, mp2_f12_ref, tol) << obs_name << " MP2-F12";
  EXPECT_NEAR(ctf12::f12_mp2_energy(input), f12_mp2_ref, tol)
      << obs_name << " F12-MP2";
  ::libint2::finalize();
}

}  // namespace

TEST(CtF12HartreeFock, NeonAugCcPvdz) {
  run_neon_f12_hf("aug-cc-pvdz", "aug-cc-pvdz-optri", -0.111555079, 1e-8,
                  -0.111127554);
}

TEST(CtF12HartreeFock, NeonAugCcPvtz) {
  run_neon_f12_hf("aug-cc-pvtz", "aug-cc-pvtz-optri", -0.042845640, 1e-8);
}

TEST(CtF12HartreeFock, NeonAugCcPvqz) {
  run_neon_f12_hf("aug-cc-pvqz", "aug-cc-pvqz-optri", -0.019939990, 1e-8);
}

TEST(CtF12Mp2, NeonAugCcPvdz) {
  run_neon_mp2("aug-cc-pvdz", "aug-cc-pvdz-optri", -0.206873509, -0.301361902,
               -0.104682301, -0.311555810, 1e-8);
}

TEST(CtF12Mp2, NeonAugCcPvtz) {
  run_neon_mp2("aug-cc-pvtz", "aug-cc-pvtz-optri", -0.272518905, -0.308391143,
               -0.043083913, -0.315602818, 1e-8);
}

TEST(CtF12Mp2, NeonAugCcPvqz) {
  run_neon_mp2("aug-cc-pvqz", "aug-cc-pvqz-optri", -0.297242806, -0.313067546,
               -0.020967256, -0.318210062, 1e-8);
}

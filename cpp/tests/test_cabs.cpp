// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/util/cabs.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <Eigen/Dense>
#include <memory>

#include "test_config.h"

using namespace qdk::chemistry::scf;

namespace {

std::shared_ptr<Molecule> make_neon() {
  auto mol = std::make_shared<Molecule>();
  mol->atomic_nums = {10};
  mol->n_atoms = 1;
  mol->atomic_charges = mol->atomic_nums;
  mol->total_nuclear_charge = 10;
  mol->n_electrons = 10;
  mol->coords = {{0.0, 0.0, 0.0}};
  return mol;
}

::libint2::BasisSet load(const std::shared_ptr<Molecule>& mol,
                         const std::string& name) {
  auto bs = BasisSet::from_database_json(mol, name, BasisMode::PSI4, true);
  return libint2_util::convert_to_libint_basisset(*bs);
}

}  // namespace

// CABS+ must be strongly orthogonal to the orbital basis and orthonormal in
// itself: S(OBS, CABS) = 0 and S(CABS, CABS) = I.
TEST(Cabs, NeonAugCcPvdzSanity) {
  QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  ::libint2::initialize();
  auto mol = make_neon();
  const auto obs = load(mol, "aug-cc-pvdz");
  const auto aux = load(mol, "aug-cc-pvdz-optri");

  const auto result = cabs::build_cabs(obs, aux);
  const int n_obs = static_cast<int>(obs.nbf());
  const int n_cabs = static_cast<int>(result.cabs_coeff.cols());
  ASSERT_GT(n_cabs, 0);
  EXPECT_EQ(result.ri_basis.nbf(), obs.nbf() + aux.nbf());

  const Eigen::MatrixXd s_ri =
      cabs::ao_overlap(result.ri_basis, result.ri_basis);

  const Eigen::MatrixXd s_obs_cabs = s_ri.topRows(n_obs) * result.cabs_coeff;
  EXPECT_LT(s_obs_cabs.cwiseAbs().maxCoeff(), 1e-8)
      << "CABS is not strongly orthogonal to OBS";

  const Eigen::MatrixXd s_cabs =
      result.cabs_coeff.transpose() * s_ri * result.cabs_coeff;
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(n_cabs, n_cabs);
  EXPECT_LT((s_cabs - identity).cwiseAbs().maxCoeff(), 1e-10)
      << "CABS is not orthonormal";

  ::libint2::finalize();
}

TEST(Cabs, TwoBasisOverlapShape) {
  QDKChemistryConfig::set_resources_dir(TEST_RESOURCES_DIR);
  ::libint2::initialize();
  auto mol = make_neon();
  const auto obs = load(mol, "aug-cc-pvdz");
  const auto aux = load(mol, "aug-cc-pvdz-optri");
  const Eigen::MatrixXd s = cabs::ao_overlap(obs, aux);
  EXPECT_EQ(static_cast<size_t>(s.rows()), obs.nbf());
  EXPECT_EQ(static_cast<size_t>(s.cols()), aux.nbf());
  // Diagonal OBS self-overlap is unity.
  const Eigen::MatrixXd s_obs = cabs::ao_overlap(obs, obs);
  EXPECT_NEAR(s_obs.diagonal().maxCoeff(), 1.0, 1e-10);
  EXPECT_NEAR(s_obs.diagonal().minCoeff(), 1.0, 1e-10);
  ::libint2::finalize();
}

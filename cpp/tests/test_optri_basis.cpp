// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <memory>
#include <numeric>

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

// Pins the resources directory to the in-source resources for the duration of a
// test so the vendored OptRI tarballs are discoverable, restoring it
// afterwards.
struct ScopedResourcesDir {
  std::filesystem::path original;
  explicit ScopedResourcesDir(const std::filesystem::path& dir) {
    original = QDKChemistryConfig::get_resources_dir();
    QDKChemistryConfig::set_resources_dir(dir);
  }
  ~ScopedResourcesDir() { QDKChemistryConfig::set_resources_dir(original); }
};

void check_optri(const std::string& basis, size_t expected_nbf,
                 int expected_max_l) {
  ScopedResourcesDir guard(TEST_RESOURCES_DIR);
  auto mol = make_neon();
  auto bs =
      BasisSet::from_database_json(mol, basis, BasisMode::PSI4, true /*pure*/);
  auto libint_bs = libint2_util::convert_to_libint_basisset(*bs);
  EXPECT_EQ(libint_bs.nbf(), expected_nbf) << basis;
  EXPECT_EQ(libint_bs.max_l(), expected_max_l) << basis;
}

}  // namespace

TEST(OptRiBasis, AugCcPvdzOptriNeon) {
  check_optri("aug-cc-pvdz-optri", 69, 4);
}

TEST(OptRiBasis, AugCcPvtzOptriNeon) {
  check_optri("aug-cc-pvtz-optri", 78, 4);
}

TEST(OptRiBasis, AugCcPvqzOptriNeon) {
  check_optri("aug-cc-pvqz-optri", 89, 5);
}

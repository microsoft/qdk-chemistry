// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <spdlog/spdlog.h>

#include <exception>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <regex>

#include "test_common.h"
#include "test_config.h"

using namespace qdk::chemistry::scf;

void assert_basis_file_existing(std::shared_ptr<Molecule> mol,
                                const std::string& basis) {
  spdlog::info("Loading basis set {}", basis);
  try {
    auto bs = BasisSet::from_database_json(mol, basis, BasisMode::PSI4,
                                           true /*pure*/);
    ASSERT_TRUE(bs->num_basis_funcs > 0);
  } catch (std::exception& e) {
    spdlog::error("Failed to load basis set {}: {}", basis, e.what());
  }
}

void test_basis_files_exist(std::vector<std::string> skip) {
  auto mol = make_molecule("h2o");
  auto path = std::filesystem::path(TEST_RESOURCES_DIR) / "basis_summary.json";
  std::ifstream fin(path);
  auto data = nlohmann::json::parse(fin);
  std::vector<std::string> basis_sets;
  for (const auto& basis : data) {
    std::string name = basis["name"];
    if (std::find(skip.begin(), skip.end(), name) != skip.end()) {
      spdlog::warn("Skip basis set {}", name);
      continue;
    }
    assert_basis_file_existing(mol, name);
    std::string alias = std::regex_replace(name, std::regex("_st_"), "*");
    alias = std::regex_replace(alias, std::regex("_sl_"), "/");
    if (name != alias) {
      assert_basis_file_existing(mol, alias);
    }
  }
}

// clang-format off
TEST(basis, basis_set_files_exist) {
  test_basis_files_exist({
    // "Engine::lmax_exceeded -- angular momentum limit exceeded"
    "7zapa-nr",
    "pv7z"
  });
}

TEST(basis, resources_dir_runtime_config) {
  // Reset the resources directory and change it back
  auto original_dir = QDKChemistryConfig::get_resources_dir();

  auto temp_dir = std::filesystem::temp_directory_path() / "test_resources";
  QDKChemistryConfig::set_resources_dir(temp_dir);
  EXPECT_EQ(QDKChemistryConfig::get_resources_dir(), temp_dir);

  QDKChemistryConfig::set_resources_dir(original_dir);

  // Verify restoration
  EXPECT_EQ(QDKChemistryConfig::get_resources_dir(), original_dir);
}
// clang-format on

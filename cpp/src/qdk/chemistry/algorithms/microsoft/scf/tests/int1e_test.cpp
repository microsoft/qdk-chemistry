// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>
#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/util/int1e.h>
#include <spdlog/spdlog.h>

#include <Eigen/Dense>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <string>

#include "test_common.h"
#include "test_config.h"
#include "util/macros.h"

using namespace qdk::chemistry::scf;
using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

nlohmann::json read_json(const char* name) {
  auto path =
      std::filesystem::path(TEST_DATA_DIR) / (std::string(name) + ".json");
  std::ifstream fin(path);
  return nlohmann::json::parse(fin);
}

RowMajorMatrix read_mat(nlohmann::json json, std::string name, int nrow,
                        int ncol) {
  auto a = json[name].get<std::vector<double>>();
  VERIFY(a.size() == static_cast<size_t>(nrow * ncol));
  return Eigen::Map<RowMajorMatrix>(a.data(), nrow, ncol);
}

void test_integral(std::string name, double tol = 1e-9) {
  auto json = read_json("int1e");
  auto mol_ptr = make_molecule(json["mol"].get<std::string>());
  auto& mol = *mol_ptr;
  auto bs_ptr = BasisSet::from_database_json(
      mol_ptr, json["basis"], BasisMode::PSI4, false, false /*sort*/);
  auto& bs = *bs_ptr;
  auto int1e = OneBodyIntegral(&bs, &mol, ParallelConfig{1, 0, 1, 0});

  auto ref = read_mat(json, name, bs.num_basis_funcs, bs.num_basis_funcs);
  RowMajorMatrix res =
      RowMajorMatrix::Zero(bs.num_basis_funcs, bs.num_basis_funcs);
  if (name == "kinetic_integral") {
    int1e.kinetic_integral(res.data());
  } else if (name == "nuclear_integral") {
    int1e.nuclear_integral(res.data());
  } else if (name == "overlap_integral") {
    int1e.overlap_integral(res.data());
  } else if (name == "ecp_integral") {
    int1e.ecp_integral(res.data());
  } else {
    ASSERT_TRUE(false);
  }

  ASSERT_NEAR((ref - res).array().abs().maxCoeff(), 0, tol);
}

void test_gradient(std::string name, double tol = 1e-9) {
  auto json = read_json("int1e");
  auto mol_ptr = make_molecule(json["mol"].get<std::string>());
  auto& mol = *mol_ptr;
  auto bs_ptr = BasisSet::from_database_json(
      mol_ptr, json["basis"], BasisMode::PSI4, false, false /*sort*/);
  auto& bs = *bs_ptr;
  auto int1e = OneBodyIntegral(&bs, &mol, ParallelConfig{1, 0, 1, 0});

  auto dm = read_mat(json, "dm", bs.num_basis_funcs, bs.num_basis_funcs);
  auto ref = read_mat(json, name, mol.n_atoms, 3);
  RowMajorMatrix res = RowMajorMatrix::Zero(3, mol.n_atoms);
  if (name == "kinetic_gradient") {
    int1e.kinetic_integral_deriv(dm.data(), res.data());
  } else if (name == "nuclear_gradient") {
    int1e.nuclear_integral_deriv(dm.data(), res.data());
  } else if (name == "overlap_gradient") {
    int1e.overlap_integral_deriv(dm.data(), res.data());
  } else if (name == "ecp_gradient") {
    int1e.ecp_integral_deriv(dm.data(), res.data());
  } else {
    ASSERT_TRUE(false);
  }

  ASSERT_NEAR((ref - res.transpose()).array().abs().maxCoeff(), 0, tol);
}

TEST(int1e, overlap_integral) { test_integral("overlap_integral"); }
TEST(int1e, kinetic_integral) { test_integral("kinetic_integral"); }
TEST(int1e, nuclear_integral) { test_integral("nuclear_integral", 1e-8); }
TEST(int1e, ecp_integral) { test_integral("ecp_integral", 1e-8); }

// Gradients are only fully supported if GPUs are enabled
#ifdef QDK_CHEMISTRY_ENABLE_GPU
TEST(int1e, overlap_graident) { test_gradient("overlap_gradient"); }
TEST(int1e, kinetic_graident) { test_gradient("kinetic_gradient"); }
TEST(int1e, nuclear_graident) { test_gradient("nuclear_gradient", 1e-8); }
TEST(int1e, ecp_graident) { test_gradient("ecp_gradient"); }
#endif

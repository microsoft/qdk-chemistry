// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <qdk/chemistry/utils/cube_generator.hpp>
#include <stdexcept>
#include <vector>

namespace {

using qdk::chemistry::data::BasisSet;
using qdk::chemistry::data::Configuration;
using qdk::chemistry::data::Element;
using qdk::chemistry::data::Orbitals;
using qdk::chemistry::data::OrbitalType;
using qdk::chemistry::data::Shell;
using qdk::chemistry::data::StateVectorContainer;
using qdk::chemistry::data::Structure;
using qdk::chemistry::data::Wavefunction;
using qdk::chemistry::utils::CubeGenerator;
using qdk::chemistry::utils::CubeGrid;
using qdk::chemistry::utils::generate_orbital_cubes;

std::shared_ptr<BasisSet> make_hydrogen_basis(
    std::vector<double> exponents = {1.0},
    std::vector<double> coefficients = {1.0}) {
  auto structure =
      std::make_shared<Structure>(std::vector<Eigen::Vector3d>{{0.0, 0.0, 0.0}},
                                  std::vector<Element>{Element::H});
  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, exponents, coefficients);
  return std::make_shared<BasisSet>("test", shells, structure);
}

std::shared_ptr<BasisSet> make_hydrogen_p_basis() {
  auto structure =
      std::make_shared<Structure>(std::vector<Eigen::Vector3d>{{0.0, 0.0, 0.0}},
                                  std::vector<Element>{Element::H});
  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::P, std::vector<double>{1.0},
                      std::vector<double>{1.0});
  return std::make_shared<BasisSet>("test-p", shells, structure);
}

std::shared_ptr<Wavefunction> make_wavefunction(
    std::shared_ptr<Orbitals> orbitals) {
  return std::make_shared<Wavefunction>(std::make_unique<StateVectorContainer>(
      Configuration::from_spin_half_string("2"), std::move(orbitals)));
}

std::shared_ptr<Wavefunction> make_restricted_wavefunction() {
  Eigen::MatrixXd coefficients(1, 1);
  coefficients << 1.0;
  auto orbitals = std::make_shared<Orbitals>(
      coefficients, std::nullopt, std::nullopt, make_hydrogen_basis());
  return make_wavefunction(orbitals);
}

std::shared_ptr<Wavefunction> make_unrestricted_wavefunction() {
  Eigen::MatrixXd coefficients_alpha(1, 1);
  coefficients_alpha << 1.0;
  Eigen::MatrixXd coefficients_beta(1, 1);
  coefficients_beta << 2.0;
  auto orbitals = std::make_shared<Orbitals>(
      coefficients_alpha, coefficients_beta, std::nullopt, std::nullopt,
      std::nullopt, make_hydrogen_basis());
  return make_wavefunction(orbitals);
}

CubeGrid single_point_grid() {
  CubeGrid grid;
  grid.origin = {0.0, 0.0, 0.0};
  grid.spacing = {1.0, 1.0, 1.0};
  grid.nx = grid.ny = grid.nz = 1;
  return grid;
}

}  // namespace

TEST(CubeGridTest, BuildsExpectedBoundingGrid) {
  const auto basis = make_hydrogen_basis();
  const auto grid = CubeGrid::from_basis_set(*basis, 3, 5, 7, 2.0);

  EXPECT_TRUE(grid.origin.isApprox(Eigen::Vector3d::Constant(-2.0)));
  EXPECT_TRUE(grid.spacing.isApprox(Eigen::Vector3d(2.0, 1.0, 2.0 / 3.0)));
  EXPECT_EQ(grid.num_points(), 105);
}

TEST(CubeGridTest, RejectsInvalidDimensionsAndOverflow) {
  const auto basis = make_hydrogen_basis();
  EXPECT_THROW(CubeGrid::from_basis_set(*basis, 0, 1, 1),
               std::invalid_argument);

  CubeGrid grid;
  grid.nx = std::numeric_limits<std::size_t>::max();
  grid.ny = 2;
  grid.nz = 1;
  EXPECT_THROW(grid.num_points(), std::overflow_error);

  if constexpr (std::numeric_limits<std::size_t>::max() >
                static_cast<std::size_t>(std::numeric_limits<int64_t>::max())) {
    grid.nx = static_cast<std::size_t>(std::numeric_limits<int64_t>::max());
    grid.ny = 2;
    EXPECT_THROW(grid.num_points(), std::overflow_error);
  }
}

TEST(CubeGeneratorTest, EvaluatesHydrogenOrbitalAndDensity) {
  CubeGenerator generator(make_hydrogen_basis());
  CubeGrid grid;
  grid.origin = {0.0, 0.0, 0.0};
  grid.spacing = {1.0, 1.0, 1.0};
  grid.nx = grid.ny = grid.nz = 1;

  Eigen::VectorXd coefficients(1);
  coefficients << 1.0;
  const auto orbital = generator.orbital(coefficients, "", grid);

  Eigen::MatrixXd density_matrix(1, 1);
  density_matrix << 1.0;
  const auto density = generator.density(density_matrix, "", grid);

  ASSERT_EQ(orbital.size(), 1);
  ASSERT_EQ(density.size(), 1);
  EXPECT_GT(orbital[0], 0.0);
  EXPECT_NEAR(density[0], orbital[0] * orbital[0], 1e-12);
}

TEST(CubeGeneratorTest, EvaluatesNormalizedHydrogenPOrbitals) {
  CubeGenerator generator(make_hydrogen_p_basis());
  CubeGrid grid;
  grid.origin = {0.25, -0.5, 0.75};
  grid.spacing = {1.0, 1.0, 1.0};
  grid.nx = grid.ny = grid.nz = 1;

  const double r_squared = grid.origin.squaredNorm();
  const double normalization = 2.0 * std::pow(2.0 / std::acos(-1.0), 0.75);
  const double radial = normalization * std::exp(-r_squared);

  for (Eigen::Index component = 0; component < 3; ++component) {
    Eigen::VectorXd coefficients = Eigen::VectorXd::Zero(3);
    coefficients[component] = 1.0;
    const auto orbital = generator.orbital(coefficients, "", grid);

    ASSERT_EQ(orbital.size(), 1);
    EXPECT_NEAR(orbital[0], grid.origin[component] * radial, 1e-12);
  }
}

TEST(CubeGeneratorTest, RejectsInvalidInputs) {
  CubeGenerator generator(make_hydrogen_basis());
  CubeGrid grid;
  grid.nx = grid.ny = grid.nz = 1;

  EXPECT_THROW(generator.orbital(Eigen::VectorXd::Zero(2), "", grid),
               std::invalid_argument);
  EXPECT_THROW(generator.density(Eigen::MatrixXd::Zero(2, 2), "", grid),
               std::invalid_argument);
  EXPECT_THROW(CubeGenerator(nullptr), std::invalid_argument);
}

TEST(CubeGeneratorTest, RejectsUnsupportedShells) {
  std::vector<double> exponents(33, 1.0);
  std::vector<double> coefficients(33, 1.0);
  EXPECT_THROW(CubeGenerator(make_hydrogen_basis(exponents, coefficients)),
               std::invalid_argument);

  auto structure =
      std::make_shared<Structure>(std::vector<Eigen::Vector3d>{{0.0, 0.0, 0.0}},
                                  std::vector<Element>{Element::H});
  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector<double>{1.0},
                      std::vector<double>{1.0}, std::vector<int>{2});
  auto radial_basis = std::make_shared<BasisSet>("radial", shells, structure);
  EXPECT_THROW(
      { CubeGenerator generator(radial_basis); }, std::invalid_argument);
}

TEST(GenerateOrbitalCubesTest, RestrictedWritesSingleZeroBasedCube) {
  const auto wavefunction = make_restricted_wavefunction();
  const auto output_dir =
      std::filesystem::temp_directory_path() /
      "qdk_cube_restricted_test";
  std::filesystem::remove_all(output_dir);

  const auto paths = generate_orbital_cubes(*wavefunction, {0},
                                            output_dir.string(),
                                            single_point_grid());

  ASSERT_EQ(paths.size(), 1u);
  // Zero-based label (index 0 -> 0000), and no spin suffix for restricted.
  EXPECT_EQ(std::filesystem::path(paths[0]).filename().string(),
            "orbital_0000.cube");
  EXPECT_TRUE(std::filesystem::exists(paths[0]));
  EXPECT_GT(std::filesystem::file_size(paths[0]), 0u);

  std::filesystem::remove_all(output_dir);
}

TEST(GenerateOrbitalCubesTest, UnrestrictedWritesAlphaAndBetaCubes) {
  const auto wavefunction = make_unrestricted_wavefunction();
  ASSERT_FALSE(wavefunction->get_orbitals()->is_restricted());

  const auto output_dir =
      std::filesystem::temp_directory_path() /
      "qdk_cube_unrestricted_test";
  std::filesystem::remove_all(output_dir);

  const auto paths = generate_orbital_cubes(*wavefunction, {0},
                                            output_dir.string(),
                                            single_point_grid());

  ASSERT_EQ(paths.size(), 2u);
  EXPECT_EQ(std::filesystem::path(paths[0]).filename().string(),
            "orbital_0000_a.cube");
  EXPECT_EQ(std::filesystem::path(paths[1]).filename().string(),
            "orbital_0000_b.cube");
  for (const auto& path : paths) {
    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_GT(std::filesystem::file_size(path), 0u);
  }

  std::filesystem::remove_all(output_dir);
}

TEST(GenerateOrbitalCubesTest, RejectsOutOfRangeIndex) {
  const auto wavefunction = make_restricted_wavefunction();
  const auto output_dir =
      std::filesystem::temp_directory_path() / "qdk_cube_oob_test";
  std::filesystem::remove_all(output_dir);

  EXPECT_THROW(generate_orbital_cubes(*wavefunction, {1}, output_dir.string(),
                                      single_point_grid()),
               std::out_of_range);

  std::filesystem::remove_all(output_dir);
}

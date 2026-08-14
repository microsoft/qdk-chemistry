// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <optional>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/utils/cube_generator.hpp>
#include <stdexcept>
#include <vector>

#include "ut_common.hpp"

namespace {

using qdk::chemistry::algorithms::ScfSolverFactory;
using qdk::chemistry::data::BasisSet;
using qdk::chemistry::data::Element;
using qdk::chemistry::data::Orbitals;
using qdk::chemistry::data::OrbitalType;
using qdk::chemistry::data::Shell;
using qdk::chemistry::data::Structure;
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

std::shared_ptr<BasisSet> make_ecp_basis(std::size_t core_electrons = 2) {
  auto structure =
      std::make_shared<Structure>(std::vector<Eigen::Vector3d>{{0.0, 0.0, 0.0}},
                                  std::vector<Element>{Element::H});
  // Valence shell: an ordinary contracted Gaussian, exactly as an ECP basis
  // stores it. The r^n projectors live in the separate ECP shell container.
  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector<double>{1.0},
                      std::vector<double>{1.0});
  std::vector<Shell> ecp_shells;
  ecp_shells.emplace_back(0, OrbitalType::S, std::vector<double>{10.0},
                          std::vector<double>{50.0}, std::vector<int>{2});
  const std::vector<std::size_t> ecp_electrons{core_electrons};
  return std::make_shared<BasisSet>("test-ecp", shells, "test-ecp-potential",
                                    ecp_shells, ecp_electrons, structure);
}

std::string first_line_of(const std::filesystem::path& path) {
  std::ifstream stream(path);
  std::string line;
  std::getline(stream, line);
  return line;
}

std::shared_ptr<Orbitals> make_restricted_orbitals() {
  Eigen::MatrixXd coefficients(1, 1);
  coefficients << 1.0;
  return std::make_shared<Orbitals>(coefficients, std::nullopt, std::nullopt,
                                    make_hydrogen_basis());
}

std::shared_ptr<Orbitals> make_unrestricted_orbitals() {
  Eigen::MatrixXd coefficients_alpha(1, 1);
  coefficients_alpha << 1.0;
  Eigen::MatrixXd coefficients_beta(1, 1);
  coefficients_beta << 2.0;
  return std::make_shared<Orbitals>(coefficients_alpha, coefficients_beta,
                                    std::nullopt, std::nullopt, std::nullopt,
                                    make_hydrogen_basis());
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

TEST(CubeGridTest, BoundsPointCountByBackendIntLimit) {
  // gauXC forwards the point count to BLAS as an int leading dimension, so
  // INT_MAX is admissible and anything beyond it must be refused before the
  // caller allocates a field of that size.
  constexpr auto limit =
      static_cast<std::size_t>(std::numeric_limits<int>::max());

  CubeGrid grid;
  grid.nx = limit;
  grid.ny = 1;
  grid.nz = 1;
  EXPECT_EQ(grid.num_points(), limit);

  grid.nz = 2;
  EXPECT_THROW(grid.num_points(), std::overflow_error);

  grid.nx = limit + 1;
  grid.nz = 1;
  EXPECT_THROW(grid.num_points(), std::overflow_error);

  // A grid that fits in int64_t but not in int was previously accepted and
  // only failed after allocation inside gauXC.
  grid.nx = 4000;
  grid.ny = 4000;
  grid.nz = 4000;
  EXPECT_THROW(grid.num_points(), std::overflow_error);
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

TEST(CubeGeneratorTest, EvaluatesHydrogenOrbitalFromNamedBasis) {
  auto structure =
      std::make_shared<Structure>(std::vector<Eigen::Vector3d>{{0.0, 0.0, 0.0}},
                                  std::vector<Element>{Element::H});
  const auto basis = BasisSet::from_basis_name("sto-3g", structure);
  ASSERT_EQ(basis->get_num_atomic_orbitals(), 1u);

  const auto shells = basis->get_shells_for_atom(0);
  ASSERT_FALSE(shells.empty());
  ASSERT_TRUE(shells.front().has_radial_powers());
  EXPECT_TRUE((shells.front().rpowers.array() == 0).all());

  CubeGenerator generator(basis);
  Eigen::VectorXd coefficients(1);
  coefficients << 1.0;
  const auto orbital = generator.orbital(coefficients, "", single_point_grid());

  ASSERT_EQ(orbital.size(), 1);
  EXPECT_GT(orbital[0], 0.0);
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

TEST(CubeGeneratorTest, AcceptsExplicitZeroRadialPowers) {
  auto structure =
      std::make_shared<Structure>(std::vector<Eigen::Vector3d>{{0.0, 0.0, 0.0}},
                                  std::vector<Element>{Element::H});
  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector<double>{1.0},
                      std::vector<double>{1.0}, std::vector<int>{0});
  auto basis = std::make_shared<BasisSet>("zero-radial", shells, structure);

  EXPECT_NO_THROW({ CubeGenerator generator(basis); });
}

TEST(CubeGeneratorTest, IndexesAtomicOrbitalsInAtomMajorOrder) {
  // AO 0 belongs to the atom at the origin and AO 1 to the atom at z = 2, so
  // selecting each in turn at the origin distinguishes the two orderings.
  auto structure = std::make_shared<Structure>(
      std::vector<Eigen::Vector3d>{{0.0, 0.0, 0.0}, {0.0, 0.0, 2.0}},
      std::vector<Element>{Element::H, Element::H});
  std::vector<Shell> shells;
  shells.emplace_back(0, OrbitalType::S, std::vector<double>{1.0},
                      std::vector<double>{1.0});
  shells.emplace_back(1, OrbitalType::S, std::vector<double>{1.0},
                      std::vector<double>{1.0});
  auto basis = std::make_shared<BasisSet>("two-atom", shells, structure);
  CubeGenerator generator(basis);

  Eigen::VectorXd first(2), second(2);
  first << 1.0, 0.0;
  second << 0.0, 1.0;
  const auto near = generator.orbital(first, "", single_point_grid());
  const auto far = generator.orbital(second, "", single_point_grid());

  const double normalization = std::pow(2.0 / std::acos(-1.0), 0.75);
  EXPECT_NEAR(near[0], normalization, 1e-12);
  EXPECT_NEAR(far[0], normalization * std::exp(-4.0), 1e-12);
}

TEST(CubeGeneratorTest, EvaluatesEcpBasisIgnoringProjectorShells) {
  // The r^2 projector lives in the ECP shell container, which the generator
  // does not traverse; only the valence Gaussians reach gauXC.
  CubeGenerator generator(make_ecp_basis());
  Eigen::VectorXd coefficients(1);
  coefficients << 1.0;

  const auto orbital = generator.orbital(coefficients, "", single_point_grid());

  ASSERT_EQ(orbital.size(), 1);
  EXPECT_NEAR(orbital[0], std::pow(2.0 / std::acos(-1.0), 0.75), 1e-12);
}

TEST(CubeGeneratorTest, EcpCubeCommentRecordsValenceOnlyField) {
  CubeGenerator generator(make_ecp_basis(28));
  Eigen::VectorXd coefficients(1);
  coefficients << 1.0;
  const auto output =
      std::filesystem::temp_directory_path() / "qdk_ecp_orbital.cube";

  generator.orbital(coefficients, output.string(), single_point_grid());

  const std::string comment = first_line_of(output);
  EXPECT_NE(comment.find("valence-only"), std::string::npos) << comment;
  EXPECT_NE(comment.find("28 core electrons"), std::string::npos) << comment;
  std::filesystem::remove(output);
}

TEST(CubeGeneratorTest, EcpAnnotationPreservesCallerComment) {
  CubeGenerator generator(make_ecp_basis(28));
  Eigen::VectorXd coefficients(1);
  coefficients << 1.0;
  const auto output =
      std::filesystem::temp_directory_path() / "qdk_ecp_annotated.cube";

  generator.orbital(coefficients, output.string(), single_point_grid(), "HOMO");

  const std::string comment = first_line_of(output);
  EXPECT_EQ(comment.rfind("HOMO", 0), 0u) << comment;
  EXPECT_NE(comment.find("valence-only"), std::string::npos) << comment;
  std::filesystem::remove(output);
}

TEST(CubeGeneratorTest, NonEcpCubeCommentIsUnchanged) {
  CubeGenerator generator(make_hydrogen_basis());
  Eigen::VectorXd coefficients(1);
  coefficients << 1.0;
  const auto output =
      std::filesystem::temp_directory_path() / "qdk_plain_orbital.cube";

  generator.orbital(coefficients, output.string(), single_point_grid(), "HOMO");

  EXPECT_EQ(first_line_of(output), "HOMO");
  std::filesystem::remove(output);
}

TEST(GenerateOrbitalCubesTest, RestrictedWritesSingleZeroBasedCube) {
  const auto orbitals = make_restricted_orbitals();
  const auto output_dir =
      std::filesystem::temp_directory_path() / "qdk_cube_restricted_test";
  std::filesystem::remove_all(output_dir);

  const auto paths = generate_orbital_cubes(*orbitals, {0}, output_dir.string(),
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
  const auto orbitals = make_unrestricted_orbitals();
  ASSERT_FALSE(orbitals->is_restricted());

  const auto output_dir =
      std::filesystem::temp_directory_path() / "qdk_cube_unrestricted_test";
  std::filesystem::remove_all(output_dir);

  const auto paths = generate_orbital_cubes(*orbitals, {0}, output_dir.string(),
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
  const auto orbitals = make_restricted_orbitals();
  const auto output_dir =
      std::filesystem::temp_directory_path() / "qdk_cube_oob_test";
  std::filesystem::remove_all(output_dir);

  EXPECT_THROW(generate_orbital_cubes(*orbitals, {1}, output_dir.string(),
                                      single_point_grid()),
               std::out_of_range);

  std::filesystem::remove_all(output_dir);
}

// Real effective core potential coverage. Every other ECP test in this file
// builds the basis by hand, so nothing exercised the path a user actually
// takes: ask for a named basis whose heavy element carries an ECP and let
// `from_basis_name` decide how to represent it.
//
// AgH in def2-SVP is the useful case because the answer is unambiguous. Ag has
// 47 electrons and H has 1, so an all-electron treatment holds 48. The def2
// ECP replaces the 28 core electrons on Ag, leaving 20. The generated field is
// valence-only by construction, so integrating it must give 20 and not 48. A
// 60^3 grid with a 6 Bohr margin is enough to see that clearly: the ECP has
// removed the sharp Ag core that a uniform grid would struggle to integrate,
// and the only remaining cusp is the hydrogen 1s.
TEST(CubeGeneratorEcpTest, AgHValenceDensityIntegratesToValenceElectronCount) {
  auto structure = testing::create_agh_structure();
  auto scf = ScfSolverFactory::create();
  const auto& [energy, wavefunction] = scf->run(structure, 0, 1, "def2-svp");
  ASSERT_NE(wavefunction, nullptr);

  const auto orbitals = wavefunction->get_orbitals();
  const auto basis_set = orbitals->get_basis_set();
  ASSERT_TRUE(basis_set->has_ecp_electrons());
  const auto& ecp_electrons = basis_set->get_ecp_electrons();
  ASSERT_EQ(ecp_electrons.size(), 2u);
  EXPECT_EQ(ecp_electrons[0], 28u);  // Ag
  EXPECT_EQ(ecp_electrons[1], 0u);   // H

  // The ECP projector shells are stored separately and must not contribute
  // atomic orbitals. If they leaked into the gauXC basis, `nbf` would grow and
  // this density matrix would be rejected for shape.
  const auto [occupations_alpha, occupations_beta] =
      wavefunction->get_total_orbital_occupations();
  const auto [density_alpha, density_beta] =
      orbitals->calculate_ao_density_matrix(occupations_alpha,
                                            occupations_beta);
  const Eigen::MatrixXd density_matrix = density_alpha + density_beta;
  ASSERT_EQ(std::size_t(density_matrix.rows()),
            basis_set->get_num_atomic_orbitals());

  CubeGenerator generator(basis_set);
  const auto grid = CubeGrid::from_basis_set(*basis_set, 60, 60, 60, 6.0);
  const auto field = generator.density(density_matrix, "", grid);
  ASSERT_EQ(field.size(), grid.num_points());

  const double volume_element =
      grid.spacing[0] * grid.spacing[1] * grid.spacing[2];
  double integral = 0.0;
  for (double value : field) integral += value;
  integral *= volume_element;

  // 47 + 1 - 28 = 20. An all-electron field would integrate to 48, so this
  // tolerance separates valence-only from total by a wide margin while still
  // being tight enough to catch a normalization error.
  EXPECT_NEAR(integral, 20.0, 0.05);
}

// The valence-only annotation must fire for a basis produced by
// `from_basis_name`, not just for the hand-built one above.
TEST(CubeGeneratorEcpTest, AgHCubeCommentRecordsRealEcpCoreCount) {
  auto structure = testing::create_agh_structure();
  const auto basis_set = BasisSet::from_basis_name("def2-svp", structure);
  ASSERT_TRUE(basis_set->has_ecp_electrons());

  CubeGenerator generator(basis_set);
  Eigen::VectorXd coefficients =
      Eigen::VectorXd::Zero(basis_set->get_num_atomic_orbitals());
  coefficients[0] = 1.0;
  const auto output =
      std::filesystem::temp_directory_path() / "qdk_agh_orbital.cube";

  generator.orbital(coefficients, output.string(), single_point_grid());

  const std::string comment = first_line_of(output);
  EXPECT_NE(comment.find("valence-only"), std::string::npos) << comment;
  EXPECT_NE(comment.find("28 core electrons"), std::string::npos) << comment;
  std::filesystem::remove(output);
}

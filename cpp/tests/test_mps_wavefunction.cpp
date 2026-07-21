/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <gtest/gtest.h>

#include <memory>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <stdexcept>
#include <type_traits>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

namespace {

MPSSite::PhysicalSlicePtr make_slice(std::size_t left, std::size_t right,
                                     double value) {
  using Slice = SymmetryBlockedTensor<2>;
  auto trivial =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  Slice::ExtentsArray extents;
  extents[0][SymmetryLabel{}] = left;
  extents[1][SymmetryLabel{}] = right;

  auto block = std::make_shared<const Eigen::MatrixXd>(
      Eigen::MatrixXd::Constant(left, right, value));
  Slice::BlockMap blocks;
  blocks[{SymmetryLabel{}, SymmetryLabel{}}] = block;
  Slice slice({trivial, trivial}, std::move(extents), std::move(blocks));
  return std::make_shared<const MPSSite::PhysicalSlice>(std::move(slice));
}

MPSSite::PhysicalSlicePtr make_symmetry_slice(bool include_particle_number) {
  using Slice = SymmetryBlockedTensor<2>;
  const auto symmetries =
      include_particle_number
          ? std::make_shared<const SymmetryProduct>(SymmetryProduct(
                {axes::spin(1, false), axes::particle_number(2)}))
          : std::make_shared<const SymmetryProduct>(
                SymmetryProduct({axes::spin(1, false)}));
  const SymmetryLabel label =
      include_particle_number
          ? SymmetryLabel({axes::alpha(), axes::particle_number_value(1)})
          : SymmetryLabel({axes::alpha()});
  Slice::ExtentsArray extents;
  extents[0][label] = 1;
  extents[1][label] = 1;
  Slice::BlockMap blocks;
  blocks[{label, label}] =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Ones(1, 1));
  return std::make_shared<const MPSSite::PhysicalSlice>(
      Slice({symmetries, symmetries}, std::move(extents), std::move(blocks)));
}

AbelianMPSContainer::SitePtr make_site(std::size_t left, std::size_t right,
                                       std::size_t physical = 4) {
  std::vector<MPSSite::PhysicalSlicePtr> slices;
  for (std::size_t p = 0; p < physical; ++p) {
    Eigen::MatrixXd values = Eigen::MatrixXd::Zero(left, right);
    for (std::size_t bond = 0; bond < left; ++bond) {
      if (right > 0 && bond / right == p) {
        values(static_cast<Eigen::Index>(bond),
               static_cast<Eigen::Index>(bond % right)) = 1.0;
      }
    }
    using Slice = SymmetryBlockedTensor<2>;
    auto trivial =
        std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
    Slice::ExtentsArray extents;
    extents[0][SymmetryLabel{}] = left;
    extents[1][SymmetryLabel{}] = right;
    Slice::BlockMap blocks;
    blocks[{SymmetryLabel{}, SymmetryLabel{}}] =
        std::make_shared<const Eigen::MatrixXd>(std::move(values));
    slices.push_back(std::make_shared<const MPSSite::PhysicalSlice>(
        Slice({trivial, trivial}, std::move(extents), std::move(blocks))));
  }
  return std::make_shared<const MPSSite>(
      std::move(slices), std::vector<SymmetryLabel>{SymmetryLabel{}},
      std::vector<SymmetryLabel>{SymmetryLabel{}});
}

std::shared_ptr<const SymmetryBlockedScalar<std::size_t>> make_particle_count(
    std::size_t count) {
  using Scalar = SymmetryBlockedScalar<std::size_t>;
  auto trivial =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  Scalar::BlockMap blocks;
  blocks[{SymmetryLabel{}}] = std::make_shared<const std::size_t>(count);
  return std::make_shared<const Scalar>(Scalar::SymmetriesArray{trivial},
                                        std::move(blocks));
}

}  // namespace

static_assert(std::is_abstract_v<MPSContainer>);
static_assert(std::is_base_of_v<MPSContainer, AbelianMPSContainer>);

TEST(AbelianMPSContainerTest, StoresSparseSitesAndMetadata) {
  std::vector<AbelianMPSContainer::SitePtr> sites = {make_site(1, 2),
                                                     make_site(2, 1)};
  auto total_num_particles = make_particle_count(4);
  const std::vector<Configuration> physical_basis = {
      Configuration::from_spin_half_string("0"),
      Configuration::from_spin_half_string("u"),
      Configuration::from_spin_half_string("d"),
      Configuration::from_spin_half_string("2")};
  const std::vector<std::size_t> site_to_orbital_order = {1, 0};

  AbelianMPSContainer wavefunction(
      std::move(sites), testing::create_test_orbitals(2, 2, true),
      total_num_particles, nullptr, 0, physical_basis, site_to_orbital_order);

  EXPECT_EQ(wavefunction.num_sites(), 2u);
  EXPECT_EQ(wavefunction.max_bond_dimension(), 2u);
  EXPECT_EQ(wavefunction.total_num_particles(), total_num_particles);
  EXPECT_EQ(wavefunction.get_container_type(), "mps");
  EXPECT_EQ(wavefunction.sectors(),
            std::vector<std::string>{Wavefunction::DEFAULT_SECTOR});
  EXPECT_EQ(wavefunction.orthogonality_center(), 0u);
  EXPECT_EQ(wavefunction.physical_basis(), physical_basis);
  EXPECT_EQ(wavefunction.site_to_orbital_order(), site_to_orbital_order);
  EXPECT_FALSE(wavefunction.is_complex());
  EXPECT_EQ(wavefunction.sites().front()->physical_slices().size(), 4u);
}

TEST(AbelianMPSContainerTest, ExposesCommonMPSInterface) {
  AbelianMPSContainer wavefunction({make_site(1, 1)},
                                   testing::create_test_orbitals(1, 1, true));
  const MPSContainer& base = wavefunction;

  EXPECT_EQ(base.num_sites(), 1u);
  EXPECT_EQ(base.get_container_type(), "mps");
  EXPECT_EQ(base.get_orbitals(), wavefunction.get_orbitals());
  EXPECT_EQ(base.site_to_orbital_order(), std::vector<std::size_t>{0});
  EXPECT_FALSE(base.is_complex());
}

TEST(AbelianMPSContainerTest, RepresentsCanonicalFormByOrthogonalityCenter) {
  AbelianMPSContainer right_canonical(
      {make_site(1, 1), make_site(1, 1), make_site(1, 1)},
      testing::create_test_orbitals(3, 3, true), nullptr, nullptr, 0);
  AbelianMPSContainer mixed({make_site(1, 1), make_site(1, 1), make_site(1, 1)},
                            testing::create_test_orbitals(3, 3, true), nullptr,
                            nullptr, 1);
  AbelianMPSContainer left_canonical(
      {make_site(1, 1), make_site(1, 1), make_site(1, 1)},
      testing::create_test_orbitals(3, 3, true), nullptr, nullptr, 2);
  AbelianMPSContainer unspecified(
      {make_site(1, 1), make_site(1, 1), make_site(1, 1)},
      testing::create_test_orbitals(3, 3, true), nullptr, nullptr,
      std::nullopt);

  EXPECT_EQ(right_canonical.orthogonality_center(), 0u);
  EXPECT_EQ(mixed.orthogonality_center(), 1u);
  EXPECT_EQ(left_canonical.orthogonality_center(), 2u);
  EXPECT_EQ(unspecified.orthogonality_center(), std::nullopt);
}

TEST(AbelianMPSContainerTest, ValidatesSitesAroundOrthogonalityCenter) {
  auto unnormalized = std::make_shared<const MPSSite>(
      std::vector<MPSSite::PhysicalSlicePtr>{make_slice(1, 1, 2.0)},
      std::vector<SymmetryLabel>{SymmetryLabel{}},
      std::vector<SymmetryLabel>{SymmetryLabel{}});

  EXPECT_THROW(AbelianMPSContainer({make_site(1, 1, 1), unnormalized},
                                   testing::create_test_orbitals(2, 2, true),
                                   nullptr, nullptr, 0),
               std::invalid_argument);
  EXPECT_NO_THROW(AbelianMPSContainer({make_site(1, 1, 1), unnormalized},
                                      testing::create_test_orbitals(2, 2, true),
                                      nullptr, nullptr, 1));
  EXPECT_NO_THROW(AbelianMPSContainer({make_site(1, 1, 1), unnormalized},
                                      testing::create_test_orbitals(2, 2, true),
                                      nullptr, nullptr, std::nullopt));
  EXPECT_THROW(AbelianMPSContainer({make_site(1, 1)},
                                   testing::create_test_orbitals(1, 1, true),
                                   nullptr, nullptr, 1),
               std::invalid_argument);
}

TEST(MPSSiteTest, MaterializesOneSiteWithLeftPhysicalPacking) {
  const auto site = make_site(2, 1, 3);
  const auto dense = std::get<Eigen::MatrixXd>(site->to_dense());

  ASSERT_EQ(dense.rows(), 6);
  ASSERT_EQ(dense.cols(), 1);
  EXPECT_EQ(dense.col(0),
            (Eigen::VectorXd(6) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0).finished());
}

TEST(MPSSiteTest, RejectsZeroBondDimensions) {
  EXPECT_THROW(make_site(0, 1), std::invalid_argument);
  EXPECT_THROW(make_site(1, 0), std::invalid_argument);
}

TEST(MPSSiteTest, RequiresParticleNumberForSpinResolvedBondSectors) {
  const auto spin_only = make_symmetry_slice(false);
  EXPECT_THROW(MPSSite({spin_only}, {SymmetryLabel({axes::alpha()})},
                       {SymmetryLabel({axes::alpha()})}),
               std::invalid_argument);

  const auto spin_and_number = make_symmetry_slice(true);
  const SymmetryLabel label({axes::alpha(), axes::particle_number_value(1)});
  EXPECT_NO_THROW(MPSSite({spin_and_number}, {label}, {label}));
}

TEST(AbelianMPSContainerTest, RejectsMismatchedAdjacentBonds) {
  std::vector<AbelianMPSContainer::SitePtr> sites = {make_site(1, 2),
                                                     make_site(3, 1)};
  EXPECT_THROW(AbelianMPSContainer(std::move(sites),
                                   testing::create_test_orbitals(2, 2, true)),
               std::invalid_argument);
}

TEST(AbelianMPSContainerTest, RejectsPhysicalBasisSizeMismatch) {
  std::vector<AbelianMPSContainer::SitePtr> sites = {make_site(1, 1, 3)};

  EXPECT_THROW(AbelianMPSContainer(std::move(sites),
                                   testing::create_test_orbitals(1, 1, true),
                                   nullptr, nullptr, 0,
                                   {Configuration::from_spin_half_string("0"),
                                    Configuration::from_spin_half_string("u")}),
               std::invalid_argument);
}

TEST(AbelianMPSContainerTest, RejectsInvalidPhysicalBasisConfigurations) {
  EXPECT_THROW(AbelianMPSContainer({make_site(1, 1)},
                                   testing::create_test_orbitals(1, 1, true),
                                   nullptr, nullptr, 0,
                                   {Configuration::from_bitstring("0"),
                                    Configuration::from_bitstring("1"),
                                    Configuration::from_bitstring("0"),
                                    Configuration::from_bitstring("1")}),
               std::invalid_argument);
}

TEST(AbelianMPSContainerTest, RejectsInvalidSiteToOrbitalOrder) {
  EXPECT_THROW(AbelianMPSContainer({make_site(1, 1), make_site(1, 1)},
                                   testing::create_test_orbitals(2, 2, true),
                                   nullptr, nullptr, 0, {}, {0}),
               std::invalid_argument);
  EXPECT_THROW(AbelianMPSContainer({make_site(1, 1), make_site(1, 1)},
                                   testing::create_test_orbitals(2, 2, true),
                                   nullptr, nullptr, 0, {}, {0, 0}),
               std::invalid_argument);
}

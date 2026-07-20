/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <gtest/gtest.h>

#include <memory>
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

AbelianMPSContainer::SitePtr make_site(std::size_t left, std::size_t right,
                                       std::size_t physical = 4) {
  std::vector<MPSSite::PhysicalSlicePtr> slices;
  for (std::size_t p = 0; p < physical; ++p) {
    slices.push_back(make_slice(left, right, static_cast<double>(p + 1)));
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
  const std::vector<std::string> physical_basis = {"empty", "alpha", "beta",
                                                   "alpha_beta"};

  AbelianMPSContainer wavefunction(
      std::move(sites), testing::create_test_orbitals(2, 2, true),
      total_num_particles, nullptr, MPSCanonicalForm::RightNormalized,
      std::nullopt, 1e-8, physical_basis);

  EXPECT_EQ(wavefunction.num_sites(), 2u);
  EXPECT_EQ(wavefunction.max_bond_dimension(), 2u);
  EXPECT_EQ(wavefunction.total_num_particles(), total_num_particles);
  EXPECT_EQ(wavefunction.get_container_type(), "mps");
  EXPECT_EQ(wavefunction.sectors(),
            std::vector<std::string>{Wavefunction::DEFAULT_SECTOR});
  EXPECT_EQ(wavefunction.canonical_form(), MPSCanonicalForm::RightNormalized);
  EXPECT_EQ(wavefunction.canonical_center(), std::nullopt);
  EXPECT_EQ(wavefunction.discarded_weight(), 1e-8);
  EXPECT_EQ(wavefunction.physical_basis(), physical_basis);
  EXPECT_FALSE(wavefunction.is_complex());
  EXPECT_EQ(wavefunction.sites().front()->physical_slices().size(), 4u);
}

TEST(AbelianMPSContainerTest, ExposesCommonMPSInterface) {
  AbelianMPSContainer wavefunction({make_site(1, 1)},
                                   testing::create_test_orbitals(1, 1, true));
  const MPSContainer& base = wavefunction;

  EXPECT_EQ(base.num_sites(), 1u);
  EXPECT_EQ(base.get_container_type(), "mps");
  EXPECT_EQ(base.orbitals(), wavefunction.orbitals());
  EXPECT_FALSE(base.is_complex());
}

TEST(MPSSiteTest, MaterializesOneSiteWithLeftPhysicalPacking) {
  const auto site = make_site(2, 1, 3);
  const auto dense = std::get<Eigen::MatrixXd>(site->to_dense());

  ASSERT_EQ(dense.rows(), 6);
  ASSERT_EQ(dense.cols(), 1);
  EXPECT_EQ(dense.col(0),
            (Eigen::VectorXd(6) << 1.0, 2.0, 3.0, 1.0, 2.0, 3.0).finished());
}

TEST(AbelianMPSContainerTest, RejectsMismatchedAdjacentBonds) {
  std::vector<AbelianMPSContainer::SitePtr> sites = {make_site(1, 2),
                                                     make_site(3, 1)};
  EXPECT_THROW(AbelianMPSContainer(std::move(sites),
                                   testing::create_test_orbitals(2, 2, true)),
               std::invalid_argument);
}

TEST(AbelianMPSContainerTest, RejectsInvalidMixedCanonicalCenter) {
  std::vector<AbelianMPSContainer::SitePtr> sites = {make_site(1, 1)};

  EXPECT_THROW(AbelianMPSContainer(
                   std::move(sites), testing::create_test_orbitals(1, 1, true),
                   nullptr, nullptr, MPSCanonicalForm::Mixed, 2),
               std::invalid_argument);
}

TEST(AbelianMPSContainerTest, RejectsPhysicalBasisSizeMismatch) {
  std::vector<AbelianMPSContainer::SitePtr> sites = {make_site(1, 1, 3)};

  EXPECT_THROW(AbelianMPSContainer(
                   std::move(sites), testing::create_test_orbitals(1, 1, true),
                   nullptr, nullptr, MPSCanonicalForm::Unspecified,
                   std::nullopt, 0.0, {"zero", "one"}),
               std::invalid_argument);
}

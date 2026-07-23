// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <memory>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/wavefunction_containers/abelian_mps_wavefunction.hpp>
#include <stdexcept>
#include <type_traits>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

namespace {

// Per-physical-state particle counts for a spin-half orbital:
// |0⟩ -> 0, |↑⟩ -> 1, |↓⟩ -> 1, |2⟩ -> 2
constexpr std::size_t DELTA_N[] = {0, 1, 1, 2};

/**
 * @brief Create a particle-number-blocked SymmetryProduct for MPS bonds.
 * @param max_n Maximum particle-number label (inclusive).
 */
std::shared_ptr<const SymmetryProduct> make_bond_symmetries(std::size_t max_n) {
  return std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(max_n)}));
}

/**
 * @brief Create a particle-number-blocked MPS site.
 *
 * Constructs a site where each bond has a single particle-number sector whose
 * label is @p left_n (left bond) and @p right_n (right bond). The sector
 * dimensions are @p left and @p right respectively.
 *
 * For each physical state p with DELTA_N[p], a block is populated only if
 * left_n + DELTA_N[p] == right_n (particle number conservation). The populated
 * block is set to a normalized isometry.
 *
 * @param left Left bond dimension (single sector).
 * @param right Right bond dimension (single sector).
 * @param left_n Particle-number label of the left bond sector.
 * @param right_n Particle-number label of the right bond sector.
 * @param max_n Maximum particle number for the axis (must be >= both labels).
 * @param physical Number of physical states (default 4).
 */
AbelianMPSContainer::SitePtr make_site(std::size_t left, std::size_t right,
                                       std::size_t left_n = 0,
                                       std::size_t right_n = 0,
                                       std::size_t max_n = 2,
                                       std::size_t physical = 4) {
  const auto symmetries = make_bond_symmetries(max_n);
  const SymmetryLabel left_label({axes::particle_number_value(left_n)});
  const SymmetryLabel right_label({axes::particle_number_value(right_n)});

  using Slice = SymmetryBlockedTensor<2>;
  Slice::ExtentsArray extents;
  extents[0][left_label] = left;
  extents[1][right_label] = right;

  const auto delta_n = right_n - left_n;
  const auto conserving_slices = static_cast<std::size_t>(
      std::count(std::begin(DELTA_N), std::end(DELTA_N), delta_n));
  std::size_t conserving_index = 0;
  std::vector<AbelianMPSSite::PhysicalSlicePtr> slices;
  for (std::size_t p = 0; p < physical; ++p) {
    Slice::BlockMap blocks;
    if (left_n + DELTA_N[p] == right_n) {
      Eigen::MatrixXd values = Eigen::MatrixXd::Zero(left, right);
      if (left * conserving_slices == right) {
        values.block(0, conserving_index * left, left, left).setIdentity();
      } else if (right * conserving_slices == left) {
        values.block(conserving_index * right, 0, right, right).setIdentity();
      } else {
        values = Eigen::MatrixXd::Identity(left, right) /
                 std::sqrt(static_cast<double>(conserving_slices));
      }
      blocks[{left_label, right_label}] =
          std::make_shared<const Eigen::MatrixXd>(std::move(values));
      ++conserving_index;
    }
    // If conservation is not satisfied, the block map is empty for this slice.
    slices.push_back(std::make_shared<const AbelianMPSSite::PhysicalSlice>(
        Slice({symmetries, symmetries}, extents, std::move(blocks))));
  }
  return std::make_shared<const AbelianMPSSite>(
      std::move(slices), std::vector<SymmetryLabel>{left_label},
      std::vector<SymmetryLabel>{right_label});
}

AbelianMPSSite::PhysicalSlicePtr make_symmetry_slice(
    bool include_particle_number) {
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
  return std::make_shared<const AbelianMPSSite::PhysicalSlice>(
      Slice({symmetries, symmetries}, std::move(extents), std::move(blocks)));
}

/**
 * @brief Create a trivially-blocked MPS site (no particle-number symmetry).
 * Used for AbelianMPSSite-only tests that don't involve AbelianMPSContainer.
 */
AbelianMPSContainer::SitePtr make_trivial_site(std::size_t left,
                                               std::size_t right,
                                               std::size_t physical = 4) {
  using Slice = SymmetryBlockedTensor<2>;
  auto trivial =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  std::vector<AbelianMPSSite::PhysicalSlicePtr> slices;
  for (std::size_t p = 0; p < physical; ++p) {
    Eigen::MatrixXd values = Eigen::MatrixXd::Zero(left, right);
    for (std::size_t bond = 0; bond < left; ++bond) {
      if (right > 0 && bond / right == p) {
        values(static_cast<Eigen::Index>(bond),
               static_cast<Eigen::Index>(bond % right)) = 1.0;
      }
    }
    Slice::ExtentsArray extents;
    extents[0][SymmetryLabel{}] = left;
    extents[1][SymmetryLabel{}] = right;
    Slice::BlockMap blocks;
    blocks[{SymmetryLabel{}, SymmetryLabel{}}] =
        std::make_shared<const Eigen::MatrixXd>(std::move(values));
    slices.push_back(std::make_shared<const AbelianMPSSite::PhysicalSlice>(
        Slice({trivial, trivial}, std::move(extents), std::move(blocks))));
  }
  return std::make_shared<const AbelianMPSSite>(
      std::move(slices), std::vector<SymmetryLabel>{SymmetryLabel{}},
      std::vector<SymmetryLabel>{SymmetryLabel{}});
}

/**
 * @brief Create a particle-number-blocked unnormalized site (for canonicality
 * rejection tests). Single sector on each bond with the given scalar value.
 */
AbelianMPSContainer::SitePtr make_unnormalized_site(std::size_t left_n = 0,
                                                    std::size_t right_n = 0,
                                                    std::size_t max_n = 2,
                                                    std::size_t physical = 4,
                                                    double value = 2.0) {
  const auto symmetries = make_bond_symmetries(max_n);
  const SymmetryLabel left_label({axes::particle_number_value(left_n)});
  const SymmetryLabel right_label({axes::particle_number_value(right_n)});
  using Slice = SymmetryBlockedTensor<2>;
  Slice::ExtentsArray extents;
  extents[0][left_label] = 1;
  extents[1][right_label] = 1;
  std::vector<AbelianMPSSite::PhysicalSlicePtr> slices;
  for (std::size_t p = 0; p < physical; ++p) {
    Slice::BlockMap blocks;
    if (left_n + DELTA_N[p] == right_n) {
      blocks[{left_label, right_label}] =
          std::make_shared<const Eigen::MatrixXd>(
              Eigen::MatrixXd::Constant(1, 1, value));
    }
    slices.push_back(std::make_shared<const AbelianMPSSite::PhysicalSlice>(
        Slice({symmetries, symmetries}, extents, std::move(blocks))));
  }
  return std::make_shared<const AbelianMPSSite>(
      std::move(slices), std::vector<SymmetryLabel>{left_label},
      std::vector<SymmetryLabel>{right_label});
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
  // 2-site chain: N=0 -> N=1 -> N=2 (one particle added per site)
  std::vector<AbelianMPSContainer::SitePtr> sites = {make_site(1, 2, 0, 1, 4),
                                                     make_site(2, 1, 1, 2, 4)};
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
  auto unnormalized = make_unnormalized_site(0, 0, 2, 4, 2.0);

  // Unnormalized at position 1 with center=0 → site 1 must be right-canonical
  // → rejected
  EXPECT_THROW(AbelianMPSContainer({make_site(1, 1), unnormalized},
                                   testing::create_test_orbitals(2, 2, true),
                                   nullptr, nullptr, 0),
               std::invalid_argument);
  // Unnormalized at position 1 with center=1 → site 1 is the center → accepted
  EXPECT_NO_THROW(AbelianMPSContainer({make_site(1, 1), unnormalized},
                                      testing::create_test_orbitals(2, 2, true),
                                      nullptr, nullptr, 1));
  // Unspecified canonicalization → accepted
  EXPECT_NO_THROW(AbelianMPSContainer({make_site(1, 1), unnormalized},
                                      testing::create_test_orbitals(2, 2, true),
                                      nullptr, nullptr, std::nullopt));
  // Out-of-range center → rejected
  EXPECT_THROW(AbelianMPSContainer({make_site(1, 1)},
                                   testing::create_test_orbitals(1, 1, true),
                                   nullptr, nullptr, 1),
               std::invalid_argument);
}

TEST(AbelianMPSSiteTest, MaterializesOneSiteWithLeftPhysicalPacking) {
  // Use trivially-blocked site to test dense layout (AbelianMPSSite doesn't
  // require particle-number blocking — only AbelianMPSContainer does).
  const auto site = make_trivial_site(2, 1, 3);
  const auto dense = std::get<Eigen::MatrixXd>(site->to_dense());

  ASSERT_EQ(dense.rows(), 6);
  ASSERT_EQ(dense.cols(), 1);
  EXPECT_EQ(dense.col(0),
            (Eigen::VectorXd(6) << 1.0, 0.0, 0.0, 0.0, 1.0, 0.0).finished());
}

TEST(AbelianMPSSiteTest, RejectsZeroBondDimensions) {
  EXPECT_THROW(make_trivial_site(0, 1), std::invalid_argument);
  EXPECT_THROW(make_trivial_site(1, 0), std::invalid_argument);
}

TEST(AbelianMPSSiteTest, RequiresParticleNumberForSpinResolvedBondSectors) {
  const auto spin_only = make_symmetry_slice(false);
  EXPECT_THROW(AbelianMPSSite({spin_only}, {SymmetryLabel({axes::alpha()})},
                              {SymmetryLabel({axes::alpha()})}),
               std::invalid_argument);

  const auto spin_and_number = make_symmetry_slice(true);
  const SymmetryLabel label({axes::alpha(), axes::particle_number_value(1)});
  EXPECT_NO_THROW(AbelianMPSSite({spin_and_number}, {label}, {label}));
}

TEST(AbelianMPSContainerTest, RejectsTriviallyBlockedSites) {
  // Trivially-blocked sites (no ParticleNumber axis) must be rejected.
  EXPECT_THROW(AbelianMPSContainer({make_trivial_site(1, 1)},
                                   testing::create_test_orbitals(1, 1, true)),
               std::invalid_argument);
}

TEST(AbelianMPSContainerTest, RejectsMismatchedAdjacentBonds) {
  // Right bond of site 0 has dim=2, left bond of site 1 has dim=3 → mismatch.
  std::vector<AbelianMPSContainer::SitePtr> sites = {make_site(1, 2, 0, 1, 4),
                                                     make_site(3, 1, 1, 2, 4)};
  EXPECT_THROW(AbelianMPSContainer(std::move(sites),
                                   testing::create_test_orbitals(2, 2, true)),
               std::invalid_argument);
}

TEST(AbelianMPSContainerTest, RejectsPhysicalBasisSizeMismatch) {
  // Site has 3 physical states but physical_basis has 2 entries → mismatch.
  std::vector<AbelianMPSContainer::SitePtr> sites = {
      make_site(1, 1, 0, 0, 2, 3)};

  EXPECT_THROW(AbelianMPSContainer(std::move(sites),
                                   testing::create_test_orbitals(1, 1, true),
                                   nullptr, nullptr, std::nullopt,
                                   {Configuration::from_spin_half_string("0"),
                                    Configuration::from_spin_half_string("u")}),
               std::invalid_argument);
}

TEST(AbelianMPSContainerTest, RejectsInvalidPhysicalBasisConfigurations) {
  EXPECT_THROW(AbelianMPSContainer({make_site(1, 1)},
                                   testing::create_test_orbitals(1, 1, true),
                                   nullptr, nullptr, std::nullopt,
                                   {Configuration::from_bitstring("0"),
                                    Configuration::from_bitstring("1"),
                                    Configuration::from_bitstring("0"),
                                    Configuration::from_bitstring("1")}),
               std::invalid_argument);
}

TEST(AbelianMPSContainerTest, RejectsInvalidSiteToOrbitalOrder) {
  EXPECT_THROW(AbelianMPSContainer({make_site(1, 1), make_site(1, 1)},
                                   testing::create_test_orbitals(2, 2, true),
                                   nullptr, nullptr, std::nullopt, {}, {0}),
               std::invalid_argument);
  EXPECT_THROW(AbelianMPSContainer({make_site(1, 1), make_site(1, 1)},
                                   testing::create_test_orbitals(2, 2, true),
                                   nullptr, nullptr, std::nullopt, {}, {0, 0}),
               std::invalid_argument);
}

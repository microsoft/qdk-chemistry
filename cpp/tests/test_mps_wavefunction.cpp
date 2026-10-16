// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <array>
#include <complex>
#include <limits>
#include <memory>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "ut_common.hpp"

using namespace qdk::chemistry::data;

namespace {

template <typename Scalar>
MPSContainer::SitePtr make_site(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>& packed,
    std::size_t physical_dimension = 4) {
  using SBT = SymmetryBlockedTensor<3, Scalar>;
  auto trivial =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  const SymmetryLabel label;
  typename SBT::ExtentsArray extents;
  extents[0][label] =
      static_cast<std::size_t>(packed.rows()) / physical_dimension;
  extents[1][label] = physical_dimension;
  extents[2][label] = static_cast<std::size_t>(packed.cols());
  typename SBT::BlockMap blocks{
      {{label, label, label},
       std::make_shared<const Tensor<3, Scalar>>(packed)}};
  auto tensor = std::make_shared<const MPSSite::TensorVariant>(
      SBT({trivial, trivial, trivial}, extents, std::move(blocks)));
  return std::make_shared<const MPSSite>(
      tensor, MPSSite::SectorOrders{{{label}, {label}, {label}}});
}

class MPSContainerTest : public ::testing::Test {
 protected:
  Eigen::MatrixXd random_matrix(Eigen::Index rows, Eigen::Index cols) {
    return Eigen::MatrixXd::NullaryExpr(rows, cols,
                                        [&] { return distribution(rng); });
  }

  void SetUp() override {
    orbitals = std::make_shared<ModelOrbitals>(2);
    data = {random_matrix(4, 3), random_matrix(12, 1)};
    sites = {make_site(data[0]), make_site(data[1])};
    counts = std::make_shared<const MPSContainer::ParticleCount>(
        MPSContainer::ParticleCount::SymmetriesArray{
            std::make_shared<const SymmetryProduct>(
                SymmetryProduct::trivial())},
        MPSContainer::ParticleCount::BlockMap{
            {{SymmetryLabel{}}, std::make_shared<const std::size_t>(2)}});
  }

  std::unique_ptr<MPSContainer> make_container() {
    return std::make_unique<MPSContainer>(sites, orbitals, counts, counts, 1,
                                          std::vector<std::size_t>{1, 0});
  }

  std::mt19937 rng{42};
  std::uniform_real_distribution<double> distribution{-1.0, 1.0};
  std::shared_ptr<Orbitals> orbitals;
  std::shared_ptr<const MPSContainer::ParticleCount> counts;
  std::array<Eigen::MatrixXd, 2> data;
  std::vector<MPSContainer::SitePtr> sites;
};

}  // namespace

TEST_F(MPSContainerTest, BasicProperties) {
  auto container = make_container();
  EXPECT_EQ(container->get_container_type(), "mps");
  EXPECT_EQ(container->get_orbitals(), orbitals);
  EXPECT_EQ(container->num_sites(), 2u);
  EXPECT_EQ(container->max_bond_dimension(), 3u);
  EXPECT_FALSE(container->is_complex());
  EXPECT_EQ(container->orthogonality_center(), 1u);
  EXPECT_EQ(container->site_to_orbital_order(),
            (std::vector<std::size_t>{1, 0}));
  EXPECT_EQ(container->total_num_particles()->value(SymmetryLabel{}), 2u);
  EXPECT_EQ(container->active_num_particles()->value(SymmetryLabel{}), 2u);
  for (std::size_t i = 0; i < sites.size(); ++i) {
    EXPECT_EQ(container->sites()[i], sites[i]);
    EXPECT_EQ(sites[i]->physical_dimension(), 4u);
    EXPECT_EQ(sites[i]->physical_basis()[1],
              Configuration::from_spin_half_string("u"));
    EXPECT_TRUE(std::get<Eigen::MatrixXd>(sites[i]->to_dense())
                    .isApprox(data[i], testing::wf_tolerance));
  }
}

TEST_F(MPSContainerTest, BinaryLocalBasisAndOptionalMetadata) {
  const Eigen::MatrixXd binary = random_matrix(2, 1);
  MPSContainer container({make_site(binary, 2)},
                         std::make_shared<ModelOrbitals>(1));
  EXPECT_EQ(container.sites()[0]->physical_basis()[1],
            Configuration::from_bitstring("1"));
  EXPECT_EQ(container.max_bond_dimension(), 1u);
  EXPECT_EQ(container.site_to_orbital_order(), (std::vector<std::size_t>{0}));
  EXPECT_FALSE(container.orthogonality_center().has_value());
  EXPECT_FALSE(container.has_total_num_particles());
  EXPECT_FALSE(container.has_active_num_particles());
  EXPECT_FALSE(container.has_one_rdm_spin_traced());
  EXPECT_FALSE(container.has_two_rdm_spin_traced());
  EXPECT_THROW(container.total_num_particles(), std::runtime_error);
  EXPECT_THROW(container.active_num_particles(), std::runtime_error);
  auto restored = MPSContainer::from_json(container.to_json());
  EXPECT_FALSE(restored->has_total_num_particles());
  EXPECT_FALSE(restored->orthogonality_center().has_value());
  EXPECT_TRUE(std::get<Eigen::MatrixXd>(restored->sites()[0]->to_dense())
                  .isApprox(binary, testing::wf_tolerance));
}

TEST_F(MPSContainerTest, BlockedSitePreservesBlocksAndSectorOrder) {
  using SBT = SymmetryBlockedTensor<3>;
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(2)}));
  const SymmetryLabel zero({axes::particle_number_value(0)});
  const SymmetryLabel one({axes::particle_number_value(1)});
  const SymmetryLabel two({axes::particle_number_value(2)});
  SBT::ExtentsArray extents;
  extents[0] = {{zero, 1}, {one, 2}};
  extents[1] = {{zero, 1}, {one, 2}, {two, 1}};
  extents[2] = {{one, 2}, {two, 1}};
  const Eigen::MatrixXd block = random_matrix(4, 2);
  auto tensor = std::make_shared<const MPSSite::TensorVariant>(
      SBT({symmetry, symmetry, symmetry}, extents,
          {{{one, one, one}, std::make_shared<const Eigen::MatrixXd>(block)}}));
  MPSSite::SectorOrders orders{{{one, zero}, {two, one, zero}, {two, one}}};
  MPSSite site(tensor, orders,
               {Configuration::from_spin_half_string("2"),
                Configuration::from_spin_half_string("u"),
                Configuration::from_spin_half_string("d"),
                Configuration::from_spin_half_string("0")});
  EXPECT_EQ(site.sector_orders(), orders);
  const auto& stored = std::get<SBT>(site.tensor());
  EXPECT_EQ(stored.extents(), extents);
  EXPECT_EQ(stored.num_blocks(), 1u);
  EXPECT_TRUE(
      stored.block({one, one, one}).isApprox(block, testing::wf_tolerance));
  Eigen::MatrixXd expected = Eigen::MatrixXd::Zero(12, 3);
  expected.block(1, 1, 2, 2) = block.topRows(2);
  expected.block(5, 1, 2, 2) = block.bottomRows(2);
  EXPECT_TRUE(std::get<Eigen::MatrixXd>(site.to_dense())
                  .isApprox(expected, testing::wf_tolerance));
  for (std::size_t slot = 0; slot < 3; ++slot) {
    auto invalid_order = orders;
    invalid_order[slot].pop_back();
    EXPECT_THROW(MPSSite(tensor, invalid_order), std::invalid_argument);
    invalid_order = orders;
    invalid_order[slot].back() = invalid_order[slot].front();
    EXPECT_THROW(MPSSite(tensor, invalid_order), std::invalid_argument);
    invalid_order = orders;
    invalid_order[slot].front() = SymmetryLabel{};
    EXPECT_THROW(MPSSite(tensor, invalid_order), std::invalid_argument);
  }
}

TEST_F(MPSContainerTest, SerializationAndClone) {
  Wavefunction original(make_container());
  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_serialization_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  original.to_hdf5(file);
  std::vector<std::shared_ptr<Wavefunction>> copies{
      std::make_shared<Wavefunction>(original),
      Wavefunction::from_json(original.to_json()),
      Wavefunction::from_hdf5(file)};
  for (const auto& copy : copies) {
    ASSERT_TRUE(copy->has_container_type<MPSContainer>());
    const auto& container = copy->get_container<MPSContainer>();
    EXPECT_EQ(copy->content_hash(), original.content_hash());
    EXPECT_EQ(container.orthogonality_center(), 1u);
    EXPECT_EQ(container.site_to_orbital_order(),
              (std::vector<std::size_t>{1, 0}));
    EXPECT_EQ(container.total_num_particles()->value(SymmetryLabel{}), 2u);
    EXPECT_EQ(container.active_num_particles()->value(SymmetryLabel{}), 2u);
    ASSERT_EQ(container.sites().size(), data.size());
    for (std::size_t i = 0; i < data.size(); ++i) {
      EXPECT_EQ(container.sites()[i]->sector_orders(),
                sites[i]->sector_orders());
      EXPECT_EQ(container.sites()[i]->physical_basis(),
                sites[i]->physical_basis());
      EXPECT_TRUE(std::get<Eigen::MatrixXd>(container.sites()[i]->to_dense())
                      .isApprox(data[i], testing::wf_tolerance));
    }
  }
}

TEST_F(MPSContainerTest, BlockedSerializationPreservesIndexSpacesAndData) {
  using SBT = SymmetryBlockedTensor<3>;
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(2)}));
  const SymmetryLabel zero({axes::particle_number_value(0)});
  const SymmetryLabel one({axes::particle_number_value(1)});
  const SymmetryLabel two({axes::particle_number_value(2)});
  SBT::ExtentsArray first_extents, second_extents;
  first_extents[0] = {{zero, 1}};
  first_extents[1] = {{zero, 1}, {one, 2}, {two, 1}};
  first_extents[2] = {{zero, 1}, {one, 2}};
  second_extents[0] = first_extents[2];
  second_extents[1] = first_extents[1];
  second_extents[2] = {{two, 1}};
  SBT::BlockMap first_blocks{
      {{zero, one, one},
       std::make_shared<const Eigen::MatrixXd>(random_matrix(2, 2))},
      {{zero, two, zero},
       std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 1))}};
  SBT::BlockMap second_blocks{
      {{one, one, two},
       std::make_shared<const Eigen::MatrixXd>(random_matrix(4, 1))},
      {{zero, zero, two},
       std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 1))}};
  const std::vector<Configuration> basis{
      Configuration::from_spin_half_string("2"),
      Configuration::from_spin_half_string("u"),
      Configuration::from_spin_half_string("d"),
      Configuration::from_spin_half_string("0")};
  auto first = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(
          SBT({symmetry, symmetry, symmetry}, first_extents, first_blocks)),
      MPSSite::SectorOrders{{{zero}, {two, one, zero}, {one, zero}}}, basis);
  auto second = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(
          SBT({symmetry, symmetry, symmetry}, second_extents, second_blocks)),
      MPSSite::SectorOrders{{{one, zero}, {two, one, zero}, {two}}}, basis);
  MPSContainer original({first, second}, orbitals);
  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_blocked_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  original.to_hdf5(file);
  auto json_copy = MPSContainer::from_json(original.to_json());
  auto hdf_copy = MPSContainer::from_hdf5(file);
  for (const auto* copy : {json_copy.get(), hdf_copy.get()}) {
    ASSERT_EQ(copy->num_sites(), original.num_sites());
    for (std::size_t i = 0; i < original.num_sites(); ++i) {
      const auto& expected_site = *original.sites()[i];
      const auto& actual_site = *copy->sites()[i];
      EXPECT_EQ(actual_site.sector_orders(), expected_site.sector_orders());
      EXPECT_EQ(actual_site.physical_basis(), basis);
      const auto& expected = std::get<SBT>(expected_site.tensor());
      const auto& actual = std::get<SBT>(actual_site.tensor());
      EXPECT_EQ(actual.extents(), expected.extents());
      for (std::size_t slot = 0; slot < 3; ++slot) {
        EXPECT_EQ(*actual.symmetries()[slot], *expected.symmetries()[slot]);
      }
      ASSERT_EQ(actual.num_blocks(), expected.num_blocks());
      for (const auto& [labels, block] : expected.blocks()) {
        ASSERT_TRUE(actual.has_block(labels));
        EXPECT_TRUE(
            actual.block(labels).isApprox(*block, testing::wf_tolerance));
      }
    }
  }
}

TEST_F(MPSContainerTest, RejectsIncorrectBlockPacking) {
  using SBT = SymmetryBlockedTensor<3>;
  auto symmetry =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  const SymmetryLabel label;
  SBT::ExtentsArray extents;
  extents[0][label] = 2;
  extents[1][label] = 2;
  extents[2][label] = 3;
  // SBT accepts the 12 entries; MPS requires a 4x3 matrix rather than 2x6.
  auto tensor = std::make_shared<const MPSSite::TensorVariant>(
      SBT({symmetry, symmetry, symmetry}, extents,
          {{{label, label, label},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(2, 6))}}));
  const MPSSite::SectorOrders orders{{{label}, {label}, {label}}};
  EXPECT_THROW(MPSSite(tensor, orders), std::invalid_argument);
}

TEST_F(MPSContainerTest, RejectsDifferentBondSectorsWithEqualDimensions) {
  using SBT = SymmetryBlockedTensor<3>;
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(2)}));
  const SymmetryLabel zero({axes::particle_number_value(0)});
  const SymmetryLabel one({axes::particle_number_value(1)});
  const SymmetryLabel two({axes::particle_number_value(2)});
  SBT::ExtentsArray first_extents;
  first_extents[0] = {{zero, 1}};
  first_extents[1] = {{zero, 2}};
  first_extents[2] = {{zero, 1}, {one, 2}};
  auto first = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(SBT(
          {symmetry, symmetry, symmetry}, first_extents,
          {{{zero, zero, one},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(2, 2))}})),
      MPSSite::SectorOrders{{{zero}, {zero}, {zero, one}}});
  for (const std::string mismatch : {"none", "labels", "order"}) {
    SCOPED_TRACE(mismatch);
    const auto label = mismatch == "labels" ? two : zero;
    SBT::ExtentsArray extents;
    extents[0] = {{label, 1}, {one, 2}};
    extents[1] = {{zero, 2}};
    extents[2] = {{two, 1}};
    const std::vector<SymmetryLabel> order =
        mismatch == "order" ? std::vector<SymmetryLabel>{one, label}
                            : std::vector<SymmetryLabel>{label, one};
    auto second = std::make_shared<const MPSSite>(
        std::make_shared<const MPSSite::TensorVariant>(SBT(
            {symmetry, symmetry, symmetry}, extents,
            {{{one, zero, two},
              std::make_shared<const Eigen::MatrixXd>(random_matrix(4, 1))}})),
        MPSSite::SectorOrders{{order, {zero}, {two}}});
    ASSERT_EQ(first->right_bond_dimension(), second->left_bond_dimension());
    if (mismatch == "none") {
      EXPECT_NO_THROW(MPSContainer({first, second}, orbitals));
      EXPECT_NO_THROW(MPSContainer::validate_sites({first, second}));
    } else {
      EXPECT_THROW(MPSContainer({first, second}, orbitals),
                   std::invalid_argument);
      EXPECT_THROW(MPSContainer::validate_sites({first, second}),
                   std::invalid_argument);
    }
  }
}

TEST_F(MPSContainerTest, ComplexSerialization) {
  using Complex = std::complex<double>;
  const Eigen::MatrixXcd first =
      data[0].cast<Complex>() +
      Complex(0, 1) * random_matrix(4, 3).cast<Complex>();
  const Eigen::MatrixXcd second =
      data[1].cast<Complex>() +
      Complex(0, 1) * random_matrix(12, 1).cast<Complex>();
  MPSContainer original({make_site(first), make_site(second)}, orbitals);
  H5::FileAccPropList access;
  access.setCore(4096, false);
  H5::H5File file("mps_complex_in_memory.h5", H5F_ACC_TRUNC,
                  H5::FileCreatPropList::DEFAULT, access);
  original.to_hdf5(file);
  auto json_copy = MPSContainer::from_json(original.to_json());
  auto hdf_copy = MPSContainer::from_hdf5(file);
  for (const auto* copy : {&original, json_copy.get(), hdf_copy.get()}) {
    EXPECT_TRUE(copy->is_complex());
    EXPECT_TRUE(std::get<Eigen::MatrixXcd>(copy->sites()[0]->to_dense())
                    .isApprox(first, testing::wf_tolerance));
    EXPECT_TRUE(std::get<Eigen::MatrixXcd>(copy->sites()[1]->to_dense())
                    .isApprox(second, testing::wf_tolerance));
  }
}

TEST_F(MPSContainerTest, SuppliedRdms) {
  for (bool complex : {false, true}) {
    std::array<ContainerTypes::MatrixVariant, 3> one;
    std::array<ContainerTypes::VectorVariant, 4> two;
    for (auto& channel : one) {
      Eigen::MatrixXd real = random_matrix(2, 2);
      channel = real;
      if (complex)
        channel = Eigen::MatrixXcd(
            real.cast<std::complex<double>>() +
            std::complex<double>(0, 1) *
                random_matrix(2, 2).cast<std::complex<double>>());
    }
    for (auto& channel : two) {
      Eigen::VectorXd real = random_matrix(16, 1);
      channel = real;
      if (complex)
        channel = Eigen::VectorXcd(
            real.cast<std::complex<double>>() +
            std::complex<double>(0, 1) *
                random_matrix(16, 1).cast<std::complex<double>>());
    }
    MPSContainer original(sites, orbitals, nullptr, nullptr, std::nullopt, {},
                          one[0], one[1], one[2], two[0], two[1], two[2],
                          two[3]);
    H5::FileAccPropList access;
    access.setCore(4096, false);
    H5::H5File file("mps_rdms_in_memory.h5", H5F_ACC_TRUNC,
                    H5::FileCreatPropList::DEFAULT, access);
    original.to_hdf5(file);
    std::vector<std::unique_ptr<WavefunctionContainer>> copies;
    copies.push_back(original.clone());
    copies.push_back(MPSContainer::from_json(original.to_json()));
    copies.push_back(MPSContainer::from_hdf5(file));
    for (const auto& copy : copies) {
      copy->clear_caches();
      ASSERT_TRUE(copy->has_one_rdm_spin_dependent());
      ASSERT_TRUE(copy->has_two_rdm_spin_dependent());
      const auto [aa, bb] = copy->get_active_one_rdm_spin_dependent();
      const auto [aaaa, aabb, bbbb] = copy->get_active_two_rdm_spin_dependent();
      const std::array<ContainerTypes::MatrixVariant, 3> actual_one{
          copy->get_active_one_rdm_spin_traced(), aa, bb};
      const std::array<ContainerTypes::VectorVariant, 4> actual_two{
          copy->get_active_two_rdm_spin_traced(), aaaa, aabb, bbbb};
      for (std::size_t i = 0; i < one.size(); ++i) {
        std::visit(
            [&](const auto& expected) {
              using Matrix = std::decay_t<decltype(expected)>;
              EXPECT_TRUE(std::get<Matrix>(actual_one[i])
                              .isApprox(expected, testing::wf_tolerance));
            },
            one[i]);
      }
      for (std::size_t i = 0; i < two.size(); ++i) {
        std::visit(
            [&](const auto& expected) {
              using Vector = std::decay_t<decltype(expected)>;
              EXPECT_TRUE(std::get<Vector>(actual_two[i])
                              .isApprox(expected, testing::wf_tolerance));
            },
            two[i]);
      }
    }
  }
}

TEST_F(MPSContainerTest, RejectsInvalidStructure) {
  EXPECT_THROW(MPSContainer({}, orbitals), std::invalid_argument);
  EXPECT_THROW(MPSContainer::validate_sites({}), std::invalid_argument);
  EXPECT_THROW(MPSContainer::validate_sites({nullptr, sites[1]}),
               std::invalid_argument);
  EXPECT_NO_THROW(MPSContainer::validate_sites(sites));
  EXPECT_THROW(MPSContainer({nullptr, sites[1]}, orbitals),
               std::invalid_argument);
  EXPECT_THROW(MPSContainer(sites, nullptr), std::invalid_argument);
  EXPECT_THROW(MPSContainer(sites, orbitals, nullptr, nullptr, 2),
               std::invalid_argument);
  EXPECT_THROW(
      MPSContainer(sites, orbitals, nullptr, nullptr, std::nullopt, {0, 0}),
      std::invalid_argument);
  const Eigen::MatrixXd wrong_bond = random_matrix(8, 1);
  EXPECT_THROW(MPSContainer({sites[0], make_site(wrong_bond)}, orbitals),
               std::invalid_argument);
  const Eigen::MatrixXd nonfinite =
      Eigen::MatrixXd::Constant(4, 1, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(make_site(nonfinite), std::invalid_argument);
  const Eigen::MatrixXd zero = Eigen::MatrixXd::Zero(4, 1);
  EXPECT_NO_THROW(make_site(zero));
  auto json = make_container()->to_json();
  json["version"] = "9.0.0";
  EXPECT_THROW(MPSContainer::from_json(json), std::runtime_error);
}

TEST_F(MPSContainerTest, EqualSpinCountsDoNotImplyClosedShellRdms) {
  auto restricted = testing::create_test_orbitals(2, 2);
  ASSERT_TRUE(restricted->is_restricted());
  using Count = MPSContainer::ParticleCount;
  auto equal_counts = std::make_shared<const Count>(
      Count::SymmetriesArray{testing::spin_symmetry(false)},
      Count::BlockMap{
          {{axes::alpha()}, std::make_shared<const std::size_t>(1)},
          {{axes::beta()}, std::make_shared<const std::size_t>(1)}});
  ContainerTypes::MatrixVariant traced = random_matrix(2, 2);
  MPSContainer traced_only(sites, restricted, equal_counts, equal_counts,
                           std::nullopt, {}, traced);
  EXPECT_TRUE(traced_only.has_one_rdm_spin_traced());
  EXPECT_FALSE(traced_only.has_one_rdm_spin_dependent());
  EXPECT_THROW(traced_only.active_one_rdm(), std::runtime_error);
  Eigen::MatrixXd aa = random_matrix(2, 2), bb = random_matrix(2, 2);
  Eigen::VectorXd aaaa = random_matrix(16, 1), aabb = random_matrix(16, 1),
                  bbbb = random_matrix(16, 1);
  MPSContainer supplied(
      sites, restricted, equal_counts, equal_counts, std::nullopt, {},
      std::nullopt, ContainerTypes::MatrixVariant{aa},
      ContainerTypes::MatrixVariant{bb}, std::nullopt,
      ContainerTypes::VectorVariant{aaaa}, ContainerTypes::VectorVariant{aabb},
      ContainerTypes::VectorVariant{bbbb});
  Eigen::VectorXd expected = aaaa + bbbb + aabb;
  for (std::size_t p = 0; p < 2; ++p)
    for (std::size_t q = 0; q < 2; ++q)
      for (std::size_t r = 0; r < 2; ++r)
        for (std::size_t s = 0; s < 2; ++s)
          expected(p * 8 + q * 4 + r * 2 + s) +=
              aabb(r * 8 + s * 4 + p * 2 + q);
  EXPECT_TRUE(
      std::get<Eigen::MatrixXd>(supplied.get_active_one_rdm_spin_traced())
          .isApprox((aa + bb).eval(), testing::wf_tolerance));
  EXPECT_TRUE(
      std::get<Eigen::VectorXd>(supplied.get_active_two_rdm_spin_traced())
          .isApprox(expected, testing::wf_tolerance));
}

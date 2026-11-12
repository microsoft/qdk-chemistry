// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <complex>
#include <limits>
#include <map>
#include <memory>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
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
    return std::make_unique<MPSContainer>(sites, orbitals, counts, 1,
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
  EXPECT_FALSE(container.has_active_num_particles());
  EXPECT_FALSE(container.has_one_rdm_spin_traced());
  EXPECT_FALSE(container.has_two_rdm_spin_traced());
  EXPECT_THROW(container.total_num_particles(), std::runtime_error);
  EXPECT_THROW(container.active_num_particles(), std::runtime_error);
  auto restored = MPSContainer::from_json(container.to_json());
  EXPECT_FALSE(restored->has_active_num_particles());
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
}

TEST_F(MPSContainerTest, RejectsPhysicalBasisThatContradictsSectorLabels) {
  using SBT = SymmetryBlockedTensor<3>;
  const SymmetryAxis spin_projection(
      AxisName::Spin, {axes::beta(), axes::spin_value(0), axes::alpha()},
      false);
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(2), spin_projection}));
  const SymmetryLabel empty(
      {axes::particle_number_value(0), axes::spin_value(0)});
  const SymmetryLabel up({axes::particle_number_value(1), axes::alpha()});
  const SymmetryLabel down({axes::particle_number_value(1), axes::beta()});
  const SymmetryLabel pair(
      {axes::particle_number_value(2), axes::spin_value(0)});
  SBT::ExtentsArray extents;
  extents[0] = {{empty, 1}};
  extents[1] = {{empty, 1}, {up, 1}, {down, 1}, {pair, 1}};
  extents[2] = {{empty, 1}};
  auto tensor = std::make_shared<const MPSSite::TensorVariant>(
      SBT({symmetry, symmetry, symmetry}, extents, SBT::BlockMap{}));
  // The default basis is (0, u, d, 2).
  auto site_with = [&](std::vector<SymmetryLabel> physical_order) {
    return MPSSite(tensor, {{{empty}, std::move(physical_order), {empty}}});
  };
  EXPECT_NO_THROW(site_with({empty, up, down, pair}));
  EXPECT_THROW(site_with({empty, down, up, pair}), std::invalid_argument);
  EXPECT_THROW(site_with({pair, up, down, empty}), std::invalid_argument);
  // Dimension two defaults to binary (0, 1). Binary states are spinless, so
  // even the labels of an alpha spin orbital are rejected.
  SBT::ExtentsArray binary_extents = extents;
  binary_extents[1] = {{empty, 1}, {up, 1}};
  auto binary = std::make_shared<const MPSSite::TensorVariant>(
      SBT({symmetry, symmetry, symmetry}, binary_extents, SBT::BlockMap{}));
  EXPECT_THROW(MPSSite(binary, {{{empty}, {empty, up}, {empty}}}),
               std::invalid_argument);
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
  // Only Wavefunction dispatches MPS; state-vector loaders reject it.
  auto state_vector_envelope = original.to_json();
  state_vector_envelope["container_type"] = "state_vector";
  EXPECT_THROW(Wavefunction::from_json(state_vector_envelope),
               std::runtime_error);
  H5::Group container_group = file.openGroup("container");
  EXPECT_THROW(StateVectorContainer::from_hdf5(container_group),
               std::runtime_error);
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
  first_extents[1] = {{zero, 1}, {one, 1}};
  first_extents[2] = {{zero, 1}, {one, 2}};
  auto first = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(SBT(
          {symmetry, symmetry, symmetry}, first_extents,
          {{{zero, zero, one},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 2))}})),
      MPSSite::SectorOrders{{{zero}, {zero, one}, {zero, one}}});
  for (const std::string mismatch : {"none", "labels", "order"}) {
    SCOPED_TRACE(mismatch);
    const auto label = mismatch == "labels" ? two : zero;
    SBT::ExtentsArray extents;
    extents[0] = {{label, 1}, {one, 2}};
    extents[1] = {{zero, 1}, {one, 1}};
    extents[2] = {{two, 1}};
    const std::vector<SymmetryLabel> order =
        mismatch == "order" ? std::vector<SymmetryLabel>{one, label}
                            : std::vector<SymmetryLabel>{label, one};
    auto second = std::make_shared<const MPSSite>(
        std::make_shared<const MPSSite::TensorVariant>(SBT(
            {symmetry, symmetry, symmetry}, extents,
            {{{one, zero, two},
              std::make_shared<const Eigen::MatrixXd>(random_matrix(2, 1))}})),
        MPSSite::SectorOrders{{order, {zero, one}, {two}}});
    ASSERT_EQ(first->right_bond_dimension(), second->left_bond_dimension());
    if (mismatch == "none") {
      EXPECT_NO_THROW(MPSContainer({first, second}, orbitals));
    } else {
      EXPECT_THROW(MPSContainer({first, second}, orbitals),
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
    MPSContainer original(sites, orbitals, nullptr, std::nullopt, {}, one[0],
                          one[1], one[2], two[0], two[1], two[2], two[3]);
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
  EXPECT_THROW(MPSContainer({nullptr, sites[1]}, orbitals),
               std::invalid_argument);
  EXPECT_THROW(MPSContainer(sites, nullptr), std::invalid_argument);
  EXPECT_THROW(MPSContainer(sites, orbitals, nullptr, 2),
               std::invalid_argument);
  EXPECT_THROW(MPSContainer(sites, orbitals, nullptr, std::nullopt, {0, 0}),
               std::invalid_argument);
  const Eigen::MatrixXd wrong_bond = random_matrix(8, 1);
  EXPECT_THROW(MPSContainer({sites[0], make_site(wrong_bond)}, orbitals),
               std::invalid_argument);
  const Eigen::MatrixXd nonfinite =
      Eigen::MatrixXd::Constant(4, 1, std::numeric_limits<double>::quiet_NaN());
  EXPECT_THROW(make_site(nonfinite), std::invalid_argument);
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
  MPSContainer traced_only(sites, restricted, equal_counts, std::nullopt, {},
                           traced);
  EXPECT_TRUE(traced_only.has_one_rdm_spin_traced());
  EXPECT_FALSE(traced_only.has_one_rdm_spin_dependent());
  EXPECT_THROW(traced_only.active_one_rdm(), std::runtime_error);
  Eigen::MatrixXd aa = random_matrix(2, 2), bb = random_matrix(2, 2);
  Eigen::VectorXd aaaa = random_matrix(16, 1), aabb = random_matrix(16, 1),
                  bbbb = random_matrix(16, 1);
  MPSContainer supplied(
      sites, restricted, equal_counts, std::nullopt, {}, std::nullopt,
      ContainerTypes::MatrixVariant{aa}, ContainerTypes::MatrixVariant{bb},
      std::nullopt, ContainerTypes::VectorVariant{aaaa},
      ContainerTypes::VectorVariant{aabb}, ContainerTypes::VectorVariant{bbbb});
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

TEST_F(MPSContainerTest, ParticleCountsFollowTheOrbitals) {
  using Count = MPSContainer::ParticleCount;
  auto spin_counts = std::make_shared<const Count>(
      Count::SymmetriesArray{testing::spin_symmetry(false)},
      Count::BlockMap{
          {{axes::alpha()}, std::make_shared<const std::size_t>(1)},
          {{axes::beta()}, std::make_shared<const std::size_t>(1)}});
  EXPECT_THROW(MPSContainer(sites, testing::create_test_orbitals(2, 2), counts),
               std::invalid_argument);
  EXPECT_THROW(MPSContainer(sites, orbitals, spin_counts),
               std::invalid_argument);
  MPSContainer spin(
      sites,
      std::make_shared<ModelOrbitals>(testing::restricted_index_set(3, {1, 2}),
                                      testing::restricted_index_set(3, {0})),
      spin_counts);
  using Electrons = std::pair<std::size_t, std::size_t>;
  EXPECT_EQ(spin.get_active_num_electrons(), Electrons(1, 1));
  EXPECT_EQ(spin.get_total_num_electrons(), Electrons(2, 2));
  // Without a spin axis an inactive mode holds two particles on spin-half
  // sites and one on binary sites, so a chain must not mix the two.
  auto spin_free = [](std::vector<std::uint32_t> indices) {
    return std::make_shared<const SymmetryBlockedIndexSet>(
        std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial()),
        std::unordered_map<SymmetryLabel, std::size_t>{{SymmetryLabel{}, 3}},
        std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>>{
            {SymmetryLabel{}, std::move(indices)}});
  };
  auto core =
      std::make_shared<ModelOrbitals>(spin_free({1, 2}), spin_free({0}));
  MPSContainer spin_half(sites, core, counts);
  EXPECT_EQ(spin_half.total_num_particles()->value(SymmetryLabel{}), 4u);
  MPSContainer binary(
      {make_site(random_matrix(2, 3), 2), make_site(random_matrix(6, 1), 2)},
      core, counts);
  EXPECT_EQ(binary.total_num_particles()->value(SymmetryLabel{}), 3u);
  EXPECT_THROW(
      MPSContainer({make_site(random_matrix(2, 3), 2), sites[1]}, core, counts),
      std::invalid_argument);
}

namespace {

/// Spin-half configuration from blocked occupation bits written
/// alpha_0 ... alpha_{n-1} beta_0 ... beta_{n-1}.
Configuration from_blocked_bits(const std::string& bits) {
  const auto n = bits.size() / 2;
  std::string states;
  for (std::size_t orbital = 0; orbital < n; ++orbital) {
    const bool alpha = bits[orbital] == '1';
    const bool beta = bits[n + orbital] == '1';
    states += alpha ? (beta ? '2' : 'u') : (beta ? 'd' : '0');
  }
  return Configuration::from_spin_half_string(states);
}

/// Sites of (|2u0> + i|u02> - |udu> - i|02u>)/2 in chain order. Each bond
/// channel carries one product state, so no decomposition enters the fixture.
std::vector<MPSContainer::SitePtr> signed_reference_sites() {
  using Complex = std::complex<double>;
  const std::array<std::string, 4> states{"2u0", "u02", "udu", "02u"};
  const std::array<Complex, 4> coefficients{
      Complex(0.5, 0.0), Complex(0.0, 0.5), Complex(-0.5, 0.0),
      Complex(0.0, -0.5)};
  const std::string order = "0ud2";
  Eigen::MatrixXcd first = Eigen::MatrixXcd::Zero(4, 4);
  Eigen::MatrixXcd middle = Eigen::MatrixXcd::Zero(16, 4);
  Eigen::MatrixXcd last = Eigen::MatrixXcd::Zero(16, 1);
  for (std::size_t bond = 0; bond < states.size(); ++bond) {
    const auto b = static_cast<Eigen::Index>(bond);
    auto local = [&](std::size_t site) {
      return static_cast<Eigen::Index>(order.find(states[bond][site]));
    };
    first(local(0), b) = coefficients[bond];
    middle(b * 4 + local(1), b) = 1.0;
    last(b * 4 + local(2), 0) = 1.0;
  }
  return {make_site(first), make_site(middle), make_site(last)};
}

}  // namespace

TEST(MPSConventionTest, SignedReferenceAmplitudes) {
  using Complex = std::complex<double>;
  // Hand-enumerated rather than computed: |2u0> creates a0 b0 a1 in identity
  // order (one exchange to reach a0 a1 b0) but a2 b2 a0 for order [2,0,1]
  // (two exchanges). Bits are alpha_0 alpha_1 alpha_2 beta_0 beta_1 beta_2.
  const std::vector<
      std::pair<std::vector<std::size_t>, std::map<std::string, Complex>>>
      cases{{{0, 1, 2},
             {{"110100", {-0.5, 0.0}},
              {"101001", {0.0, 0.5}},
              {"101010", {0.5, 0.0}},
              {"011010", {0.0, 0.5}}}},
            {{2, 0, 1},
             {{"101001", {0.5, 0.0}},
              {"011010", {0.0, -0.5}},
              {"011100", {-0.5, 0.0}},
              {"110100", {0.0, 0.5}}}}};
  for (const auto& [order, expected] : cases) {
    const MPSContainer original(signed_reference_sites(),
                                std::make_shared<ModelOrbitals>(3), nullptr,
                                std::nullopt, order);
    const std::shared_ptr<const MPSContainer> restored =
        MPSContainer::from_json(original.to_json());
    for (const auto* container : {&original, restored.get()}) {
      for (unsigned mask = 0; mask < 64; ++mask) {
        std::string bits(6, '0');
        for (std::size_t i = 0; i < bits.size(); ++i) {
          if ((mask >> i) & 1u) bits[i] = '1';
        }
        const auto found = expected.find(bits);
        const Complex reference =
            found == expected.end() ? Complex{} : found->second;
        const auto actual = std::get<Complex>(
            container->get_coefficient(from_blocked_bits(bits)));
        EXPECT_NEAR(std::abs(actual - reference), 0.0, testing::wf_tolerance)
            << bits;
      }
    }
  }
}

TEST(MPSConventionTest, NormAndOverlapOfSignedReference) {
  using Complex = std::complex<double>;
  // Real bond-dimension-one chain holding one product state.
  auto product_sites = [](const std::string& states, double scale) {
    const std::string order = "0ud2";
    std::vector<MPSContainer::SitePtr> sites;
    for (const char state : states) {
      Eigen::MatrixXd site = Eigen::MatrixXd::Zero(4, 1);
      site(static_cast<Eigen::Index>(order.find(state)), 0) =
          sites.empty() ? scale : 1.0;
      sites.push_back(make_site(site));
    }
    return sites;
  };
  auto orbitals = std::make_shared<ModelOrbitals>(3);
  const std::vector<std::size_t> order{2, 0, 1};
  const MPSContainer reference(signed_reference_sites(), orbitals, nullptr,
                               std::nullopt, order);
  const MPSContainer product(product_sites("u02", 3.0), orbitals, nullptr,
                             std::nullopt, order);
  EXPECT_NEAR(reference.norm(), 1.0, testing::wf_tolerance);
  EXPECT_NEAR(product.norm(), 3.0, testing::wf_tolerance);
  // The reference holds i/2 |u02>, and the bra is conjugated.
  EXPECT_NEAR(std::abs(std::get<Complex>(reference.overlap(product)) -
                       Complex(0.0, -1.5)),
              0.0, testing::wf_tolerance);
  EXPECT_NEAR(std::abs(std::get<Complex>(product.overlap(reference)) -
                       Complex(0.0, 1.5)),
              0.0, testing::wf_tolerance);

  // Against a determinant expansion the signed amplitudes enter: for order
  // [2,0,1] they are 1/2, -i/2, -1/2 and i/2 on the determinants below.
  StateVectorContainer::DeterminantVector determinants;
  for (const auto* bits : {"101001", "011010", "011100", "110100"}) {
    determinants.push_back(from_blocked_bits(bits));
  }
  Eigen::VectorXcd coefficients(4);
  coefficients << Complex(0.5, 0.0), Complex(0.0, 0.5), Complex(0.5, 0.0),
      Complex(0.5, 0.0);
  const StateVectorContainer expansion(coefficients, determinants, orbitals);
  EXPECT_NEAR(std::abs(std::get<Complex>(reference.overlap(expansion)) -
                       Complex(-0.25, -0.25)),
              0.0, testing::wf_tolerance);

  // A different site order would need fermionic swaps.
  const MPSContainer identity(product_sites("u02", 1.0), orbitals);
  EXPECT_THROW(reference.overlap(identity), std::runtime_error);
}

TEST_F(MPSContainerTest, CoefficientsContractBlockedSitesInBasisOrder) {
  using SBT = SymmetryBlockedTensor<3>;
  auto symmetry = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::particle_number(4)}));
  const SymmetryLabel zero({axes::particle_number_value(0)});
  const SymmetryLabel one({axes::particle_number_value(1)});
  const SymmetryLabel two({axes::particle_number_value(2)});
  // Physical index order (2,u,d,0) groups the sectors as two, one, zero.
  const std::string states = "2ud0";
  std::vector<Configuration> basis;
  for (const char state : states) {
    basis.push_back(Configuration::from_spin_half_string(std::string{state}));
  }
  const std::vector<SymmetryLabel> physical_order{two, one, zero};
  SBT::ExtentsArray first_extents, second_extents;
  first_extents[0] = {{zero, 1}};
  first_extents[1] = {{two, 1}, {one, 2}, {zero, 1}};
  first_extents[2] = {{one, 2}, {two, 1}};
  second_extents[0] = first_extents[2];
  second_extents[1] = first_extents[1];
  second_extents[2] = {{two, 1}};
  auto first = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(SBT(
          {symmetry, symmetry, symmetry}, first_extents,
          {{{zero, one, one},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(2, 2))},
           {{zero, two, two},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 1))}})),
      MPSSite::SectorOrders{{{zero}, physical_order, {two, one}}}, basis);
  auto second = std::make_shared<const MPSSite>(
      std::make_shared<const MPSSite::TensorVariant>(SBT(
          {symmetry, symmetry, symmetry}, second_extents,
          {{{one, one, two},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(4, 1))},
           {{two, zero, two},
            std::make_shared<const Eigen::MatrixXd>(random_matrix(1, 1))}})),
      MPSSite::SectorOrders{{{two, one}, physical_order, {two}}}, basis);
  const MPSContainer container({first, second}, orbitals);
  const auto left = std::get<Eigen::MatrixXd>(first->to_dense());
  const auto right = std::get<Eigen::MatrixXd>(second->to_dense());
  std::size_t nonzero = 0;
  double squared_norm = 0.0;
  for (Eigen::Index p0 = 0; p0 < 4; ++p0) {
    for (Eigen::Index p1 = 0; p1 < 4; ++p1) {
      double expected = 0.0;
      for (Eigen::Index bond = 0; bond < 3; ++bond) {
        expected += left(p0, bond) * right(bond * 4 + p1, 0);
      }
      // Chain order creates a0 b0 a1 b1, so one exchange is needed exactly
      // when site 0 holds a beta electron and site 1 an alpha electron.
      const char s0 = states[static_cast<std::size_t>(p0)];
      const char s1 = states[static_cast<std::size_t>(p1)];
      if ((s0 == 'd' || s0 == '2') && (s1 == 'u' || s1 == '2')) {
        expected = -expected;
      }
      if (expected != 0.0) ++nonzero;
      squared_norm += expected * expected;
      const auto det =
          Configuration::from_spin_half_string(std::string{s0, s1});
      EXPECT_NEAR(std::get<double>(container.get_coefficient(det)), expected,
                  testing::wf_tolerance)
          << s0 << s1;
    }
  }
  EXPECT_EQ(nonzero, 5u);
  // The norm contracts the blocked sites and agrees with the amplitudes.
  EXPECT_NEAR(container.norm(), std::sqrt(squared_norm), testing::wf_tolerance);
  // Overlaps match local states through the physical bases: |ud> stored in
  // the default (0,u,d,2) order picks rows u = 1 and d = 2 of (2,u,d,0).
  Eigen::MatrixXd up = Eigen::MatrixXd::Zero(4, 1);
  Eigen::MatrixXd down = Eigen::MatrixXd::Zero(4, 1);
  up(1, 0) = 1.0;
  down(2, 0) = 1.0;
  const MPSContainer product({make_site(up), make_site(down)}, orbitals);
  double amplitude = 0.0;
  for (Eigen::Index bond = 0; bond < 3; ++bond) {
    amplitude += left(1, bond) * right(bond * 4 + 2, 0);
  }
  EXPECT_NE(amplitude, 0.0);
  EXPECT_NEAR(std::get<double>(product.overlap(container)), amplitude,
              testing::wf_tolerance);
}

TEST_F(MPSContainerTest, SpinlessCoefficientsCarryPermutationSign) {
  const Eigen::MatrixXd first = random_matrix(2, 2);
  const Eigen::MatrixXd second = random_matrix(4, 1);
  // Chain site 0 holds orbital 1, so |11> in chain order is a1^ a0^ |vac>.
  const MPSContainer container({make_site(first, 2), make_site(second, 2)},
                               orbitals, nullptr, std::nullopt, {1, 0});
  for (Eigen::Index orbital0 = 0; orbital0 < 2; ++orbital0) {
    for (Eigen::Index orbital1 = 0; orbital1 < 2; ++orbital1) {
      const Eigen::Index p0 = orbital1, p1 = orbital0;
      double expected = 0.0;
      for (Eigen::Index bond = 0; bond < 2; ++bond) {
        expected += first(p0, bond) * second(bond * 2 + p1, 0);
      }
      if (orbital0 == 1 && orbital1 == 1) expected = -expected;
      const auto det = Configuration::from_bitstring(std::to_string(orbital0) +
                                                     std::to_string(orbital1));
      EXPECT_NEAR(std::get<double>(container.get_coefficient(det)), expected,
                  testing::wf_tolerance);
    }
  }
}

TEST_F(MPSContainerTest, CoefficientRejectsMismatchedConfigurations) {
  const auto container = make_container();
  EXPECT_THROW(container->get_coefficient(Configuration::from_bitstring("10")),
               std::invalid_argument);
  // A mode outside the active sites lies outside the represented space.
  EXPECT_EQ(std::get<double>(container->get_coefficient(
                Configuration::from_spin_half_string("u0u"))),
            0.0);
}

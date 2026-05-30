// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <memory>
#include <stdexcept>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>

using namespace qdk::chemistry::data;

namespace {

using SBT2 = SymmetryBlockedTensor<2, double>;

SBT2::Labels aa() { return {SymmetryLabel({axes::alpha()}),
                            SymmetryLabel({axes::alpha()})}; }
SBT2::Labels bb() { return {SymmetryLabel({axes::beta()}),
                            SymmetryLabel({axes::beta()})}; }

std::array<std::unordered_map<SymmetryLabel, std::size_t>, 2> extents2(
    std::size_t n) {
  std::unordered_map<SymmetryLabel, std::size_t> slot;
  slot.emplace(SymmetryLabel({axes::alpha()}), n);
  slot.emplace(SymmetryLabel({axes::beta()}), n);
  return {slot, slot};
}

}  // namespace

TEST(SymmetryBlockedTensorTest, RestrictedAutoAliasesPartner) {
  auto sym = std::make_shared<const Symmetries>(
      Symmetries({axes::spin(0, /*equivalent=*/true)}));
  auto block = std::make_shared<const Eigen::MatrixXd>(
      Eigen::MatrixXd::Identity(3, 3));

  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block);

  SBT2 tensor({sym, sym}, extents2(3), blocks);

  EXPECT_TRUE(tensor.has_block(aa()));
  EXPECT_TRUE(tensor.has_block(bb()));
  // Symmetry-equivalent partner is the same storage.
  EXPECT_EQ(tensor.block_ptr(aa()).get(), tensor.block_ptr(bb()).get());
  EXPECT_EQ(tensor.num_blocks(), 2u);
}

TEST(SymmetryBlockedTensorTest, UnrestrictedKeepsDistinctBlocks) {
  auto sym = std::make_shared<const Symmetries>(
      Symmetries({axes::spin(0, /*equivalent=*/false)}));
  auto block_aa = std::make_shared<const Eigen::MatrixXd>(
      Eigen::MatrixXd::Constant(2, 2, 1.0));
  auto block_bb = std::make_shared<const Eigen::MatrixXd>(
      Eigen::MatrixXd::Constant(2, 2, 2.0));

  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block_aa);
  blocks.emplace(bb(), block_bb);

  SBT2 tensor({sym, sym}, extents2(2), blocks);

  EXPECT_NE(tensor.block_ptr(aa()).get(), tensor.block_ptr(bb()).get());
  EXPECT_EQ(tensor.num_blocks(), 2u);
}

TEST(SymmetryBlockedTensorTest, ExtentMismatchRejected) {
  auto sym = std::make_shared<const Symmetries>(
      Symmetries({axes::spin(0, true)}));
  auto block = std::make_shared<const Eigen::MatrixXd>(
      Eigen::MatrixXd::Zero(3, 2));

  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block);

  EXPECT_THROW(SBT2({sym, sym}, extents2(3), blocks),
               std::invalid_argument);
}

TEST(SymmetryBlockedTensorTest, InvalidLabelRejected) {
  auto sym = std::make_shared<const Symmetries>(
      Symmetries({axes::spin(0, true)}));
  auto block =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Zero(1, 1));

  // Label carrying a spin value (2*Ms = 3) not admitted by the axis.
  SBT2::Labels bad = {SymmetryLabel({axes::spin_value(3)}),
                      SymmetryLabel({axes::spin_value(3)})};
  std::unordered_map<SymmetryLabel, std::size_t> slot;
  slot.emplace(SymmetryLabel({axes::spin_value(3)}), 1);
  std::array<std::unordered_map<SymmetryLabel, std::size_t>, 2> ext = {slot,
                                                                       slot};
  SBT2::BlockMap blocks;
  blocks.emplace(bad, block);

  EXPECT_THROW(SBT2({sym, sym}, ext, blocks), std::invalid_argument);
}

TEST(SymmetryBlockedTensorTest, AliasMismatchRejected) {
  auto sym = std::make_shared<const Symmetries>(
      Symmetries({axes::spin(0, /*equivalent=*/true)}));
  auto block_aa =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Zero(2, 2));
  auto block_bb =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Zero(2, 2));

  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block_aa);
  blocks.emplace(bb(), block_bb);  // distinct storage, but equivalent axis

  EXPECT_THROW(SBT2({sym, sym}, extents2(2), blocks), std::invalid_argument);
}

TEST(SymmetryBlockedTensorTest, MissingBlockThrows) {
  auto sym = std::make_shared<const Symmetries>(
      Symmetries({axes::spin(0, false)}));
  auto block =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Zero(2, 2));
  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block);
  SBT2 tensor({sym, sym}, extents2(2), blocks);

  EXPECT_FALSE(tensor.has_block(bb()));
  EXPECT_THROW(tensor.block(bb()), std::invalid_argument);
}

TEST(SymmetryBlockedTensorTest, JsonRoundTripRestricted) {
  auto sym = std::make_shared<const Symmetries>(
      Symmetries({axes::spin(0, true)}));
  Eigen::MatrixXd data(2, 2);
  data << 1.0, 2.0, 3.0, 4.0;
  auto block = std::make_shared<const Eigen::MatrixXd>(data);
  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block);
  SBT2 tensor({sym, sym}, extents2(2), blocks);

  auto restored = SBT2::from_json(tensor.to_json());
  EXPECT_EQ(restored->block_ptr(aa()).get(), restored->block_ptr(bb()).get());
  EXPECT_TRUE(restored->block(aa()).isApprox(data));
}

TEST(SymmetryBlockedTensorTest, JsonRoundTripUnrestricted) {
  auto sym = std::make_shared<const Symmetries>(
      Symmetries({axes::spin(0, false)}));
  auto block_aa =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Constant(2, 2, 1.0));
  auto block_bb =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Constant(2, 2, 5.0));
  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block_aa);
  blocks.emplace(bb(), block_bb);
  SBT2 tensor({sym, sym}, extents2(2), blocks);

  auto restored = SBT2::from_json(tensor.to_json());
  EXPECT_NE(restored->block_ptr(aa()).get(), restored->block_ptr(bb()).get());
  EXPECT_EQ(restored->block(aa())(0, 0), 1.0);
  EXPECT_EQ(restored->block(bb())(0, 0), 5.0);
}

TEST(SymmetryBlockedTensorTest, ComplexRank1) {
  using SBT1c = SymmetryBlockedTensor<1, std::complex<double>>;
  auto sym = std::make_shared<const Symmetries>(
      Symmetries({axes::spin(0, false)}));
  Eigen::VectorXcd v(2);
  v << std::complex<double>(1.0, -1.0), std::complex<double>(2.0, 3.0);
  auto block = std::make_shared<const Eigen::VectorXcd>(v);
  std::unordered_map<SymmetryLabel, std::size_t> slot;
  slot.emplace(SymmetryLabel({axes::alpha()}), 2);
  SBT1c::BlockMap blocks;
  blocks.emplace(SBT1c::Labels{SymmetryLabel({axes::alpha()})}, block);
  SBT1c tensor({sym}, {slot}, blocks);

  auto restored = SBT1c::from_json(tensor.to_json());
  EXPECT_TRUE(restored->block(SBT1c::Labels{SymmetryLabel({axes::alpha()})})
                  .isApprox(v));
}

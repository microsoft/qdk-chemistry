// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>
#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <filesystem>
#include <memory>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <stdexcept>

using namespace qdk::chemistry::data;

using SBT2 = SymmetryBlockedTensor<2, double>;

static SBT2::Labels aa() {
  return {SymmetryLabel({axes::alpha()}), SymmetryLabel({axes::alpha()})};
}
static SBT2::Labels bb() {
  return {SymmetryLabel({axes::beta()}), SymmetryLabel({axes::beta()})};
}

static std::array<std::unordered_map<SymmetryLabel, std::size_t>, 2> extents2(
    std::size_t n) {
  std::unordered_map<SymmetryLabel, std::size_t> slot;
  slot.emplace(SymmetryLabel({axes::alpha()}), n);
  slot.emplace(SymmetryLabel({axes::beta()}), n);
  return {slot, slot};
}

static SBT2 make_simple_tensor() {
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, true)}));
  Eigen::MatrixXd data(2, 2);
  data << 1.0, 2.0, 3.0, 4.0;
  auto block = std::make_shared<const Eigen::MatrixXd>(data);
  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block);
  return SBT2({sym, sym}, extents2(2), blocks);
}

TEST(SymmetryBlockedTensorTest, RestrictedAutoAliasesPartner) {
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, /*equivalent=*/true)}));
  auto block =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Identity(3, 3));

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
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, /*equivalent=*/false)}));
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
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, true)}));
  auto block =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Zero(3, 2));

  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block);

  EXPECT_THROW(SBT2({sym, sym}, extents2(3), blocks), std::invalid_argument);
}

TEST(SymmetryBlockedTensorTest, InvalidLabelRejected) {
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, true)}));
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
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, /*equivalent=*/true)}));
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
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, false)}));
  auto block =
      std::make_shared<const Eigen::MatrixXd>(Eigen::MatrixXd::Zero(2, 2));
  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block);
  SBT2 tensor({sym, sym}, extents2(2), blocks);

  EXPECT_FALSE(tensor.has_block(bb()));
  EXPECT_THROW(tensor.block(bb()), std::invalid_argument);
}

TEST(SymmetryBlockedTensorTest, JsonRoundTripRestricted) {
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, true)}));
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
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, false)}));
  auto block_aa = std::make_shared<const Eigen::MatrixXd>(
      Eigen::MatrixXd::Constant(2, 2, 1.0));
  auto block_bb = std::make_shared<const Eigen::MatrixXd>(
      Eigen::MatrixXd::Constant(2, 2, 5.0));
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
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, false)}));
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

TEST(SymmetryBlockedTensorTest, Hdf5RoundTripStoresNativeDoubleBlocks) {
  const std::filesystem::path filename =
      "symmetry_blocked_tensor_native_real.h5";
  std::filesystem::remove(filename);

  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, true)}));
  Eigen::MatrixXd data(2, 2);
  data << 1.0, 2.0, 3.0, 4.0;
  auto block = std::make_shared<const Eigen::MatrixXd>(data);
  SBT2::BlockMap blocks;
  blocks.emplace(aa(), block);
  SBT2 tensor({sym, sym}, extents2(2), blocks);

  tensor.to_hdf5_file(filename.string());
  auto restored = SBT2::from_hdf5_file(filename.string());

  EXPECT_EQ(restored->block_ptr(aa()).get(), restored->block_ptr(bb()).get());
  EXPECT_TRUE(restored->block(aa()).isApprox(data));

  H5::H5File file(filename.string(), H5F_ACC_RDONLY);
  auto metadata = file.openDataSet("symmetry_blocked_tensor_metadata");
  EXPECT_EQ(metadata.getTypeClass(), H5T_STRING);

  auto block_dataset = file.openDataSet("block_0");
  EXPECT_EQ(block_dataset.getTypeClass(), H5T_FLOAT);
  EXPECT_EQ(block_dataset.getDataType().getSize(), sizeof(double));

  auto dataspace = block_dataset.getSpace();
  EXPECT_EQ(dataspace.getSimpleExtentNdims(), 2);
  hsize_t dims[2] = {0, 0};
  dataspace.getSimpleExtentDims(dims);
  EXPECT_EQ(dims[0], 2u);
  EXPECT_EQ(dims[1], 2u);

  std::filesystem::remove(filename);
}

TEST(SymmetryBlockedTensorTest,
     Hdf5RoundTripStoresComplexBlocksAsCompoundDataset) {
  using SBT1c = SymmetryBlockedTensor<1, std::complex<double>>;
  const std::filesystem::path filename =
      "symmetry_blocked_tensor_native_complex.h5";
  std::filesystem::remove(filename);

  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, false)}));
  Eigen::VectorXcd v(2);
  v << std::complex<double>(1.0, -1.0), std::complex<double>(2.0, 3.0);
  auto block = std::make_shared<const Eigen::VectorXcd>(v);
  std::unordered_map<SymmetryLabel, std::size_t> slot;
  slot.emplace(SymmetryLabel({axes::alpha()}), 2);
  SBT1c::BlockMap blocks;
  blocks.emplace(SBT1c::Labels{SymmetryLabel({axes::alpha()})}, block);
  SBT1c tensor({sym}, {slot}, blocks);

  tensor.to_hdf5_file(filename.string());
  auto restored = SBT1c::from_hdf5_file(filename.string());

  EXPECT_TRUE(restored->block(SBT1c::Labels{SymmetryLabel({axes::alpha()})})
                  .isApprox(v));

  H5::H5File file(filename.string(), H5F_ACC_RDONLY);
  auto block_dataset = file.openDataSet("block_0");
  EXPECT_EQ(block_dataset.getTypeClass(), H5T_COMPOUND);
  EXPECT_FALSE(file.nameExists("block_0_real"));
  EXPECT_FALSE(file.nameExists("block_0_imag"));

  std::filesystem::remove(filename);
}

TEST(SymmetryBlockedTensorTest, JsonRoundTripStampsSerializationVersion) {
  auto j = make_simple_tensor().to_json();
  ASSERT_TRUE(j.contains("version"));
  EXPECT_TRUE(j["version"].is_string());
  EXPECT_FALSE(j["version"].get<std::string>().empty());
  EXPECT_NO_THROW(SBT2::from_json(j));
}

TEST(SymmetryBlockedTensorTest, JsonFromJsonRejectsMissingVersion) {
  auto j = make_simple_tensor().to_json();
  j.erase("version");
  EXPECT_THROW(SBT2::from_json(j), std::runtime_error);
}

TEST(SymmetryBlockedTensorTest, JsonFromJsonRejectsMismatchedVersion) {
  auto j = make_simple_tensor().to_json();
  j["version"] = "99.0.0";
  EXPECT_THROW(SBT2::from_json(j), std::runtime_error);
}

TEST(SymmetryBlockedTensorTest, Hdf5MetadataCarriesSerializationVersion) {
  const std::filesystem::path filename = "symmetry_blocked_tensor_version.h5";
  std::filesystem::remove(filename);

  make_simple_tensor().to_hdf5_file(filename.string());

  H5::H5File file(filename.string(), H5F_ACC_RDONLY);
  auto dataset = file.openDataSet("symmetry_blocked_tensor_metadata");
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  std::string metadata_str;
  dataset.read(metadata_str, str_type);
  auto metadata = nlohmann::json::parse(metadata_str);
  ASSERT_TRUE(metadata.contains("version"));
  EXPECT_TRUE(metadata["version"].is_string());
  EXPECT_FALSE(metadata["version"].get<std::string>().empty());

  std::filesystem::remove(filename);
}

TEST(SymmetryBlockedTensorTest, Hdf5FromHdf5RejectsMissingVersion) {
  const std::filesystem::path filename =
      "symmetry_blocked_tensor_missing_version.h5";
  std::filesystem::remove(filename);

  make_simple_tensor().to_hdf5_file(filename.string());

  // Strip the version field from the metadata payload in-place.
  {
    H5::H5File file(filename.string(), H5F_ACC_RDWR);
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    auto dataset = file.openDataSet("symmetry_blocked_tensor_metadata");
    std::string metadata_str;
    dataset.read(metadata_str, str_type);
    auto metadata = nlohmann::json::parse(metadata_str);
    metadata.erase("version");
    dataset.write(metadata.dump(), str_type);
  }

  EXPECT_THROW(SBT2::from_hdf5_file(filename.string()), std::runtime_error);
  std::filesystem::remove(filename);
}

TEST(SymmetryBlockedTensorTest, Hdf5FromHdf5RejectsMismatchedVersion) {
  const std::filesystem::path filename =
      "symmetry_blocked_tensor_bad_version.h5";
  std::filesystem::remove(filename);

  make_simple_tensor().to_hdf5_file(filename.string());

  // Inject an incompatible version into the metadata payload in-place.
  {
    H5::H5File file(filename.string(), H5F_ACC_RDWR);
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    auto dataset = file.openDataSet("symmetry_blocked_tensor_metadata");
    std::string metadata_str;
    dataset.read(metadata_str, str_type);
    auto metadata = nlohmann::json::parse(metadata_str);
    metadata["version"] = "99.0.0";
    dataset.write(metadata.dump(), str_type);
  }

  EXPECT_THROW(SBT2::from_hdf5_file(filename.string()), std::runtime_error);
  std::filesystem::remove(filename);
}

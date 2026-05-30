// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>

#include <H5Cpp.h>

namespace qdk::chemistry::data {

namespace {

// Symmetry-blocked tensors are serialized to HDF5 as a single JSON-encoded
// string dataset. This keeps the on-disk representation aligned with the JSON
// path while round-tripping the full block/aliasing structure.
constexpr const char* kHdf5JsonDataset = "symmetry_blocked_tensor_json";

void write_json_string(H5::Group& group, const std::string& payload) {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSpace scalar_space(H5S_SCALAR);
  auto dataset =
      group.createDataSet(kHdf5JsonDataset, str_type, scalar_space);
  dataset.write(payload, str_type);
}

std::string read_json_string(H5::Group& group) {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  auto dataset = group.openDataSet(kHdf5JsonDataset);
  std::string payload;
  dataset.read(payload, str_type);
  return payload;
}

}  // namespace

template <std::size_t Rank, class Scalar>
void SymmetryBlockedTensor<Rank, Scalar>::to_hdf5(H5::Group& group) const {
  write_json_string(group, to_json().dump());
}

template <std::size_t Rank, class Scalar>
void SymmetryBlockedTensor<Rank, Scalar>::to_hdf5_file(
    const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

template <std::size_t Rank, class Scalar>
std::shared_ptr<SymmetryBlockedTensor<Rank, Scalar>>
SymmetryBlockedTensor<Rank, Scalar>::from_hdf5(H5::Group& group) {
  return from_json(nlohmann::json::parse(read_json_string(group)));
}

template <std::size_t Rank, class Scalar>
std::shared_ptr<SymmetryBlockedTensor<Rank, Scalar>>
SymmetryBlockedTensor<Rank, Scalar>::from_hdf5_file(
    const std::string& filename) {
  H5::H5File file(filename, H5F_ACC_RDONLY);
  return from_hdf5(file);
}

template class SymmetryBlockedTensor<1, double>;
template class SymmetryBlockedTensor<2, double>;
template class SymmetryBlockedTensor<4, double>;
template class SymmetryBlockedTensor<1, std::complex<double>>;
template class SymmetryBlockedTensor<2, std::complex<double>>;
template class SymmetryBlockedTensor<4, std::complex<double>>;

}  // namespace qdk::chemistry::data

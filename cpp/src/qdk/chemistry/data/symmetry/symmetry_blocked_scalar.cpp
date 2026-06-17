// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <qdk/chemistry/data/symmetry/symmetry_blocked_scalar.hpp>

namespace qdk::chemistry::data {

template <class Scalar>
std::shared_ptr<SymmetryBlockedScalar<Scalar>>
SymmetryBlockedScalar<Scalar>::from_json(const nlohmann::json& j) {
  if (!j.contains("version")) {
    throw std::runtime_error(
        "SymmetryBlockedScalar JSON missing required 'version' field.");
  }
  _validate_version(j.at("version").template get<std::string>());

  const std::string expected_scalar = _scalar_tag();
  if (j.contains("scalar") &&
      j.at("scalar").template get<std::string>() != expected_scalar) {
    throw std::invalid_argument(
        "SymmetryBlockedScalar JSON scalar type does not match the requested "
        "type.");
  }

  auto symmetries = Base::_symmetries_from_json(j);

  BlockMap blocks;
  for (const auto& entry : j.at("blocks")) {
    auto value =
        std::make_shared<const Scalar>(_value_from_json(entry.at("value")));
    for (const auto& key_json : entry.at("keys")) {
      std::vector<SymmetryLabel> labels;
      for (const auto& label_json : key_json) {
        labels.push_back(SymmetryLabel::from_json(label_json));
      }
      blocks.emplace(detail::make_labels<1>(labels), value);
    }
  }
  return std::make_shared<SymmetryBlockedScalar>(std::move(symmetries),
                                                 std::move(blocks));
}

template <class Scalar>
void SymmetryBlockedScalar<Scalar>::to_hdf5(H5::Group& group) const {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSpace scalar_space(H5S_SCALAR);
  auto dataset = group.createDataSet("symmetry_blocked_scalar_metadata",
                                     str_type, scalar_space);
  std::string payload = to_json().dump(2);
  dataset.write(payload, str_type);
}

template <class Scalar>
void SymmetryBlockedScalar<Scalar>::to_hdf5_file(
    const std::string& filename) const {
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

template <class Scalar>
std::shared_ptr<SymmetryBlockedScalar<Scalar>>
SymmetryBlockedScalar<Scalar>::from_hdf5(H5::Group& group) {
  if (!group.nameExists("symmetry_blocked_scalar_metadata")) {
    throw std::runtime_error(
        "SymmetryBlockedScalar HDF5 metadata dataset not found.");
  }
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  auto dataset = group.openDataSet("symmetry_blocked_scalar_metadata");
  std::string payload;
  dataset.read(payload, str_type);
  return from_json(nlohmann::json::parse(payload));
}

template <class Scalar>
std::shared_ptr<SymmetryBlockedScalar<Scalar>>
SymmetryBlockedScalar<Scalar>::from_hdf5_file(const std::string& filename) {
  H5::H5File file(filename, H5F_ACC_RDONLY);
  return from_hdf5(file);
}

template class SymmetryBlockedScalar<std::size_t>;

}  // namespace qdk::chemistry::data

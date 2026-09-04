// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>

#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

namespace detail {

void write_string_dataset(H5::Group& group, const std::string& name,
                          const std::string& payload) {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  H5::DataSpace scalar_space(H5S_SCALAR);
  auto dataset = group.createDataSet(name, str_type, scalar_space);
  dataset.write(payload, str_type);
}

std::string read_string_dataset(H5::Group& group, const std::string& name) {
  H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
  auto dataset = group.openDataSet(name);
  std::string payload;
  dataset.read(payload, str_type);
  return payload;
}

}  // namespace detail

template <std::size_t Rank, class Scalar>
std::shared_ptr<SymmetryBlockedTensor<Rank, Scalar>>
SymmetryBlockedTensor<Rank, Scalar>::from_json(const nlohmann::json& j) {
  if (!j.contains("version")) {
    throw std::runtime_error(
        "SymmetryBlockedTensor JSON missing required 'version' field.");
  }
  validate_serialization_version(SERIALIZATION_VERSION,
                                 j.at("version").template get<std::string>());

  auto symmetries = Base::_symmetries_from_json(j);
  auto extents = Base::_extents_from_json(j);

  BlockMap blocks;
  for (const auto& entry : j.at("blocks")) {
    auto blk = std::make_shared<const Tensor<Rank, Scalar>>(
        _block_from_json(entry.at("block")));
    for (const auto& key_json : entry.at("keys")) {
      std::vector<SymmetryLabel> labels;
      for (const auto& label_json : key_json) {
        labels.push_back(SymmetryLabel::from_json(label_json));
      }
      blocks.emplace(detail::make_labels<Rank>(labels), blk);
    }
  }
  return std::make_shared<SymmetryBlockedTensor>(
      std::move(symmetries), std::move(extents), std::move(blocks));
}

template <std::size_t Rank, class Scalar>
void SymmetryBlockedTensor<Rank, Scalar>::to_hdf5(H5::Group& group) const {
  nlohmann::json metadata;
  metadata["version"] = SERIALIZATION_VERSION;
  metadata["type"] = "SymmetryBlockedTensor";
  metadata["rank"] = Rank;
  metadata["scalar"] = utils::is_complex_scalar_v<Scalar> ? "complex" : "real";
  metadata["symmetries"] = this->_symmetries_to_json();
  metadata["extents"] = this->_extents_to_json();
  metadata["blocks"] = nlohmann::json::array();

  std::size_t block_index = 0;
  for (const auto& pointer_group : this->_group_by_pointer()) {
    const std::string block_name = "block_" + std::to_string(block_index++);

    nlohmann::json keys = nlohmann::json::array();
    for (const auto& key : pointer_group.keys) {
      nlohmann::json key_json = nlohmann::json::array();
      for (const auto& label : key) {
        key_json.push_back(label.to_json());
      }
      keys.push_back(std::move(key_json));
    }

    if constexpr (utils::is_complex_scalar_v<Scalar>) {
      if constexpr (Rank == 2 || Rank == 3) {
        auto variant = std::make_shared<MatrixVariant>(*pointer_group.ptr);
        save_matrix_variant_to_group(/*is_complex=*/true, variant, group,
                                     block_name);
      } else {
        auto variant = std::make_shared<VectorVariant>(*pointer_group.ptr);
        save_vector_variant_to_group(/*is_complex=*/true, variant, group,
                                     block_name);
      }
    } else {
      if constexpr (Rank == 2 || Rank == 3) {
        save_matrix_to_group(group, block_name, *pointer_group.ptr);
      } else {
        save_vector_to_group(group, block_name, *pointer_group.ptr);
      }
    }
    metadata["blocks"].push_back(
        nlohmann::json{{"keys", std::move(keys)}, {"dataset", block_name}});
  }

  detail::write_string_dataset(group, "symmetry_blocked_tensor_metadata",
                               metadata.dump());
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
  if (!group.nameExists("symmetry_blocked_tensor_metadata")) {
    throw std::runtime_error(
        "SymmetryBlockedTensor HDF5 metadata dataset not found.");
  }

  auto metadata = nlohmann::json::parse(
      detail::read_string_dataset(group, "symmetry_blocked_tensor_metadata"));
  if (!metadata.contains("version")) {
    throw std::runtime_error(
        "SymmetryBlockedTensor HDF5 metadata missing required 'version' "
        "field.");
  }
  validate_serialization_version(
      SERIALIZATION_VERSION,
      metadata.at("version").template get<std::string>());

  if (metadata.at("rank").get<std::size_t>() != Rank) {
    throw std::invalid_argument(
        "SymmetryBlockedTensor HDF5 rank does not match the requested type.");
  }

  const std::string expected_scalar =
      utils::is_complex_scalar_v<Scalar> ? "complex" : "real";
  if (metadata.at("scalar").get<std::string>() != expected_scalar) {
    throw std::invalid_argument(
        "SymmetryBlockedTensor HDF5 scalar type does not match the requested "
        "type.");
  }

  auto symmetries = Base::_symmetries_from_json(metadata);
  auto extents = Base::_extents_from_json(metadata);

  BlockMap blocks;
  std::unordered_map<std::string, BlockPtr> block_cache;
  for (const auto& entry : metadata.at("blocks")) {
    const auto dataset_name = entry.at("dataset").get<std::string>();

    BlockPtr block_ptr;
    auto cache_it = block_cache.find(dataset_name);
    if (cache_it == block_cache.end()) {
      Tensor<Rank, Scalar> block;
      if constexpr (utils::is_complex_scalar_v<Scalar>) {
        if constexpr (Rank == 2 || Rank == 3) {
          auto variant =
              load_matrix_variant_from_group(group, dataset_name, true);
          block = std::get<Eigen::MatrixXcd>(variant);
        } else {
          auto variant =
              load_vector_variant_from_group(group, dataset_name, true);
          block = std::get<Eigen::VectorXcd>(variant);
        }
      } else {
        if constexpr (Rank == 2 || Rank == 3) {
          block = load_matrix_from_group(group, dataset_name);
        } else {
          block = load_vector_from_group(group, dataset_name);
        }
      }
      block_ptr =
          std::make_shared<const Tensor<Rank, Scalar>>(std::move(block));
      block_cache.emplace(dataset_name, block_ptr);
    } else {
      block_ptr = cache_it->second;
    }

    for (const auto& key_json : entry.at("keys")) {
      std::vector<SymmetryLabel> labels;
      for (const auto& label_json : key_json) {
        labels.push_back(SymmetryLabel::from_json(label_json));
      }
      blocks.emplace(detail::make_labels<Rank>(labels), block_ptr);
    }
  }

  return std::make_shared<SymmetryBlockedTensor>(
      std::move(symmetries), std::move(extents), std::move(blocks));
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
template class SymmetryBlockedTensor<3, double>;
template class SymmetryBlockedTensor<4, double>;
template class SymmetryBlockedTensor<1, std::complex<double>>;
template class SymmetryBlockedTensor<2, std::complex<double>>;
template class SymmetryBlockedTensor<3, std::complex<double>>;
template class SymmetryBlockedTensor<4, std::complex<double>>;

}  // namespace qdk::chemistry::data

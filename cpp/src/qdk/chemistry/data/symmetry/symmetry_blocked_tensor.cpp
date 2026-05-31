// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>

#include <H5Cpp.h>

namespace qdk::chemistry::data {

namespace {

constexpr const char* kHdf5MetadataDataset = "symmetry_blocked_tensor_metadata";
constexpr const char* kLegacyJsonDataset = "symmetry_blocked_tensor_json";

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

template <class Block, class Extractor>
void write_double_dataset(H5::Group& group, const std::string& name,
                          const Block& block, Extractor&& extractor) {
  const hsize_t dims[2] = {static_cast<hsize_t>(block.rows()),
                           static_cast<hsize_t>(block.cols())};
  H5::DataSpace dataspace(2, dims);
  auto dataset =
      group.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);

  std::vector<double> buffer(static_cast<std::size_t>(block.rows() * block.cols()));
  std::size_t k = 0;
  for (Eigen::Index r = 0; r < block.rows(); ++r) {
    for (Eigen::Index c = 0; c < block.cols(); ++c) {
      buffer[k++] = extractor(block(r, c));
    }
  }
  dataset.write(buffer.data(), H5::PredType::NATIVE_DOUBLE);
}

struct DoubleDatasetBuffer {
  Eigen::Index rows = 0;
  Eigen::Index cols = 0;
  std::vector<double> values;
};

template <std::size_t Rank, class Scalar>
Tensor<Rank, Scalar> make_block_storage(Eigen::Index rows, Eigen::Index cols) {
  if constexpr (Rank == 2) {
    return Tensor<Rank, Scalar>(rows, cols);
  } else {
    if (cols != 1) {
      throw std::runtime_error(
          "SymmetryBlockedTensor vector block datasets must have one column.");
    }
    return Tensor<Rank, Scalar>(rows);
  }
}

DoubleDatasetBuffer read_double_dataset(H5::Group& group, const std::string& name) {
  auto dataset = group.openDataSet(name);
  auto dataspace = dataset.getSpace();

  const int ndims = dataspace.getSimpleExtentNdims();
  hsize_t dims[2] = {0, 1};
  if (ndims == 1) {
    dataspace.getSimpleExtentDims(dims);
  } else if (ndims == 2) {
    dataspace.getSimpleExtentDims(dims);
  } else {
    throw std::runtime_error("SymmetryBlockedTensor block dataset '" + name +
                             "' must be 1-D or 2-D.");
  }

  DoubleDatasetBuffer buffer;
  buffer.rows = static_cast<Eigen::Index>(dims[0]);
  buffer.cols = static_cast<Eigen::Index>(ndims == 1 ? 1 : dims[1]);
  buffer.values.resize(static_cast<std::size_t>(buffer.rows * buffer.cols));
  dataset.read(buffer.values.data(), H5::PredType::NATIVE_DOUBLE);
  return buffer;
}

template <std::size_t Rank, class Scalar>
Tensor<Rank, Scalar> block_from_real_buffer(const DoubleDatasetBuffer& buffer) {
  auto block = make_block_storage<Rank, Scalar>(buffer.rows, buffer.cols);
  std::size_t k = 0;
  for (Eigen::Index r = 0; r < buffer.rows; ++r) {
    for (Eigen::Index c = 0; c < buffer.cols; ++c) {
      block(r, c) = static_cast<Scalar>(buffer.values[k++]);
    }
  }
  return block;
}

template <std::size_t Rank>
Tensor<Rank, std::complex<double>> block_from_complex_buffers(
    const DoubleDatasetBuffer& real_buffer, const DoubleDatasetBuffer& imag_buffer,
    const std::string& real_name, const std::string& imag_name) {
  if (real_buffer.rows != imag_buffer.rows || real_buffer.cols != imag_buffer.cols) {
    throw std::runtime_error("SymmetryBlockedTensor complex datasets '" + real_name +
                             "' and '" + imag_name + "' have mismatched shapes.");
  }

  auto block = make_block_storage<Rank, std::complex<double>>(
      real_buffer.rows, real_buffer.cols);
  std::size_t k = 0;
  for (Eigen::Index r = 0; r < real_buffer.rows; ++r) {
    for (Eigen::Index c = 0; c < real_buffer.cols; ++c) {
      block(r, c) =
          std::complex<double>(real_buffer.values[k], imag_buffer.values[k]);
      ++k;
    }
  }
  return block;
}

}  // namespace

template <std::size_t Rank, class Scalar>
void SymmetryBlockedTensor<Rank, Scalar>::to_hdf5(H5::Group& group) const {
  nlohmann::json metadata;
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

    nlohmann::json block_metadata{{"keys", std::move(keys)}};
    if constexpr (utils::is_complex_scalar_v<Scalar>) {
      const std::string real_name = block_name + "_real";
      const std::string imag_name = block_name + "_imag";
      write_double_dataset(group, real_name, *pointer_group.ptr,
                           [](const Scalar& value) { return value.real(); });
      write_double_dataset(group, imag_name, *pointer_group.ptr,
                           [](const Scalar& value) { return value.imag(); });
      block_metadata["real_dataset"] = real_name;
      block_metadata["imag_dataset"] = imag_name;
    } else {
      write_double_dataset(group, block_name, *pointer_group.ptr,
                           [](const Scalar& value) {
                             return static_cast<double>(value);
                           });
      block_metadata["dataset"] = block_name;
    }
    metadata["blocks"].push_back(std::move(block_metadata));
  }

  write_string_dataset(group, kHdf5MetadataDataset, metadata.dump());
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
  if (!group.nameExists(kHdf5MetadataDataset)) {
    if (group.nameExists(kLegacyJsonDataset)) {
      return from_json(
          nlohmann::json::parse(read_string_dataset(group, kLegacyJsonDataset)));
    }
    throw std::runtime_error(
        "SymmetryBlockedTensor HDF5 metadata dataset not found.");
  }

  auto metadata =
      nlohmann::json::parse(read_string_dataset(group, kHdf5MetadataDataset));
  if (metadata.at("rank").get<std::size_t>() != Rank) {
    throw std::invalid_argument(
        "SymmetryBlockedTensor HDF5 rank does not match the requested type.");
  }

  const std::string expected_scalar =
      utils::is_complex_scalar_v<Scalar> ? "complex" : "real";
  if (metadata.at("scalar").get<std::string>() != expected_scalar) {
    throw std::invalid_argument(
        "SymmetryBlockedTensor HDF5 scalar type does not match the requested type.");
  }

  auto symmetries = Base::_symmetries_from_json(metadata);
  auto extents = Base::_extents_from_json(metadata);

  BlockMap blocks;
  std::unordered_map<std::string, BlockPtr> block_cache;
  for (const auto& entry : metadata.at("blocks")) {
    BlockPtr block_ptr;
    if constexpr (utils::is_complex_scalar_v<Scalar>) {
      const auto real_name = entry.at("real_dataset").get<std::string>();
      const auto imag_name = entry.at("imag_dataset").get<std::string>();
      const auto cache_key = real_name + "\n" + imag_name;

      auto cache_it = block_cache.find(cache_key);
      if (cache_it == block_cache.end()) {
        auto real_buffer = read_double_dataset(group, real_name);
        auto imag_buffer = read_double_dataset(group, imag_name);
        auto block = block_from_complex_buffers<Rank>(real_buffer, imag_buffer,
                                                      real_name, imag_name);
        block_ptr =
            std::make_shared<const Tensor<Rank, Scalar>>(std::move(block));
        block_cache.emplace(cache_key, block_ptr);
      } else {
        block_ptr = cache_it->second;
      }
    } else {
      const auto dataset_name = entry.at("dataset").get<std::string>();
      auto cache_it = block_cache.find(dataset_name);
      if (cache_it == block_cache.end()) {
        auto buffer = read_double_dataset(group, dataset_name);
        auto block = block_from_real_buffer<Rank, Scalar>(buffer);
        block_ptr =
            std::make_shared<const Tensor<Rank, Scalar>>(std::move(block));
        block_cache.emplace(dataset_name, block_ptr);
      } else {
        block_ptr = cache_it->second;
      }
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

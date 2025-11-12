// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <H5Cpp.h>

#include <vector>

// Template implementations for hdf5_serialization.hpp
// This file should only be included by hdf5_serialization.hpp

namespace qdk::chemistry {
namespace data {

template <typename T>
void save_stl_to_hdf5(H5::H5File& file, const std::string& dataset_name,
                      const std::vector<T>& data) {
  auto data_type = h5_pred_type<T>::value();

  // Create dataspace with vector dimensions
  hsize_t dims[1] = {data.size()};
  H5::DataSpace dataspace(1, dims);

  // Create dataset and write data
  H5::DataSet dataset = file.createDataSet(dataset_name, data_type, dataspace);
  if (!data.empty()) {
    dataset.write(data.data(), data_type);
  }
}

template <typename T>
std::vector<T> load_std_vector_from_hdf5(H5::H5File& file,
                                         const std::string& dataset_name) {
  H5::DataSet dataset = file.openDataSet(dataset_name);
  H5::DataSpace dataspace = dataset.getSpace();

  // Get dimensions
  hsize_t dims[1];
  if (dataspace.getSimpleExtentNdims() == 1) {
    dataspace.getSimpleExtentDims(dims, nullptr);
  } else {
    throw std::runtime_error("Expected dataspace rank of 1.");
  }

  // Prepare output vector
  std::vector<T> data(dims[0]);

  // Read data if vector is non-empty
  if (dims[0] > 0) {
    dataset.read(data.data(), h5_pred_type<T>::value());
  }

  return data;
}

template <typename T>
void save_vector_to_group(H5::Group& group, const std::string& dataset_name,
                          const std::vector<T>& vector) {
  if (!vector.empty()) {
    hsize_t dims[1] = {vector.size()};
    H5::DataSpace dataspace(1, dims);
    H5::DataSet dataset =
        group.createDataSet(dataset_name, h5_pred_type<T>::value(), dataspace);
    dataset.write(vector.data(), h5_pred_type<T>::value());
  }
}

template <typename T>
std::vector<T> load_vector_from_group(H5::Group& group,
                                      const std::string& dataset_name) {
  H5::DataSet dataset = group.openDataSet(dataset_name);
  H5::DataSpace dataspace = dataset.getSpace();
  if (dataspace.getSimpleExtentNdims() != 1) {
    throw std::runtime_error("Dataspace is not one-dimensional.");
  }
  hsize_t dims[1];
  dataspace.getSimpleExtentDims(dims);

  std::vector<T> vector(dims[0]);
  if (dims[0] > 0) {
    dataset.read(vector.data(), h5_pred_type<T>::value());
  }
  return vector;
}

}  // namespace data
}  // namespace qdk::chemistry

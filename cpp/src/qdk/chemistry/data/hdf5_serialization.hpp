// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @file hdf5_serialization.hpp
 * @brief HDF5 group-based serialization helpers
 *
 * This file contains helper functions for HDF5 group-based serialization of
 * various data types including Eigen matrices/vectors and STL containers.
 */

/**
 * @brief Template struct for mapping C++ types to HDF5 predefined types.
 */
template <typename T>
struct h5_pred_type;

#define DECLARE_H5_PRED_TYPE(type, pred_type) \
  template <>                                 \
  struct h5_pred_type<type> {                 \
    static auto value() { return pred_type; } \
  };

// Specializations for common types
DECLARE_H5_PRED_TYPE(int, H5::PredType::NATIVE_INT)
DECLARE_H5_PRED_TYPE(unsigned int, H5::PredType::NATIVE_UINT)
DECLARE_H5_PRED_TYPE(char, H5::PredType::NATIVE_CHAR)
DECLARE_H5_PRED_TYPE(float, H5::PredType::NATIVE_FLOAT)
DECLARE_H5_PRED_TYPE(double, H5::PredType::NATIVE_DOUBLE)

#undef DECLARE_H5_PRED_TYPE

// Eigen matrix/vector operations with files
void save_matrix_to_hdf5(H5::H5File& file, const std::string& dataset_name,
                         const Eigen::MatrixXd& matrix);
void save_vector_to_hdf5(H5::H5File& file, const std::string& dataset_name,
                         const Eigen::VectorXd& vector);
Eigen::MatrixXd load_matrix_from_hdf5(H5::H5File& file,
                                      const std::string& dataset_name);
Eigen::VectorXd load_vector_from_hdf5(H5::H5File& file,
                                      const std::string& dataset_name);

// STL container operations with files
template <typename T>
void save_stl_to_hdf5(H5::H5File& file, const std::string& dataset_name,
                      const std::vector<T>& data);
template <typename T>
std::vector<T> load_std_vector_from_hdf5(H5::H5File& file,
                                         const std::string& dataset_name);

// Utility functions for files
bool dataset_exists(H5::H5File& file, const std::string& dataset_name);
bool group_exists(H5::H5File& file, const std::string& group_name);

// Eigen matrix/vector operations with groups
void save_matrix_to_group(H5::Group& group, const std::string& dataset_name,
                          const Eigen::MatrixXd& matrix);
void save_vector_to_group(H5::Group& group, const std::string& dataset_name,
                          const Eigen::VectorXd& vector);
void save_vector_to_group(H5::Group& group, const std::string& dataset_name,
                          const std::vector<size_t>& vector);
Eigen::MatrixXd load_matrix_from_group(H5::Group& group,
                                       const std::string& dataset_name);
Eigen::VectorXd load_vector_from_group(H5::Group& group,
                                       const std::string& dataset_name);
std::vector<size_t> load_size_vector_from_group(
    H5::Group& group, const std::string& dataset_name);

// STL container operations with groups
template <typename T>
void save_stl_to_group(H5::Group& group, const std::string& dataset_name,
                       const std::vector<T>& data);
template <typename T>
std::vector<T> load_std_vector_from_group(H5::Group& group,
                                          const std::string& dataset_name);

// Utility functions for groups
bool dataset_exists_in_group(H5::Group& group, const std::string& dataset_name);
bool group_exists_in_group(H5::Group& group, const std::string& group_name);

template <typename T>
void save_stl_to_hdf5(H5::H5File& file, const std::string& dataset_name,
                      const std::vector<T>& data) {
  auto data_type = h5_pred_type<T>::value();
  hsize_t dims[1] = {data.size()};
  H5::DataSpace dataspace(1, dims);
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
  hsize_t dims[1];
  dataspace.getSimpleExtentDims(dims, NULL);
  std::vector<T> data(dims[0]);
  if (dims[0] > 0) {
    dataset.read(data.data(), h5_pred_type<T>::value());
  }
  return data;
}

template <typename T>
void save_stl_to_group(H5::Group& group, const std::string& dataset_name,
                       const std::vector<T>& data) {
  auto data_type = h5_pred_type<T>::value();
  hsize_t dims[1] = {data.size()};
  H5::DataSpace dataspace(1, dims);
  H5::DataSet dataset = group.createDataSet(dataset_name, data_type, dataspace);
  if (!data.empty()) {
    dataset.write(data.data(), data_type);
  }
}

template <typename T>
std::vector<T> load_std_vector_from_group(H5::Group& group,
                                          const std::string& dataset_name) {
  H5::DataSet dataset = group.openDataSet(dataset_name);
  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[1];
  dataspace.getSimpleExtentDims(dims, NULL);
  std::vector<T> data(dims[0]);
  if (dims[0] > 0) {
    dataset.read(data.data(), h5_pred_type<T>::value());
  }
  return data;
}

}  // namespace qdk::chemistry::data

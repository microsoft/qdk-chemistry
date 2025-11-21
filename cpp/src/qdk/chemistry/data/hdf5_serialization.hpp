// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <complex>
#include <string>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

using VectorVariant = std::variant<Eigen::VectorXd, Eigen::VectorXcd>;

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
DECLARE_H5_PRED_TYPE(long, H5::PredType::NATIVE_LONG)
DECLARE_H5_PRED_TYPE(unsigned long, H5::PredType::NATIVE_ULONG)
DECLARE_H5_PRED_TYPE(long long, H5::PredType::NATIVE_LLONG)
DECLARE_H5_PRED_TYPE(unsigned long long, H5::PredType::NATIVE_ULLONG)
#if !defined(__APPLE__) && (SIZE_MAX != ULONG_MAX)
DECLARE_H5_PRED_TYPE(size_t, H5::PredType::NATIVE_ULLONG)
#endif
DECLARE_H5_PRED_TYPE(char, H5::PredType::NATIVE_CHAR)
DECLARE_H5_PRED_TYPE(float, H5::PredType::NATIVE_FLOAT)
DECLARE_H5_PRED_TYPE(double, H5::PredType::NATIVE_DOUBLE)

#undef DECLARE_H5_PRED_TYPE

// Eigen matrix/vector operations with files

/**
 * @brief Save an Eigen matrix to an HDF5 file as a dataset.
 * @param file HDF5 file handle
 * @param dataset_name Name for the dataset in the HDF5 file
 * @param matrix Eigen matrix to save
 */
void save_matrix_to_hdf5(H5::H5File& file, const std::string& dataset_name,
                         const Eigen::MatrixXd& matrix);

/**
 * @brief Save an Eigen vector to an HDF5 file as a dataset.
 * @param file HDF5 file handle
 * @param dataset_name Name for the dataset in the HDF5 file
 * @param vector Eigen vector to save
 */
void save_vector_to_hdf5(H5::H5File& file, const std::string& dataset_name,
                         const Eigen::VectorXd& vector);

/**
 * @brief Load an Eigen matrix from an HDF5 file dataset.
 * @param file HDF5 file handle
 * @param dataset_name Name of the dataset in the HDF5 file
 * @return Loaded Eigen matrix
 */
Eigen::MatrixXd load_matrix_from_hdf5(H5::H5File& file,
                                      const std::string& dataset_name);

/**
 * @brief Load an Eigen vector from an HDF5 file dataset.
 * @param file HDF5 file handle
 * @param dataset_name Name of the dataset in the HDF5 file
 * @return Loaded Eigen vector
 */
Eigen::VectorXd load_vector_from_hdf5(H5::H5File& file,
                                      const std::string& dataset_name);

/**
 * @brief Load a vector variant (real or complex) from an HDF5 group.
 *
 * Loads either a real (VectorXd) or complex (VectorXcd) Eigen vector from
 * an HDF5 group dataset. For complex vectors, expects HDF5 compound type
 * with "real" and "imag" fields.
 *
 * @param grp HDF5 group handle
 * @param name Name of the dataset in the group
 * @param is_complex If true, loads as complex vector; if false, loads as real
 * vector
 * @return VectorVariant containing either Eigen::VectorXd or Eigen::VectorXcd
 * @throws std::runtime_error if dataset not found or has incorrect format
 */
VectorVariant load_vector_variant_from_group(H5::Group& grp,
                                             const std::string& name,
                                             bool is_complex = false);

// STL container operations with files

/**
 * @brief Save an STL vector to an HDF5 file as a dataset.
 * @tparam T Element type (must have corresponding h5_pred_type specialization)
 * @param file HDF5 file handle
 * @param dataset_name Name for the dataset in the HDF5 file
 * @param data STL vector to save
 */
template <typename T>
void save_stl_to_hdf5(H5::H5File& file, const std::string& dataset_name,
                      const std::vector<T>& data);

/**
 * @brief Load an STL vector from an HDF5 file dataset.
 * @tparam T Element type (must have corresponding h5_pred_type specialization)
 * @param file HDF5 file handle
 * @param dataset_name Name of the dataset in the HDF5 file
 * @return Loaded STL vector
 */
template <typename T>
std::vector<T> load_std_vector_from_hdf5(H5::H5File& file,
                                         const std::string& dataset_name);

// Utility functions for files

/**
 * @brief Check if a dataset exists in an HDF5 file.
 * @param file HDF5 file handle
 * @param dataset_name Name of the dataset to check
 * @return true if dataset exists, false otherwise
 */
bool dataset_exists(H5::H5File& file, const std::string& dataset_name);

/**
 * @brief Check if a group exists in an HDF5 file.
 * @param file HDF5 file handle
 * @param group_name Name of the group to check
 * @return true if group exists, false otherwise
 */
bool group_exists(H5::H5File& file, const std::string& group_name);

// Eigen matrix/vector operations with groups

/**
 * @brief Save an Eigen matrix to an HDF5 group as a dataset.
 * @param group HDF5 group handle
 * @param dataset_name Name for the dataset in the group
 * @param matrix Eigen matrix to save
 */
void save_matrix_to_group(H5::Group& group, const std::string& dataset_name,
                          const Eigen::MatrixXd& matrix);

/**
 * @brief Save an Eigen vector to an HDF5 group as a dataset.
 * @param group HDF5 group handle
 * @param dataset_name Name for the dataset in the group
 * @param vector Eigen vector to save
 */
void save_vector_to_group(H5::Group& group, const std::string& dataset_name,
                          const Eigen::VectorXd& vector);

/**
 * @brief Save an STL vector of size_t to an HDF5 group as a dataset.
 * @param group HDF5 group handle
 * @param dataset_name Name for the dataset in the group
 * @param vector STL vector of size_t to save
 */
void save_vector_to_group(H5::Group& group, const std::string& dataset_name,
                          const std::vector<size_t>& vector);

/**
 * @brief Load an Eigen matrix from an HDF5 group dataset.
 * @param group HDF5 group handle
 * @param dataset_name Name of the dataset in the group
 * @return Loaded Eigen matrix
 */
Eigen::MatrixXd load_matrix_from_group(H5::Group& group,
                                       const std::string& dataset_name);

/**
 * @brief Load an Eigen vector from an HDF5 group dataset.
 * @param group HDF5 group handle
 * @param dataset_name Name of the dataset in the group
 * @return Loaded Eigen vector
 */
Eigen::VectorXd load_vector_from_group(H5::Group& group,
                                       const std::string& dataset_name);

/**
 * @brief Load an STL vector of size_t from an HDF5 group dataset.
 *
 * This function is useful for loading index arrays, dimensions, or other
 * unsigned integer metadata stored in HDF5 groups.
 *
 * @param group HDF5 group handle
 * @param dataset_name Name of the dataset in the group
 * @return Loaded STL vector of size_t values
 */
std::vector<size_t> load_size_vector_from_group(
    H5::Group& group, const std::string& dataset_name);

// STL container operations with groups

/**
 * @brief Save an STL vector to an HDF5 group as a dataset.
 * @tparam T Element type (must have corresponding h5_pred_type specialization)
 * @param group HDF5 group handle
 * @param dataset_name Name for the dataset in the group
 * @param data STL vector to save
 */
template <typename T>
void save_stl_to_group(H5::Group& group, const std::string& dataset_name,
                       const std::vector<T>& data);

/**
 * @brief Load an STL vector from an HDF5 group dataset.
 * @tparam T Element type (must have corresponding h5_pred_type specialization)
 * @param group HDF5 group handle
 * @param dataset_name Name of the dataset in the group
 * @return Loaded STL vector
 */
template <typename T>
std::vector<T> load_std_vector_from_group(H5::Group& group,
                                          const std::string& dataset_name);

// Utility functions for groups

/**
 * @brief Check if a dataset exists in an HDF5 group.
 * @param group HDF5 group handle
 * @param dataset_name Name of the dataset to check
 * @return true if dataset exists in the group, false otherwise
 */
bool dataset_exists_in_group(H5::Group& group, const std::string& dataset_name);

/**
 * @brief Check if a subgroup exists in an HDF5 group.
 * @param group HDF5 group handle
 * @param group_name Name of the subgroup to check
 * @return true if subgroup exists, false otherwise
 */
bool group_exists_in_group(H5::Group& group, const std::string& group_name);

// Template implementations

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

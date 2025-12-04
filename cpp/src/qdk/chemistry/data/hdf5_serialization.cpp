// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "hdf5_serialization.hpp"

#include <complex>
#include <variant>

namespace qdk::chemistry::data {

void save_matrix_to_hdf5(H5::H5File& file, const std::string& dataset_name,
                         const Eigen::MatrixXd& matrix) {
  hsize_t dims[2] = {static_cast<hsize_t>(matrix.rows()),
                     static_cast<hsize_t>(matrix.cols())};
  H5::DataSpace dataspace(2, dims);
  H5::DataSet dataset =
      file.createDataSet(dataset_name, H5::PredType::NATIVE_DOUBLE, dataspace);
  dataset.write(matrix.data(), H5::PredType::NATIVE_DOUBLE);
}

void save_vector_to_hdf5(H5::H5File& file, const std::string& dataset_name,
                         const Eigen::VectorXd& vector) {
  hsize_t dims[1] = {static_cast<hsize_t>(vector.size())};
  H5::DataSpace dataspace(1, dims);
  H5::DataSet dataset =
      file.createDataSet(dataset_name, H5::PredType::NATIVE_DOUBLE, dataspace);
  dataset.write(vector.data(), H5::PredType::NATIVE_DOUBLE);
}

Eigen::MatrixXd load_matrix_from_hdf5(H5::H5File& file,
                                      const std::string& dataset_name) {
  H5::DataSet dataset = file.openDataSet(dataset_name);
  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[2];
  dataspace.getSimpleExtentDims(dims);
  Eigen::MatrixXd matrix(dims[0], dims[1]);
  dataset.read(matrix.data(), H5::PredType::NATIVE_DOUBLE);
  return matrix;
}

Eigen::VectorXd load_vector_from_hdf5(H5::H5File& file,
                                      const std::string& dataset_name) {
  H5::DataSet dataset = file.openDataSet(dataset_name);
  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[1];
  dataspace.getSimpleExtentDims(dims);
  Eigen::VectorXd vector(dims[0]);
  dataset.read(vector.data(), H5::PredType::NATIVE_DOUBLE);
  return vector;
}

VectorVariant load_vector_variant_from_group(H5::Group& grp,
                                             const std::string& name,
                                             bool is_complex) {
  VectorVariant vec_var;
  if (grp.nameExists(name)) {
    if (is_complex) {
      H5::DataSet dataset = grp.openDataSet(name);
      H5::DataSpace dataspace = dataset.getSpace();
      hsize_t dim = dataspace.getSimpleExtentNpoints();

      H5::DataType datatype = dataset.getDataType();
      if (datatype.getClass() != H5T_COMPOUND) {
        throw std::runtime_error(
            "Expected complex compound type in HDF5 coefficients "
            "dataset. Error reading dataset: " +
            name);
      }

      H5::CompType complex_type(sizeof(std::complex<double>));
      complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
      complex_type.insertMember("imag", sizeof(double),
                                H5::PredType::NATIVE_DOUBLE);

      Eigen::VectorXcd vec_c(dim);
      dataset.read(vec_c.data(), complex_type);
      vec_var = vec_c;
    } else {
      H5::DataSet dataset = grp.openDataSet(name);
      H5::DataSpace dataspace = dataset.getSpace();
      hsize_t dim = dataspace.getSimpleExtentNpoints();

      Eigen::VectorXd vec_r(dim);
      dataset.read(vec_r.data(), H5::PredType::NATIVE_DOUBLE);
      vec_var = vec_r;
    }
  } else {
    throw std::runtime_error("Dataset not found in HDF5 group: " + name);
  }
  return vec_var;
}

bool dataset_exists(H5::H5File& file, const std::string& dataset_name) {
  try {
    H5::DataSet dataset = file.openDataSet(dataset_name);
    return true;
  } catch (const H5::Exception&) {
    return false;
  }
}

bool group_exists(H5::H5File& file, const std::string& group_name) {
  try {
    H5::Group group = file.openGroup(group_name);
    return true;
  } catch (const H5::Exception&) {
    return false;
  }
}

void save_matrix_to_group(H5::Group& group, const std::string& dataset_name,
                          const Eigen::MatrixXd& matrix) {
  hsize_t dims[2] = {static_cast<hsize_t>(matrix.rows()),
                     static_cast<hsize_t>(matrix.cols())};
  H5::DataSpace dataspace(2, dims);
  H5::DataSet dataset =
      group.createDataSet(dataset_name, H5::PredType::NATIVE_DOUBLE, dataspace);
  dataset.write(matrix.data(), H5::PredType::NATIVE_DOUBLE);
}

void save_vector_to_group(H5::Group& group, const std::string& dataset_name,
                          const Eigen::VectorXd& vector) {
  hsize_t dims[1] = {static_cast<hsize_t>(vector.size())};
  H5::DataSpace dataspace(1, dims);
  H5::DataSet dataset =
      group.createDataSet(dataset_name, H5::PredType::NATIVE_DOUBLE, dataspace);
  dataset.write(vector.data(), H5::PredType::NATIVE_DOUBLE);
}

void save_vector_to_group(H5::Group& group, const std::string& dataset_name,
                          const std::vector<size_t>& vector) {
  if (!vector.empty()) {
    hsize_t dims[1] = {vector.size()};
    H5::DataSpace dataspace(1, dims);
    H5::DataSet dataset = group.createDataSet(
        dataset_name, H5::PredType::NATIVE_ULONG, dataspace);
    dataset.write(vector.data(), H5::PredType::NATIVE_ULONG);
  }
}

Eigen::MatrixXd load_matrix_from_group(H5::Group& group,
                                       const std::string& dataset_name) {
  H5::DataSet dataset = group.openDataSet(dataset_name);
  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[2];
  dataspace.getSimpleExtentDims(dims);
  Eigen::MatrixXd matrix(dims[0], dims[1]);
  dataset.read(matrix.data(), H5::PredType::NATIVE_DOUBLE);
  return matrix;
}

Eigen::VectorXd load_vector_from_group(H5::Group& group,
                                       const std::string& dataset_name) {
  H5::DataSet dataset = group.openDataSet(dataset_name);
  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[1];
  dataspace.getSimpleExtentDims(dims);
  Eigen::VectorXd vector(dims[0]);
  dataset.read(vector.data(), H5::PredType::NATIVE_DOUBLE);
  return vector;
}

std::vector<size_t> load_size_vector_from_group(
    H5::Group& group, const std::string& dataset_name) {
  H5::DataSet dataset = group.openDataSet(dataset_name);
  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[1];
  dataspace.getSimpleExtentDims(dims);
  std::vector<size_t> vector(dims[0]);
  dataset.read(vector.data(), H5::PredType::NATIVE_ULONG);
  return vector;
}

bool dataset_exists_in_group(H5::Group& group,
                             const std::string& dataset_name) {
  return group.nameExists(dataset_name);
}

bool group_exists_in_group(H5::Group& group, const std::string& group_name) {
  return group.nameExists(group_name);
}

void write_vector_to_hdf5(H5::Group& grp, const std::string& name,
                          const std::shared_ptr<VectorVariant>& vec,
                          bool is_complex) {
  if (vec) {
    if (is_complex) {
      const auto& data = std::get<Eigen::VectorXcd>(*vec);
      hsize_t dim = data.size();
      H5::DataSpace dataspace(1, &dim);

      H5::CompType complex_type(sizeof(std::complex<double>));
      complex_type.insertMember("real", 0, H5::PredType::NATIVE_DOUBLE);
      complex_type.insertMember("imag", sizeof(double),
                                H5::PredType::NATIVE_DOUBLE);

      H5::DataSet dataset = grp.createDataSet(name, complex_type, dataspace);
      dataset.write(data.data(), complex_type);
    } else {
      const auto& data = std::get<Eigen::VectorXd>(*vec);
      hsize_t dim = data.size();
      H5::DataSpace dataspace(1, &dim);

      H5::DataSet dataset =
          grp.createDataSet(name, H5::PredType::NATIVE_DOUBLE, dataspace);
      dataset.write(data.data(), H5::PredType::NATIVE_DOUBLE);
    }
  }
}

}  // namespace qdk::chemistry::data

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <Eigen/Dense>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>
#include <string>

namespace qdk::chemistry::data {

/**
 * @class NuclearHessian
 * @brief Nuclear second derivatives for a specific molecular structure.
 *
 * Stores the Cartesian Hessian of the total energy with respect to nuclear
 * coordinates. The matrix is ordered atom-major by x, y, z components and is
 * associated with the structure for which it was computed.
 */
class NuclearHessian : public DataClass,
                       public std::enable_shared_from_this<NuclearHessian> {
 public:
  /**
   * @brief Construct a nuclear Hessian for a structure.
   *
   * @param structure Molecular structure used to compute the Hessian.
   * @param matrix Square 3N by 3N Hessian matrix in Hartree/Bohr^2.
   * @throws std::invalid_argument If the structure is null or the matrix shape
   * does not match the structure.
   */
  NuclearHessian(std::shared_ptr<Structure> structure,
                 const Eigen::MatrixXd& matrix);

  /**
   * @brief Get the molecular structure associated with this Hessian.
   */
  const std::shared_ptr<Structure> get_structure() const;

  /**
   * @brief Get the Hessian matrix in Hartree/Bohr^2.
   */
  const Eigen::MatrixXd& get_matrix() const { return matrix_; }

  /**
   * @brief Return the serialized data type name.
   */
  std::string get_data_type_name() const override {
    return DATACLASS_TO_SNAKE_CASE(NuclearHessian);
  }

  /**
   * @brief Return a short summary of the Hessian data.
   */
  std::string get_summary() const override;

  /**
   * @brief Save the Hessian to a JSON or HDF5 file.
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Convert the Hessian to JSON.
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save the Hessian to a JSON file.
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Save the Hessian to an HDF5 group.
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save the Hessian to an HDF5 file.
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load the Hessian from a JSON or HDF5 file.
   */
  static std::shared_ptr<NuclearHessian> from_file(const std::string& filename,
                                                   const std::string& type);

  /**
   * @brief Load the Hessian from a JSON file.
   */
  static std::shared_ptr<NuclearHessian> from_json_file(
      const std::string& filename);

  /**
   * @brief Load the Hessian from JSON.
   */
  static std::shared_ptr<NuclearHessian> from_json(const nlohmann::json& j);

  /**
   * @brief Load the Hessian from an HDF5 file.
   */
  static std::shared_ptr<NuclearHessian> from_hdf5_file(
      const std::string& filename);

  /**
   * @brief Load the Hessian from an HDF5 group.
   */
  static std::shared_ptr<NuclearHessian> from_hdf5(H5::Group& group);

 private:
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  bool _is_valid() const;
  void _to_json_file(const std::string& filename) const;
  void _to_hdf5_file(const std::string& filename) const;
  static std::shared_ptr<NuclearHessian> _from_json_file(
      const std::string& filename);
  static std::shared_ptr<NuclearHessian> _from_hdf5_file(
      const std::string& filename);

  std::shared_ptr<Structure> structure_;
  Eigen::MatrixXd matrix_;
};

static_assert(DataClassCompliant<NuclearHessian>,
              "NuclearHessian must implement the DataClass interface");

}  // namespace qdk::chemistry::data

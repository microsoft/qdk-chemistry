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
 * @class NuclearGradients
 * @brief Nuclear energy gradients for a specific molecular structure.
 *
 * Stores the Cartesian derivative of the total energy with respect to nuclear
 * coordinates. Values are stored atom-major as x, y, z components for each
 * atom and are associated with the structure for which they were computed.
 */
class NuclearGradients : public DataClass,
                         public std::enable_shared_from_this<NuclearGradients> {
 public:
  /**
   * @brief Construct nuclear gradients for a structure.
   *
   * @param structure Molecular structure used to compute the gradients.
   * @param gradient_values Atom-major gradient vector with length 3 * number of
   * atoms.
   * @throws std::invalid_argument If the structure is null or the vector size
   * does not match the structure.
   */
  NuclearGradients(std::shared_ptr<Structure> structure,
                   const Eigen::VectorXd& gradient_values);

  /**
   * @brief Get the molecular structure associated with these gradients.
   */
  const std::shared_ptr<Structure> get_structure() const;

  /**
   * @brief Get the atom-major gradient vector in Hartree/Bohr.
   */
  const Eigen::VectorXd& get_values() const { return values_; }

  /**
   * @brief Return gradients as a num_atoms by 3 matrix.
   */
  Eigen::MatrixXd as_matrix() const;

  /**
   * @brief Return the serialized data type name.
   */
  std::string get_data_type_name() const override {
    return DATACLASS_TO_SNAKE_CASE(NuclearGradients);
  }

  /**
   * @brief Return a short summary of the gradient data.
   */
  std::string get_summary() const override;

  /**
   * @brief Save gradients to a JSON or HDF5 file.
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Convert gradients to JSON.
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save gradients to a JSON file.
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Save gradients to an HDF5 group.
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save gradients to an HDF5 file.
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load gradients from a JSON or HDF5 file.
   */
  static std::shared_ptr<NuclearGradients> from_file(
      const std::string& filename, const std::string& type);

  /**
   * @brief Load gradients from a JSON file.
   */
  static std::shared_ptr<NuclearGradients> from_json_file(
      const std::string& filename);

  /**
   * @brief Load gradients from JSON.
   */
  static std::shared_ptr<NuclearGradients> from_json(const nlohmann::json& j);

  /**
   * @brief Load gradients from an HDF5 file.
   */
  static std::shared_ptr<NuclearGradients> from_hdf5_file(
      const std::string& filename);

  /**
   * @brief Load gradients from an HDF5 group.
   */
  static std::shared_ptr<NuclearGradients> from_hdf5(H5::Group& group);

 private:
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  bool _is_valid() const;
  void _to_json_file(const std::string& filename) const;
  void _to_hdf5_file(const std::string& filename) const;
  static std::shared_ptr<NuclearGradients> _from_json_file(
      const std::string& filename);
  static std::shared_ptr<NuclearGradients> _from_hdf5_file(
      const std::string& filename);

  std::shared_ptr<Structure> structure_;
  Eigen::VectorXd values_;
};

static_assert(DataClassCompliant<NuclearGradients>,
              "NuclearGradients must implement the DataClass interface");

}  // namespace qdk::chemistry::data

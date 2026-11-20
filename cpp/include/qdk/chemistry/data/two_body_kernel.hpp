// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <string>

namespace qdk::chemistry::data {

/**
 * @class TwoBodyKernel
 * @brief Interaction kernel g(r_12) of a two-electron operator.
 *
 * The integrals of a kernel are
 * (mn|g|ls) = int phi_m(1) phi_n(1) g(r_12) phi_l(2) phi_s(2). Kernels are
 * immutable and serialized with their type and parameters.
 */
class TwoBodyKernel : public DataClass,
                      public std::enable_shared_from_this<TwoBodyKernel> {
 public:
  /**
   * @brief Virtual destructor for proper inheritance
   */
  virtual ~TwoBodyKernel() = default;

  /**
   * @brief Return the static serialized data type name.
   * @return "two_body_kernel"
   */
  static std::string data_type_name() {
    return DATACLASS_TO_SNAKE_CASE(TwoBodyKernel);
  }

  /**
   * @brief Return the serialized data type name for this instance.
   * @return "two_body_kernel"
   */
  std::string get_data_type_name() const override { return data_type_name(); }

  /**
   * @brief Save the kernel to a JSON or HDF5 file.
   * @param filename Path of the file
   * @param type "json" or "hdf5"
   * @throws std::invalid_argument If type is not supported
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Save the kernel to a JSON file.
   * @param filename Path of the file, with suffix .two_body_kernel.json
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Save the kernel to an HDF5 file.
   * @param filename Path of the file, with suffix .two_body_kernel.h5
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load a kernel from a JSON or HDF5 file.
   * @param filename Path of the file
   * @param type "json" or "hdf5"
   * @return The kernel, of the serialized type
   * @throws std::invalid_argument If type is not supported
   */
  static std::shared_ptr<TwoBodyKernel> from_file(const std::string& filename,
                                                  const std::string& type);

  /**
   * @brief Load a kernel from a JSON file.
   * @param filename Path of the file, with suffix .two_body_kernel.json
   * @return The kernel, of the serialized type
   */
  static std::shared_ptr<TwoBodyKernel> from_json_file(
      const std::string& filename);

  /**
   * @brief Load a kernel from JSON.
   * @param j JSON with the type and parameters of the kernel
   * @return The kernel, of the serialized type
   * @throws std::runtime_error If the type is not a known kernel
   */
  static std::shared_ptr<TwoBodyKernel> from_json(const nlohmann::json& j);

  /**
   * @brief Load a kernel from an HDF5 file.
   * @param filename Path of the file, with suffix .two_body_kernel.h5
   * @return The kernel, of the serialized type
   */
  static std::shared_ptr<TwoBodyKernel> from_hdf5_file(
      const std::string& filename);

  /**
   * @brief Load a kernel from an HDF5 group.
   * @param group Group with the type and parameters of the kernel
   * @return The kernel, of the serialized type
   * @throws std::runtime_error If the type is not a known kernel
   */
  static std::shared_ptr<TwoBodyKernel> from_hdf5(H5::Group& group);

 protected:
  /// Serialization format version of the kernels
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

/**
 * @class CoulombKernel
 * @brief The Coulomb kernel g(r) = 1 / r.
 */
class CoulombKernel final : public TwoBodyKernel {
 public:
  /**
   * @brief Construct the Coulomb kernel
   */
  CoulombKernel() = default;

  /**
   * @brief Return a short summary of the kernel.
   * @return "CoulombKernel(1/r)"
   */
  std::string get_summary() const override;

  /**
   * @brief Convert the kernel to JSON.
   * @return JSON with the type of the kernel
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save the kernel to an HDF5 group.
   * @param group Group receiving the type of the kernel
   */
  void to_hdf5(H5::Group& group) const override;

 private:
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;
};

/**
 * @class ErfCoulombKernel
 * @brief The long-range Coulomb kernel g(r) = erf(omega r) / r.
 */
class ErfCoulombKernel final : public TwoBodyKernel {
 public:
  /**
   * @brief Construct the kernel of a range-separation parameter
   * @param omega Range-separation parameter in inverse bohr
   * @throws std::invalid_argument If omega is not finite and positive
   */
  explicit ErfCoulombKernel(double omega);

  /**
   * @brief Access the range-separation parameter
   * @return omega in inverse bohr
   */
  double get_omega() const { return _omega; }

  /**
   * @brief Return a short summary of the kernel.
   * @return "ErfCoulombKernel(omega=...)"
   */
  std::string get_summary() const override;

  /**
   * @brief Convert the kernel to JSON.
   * @return JSON with the type and omega of the kernel
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save the kernel to an HDF5 group.
   * @param group Group receiving the type and omega of the kernel
   */
  void to_hdf5(H5::Group& group) const override;

 private:
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  /// Range-separation parameter in inverse bohr
  double _omega;
};

/**
 * @class ErfcCoulombKernel
 * @brief The short-range Coulomb kernel g(r) = erfc(omega r) / r.
 */
class ErfcCoulombKernel final : public TwoBodyKernel {
 public:
  /**
   * @brief Construct the kernel of a range-separation parameter
   * @param omega Range-separation parameter in inverse bohr
   * @throws std::invalid_argument If omega is not finite and positive
   */
  explicit ErfcCoulombKernel(double omega);

  /**
   * @brief Access the range-separation parameter
   * @return omega in inverse bohr
   */
  double get_omega() const { return _omega; }

  /**
   * @brief Return a short summary of the kernel.
   * @return "ErfcCoulombKernel(omega=...)"
   */
  std::string get_summary() const override;

  /**
   * @brief Convert the kernel to JSON.
   * @return JSON with the type and omega of the kernel
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save the kernel to an HDF5 group.
   * @param group Group receiving the type and omega of the kernel
   */
  void to_hdf5(H5::Group& group) const override;

 private:
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  /// Range-separation parameter in inverse bohr
  double _omega;
};

static_assert(DataClassCompliant<TwoBodyKernel>,
              "TwoBodyKernel must implement the DataClass interface");

}  // namespace qdk::chemistry::data

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_container.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

class HamiltonianContainer;

/**
 * @class Hamiltonian
 * @brief Interfaces a molecular Hamiltonian in the molecular orbital
 * basis by wrapping an implementation from @ref HamiltonianContainer.
 *
 * This class provides an interface to molecular Hamiltonian data for quantum
 * chemistry calculations, specifically designed for active space methods. It
 * interfaces with a HamiltonianContainer that stores:
 * - One-electron integrals (kinetic + nuclear attraction) in MO representation
 * - Two-electron integrals (electron-electron repulsion) in MO representation
 * - Molecular orbital information for the active space
 * - Core energy contributions from inactive orbitals and nuclear repulsion
 */
class Hamiltonian : public DataClass,
                    public std::enable_shared_from_this<Hamiltonian> {
 public:
  /**
   * @brief Constructor for Hamiltonian with a HamiltonianContainer
   * @param container Unique pointer to HamiltonianContainer holding the data
   */
  Hamiltonian(std::unique_ptr<HamiltonianContainer> container);

  /**
   * @brief Copy constructor
   */
  Hamiltonian(const Hamiltonian& other);

  /**
   * @brief Move constructor
   */
  Hamiltonian(Hamiltonian&& other) noexcept = default;

  /**
   * @brief Copy assignment operator
   */
  Hamiltonian& operator=(const Hamiltonian& other);

  /**
   * @brief Move assignment operator
   */
  Hamiltonian& operator=(Hamiltonian&& other) noexcept = default;

  /**
   * @brief Destructor
   */
  virtual ~Hamiltonian() = default;

  /**
   * @brief Get one-electron integrals in MO basis
   * @return Reference to one-electron integrals matrix
   * @throws std::runtime_error if integrals are not set
   */
  const Eigen::MatrixXd& get_one_body_integrals() const;

  /**
   * @brief Check if one-body integrals are available
   * @return True if one-body integrals are set
   */
  bool has_one_body_integrals() const;

  /**
   * @brief Get alpha one-electron integrals in MO basis
   * @return Reference to alpha one-electron integrals matrix
   * @throws std::runtime_error if integrals are not set
   */
  const Eigen::MatrixXd& get_one_body_integrals_alpha() const;

  /**
   * @brief Get beta one-electron integrals in MO basis
   * @return Reference to beta one-electron integrals matrix
   * @throws std::runtime_error if integrals are not set
   */
  const Eigen::MatrixXd& get_one_body_integrals_beta() const;

  /**
   * @brief Get two-electron integrals in MO basis
   * @return Reference to two-electron integrals vector
   * @throws std::runtime_error if integrals are not set
   */
  const Eigen::VectorXd& get_two_body_integrals() const;

  /**
   * @brief Get specific two-electron integral element
   * @param i First orbital index
   * @param j Second orbital index
   * @param k Third orbital index
   * @param l Fourth orbital index
   * @param channel Spin channel to query (aaaa, aabb, or bbbb), defaults to
   * aaaa
   * @return Two-electron integral <ij|kl>
   * @throws std::out_of_range if indices are invalid
   */
  double get_two_body_element(unsigned i, unsigned j, unsigned k, unsigned l,
                              SpinChannel channel = SpinChannel::aaaa) const;

  /**
   * @brief Check if two-body integrals are available
   * @return True if two-body integrals are set
   */
  bool has_two_body_integrals() const;

  /**
   * @brief Get alpha-alpha two-electron integrals in MO basis
   * @return Reference to alpha-alpha two-electron integrals vector
   * @throws std::runtime_error if integrals are not set
   */
  const Eigen::VectorXd& get_two_body_integrals_aaaa() const;

  /**
   * @brief Get alpha-beta two-electron integrals in MO basis
   * @return Reference to alpha-beta two-electron integrals vector
   * @throws std::runtime_error if integrals are not set
   */
  const Eigen::VectorXd& get_two_body_integrals_aabb() const;

  /**
   * @brief Get beta-beta two-electron integrals in MO basis
   * @return Reference to beta-beta two-electron integrals vector
   * @throws std::runtime_error if integrals are not set
   */
  const Eigen::VectorXd& get_two_body_integrals_bbbb() const;

  /**
   * @brief Get inactive Fock matrix for the selected active space
   * @return Reference to the inactive Fock matrix
   * @throws std::runtime_error if inactive Fock matrix is not set
   */
  std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
  get_inactive_fock_matrix() const;

  /**
   * @brief Check if inactive Fock matrix is available
   * @return True if inactive Fock matrix is set
   */
  bool has_inactive_fock_matrix() const;

  /**
   * @brief Get alpha inactive Fock matrix for the selected active space
   * @return Reference to the alpha inactive Fock matrix
   * @throws std::runtime_error if inactive Fock matrix is not set
   */
  const Eigen::MatrixXd& get_inactive_fock_matrix_alpha() const;

  /**
   * @brief Get beta inactive Fock matrix for the selected active space
   * @return Reference to the beta inactive Fock matrix
   * @throws std::runtime_error if inactive Fock matrix is not set
   */
  const Eigen::MatrixXd& get_inactive_fock_matrix_beta() const;

  /**
   * @brief Get molecular orbital data
   * @return Reference to the orbitals object
   * @throws std::runtime_error if orbitals are not set
   */
  const std::shared_ptr<Orbitals> get_orbitals() const;

  /**
   * @brief Check if orbital data is available
   * @return True if orbitals are set
   */
  bool has_orbitals() const;

  /**
   * @brief Get core energy
   * @return Core energy in atomic units
   */
  double get_core_energy() const;

  /**
   * @brief Get the type of Hamiltonian (Hermitian or NonHermitian)
   * @return HamiltonianType enum value
   */
  HamiltonianType get_type() const;

  /**
   * @brief Get the type of the underlying container
   * @return String identifying the container type (e.g., "canonical_4_center",
   * "desnity_fitted", "DoubleFactorizedTHC")
   */
  virtual std::string get_container_type() const;

  /**
   * @brief Get typed reference to the underlying container
   * @tparam T Container type to cast to
   * @return Reference to container as type T
   * @throws std::bad_cast if container is not of type T
   */
  template <typename T>
  const T& get_container() const {
    const T* ptr = dynamic_cast<const T*>(_container.get());
    if (!ptr) {
      throw std::bad_cast();
    }
    return *ptr;
  }

  /**
   * @brief Check if container is of specific type
   * @tparam T Container type to check
   * @return True if container is of type T
   */
  template <typename T>
  bool has_container_type() const {
    return dynamic_cast<const T*>(_container.get()) != nullptr;
  }

  /**
   * @brief Check if the Hamiltonian is Hermitian
   * @return True if the Hamiltonian type is Hermitian
   */
  bool is_hermitian() const;

  /**
   * @brief Check if the Hamiltonian is restricted
   * @return True if alpha and beta integrals are identical
   */
  bool is_restricted() const;

  /**
   * @brief Check if the Hamiltonian is unrestricted
   * @return True if alpha and beta integrals are different
   */
  bool is_unrestricted() const;

  /**
   * @brief Get summary string of Hamiltonian information
   * @return String describing the Hamiltonian
   */

  std::string get_summary() const override;

  /**
   * @brief Generic file I/O - save to file based on type parameter
   * @param filename Path to file to create/overwrite
   * @param type File format type ("json" or "hdf5")
   * @throws std::runtime_error if unsupported type or I/O error occurs
   */

  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Convert Hamiltonian to JSON
   * @return JSON object containing Hamiltonian data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save Hamiltonian to JSON file (with validation)
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Serialize Hamiltonian data to HDF5 group
   * @param group HDF5 group to write data to
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save Hamiltonian to HDF5 file (with validation)
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Generic file I/O - load from file based on type parameter
   * @param filename Path to file to read
   * @param type File format type ("json" or "hdf5")
   * @return Shared pointer to const Hamiltonian loaded from file
   * @throws std::runtime_error if file doesn't exist, unsupported type, or I/O
   * error occurs
   */
  static std::shared_ptr<Hamiltonian> from_file(const std::string& filename,
                                                const std::string& type);

  /**
   * @brief Load Hamiltonian from HDF5 file (with validation)
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to const Hamiltonian loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Hamiltonian> from_hdf5_file(
      const std::string& filename);

  /**
   * @brief Deserialize Hamiltonian data from HDF5 group
   * @param group HDF5 group to read data from
   * @return Shared pointer to const Hamiltonian loaded from group
   * @throws std::runtime_error if I/O error occurs
   */
  static std::shared_ptr<Hamiltonian> from_hdf5(H5::Group& group);

  /**
   * @brief Load Hamiltonian from JSON file (with validation)
   * @param filename Path to JSON file to read
   * @return Shared pointer to const Hamiltonian loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Hamiltonian> from_json_file(
      const std::string& filename);

  /**
   * @brief Load Hamiltonian from JSON
   * @param j JSON object containing Hamiltonian data
   * @return Shared pointer to const Hamiltonian loaded from JSON
   * @throws std::runtime_error if JSON is malformed
   */
  static std::shared_ptr<Hamiltonian> from_json(const nlohmann::json& j);

  /**
   * @brief Save Hamiltonian to an FCIDUMP file
   * @param filename Path to FCIDUMP file to create/overwrite
   * @param nalpha Number of alpha electrons
   * @param nbeta Number of beta electrons
   * @throws std::runtime_error if I/O error occurs
   */
  void to_fcidump_file(const std::string& filename, size_t nalpha,
                       size_t nbeta) const;

 private:
  /// Container holding the wavefunction implementation
  std::unique_ptr<const HamiltonianContainer> _container;

  //   /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /**
   * @brief Save to JSON file without filename validation (internal use)
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_json_file(const std::string& filename) const;

  /**
   * @brief Load from JSON file without filename validation (internal use)
   * @param filename Path to JSON file to read
   * @return Shared pointer to const Hamiltonian loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Hamiltonian> _from_json_file(
      const std::string& filename);

  /**
   * @brief Save to HDF5 file without filename validation (internal use)
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_hdf5_file(const std::string& filename) const;

  /**
   * @brief Load from HDF5 file without filename validation (internal use)
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to const Hamiltonian loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Hamiltonian> _from_hdf5_file(
      const std::string& filename);

  /**
   * @brief Save FCIDUMP file without filename validation (internal use)
   * @param filename Path to FCIDUMP file to create/overwrite
   * @param nalpha Number of alpha electrons
   * @param nbeta Number of beta electrons
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_fcidump_file(const std::string& filename, size_t nalpha,
                        size_t nbeta) const;
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(
    DataClassCompliant<Hamiltonian>,
    "Hamiltonian must derive from DataClass and implement all required "
    "deserialization methods");

}  // namespace qdk::chemistry::data

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/data/basis_set.hpp>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @class Orbitals
 * @brief Represents molecular orbitals with coefficients and energies.
 *
 * This class stores molecular orbital data including:
 * - Orbital coefficients (alpha/beta spin channels)
 * - Orbital energies (alpha/beta spin channels)
 * - Atomic orbital (AO) overlap matrix
 * - Basis set information
 *
 * Supports both restricted (RHF/RKS) and unrestricted (UHF/UKS) calculations.
 *
 * This class is immutable after construction - all data must be provided
 * during construction and cannot be modified afterward.
 */
class Orbitals : public DataClass,
                 public std::enable_shared_from_this<Orbitals> {
 public:
  typedef std::tuple<std::vector<size_t>, std::vector<size_t>>
      RestrictedCASIndices;
  typedef std::tuple<std::vector<size_t>, std::vector<size_t>,
                     std::vector<size_t>, std::vector<size_t>>
      UnrestrictedCASIndices;
  /**
   * @brief Constructor for restricted orbitals with shared pointer to basis set
   * @param coefficients The molecular orbital coefficients matrix
   * @param energies The orbital energies (optional)
   * @param ao_overlap The atomic orbital overlap matrix (optional)
   * @param basis_set The basis set as shared pointer
   * @param indices Orbital indices (shared for alpha/beta)
   * @todo TODO (NAB):  need to clarify the purpose/scope of these indices
   * 41469
   */
  Orbitals(const Eigen::MatrixXd& coefficients,
           const std::optional<Eigen::VectorXd>& energies,
           const std::optional<Eigen::MatrixXd>& ao_overlap,
           std::shared_ptr<BasisSet> basis_set,
           const std::optional<RestrictedCASIndices>& indices = std::nullopt);

  /**
   * @brief Constructor for unrestricted orbitals with shared pointer to basis
   * set
   * @param coefficients_alpha The alpha molecular orbital coefficients matrix
   * @param coefficients_beta The beta molecular orbital coefficients matrix
   * @param energies_alpha The alpha orbital energies (optional)
   * @param energies_beta The beta orbital energies (optional)
   * @param ao_overlap The atomic orbital overlap matrix (optional)
   * @param basis_set The basis set as shared pointer
   * @param indices Orbital indices (shared for alpha/beta)
   * @todo TODO (NAB):  need to clarify the purpose/scope of these indices
   * 41469
   */
  Orbitals(const Eigen::MatrixXd& coefficients_alpha,
           const Eigen::MatrixXd& coefficients_beta,
           const std::optional<Eigen::VectorXd>& energies_alpha,
           const std::optional<Eigen::VectorXd>& energies_beta,
           const std::optional<Eigen::MatrixXd>& ao_overlap,
           std::shared_ptr<BasisSet> basis_set,
           const std::optional<UnrestrictedCASIndices>& indices = std::nullopt);

  /**
   * @brief Destructor
   */
  virtual ~Orbitals();

  /**
   * @brief Copy constructor
   * @param other The orbitals object to copy from
   */
  Orbitals(const Orbitals& other);

  /**
   * @brief Copy assignment operator
   * @param other The orbitals object to copy from
   * @return Reference to this object
   */
  Orbitals& operator=(const Orbitals& other);

  /**
   * @brief Get orbital coefficients
   * @return Pair of references to (alpha, beta) coefficient matrices
   * @throws std::runtime_error if coefficients are not set
   */
  virtual std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&>
  get_coefficients() const;

  /**
   * @brief Get orbital energies
   * @return Pair of references to (alpha, beta) energy vectors
   * @throws std::runtime_error if energies are not set
   */
  virtual std::pair<const Eigen::VectorXd&, const Eigen::VectorXd&>
  get_energies() const;

  /**
   * @brief Check if energies are set
   * @return True if energies are set
   */
  bool has_energies() const;

  /**
   * @brief Get active space information
   * @return Pair of (alpha, beta) active space indices
   */
  virtual std::pair<const std::vector<size_t>&, const std::vector<size_t>&>
  get_active_space_indices() const;

  /**
   * @brief Get inactive space information
   * @return Pair of (alpha, beta) inactive space indices
   */
  std::pair<const std::vector<size_t>&, const std::vector<size_t>&>
  get_inactive_space_indices() const;

  /**
   * @brief Get virtual space information (orbitals not in active or inactive)
   * @return Pair of (alpha, beta) virtual space indices
   */
  std::pair<std::vector<size_t>, std::vector<size_t>>
  get_virtual_space_indices() const;

  // === Molecular orbital overlap ===

  /**
   * @brief Get all molecular orbital (MO) overlap matrices
   * @return Tuple of (alpha-alpha, alpha-beta, beta-beta) MO overlap matrices
   * @throws std::runtime_error if atomic orbital (AO) overlap matrix or
   * coefficients are not set
   *
   * Computes the overlap matrices between molecular orbitals by transforming
   * the atomic orbital overlap matrix to the molecular orbital basis:
   * - S_MO^αα = C_α^T * S_AO * C_α (alpha-alpha overlap)
   * - S_MO^αβ = C_α^T * S_AO * C_β (alpha-beta overlap)
   * - S_MO^ββ = C_β^T * S_AO * C_β (beta-beta overlap)
   *
   * For restricted calculations, alpha-alpha = beta-beta and alpha-beta =
   * alpha-alpha. For orthonormal MOs, alpha-alpha and beta-beta should be
   * identity matrices.
   */
  virtual std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd>
  get_mo_overlap() const;

  /**
   * @brief Get alpha-alpha molecular orbital (MO) overlap matrix
   * @return MO overlap matrix S_MO^αα = C_α^T * S_AO * C_α
   * @throws std::runtime_error if AO overlap matrix or alpha coefficients are
   * not set
   *
   * Computes the overlap matrix between alpha molecular orbitals by
   * transforming the atomic orbital overlap matrix using the alpha MO
   * coefficients. For orthonormal alpha MOs, this should be the identity
   * matrix.
   */
  virtual Eigen::MatrixXd get_mo_overlap_alpha_alpha() const;

  /**
   * @brief Get alpha-beta molecular orbital (MO) overlap matrix
   * @return MO overlap matrix S_MO^αβ = C_α^T * S_AO * C_β
   * @throws std::runtime_error if AO overlap matrix or coefficients are not set
   *
   * Computes the overlap matrix between alpha and beta molecular orbitals.
   * For restricted calculations, this equals the alpha-alpha overlap matrix.
   */
  virtual Eigen::MatrixXd get_mo_overlap_alpha_beta() const;

  /**
   * @brief Get beta-beta molecular orbital (MO) overlap matrix
   * @return MO overlap matrix S_MO^ββ = C_β^T * S_AO * C_β
   * @throws std::runtime_error if AO overlap matrix or beta coefficients are
   * not set
   *
   * Computes the overlap matrix between beta molecular orbitals by transforming
   * the atomic orbital overlap matrix using the beta MO coefficients.
   * For orthonormal beta MOs, this should be the identity matrix.
   * For restricted calculations, this equals the alpha-alpha overlap matrix.
   */
  virtual Eigen::MatrixXd get_mo_overlap_beta_beta() const;

  /**
   * @brief Get basis set information
   * @return Reference to the basis set object
   * @throws std::runtime_error if basis set is not set
   */
  virtual std::shared_ptr<BasisSet> get_basis_set() const;

  /**
   * @brief Check if basis set information is available
   * @return True if basis set is set
   */
  bool has_basis_set() const;

  /**
   * @brief Get number of molecular orbitals
   * @return Number of molecular orbitals (columns in coefficient matrix)
   */
  virtual size_t get_num_molecular_orbitals() const;

  /**
   * @brief Get number of atomic orbitals
   * @return Number of AOs (rows in coefficient matrix)
   */
  virtual size_t get_num_atomic_orbitals() const;

  /**
   * @brief Get all molecular orbital indices as a vector
   * @return Vector containing indices [0, 1, 2, ..., num_molecular_orbitals-1]
   */
  virtual std::vector<size_t> get_all_mo_indices() const;

  /**
   * @brief Check if calculation is restricted (RHF/RKS)
   * @return True if alpha and beta coefficients are identical
   */
  virtual bool is_restricted() const;

  /**
   * @brief Check if calculation is unrestricted (UHF/UKS)
   * @return True if alpha and beta coefficients are different
   */
  virtual bool is_unrestricted() const;

  /**
   * @brief Check if calculation has an active space
   * @return True if active space is set
   */
  bool has_active_space() const;

  /**
   * @brief Check if calculation has an inactive space
   * @return True if inactive space is set
   */
  bool has_inactive_space() const;

  /**
   * @brief Get alpha orbital coefficients
   * @return Reference to alpha coefficient matrix
   */
  virtual const Eigen::MatrixXd& get_coefficients_alpha() const;

  /**
   * @brief Get beta orbital coefficients
   * @return Reference to beta coefficient matrix
   */
  virtual const Eigen::MatrixXd& get_coefficients_beta() const;

  /**
   * @brief Get alpha orbital energies
   * @return Reference to alpha energy vector
   */
  virtual const Eigen::VectorXd& get_energies_alpha() const;

  /**
   * @brief Get beta orbital energies
   * @return Reference to beta energy vector
   */
  virtual const Eigen::VectorXd& get_energies_beta() const;

  /**
   * @brief Get overlap matrix
   * @return Reference to overlap matrix
   */
  virtual const Eigen::MatrixXd& get_overlap_matrix() const;

  /**
   * @brief Check if instance has an overlap matrix
   * @return True if overlap matrix is set
   */
  bool has_overlap_matrix() const;

  // === Density matrix calculations ===

  /**
   * @brief Calculate AO density matrix from occupation vectors
   * @param occupations_alpha Alpha spin occupation vector (size must match
   * number of MOs)
   * @param occupations_beta Beta spin occupation vector (size must match number
   * of MOs)
   * @return Pair of (alpha, beta) AO density matrices
   * @throws std::runtime_error if occupation vector sizes don't match number of
   * MOs
   */
  virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
  calculate_ao_density_matrix(const Eigen::VectorXd& occupations_alpha,
                              const Eigen::VectorXd& occupations_beta) const;

  /**
   * @brief Calculate atomic orbital (AO) density matrix from restricted
   * occupation vector
   * @param occupations Occupation vector (size must match number of MOs)
   * @return AO density matrix (total alpha + beta)
   * @throws std::runtime_error if occupation vector size doesn't match number
   * of MOs
   */
  virtual Eigen::MatrixXd calculate_ao_density_matrix(
      const Eigen::VectorXd& occupations) const;

  /**
   * @brief Calculate atomic orbital (AO) density matrix from 1RDM in molecular
   * orbital (MO) space
   * @param rdm_alpha Alpha 1RDM in MO basis (size must match number of MOs x
   * MOs)
   * @param rdm_beta Beta 1RDM in MO basis (size must match number of MOs x MOs)
   * @return Pair of (alpha, beta) AO density matrices
   * @throws std::runtime_error if 1RDM matrix sizes don't match number of MOs
   */
  virtual std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
  calculate_ao_density_matrix_from_rdm(const Eigen::MatrixXd& rdm_alpha,
                                       const Eigen::MatrixXd& rdm_beta) const;

  /**
   * @brief Calculate atomic orbital (AO) density matrix from 1RDM in molecular
   * orbital (MO) space (restricted)
   * @param rdm 1RDM in MO basis (size must match number of MOs x MOs)
   * @return AO density matrix (total alpha + beta)
   * @throws std::runtime_error if 1RDM matrix size doesn't match number of MOs
   */
  virtual Eigen::MatrixXd calculate_ao_density_matrix_from_rdm(
      const Eigen::MatrixXd& rdm) const;

  /**
   * @brief Get summary string of orbital information
   * @return String describing the orbital data
   */
  std::string get_summary() const override;

  /**
   * @brief Save orbital data to file based on type parameter
   * @param filename Path to file to create/overwrite
   * @param type File format type ("json" or "hdf5")
   * @throws std::runtime_error if data is invalid, unsupported type, or I/O
   * error occurs
   */

  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Convert orbital data to JSON
   * @return JSON object containing orbital data
   * @throws std::runtime_error if data is invalid
   */
  virtual nlohmann::json to_json() const override;

  /**
   * @brief Save orbital data to JSON file (with validation)
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if data is invalid or I/O error occurs
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Serialize orbital data to HDF5 group
   * @param group HDF5 group to write data to
   * @throws std::runtime_error if I/O error occurs
   */
  virtual void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save orbital data to HDF5 file (with validation)
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if data is invalid or I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load orbital data from file based on type parameter
   * @param filename Path to file to read
   * @param type File format type ("json" or "hdf5")
   * @return New Orbitals object loaded from file
   * @throws std::runtime_error if file doesn't exist, unsupported type, or I/O
   * error occurs
   */
  static std::shared_ptr<Orbitals> from_file(const std::string& filename,
                                             const std::string& type);

  /**
   * @brief Load orbital data from HDF5 file
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to const Orbitals object loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Orbitals> from_hdf5_file(const std::string& filename);

  /**
   * @brief Load orbital data from HDF5 group
   * @param group HDF5 group containing orbital data
   * @return Shared pointer to Orbitals object loaded from group
   * @throws std::runtime_error if HDF5 data is malformed or I/O error occurs
   */
  static std::shared_ptr<Orbitals> from_hdf5(H5::Group& group);

  /**
   * @brief Load orbital data from JSON file
   * @param filename Path to JSON file to read
   * @return Shared pointer to const Orbitals object loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Orbitals> from_json_file(const std::string& filename);

  /**
   * @brief Load orbital data from JSON
   * @param j JSON object containing orbital data
   * @return Shared pointer to const Orbitals object loaded from JSON
   * @throws std::runtime_error if JSON is malformed or missing required data
   */
  static std::shared_ptr<Orbitals> from_json(const nlohmann::json& j);

 protected:
  /**
   * Orbital coefficients [AO x MO] for (alpha, beta) spin channels
   */
  std::pair<std::shared_ptr<Eigen::MatrixXd>, std::shared_ptr<Eigen::MatrixXd>>
      _coefficients = {nullptr, nullptr};

  /**
   * Orbital energies for (alpha, beta) spin channels
   */
  std::pair<std::shared_ptr<Eigen::VectorXd>, std::shared_ptr<Eigen::VectorXd>>
      _energies = {nullptr, nullptr};

  /**
   * Active space indices for (alpha, beta) spin channels
   */
  std::pair<std::vector<size_t>, std::vector<size_t>> _active_space_indices = {
      {}, {}};

  /**
   * Inactive space indices for (alpha, beta) spin channels
   */
  std::pair<std::vector<size_t>, std::vector<size_t>> _inactive_space_indices =
      {{}, {}};

  /**
   * Atomic orbital overlap matrix [AO x AO]
   */
  std::unique_ptr<Eigen::MatrixXd> _ao_overlap = nullptr;

  /**
   * Comprehensive basis set information
   */
  std::shared_ptr<BasisSet> _basis_set;

  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /**
   * Save orbital metadata to HDF5 file
   */
  void _save_metadata_to_hdf5(H5::H5File& file) const;

  /**
   * Save orbital metadata to HDF5 file with direct parameters
   */
  static void _save_orbital_metadata_to_hdf5(H5::H5File& file,
                                             size_t num_atomic_orbitals,
                                             size_t num_molecular_orbitals,
                                             bool is_restricted,
                                             bool has_overlap,
                                             bool has_basis_set);

  /**
   * @brief Check if orbital data is complete and consistent
   * @return True if all required data is set and consistent
   */
  virtual bool _is_valid() const;

  /**
   * @brief Perform post-construction validation
   * @throws std::runtime_error if validation fails
   */
  virtual void _post_construction_validate();

  /**
   * @brief Save to JSON file without filename validation (internal use)
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if data is invalid or I/O error occurs
   */
  void _to_json_file(const std::string& filename) const;

  /**
   * @brief Save to HDF5 file without filename validation (internal use)
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if data is invalid or I/O error occurs
   */
  void _to_hdf5_file(const std::string& filename) const;

  /**
   * @brief Load from JSON file without filename validation (internal static
   * use)
   * @param filename Path to JSON file to read
   * @return Shared pointer to const Orbitals object loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Orbitals> _from_json_file(const std::string& filename);

  /**
   * @brief Load from HDF5 file without filename validation (internal static
   * use)
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to const Orbitals object loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Orbitals> _from_hdf5_file(const std::string& filename);

 protected:
  // Protected default constructor for use by subclasses
  Orbitals() = default;
};

/**
 * @class ModelOrbitals
 * @brief Simple subclass of Orbitals for model systems without basis set
 * information.
 *
 * This class allows creating Orbitals objects with a specified basis size
 * and whether the calculation is restricted or unrestricted, without needing
 * to provide full coefficient or energy data. The class allows for model
 * Hamiltonians and Wavefunctions to be fully specified without explicit basis
 * set details.
 *
 * Calls to any functions requiring actual data (e.g. get_coefficients,
 * get_energies, calculate_ao_density_matrix, etc.) will throw runtime errors.
 */
class ModelOrbitals : public Orbitals {
 public:
  ModelOrbitals(size_t basis_size, bool restricted);

  /**
   * @brief Constructor with active and inactive space indices (restricted)
   * @param basis_size Number of basis functions (and molecular orbitals)
   * @param indices Orbital indices (shared for alpha/beta)
   * @todo TODO (NAB):  need to clarify the purpose/scope of these indices
   * 41469
   */
  ModelOrbitals(
      size_t basis_size,
      const std::tuple<std::vector<size_t>, std::vector<size_t>>& indices);

  /**
   * @brief Constructor with active and inactive space indices (unrestricted)
   * @param basis_size Number of basis functions (and molecular orbitals)
   * @param indices Orbital indices (shared for alpha/beta)
   * @todo TODO (NAB):  need to clarify the purpose/scope of these indices
   * 41469
   */
  ModelOrbitals(
      size_t basis_size,
      const std::tuple<std::vector<size_t>, std::vector<size_t>,
                       std::vector<size_t>, std::vector<size_t>>& indices);

  // Copy constructor and assignment operator
  ModelOrbitals(const ModelOrbitals& other);
  ModelOrbitals& operator=(const ModelOrbitals& other);

  virtual ~ModelOrbitals() = default;

  // Override methods that should throw errors for model systems
  std::pair<const Eigen::MatrixXd&, const Eigen::MatrixXd&> get_coefficients()
      const override;
  std::pair<const Eigen::VectorXd&, const Eigen::VectorXd&> get_energies()
      const override;
  std::shared_ptr<BasisSet> get_basis_set() const override;
  const Eigen::MatrixXd& get_coefficients_alpha() const override;
  const Eigen::MatrixXd& get_coefficients_beta() const override;
  const Eigen::VectorXd& get_energies_alpha() const override;
  const Eigen::VectorXd& get_energies_beta() const override;
  const Eigen::MatrixXd& get_overlap_matrix() const override;

  // Density matrix methods should also error out
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd> calculate_ao_density_matrix(
      const Eigen::VectorXd& occupations_alpha,
      const Eigen::VectorXd& occupations_beta) const override;
  Eigen::MatrixXd calculate_ao_density_matrix(
      const Eigen::VectorXd& occupations) const override;
  std::pair<Eigen::MatrixXd, Eigen::MatrixXd>
  calculate_ao_density_matrix_from_rdm(
      const Eigen::MatrixXd& rdm_alpha,
      const Eigen::MatrixXd& rdm_beta) const override;
  Eigen::MatrixXd calculate_ao_density_matrix_from_rdm(
      const Eigen::MatrixXd& rdm) const override;

  // MO overlap methods return identity matrices for model systems
  std::tuple<Eigen::MatrixXd, Eigen::MatrixXd, Eigen::MatrixXd> get_mo_overlap()
      const override;
  Eigen::MatrixXd get_mo_overlap_alpha_alpha() const override;
  Eigen::MatrixXd get_mo_overlap_alpha_beta() const override;
  Eigen::MatrixXd get_mo_overlap_beta_beta() const override;

  // Override size query methods
  size_t get_num_molecular_orbitals() const override { return _num_orbitals; }
  size_t get_num_atomic_orbitals() const override { return _num_orbitals; }
  std::vector<size_t> get_all_mo_indices() const override;

  // Override calculation type queries
  bool is_restricted() const override;
  bool is_unrestricted() const override;

  // Override serialization
  nlohmann::json to_json() const override;
  static std::shared_ptr<ModelOrbitals> from_json(const nlohmann::json& j);

  /**
   * @brief Serialize ModelOrbitals to HDF5 group
   * @param group HDF5 group to write data to
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Load ModelOrbitals from HDF5 group
   * @param group HDF5 group containing ModelOrbitals data
   * @return Shared pointer to ModelOrbitals object loaded from group
   * @throws std::runtime_error if HDF5 data is malformed or I/O error occurs
   */
  static std::shared_ptr<ModelOrbitals> from_hdf5(H5::Group& group);

 private:
  size_t _num_orbitals;  ///< Number of molecular orbitals
  bool _is_restricted;   ///< Whether this is a restricted calculation

  // Override validation
  bool _is_valid() const override;
  void _post_construction_validate() override;
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(DataClassCompliant<Orbitals>,
              "Orbitals must derive from DataClass and implement all required "
              "deserialization methods");

}  // namespace qdk::chemistry::data

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <qdk/chemistry/data/data_class.hpp>

namespace qdk::chemistry::data {

/**
 * @class StabilityResult
 * @brief Result structure for wavefunction stability analysis
 *
 * The StabilityResult class encapsulates the results of a stability check
 * performed on a wavefunction. It contains information about whether the
 * wavefunction is stable, along with the eigenvalues and eigenvectors
 * of the stability matrix for both internal and external stability. (See
 * J. Chem. Phys. 66, 3045–3050 (1977) for classification of stability types.)
 *
 * This class provides:
 * - Internal and external stability status
 * - Overall stability status (stable only if both internal and external are
 * stable)
 * - Internal and external eigenvalues of the stability matrices
 * - Internal and external eigenvectors of the stability matrices
 * - Convenient access methods for stability analysis results
 *
 * @note Internal stability typically refers to stability against perturbations
 * within the same method (e.g. RHF -> RHF), while external stability refers to
 * stability against perturbations between different methods (e.g. RHF -> UHF).
 *
 * @section eigenvector_format Eigenvector Format
 *
 * The eigenvectors encode orbital rotation parameters between occupied
 * and virtual orbitals. The required size depends on the orbital type:
 *
 * **RHF (Restricted Hartree-Fock):**
 * - Size: num_occupied_orbitals * num_virtual_orbitals
 * - Where: num_virtual_orbitals = num_molecular_orbitals -
 *                                 num_occupied_orbitals
 * - Elements represent rotations between occupied and virtual spatial orbitals
 * - Both spins rotate together (spin symmetry preserved)
 *
 * **UHF (Unrestricted Hartree-Fock):**
 * - Size: num_alpha_occupied_orbitals * num_alpha_virtual_orbitals +
 *         num_beta_occupied_orbitals * num_beta_virtual_orbitals
 * - Where: num_alpha_virtual_orbitals = num_molecular_orbitals -
 *                                       num_alpha_occupied_orbitals
 *          num_beta_virtual_orbitals = num_molecular_orbitals -
 *                                      num_beta_occupied_orbitals
 * - First num_alpha_occupied_orbitals * num_alpha_virtual_orbitals elements:
 *   alpha rotations
 * - Last num_beta_occupied_orbitals * num_beta_virtual_orbitals elements:
 *   beta rotations
 * - Alpha and beta orbitals rotate independently
 *
 * **ROHF (Restricted Open-shell Hartree-Fock):**
 * - The rotation mask is the union of two rectangular blocks:
 *   1. Alpha block: num_alpha_occupied_orbitals * num_alpha_virtual_orbitals
 *   2. Beta block: num_beta_occupied_orbitals * num_beta_virtual_orbitals
 * - Size calculation for union (assuming num_alpha_occupied >=
 * num_beta_occupied): num_alpha_occupied_orbitals * (num_molecular_orbitals -
 * num_alpha_occupied_orbitals) + (num_alpha_occupied_orbitals -
 * num_beta_occupied_orbitals) * num_beta_occupied_orbitals
 * - This equals the virtual-occupied block plus the additional
 *   doubly-occupied to singly-occupied block
 *
 * **Indexing Convention:**
 * The occupied orbital index varies fastest. For the element corresponding to
 * occupied orbital i and virtual orbital a, the index is computed as:
 * index = i + a * num_occupied. This convention is from row-major eigenvector
 * (num_virtual, num_occupied).
 */
class StabilityResult : public DataClass,
                        public std::enable_shared_from_this<StabilityResult> {
 public:
  // === Constructors and destructor ===

  /**
   * @brief Default constructor
   */
  StabilityResult() = default;

  /**
   * @brief Construct a StabilityResult with specific values
   *
   * @param internal_stable True if internal stability is satisfied
   * @param external_stable True if external stability is satisfied
   * @param internal_eigenvalues Eigenvalues of the internal stability matrix
   * @param internal_eigenvectors Eigenvectors of the internal stability matrix
   * @param external_eigenvalues Eigenvalues of the external stability matrix
   * @param external_eigenvectors Eigenvectors of the external stability matrix
   */
  StabilityResult(bool internal_stable, bool external_stable,
                  const Eigen::VectorXd& internal_eigenvalues,
                  const Eigen::MatrixXd& internal_eigenvectors,
                  const Eigen::VectorXd& external_eigenvalues,
                  const Eigen::MatrixXd& external_eigenvectors)
      : internal_stable_(internal_stable),
        external_stable_(external_stable),
        internal_eigenvalues_(internal_eigenvalues),
        internal_eigenvectors_(internal_eigenvectors),
        external_eigenvalues_(external_eigenvalues),
        external_eigenvectors_(external_eigenvectors) {}

  /**
   * @brief Copy constructor
   */
  StabilityResult(const StabilityResult&) = default;

  /**
   * @brief Move constructor
   */
  StabilityResult(StabilityResult&&) noexcept = default;

  /**
   * @brief Copy assignment operator
   */
  StabilityResult& operator=(const StabilityResult&) = default;

  /**
   * @brief Move assignment operator
   */
  StabilityResult& operator=(StabilityResult&&) noexcept = default;

  /**
   * @brief Destructor
   */
  virtual ~StabilityResult() = default;

  // === Data access methods ===

  /**
   * @brief Check if the wavefunction is stable overall
   *
   * @return True if both internal and external stability are satisfied
   */
  bool is_stable() const { return internal_stable_ && external_stable_; }

  /**
   * @brief Check if internal stability is satisfied
   *
   * @return True if internal stability is satisfied
   */
  bool is_internal_stable() const { return internal_stable_; }

  /**
   * @brief Check if external stability is satisfied
   *
   * @return True if external stability is satisfied
   */
  bool is_external_stable() const { return external_stable_; }

  /**
   * @brief Get the internal eigenvalues of the stability matrix
   *
   * @return Reference to the internal eigenvalues vector
   */
  const Eigen::VectorXd& get_internal_eigenvalues() const {
    return internal_eigenvalues_;
  }

  /**
   * @brief Get the internal eigenvectors of the stability matrix
   *
   * @return Reference to the internal eigenvectors matrix
   */
  const Eigen::MatrixXd& get_internal_eigenvectors() const {
    return internal_eigenvectors_;
  }

  /**
   * @brief Get the external eigenvalues of the stability matrix
   *
   * @return Reference to the external eigenvalues vector
   */
  const Eigen::VectorXd& get_external_eigenvalues() const {
    return external_eigenvalues_;
  }

  /**
   * @brief Get the external eigenvectors of the stability matrix
   *
   * @return Reference to the external eigenvectors matrix
   */
  const Eigen::MatrixXd& get_external_eigenvectors() const {
    return external_eigenvectors_;
  }

  // === Data modification methods ===

  /**
   * @brief Set the internal stability status
   *
   * @param internal_stable True if internal stability is satisfied
   */
  void set_internal_stable(bool internal_stable) {
    internal_stable_ = internal_stable;
  }

  /**
   * @brief Set the external stability status
   *
   * @param external_stable True if external stability is satisfied
   */
  void set_external_stable(bool external_stable) {
    external_stable_ = external_stable;
  }

  /**
   * @brief Set the internal eigenvalues
   *
   * @param internal_eigenvalues The internal eigenvalues of the stability
   * matrix
   */
  void set_internal_eigenvalues(const Eigen::VectorXd& internal_eigenvalues) {
    internal_eigenvalues_ = internal_eigenvalues;
  }

  /**
   * @brief Set the internal eigenvectors
   *
   * @param internal_eigenvectors The internal eigenvectors of the stability
   * matrix
   */
  void set_internal_eigenvectors(const Eigen::MatrixXd& internal_eigenvectors) {
    internal_eigenvectors_ = internal_eigenvectors;
  }

  /**
   * @brief Set the external eigenvalues
   *
   * @param external_eigenvalues The external eigenvalues of the stability
   * matrix
   */
  void set_external_eigenvalues(const Eigen::VectorXd& external_eigenvalues) {
    external_eigenvalues_ = external_eigenvalues;
  }

  /**
   * @brief Set the external eigenvectors
   *
   * @param external_eigenvectors The external eigenvectors of the stability
   * matrix
   */
  void set_external_eigenvectors(const Eigen::MatrixXd& external_eigenvectors) {
    external_eigenvectors_ = external_eigenvectors;
  }

  // === Utility methods ===

  /**
   * @brief Get the number of internal eigenvalues
   * @return Number of internal eigenvalues in the stability matrix
   */
  size_t internal_size() const {
    return static_cast<size_t>(internal_eigenvalues_.size());
  }

  /**
   * @brief Get the number of external eigenvalues
   * @return Number of external eigenvalues in the stability matrix
   */
  size_t external_size() const {
    return static_cast<size_t>(external_eigenvalues_.size());
  }

  /**
   * @brief Get the smallest internal eigenvalue
   * @return Smallest internal eigenvalue (most negative for unstable systems)
   * @throws std::runtime_error if no internal eigenvalues are present
   */
  double get_smallest_internal_eigenvalue() const;

  /**
   * @brief Get the smallest external eigenvalue
   * @return Smallest external eigenvalue (most negative for unstable systems)
   * @throws std::runtime_error if no external eigenvalues are present
   */
  double get_smallest_external_eigenvalue() const;

  /**
   * @brief Get the smallest eigenvalue across both internal and external
   * @return Smallest eigenvalue overall (most negative for unstable systems)
   * @throws std::runtime_error if no eigenvalues are present
   */
  double get_smallest_eigenvalue() const;

  /**
   * @brief Get the smallest internal eigenvalue and its corresponding
   * eigenvector
   * @return Pair of (eigenvalue, eigenvector) for the smallest internal
   * eigenvalue
   * @throws std::runtime_error if no internal eigenvalues are present
   */
  std::pair<double, Eigen::VectorXd>
  get_smallest_internal_eigenvalue_and_vector() const;

  /**
   * @brief Get the smallest external eigenvalue and its corresponding
   * eigenvector
   * @return Pair of (eigenvalue, eigenvector) for the smallest external
   * eigenvalue
   * @throws std::runtime_error if no external eigenvalues are present
   */
  std::pair<double, Eigen::VectorXd>
  get_smallest_external_eigenvalue_and_vector() const;

  /**
   * @brief Get the smallest eigenvalue and its corresponding eigenvector across
   * both internal and external
   * @return Pair of (eigenvalue, eigenvector) for the smallest eigenvalue
   * overall
   * @throws std::runtime_error if no eigenvalues are present
   */
  std::pair<double, Eigen::VectorXd> get_smallest_eigenvalue_and_vector() const;

  // === DataClass interface implementation ===

  /**
   * @brief Get summary string of stability result information
   * @return String describing the stability result
   */
  std::string get_summary() const override;

  /**
   * @brief Check if the stability result is empty
   * @return true if no eigenvalues/eigenvectors are present for internal or
   * external
   */
  bool empty() const;

  /**
   * @brief Check if internal stability data is present
   * @return true if internal eigenvalues are present
   */
  bool has_internal_result() const;

  /**
   * @brief Check if external stability data is present
   * @return true if external eigenvalues are present
   */
  bool has_external_result() const;

  // === File I/O methods (DataClass interface) ===

  /**
   * @brief Save object to file in the specified format
   * @param filename Path to the output file
   * @param type Format type (e.g., "json", "hdf5")
   * @throws std::invalid_argument if format type is not supported
   * @throws std::runtime_error if I/O error occurs
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Convert object to JSON representation
   * @return JSON object containing the serialized data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save object to JSON file
   * @param filename Path to the output JSON file
   * @throws std::runtime_error if I/O error occurs
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Save object to HDF5 group
   * @param group HDF5 group to save data to
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save object to HDF5 file
   * @param filename Path to the output HDF5 file
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  // === Static deserialization methods ===

  /**
   * @brief Load object from file in the specified format
   * @param filename Path to the input file
   * @param type Format type (e.g., "json", "hdf5")
   * @return Shared pointer to the loaded StabilityResult
   * @throws std::invalid_argument if format type is not supported
   * @throws std::runtime_error if I/O error occurs
   */
  static std::shared_ptr<StabilityResult> from_file(const std::string& filename,
                                                    const std::string& type);

  /**
   * @brief Load object from JSON file
   * @param filename Path to the input JSON file
   * @return Shared pointer to the loaded StabilityResult
   * @throws std::runtime_error if I/O error occurs
   */
  static std::shared_ptr<StabilityResult> from_json_file(
      const std::string& filename);

  /**
   * @brief Load object from JSON data
   * @param j JSON object containing the serialized data
   * @return Shared pointer to the loaded StabilityResult
   * @throws std::runtime_error if JSON data is invalid
   */
  static std::shared_ptr<StabilityResult> from_json(const nlohmann::json& j);

  /**
   * @brief Load object from HDF5 file
   * @param filename Path to the input HDF5 file
   * @return Shared pointer to the loaded StabilityResult
   * @throws std::runtime_error if I/O error occurs
   */
  static std::shared_ptr<StabilityResult> from_hdf5_file(
      const std::string& filename);

  /**
   * @brief Load object from HDF5 group
   * @param group HDF5 group to load data from
   * @return Shared pointer to the loaded StabilityResult
   * @throws std::runtime_error if I/O error occurs
   */
  static std::shared_ptr<StabilityResult> from_hdf5(H5::Group& group);

 private:
  /// True if internal stability is satisfied (default: true)
  bool internal_stable_ = true;

  /// True if external stability is satisfied (default: true)
  bool external_stable_ = true;

  /// Internal eigenvalues of the stability matrix
  Eigen::VectorXd internal_eigenvalues_;

  /// Internal eigenvectors of the stability matrix
  Eigen::MatrixXd internal_eigenvectors_;

  /// External eigenvalues of the stability matrix
  Eigen::VectorXd external_eigenvalues_;

  /// External eigenvectors of the stability matrix
  Eigen::MatrixXd external_eigenvectors_;

  /// Serialization version for compatibility checking
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /**
   * @brief Check if the StabilityResult object is in a valid state
   *
   * This method performs internal consistency checks on the object's data
   * without throwing exceptions. It verifies dimensional consistency between
   * eigenvalues and eigenvectors and other internal state requirements.
   *
   * @return true if the object is in a valid state, false otherwise
   */
  bool _is_valid() const;

  /**
   * @brief Private function to save to JSON file without validation
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_json_file(const std::string& filename) const;

  /**
   * @brief Private function to save to HDF5 file without validation
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_hdf5_file(const std::string& filename) const;

  /**
   * @brief Private function to load from JSON file without validation
   * @param filename Path to JSON file to read
   * @return Shared pointer to new StabilityResult instance
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<StabilityResult> _from_json_file(
      const std::string& filename);

  /**
   * @brief Private function to load from HDF5 file without validation
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to new StabilityResult instance
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<StabilityResult> _from_hdf5_file(
      const std::string& filename);
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(DataClassCompliant<StabilityResult>,
              "StabilityResult must derive from DataClass and implement all "
              "required deserialization methods");

}  // namespace qdk::chemistry::data

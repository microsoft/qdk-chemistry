// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/configuration_set.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Enum to distinguish between different wavefunction representations
 *
 * This enum allows tagging wavefunctions based on their mathematical role:
 * - SelfDual: Wavefunctions that can be used as both bra and ket
 * - NotSelfDual: Wavefunctions that are strictly bra or ket
 *
 * This distinction maps to the use of hermitian or non-hermitian operators.
 */
enum class WavefunctionType { SelfDual, NotSelfDual };

/**
 * @brief Common type definitions used across all container types
 *
 * This struct provides centralized type definitions for scalar, matrix, and
 * vector types that can be either real or complex, used throughout the
 * wavefunction system.
 */
namespace ContainerTypes {
// Basic types that can be real or complex
using ScalarVariant = std::variant<double, std::complex<double>>;

// Vector types
using VectorVariant = std::variant<Eigen::VectorXd, Eigen::VectorXcd>;

// Matrix types
using MatrixVariant = std::variant<Eigen::MatrixXd, Eigen::MatrixXcd>;

// Common types
using DeterminantVector = std::vector<Configuration>;
};  // namespace ContainerTypes

namespace detail {
/**
 * @brief Helper to create a ContainerTypes::MatrixVariant from scalar
 * multiplication
 * @param matrix The matrix variant to multiply
 * @param scalar The scalar value to multiply by
 * @return Shared pointer to new ContainerTypes::MatrixVariant containing the
 * result
 */
template <typename Scalar>
std::shared_ptr<ContainerTypes::MatrixVariant> multiply_matrix_variant(
    const ContainerTypes::MatrixVariant& matrix, Scalar scalar) {
  return std::visit(
      [scalar](
          const auto& mat) -> std::shared_ptr<ContainerTypes::MatrixVariant> {
        using MatType = std::decay_t<decltype(mat)>;
        return std::make_shared<ContainerTypes::MatrixVariant>(
            std::in_place_type<MatType>, mat * scalar);
      },
      matrix);
}

/**
 * @brief Helper to create a ContainerTypes::VectorVariant from scalar
 * multiplication
 * @param vector The vector variant to multiply
 * @param scalar The scalar value to multiply by
 * @return Shared pointer to new ContainerTypes::VectorVariant containing the
 * result
 */
template <typename Scalar>
std::shared_ptr<ContainerTypes::VectorVariant> multiply_vector_variant(
    const ContainerTypes::VectorVariant& vector, Scalar scalar) {
  return std::visit(
      [scalar](
          const auto& vec) -> std::shared_ptr<ContainerTypes::VectorVariant> {
        using VecType = std::decay_t<decltype(vec)>;
        return std::make_shared<ContainerTypes::VectorVariant>(
            std::in_place_type<VecType>, vec * scalar);
      },
      vector);
}

/**
 * @brief Helper to add two ContainerTypes::MatrixVariants
 * @param mat1 First matrix variant
 * @param mat2 Second matrix variant
 * @return Shared pointer to new ContainerTypes::MatrixVariant containing the
 * sum
 */
std::shared_ptr<ContainerTypes::MatrixVariant> add_matrix_variants(
    const ContainerTypes::MatrixVariant& mat1,
    const ContainerTypes::MatrixVariant& mat2);

/**
 * @brief Helper to add two ContainerTypes::VectorVariants
 * @param vec1 First vector variant
 * @param vec2 Second vector variant
 * @return Shared pointer to new ContainerTypes::VectorVariant containing the
 * sum
 */
std::shared_ptr<ContainerTypes::VectorVariant> add_vector_variants(
    const ContainerTypes::VectorVariant& vec1,
    const ContainerTypes::VectorVariant& vec2);

/**
 * @brief Check if a ContainerTypes::MatrixVariant contains complex type
 * @param variant The matrix variant to check
 * @return True if contains Eigen::MatrixXcd, false if Eigen::MatrixXd
 */
bool is_matrix_variant_complex(const ContainerTypes::MatrixVariant& variant);

/**
 * @brief Check if a ContainerTypes::VectorVariant contains complex type
 * @param variant The vector variant to check
 * @return True if contains Eigen::VectorXcd, false if Eigen::VectorXd
 */
bool is_vector_variant_complex(const ContainerTypes::VectorVariant& variant);

/**
 * @brief Transpose a 2-RDM vector from (ij|kl) to (kl|ij) ordering
 * @param variant The vector variant to transpose
 * @param norbs Number of orbitals
 * @return shared pointer to new ContainerTypes::VectorVariant containing the
 * transposed data
 */
std::shared_ptr<ContainerTypes::VectorVariant>
transpose_ijkl_klij_vector_variant(const ContainerTypes::VectorVariant& variant,
                                   int norbs);
}  // namespace detail

/**
 * @brief Abstract base class for wavefunction containers
 *
 * This class provides the interface for different types of wavefunction
 * representations (e.g., CI, MCSCF, coupled cluster). It uses variant types to
 * support both real and complex arithmetic, and provides methods for accessing
 * coefficients, reduced density matrices (RDMs), and other wavefunction
 * properties.
 */
class WavefunctionContainer {
 public:
  // Using variant types to support both real and complex data
  using MatrixVariant = ContainerTypes::MatrixVariant;
  using VectorVariant = ContainerTypes::VectorVariant;
  using ScalarVariant = ContainerTypes::ScalarVariant;
  using DeterminantVector = ContainerTypes::DeterminantVector;

  /**
   * @brief Constructor
   * @param type Type of wavefunction (SelfDual or NotSelfDual)
   */
  WavefunctionContainer(WavefunctionType type = WavefunctionType::SelfDual);

  /**
   * @brief Constructs a wavefunction with spin-traced reduced density matrix
   * (RDM) data
   *
   * @param one_rdm_spin_traced Spin-traced 1-RDM for active orbitals (optional)
   * @param two_rdm_spin_traced Spin-traced 2-RDM for active orbitals (optional)
   * @param type The type of wavefunction
   */
  WavefunctionContainer(const std::optional<MatrixVariant>& one_rdm_spin_traced,
                        const std::optional<VectorVariant>& two_rdm_spin_traced,
                        WavefunctionType type = WavefunctionType::SelfDual);

  /**
   * @brief Constructs a wavefunction container with reduced density matrix
   * (RDM) data
   *
   * @param one_rdm_spin_traced Spin-traced 1-RDM for active orbitals (optional)
   * @param one_rdm_aa Alpha-alpha block of 1-RDM for active orbitals (optional)
   * @param one_rdm_bb Beta-beta block of 1-RDM for active orbitals (optional)
   * @param two_rdm_spin_traced Spin-traced 2-RDM for active orbitals (optional)
   * @param two_rdm_aabb Alpha-beta-alpha-beta block of 2-RDM for active
   * orbitals (optional)
   * @param two_rdm_aaaa Alpha-alpha-alpha-alpha block of 2-RDM for active
   * orbitals (optional)
   * @param two_rdm_bbbb Beta-beta-beta-beta block of 2-RDM for active orbitals
   * (optional)
   * @param type The type of wavefunction
   */
  WavefunctionContainer(const std::optional<MatrixVariant>& one_rdm_spin_traced,
                        const std::optional<MatrixVariant>& one_rdm_aa,
                        const std::optional<MatrixVariant>& one_rdm_bb,
                        const std::optional<VectorVariant>& two_rdm_spin_traced,
                        const std::optional<VectorVariant>& two_rdm_aabb,
                        const std::optional<VectorVariant>& two_rdm_aaaa,
                        const std::optional<VectorVariant>& two_rdm_bbbb,
                        WavefunctionType type = WavefunctionType::SelfDual);

  virtual ~WavefunctionContainer() = default;

  /**
   * @brief Create a deep copy of this container
   * @return Unique pointer to a cloned container
   */
  virtual std::unique_ptr<WavefunctionContainer> clone() const = 0;

  /**
   * @brief Get all coefficients
   * @return Vector of all coefficients (real or complex)
   */
  virtual const VectorVariant& get_coefficients() const = 0;

  /**
   * @brief Get coefficient for a specific determinant
   * @param det Configuration/determinant to get coefficient for
   * @return Scalar coefficient (real or complex)
   */
  virtual ScalarVariant get_coefficient(const Configuration& det) const = 0;

  /**
   * @brief Get all determinants in the wavefunction
   * @return Vector of all configurations/determinants
   */
  virtual const DeterminantVector& get_active_determinants() const = 0;

  /**
   * @brief Get number of determinants
   * @return Number of determinants in the wavefunction
   */
  virtual size_t size() const = 0;

  /**
   * @brief Calculate overlap with another wavefunction
   * @param other Other wavefunction container
   * @return Overlap value (real or complex)
   */
  virtual ScalarVariant overlap(const WavefunctionContainer& other) const = 0;

  /**
   * @brief Calculate norm of the wavefunction
   * @return Norm (always real)
   */
  virtual double norm() const = 0;

  /**
   * @brief Get spin-dependent one-particle RDMs for active orbitals only
   * @return Tuple of (alpha-alpha, beta-beta) one-particle RDMs for active
   * orbitals
   */
  virtual std::tuple<const MatrixVariant&, const MatrixVariant&>
  get_active_one_rdm_spin_dependent() const;

  /**
   * @brief Get spin-dependent two-particle RDMs for active orbitals only
   * @return Tuple of (aabb, aaaa, bbbb) two-particle RDMs for active orbitals
   */
  virtual std::tuple<const VectorVariant&, const VectorVariant&,
                     const VectorVariant&>
  get_active_two_rdm_spin_dependent() const;

  /**
   * @brief Get spin-traced one-particle RDM for active orbitals only
   * @return Spin-traced one-particle RDM for active orbitals
   */
  virtual const MatrixVariant& get_active_one_rdm_spin_traced() const;

  /**
   * @brief Get spin-traced two-particle RDM for active orbitals only
   * @return Spin-traced two-particle RDM for active orbitals
   */
  virtual const VectorVariant& get_active_two_rdm_spin_traced() const;

  /**
   * @brief Checks if single-orbital entropies for active orbitals are available
   *
   * @return True if single-orbital entropies are available, false otherwise
   */
  virtual bool has_single_orbital_entropies() const;

  /**
   * @brief Calculate single orbital entropies for active orbitals only
   *
   * This function uses the method of Boguslawski & Tecmer (2015),
   * doi:10.1002/qua.24832, :cite:`Boguslawski2015`.
   * @return Vector of orbital entropies for active orbitals (always real)
   */
  virtual Eigen::VectorXd get_single_orbital_entropies() const;

  /**
   * @brief Get total number of alpha and beta electrons (active + inactive)
   * @return Pair of (n_alpha_total, n_beta_total) electrons
   */
  virtual std::pair<size_t, size_t> get_total_num_electrons() const = 0;

  /**
   * @brief Get number of active alpha and beta electrons
   * @return Pair of (n_alpha_active, n_beta_active) electrons
   */
  virtual std::pair<size_t, size_t> get_active_num_electrons() const = 0;

  /**
   * @brief Get orbital occupations for all orbitals (total = active + inactive
   * + virtual)
   * @return Pair of (alpha_occupations_total, beta_occupations_total)
   */
  virtual std::pair<Eigen::VectorXd, Eigen::VectorXd>
  get_total_orbital_occupations() const = 0;

  /**
   * @brief Get orbital occupations for active orbitals only
   * @return Pair of (alpha_active_occupations, beta_active_occupations)
   */
  virtual std::pair<Eigen::VectorXd, Eigen::VectorXd>
  get_active_orbital_occupations() const = 0;

  /**
   * @brief Check if spin-dependent one-particle RDMs for active orbitals are
   * available
   * @return True if available
   */
  virtual bool has_one_rdm_spin_dependent() const;

  /**
   * @brief Check if spin-traced one-particle RDM for active orbitals is
   * available
   * @return True if available
   */
  virtual bool has_one_rdm_spin_traced() const;

  /**
   * @brief Check if spin-dependent two-particle RDMs for active orbitals are
   * available
   * @return True if available
   */
  virtual bool has_two_rdm_spin_dependent() const;

  /**
   * @brief Check if spin-traced two-particle RDM for active orbitals is
   * available
   * @return True if available
   */
  virtual bool has_two_rdm_spin_traced() const;

  /**
   * @brief Clear cached data to release memory
   *
   * This method cleans up memoized evaluations of data derived from the
   * wavefunction, such as reduced density matrices (RDMs), and other computed
   * properties. Each derived container implementation decides specifically what
   * cached data to clear based on its internal structure.
   *
   * Calling this method can help manage memory usage for large systems.
   */
  virtual void clear_caches() const = 0;

  /**
   * @brief Convert container to JSON format
   * @return JSON object containing container data
   */
  virtual nlohmann::json to_json() const = 0;

  /**
   * @brief Load container from JSON format
   * @param j JSON object containing container data
   * @return Unique pointer to container created from JSON data
   * @throws std::runtime_error if JSON is malformed
   */
  static std::unique_ptr<WavefunctionContainer> from_json(
      const nlohmann::json& j);

  /**
   * @brief Convert container to HDF5 group
   * @param group HDF5 group to write container data to
   * @throws std::runtime_error if HDF5 I/O error occurs
   */
  virtual void to_hdf5(H5::Group& group) const;

  /**
   * @brief Load container from HDF5 group
   * @param group HDF5 group containing container data
   * @return Unique pointer to container created from HDF5 group
   * @throws std::runtime_error if HDF5 data is malformed or I/O error occurs
   */
  static std::unique_ptr<WavefunctionContainer> from_hdf5(H5::Group& group);

  /**
   * @brief Get container type identifier for serialization
   * @return String identifying the container type (e.g., "cas", "sci", "sd")
   */
  virtual std::string get_container_type() const = 0;

  /**
   * @brief Get reference to orbital basis set
   * @return Shared pointer to orbitals
   */
  virtual std::shared_ptr<Orbitals> get_orbitals() const = 0;

  /**
   * @brief Get the wavefunction type (SelfDual or NotSelfDual)
   * @return WavefunctionType enum value
   */
  WavefunctionType get_type() const;

  /**
   * @brief Check if the wavefunction is complex-valued
   * @return True if complex, false if real
   */
  virtual bool is_complex() const = 0;

  /**
   * @brief Check if this container has coefficients data
   * @return True if coefficients are available, false otherwise
   */
  virtual bool has_coefficients() const { return false; }

  /**
   * @brief Check if this container has configuration set data
   * @return True if configuration set is available, false otherwise
   */
  virtual bool has_configuration_set() const { return false; }

  /**
   * @brief Get the configuration set for this wavefunction
   * @return Reference to the configuration set containing determinants and
   * orbitals
   * @throws std::runtime_error if configuration set is not available
   */
  virtual const ConfigurationSet& get_configuration_set() const {
    throw std::runtime_error(
        "Configuration set not available for this container type");
  }

 protected:
  /// Wavefunction type (SelfDual or NotSelfDual)
  WavefunctionType _type;
  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  // spin-traced RDMs
  mutable std::shared_ptr<MatrixVariant> _one_rdm_spin_traced = nullptr;
  mutable std::shared_ptr<VectorVariant> _two_rdm_spin_traced = nullptr;
  // spin-dependent RDMs
  mutable std::shared_ptr<MatrixVariant> _one_rdm_spin_dependent_aa = nullptr;
  mutable std::shared_ptr<MatrixVariant> _one_rdm_spin_dependent_bb = nullptr;
  mutable std::shared_ptr<VectorVariant> _two_rdm_spin_dependent_aaaa = nullptr;
  mutable std::shared_ptr<VectorVariant> _two_rdm_spin_dependent_aabb = nullptr;
  mutable std::shared_ptr<VectorVariant> _two_rdm_spin_dependent_bbbb = nullptr;

  /** @brief Clear cached RDMs */
  void _clear_rdms() const;
};

/**
 * @brief Main wavefunction class that wraps container implementations
 *
 * This class provides a high-level interface to wavefunction data by delegating
 * to specific container implementations. It combines orbital information with
 * wavefunction coefficients and provides convenient access to all wavefunction
 * properties. All methods redirect to the underlying container.
 */
class Wavefunction : public DataClass,
                     public std::enable_shared_from_this<Wavefunction> {
 public:
  /**
   * @brief Get a summary string
   * @return String containing summary
   */
  std::string get_summary() const override;

  // Type aliases for convenience
  using ScalarVariant = ContainerTypes::ScalarVariant;
  using MatrixVariant = ContainerTypes::MatrixVariant;
  using VectorVariant = ContainerTypes::VectorVariant;
  using DeterminantVector = ContainerTypes::DeterminantVector;

  /**
   * @brief Construct wavefunction with container (orbitals are stored in
   * container)
   * @param container Wavefunction container implementation
   */
  Wavefunction(std::unique_ptr<WavefunctionContainer> container);

  ~Wavefunction() = default;

  /**
   * @brief Copy constructor
   */
  Wavefunction(const Wavefunction& other);

  /**
   * @brief Move constructor
   */
  Wavefunction(Wavefunction&& other) noexcept = default;

  /**
   * @brief Copy assignment operator
   */
  Wavefunction& operator=(const Wavefunction& other);

  /**
   * @brief Move assignment operator
   */
  Wavefunction& operator=(Wavefunction&& other) noexcept = default;

  /**
   * @brief Get reference to orbital basis set
   * @return Shared pointer to orbitals
   */
  virtual std::shared_ptr<Orbitals> get_orbitals() const;

  /**
   * @brief Get the type of the underlying container
   * @return String identifying the container type (e.g., "cas", "sci", "sd")
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
   * @brief Get total number of alpha and beta electrons (active + inactive)
   * @return Pair of (n_alpha_total, n_beta_total) electrons
   */
  virtual std::pair<size_t, size_t> get_total_num_electrons() const;

  /**
   * @brief Get number of active alpha and beta electrons
   * @return Pair of (n_alpha_active, n_beta_active) electrons
   */
  virtual std::pair<size_t, size_t> get_active_num_electrons() const;

  /**
   * @brief Get orbital occupations for all orbitals (total = active + inactive
   * + virtual)
   * @return Pair of (alpha_occupations_total, beta_occupations_total)
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd> get_total_orbital_occupations()
      const;

  /**
   * @brief Get orbital occupations for active orbitals only
   * @return Pair of (alpha_active_occupations, beta_active_occupations)
   */
  std::pair<Eigen::VectorXd, Eigen::VectorXd> get_active_orbital_occupations()
      const;

  /**
   * @brief Get coefficient for a specific determinant
   *
   * The configuration is expected to be a determinant describing only
   * the wavefunction's active space.
   *
   * @param det Configuration/determinant to get coefficient for
   * @return Scalar coefficient (real or complex)
   */
  ScalarVariant get_coefficient(const Configuration& det) const;

  /**
   * @brief Get coefficients for all determinants as a vector, in which the
   * sequence of coefficients is consistent with the vector from
   * get_active_determinants()
   * @return Vector of coefficients (real or complex)
   */
  const VectorVariant& get_coefficients() const;

  /**
   * @brief Get all determinants in the wavefunction
   * @return Vector of all configurations/determinants
   */
  const DeterminantVector& get_active_determinants() const;

  /**
   * @brief Get all determinants in the wavefunction with full orbital space
   * @return Vector of all configurations/determinants including inactive and
   * virtual orbitals
   *
   * Converts stored active-space-only determinants to full orbital space by
   * prepending doubly occupied inactive orbitals and appending unoccupied
   * virtual orbitals.
   */
  DeterminantVector get_total_determinants() const;

  /**
   * @brief Extract active space determinant from a full orbital space
   * determinant
   * @param total_determinant Configuration representing full orbital space
   * @return Configuration representing only the active space portion
   *
   * Removes inactive and virtual orbital information, keeping only the active
   * space orbitals.
   */
  Configuration get_active_determinant(
      const Configuration& total_determinant) const;

  /**
   * @brief Convert active space determinant to full orbital space determinant
   * @param active_determinant Configuration representing only active space
   * @return Configuration representing full orbital space
   *
   * Expands active-space-only determinant to full orbital space by prepending
   * doubly occupied inactive orbitals and appending unoccupied virtual
   * orbitals.
   */
  Configuration get_total_determinant(
      const Configuration& active_determinant) const;

  /**
   * @brief Get number of determinants
   * @return Number of determinants in the wavefunction
   */
  size_t size() const;

  /**
   * @brief Calculate norm of the wavefunction
   * @return Norm (always real)
   */
  double norm() const;

  /**
   * @brief Calculate overlap with another wavefunction
   * @param other Other wavefunction
   * @return Overlap value (real or complex)
   */
  ScalarVariant overlap(const Wavefunction& other) const;

  /**
   * @brief Get spin-dependent one-particle RDMs
   * @return Tuple of (alpha-alpha, beta-beta) one-particle RDMs
   */
  std::tuple<const MatrixVariant&, const MatrixVariant&>
  get_active_one_rdm_spin_dependent() const;

  /**
   * @brief Get spin-dependent two-particle RDMs
   * @return Tuple of (aabb, aaaa, bbbb) two-particle RDMs
   */
  std::tuple<const VectorVariant&, const VectorVariant&, const VectorVariant&>
  get_active_two_rdm_spin_dependent() const;

  /**
   * @brief Get spin-traced one-particle RDM
   * @return Spin-traced one-particle RDM
   */
  const MatrixVariant& get_active_one_rdm_spin_traced() const;

  /**
   * @brief Get spin-traced two-particle RDM
   * @return Spin-traced two-particle RDM
   */
  const VectorVariant& get_active_two_rdm_spin_traced() const;

  /**
   * @brief Checks if single-orbital entropies for active orbitals are available
   * @return True if single-orbital entropies are available, false otherwise
   */
  virtual bool has_single_orbital_entropies() const;

  /**
   * @brief Calculate single orbital entropies for active orbitals only
   * @return Vector of orbital entropies for active orbitals (always real)
   */
  virtual Eigen::VectorXd get_single_orbital_entropies() const;

  /**
   * @brief Check if spin-dependent one-particle RDMs are available
   * @return True if available
   */
  bool has_one_rdm_spin_dependent() const;

  /**
   * @brief Check if spin-traced one-particle RDM is available
   * @return True if available
   */
  bool has_one_rdm_spin_traced() const;

  /**
   * @brief Check if spin-dependent two-particle RDMs are available
   * @return True if available
   */
  bool has_two_rdm_spin_dependent() const;

  /**
   * @brief Check if spin-traced two-particle RDM is available
   * @return True if available
   */
  bool has_two_rdm_spin_traced() const;

  /**
   * @brief Convert wavefunction to JSON format
   * @return JSON object containing wavefunction data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Load wavefunction from JSON format
   * @param j JSON object containing wavefunction data
   * @return Shared pointer to Wavefunction object created from JSON data
   * @throws std::runtime_error if JSON is malformed
   */
  static std::shared_ptr<Wavefunction> from_json(const nlohmann::json& j);

  /**
   * @brief Save wavefunction to JSON file
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Load wavefunction from JSON file
   * @param filename Path to JSON file to read
   * @return Shared pointer to Wavefunction object created from JSON file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Wavefunction> from_json_file(
      const std::string& filename);

  /**
   * @brief Convert wavefunction to HDF5 group
   * @param group HDF5 group to write wavefunction data to
   * @throws std::runtime_error if HDF5 I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Load wavefunction from HDF5 group. Note that due to significant code
   * duplication in the cas and sci containers, their common logic is shared in
   * this base class, and not re-implemented in the cas and sci containers.
   * @param group HDF5 group containing wavefunction data
   * @return Shared pointer to Wavefunction object created from HDF5 group
   * @throws std::runtime_error if HDF5 data is malformed or I/O error occurs
   */
  static std::shared_ptr<Wavefunction> from_hdf5(H5::Group& group);

  /**
   * @brief Save wavefunction to HDF5 file
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load wavefunction from HDF5 file
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to Wavefunction object created from HDF5 file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Wavefunction> from_hdf5_file(
      const std::string& filename);

  /**
   * @brief Save wavefunction to file in specified format
   * @param filename Path to file to create/overwrite
   * @param type Format type ("json" or "hdf5")
   * @throws std::invalid_argument if unknown type
   * @throws std::runtime_error if I/O error occurs
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Load wavefunction from file in specified format
   * @param filename Path to file to read
   * @param type Format type ("json" or "hdf5")
   * @return Shared pointer to Wavefunction object created from file
   * @throws std::invalid_argument if unknown type
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Wavefunction> from_file(const std::string& filename,
                                                 const std::string& type);

  /**
   * @brief Get the wavefunction type (SelfDual or NotSelfDual)
   * @return WavefunctionType enum value
   */
  WavefunctionType get_type() const;

  /**
   * @brief Check if the wavefunction is complex-valued
   * @return True if complex, false if real
   */
  bool is_complex() const;

 private:
  /// Container holding the wavefunction implementation
  std::unique_ptr<const WavefunctionContainer> _container;

  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /**
   * @brief Clear cached data to release memory
   *
   * This private method delegates to the underlying WavefunctionContainer's
   * clear_caches() method to clean up memoized evaluations of derived data.
   * This includes cached reduced density matrices (RDMs) and other computed
   * properties specific to each container implementation.
   */
  void _clear_caches() const;

  /**
   * @brief Save to JSON file without filename validation (internal use)
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_json_file(const std::string& filename) const;

  /**
   * @brief Save to HDF5 file without filename validation (internal use)
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_hdf5_file(const std::string& filename) const;

  /**
   * @brief Load from JSON file without filename validation (internal use)
   * @param filename Path to JSON file to read
   * @return Shared pointer to Wavefunction object created from JSON file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Wavefunction> _from_json_file(
      const std::string& filename);

  /**
   * @brief Load from HDF5 file without filename validation (internal use)
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to Wavefunction object created from HDF5 file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Wavefunction> _from_hdf5_file(
      const std::string& filename);
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(
    DataClassCompliant<Wavefunction>,
    "Wavefunction must derive from DataClass and implement all required "
    "deserialization methods");
}  // namespace qdk::chemistry::data

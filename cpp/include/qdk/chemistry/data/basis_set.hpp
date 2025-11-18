// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <initializer_list>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

// Forward declaration
class Structure;

/// @brief Maximum angular momentum for basis functions supported in
/// QDK/Chemistry
inline static constexpr size_t MAX_ORBITAL_ANGULAR_MOMENTUM =
    6;  // Up to i-orbitals

/**
 * @enum OrbitalType
 * @brief Enumeration for different types of atomic orbitals
 */
enum class OrbitalType {
  UL = -1,  ///< ECP local potential (l=-1)
  S = 0,    ///< S orbital (angular momentum l=0)
  P = 1,    ///< P orbital (angular momentum l=1)
  D = 2,    ///< D orbital (angular momentum l=2)
  F = 3,    ///< F orbital (angular momentum l=3)
  G = 4,    ///< G orbital (angular momentum l=4)
  H = 5,    ///< H orbital (angular momentum l=5)
  I = 6     ///< I orbital (angular momentum l=6)
};

/**
 * @enum BasisType
 * @brief Enumeration for basis function types (spherical vs cartesian)
 */
enum class BasisType {
  Spherical,  ///< Spherical harmonics (2l+1 functions per shell)
  Cartesian   ///< Cartesian coordinates (more functions for l>=2)
};

/**
 * @struct Shell
 * @brief Information about a shell of basis functions
 *
 * A shell represents a group of basis functions that share the same atom,
 * angular momentum, and primitive functions, but differ in magnetic quantum
 * numbers. For example, a p-shell contains px, py, pz functions.
 *
 * Primitive data is stored as raw vectors instead of Primitive objects
 * for better performance and simpler data handling.
 *
 * By convention, the coefficients are stored as the raw, unnormalized
 * contraction coefficients for the primitives. This convention is adopted
 * to facilitate compatibility with various quantum chemistry software
 * packages and libraries, which often use raw coefficients in their basis set
 * definitions. The normalization of these coefficients is typically handled
 * during the computation of integrals or other operations, rather than being
 * stored in the basis set itself.
 */
struct Shell {
  size_t atom_index = 0ul;  ///< Index of the atom this shell belongs to
  OrbitalType orbital_type =
      OrbitalType::S;            ///< Type of orbital (s, p, d, f, etc.)
  Eigen::VectorXd exponents;     ///< Orbital exponents for primitive Gaussians
  Eigen::VectorXd coefficients;  ///< Contraction coefficients for primitives
  Eigen::VectorXi rpowers;       ///< Radial powers for ECP shells (r^n terms)

  /**
   * @brief Constructor with primitive data
   */
  Shell(size_t atom_idx, OrbitalType orb_type, const Eigen::VectorXd& exp,
        const Eigen::VectorXd& coeff)
      : atom_index(atom_idx),
        orbital_type(orb_type),
        exponents(exp),
        coefficients(coeff),
        rpowers(Eigen::VectorXi::Zero(0)) {
    if (exponents.size() != coefficients.size()) {
      throw std::invalid_argument(
          "Exponents and coefficients must have the same size");
    }
  }

  /**
   * @brief Constructor with primitive data and radial powers (for ECP shells)
   */
  Shell(size_t atom_idx, OrbitalType orb_type, const Eigen::VectorXd& exp,
        const Eigen::VectorXd& coeff, const Eigen::VectorXi& rpow)
      : atom_index(atom_idx),
        orbital_type(orb_type),
        exponents(exp),
        coefficients(coeff),
        rpowers(rpow) {
    if (exponents.size() != coefficients.size()) {
      throw std::invalid_argument(
          "Exponents and coefficients must have the same size");
    }
    if (rpowers.size() > 0 && rpowers.size() != exponents.size()) {
      throw std::invalid_argument(
          "Radial powers must have the same size as exponents and "
          "coefficients");
    }
  }

  /**
   * @brief Constructor with vectors for primitives
   */
  Shell(size_t atom_idx, OrbitalType orb_type,
        const std::vector<double>& exp_list,
        const std::vector<double>& coeff_list);

  /**
   * @brief Constructor with vectors for primitives and radial powers (for ECP
   * shells)
   */
  Shell(size_t atom_idx, OrbitalType orb_type,
        const std::vector<double>& exp_list,
        const std::vector<double>& coeff_list,
        const std::vector<int>& rpow_list);

  /**
   * @brief Get number of primitives in this shell
   */
  size_t get_num_primitives() const { return exponents.size(); }

  /**
   * @brief Check if this shell has radial powers (i.e., is an ECP shell)
   */
  bool has_radial_powers() const { return rpowers.size() > 0; }

  /**
   * @brief Get number of basis functions in this shell
   * @param basis_type Whether to use spherical or cartesian basis functions
   */
  size_t get_num_basis_functions(
      BasisType basis_type = BasisType::Spherical) const {
    int l = static_cast<int>(orbital_type);
    if (basis_type == BasisType::Spherical) {
      return 2 * l + 1;  // Spherical harmonics: 2l+1
    } else {
      // Cartesian: (l+1)(l+2)/2
      return (l + 1) * (l + 2) / 2;
    }
  }

  /**
   * @brief Get angular momentum quantum number
   */
  int get_angular_momentum() const { return static_cast<int>(orbital_type); }
};

/**
 * @class BasisSet
 * @brief Represents an atomic orbital basis set using shell-based organization
 *
 * This class stores and manages atomic orbital basis set information using
 * shells as the primary organizational unit. A shell represents a group of
 * basis functions with the same atom, angular momentum, and primitives.
 * The class is designed to be immutable after construction, ensuring data
 * integrity.
 *
 * Features:
 * - Shell-based storage for memory efficiency
 * - Support for spherical or cartesian basis functions
 * - Mapping between shells/basis functions and atoms
 * - Mapping between shells/basis functions and orbital types
 * - Basis set metadata (name, parameters, references)
 * - Integration with molecular structure information
 * - On-demand expansion of shells to individual basis functions
 *
 * The shell-based approach matches standard quantum chemistry practices
 * and provides efficient storage and computation.
 */
class BasisSet : public DataClass,
                 public std::enable_shared_from_this<BasisSet> {
 public:
  /**
   * @brief Constructor with basis set name and structure
   * @param name Name of the basis set (e.g., "6-31G", "cc-pVDZ")
   * @param structure The molecular structure
   * @param basis_type Whether to use spherical or cartesian basis functions
   */
  BasisSet(const std::string& name, const Structure& structure,
           BasisType basis_type = BasisType::Spherical);

  /**
   * @brief Constructor with shells
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param basis_type Whether to use spherical or cartesian basis functions
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           BasisType basis_type = BasisType::Spherical);

  /**
   * @brief Constructor with shells and structure
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param structure The molecular structure
   * @param basis_type Whether to use spherical or cartesian basis functions
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           const Structure& structure,
           BasisType basis_type = BasisType::Spherical);

  /**
   * @brief Constructor with basis set name and structure shared pointer
   * @param name Name of the basis set (e.g., "6-31G", "cc-pVDZ")
   * @param structure Shared pointer to the molecular structure
   * @param basis_type Whether to use spherical or cartesian basis functions
   */
  BasisSet(const std::string& name, std::shared_ptr<Structure> structure,
           BasisType basis_type = BasisType::Spherical);

  /**
   * @brief Constructor with shells and structure shared pointer
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param structure Shared pointer to the molecular structure
   * @param basis_type Whether to use spherical or cartesian basis functions
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           std::shared_ptr<Structure> structure,
           BasisType basis_type = BasisType::Spherical);

  /**
   * @brief Constructor with shells, ECP shells, and structure
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param ecp_shells Vector of ECP shells to initialize the basis set with
   * @param structure The molecular structure
   * @param basis_type Whether to use spherical or cartesian basis functions
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           const std::vector<Shell>& ecp_shells, const Structure& structure,
           BasisType basis_type = BasisType::Spherical);

  /**
   * @brief Constructor with shells, ECP shells, and structure shared pointer
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param ecp_shells Vector of ECP shells to initialize the basis set with
   * @param structure Shared pointer to the molecular structure
   * @param basis_type Whether to use spherical or cartesian basis functions
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           const std::vector<Shell>& ecp_shells,
           std::shared_ptr<Structure> structure,
           BasisType basis_type = BasisType::Spherical);

  /**
   * @brief Constructor with shells, ECP shells, ECP name, ECP electrons, and
   * structure
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param ecp_name Name of the ECP basis set
   * @param ecp_shells Vector of ECP shells to initialize the basis set with
   * @param ecp_electrons Vector containing numbers of ECP electrons for each
   * atom
   * @param structure The molecular structure
   * @param basis_type Whether to use spherical or cartesian basis functions
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           const std::string& ecp_name, const std::vector<Shell>& ecp_shells,
           const std::vector<size_t>& ecp_electrons, const Structure& structure,
           BasisType basis_type = BasisType::Spherical);

  /**
   * @brief Constructor with shells, ECP shells, ECP name, ECP electrons, and
   * structure shared pointer
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param ecp_shells Vector of ECP shells to initialize the basis set with
   * @param ecp_name Name of the ECP basis set
   * @param ecp_electrons Vector containing numbers of ECP electrons for each
   * atom
   * @param structure Shared pointer to the molecular structure
   * @param basis_type Whether to use spherical or cartesian basis functions
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           const std::string& ecp_name, const std::vector<Shell>& ecp_shells,
           const std::vector<size_t>& ecp_electrons,
           std::shared_ptr<Structure> structure,
           BasisType basis_type = BasisType::Spherical);

  /**
   * @brief Default destructor
   */
  virtual ~BasisSet() = default;

  /**
   * @brief Copy constructor
   *
   * Note: this function generates a deep copy of the basis set.
   */
  BasisSet(const BasisSet& other);

  /**
   * @brief Move constructor
   */
  BasisSet(BasisSet&& other) noexcept = default;

  /**
   * @brief Copy assignment operator
   *
   * Note: this function generates a deep copy of the basis set.
   */
  BasisSet& operator=(const BasisSet& other);

  /**
   * @brief Move assignment operator
   */
  BasisSet& operator=(BasisSet&& other) noexcept = default;

  /**
   * @brief Get the basis type
   * @return Current basis type (spherical or cartesian)
   */
  BasisType get_basis_type() const;

  /**
   * @brief Get all shells (flattened from per-atom storage)
   * @return Vector of all shells
   */
  std::vector<Shell> get_shells() const;

  /**
   * @brief Get shells for a specific atom
   * @param atom_index Index of the atom
   * @return Vector of shells for this atom
   */
  const std::vector<Shell>& get_shells_for_atom(size_t atom_index) const;

  /**
   * @brief Get a specific shell by global index
   * @param shell_index Global index of the shell
   * @return Reference to the shell
   * @throws std::out_of_range if index is invalid
   */
  const Shell& get_shell(size_t shell_index) const;

  /**
   * @brief Get total number of shells across all atoms
   * @return Total number of shells
   */
  size_t get_num_shells() const;

  /**
   * @brief Get number of atoms that have shells
   * @return Number of atoms with shells
   */
  size_t get_num_atoms() const;

  /**
   * @brief Get all ECP shells (flattened from per-atom storage)
   * @return Vector of all ECP shells
   */
  std::vector<Shell> get_ecp_shells() const;

  /**
   * @brief Get ECP shells for a specific atom
   * @param atom_index Index of the atom
   * @return Vector of ECP shells for this atom
   */
  const std::vector<Shell>& get_ecp_shells_for_atom(size_t atom_index) const;

  /**
   * @brief Get a specific ECP shell by global index
   * @param shell_index Global index of the ECP shell
   * @return Reference to the ECP shell
   * @throws std::out_of_range if index is invalid
   */
  const Shell& get_ecp_shell(size_t shell_index) const;

  /**
   * @brief Get total number of ECP shells across all atoms
   * @return Total number of ECP shells
   */
  size_t get_num_ecp_shells() const;

  /**
   * @brief Check if this basis set has ECP shells
   * @return True if there are any ECP shells
   */
  bool has_ecp_shells() const;

  /**
   * @brief Get the shell index and magnetic quantum number for a basis function
   * index
   * @param basis_index Index of the basis function
   * @return Pair containing (shell_index, magnetic_quantum_number)
   * @throws std::out_of_range if index is invalid
   */
  std::pair<size_t, int> get_basis_function_info(size_t basis_index) const;

  /**
   * @brief Get number of basis functions (total from all shells)
   * @return Total number of basis functions
   */
  size_t get_num_basis_functions() const;

  /**
   * @brief Get the atom index for a basis function
   * @param contracted_basis_function_index Index of the basis function
   * @return Index of the atom this basis function belongs to
   * @throws std::out_of_range if basis_index is invalid
   */
  size_t get_atom_index_for_basis_function(
      size_t contracted_basis_function_index) const;

  /**
   * @brief Get all basis function indices for a specific atom
   * @param atom_index Index of the atom
   * @return Vector of basis function indices for this atom
   */
  std::vector<size_t> get_basis_function_indices_for_atom(
      size_t atom_index) const;

  /**
   * @brief Get shell indices for a specific atom
   * @param atom_index Index of the atom
   * @return Vector of shell indices for this atom
   */
  std::vector<size_t> get_shell_indices_for_atom(size_t atom_index) const;

  /**
   * @brief Get number of basis functions for a specific atom
   * @param atom_index Index of the atom
   * @return Number of basis functions for this atom
   */
  size_t get_num_basis_functions_for_atom(size_t atom_index) const;

  /**
   * @brief Get shell indices for a specific orbital type
   * @param orbital_type Type of orbital
   * @return Vector of shell indices of this type
   */
  std::vector<size_t> get_shell_indices_for_orbital_type(
      OrbitalType orbital_type) const;

  /**
   * @brief Get number of basis functions for a specific orbital type
   * @param orbital_type Type of orbital
   * @return Number of basis functions of this type
   */
  size_t get_num_basis_functions_for_orbital_type(
      OrbitalType orbital_type) const;

  /**
   * @brief Get shell indices for a specific atom and orbital type
   * @param atom_index Index of the atom
   * @param orbital_type Type of orbital
   * @return Vector of shell indices matching both criteria
   */
  std::vector<size_t> get_shell_indices_for_atom_and_orbital_type(
      size_t atom_index, OrbitalType orbital_type) const;

  /**
   * @brief Get ECP shell indices for a specific atom
   * @param atom_index Index of the atom
   * @return Vector of ECP shell indices for this atom
   */
  std::vector<size_t> get_ecp_shell_indices_for_atom(size_t atom_index) const;

  /**
   * @brief Get ECP shell indices for a specific orbital type
   * @param orbital_type Type of orbital
   * @return Vector of ECP shell indices of this type
   */
  std::vector<size_t> get_ecp_shell_indices_for_orbital_type(
      OrbitalType orbital_type) const;

  /**
   * @brief Get ECP shell indices for a specific atom and orbital type
   * @param atom_index Index of the atom
   * @param orbital_type Type of orbital
   * @return Vector of ECP shell indices matching both criteria
   */
  std::vector<size_t> get_ecp_shell_indices_for_atom_and_orbital_type(
      size_t atom_index, OrbitalType orbital_type) const;

  /**
   * @brief Get the basis set name
   * @return Name of the basis set
   */
  const std::string& get_name() const;
  ;

  /**
   * @brief Get the molecular structure
   * @return Pointer to the molecular structure
   * @throws std::runtime_error if no structure is set
   */
  const std::shared_ptr<Structure> get_structure() const;

  /**
   * @brief Check if a structure is associated with this basis set
   * @return True if structure is set
   */
  bool has_structure() const;

  /**
   * @brief Get the ECP name
   * @return Name of the ECP
   */
  const std::string& get_ecp_name() const;

  /**
   * @brief Get the ECP electrons vector
   * @return Vector containing numbers of ECP electrons for each atom
   */
  const std::vector<size_t>& get_ecp_electrons() const;

  /**
   * @brief Check if ECP electrons are present
   * @return True if ECP electrons are present
   */
  bool has_ecp_electrons() const;

  /**
   * @brief Get summary string of basis set information
   * @return String describing the basis set
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
   * @brief Convert basis set to JSON
   * @return JSON object containing basis set data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save basis set to JSON file (with validation)
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Serialize basis set to HDF5 group
   * @param group HDF5 group to write data to
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save basis set to HDF5 file (with validation)
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Generic file I/O - load from file based on type parameter
   * @param filename Path to file to read
   * @param type File format type ("json" or "hdf5")
   * @return New BasisSet instance loaded from file
   * @throws std::runtime_error if file doesn't exist, unsupported type, or I/O
   * error occurs
   */
  static std::shared_ptr<BasisSet> from_file(const std::string& filename,
                                             const std::string& type);

  /**
   * @brief Load basis set from HDF5 file (with validation)
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to const BasisSet instance loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<BasisSet> from_hdf5_file(const std::string& filename);

  /**
   * @brief Load basis set from HDF5 group
   * @param group HDF5 group to read data from
   * @return Shared pointer to const BasisSet instance loaded from group
   * @throws std::runtime_error if I/O error occurs
   */
  static std::shared_ptr<BasisSet> from_hdf5(H5::Group& group);

  /**
   * @brief Load basis set from JSON file (with validation)
   * @param filename Path to JSON file to read
   * @return Shared pointer to const BasisSet instance loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<BasisSet> from_json_file(const std::string& filename);

  /**
   * @brief Load basis set from JSON
   * @param j JSON object containing basis set data
   * @return Shared pointer to const BasisSet instance loaded from JSON
   * @throws std::runtime_error if JSON is malformed
   */
  static std::shared_ptr<BasisSet> from_json(const nlohmann::json& j);

  /**
   * @brief Convert orbital type to string
   * @param orbital_type Type of orbital
   * @return String representation (e.g., "s", "p", "d")
   */
  static std::string orbital_type_to_string(OrbitalType orbital_type);

  /**
   * @brief Convert an integer orbital angular momentum quantum number to
   * orbital type
   * @param l Angular momentum quantum number
   * @return Orbital type enumeration
   * @throws std::invalid_argument if l is out of range
   */
  static OrbitalType l_to_orbital_type(int l);

  /**
   * @brief Convert string to orbital type
   * @param orbital_string String representation (e.g., "s", "p", "d")
   * @return Orbital type enumeration
   * @throws std::invalid_argument if string is invalid
   */
  static OrbitalType string_to_orbital_type(const std::string& orbital_string);

  /**
   * @brief Get angular momentum quantum number for orbital type
   * @param orbital_type Type of orbital
   * @return Angular momentum quantum number (l)
   */
  static int get_angular_momentum(OrbitalType orbital_type);

  /**
   * @brief Get number of orbitals for given angular momentum
   * @param l Angular momentum quantum number
   * @param basis_type Whether to use spherical or cartesian basis functions
   * @return Number of orbitals
   */
  static int get_num_orbitals_for_l(
      int l, BasisType basis_type = BasisType::Spherical);

  inline static int get_orbital_size(OrbitalType orbital_type) {
    return get_num_orbitals_for_l(static_cast<int>(orbital_type));
  }

  /**
   * @brief Convert basis function index to shell index and magnetic quantum
   * number
   * @param basis_index Global basis function index
   * @return Pair of (shell_index, magnetic_quantum_number)
   */
  std::pair<size_t, int> basis_to_shell_index(size_t basis_index) const;

  /**
   * @brief Convert basis type to string
   * @param basis_type The basis type to convert
   * @return String representation ("spherical" or "cartesian")
   */
  static std::string basis_type_to_string(BasisType basis_type);

  /**
   * @brief Convert string to basis type
   * @param basis_string String representation ("spherical" or "cartesian")
   * @return Basis type enumeration
   * @throws std::invalid_argument if string is invalid
   */
  static BasisType string_to_basis_type(const std::string& basis_string);

 private:
  /// Basis set name (e.g., "6-31G", "cc-pVDZ")
  std::string _name;

  /// Basis type (spherical or cartesian)
  BasisType _basis_type;

  /// Molecular structure associated with this basis set
  std::shared_ptr<Structure> _structure;

  /// Shells organized by atom index - each atom has a vector of shells
  std::vector<std::vector<Shell>> _shells_per_atom;

  /// ECP shells organized by atom index - each atom has a vector of ECP shells
  std::vector<std::vector<Shell>> _ecp_shells_per_atom;

  /// Effective Core Potential (ECP) name (basis set name)
  std::string _ecp_name;

  /// Number of ECP electrons replaced for each atom
  std::vector<size_t> _ecp_electrons;

  /// Lazily computed cache for basis function to atom mapping
  mutable std::vector<size_t> _basis_to_atom_map;

  /// Lazily computed cache for basis function to shell mapping
  mutable std::vector<size_t> _basis_to_shell_map;

  /// Lazily computed cache for the total number of basis functions
  mutable size_t _cached_num_basis_functions = 0;

  /// Lazily computed cache for the total number of shells
  mutable size_t _cached_num_shells = 0;

  /// Flag to track if cached data is valid
  mutable bool _cache_valid = false;

  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /**
   * @brief Check if basis set is valid and complete
   * @return True if basis set is valid
   */
  bool _is_valid() const;

  /**
   * @brief Validate consistency with the associated molecular structure
   * @return True if basis set is consistent with its structure
   */
  bool _is_consistent_with_structure() const;

  /**
   * @brief Clear internal mapping structures and invalidate cache
   */
  void _clear_maps();

  /**
   * @brief Compute and cache mapping data if not already cached
   */
  void _compute_mappings() const;

  /**
   * @brief Validate basis function index
   * @param basis_index Index to validate
   * @throws std::out_of_range if index is invalid
   */
  void _validate_basis_index(size_t basis_index) const;

  /**
   * @brief Validate shell index
   * @param shell_index Index to validate
   * @throws std::out_of_range if index is invalid
   */
  void _validate_shell_index(size_t shell_index) const;

  /**
   * @brief Validate atom index
   * @param atom_index Index to validate
   * @throws std::out_of_range if index is invalid
   */
  void _validate_atom_index(size_t atom_index) const;

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
   * @return Shared pointer to const BasisSet instance loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<BasisSet> _from_json_file(const std::string& filename);

  /**
   * @brief Load from HDF5 file without filename validation (internal use)
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to const BasisSet instance loaded from file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<BasisSet> _from_hdf5_file(const std::string& filename);
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(DataClassCompliant<BasisSet>,
              "BasisSet must derive from DataClass and implement all required "
              "deserialization methods");

}  // namespace qdk::chemistry::data

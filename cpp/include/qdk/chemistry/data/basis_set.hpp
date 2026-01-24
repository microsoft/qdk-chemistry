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
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {
// Forward declaration
class Structure;

/// @brief Maximum angular momentum for atomic orbitals supported in
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
 * @enum AOType
 * @brief Enumeration for atomic orbital types (spherical vs cartesian)
 */
enum class AOType {
  Spherical,  ///< Spherical harmonics (2l+1 functions per shell)
  Cartesian   ///< Cartesian coordinates (more functions for l>=2)
};

/**
 * @struct Shell
 * @brief Information about a shell of atomic orbitals
 *
 * A shell represents a group of atomic orbitals that share the same atom,
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
   * @brief Get number of atomic orbitals in this shell
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   */
  size_t get_num_atomic_orbitals(
      AOType atomic_orbital_type = AOType::Spherical) const {
    int l = static_cast<int>(orbital_type);
    if (atomic_orbital_type == AOType::Spherical) {
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
 * atomic orbitals with the same atom, angular momentum, and primitives.
 * The class is designed to be immutable after construction, ensuring data
 * integrity.
 *
 * Features:
 * - Shell-based storage for memory efficiency
 * - Support for spherical or cartesian atomic orbitals
 * - Mapping between shells/atomic orbitals and atoms
 * - Mapping between shells/atomic orbitals and orbital types
 * - Basis set metadata (name, parameters, references)
 * - Integration with molecular structure information
 * - On-demand expansion of shells to individual atomic orbitals
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
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   */
  BasisSet(const std::string& name, const Structure& structure,
           AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Constructor with shells
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Constructor with shells and structure
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param structure The molecular structure
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           const Structure& structure,
           AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Constructor with basis set name and structure shared pointer
   * @param name Name of the basis set (e.g., "6-31G", "cc-pVDZ")
   * @param structure Shared pointer to the molecular structure
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   */
  BasisSet(const std::string& name, std::shared_ptr<Structure> structure,
           AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Constructor with shells and structure shared pointer
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param structure Shared pointer to the molecular structure
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           std::shared_ptr<Structure> structure,
           AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Constructor with shells, ECP shells, and structure
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param ecp_shells Vector of ECP shells to initialize the basis set with
   * @param structure The molecular structure
   * @param basis_type Whether to use spherical or cartesian atomic orbitals
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           const std::vector<Shell>& ecp_shells, const Structure& structure,
           AOType basis_type = AOType::Spherical);

  /**
   * @brief Constructor with shells, ECP shells, and structure shared pointer
   * @param name Name of the basis set
   * @param shells Vector of shells to initialize the basis set with
   * @param ecp_shells Vector of ECP shells to initialize the basis set with
   * @param structure Shared pointer to the molecular structure
   * @param basis_type Whether to use spherical or cartesian atomic orbitals
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           const std::vector<Shell>& ecp_shells,
           std::shared_ptr<Structure> structure,
           AOType basis_type = AOType::Spherical);

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
   * @param basis_type Whether to use spherical or cartesian atomic orbitals
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           const std::string& ecp_name, const std::vector<Shell>& ecp_shells,
           const std::vector<size_t>& ecp_electrons, const Structure& structure,
           AOType basis_type = AOType::Spherical);

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
   * @param basis_type Whether to use spherical or cartesian atomic orbitals
   */
  BasisSet(const std::string& name, const std::vector<Shell>& shells,
           const std::string& ecp_name, const std::vector<Shell>& ecp_shells,
           const std::vector<size_t>& ecp_electrons,
           std::shared_ptr<Structure> structure,
           AOType basis_type = AOType::Spherical);

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

  /** @brief Name for custom basis sets */
  static constexpr std::string_view custom_name = "custom_basis_set";

  /** @brief Name for custom ecps */
  static constexpr std::string_view custom_ecp_name = "custom_ecp";

  /**
   * @brief Get the data type name for this class
   * @return "basis_set"
   */
  std::string get_data_type_name() const override {
    return DATACLASS_TO_SNAKE_CASE(BasisSet);
  }
  /**
   * @brief Get supported basis set names
   * @return Vector of supported basis set names
   */
  static std::vector<std::string> get_supported_basis_set_names();

  /**
   * @brief Get supported elements for a given basis set
   * @param basis_name Name of the basis set
   * @return Vector of supported elements as enum
   */
  static std::vector<Element> get_supported_elements_for_basis_set(
      std::string basis_name);

  /**
   * @brief Constructor with basis set name and structure
   * @param basis_name Name of the basis set (e.g., "6-31G", "cc-pVDZ")
   * @param structure The molecular structure
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   * @return Shared pointer to the created BasisSet
   */
  static std::shared_ptr<BasisSet> from_basis_name(
      const std::string& basis_name, const Structure& structure,
      AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Constructor with basis set name and structure
   * @param basis_name Name of the basis set (e.g., "6-31G", "cc-pVDZ")
   * @param structure Shared pointer to the molecular structure
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   * @return Shared pointer to the created BasisSet
   */
  static std::shared_ptr<BasisSet> from_basis_name(
      std::string basis_name, std::shared_ptr<Structure> structure,
      AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Constructor with basis set name and structure
   * @param element_to_basis_map Mapping from element symbols to basis set names
   * @param structure The molecular structure
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   * @return Shared pointer to the created BasisSet
   */
  static std::shared_ptr<BasisSet> from_element_map(
      const std::map<std::string, std::string>& element_to_basis_map,
      const Structure& structure,
      AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Constructor with basis set name and structure
   * @param element_to_basis_map Mapping from element symbols to basis set names
   * @param structure Shared pointer to the molecular structure
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   * @return Shared pointer to the created BasisSet
   */
  static std::shared_ptr<BasisSet> from_element_map(
      const std::map<std::string, std::string>& element_to_basis_map,
      std::shared_ptr<Structure> structure,
      AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Constructor with basis set name and structure
   * @param index_to_basis_map Mapping from atom indices (as strings) to basis
   * set names
   * @param structure The molecular structure
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   * @return Shared pointer to the created BasisSet
   */
  static std::shared_ptr<BasisSet> from_index_map(
      const std::map<size_t, std::string>& index_to_basis_map,
      const Structure& structure,
      AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Constructor with basis set name and structure
   * @param index_to_basis_map Mapping from atom indices (as strings) to basis
   * set names
   * @param structure Shared pointer to the molecular structure
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   * @return Shared pointer to the created BasisSet
   */
  static std::shared_ptr<BasisSet> from_index_map(
      const std::map<size_t, std::string>& index_to_basis_map,
      std::shared_ptr<Structure> structure,
      AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Get the basis type
   * @return Current basis type (spherical or cartesian)
   */
  AOType get_atomic_orbital_type() const;

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
   * @brief Get the shell index and magnetic quantum number for a atomic orbital
   * index
   * @param atomic_orbital_index Index of the atomic orbital
   * @return Pair containing (shell_index, magnetic_quantum_number)
   * @throws std::out_of_range if index is invalid
   */
  std::pair<size_t, int> get_atomic_orbital_info(
      size_t atomic_orbital_index) const;

  /**
   * @brief Get number of atomic orbitals (total from all shells)
   * @return Total number of atomic orbitals
   */
  size_t get_num_atomic_orbitals() const;

  /**
   * @brief Get the atom index for a atomic orbital
   * @param contracted_atomic_orbital_index Index of the atomic orbital
   * @return Index of the atom this atomic orbital belongs to
   * @throws std::out_of_range if atomic_orbital_index is invalid
   */
  size_t get_atom_index_for_atomic_orbital(
      size_t contracted_atomic_orbital_index) const;

  /**
   * @brief Get all atomic orbital indices for a specific atom
   * @param atom_index Index of the atom
   * @return Vector of atomic orbital indices for this atom
   */
  std::vector<size_t> get_atomic_orbital_indices_for_atom(
      size_t atom_index) const;

  /**
   * @brief Get shell indices for a specific atom
   * @param atom_index Index of the atom
   * @return Vector of shell indices for this atom
   */
  std::vector<size_t> get_shell_indices_for_atom(size_t atom_index) const;

  /**
   * @brief Get number of atomic orbitals for a specific atom
   * @param atom_index Index of the atom
   * @return Number of atomic orbitals for this atom
   */
  size_t get_num_atomic_orbitals_for_atom(size_t atom_index) const;

  /**
   * @brief Get shell indices for a specific orbital type
   * @param orbital_type Type of orbital
   * @return Vector of shell indices of this type
   */
  std::vector<size_t> get_shell_indices_for_orbital_type(
      OrbitalType orbital_type) const;

  /**
   * @brief Get number of atomic orbitals for a specific orbital type
   * @param orbital_type Type of orbital
   * @return Number of atomic orbitals of this type
   */
  size_t get_num_atomic_orbitals_for_orbital_type(
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
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   * orbitals
   * @return Number of orbitals
   */
  static int get_num_orbitals_for_l(
      int l, AOType atomic_orbital_type = AOType::Spherical);

  inline static int get_orbital_size(OrbitalType orbital_type) {
    return get_num_orbitals_for_l(static_cast<int>(orbital_type));
  }

  /**
   * @brief Convert atomic orbital index to shell index and magnetic quantum
   * number
   * @param atomic_orbital_index Global atomic orbital index
   * @return Pair of (shell_index, magnetic_quantum_number)
   */
  std::pair<size_t, int> basis_to_shell_index(
      size_t atomic_orbital_index) const;

  /**
   * @brief Convert basis type to string
   * @param atomic_orbital_type The basis type to convert
   * @return String representation ("spherical" or "cartesian")
   */
  static std::string atomic_orbital_type_to_string(AOType atomic_orbital_type);

  /**
   * @brief Convert string to basis type
   * @param basis_string String representation ("spherical" or "cartesian")
   * @return Basis type enumeration
   * @throws std::invalid_argument if string is invalid
   */
  static AOType string_to_atomic_orbital_type(const std::string& basis_string);

 private:
  /// Basis set name (e.g., "6-31G", "cc-pVDZ")
  std::string _name;

  /// Basis type (spherical or cartesian)
  AOType _atomic_orbital_type;

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

  /// Lazily computed cache for atomic orbital to atom mapping
  mutable std::vector<size_t> _basis_to_atom_map;

  /// Lazily computed cache for atomic orbital to shell mapping
  mutable std::vector<size_t> _basis_to_shell_map;

  /// Lazily computed cache for the total number of atomic orbitals
  mutable size_t _cached_num_atomic_orbitals = 0;

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
   * @brief Validate atomic orbital index
   * @param atomic_orbital_index Index to validate
   * @throws std::out_of_range if index is invalid
   */
  void _validate_atomic_orbital_index(size_t atomic_orbital_index) const;

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

namespace detail {
/**
 * @brief Normalize basis set name for filesystem usage.
 * Replaces special characters that are problematic in filenames:
 * - '*' -> '_st_' (star)
 * - '/' -> '_sl_' (slash)
 * - '+' -> '_pl_' (plus)
 * @param name Original basis set name (e.g., "6-31g*+")
 * @return Normalized name safe for filesystem (e.g., "6-31g_st__pl_")
 */
std::string normalize_basis_set_name(const std::string& name);

/**
 * @brief Denormalize basis set name from filesystem representation.
 * Reverses the normalization:
 * - '_st_' -> '*'
 * - '_sl_' -> '/'
 * - '_pl_' -> '+'
 * @param normalized Normalized name from filesystem
 * @return Original basis set name
 */
std::string denormalize_basis_set_name(const std::string& normalized);
}  // namespace detail

}  // namespace qdk::chemistry::data

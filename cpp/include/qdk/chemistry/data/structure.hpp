// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @enum Element
 * @brief Enumeration for chemical elements in the periodic table (1-118)
 */
enum class Element : unsigned {
  // Period 1
  H = 1,
  He = 2,
  // Period 2
  Li = 3,
  Be = 4,
  B = 5,
  C = 6,
  N = 7,
  O = 8,
  F = 9,
  Ne = 10,
  // Period 3
  Na = 11,
  Mg = 12,
  Al = 13,
  Si = 14,
  P = 15,
  S = 16,
  Cl = 17,
  Ar = 18,
  // Period 4
  K = 19,
  Ca = 20,
  Sc = 21,
  Ti = 22,
  V = 23,
  Cr = 24,
  Mn = 25,
  Fe = 26,
  Co = 27,
  Ni = 28,
  Cu = 29,
  Zn = 30,
  Ga = 31,
  Ge = 32,
  As = 33,
  Se = 34,
  Br = 35,
  Kr = 36,
  // Period 5
  Rb = 37,
  Sr = 38,
  Y = 39,
  Zr = 40,
  Nb = 41,
  Mo = 42,
  Tc = 43,
  Ru = 44,
  Rh = 45,
  Pd = 46,
  Ag = 47,
  Cd = 48,
  In = 49,
  Sn = 50,
  Sb = 51,
  Te = 52,
  I = 53,
  Xe = 54,
  // Period 6
  Cs = 55,
  Ba = 56,
  La = 57,
  Ce = 58,
  Pr = 59,
  Nd = 60,
  Pm = 61,
  Sm = 62,
  Eu = 63,
  Gd = 64,
  Tb = 65,
  Dy = 66,
  Ho = 67,
  Er = 68,
  Tm = 69,
  Yb = 70,
  Lu = 71,
  Hf = 72,
  Ta = 73,
  W = 74,
  Re = 75,
  Os = 76,
  Ir = 77,
  Pt = 78,
  Au = 79,
  Hg = 80,
  Tl = 81,
  Pb = 82,
  Bi = 83,
  Po = 84,
  At = 85,
  Rn = 86,
  // Period 7
  Fr = 87,
  Ra = 88,
  Ac = 89,
  Th = 90,
  Pa = 91,
  U = 92,
  Np = 93,
  Pu = 94,
  Am = 95,
  Cm = 96,
  Bk = 97,
  Cf = 98,
  Es = 99,
  Fm = 100,
  Md = 101,
  No = 102,
  Lr = 103,
  Rf = 104,
  Db = 105,
  Sg = 106,
  Bh = 107,
  Hs = 108,
  Mt = 109,
  Ds = 110,
  Rg = 111,
  Cn = 112,
  Nh = 113,
  Fl = 114,
  Mc = 115,
  Lv = 116,
  Ts = 117,
  Og = 118
};

/**
 * @class Structure
 * @brief Represents a molecular structure with atomic coordinates, elements,
 * masses, and nuclear charges
 *
 * This class stores and manipulates molecular structure data including:
 * - Atomic coordinates in 3D space
 * - Atomic element identifiers using enum
 * - Atomic masses (in atomic mass units)
 * - Nuclear charges (atomic numbers) for each atom
 * - Serialization to/from JSON and XYZ formats
 * - Basic geometric operations and validation
 *
 * The structure can be constructed from various input formats and provides
 * convenient access to atomic properties and molecular geometry.
 * Standard atomic masses and nuclear charges are used by default unless
 * otherwise specified.
 * The class is designed to be immutable after construction, ensuring data
 * integrity.
 *
 * Units and conventions
 * ---------------------
 * - Internal storage: atomic coordinates are stored internally in Bohr.
 * - Getters/Setters: get_* and set_* coordinate APIs operate in Bohr.
 * - Serialization: JSON read/write uses Bohr for coordinates.
 * - XYZ I/O: XYZ files are read/written in Angstrom (according to their
 *   standard) and converted at the API boundary.
 */
class Structure : public DataClass,
                  public std::enable_shared_from_this<Structure> {
 public:
  /**
   * @brief Constructor with coordinates, elements, masses, and nuclear charges
   * @param coordinates Matrix of atomic coordinates (N x 3) in Bohr
   * @param elements Vector of atomic elements using enum
   * @param masses Vector of atomic masses in AMU (default: use standard masses)
   * @param nuclear_charges Vector of nuclear charges (default: use standard
   * charges)
   * @throws std::invalid_argument if dimensions don't match
   */
  Structure(const Eigen::MatrixXd& coordinates,
            const std::vector<Element>& elements,
            const Eigen::VectorXd& masses = {},
            const Eigen::VectorXd& nuclear_charges = {});

  /**
   * @brief Constructor from atomic symbols and coordinates
   * @param coordinates Matrix of atomic coordinates (N x 3) in Bohr
   * @param symbols Vector of atomic symbols (e.g., "H", "C", "O")
   * @param masses Vector of atomic masses in AMU (default: use standard masses)
   * @param nuclear_charges Vector of nuclear charges (default: use standard
   * charges)
   * @throws std::invalid_argument if dimensions don't match or unknown symbols
   */
  Structure(const Eigen::MatrixXd& coordinates,
            const std::vector<std::string>& symbols,
            const Eigen::VectorXd& masses = {},
            const Eigen::VectorXd& nuclear_charges = {});

  /**
   * @brief Constructor from atomic symbols and coordinates as vector
   * @param coordinates Vector of atomic coordinates (N x 3) in Bohr
   * @param symbols Vector of atomic symbols (e.g., "H", "C", "O")
   * @param masses Vector of atomic masses in AMU (default: use standard masses)
   * @param nuclear_charges Vector of nuclear charges (default: use standard
   * charges)
   * @throws std::invalid_argument if dimensions don't match or unknown symbols
   */
  Structure(const std::vector<Eigen::Vector3d>& coordinates,
            const std::vector<std::string>& symbols,
            const std::vector<double>& masses = {},
            const std::vector<double>& nuclear_charges = {});

  /**
   * @brief Constructor from atomic elements and coordinates as vector
   * @param coordinates Vector of atomic coordinates (N x 3) in Bohr
   * @param elements Vector of atomic elements using enum
   * @param masses Vector of atomic masses in AMU (default: use standard masses)
   * @param nuclear_charges Vector of nuclear charges (default: use standard
   * charges)
   * @throws std::invalid_argument if dimensions don't match
   */
  Structure(const std::vector<Eigen::Vector3d>& coordinates,
            const std::vector<Element>& elements,
            const std::vector<double>& masses = {},
            const std::vector<double>& nuclear_charges = {});

  /**
   * @brief Copy constructor
   */
  Structure(const Structure& other) = default;

  /**
   * @brief Move constructor
   */
  Structure(Structure&& other) noexcept = default;

  /**
   * @brief Copy assignment operator
   */
  Structure& operator=(const Structure& other) = default;

  /**
   * @brief Move assignment operator
   */
  Structure& operator=(Structure&& other) noexcept = default;

  /**
   * @brief Destructor
   */
  virtual ~Structure() = default;

  /**
   * @brief Get the atomic coordinates matrix
   * @return Matrix of coordinates (N x 3) in Bohr
   */
  const Eigen::MatrixXd& get_coordinates() const { return _coordinates; }

  /**
   * @brief Get the atomic elements vector
   * @return Vector of atomic elements
   */
  const std::vector<Element>& get_elements() const { return _elements; }

  /**
   * @brief Get the atomic masses vector
   * @return Vector of atomic masses in AMU
   */
  const Eigen::VectorXd& get_masses() const { return _masses; }

  /**
   * @brief Get the nuclear charges vector
   * @return Vector of nuclear charges (atomic numbers)
   */
  const Eigen::VectorXd& get_nuclear_charges() const {
    return _nuclear_charges;
  }

  /**
   * @brief Get coordinates for a specific atom
   * @param atom_index Index of the atom (0-based)
   * @return 3D coordinates as Eigen::Vector3d
   * @throws std::out_of_range if atom_index is invalid
   */
  Eigen::Vector3d get_atom_coordinates(size_t atom_index) const;

  /**
   * @brief Get element for a specific atom
   * @param atom_index Index of the atom (0-based)
   * @return Atomic element enum
   * @throws std::out_of_range if atom_index is invalid
   */
  Element get_atom_element(size_t atom_index) const;

  /**
   * @brief Get mass for a specific atom
   * @param atom_index Index of the atom (0-based)
   * @return Atomic mass in AMU
   * @throws std::out_of_range if atom_index is invalid
   */
  double get_atom_mass(size_t atom_index) const;

  /**
   * @brief Get nuclear charge for a specific atom
   * @param atom_index Index of the atom (0-based)
   * @return Nuclear charge (atomic number)
   * @throws std::out_of_range if atom_index is invalid
   */
  double get_atom_nuclear_charge(size_t atom_index) const;

  /**
   * @brief Get atomic symbol for a specific atom
   * @param atom_index Index of the atom (0-based)
   * @return Atomic symbol (e.g., "H", "C", "O")
   * @throws std::out_of_range if atom_index is invalid
   */
  std::string get_atom_symbol(size_t atom_index) const;

  /**
   * @brief Get all atomic symbols
   * @return Vector of atomic symbols
   */
  std::vector<std::string> get_atomic_symbols() const;

  /**
   * @brief Get the number of atoms in the structure
   * @return Number of atoms
   */
  size_t get_num_atoms() const { return _elements.size(); }

  /**
   * @brief Check if the structure is empty
   * @return True if no atoms are present
   */
  bool is_empty() const { return _elements.empty(); }

  /**
   * @brief Calculate the total molecular mass
   * @return Total mass in AMU
   */
  double get_total_mass() const;

  /**
   * @brief Get summary string of structure information
   * @return String describing the structure
   */
  std::string get_summary() const override;

  /**
   * @brief Convert structure to JSON format
   * @return JSON object containing structure data with coordinates in Bohr
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Load structure from JSON format
   * @param j JSON object containing structure data with coordinates in Bohr
   * @return Shared pointer to const Structure object created from JSON data
   * @throws std::runtime_error if JSON is malformed
   */
  static std::shared_ptr<Structure> from_json(const nlohmann::json& j);

  /**
   * @brief Save structure to JSON file
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Load structure from JSON file
   * @param filename Path to JSON file to read
   * @return Shared pointer to const Structure object created from JSON file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Structure> from_json_file(const std::string& filename);

  /**
   * @brief Convert structure to HDF5 group
   * @param group HDF5 group to write structure data to
   * @throws std::runtime_error if HDF5 I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Load structure from HDF5 group
   * @param group HDF5 group containing structure data
   * @return Shared pointer to Structure object created from HDF5 group
   * @throws std::runtime_error if HDF5 data is malformed or I/O error occurs
   */
  static std::shared_ptr<Structure> from_hdf5(H5::Group& group);

  /**
   * @brief Save structure to HDF5 file
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load structure from HDF5 file
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to Structure object created from HDF5 file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Structure> from_hdf5_file(const std::string& filename);

  /**
   * @brief Convert structure to XYZ format string
   * @param comment Optional comment line (default: empty)
   * @return XYZ format string with coordinates in Angstrom
   */
  std::string to_xyz(const std::string& comment = "") const;

  /**
   * @brief Load structure from XYZ format string
   * @param xyz_string XYZ format string with coordinates in Angstrom
   * @return Structure object created from XYZ string
   * @throws std::runtime_error if XYZ format is invalid
   */
  static std::shared_ptr<Structure> from_xyz(const std::string& xyz_string);

  /**
   * @brief Save structure to XYZ file
   * @param filename Path to XYZ file to create/overwrite
   * @param comment Optional comment line (default: empty)
   * @throws std::runtime_error if I/O error occurs
   */
  void to_xyz_file(const std::string& filename,
                   const std::string& comment = "") const;

  /**
   * @brief Load structure from XYZ file
   * @param filename Path to XYZ file to read
   * @return Shared pointer to a Structure object created from XYZ file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Structure> from_xyz_file(const std::string& filename);

  /**
   * @brief Save structure to file in specified format
   * @param filename Path to file to create/overwrite
   * @param type Format type ("json", "xyz", or "hdf5")
   * @throws std::invalid_argument if unknown type
   * @throws std::runtime_error if I/O error occurs
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Load structure from file in specified format
   * @param filename Path to file to create/overwrite
   * @param type Format type ("json", "xyz", or "hdf5")
   * @return Structure object created from file
   * @throws std::invalid_argument if unknown type
   * @throws std::runtime_error if I/O error occurs
   */
  static std::shared_ptr<Structure> from_file(const std::string& filename,
                                              const std::string& type);

  /**
   * @brief Calculate nuclear-nuclear repulsion energy
   * @return Nuclear repulsion energy in atomic units (Hartree)
   *
   * This function calculates the Coulombic repulsion energy between all nuclei
   * in the structure using the formula:
   * E_nn = sum_{i<j} Z_i * Z_j / |R_i - R_j|
   * where Z_i is the nuclear charge of atom i and R_i is its position vector.
   */
  double calculate_nuclear_repulsion_energy() const;

  /**
   * @brief Convert atomic symbol to element enum
   * @param symbol Atomic symbol (e.g., "H", "C", "O")
   * @return Atomic element enum
   * @throws std::invalid_argument if unknown symbol
   */
  static Element symbol_to_element(const std::string& symbol);

  /**
   * @brief Convert element enum to atomic symbol
   * @param element Atomic element enum
   * @return Atomic symbol (e.g., "H", "C", "O")
   */
  static std::string element_to_symbol(Element element);

  /**
   * @brief Convert atomic symbol to nuclear charge
   * @param symbol Atomic symbol (e.g., "H", "C", "O")
   * @return Nuclear charge (atomic number)
   * @throws std::invalid_argument if unknown symbol
   */
  static unsigned symbol_to_nuclear_charge(const std::string& symbol);

  /**
   * @brief Convert nuclear charge to atomic symbol
   * @param nuclear_charge Nuclear charge (atomic number)
   * @return Atomic symbol (e.g., "H", "C", "O")
   * @throws std::invalid_argument if unknown nuclear charge
   */
  static std::string nuclear_charge_to_symbol(unsigned nuclear_charge);

  /**
   * @brief Convert element enum to nuclear charge
   * @param element Atomic element enum
   * @return Nuclear charge (atomic number)
   */
  static unsigned element_to_nuclear_charge(Element element);

  /**
   * @brief Convert nuclear charge to element enum
   * @param nuclear_charge Nuclear charge (atomic number)
   * @return Atomic element enum
   * @throws std::invalid_argument if unknown nuclear charge
   */
  static Element nuclear_charge_to_element(unsigned nuclear_charge);

  /**
   * @brief Get standard atomic mass for an element
   * @param element Atomic element enum
   * @return Standard atomic mass in AMU
   */
  static double get_standard_atomic_mass(Element element);

  /**
   * @brief Get standard nuclear charge for an element
   * @param element Atomic element enum
   * @return Standard nuclear charge (atomic number)
   */
  static unsigned get_standard_nuclear_charge(Element element);

 private:
  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  /**
   * @brief Check if the structure is valid
   * @return True if all data arrays are consistent
   */
  bool _is_valid() const;

  /**
   * @brief Fix capitalization of atomic symbol (e.g., "he" -> "He", "CA" ->
   * "Ca")
   * @param symbol Input atomic symbol with potentially incorrect capitalization
   * @return Properly capitalized atomic symbol
   */
  static std::string _fix_symbol_capitalization(const std::string& symbol);

  /**
   * @brief Private function to save structure to JSON file without validation
   * @param filename Path to JSON file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_json_file(const std::string& filename) const;

  /**
   * @brief Private function to load structure from JSON file without validation
   * @param filename Path to JSON file to read
   * @return Shared pointer to const Structure object created from JSON file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Structure> _from_json_file(
      const std::string& filename);

  /**
   * @brief Private function to save structure to XYZ file without validation
   * @param filename Path to XYZ file to create/overwrite
   * @param comment Optional comment line (default: empty)
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_xyz_file(const std::string& filename,
                    const std::string& comment = "") const;

  /**
   * @brief Private function to load structure from XYZ file without validation
   * @param filename Path to XYZ file to read
   * @return Shared pointer to a Structure object created from XYZ file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Structure> _from_xyz_file(const std::string& filename);

  /**
   * @brief Private function to save structure to HDF5 file without validation
   * @param filename Path to HDF5 file to create/overwrite
   * @throws std::runtime_error if I/O error occurs
   */
  void _to_hdf5_file(const std::string& filename) const;

  /**
   * @brief Private function to load structure from HDF5 file without validation
   * @param filename Path to HDF5 file to read
   * @return Shared pointer to Structure object created from HDF5 file
   * @throws std::runtime_error if file doesn't exist or I/O error occurs
   */
  static std::shared_ptr<Structure> _from_hdf5_file(
      const std::string& filename);

  /// Atomic coordinates matrix (N x 3) stored internally in Bohr
  const Eigen::MatrixXd _coordinates;

  /// Atomic elements for each atom
  const std::vector<Element> _elements;

  /// Atomic masses for each atom in AMU
  const Eigen::VectorXd _masses;

  /// Nuclear charges (atomic numbers) for each atom
  const Eigen::VectorXd _nuclear_charges;

  /**
   * @brief Validate that all data arrays have consistent dimensions
   * @throws std::invalid_argument if inconsistent
   */
  void _validate_dimensions() const;
};

// Enforce inheritance from base class and presence of required methods.
// This checks the presence of key methods (serialization, deserialization) and
// get_summary.
static_assert(DataClassCompliant<Structure>,
              "Structure must derive from DataClass and implement all required "
              "deserialization methods");
}  // namespace qdk::chemistry::data

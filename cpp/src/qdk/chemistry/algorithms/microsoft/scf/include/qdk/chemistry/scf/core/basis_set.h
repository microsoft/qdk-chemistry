// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/basis_func.h>
#include <qdk/chemistry/scf/core/molecule.h>

#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdk::chemistry::scf {
/**
 * @brief Basis set modes for different quantum chemistry packages
 */
enum BasisMode { PSI4, RAW };

/**
 * @brief Represents a single shell in a basis set
 *
 * Contains angular momentum, contraction information, and coefficients
 * for a shell centered on a particular atom.
 */
struct Shell {
  uint64_t atom_index;        ///< Index of the atom this shell is centered on
  std::array<double, 3> O;    ///< Cartesian coordinates of the shell center
  uint64_t angular_momentum;  ///< Angular momentum quantum number
  uint64_t contraction;       ///< Number of primitive Gaussians
  double exponents[MAX_CONTRACTION];     ///< Gaussian exponents
  double coefficients[MAX_CONTRACTION];  ///< Contraction coefficients
  int rpowers[MAX_CONTRACTION];  ///< Radial powers for ECP shells (r^n terms)

  /**
   * @brief Constructs a Shell from JSON representation
   *
   * @param rec JSON object containing shell data
   * @param mol Pointer to the molecule
   *
   * @return Shell object
   */
  static Shell from_json(const nlohmann::ordered_json& rec,
                         const std::shared_ptr<Molecule> mol);

  /**
   * @brief Converts the Shell to JSON representation
   *
   * @param is_ecp Whether this is an ECP shell
   *
   * @return JSON object
   */
  nlohmann::ordered_json to_json(const bool& is_ecp = false) const;
};

/**
 * @brief Manages basis set data for quantum chemistry calculations
 */
class BasisSet {
 public:
  std::shared_ptr<Molecule> mol;  ///< Pointer to the molecular structure
  std::string name;           ///< Basis set name (e.g., "6-31G", "def2-TZVP")
  BasisMode mode;             ///< Normalization convention (PSI4 or RAW)
  std::vector<Shell> shells;  ///< All orbital atomic orbital shells
  bool pure;  ///< Whether to use pure spherical harmonics or Cartesian
  uint64_t num_atomic_orbitals;  ///< Total number of atomic orbitals

  std::vector<Shell> ecp_shells;  ///< Effective core potential shells
  std::unordered_map<int, int>
      element_ecp_electrons;  ///< Map from atomic number to ECP electron number
  int n_ecp_electrons = 0;  ///< Total number of core electrons replaced by ECPs

  /**
   * @brief Load basis set from database (QCSchema) JSON file
   *
   * @param mol Molecular structure
   * @param path Path to basis set file or basis set name
   * @param mode Normalization convention to use
   * @param pure Whether to use spherical harmonics
   * @param sort Whether to sort shells by angular momentum
   *
   * @return Unique pointer to constructed BasisSet
   */
  static std::shared_ptr<BasisSet> from_database_json(
      std::shared_ptr<Molecule> mol, const std::string& path, BasisMode mode,
      bool pure, bool sort = true);

  /**
   * @brief Load basis set from serialized (internal schema) JSON file
   *
   * @param mol Molecular structure
   * @param path Path to serialized basis set JSON file
   *
   * @return Unique pointer to constructed BasisSet
   */
  static std::shared_ptr<BasisSet> from_serialized_json(
      std::shared_ptr<Molecule> mol, std::string path);

  /**
   * @brief Load basis set from serialized (internal schema) JSON data
   *
   * @param mol Molecular structure
   * @param json JSON data containing basis set information
   *
   * @return Unique pointer to constructed BasisSet
   */
  static std::shared_ptr<BasisSet> from_serialized_json(
      std::shared_ptr<Molecule> mol, const nlohmann::ordered_json& json);

  /**
   * @brief Get atom-to-atomic-orbital mapping (const version)
   * @return Vector indicating which atomic orbitals belong to which atoms
   * @throws std::runtime_error if atom2ao data is not available
   */
  const std::vector<uint8_t>& get_atom2ao() const {
    if (atom2ao_.empty()) {
      throw std::runtime_error(
          "atom2ao data not available. Call non-const get_atom2ao() first to "
          "compute it.");
    }
    return atom2ao_;
  }

  /**
   * @brief Get atom-to-atomic-orbital mapping (non-const version)
   * @return Vector indicating which atomic orbitals belong to which atoms
   */
  const std::vector<uint8_t>& get_atom2ao() {
    if (atom2ao_.empty()) {
      calc_atom2ao();
    }
    return atom2ao_;
  }

  /**
   * @brief Get significant shell pairs for integral screening (const version)
   * @return Vector of shell index pairs (i, j) with non-negligible overlap
   * @throws std::runtime_error if shell_pairs data is not available
   */
  const std::vector<std::pair<int, int>>& get_shell_pairs() const;

  /**
   * @brief Get significant shell pairs for integral screening (non-const
   * version)
   * @return Vector of shell index pairs (i, j) with non-negligible overlap
   */
  const std::vector<std::pair<int, int>>& get_shell_pairs();

  /**
   * @brief Get the maximum angular momentum in the basis set
   * @return Maximum angular momentum quantum number
   */
  inline unsigned max_angular_momentum() const {
    return std::max_element(shells.begin(), shells.end(),
                            [](auto& a, auto& b) {
                              return a.angular_momentum < b.angular_momentum;
                            })
        ->angular_momentum;
  }

  /**
   * @brief Convert basis set to JSON representation
   * @return Ordered JSON object containing all basis set data
   */
  nlohmann::ordered_json to_json() const;

  /**
   * @brief Constructor for basis set from shells
   *
   * @param mol Molecular structure
   * @param input_shells Input shells
   * @param mode Normalization convention
   * @param pure Whether to use spherical harmonics
   * @param sort Whether to sort shells
   */
  explicit BasisSet(std::shared_ptr<Molecule> mol,
                    const std::vector<Shell>& input_shells, BasisMode mode,
                    bool pure, bool sort);
  /**
   * @brief Constructor for basis set from shells with ECPs
   * @param mol Molecular structure
   * @param input_shells Orbital atomic orbital shells
   * @param ecp_shells Effective core potential shells
   * @param element_ecp_electrons Map from atomic number to ECP electron number
   * @param n_ecp_electrons Total number of core electrons replaced by ECPs
   * @param mode Normalization convention
   * @param pure Whether to use spherical harmonics
   * @param sort Whether to sort shells
   */
  explicit BasisSet(std::shared_ptr<Molecule> mol,
                    const std::vector<Shell>& input_shells,
                    const std::vector<Shell>& ecp_shells,
                    const std::unordered_map<int, int>& element_ecp_electrons,
                    int n_ecp_electrons, BasisMode mode, bool pure, bool sort);

 private:
  /**
   * @brief Private constructor for basis set from database
   *
   * @param mol Molecular structure
   * @param path Path to basis set file
   * @param mode Normalization convention
   * @param pure Whether to use spherical harmonics
   * @param sort Whether to sort shells
   */
  explicit BasisSet(std::shared_ptr<Molecule> mol, const std::string& path,
                    BasisMode mode, bool pure, bool sort);

  /**
   * @brief Default private constructor
   */
  explicit BasisSet();

  std::vector<std::pair<int, int>>
      shell_pairs_;               ///< Significant shell pairs for screening
  std::vector<uint8_t> atom2ao_;  ///< Atom-to-atomic-orbital mapping matrix
                                  ///< (n_atoms × num_atomic_orbitals)

  /**
   * @brief Calculate atom-to-atomic-orbital mapping
   */
  void calc_atom2ao();
};
}  // namespace qdk::chemistry::scf

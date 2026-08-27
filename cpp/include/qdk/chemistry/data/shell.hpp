// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <cstddef>
#include <nlohmann/json_fwd.hpp>
#include <stdexcept>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

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
   * @param atom_idx Index of the atom on which the shell is centered
   * @param orb_type Shell angular momentum
   * @param exp Primitive Gaussian exponents
   * @param coeff Primitive contraction coefficients
   * @throws std::invalid_argument if exponent and coefficient counts differ
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
   * @param atom_idx Index of the atom on which the shell is centered
   * @param orb_type Shell angular momentum or ECP local-potential type
   * @param exp Primitive Gaussian exponents
   * @param coeff Primitive contraction coefficients
   * @param rpow Radial powers for ECP primitives
   * @throws std::invalid_argument if primitive-array lengths are inconsistent
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
   * @param atom_idx Index of the atom on which the shell is centered
   * @param orb_type Shell angular momentum
   * @param exp_list Primitive Gaussian exponents
   * @param coeff_list Primitive contraction coefficients
   * @throws std::invalid_argument if exponent and coefficient counts differ
   */
  Shell(size_t atom_idx, OrbitalType orb_type,
        const std::vector<double>& exp_list,
        const std::vector<double>& coeff_list);

  /**
   * @brief Constructor with vectors for primitives and radial powers (for ECP
   * shells)
   * @param atom_idx Index of the atom on which the shell is centered
   * @param orb_type Shell angular momentum or ECP local-potential type
   * @param exp_list Primitive Gaussian exponents
   * @param coeff_list Primitive contraction coefficients
   * @param rpow_list Radial powers for ECP primitives
   * @throws std::invalid_argument if primitive-array lengths are inconsistent
   */
  Shell(size_t atom_idx, OrbitalType orb_type,
        const std::vector<double>& exp_list,
        const std::vector<double>& coeff_list,
        const std::vector<int>& rpow_list);

  /**
   * @brief Get number of primitives in this shell
   * @return Number of exponent/coefficient pairs
   */
  size_t get_num_primitives() const { return exponents.size(); }

  /**
   * @brief Check if this shell has radial powers (i.e., is an ECP shell)
   * @return Whether radial powers are stored
   */
  bool has_radial_powers() const { return rpowers.size() > 0; }

  /**
   * @brief Get number of atomic orbitals in this shell
   * @param atomic_orbital_type Whether to use spherical or cartesian atomic
   *        orbitals
   * @return Number of basis functions contributed by the shell
   */
  size_t get_num_atomic_orbitals(
      AOType atomic_orbital_type = AOType::Spherical) const {
    int l = static_cast<int>(orbital_type);
    if (atomic_orbital_type == AOType::Spherical) {
      return 2 * l + 1;
    }
    return (l + 1) * (l + 2) / 2;
  }

  /**
   * @brief Get angular momentum quantum number
   * @return Angular momentum @f$l@f$, or -1 for the ECP local potential
   */
  int get_angular_momentum() const { return static_cast<int>(orbital_type); }

  /**
   * @brief Serialize this shell.
   * @param include_radial_powers Whether to include ECP radial powers
   */
  nlohmann::json to_json(bool include_radial_powers = false) const;

  /**
   * @brief Deserialize a shell.
   * @param json Serialized shell data
   * @param atom_index Index of the atom on which the shell is centered
   * @param allow_radial_powers Whether ECP radial powers are permitted
   */
  static Shell from_json(const nlohmann::json& json, size_t atom_index,
                         bool allow_radial_powers = false);
};

/**
 * @brief Convert an orbital type to its canonical lowercase name.
 * @param orbital_type Orbital type to convert
 * @return @c ul, @c s, @c p, @c d, @c f, @c g, @c h, @c i, or @c unknown
 */
std::string orbital_type_to_string(OrbitalType orbital_type);

/**
 * @brief Convert angular momentum to an orbital type.
 * @param l Angular momentum, from -1 through
 *        @ref MAX_ORBITAL_ANGULAR_MOMENTUM
 * @return Corresponding orbital type
 * @throws std::invalid_argument if @p l is unsupported
 */
OrbitalType l_to_orbital_type(int l);

/**
 * @brief Parse an orbital type name case-insensitively.
 * @param orbital_string Orbital name
 * @return Corresponding orbital type
 * @throws std::invalid_argument if the name is unknown
 */
OrbitalType string_to_orbital_type(const std::string& orbital_string);

/**
 * @brief Convert an atomic-orbital representation to its canonical name.
 * @param atomic_orbital_type Representation to convert
 * @return @c spherical, @c cartesian, or @c unknown
 */
std::string atomic_orbital_type_to_string(AOType atomic_orbital_type);

/**
 * @brief Parse an atomic-orbital representation case-insensitively.
 * @param basis_string @c spherical, @c sph, @c cartesian, or @c cart
 * @return Parsed atomic-orbital representation
 * @throws std::invalid_argument if the name is unknown
 */
AOType string_to_atomic_orbital_type(const std::string& basis_string);

/// @brief Shells grouped by the index of the atom they are centered on
using ShellsPerAtom = std::vector<std::vector<Shell>>;

/**
 * @brief Sort shells into canonical order, in place
 *
 * Shells are ordered by angular momentum and then by decreasing largest
 * exponent. Within each shell the primitives are ordered by decreasing
 * exponent, keeping exponents, coefficients and radial powers aligned.
 * @param shells Shells to sort
 */
void sort_shells_inplace(std::vector<Shell>& shells);

/**
 * @brief Group shells by their owning atom
 *
 * @param shells Shells to group
 * @param num_atoms Number of atoms; the result always has this many entries
 * @param shell_description Noun used when reporting an out-of-range atom index
 * @return Per-atom shell vectors with exactly @p num_atoms entries
 * @throws std::invalid_argument if a shell references an atom outside the range
 */
ShellsPerAtom group_shells_by_atom(
    const std::vector<Shell>& shells, size_t num_atoms,
    const std::string& shell_description = "Shell");

/**
 * @brief Flatten per-atom shells back into a single ordered vector.
 * @param shells_per_atom Per-atom shell vectors
 * @return Flattened shells in atom order
 */
std::vector<Shell> flatten_shells(const ShellsPerAtom& shells_per_atom);

/**
 * @brief Count shells across all atoms.
 * @param shells_per_atom Per-atom shell vectors
 * @return Total shell count
 */
size_t count_shells(const ShellsPerAtom& shells_per_atom);

/**
 * @brief Count atomic orbitals across all atoms.
 * @param shells_per_atom Per-atom shell vectors
 * @param atomic_orbital_type Spherical or Cartesian representation
 * @return Total orbital count
 */
size_t count_orbitals(const ShellsPerAtom& shells_per_atom,
                      AOType atomic_orbital_type = AOType::Spherical);

}  // namespace qdk::chemistry::data

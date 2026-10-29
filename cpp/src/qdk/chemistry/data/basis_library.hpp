// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <qdk/chemistry/data/shell.hpp>
#include <string>
#include <tuple>
#include <vector>

// Shared shell utilities and access to the bundled basis set library.
namespace qdk::chemistry::data::detail {

/// @brief Shells grouped by the index of the atom they are centered on
using ShellsPerAtom = std::vector<std::vector<Shell>>;

/**
 * @brief Sort shells into canonical order, in place
 *
 * Shells are ordered by angular momentum and then by decreasing largest
 * exponent. Within each shell the primitives are ordered by decreasing
 * exponent, keeping exponents, coefficients and radial powers aligned.
 * Primitives with equal exponents retain their relative order.
 * @param shells Shells to sort; an empty vector is allowed
 * @throws std::invalid_argument if any shell has no primitives
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

/**
 * @brief Normalize a basis set name for filesystem usage.
 * Replaces '*' with "_st_", '/' with "_sl_", and '+' with "_pl_".
 */
std::string normalize_basis_set_name(const std::string& name);

/** @brief Inverse of @ref normalize_basis_set_name. */
std::string denormalize_basis_set_name(const std::string& normalized);

/** @brief Lowercase a basis set name for case-insensitive library lookup. */
std::string lowercase_basis_name(std::string name);

/**
 * @brief Read one atom's shells from the basis set library.
 * @param nuclear_charge Nuclear charge of the element
 * @param basis_set_name Name of the basis set to read
 * @param atom_index Atom the returned shells are centered on
 * @return Primary shells, ECP shells, and the number of replaced electrons
 *
 * Primary shells have no radial powers; ECP shells retain the radial powers
 * specified by the library definition.
 */
std::tuple<std::vector<Shell>, std::vector<Shell>, size_t>
get_basis_for_nuclear_charge(double nuclear_charge, std::string basis_set_name,
                             size_t atom_index);

}  // namespace qdk::chemistry::data::detail

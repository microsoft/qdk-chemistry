// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <qdk/chemistry/data/shell.hpp>
#include <string>
#include <tuple>
#include <vector>

// Access to the bundled basis set library, shared by every basis class.
namespace qdk::chemistry::data::detail {

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
 */
std::tuple<std::vector<Shell>, std::vector<Shell>, size_t>
get_basis_for_nuclear_charge(double nuclear_charge, std::string basis_set_name,
                             size_t atom_index);

}  // namespace qdk::chemistry::data::detail

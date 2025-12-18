// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/molecule.h>

#include <libint2.hpp>  // for Shell class
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/structure.hpp>

namespace qdk::chemistry::utils::microsoft {

namespace qcs = qdk::chemistry::scf;

/**
 * @brief Normalize shells for PSI4 mode
 *
 * @param shells Vector of shells to be normalized for PSI4 mode
 * @note This transformation happens in place. For correct results, the shells
 * should be initialized with *raw* coefficients (i.e. those obtained from the
 * BasisSetExchange).
 */
void _norm_psi4_mode(std::vector<qcs::Shell>& shells);

/**
 * @brief Initialize the backend for internal algorithms
 * This function sets up the necessary environment for the internal
 * algorithms to run. It should be called before any internal algorithm
 * is executed to ensure that all dependencies and configurations are properly
 * initialized.
 */
void initialize_backend();

/**
 * @brief Finalize the backend for internal algorithms
 * This function cleans up resources and finalizes the environment used by
 * internal algorithms. It should be called after all internal algorithms
 * have been executed to ensure proper resource management.
 */
void finalize_backend();

/**
 * @brief Convert a Molecule to a qdk::chemistry::data::Structure
 *
 * This function takes a Molecule object and converts it into a
 * qdk::chemistry::data::Structure by extracting atomic coordinates and nuclear
 * charges.
 *
 * @param molecule The Molecule object to convert.
 * @return A qdk::chemistry::data::Structure representing the same molecular
 * data.
 */
qdk::chemistry::data::Structure convert_to_structure(
    const qcs::Molecule& molecule);

/**
 * @brief Convert a qdk::chemistry::data::Structure to a Molecule
 *
 * This function takes a qdk::chemistry::data::Structure and converts it into a
 * Molecule object by extracting atomic coordinates and nuclear charges.
 *
 * @param structure The Structure to convert.
 * @param charge The total charge of the molecule.
 * @param multiplicity The spin multiplicity of the molecule.
 * @return A Molecule object representing the same molecular data.
 */
std::shared_ptr<qcs::Molecule> convert_to_molecule(
    const qdk::chemistry::data::Structure& structure, int64_t charge,
    int64_t multiplicity);

/**
 * @brief Convert a qdk::chemistry::data::BasisSet to JSON format
 *
 * This function converts a qdk::chemistry::data::BasisSet object into a JSON
 * format that can be easily serialized and used in other applications.
 *
 * @param basis_set The qdk::chemistry::data::BasisSet object to convert.
 * @return A nlohmann::ordered_json object representing the basis set.
 */
nlohmann::ordered_json convert_to_json(
    const qdk::chemistry::data::BasisSet& basis_set);

/**
 * @brief Convert a qdk::chemistry::data::Shell to JSON format
 *
 * This function converts a qdk::chemistry::data::Shell object into a JSON
 * format that can be easily serialized and used in other applications.
 *
 * @param shell The qdk::chemistry::data::Shell object to convert.
 * @return A nlohmann::ordered_json object representing the shell.
 */
nlohmann::ordered_json convert_to_json(
    const qdk::chemistry::data::Shell& shell);

/**
 * @brief Convert a BasisSet from the internal library to a
 * qdk::chemistry::data::BasisSet
 *
 * This function converts a BasisSet object from the internal library into
 * a qdk::chemistry::data::BasisSet, ensuring that it is compatible with the
 * QDK framework.
 *
 * @param basis_set The BasisSet object to convert.
 * @return A qdk::chemistry::data::BasisSet representing the same basis set
 * data.
 */
qdk::chemistry::data::BasisSet convert_basis_set_to_qdk(
    const qcs::BasisSet& basis_set);

/**
 * @brief Convert a qdk::chemistry::data::BasisSet to the internal library
 * BasisSet
 *
 * This function converts a qdk::chemistry::data::BasisSet object into a
 * BasisSet compatible with the internal library, ensuring proper integration
 * with internal algorithms.
 *
 * @param qdk_basis_set The qdk::chemistry::data::BasisSet object to convert.
 * @param  normalize Whether to normalize the basis set after conversion.
 * Default is true.
 * @return A std::unique_ptr<BasisSet> representing the same basis set
 * data.
 * @throws std::runtime_error If the basis set is not spherical(pure)
 */
std::shared_ptr<qcs::BasisSet> convert_basis_set_from_qdk(
    const qdk::chemistry::data::BasisSet& qdk_basis_set, bool normalize = true);

/**
 * @brief Compute a mapping between QDK and internal basis set shells
 *
 * @param qdk_basis_set The QDK representation of the basis set.
 * @param itrn_basis_set The internal representation of the basis set.
 * @return A vector of indices where each element at index i maps the i-th QDK
 * shell to the corresponding index in the internal representation.
 * @throws std::runtime_error If the number of shells differs between
 * representations or if a one-to-one mapping cannot be established.
 */
std::vector<unsigned> compute_shell_map(
    const qdk::chemistry::data::BasisSet& qdk_basis_set,
    const qcs::BasisSet& itrn_basis_set);

}  // namespace qdk::chemistry::utils::microsoft

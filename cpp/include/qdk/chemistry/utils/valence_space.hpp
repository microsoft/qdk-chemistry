// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <vector>

namespace qdk::chemistry::utils {

/**
 * @brief Computes the default number of valence orbitals and electrons for a
 * given structure and wavefunction
 *
 * This function analyzes the provided molecular structure and wavefunction to
 * determine the number of electrons and orbitals in valence shell
 *
 * @param wavefunction Shared pointer to the wavefunction containing electronic
 *                     structure information (structure is extracted from
 *                     wavefunction->orbitals->basis_set)
 * @param charge The total charge of the molecular system, which should be equal
 *               to the charge set in scf calculation
 * @param include_double_d_shell If true, add 5 correlating d' orbitals to the
 *                               valence space for d-block elements (Sc-Zn,
 *                               Y-Cd, Hf-Hg) to capture the strong nd /
 *                               (n+1)d' radial correlation in transition
 *                               metals (the "double d-shell" effect). Defaults
 *                               to false. This option mirrors the
 *                               include_double_d_shell setting on
 *                               ValenceActiveSpaceSettings and should be kept
 *                               consistent with it when the computed values
 *                               are passed to a ValenceActiveSpaceSelector.
 *
 * @return A pair where:
 *         - first: Number of valence electrons
 *         - second: Number of valence orbitals
 */
std::pair<size_t, size_t> compute_valence_space_parameters(
    std::shared_ptr<qdk::chemistry::data::Wavefunction> wavefunction,
    int charge, bool include_double_d_shell = false);

}  // namespace qdk::chemistry::utils

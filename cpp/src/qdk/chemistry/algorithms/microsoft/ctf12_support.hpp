// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>

#include "ctf12_f12.hpp"

namespace qdk::chemistry::algorithms::microsoft::ctf12 {

/**
 * @brief Build a closed-shell F12-HF input from a reference wavefunction.
 *
 * Extracts the orbital basis, molecular-orbital coefficients, orbital energies
 * and nuclear data from a restricted single-determinant reference and builds
 * the complementary auxiliary basis set (CABS). The CABS auxiliary basis is the
 * named @p cabs_basis, or @c "<orbital-basis>-optri" when @p cabs_basis is
 * empty.
 *
 * @param reference The restricted reference wavefunction.
 * @param gamma The Slater geminal exponent (atomic units).
 * @param cabs_basis The named OptRI/CABS auxiliary basis (empty to derive one).
 * @param frozen_core The number of frozen core orbitals (formulation a).
 * @return The F12-HF reference description.
 * @throws std::invalid_argument if the reference is not a closed-shell
 *         single-determinant wavefunction with an associated basis set.
 */
F12HartreeFockInput f12_input_from_wavefunction(
    const data::Wavefunction& reference, double gamma,
    const std::string& cabs_basis, std::size_t frozen_core);

}  // namespace qdk::chemistry::algorithms::microsoft::ctf12

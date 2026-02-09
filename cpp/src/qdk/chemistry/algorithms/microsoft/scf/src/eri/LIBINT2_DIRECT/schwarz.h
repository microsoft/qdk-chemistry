// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/types.h>

#include <libint2.hpp>

namespace qdk::chemistry::scf::libint2::direct {

/**
 * @brief Compute Schwarz integral bounds for screening
 *
 * Calculates the Schwarz inequality bounds (μν|μν)^{1/2} for all shell pairs,
 * which provide upper bounds for two-electron integrals. These bounds are used
 * for efficient integral screening: if (μν|λσ) ≤ K(μν) × K(λσ), then the
 * integral can be neglected if this product is below the required precision.
 *
 * @param obs Libint2 orbital basis set
 * @param use_2norm Whether to use 2-norm (true) or infinity norm (false)
 * @return Matrix of Schwarz bounds K(μν) = sqrt(|(μν|μν)|)
 *
 * @note The returned matrix is symmetric: K(μν) = K(νμ)
 * @note 2-norm typically provides tighter bounds but is more expensive
 */
RowMajorMatrix compute_schwarz_ints(const ::libint2::BasisSet& obs,
                                    bool use_2norm = true);

}  // namespace qdk::chemistry::scf::libint2::direct

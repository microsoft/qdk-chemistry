// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <array>
#include <cmath>

namespace qdk::chemistry::data::detail {

/**
 * @brief Compute the single-orbital von Neumann entropy from diagonal RDM
 * elements.
 *
 * This internal helper is shared by Wavefunction entropy evaluation and
 * algorithms that optimize the same entropy convention.
 *
 * @param occ_alpha Alpha-spin occupation of the spatial orbital.
 * @param occ_beta Beta-spin occupation of the spatial orbital.
 * @param double_occ Alpha-beta double occupation of the spatial orbital.
 * @return The single-orbital entropy using natural logarithms.
 */
inline double single_orbital_entropy(double occ_alpha, double occ_beta,
                                     double double_occ) {
  const double empty = 1.0 - occ_alpha - occ_beta + double_occ;
  const std::array<double, 4> weights{empty, occ_alpha - double_occ,
                                      occ_beta - double_occ, double_occ};

  double entropy = 0.0;
  for (const double weight : weights) {
    if (weight > 0.0) {
      entropy -= weight * std::log(weight);
    }
  }
  return entropy;
}

}  // namespace qdk::chemistry::data::detail

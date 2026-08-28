// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <functional>
#include <utility>

namespace qdk::chemistry::utils {

/**
 * @brief Refine a bracketed scalar minimum to an absolute argument tolerance.
 *
 * @param objective Scalar function to minimize.
 * @param lower_bound Inclusive lower end of the bracketing interval.
 * @param upper_bound Inclusive upper end of the bracketing interval.
 * @param argument_tolerance Maximum final interval width.
 * @return The best sampled argument and its objective value.
 *
 * @throws std::invalid_argument if the bounds or tolerance are not finite, the
 * interval is not ordered, or the tolerance is not positive.
 *
 * @note Golden-section contraction is used rather than a Brent-style
 * minimizer because an objective built from absolute values has minima at
 * cusps, where a relative, machine-epsilon-dependent stopping rule leaves
 * platform-dependent residuals. Contraction stops early if the interval
 * reaches the spacing of its own floating-point representation, so a
 * tolerance finer than that terminates rather than spinning.
 * @note The bracket is assumed to hold one minimum; with several, the
 * returned point is not guaranteed to beat the sampling that placed the
 * bracket, so callers that sample first should keep the better of the two.
 */
std::pair<double, double> golden_section_minimum(
    const std::function<double(double)>& objective, double lower_bound,
    double upper_bound, double argument_tolerance = 1e-13);

}  // namespace qdk::chemistry::utils

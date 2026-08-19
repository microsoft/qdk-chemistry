// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "golden_section.hpp"

#include <cmath>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

namespace qdk::chemistry::utils {

std::pair<double, double> golden_section_minimum(
    const std::function<double(double)>& objective, double lower_bound,
    double upper_bound, double argument_tolerance) {
  QDK_LOG_TRACE_ENTERING();
  if (upper_bound <= lower_bound) {
    throw std::invalid_argument("upper_bound must be greater than lower_bound");
  }
  if (argument_tolerance <= 0.0) {
    throw std::invalid_argument("argument_tolerance must be positive");
  }

  const double inverse_golden_ratio = (std::sqrt(5.0) - 1.0) / 2.0;
  double left = lower_bound;
  double right = upper_bound;
  double inner_left = right - inverse_golden_ratio * (right - left);
  double inner_right = left + inverse_golden_ratio * (right - left);
  double value_left = objective(inner_left);
  double value_right = objective(inner_right);

  while (right - left > argument_tolerance) {
    const double previous_width = right - left;
    if (value_left <= value_right) {
      right = inner_right;
      inner_right = inner_left;
      value_right = value_left;
      inner_left = right - inverse_golden_ratio * (right - left);
      value_left = objective(inner_left);
    } else {
      left = inner_left;
      inner_left = inner_right;
      value_left = value_right;
      inner_right = left + inverse_golden_ratio * (right - left);
      value_right = objective(inner_right);
    }
    // A tolerance below the spacing of the bracket's own floating-point
    // representation is unreachable: the interval stops contracting while it
    // is still wider than the request.
    if (right - left >= previous_width) {
      break;
    }
  }

  const double midpoint = (left + right) / 2.0;
  std::pair<double, double> best{inner_left, value_left};
  for (const std::pair<double, double>& candidate :
       {std::pair<double, double>{inner_right, value_right},
        std::pair<double, double>{midpoint, objective(midpoint)}}) {
    if (candidate.second < best.second ||
        (candidate.second == best.second && candidate.first < best.first)) {
      best = candidate;
    }
  }
  return best;
}

}  // namespace qdk::chemistry::utils

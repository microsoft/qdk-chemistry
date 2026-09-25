// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "iterative_localizer_base.hpp"

#include <limits>
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::algorithms::microsoft {

IterativeOrbitalLocalizationSettings::IterativeOrbitalLocalizationSettings() {
  set_default("tolerance", 1e-6);
  set_default("max_iterations", 10000,
              "Maximum allowed number of localization iterations",
              qdk::chemistry::data::BoundConstraint<int64_t>{
                  2, std::numeric_limits<int64_t>::max()});
  set_default("small_rotation_tolerance", 1e-12);
}

IterativeOrbitalLocalizationScheme::IterativeOrbitalLocalizationScheme(
    IterativeOrbitalLocalizationSettings settings)
    : settings_(settings) {
  QDK_LOG_TRACE_ENTERING();
}

}  // namespace qdk::chemistry::algorithms::microsoft

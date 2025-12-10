// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "iterative_localizer_base.hpp"

#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::algorithms::microsoft {

IterativeOrbitalLocalizationScheme::IterativeOrbitalLocalizationScheme(
    IterativeOrbitalLocalizationSettings settings)
    : settings_(settings) {
  QDK_LOG_TRACE_ENTERING();
}

}  // namespace qdk::chemistry::algorithms::microsoft

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "microsoft/localization/mp2_natural_orbitals.hpp"
#include "microsoft/localization/pipek_mezey.hpp"
#include "microsoft/localization/vvhv.hpp"

namespace qdk::chemistry::algorithms {

std::unique_ptr<Localizer> make_pipek_mezey_localizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::PipekMezeyLocalizer>();
}

std::unique_ptr<Localizer> make_mp2_natural_orbital_localizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::MP2NaturalOrbitalLocalizer>();
}

std::unique_ptr<Localizer> make_vvhv_localizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::VVHVLocalizer>();
}

void LocalizerFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  LocalizerFactory::register_instance(&make_pipek_mezey_localizer);
  LocalizerFactory::register_instance(&make_mp2_natural_orbital_localizer);
  LocalizerFactory::register_instance(&make_vvhv_localizer);
}

}  // namespace qdk::chemistry::algorithms

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "microsoft/macis_asci.hpp"
#include "microsoft/macis_cas.hpp"

namespace qdk::chemistry::algorithms {

std::unique_ptr<MultiConfigurationCalculator> make_macis_cas_mc() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<qdk::chemistry::algorithms::microsoft::MacisCas>();
}
std::unique_ptr<MultiConfigurationCalculator> make_macis_asci_mc() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<qdk::chemistry::algorithms::microsoft::MacisAsci>();
}

void MultiConfigurationCalculatorFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  MultiConfigurationCalculatorFactory::register_instance(&make_macis_cas_mc);
  MultiConfigurationCalculatorFactory::register_instance(&make_macis_asci_mc);
}

}  // namespace qdk::chemistry::algorithms

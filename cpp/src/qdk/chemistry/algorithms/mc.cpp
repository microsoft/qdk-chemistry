// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/data/structure.hpp>

#include "microsoft/macis_asci.hpp"
#include "microsoft/macis_cas.hpp"

namespace qdk::chemistry::algorithms {

std::unique_ptr<MultiConfigurationCalculator> make_macis_cas_mc() {
  return std::make_unique<qdk::chemistry::algorithms::microsoft::MacisCas>();
}
std::unique_ptr<MultiConfigurationCalculator> make_macis_asci_mc() {
  return std::make_unique<qdk::chemistry::algorithms::microsoft::MacisAsci>();
}

void MultiConfigurationCalculatorFactory::register_default_instances() {
  MultiConfigurationCalculatorFactory::register_instance(&make_macis_cas_mc);
  MultiConfigurationCalculatorFactory::register_instance(&make_macis_asci_mc);
}

}  // namespace qdk::chemistry::algorithms

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/pmc.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/data/structure.hpp>

#include "microsoft/macis_pmc.hpp"

namespace qdk::chemistry::algorithms {

std::unique_ptr<ProjectedMultiConfigurationCalculator> make_macis_pmc() {
  return std::make_unique<qdk::chemistry::algorithms::microsoft::MacisPmc>();
}

void ProjectedMultiConfigurationCalculatorFactory::
    register_default_instances() {
  ProjectedMultiConfigurationCalculatorFactory::register_instance(
      &make_macis_pmc);
}

}  // namespace qdk::chemistry::algorithms

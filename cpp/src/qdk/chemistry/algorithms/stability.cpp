// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/stability.hpp"

#include <qdk/chemistry/algorithms/stability.hpp>
#include <qdk/chemistry/config.hpp>

namespace qdk::chemistry::algorithms {

std::unique_ptr<StabilityChecker> make_qdk_stability_checker() {
  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::StabilityChecker>();
}

void StabilityCheckerFactory::register_default_instances() {
  StabilityCheckerFactory::register_instance(&make_qdk_stability_checker);
}

}  // namespace qdk::chemistry::algorithms

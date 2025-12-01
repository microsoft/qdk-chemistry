/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>

#include "microsoft/mp2.hpp"

namespace qdk::chemistry::algorithms {
/**
 * @brief Factory function to create Microsoft MP2 calculator
 */
std::unique_ptr<DynamicalCorrelationCalculator> make_qdk_mp2_calculator() {
  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::MP2Calculator>();
}
/**
 * @brief Register default reference-derived calculator implementations
 */
void DynamicalCorrelationCalculatorFactory::register_default_instances() {
  // Register MP2 algorithms
  DynamicalCorrelationCalculatorFactory::register_instance(
      make_qdk_mp2_calculator);
}

}  // namespace qdk::chemistry::algorithms

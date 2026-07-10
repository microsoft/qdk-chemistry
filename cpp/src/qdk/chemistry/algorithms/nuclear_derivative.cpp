// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/nuclear_derivative.hpp>

#include "finite_difference_nuclear_derivative.hpp"
#include "qdk_nuclear_derivative.hpp"

namespace qdk::chemistry::algorithms {

void NuclearDerivativeCalculatorFactory::register_default_instances() {
  NuclearDerivativeCalculatorFactory::register_instance(
      &make_finite_difference_nuclear_derivative_calculator);
  NuclearDerivativeCalculatorFactory::register_instance(
      &make_qdk_nuclear_derivative_calculator);
}

}  // namespace qdk::chemistry::algorithms

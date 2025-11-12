// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/hamiltonian.hpp"

#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/data/structure.hpp>

namespace qdk::chemistry::algorithms {

std::unique_ptr<HamiltonianConstructor> make_microsoft_hamiltonian() {
  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::HamiltonianConstructor>();
}

void HamiltonianConstructorFactory::register_default_instances() {
  HamiltonianConstructorFactory::register_instance(&make_microsoft_hamiltonian);
}

}  // namespace qdk::chemistry::algorithms

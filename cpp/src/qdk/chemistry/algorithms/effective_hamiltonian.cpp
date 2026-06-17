// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "microsoft/ctf12_hamiltonian.hpp"

namespace qdk::chemistry::algorithms {

std::unique_ptr<EffectiveHamiltonianConstructor>
make_microsoft_ctf12_hamiltonian() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::CtF12HamiltonianConstructor>();
}

void EffectiveHamiltonianConstructorFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  EffectiveHamiltonianConstructorFactory::register_instance(
      &make_microsoft_ctf12_hamiltonian);
}

}  // namespace qdk::chemistry::algorithms

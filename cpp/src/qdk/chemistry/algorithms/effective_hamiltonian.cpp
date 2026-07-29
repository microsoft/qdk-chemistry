// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "microsoft/effective_hamiltonian/swpt2.hpp"

namespace qdk::chemistry::algorithms {

std::unique_ptr<EffectiveHamiltonianConstructor> make_swpt2_constructor() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::SchriefferWolffPT2Constructor>();
}

void EffectiveHamiltonianConstructorFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();
  EffectiveHamiltonianConstructorFactory::register_instance(
      &make_swpt2_constructor);
}

}  // namespace qdk::chemistry::algorithms

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::algorithms {

/**
 * @brief Register default effective-Hamiltonian implementations.
 *
 * There is currently no native C++ effective-Hamiltonian implementation, so no
 * default instance is registered here. The available implementations are
 * supplied by Python plugins, which derive from @ref EffectiveHamiltonian
 * through the pybind11 trampoline and register themselves with the registry.
 */
void EffectiveHamiltonianFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();
}

}  // namespace qdk::chemistry::algorithms

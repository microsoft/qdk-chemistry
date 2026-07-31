// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/effective_hamiltonian.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "microsoft/ducc/ducc.hpp"

namespace qdk::chemistry::algorithms {

/**
 * @brief Factory function to create the BTAS DUCC solver.
 */
std::unique_ptr<EffectiveHamiltonian> make_ducc_solver() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<qdk::chemistry::algorithms::microsoft::DuccSolver>();
}

/**
 * @brief Register default effective-Hamiltonian implementations.
 */
void EffectiveHamiltonianFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();
  EffectiveHamiltonianFactory::register_instance(make_ducc_solver);
}

}  // namespace qdk::chemistry::algorithms

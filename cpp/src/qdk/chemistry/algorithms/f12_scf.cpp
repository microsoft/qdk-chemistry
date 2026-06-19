// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/f12_scf.hpp"

#include <qdk/chemistry/algorithms/f12_scf.hpp>
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::algorithms {

std::unique_ptr<F12HartreeFockSolver>
make_microsoft_ctf12_hartree_fock_solver() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::CtF12HartreeFockSolver>();
}

void F12HartreeFockSolverFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  F12HartreeFockSolverFactory::register_instance(
      &make_microsoft_ctf12_hartree_fock_solver);
}

}  // namespace qdk::chemistry::algorithms

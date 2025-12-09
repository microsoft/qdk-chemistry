// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/scf.hpp"

#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::algorithms {

std::unique_ptr<ScfSolver> make_microsoft_scf_solver() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<qdk::chemistry::algorithms::microsoft::ScfSolver>();
}

void ScfSolverFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  ScfSolverFactory::register_instance(&make_microsoft_scf_solver);
}

}  // namespace qdk::chemistry::algorithms

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/scf.hpp"

#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/config.hpp>

namespace qdk::chemistry::algorithms {

std::unique_ptr<ScfSolver> make_microsoft_scf_solver() {
  return std::make_unique<qdk::chemistry::algorithms::microsoft::ScfSolver>();
}

void ScfSolverFactory::register_default_instances() {
  ScfSolverFactory::register_instance(&make_microsoft_scf_solver);
}

}  // namespace qdk::chemistry::algorithms

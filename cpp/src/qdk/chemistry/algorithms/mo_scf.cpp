// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/mo_scf.hpp"

#include <qdk/chemistry/algorithms/mo_scf.hpp>

namespace qdk::chemistry::algorithms {

void MoScfSolverFactory::register_default_instances() {
  register_instance(
      []() { return std::make_unique<microsoft::MoScfSolver>(); });
}

}  // namespace qdk::chemistry::algorithms

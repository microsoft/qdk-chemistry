// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/scf.hpp"

#include <limits>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "microsoft/stabilized_scf.hpp"

namespace qdk::chemistry::algorithms {

ElectronicStructureSettings::ElectronicStructureSettings() {
  set_default(
      "method", "hf",
      "SCF method: 'hf' for Hartree-Fock, or a DFT functional name. "
      "See the user manual for the complete list of available options.");
  set_default("convergence_threshold", 1e-7);
  set_default("max_iterations", 50, "Maximum number of SCF iterations",
              qdk::chemistry::data::BoundConstraint<int64_t>{
                  1, std::numeric_limits<int64_t>::max()});
  set_default("scf_type", std::string("auto"),
              "SCF orbital type: 'auto' selects based on spin multiplicity",
              data::ListConstraint<std::string>{{std::vector<std::string>{
                  "auto", "restricted", "unrestricted"}}});
}

std::unique_ptr<ScfSolver> make_microsoft_scf_solver() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<qdk::chemistry::algorithms::microsoft::ScfSolver>();
}

std::unique_ptr<ScfSolver> make_microsoft_stabilized_scf_solver() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::StabilizedScfSolver>();
}

void ScfSolverFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  ScfSolverFactory::register_instance(&make_microsoft_scf_solver);
  ScfSolverFactory::register_instance(&make_microsoft_stabilized_scf_solver);
}

}  // namespace qdk::chemistry::algorithms

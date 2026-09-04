// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/scf.hpp"

#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <utility>

#include "microsoft/stabilized_scf.hpp"

namespace qdk::chemistry::algorithms {

std::pair<double, std::shared_ptr<data::Wavefunction>> ScfSolver::run(
    std::shared_ptr<data::Structure> structure, int charge,
    int spin_multiplicity, BasisOrGuessType basis_or_guess,
    std::shared_ptr<data::AuxiliaryBasisCollection> auxiliary_bases) const {
  return Algorithm::run(std::move(structure), charge, spin_multiplicity,
                        std::move(basis_or_guess), std::move(auxiliary_bases));
}

std::string ScfSolver::hash(
    std::shared_ptr<data::Structure> structure, int charge,
    int spin_multiplicity, BasisOrGuessType basis_or_guess,
    std::shared_ptr<data::AuxiliaryBasisCollection> auxiliary_bases) const {
  return Algorithm::hash(std::move(structure), charge, spin_multiplicity,
                         std::move(basis_or_guess), std::move(auxiliary_bases));
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

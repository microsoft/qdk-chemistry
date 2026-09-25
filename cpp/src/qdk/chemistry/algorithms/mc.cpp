// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <limits>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "microsoft/macis_asci.hpp"
#include "microsoft/macis_cas.hpp"

namespace qdk::chemistry::algorithms {

MultiConfigurationSettings::MultiConfigurationSettings() {
  // Initialize with default settings for MC calculations
  // evaluate 1 RDM
  set_default<bool>("calculate_one_rdm", false);
  // evaluate 2 RDM
  set_default<bool>("calculate_two_rdm", false);
  // evaluate single orbital entropies
  set_default<bool>("calculate_single_orbital_entropies", false);
  // evaluate two-orbital entropies
  set_default<bool>("calculate_two_orbital_entropies", false);
  // evaluate mutual information
  set_default<bool>("calculate_mutual_information", false);
  // energy convergence threshold
  set_default<double>("ci_residual_tolerance", 1.0e-6,
                      "CI residual convergence tolerance",
                      qdk::chemistry::data::BoundConstraint<double>{0.0, 1.0});
  // maximum number of iterations any Davidson
  set_default<int64_t>("max_solver_iterations", 200,
                       "Maximum number of Davidson iterations",
                       qdk::chemistry::data::BoundConstraint<int64_t>{
                           1, std::numeric_limits<int64_t>::max()});
  // Matrix size cutoff for using dense vs iterative eigensolver.
  // If the number of determinants is at or below this value, dense
  // diagonalization is used; otherwise the iterative (Davidson) solver.
  set_default<int64_t>("iterative_solver_dimension_cutoff", 2000,
                       "Matrix size cutoff for using iterative eigensolver",
                       qdk::chemistry::data::BoundConstraint<int64_t>{
                           1, std::numeric_limits<int64_t>::max()});
}

std::unique_ptr<MultiConfigurationCalculator> make_macis_cas_mc() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<qdk::chemistry::algorithms::microsoft::MacisCas>();
}
std::unique_ptr<MultiConfigurationCalculator> make_macis_asci_mc() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<qdk::chemistry::algorithms::microsoft::MacisAsci>();
}

void MultiConfigurationCalculatorFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  MultiConfigurationCalculatorFactory::register_instance(&make_macis_cas_mc);
  MultiConfigurationCalculatorFactory::register_instance(&make_macis_asci_mc);
}

}  // namespace qdk::chemistry::algorithms

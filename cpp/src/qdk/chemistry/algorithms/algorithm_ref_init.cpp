// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/active_space.hpp>
#include <qdk/chemistry/algorithms/algorithm_ref_init.hpp>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/algorithms/mcscf.hpp>
#include <qdk/chemistry/algorithms/pmc.hpp>
#include <qdk/chemistry/algorithms/scf.hpp>
#include <qdk/chemistry/algorithms/stability.hpp>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms {

namespace {

/// Try to create an algorithm via @p Factory and return a copy of its settings.
/// Returns nullptr if the name is not found.
template <typename Factory>
std::shared_ptr<data::Settings> try_factory(const std::string& name) {
  auto algo = Factory::create(name);
  return std::make_shared<data::Settings>(algo->settings());
}

}  // namespace

void init_algorithm_ref_resolver() {
  data::AlgorithmRef::create_default_settings =
      [](const std::string& type,
         const std::string& name) -> std::shared_ptr<data::Settings> {
    // Dispatch to the matching factory by type name.
    if (type == ScfSolverFactory::algorithm_type_name())
      return try_factory<ScfSolverFactory>(name);
    if (type == ActiveSpaceSelectorFactory::algorithm_type_name())
      return try_factory<ActiveSpaceSelectorFactory>(name);
    if (type == HamiltonianConstructorFactory::algorithm_type_name())
      return try_factory<HamiltonianConstructorFactory>(name);
    if (type == MultiConfigurationCalculatorFactory::algorithm_type_name())
      return try_factory<MultiConfigurationCalculatorFactory>(name);
    if (type ==
        ProjectedMultiConfigurationCalculatorFactory::algorithm_type_name())
      return try_factory<ProjectedMultiConfigurationCalculatorFactory>(name);
    if (type == DynamicalCorrelationCalculatorFactory::algorithm_type_name())
      return try_factory<DynamicalCorrelationCalculatorFactory>(name);
    if (type == MultiConfigurationScfFactory::algorithm_type_name())
      return try_factory<MultiConfigurationScfFactory>(name);
    if (type == LocalizerFactory::algorithm_type_name())
      return try_factory<LocalizerFactory>(name);
    if (type == StabilityCheckerFactory::algorithm_type_name())
      return try_factory<StabilityCheckerFactory>(name);
    return nullptr;
  };
}

}  // namespace qdk::chemistry::algorithms

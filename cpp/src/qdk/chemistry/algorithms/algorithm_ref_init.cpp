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

namespace detail {

std::shared_ptr<data::Settings> resolve_algorithm_defaults(
    const std::string& type, const std::string& name) {
#define REGISTER_FACTORY_SETTINGS_INIT(NAME) \
  if (type == NAME::algorithm_type_name()) return try_factory<NAME>(name);
  REGISTER_FACTORY_SETTINGS_INIT(ScfSolverFactory)
  REGISTER_FACTORY_SETTINGS_INIT(ActiveSpaceSelectorFactory)
  REGISTER_FACTORY_SETTINGS_INIT(HamiltonianConstructorFactory)
  REGISTER_FACTORY_SETTINGS_INIT(MultiConfigurationCalculatorFactory)
  REGISTER_FACTORY_SETTINGS_INIT(ProjectedMultiConfigurationCalculatorFactory)
  REGISTER_FACTORY_SETTINGS_INIT(DynamicalCorrelationCalculatorFactory)
  REGISTER_FACTORY_SETTINGS_INIT(MultiConfigurationScfFactory)
  REGISTER_FACTORY_SETTINGS_INIT(LocalizerFactory)
  REGISTER_FACTORY_SETTINGS_INIT(StabilityCheckerFactory)

#undef REGISTER_FACTORY_SETTINGS_INIT

  return nullptr;
}

}  // namespace detail

}  // namespace qdk::chemistry::algorithms

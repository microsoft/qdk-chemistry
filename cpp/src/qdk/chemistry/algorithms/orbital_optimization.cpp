// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/orbital_optimization.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "microsoft/qio/qio_orbital_optimizer.hpp"

namespace qdk::chemistry::algorithms {

namespace {

std::unique_ptr<OrbitalOptimizer> make_qio_orbital_optimizer() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<microsoft::QIOOrbitalOptimizer>();
}

}  // namespace

void OrbitalOptimizerFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();
  OrbitalOptimizerFactory::register_instance(&make_qio_orbital_optimizer);
}

}  // namespace qdk::chemistry::algorithms

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/hamiltonian_regularizer.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "microsoft/flr_bliss/flr_bliss_regularizer.hpp"

namespace qdk::chemistry::algorithms {

namespace {

std::unique_ptr<HamiltonianRegularizer> make_flr_bliss_regularizer() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::FlrBlissRegularizer>();
}

}  // namespace

void HamiltonianRegularizerFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  HamiltonianRegularizerFactory::register_instance(&make_flr_bliss_regularizer);
}

}  // namespace qdk::chemistry::algorithms

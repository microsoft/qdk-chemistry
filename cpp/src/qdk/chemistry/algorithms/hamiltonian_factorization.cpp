// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <memory>
#include <qdk/chemistry/algorithms/double_factorization.hpp>
#include <qdk/chemistry/algorithms/hamiltonian_factorization.hpp>
#include <qdk/chemistry/utils/logger.hpp>

namespace qdk::chemistry::algorithms {

namespace {

std::unique_ptr<HamiltonianFactorization> make_double_factorization() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<DoubleFactorization>();
}

}  // namespace

void HamiltonianFactorizationFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  HamiltonianFactorizationFactory::register_instance(
      &make_double_factorization);
}

}  // namespace qdk::chemistry::algorithms

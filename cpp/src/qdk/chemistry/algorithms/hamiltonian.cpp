// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/hamiltonian.hpp"

#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#include "microsoft/cholesky_hamiltonian.hpp"
#include "microsoft/hamiltonian_basis_transformer.hpp"

namespace qdk::chemistry::algorithms {

std::unique_ptr<HamiltonianConstructor> make_microsoft_cholesky_hamiltonian() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::CholeskyHamiltonianConstructor>();
}

std::unique_ptr<HamiltonianConstructor> make_microsoft_hamiltonian() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<
      qdk::chemistry::algorithms::microsoft::HamiltonianConstructor>();
}

void HamiltonianConstructorFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  HamiltonianConstructorFactory::register_instance(&make_microsoft_hamiltonian);
  HamiltonianConstructorFactory::register_instance(
      &make_microsoft_cholesky_hamiltonian);
}

std::unique_ptr<HamiltonianBasisTransformer>
make_microsoft_hamiltonian_basis_transformer() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<microsoft::QdkHamiltonianBasisTransformer>();
}

void HamiltonianBasisTransformerFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();
  HamiltonianBasisTransformerFactory::register_instance(
      &make_microsoft_hamiltonian_basis_transformer);
}

}  // namespace qdk::chemistry::algorithms

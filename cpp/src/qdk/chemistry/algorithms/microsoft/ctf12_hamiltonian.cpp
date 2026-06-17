// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ctf12_hamiltonian.hpp"

#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Hamiltonian> CtF12HamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Wavefunction> reference) const {
  QDK_LOG_TRACE_ENTERING();

  if (!reference) {
    throw std::invalid_argument(
        "CtF12HamiltonianConstructor: reference wavefunction is null");
  }

  throw std::runtime_error(
      "CtF12HamiltonianConstructor: CT-F12 effective-Hamiltonian construction "
      "is not yet implemented");
}

}  // namespace qdk::chemistry::algorithms::microsoft

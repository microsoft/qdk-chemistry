// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <memory>
#include <qdk/chemistry/algorithms/symmetry_shift.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <string>

#include "microsoft/symmetry_shift/fermionic_low_rank.hpp"

namespace qdk::chemistry::algorithms {

// ---------------------------------------------------------------------------
// Factory registration.
// ---------------------------------------------------------------------------

namespace {

std::unique_ptr<SymmetryShifter> make_fermionic_low_rank_shifter() {
  QDK_LOG_TRACE_ENTERING();
  return std::make_unique<microsoft::FermionicLowRankShifter>();
}

}  // namespace

void SymmetryShifterFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  SymmetryShifterFactory::register_instance(&make_fermionic_low_rank_shifter);
}

}  // namespace qdk::chemistry::algorithms

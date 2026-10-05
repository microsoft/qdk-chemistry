// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/mo_scf.hpp>

#include "scf.hpp"

namespace qdk::chemistry::algorithms::microsoft {

/** @brief MO-basis HF using the native QDK SCF iteration and integral engines.
 */
class MoScfSolver final : public algorithms::MoScfSolver {
 public:
  MoScfSolver() { _settings = std::make_unique<ScfIterationSettings>(); }
  std::string name() const final { return "qdk"; }

 protected:
  std::pair<double, std::shared_ptr<data::Ansatz>> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

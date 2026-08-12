// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include "hamiltonian.hpp"

namespace qdk::chemistry::algorithms::microsoft {

class ScalarRelativisticHamiltonianSettings : public HamiltonianSettings {
 public:
  ScalarRelativisticHamiltonianSettings() {
    set_default("xuncontract", true,
                "Decontract the orbital basis for the X2C transformation and "
                "recontract the resulting one-electron Hamiltonian");
  }
  ~ScalarRelativisticHamiltonianSettings() override = default;
};

/**
 * @class ScalarRelativisticHamiltonianConstructor
 * @brief Constructs a scalar-relativistic Hamiltonian using the exact
 *        two-component (X2C) approach for one-electron integrals.
 *
 * This class follows the same workflow as the nonrelativistic
 * @ref HamiltonianConstructor, but replaces the core one-electron
 * Hamiltonian \f$ h = T + V \f$ with the scalar-relativistic X2C
 * one-electron Hamiltonian \f$ h^{\text{X2C}} \f$.  The two-electron
 * integrals (ERI) are left unchanged (untransformed), which is the
 * standard "one-electron X2C" or "X2C-1e" approximation.
 *
 * @note Effective core potentials are not supported. Use an all-electron
 *       relativistic basis set such as cc-pVXZ-DK or ANO-RCC.
 *
 * The X2C procedure:
 *   1. Compute one-electron integrals S, T, V in the AO basis,
 *      plus the spin-free pVp integrals \f$ W^{SF}_{\mu\nu} =
 *      \langle\chi_\mu|\hat{p}\cdot V\hat{p}|\chi_\nu\rangle \f$
 *      analytically via Libint2 Operator::opVop.
 *   2. Build and diagonalise the modified Dirac Hamiltonian.
 *   3. Select the positive-energy eigenvectors and project their energies
 *      back to the original AO metric.
 *   4. Form the spin-free two-component one-electron Hamiltonian.
 *
 * @see HamiltonianConstructor (nonrelativistic counterpart)
 */
class ScalarRelativisticHamiltonianConstructor
    : public qdk::chemistry::algorithms::HamiltonianConstructor {
 public:
  ScalarRelativisticHamiltonianConstructor() {
    _settings = std::make_unique<ScalarRelativisticHamiltonianSettings>();
  };
  ~ScalarRelativisticHamiltonianConstructor() override = default;

  virtual std::string name() const final { return "qdk_x2c"; };

 protected:
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Orbitals> orbitals) const override;
};

}  // namespace qdk::chemistry::algorithms::microsoft

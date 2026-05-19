// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/// @name X2C numeric thresholds
/// @{

/// Threshold for discarding near-zero eigenvalues in the 4-component
/// metric during canonical orthogonalisation.  Must be very tight because
/// the small-component metric T/(2c²) has inherently small eigenvalues.
/// Matches PySCF's ``x2c.LINEAR_DEP_THRESHOLD``.
static constexpr double x2c_metric_lindep_threshold = 1e-14;

/// Number of decimal digits retained when rounding Gaussian exponents
/// during basis decontraction.  Two exponents that agree to this many
/// decimals are considered duplicates and merged.
static constexpr int x2c_exponent_rounding_digits = 9;

/// Derived precision for rounding and comparing exponents.
static constexpr double x2c_exponent_rounding_factor = 1e9;
static constexpr double x2c_exponent_duplicate_tolerance = 1e-9;

/// @}

class ScalarRelativisticHamiltonianSettings
    : public qdk::chemistry::data::Settings {
 public:
  ScalarRelativisticHamiltonianSettings() {
    set_default("eri_method", "direct");
    set_default("scf_type", "auto");
    set_default("xuncontract", true);
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
 * @note The user should supply an appropriate uncontracted or
 *       relativistic basis set (e.g. cc-pVXZ-DK, ANO-RCC).
 *
 * The X2C procedure:
 *   1. Compute one-electron integrals S, T, V in the AO basis,
 *      plus the spin-free pVp integrals \f$ W^{SF}_{\mu\nu} =
 *      \langle\chi_\mu|\hat{p}\cdot V\hat{p}|\chi_\nu\rangle \f$
 *      analytically via Libint2 Operator::opVop.
 *   2. Build and diagonalise the modified Dirac Hamiltonian.
 *   3. Construct the X2C decoupling matrix X from the large/small
 *      component ratio.
 *   4. Build the renormalisation matrix R.
 *   5. Form the 2-component (scalar-relativistic) core Hamiltonian
 *      \f$ h^{\text{X2C}} = R^{\dagger} (V + X^{\dagger} T +
 *      T X + X^{\dagger} (W^{SF}/(4c^2) - T) X) R \f$
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

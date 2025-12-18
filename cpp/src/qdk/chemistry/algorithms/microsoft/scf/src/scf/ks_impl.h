// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/exc.h>

#include "scf/scf_impl.h"

namespace qdk::chemistry::scf {
/**
 * @brief Kohn-Sham density functional theory (DFT) SCF solver implementation
 *
 * Implements the Kohn-Sham formulation of density functional theory, extending
 * the base SCF solver with exchange-correlation (XC) functional evaluation.
 *
 * The Kohn-Sham Fock matrix has the form:
 *   F = H + J + V_XC
 * where H is the core Hamiltonian, J is the Coulomb matrix, and V_XC is the
 * exchange-correlation potential obtained by functional derivative of E_XC.
 */
class KSImpl : public SCFImpl {
 public:
  /**
   * @brief Construct Kohn-Sham solver with configured initial guess
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration
   */
  KSImpl(std::shared_ptr<Molecule> mol, const SCFConfig& cfg);

  /**
   * @brief Construct Kohn-Sham solver with initial density matrix
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration
   * @param density_matrix Initial density matrix guess
   */
  KSImpl(std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
         const RowMajorMatrix& density_matrix);

  /**
   * @brief Construct Kohn-Sham solver with initial density matrix
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration
   * @param basis_set Basis set to use
   * @param raw_basis_set Raw (unnormalized) basis set for output
   */
  KSImpl(std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
         std::shared_ptr<BasisSet> basis_set,
         std::shared_ptr<BasisSet> raw_basis_set);

 private:
  /**
   * @brief Update Fock matrix with exchange-correlation contributions
   *
   * Computes the XC potential matrix V_XC from the current density matrix and
   * updates the Fock matrix: F = H + J + K (for hybrids) + V_XC
   */
  void update_fock_() override;

  /**
   * @brief Reset Fock matrix to initial state for new SCF iteration
   *
   * Reinitializes the Fock matrix to F = H + V_XC
   */
  void reset_fock_() override;

  /**
   * @brief Compute total electronic energy for current SCF state
   * @return Total electronic + nuclear energy in atomic units (Hartree)
   * @note Energy components are stored in ctx.result for later access
   */
  double total_energy_() override;

  /**
   * @brief Get hybridization coefficients for range-separated functionals
   *
   * @returns A tuple containing the coefficients
   *   0: alpha Fraction of long-range HF exchange (0.0 to 1.0)
   *   1: beta Fraction of short-range HF exchange (for RSH functionals)
   *   2: omega Range-separation parameter (for RSH functionals)
   *
   * @note For pure DFT functionals, alpha = beta = 0.0
   */
  std::tuple<double, double, double> get_hyb_coeff_() const override;

  /**
   * @brief Compute nuclear gradient contribution from XC functional
   * @return Matrix of XC gradients with shape (3, n_atoms) in atomic units
   * (Hartree/bohr) Each column contains [∂E/∂x, ∂E/∂y, ∂E/∂z] for one atom
   */
  const RowMajorMatrix get_vxc_grad_() const override;

  /// Exchange-correlation potential matrix V_XC in AO basis
  RowMajorMatrix XC_;

  /// Exchange-correlation functional evaluator
  std::shared_ptr<EXC> exc_;

  // CPSCF/TDDFT support for response properties

  /**
   * @brief Update trial Fock matrix for CPSCF/TDDFT response calculations
   *
   * Computes the XC kernel contribution to the trial Fock matrix used in
   * coupled-perturbed SCF (CPSCF) or time-dependent DFT (TDDFT) calculations
   * for computing response properties such as polarizabilities and excitation
   * energies.
   */
  void update_trial_fock_() override;

  /// Trial XC potential for CPSCF/TDDFT response calculations
  RowMajorMatrix tXC_;
};
}  // namespace qdk::chemistry::scf

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/scf.h>

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

namespace qdk::chemistry::scf {
/**
 * @brief Base class for exchange-correlation (XC) functional evaluation
 *
 * Abstract base class providing the interface for computing
 * exchange-correlation contributions to the Fock matrix, gradients, and
 * response properties in DFT calculations.
 *
 * Derived classes implement specific backend integrators (GauXC, etc.).
 */
class EXC {
 public:
  /**
   * @brief Construct XC functional evaluator
   *
   * Populates metadata shared by all EXC implementations
   *
   * @param basis_set The atomic orbital basis set
   * @param cfg SCF configuration
   */
  EXC(std::shared_ptr<BasisSet> basis_set, const SCFConfig& cfg);

  /**
   * @brief Virtual destructor for proper cleanup of derived classes
   */
  virtual ~EXC() = default;

  /**
   * @brief Get hybrid functional mixing parameters
   *
   * Returns the exact exchange mixing coefficients for hybrid and
   * range-separated hybrid functionals. For pure DFT functionals,
   * all parameters are zero.
   *
   * @returns A tuple containing the coefficients
   *   0: alpha Fraction of long-range HF exchange (0.0 to 1.0)
   *   1: beta Fraction of short-range HF exchange (for RSH functionals)
   *   2: omega Range-separation parameter (for RSH functionals)
   */
  std::tuple<double, double, double> get_hyb() const {
    return std::make_tuple(x_alpha_, x_beta_, x_omega_);
  }

  /**
   * @brief Build exchange-correlation contribution to Fock matrix
   *
   * Evaluates the XC functional on a numerical grid and computes the
   * XC matrix contribution to the Fock matrix via:
   *   V_XC[μν] = ∫ φ_μ(r) * (δE_XC/δρ(r)) * φ_ν(r) dr
   *
   * This is a pure virtual function that must be implemented by derived
   * classes.
   *
   * @param[in] P Density matrix in AO basis (size: num_density_matrices *
   * num_basis_funcs * num_basis_funcs)
   * @param[out] VXC XC matrix contribution to Fock matrix (size:
   * num_density_matrices * num_basis_funcs * num_basis_funcs)
   * @param[out] xc_energy Total XC energy E_XC[ρ]
   * @note ``ndm`` is the number of density matrices; 1 for restricted, 2 for
   *       unrestricted calculations.
   */
  virtual void build_XC(const double* P, double* VXC, double* xc_energy) = 0;

  /**
   * @brief Compute XC contribution to nuclear gradients
   *
   * Evaluates the derivative of the XC energy with respect to nuclear
   * coordinates.
   *
   * This is a pure virtual function that must be implemented by derived
   * classes.
   *
   * @param[in] P Density matrix in AO basis (size: (num_density_matrices,
   * num_basis_funcs, num_basis_funcs))
   * @param[out] dXC XC contribution to energy gradient (size: (3,natoms))
   * @note ``ndm`` is the number of density matrices; 1 for restricted, 2 for
   *       unrestricted calculations.
   */
  virtual void get_gradients(const double* P, double* dXC) = 0;

  /**
   * @brief Evaluate XC kernel contraction for linear response
   *
   * Computes the contraction of the XC kernel (second functional derivative)
   * with a trial density for TDDFT (time-dependent DFT) or CPSCF (coupled-
   * perturbed SCF) calculations:
   *
   *   F_XC[μν] = ∫∫ φ_μ(r) * (δ²E_XC/δρ(r)δρ(r)) * δρ_trial(r) * φ_ν(r) dr
   *
   * where δρ is the density perturbation from the trial density tD.
   *
   * This is a pure virtual function that must be implemented by derived
   * classes.
   *
   * @param[in] D Ground state density matrix in the AO basis (side:
   * num_density_matrices * num_basis_funcs * num_basis_funcs)
   * @param[in] tD Trial/perturbed density matrix in the AO basis (size:
   * num_density_matrices * num_basis_funcs * num_basis_funcs)
   * @param[out] Fxc XC kernel contribution to response matrix
   * @note ``ndm`` is the number of density matrices; 1 for restricted, 2 for
   *       unrestricted calculations.
   */
  virtual void eval_fxc_contraction(const double* D, const double* tD,
                                    double* Fxc) = 0;

  /**
   * @brief Factory method to create appropriate EXC implementation
   *
   * Creates an EXC object of the appropriate derived type based on the
   * method specified in the configuration.
   *
   * @param basis_set Basis set for the calculation
   * @param cfg SCF configuration specifying XC method and functional
   * @return std::shared_ptr<EXC> Pointer to newly created EXC instance
   */
  static std::shared_ptr<EXC> create(std::shared_ptr<BasisSet> basis_set,
                                     const SCFConfig& cfg);

 protected:
  std::shared_ptr<BasisSet> basis_set_;  ///< The AO basis set
  const EXCConfig& cfg_;                 ///< XC configuration
  const ParallelConfig& mpi_;            ///< Parallelization configuration

  std::vector<std::pair<int, double>>
      x_functionals_;  ///< Exchange functional components with weights
  std::vector<std::pair<int, double>>
      c_functionals_;  ///< Correlation functional components with weights

  double x_alpha_;  ///< Short-range HF exchange fraction α
  double x_beta_;   ///< Long-range HF exchange fraction β
  double x_omega_;  ///< Range-separation parameter ω
};
}  // namespace qdk::chemistry::scf

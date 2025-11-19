// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/core/exc.h>
#include <qdk/chemistry/scf/util/gauxc_registry.h>

namespace qdk::chemistry::scf {

// Forward decl for implementation classes
namespace impl {
class GAUXC;
}

/**
 * @brief Exchange-correlation functional evaluator using GauXC library
 *
 * Wrapper class providing the EXC interface for GauXC-based XC functional
 * evaluation. Manages registration and lookup of GauXC implementation objects
 * via a global registry to enable efficient reuse across SCF iterations and
 * multiple calculations with identical settings.
 *
 * @see impl::GAUXC for the underlying implementation details
 */
class GAUXC : public EXC {
 public:
  /**
   * @brief Construct GauXC-based XC functional evaluator
   *
   * @param basis_set Basis set defining atomic orbital functions
   * @param cfg SCF configuration
   *
   * @throws std::runtime_error If GauXC initialization fails
   * @throws std::runtime_error If functional name is invalid/unsupported
   *
   * @note Constructor does NOT build grids or allocate GPU memory yet;
   *       this happens lazily on first build_XC() call via registry
   */
  GAUXC(std::shared_ptr<BasisSet> basis_set, const SCFConfig& cfg);

  /**
   * @brief Destructor
   */
  ~GAUXC() noexcept;

  /**
   * @brief Build exchange-correlation matrix and compute XC energy
   * @see impl::GAUXC::build_XC for the underlying implementation details
   */
  void build_XC(const double* D, double* XC, double* xc_energy) override;

  /**
   * @brief Compute XC contribution to nuclear gradients
   * @see impl::GAUXC::get_gradients for the underlying implementation details
   */
  void get_gradients(const double* D, double* dXC) override;

  /**
   * @brief Evaluate XC kernel contraction for linear response calculations
   * @see impl::GAUXC::eval_fxc_contraction for the underlying implementation
   * details
   */
  void eval_fxc_contraction(const double* D, const double* tD,
                            double* Fxc) override;

 private:
  /// GauXC configuration (grid settings, pruning scheme, etc.)
  GAUXCInput gauxc_input_;

  /// Whether calculation uses unrestricted (UKS) vs restricted (RKS) formalism
  bool unrestricted_;

  /// XC functional name string (e.g., "B3LYP", "PBE0", "ωB97X")
  std::string xc_name_;
};
}  // namespace qdk::chemistry::scf

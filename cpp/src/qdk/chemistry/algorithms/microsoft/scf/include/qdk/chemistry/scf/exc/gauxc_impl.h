// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/exc.h>

#include <Eigen/Core>
#include <gauxc/xc_integrator.hpp>

namespace qdk::chemistry::scf::impl {

/**
 * @brief GauXC backend implementation for XC and sn-K evaluation
 *
 * @see https://github.com/wavefunction/GauXC for GauXC documentation
 */
class GAUXC {
 public:
  /**
   * @brief Construct GauXC XC functional evaluator
   *
   * Initializes the GauXC runtime environment and XC integrator with
   * the specified functional, grid settings, and execution backend.
   *
   * @param basis_set Basis set for evaluating basis functions on grids
   * @param gauxc_input GauXC configuration
   * @param unrestricted If true, use spin-unrestricted (UKS) formalism
   * @param xc_name Functional name (e.g., "B3LYP", "PBE0", "wB97X")
   *
   * @throws std::invalid_argument if xc_name is not supported in GauXC
   */
  GAUXC(BasisSet& basis_set, const GAUXCInput& gauxc_input, bool unrestricted,
        const std::string& xc_name);

  /**
   * @brief Destructor
   */
  ~GAUXC() noexcept;

  /**
   * @brief Build XC matrix and compute XC energy
   * @see EXC::build_XC for API documentation
   */
  void build_XC(const double* D, double* VXC, double* xc_energy);

  /**
   * @brief Compute XC contribution to nuclear gradients
   * @see EXC::get_gradients for API documentation
   */
  void get_gradients(const double* D, double* dXC);

  /**
   * @brief Build semi-numerical exact exchange (snK) matrix
   *
   * Computes the exact exchange matrix using semi-numerical integration,
   * where one electron is integrated numerically on a grid while the other
   * is handled analytically.
   *
   * @param[in] D Density matrix (size: (ndm, num_basis_funcs, num_basis_funcs))
   * @param[out] K Semi-numerical exchange matrix (size: (ndm, num_basis_funcs,
   * num_basis_funcs))
   * @note ``ndm`` is the number of density matrices; 1 for restricted, 2 for
   *       unrestricted calculations.
   */
  void build_snK(const double* D, double* K);

  /**
   * @brief Evaluate density-dependent multipole moments
   *
   * Computes atom-partitioned multipole moments (q_lm) that depend on the
   * electron density, used in domain decomposition methods (e.g., ddCOSMO,
   * ddPCM) for solvation.
   *
   * @param[in] lmax Maximum angular momentum for multipole expansion
   * @param[in] D Density matrix (size:
   * num_density_matrices * num_basis_funcs × num_basis_funcs)
   * @param[out] dd_psi Density-dependent (size: ((2*lmax+1),natoms))
   * @note ``ndm`` is the number of density matrices; 1 for restricted, 2 for
   *       unrestricted calculations.
   */
  void eval_dd_psi(int lmax, const double* D, double* dd_psi);

  /**
   * @brief Evaluate density-dependent potential from multipoles
   *
   * Computes the potential arising from external multipole moments,
   * which couples back to the electronic structure in domain decomposition
   * solvation methods.
   *
   * @param[in] lmax Maximum angular momentum for multipole expansion
   * @param[in] x Input multipole coefficients from external environment (size:
   * ((2*lmax+1),natoms))
   * @param[out] dd_psi_potential Potential contribution to Fock matrix (size:
   * (num_density_matrices,num_basis_funcs,num_basis_funcs))
   */
  void eval_dd_psi_potential(int lmax, const double* x,
                             double* dd_psi_potential);

  /**
   * @brief Evaluate XC kernel contraction for TDDFT/CPSCF
   * @see EXC::eval_fxc_contraction for API documentation
   */
  void eval_fxc_contraction(const double* D, const double* tD, double* Fxc);

  double x_alpha;  ///< Short-range HF exchange fraction α
  double x_beta;   ///< Long-range HF exchange fraction β
  double x_omega;  ///< Range-separation parameter ω

 private:
  std::shared_ptr<GauXC::RuntimeEnvironment>
      rt_;  ///< GauXC runtime (manages execution backend and resources)
  std::shared_ptr<GauXC::XCIntegrator<Eigen::MatrixXd>>
      integrator_;     ///< GauXC XC integrator instance
  bool unrestricted_;  ///< True for spin-unrestricted (UKS), false for
                       ///< restricted (RKS)
#ifdef QDK_CHEMISTRY_ENABLE_GPU
  size_t device_buffer_sz_;  ///< Size of allocated GPU device buffer in bytes

  /**
   * @brief Allocate GPU device buffer asynchronously
   *
   * Allocates temporary GPU memory for GauXC operations in the given stream.
   *
   * @param sz Size in bytes to allocate
   * @param stream CUDA stream for asynchronous allocation
   */
  void allocate_device_buffer_async_(size_t sz, cudaStream_t stream);

  /**
   * @brief Free GPU device buffer asynchronously
   *
   * Deallocates GPU memory previously allocated by
   * allocate_device_buffer_async_.
   *
   * @param stream CUDA stream for asynchronous deallocation
   */
  void free_device_buffer_async_(cudaStream_t stream);
#endif  // QDK_CHEMISTRY_ENABLE_GPU
};

}  // namespace qdk::chemistry::scf::impl

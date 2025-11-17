// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "snk.h"

#include <qdk/chemistry/scf/util/gauxc_registry.h>

#include <stdexcept>

#include "util/timer.h"

namespace qdk::chemistry::scf {
namespace snk {

/**
 * @brief ERI class for Semi-Numerical Kohn-Sham (SNK) electron repulsion
 * integrals
 *
 * This class provides an interface for computing exchange (K) matrices using
 * the Semi-Numerical Kohn-Sham approach. SNK uses numerical integration via
 * GauXC for the exchange integrals, providing improved computational efficiency
 * for large molecular systems.
 *
 * The SNK approach leverages GauXC's efficient numerical integration
 * infrastructure to evaluate the exchange integrals on real-space grids,
 * avoiding the costly four-center two-electron repulsion integrals required in
 * traditional methods. This results in linear scaling behavior for the exchange
 * contribution.
 *
 * @note SNK ONLY constructs exchange (K) matrices - Coulomb (J) matrices are
 * NOT available
 * @note Range-separated hybrid functionals are not currently supported
 */
class ERI {
  GAUXCInput
      gauxc_input_;  ///< Configuration for GauXC numerical integration settings
  bool unrestricted_;  ///< Whether to use unrestricted (UKS) formalism
  std::string
      xc_name_;  ///< Functional name (used for GauXC grid configuration)
  size_t num_basis_funcs_;  ///< Number of basis functions

 public:
  /**
   * @brief Construct SNK ERI engine
   *
   * Initializes the SNK engine with the given basis set and GauXC
   * configuration. The GauXC implementation is obtained from the registry for
   * efficient reuse.
   *
   * @param unr Whether to use unrestricted formalism
   * @param basis_set Orbital basis set
   * @param gauxc_input GauXC configuration for numerical integration
   * @param xc_name Functional name (used for grid setup, not XC evaluation)
   */
  ERI(bool unr, const BasisSet& basis_set, GAUXCInput gauxc_input,
      std::string xc_name) {
    unrestricted_ = unr;
    gauxc_input_ = gauxc_input;
    xc_name_ = xc_name == "HF" ? "PBE0" : xc_name;  // does not matter
    num_basis_funcs_ = basis_set.num_basis_funcs;

    // Get or create the GAUXC implementation from the registry
    util::GAUXCRegistry::get_or_create(const_cast<BasisSet&>(basis_set),
                                       gauxc_input_, unrestricted_, xc_name_);
  }

  /**
   * @brief Build exchange (K) matrix using SNK approach
   *
   * Computes ONLY the exchange matrix K using semi-numerical integration via
   * GauXC. The Coulomb matrix J is NOT supported and will cause an error if
   * requested. The exchange integrals are evaluated numerically on real-space
   * grids, providing linear scaling behavior.
   *
   * @param P Density matrix (AO basis)
   * @param J Coulomb matrix output (MUST be nullptr - J matrices are NOT
   * supported)
   * @param K Exchange matrix output (this is what SNK computes)
   * @param alpha Scaling factor for Hartree-Fock exchange
   * @param beta Scaling factor for DFT exchange (added to alpha)
   * @param omega Range-separation parameter (not supported, must be ~0)
   *
   * @throws std::runtime_error If J matrix is requested (J matrices are NOT
   * available)
   * @throws std::runtime_error If range-separated hybrid is requested
   *
   * @note SNK ONLY builds K matrices - J matrices are NOT available in this
   * module
   * @note Final K matrix is scaled by (alpha + beta)
   */
  void build_JK(const double* P, double* J, double* K, double alpha,
                double beta, double omega) {
    AutoTimer t("ERI::build_JK");
    const size_t mat_size =
        (unrestricted_ ? 2 : 1) * num_basis_funcs_ * num_basis_funcs_;

    // RSH check
    const bool is_rsx = std::abs(omega) > 1e-12;
    if (is_rsx) {
      throw std::runtime_error(
          "SNK + Range Separated Hybrid is not yet implemented");
    }

    // Pure early exit
    if (std::abs(alpha + beta) < 1e-12) {
      if (K) std::memset(K, 0, mat_size * sizeof(double));
      return;
    }
    // J-request check - SNK does not provide J matrices
    if (J) {
      throw std::runtime_error(
          "SNK Cannot Construct J - Only K matrices are available");
    }

    auto gauxc_impl = util::GAUXCRegistry::find(gauxc_input_);
    gauxc_impl->build_snK(P, K);
    for (size_t i = 0; i < mat_size; ++i) K[i] *= alpha + beta;
  }

  /**
   * @brief Compute nuclear gradients for SNK exchange integrals
   *
   * Computes the contribution to nuclear gradients from the exchange matrix
   * in the SNK formalism. This functionality is not yet implemented.
   * Note: SNK only provides exchange (K) contributions - no Coulomb (J)
   * gradients.
   *
   * @param P Density matrix (AO basis)
   * @param dJ Coulomb gradient contribution (NOT supported - SNK has no J)
   * @param dK Exchange gradient contribution
   * @param alpha Scaling factor for Hartree-Fock exchange
   * @param beta Scaling factor for DFT exchange
   * @param omega Range-separation parameter
   *
   * @throws std::runtime_error Always - gradients not yet implemented
   */
  void get_gradients(const double* P, double* dJ, double* dK, double alpha,
                     double beta, double omega) {
    throw std::runtime_error("SNK + Gradients Not Yet Implemented");
  }

  /**
   * @brief Perform quarter transformation of integrals
   *
   * Quarter transformation involves transforming one index of the four-center
   * integrals from AO to MO basis. This operation is not supported in SNK
   * since it operates directly on density matrices rather than storing
   * integrals.
   *
   * @param nt Number of transformed orbitals
   * @param C Transformation coefficients
   * @param out Output buffer for transformed integrals
   *
   * @throws std::runtime_error Always - quarter transformation not supported
   */
  void quarter_trans(size_t nt, const double* C, double* out) {
    throw std::runtime_error("SNK cannot be used for quarter transformation");
  };

  /**
   * @brief Factory method to create SNK ERI instance
   *
   * Static factory method that creates a new SNK ERI instance with the
   * specified configuration. This provides a clean interface for creating
   * SNK instances from external code.
   *
   * @param unr Whether to use unrestricted formalism
   * @param basis_set Orbital basis set
   * @param gauxc_input GauXC configuration for numerical integration
   * @param xc_name Functional name (used for grid configuration)
   * @return Unique pointer to new SNK ERI instance
   */
  static std::unique_ptr<ERI> make_gauxc_snk(bool unr,
                                             const BasisSet& basis_set,
                                             GAUXCInput gauxc_input,
                                             std::string xc_name) {
    return std::make_unique<ERI>(unr, basis_set, gauxc_input, xc_name);
  }
};

}  // namespace snk

SNK::SNK(bool unr, BasisSet& basis_set, GAUXCInput gauxc_input,
         std::string xc_name, ParallelConfig _mpi)
    : ERI(unr, 0.0, basis_set, _mpi),
      eri_impl_(
          snk::ERI::make_gauxc_snk(unr, basis_set, gauxc_input, xc_name)) {}

SNK::~SNK() noexcept = default;

// Public interface implementation - delegates to internal ERI implementation
void SNK::build_JK(const double* P, double* J, double* K, double alpha,
                   double beta, double omega) {
  if (!eri_impl_) throw std::runtime_error("SNK NOT INITIALIZED");
  eri_impl_->build_JK(P, J, K, alpha, beta, omega);
}

// Override implementation for ERI base class
void SNK::build_JK_impl_(const double* P, double* J, double* K, double alpha,
                         double beta, double omega) {
  if (!eri_impl_) throw std::runtime_error("SNK NOT INITIALIZED");
  eri_impl_->build_JK(P, J, K, alpha, beta, omega);
}

// Gradient computation interface
void SNK::get_gradients(const double* P, double* dJ, double* dK, double alpha,
                        double beta, double omega) {
  if (!eri_impl_) throw std::runtime_error("SNK NOT INITIALIZED");
  eri_impl_->get_gradients(P, dJ, dK, alpha, beta, omega);
}

// Quarter transformation interface
void SNK::quarter_trans_impl(size_t nt, const double* C, double* out) {
  if (!eri_impl_) throw std::runtime_error("SNK NOT INITIALIZED");
  eri_impl_->quarter_trans(nt, C, out);
};
}  // namespace qdk::chemistry::scf

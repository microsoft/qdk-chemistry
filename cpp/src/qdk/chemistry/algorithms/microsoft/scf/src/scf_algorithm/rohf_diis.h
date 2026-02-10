// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include "diis_base.h"

namespace qdk::chemistry::scf {

/**
 * @brief Pulay DIIS accelerator specialized for ROHF spin blocking
 *
 * `ROHFDIIS` wraps the common DIIS iteration loop from `DIISBase` and adds the
 * ROHF-specific handling that converts spin-blocked Fock and density matrices
 * into their effective total forms. It maintains cached copies of the
 * effective Fock matrix (`effective_F_`) and total density (`total_P_`) so that
 * repeated invocations within the same SCF step can reuse the already built
 * data when the density is unchanged.
 */
class ROHFDIIS : public DIISBase {
 public:
  /**
  * @brief Construct the ROHF DIIS helper
  * @param ctx SCF context forwarded to the base class
  * @param subspace_size Maximum number of stored error vectors used by DIIS
  */
  explicit ROHFDIIS(const SCFContext& ctx, size_t subspace_size = 8);

  ~ROHFDIIS() noexcept override = default;

  /**
  * @brief Build cached ROHF Fock/density matrices from spin-blocked inputs
  * References: 
  * Guest, M. F., and V. R. Saunders. "On methods for converging open-shell 
  * Hartree-Fock wave-functions." Molecular Physics 28, no. 3 (1974): 819-828.
  * Plakhutin, Boris N., and Ernest R. Davidson. "Canonical form of the
  * Hartree-Fock orbitals in open-shell systems." The Journal of Chemical Physics
  * 140, no. 1 (2014): 014102.
  * @param F Spin-blocked Fock matrix (alpha stacked on beta)
  * @param C Molecular orbital coefficient matrix
  * @param P Spin-blocked density matrix
  * @param nelec_alpha Number of alpha electrons
  * @param nelec_beta Number of beta electrons
  */
  void build_rohf_f_p_matrix(const RowMajorMatrix& F, const RowMajorMatrix& C,
                             const RowMajorMatrix& P, int nelec_alpha,
                             int nelec_beta);

  /**
  * @brief Retrieve the cached effective Fock matrix
  */
  const RowMajorMatrix& get_fock_matrix() const;

  /**
  * @brief Retrieve the cached total density matrix
  */
  const RowMajorMatrix& get_density_matrix() const;

  /**
  * @brief Mutable access to the cached total density matrix
  */
  RowMajorMatrix& density_matrix();

 protected:
  /**
  * @brief Provide DIIS with the ROHF-effective Fock matrix view
  */
  const RowMajorMatrix& get_active_fock(const SCFImpl& scf_impl) const override;

  /**
  * @brief Provide DIIS with the total-density view used for ROHF
  */
  RowMajorMatrix& active_density(SCFImpl& scf_impl) override;

  /**
  * @brief Update the spin-blocked density matrix after solving the eigenproblem
  * @param P Spin-blocked density matrix to overwrite
  * @param C Molecular orbital coefficients used to rebuild densities
  * @param unrestricted Indicates if the SCF run is unrestricted
  * @param nelec_alpha Number of alpha electrons
  * @param nelec_beta Number of beta electrons
  */
  void update_density_matrix(RowMajorMatrix& P, const RowMajorMatrix& C,
                             bool unrestricted, int nelec_alpha,
                             int nelec_beta) override;

  /// Cached effective Fock matrix expressed in the AO basis
  RowMajorMatrix effective_F_;
  /// Cached total density matrix (alpha + beta) exposed to DIIS
  RowMajorMatrix total_P_;
};

}  // namespace qdk::chemistry::scf

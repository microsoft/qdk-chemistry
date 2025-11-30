// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/scf.h>
#include <qdk/chemistry/scf/core/scf_algorithm.h>
#include <qdk/chemistry/scf/core/types.h>

#include <memory>
#include <vector>

namespace qdk::chemistry::scf {

namespace impl {
class GDM;
}  // namespace impl

/**
 * @brief Geometric Direct Minimization (GDM) class
 *
 * The GDM class implements a quasi-Newton orbital optimization method that
 * directly minimizes the energy with respect to occupied-virtual orbital
 * rotations. This method is particularly effective for difficult SCF
 * convergence cases where traditional DIIS methods may fail.
 *
 * Reference: Van Voorhis, Troy, and Martin Head-Gordon. "A geometric approach
 * to direct minimization." Molecular Physics 100, no. 11 (2002): 1713-1721.
 * https://doi.org/10.1080/00268970110103642
 *
 */
class GDM : public SCFAlgorithm {
 public:
  /**
   * @brief Constructor for the GDM (Geometric Direct Minimization) class
   * @param[in] ctx Reference to SCFContext
   * @param[in] gdm_config GDM configuration parameters
   *
   */
  explicit GDM(const SCFContext& ctx, const GDMConfig& gdm_config);

  /**
   * @brief Destructor
   */
  ~GDM() noexcept;

  /**
   * @brief Perform one GDM SCF iteration
   *
   * @param[in,out] scf_impl Reference to SCFImpl containing all matrices and
   * energy
   */
  void iterate(SCFImpl& scf_impl) override;

  /**
   * @brief Set the energy change from last two DIIS cycles for GDM algorithm
   *
   * @param[in] delta_energy_diis Energy change from DIIS algorithm
   */
  void set_delta_energy_diis(const double delta_energy_diis);

 private:
  /// PIMPL pointer to implementation
  std::unique_ptr<impl::GDM> gdm_impl_;
};

}  // namespace qdk::chemistry::scf

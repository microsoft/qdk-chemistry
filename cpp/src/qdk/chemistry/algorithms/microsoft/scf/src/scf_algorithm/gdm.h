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
 * Reference: Troy Van Voorhis and Martin Head-Gordon (2002).
 *   doi: 10.1080/00268970110103642 :cite:`VanVoorhis2002`
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
   * @brief Initialize GDM state when switching from DIIS
   *
   * @param[in] delta_energy_diis Energy change from DIIS algorithm
   * @param[in] total_energy Current SCF total energy
   */
  void initialize_from_diis(const double delta_energy_diis,
                            const double total_energy);

 private:
  /// PIMPL pointer to implementation
  std::unique_ptr<impl::GDM> gdm_impl_;
};

}  // namespace qdk::chemistry::scf

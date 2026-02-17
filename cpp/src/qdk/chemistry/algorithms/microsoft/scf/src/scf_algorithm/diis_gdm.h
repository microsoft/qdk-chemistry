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

// Forward declarations
class DIIS;
class GDM;

/**
 * @brief Composite DIIS-GDM SCF algorithm class
 *
 * This class implements a hybrid SCF algorithm that starts with DIIS
 * (Direct Inversion in the Iterative Subspace) for rapid initial convergence
 * and automatically switches to GDM (Geometric Direct Minimization) when
 * DIIS encounters convergence difficulties. The algorithm uses internal
 * switching criteria based on energy changes and iteration counts.
 *
 * The switching logic:
 * - Starts with DIIS for initial iterations (up to gdm_max_diis_step)
 * - Switches to GDM if energy change exceeds energy_thresh_diis_switch
 * threshold
 * - Once switched to GDM, continues with GDM until convergence
 */
class DIIS_GDM : public SCFAlgorithm {
 public:
  /**
   * @brief Constructor for the DIIS-GDM composite algorithm
   * @param[in] ctx Reference to SCFContext
   * @param[in] subspace_size Maximum number of vectors to retain in DIIS
   * subspace (default: 8)
   * @param[in] gdm_config GDM configuration parameters
   *
   * @note If max_diis_step is less than 2, it will be automatically set to 2.
   * @note energy_thresh_diis_switch must be greater than 0.0.
   */
  explicit DIIS_GDM(const SCFContext& ctx, const size_t subspace_size,
                    const GDMConfig& gdm_config);

  /**
   * @brief Destructor
   */
  ~DIIS_GDM() noexcept;

  /**
   * @brief Perform one DIIS-GDM SCF iteration
   *
   * Decides whether to use DIIS or GDM based on internal switching criteria
   * and delegates to the appropriate algorithm implementation.
   *
   * @param[in,out] scf_impl Reference to SCFImpl containing all matrices and
   * energy
   */
  void iterate(SCFImpl& scf_impl) override;

 private:
  /// Pointer to DIIS algorithm instance
  std::unique_ptr<DIIS> diis_algorithm_;

  /// Pointer to GDM algorithm instance
  std::unique_ptr<GDM> gdm_algorithm_;

  // Algorithm switching parameters
  GDMConfig gdm_config_;  ///< GDM configuration parameters
  bool use_gdm_ = false;  ///< Flag indicating if currently using GDM

  /**
   * @brief Determine if we should switch from DIIS to GDM
   *
   * @param[in] delta_energy Energy change from previous iteration
   * @param[in] step Current iteration step
   * @return true if we should switch to GDM
   */
  bool should_switch_to_gdm_(double delta_energy, int step) const;
};

}  // namespace qdk::chemistry::scf

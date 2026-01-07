// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "diis_gdm.h"

#include <qdk/chemistry/scf/core/scf.h>

#include <cmath>
#include <limits>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>

#include "../scf/scf_impl.h"
#include "diis.h"
#include "gdm.h"

namespace qdk::chemistry::scf {

DIIS_GDM::DIIS_GDM(const SCFContext& ctx, const size_t subspace_size,
                   const GDMConfig& gdm_config)
    : SCFAlgorithm(ctx), gdm_config_(gdm_config), use_gdm_(false) {
  QDK_LOG_TRACE_ENTERING();
  // Initialize DIIS algorithm
  diis_algorithm_ = std::make_unique<DIIS>(ctx, subspace_size);

  // Validate energy_thresh_diis_switch must be positive
  if (gdm_config_.energy_thresh_diis_switch <= 0.0) {
    throw std::invalid_argument(
        "energy_thresh_diis_switch must be greater than 0.0, got: " +
        std::to_string(gdm_config_.energy_thresh_diis_switch));
  }

  // Ensure max_diis_step is at least 2
  if (gdm_config_.gdm_max_diis_iteration < 2) {
    gdm_config_.gdm_max_diis_iteration = 2;
    QDK_LOGGER().info("max_diis_step was < 2, set to 2");
  }

  // Initialize GDM algorithm
  gdm_algorithm_ = std::make_unique<GDM>(ctx, gdm_config_);

  // Log initialization parameters
  QDK_LOGGER().debug(
      "DIIS_GDM initialized: subspace_size={}, max_diis_step={}, "
      "energy_thresh_diis_switch={}",
      subspace_size, gdm_config_.gdm_max_diis_iteration,
      gdm_config_.energy_thresh_diis_switch);
}

DIIS_GDM::~DIIS_GDM() noexcept = default;

void DIIS_GDM::iterate(SCFImpl& scf_impl) {
  QDK_LOG_TRACE_ENTERING();
  // Determine if we should switch to GDM (note: step_count_ and delta_energy_
  // has been updated in check_convergence prior to this call)
  if (!use_gdm_ && should_switch_to_gdm_(delta_energy_, step_count_)) {
    use_gdm_ = true;
    QDK_LOGGER().info(
        "Switching from DIIS to GDM at step {} (delta_energy={}, "
        "max_diis_step={})",
        step_count_, delta_energy_, gdm_config_.gdm_max_diis_iteration);
    double total_energy = scf_impl.context().result.scf_total_energy;
    gdm_algorithm_->initialize_from_diis(std::abs(delta_energy_), total_energy);
  }

  // Delegate to appropriate algorithm
  if (use_gdm_) {
    gdm_algorithm_->iterate(scf_impl);
  } else {
    diis_algorithm_->iterate(scf_impl);
  }
}

bool DIIS_GDM::should_switch_to_gdm_(const double delta_energy,
                                     const int step) const {
  QDK_LOG_TRACE_ENTERING();
  // Switch conditions:
  // 1. Exceeded maximum DIIS steps
  if (step > gdm_config_.gdm_max_diis_iteration) {
    return true;
  }

  // 2. Energy change exceeds threshold (indicating DIIS difficulties)
  if (step > 1 &&
      std::abs(delta_energy) < gdm_config_.energy_thresh_diis_switch) {
    return true;
  }

  return false;
}

}  // namespace qdk::chemistry::scf

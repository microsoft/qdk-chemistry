// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "macis_base.hpp"

#include <qdk/chemistry/utils/logger.hpp>

#define SET_MACIS_SETTING(qdk_settings, macis_settings, param_name, type) \
  macis_settings.param_name = qdk_settings.get_or_default<type>(          \
      #param_name, macis_settings.param_name)

#define SET_MACIS_ENUM_SETTING(qdk_settings, macis_settings, param_name, \
                               converter)                                \
  if (qdk_settings.has(#param_name)) {                                   \
    macis_settings.param_name =                                          \
        converter(qdk_settings.get<std::string>(#param_name));           \
  }

namespace qdk::chemistry::algorithms::microsoft {

macis::CoreSelectionStrategy string_to_core_selection_strategy(
    const std::string& strategy) {
  if (strategy == "fixed") {
    return macis::CoreSelectionStrategy::Fixed;
  } else if (strategy == "percentage") {
    return macis::CoreSelectionStrategy::Percentage;
  } else {
    throw std::invalid_argument(
        "Invalid core_selection_strategy: '" + strategy +
        "'. Valid options are 'fixed' or 'percentage'.");
  }
}

macis::MCSCFSettings get_mcscf_settings_(const data::Settings& settings_) {
  QDK_LOG_TRACE_ENTERING();

  macis::MCSCFSettings mcscf_settings;
  // Respect MACIS native setting names
  if (settings_.has("ci_res_tol")) {
    SET_MACIS_SETTING(settings_, mcscf_settings, ci_res_tol, double);
  } else {
    mcscf_settings.ci_res_tol = settings_.get<double>("ci_residual_tolerance");
  }
  if (settings_.has("ci_max_subspace")) {
    SET_MACIS_SETTING(settings_, mcscf_settings, ci_max_subspace, size_t);
  } else {
    mcscf_settings.ci_max_subspace =
        settings_.get<int64_t>("davidson_iterations");
  }
  SET_MACIS_SETTING(settings_, mcscf_settings, ci_matel_tol, double);
  return mcscf_settings;
}

macis::ASCISettings get_asci_settings_(const data::Settings& settings_) {
  QDK_LOG_TRACE_ENTERING();

  macis::ASCISettings asci_settings;
  SET_MACIS_SETTING(settings_, asci_settings, ntdets_max, size_t);
  SET_MACIS_SETTING(settings_, asci_settings, ntdets_min, size_t);
  SET_MACIS_SETTING(settings_, asci_settings, ncdets_max, size_t);
  SET_MACIS_SETTING(settings_, asci_settings, h_el_tol, double);
  SET_MACIS_SETTING(settings_, asci_settings, rv_prune_tol, double);
  SET_MACIS_SETTING(settings_, asci_settings, pair_size_max, size_t);
  SET_MACIS_SETTING(settings_, asci_settings, pt2_tol, double);
  SET_MACIS_SETTING(settings_, asci_settings, pt2_reserve_count, size_t);
  SET_MACIS_SETTING(settings_, asci_settings, pt2_prune, bool);
  SET_MACIS_SETTING(settings_, asci_settings, pt2_precompute_eps, bool);
  SET_MACIS_SETTING(settings_, asci_settings, pt2_precompute_idx, bool);
  SET_MACIS_SETTING(settings_, asci_settings, pt2_print_progress, bool);
  SET_MACIS_SETTING(settings_, asci_settings, pt2_bigcon_thresh, size_t);
  SET_MACIS_SETTING(settings_, asci_settings, nxtval_bcount_thresh, size_t);
  SET_MACIS_SETTING(settings_, asci_settings, nxtval_bcount_inc, size_t);
  SET_MACIS_SETTING(settings_, asci_settings, just_singles, bool);
  SET_MACIS_SETTING(settings_, asci_settings, grow_factor, double);
  SET_MACIS_SETTING(settings_, asci_settings, min_grow_factor, double);
  SET_MACIS_SETTING(settings_, asci_settings, growth_backoff_rate, double);
  SET_MACIS_SETTING(settings_, asci_settings, growth_recovery_rate, double);
  SET_MACIS_SETTING(settings_, asci_settings, max_refine_iter, size_t);
  SET_MACIS_SETTING(settings_, asci_settings, refine_energy_tol, double);
  SET_MACIS_SETTING(settings_, asci_settings, grow_with_rot, bool);
  SET_MACIS_SETTING(settings_, asci_settings, rot_size_start, size_t);
  SET_MACIS_SETTING(settings_, asci_settings, constraint_level, int);
  SET_MACIS_SETTING(settings_, asci_settings, pt2_max_constraint_level, int);
  SET_MACIS_SETTING(settings_, asci_settings, pt2_min_constraint_level, int);
  SET_MACIS_SETTING(settings_, asci_settings, pt2_constraint_refine_force,
                    int64_t);
  SET_MACIS_ENUM_SETTING(settings_, asci_settings, core_selection_strategy,
                         string_to_core_selection_strategy);
  SET_MACIS_SETTING(settings_, asci_settings, core_selection_threshold, double);

  // Validate grow_factor and related parameters
  if (asci_settings.grow_factor <= 1.0) {
    throw std::runtime_error("grow_factor must be > 1.0, got " +
                             std::to_string(asci_settings.grow_factor));
  }
  if (asci_settings.min_grow_factor <= 1.0) {
    throw std::runtime_error("min_grow_factor must be > 1.0, got " +
                             std::to_string(asci_settings.min_grow_factor));
  }
  if (asci_settings.min_grow_factor > asci_settings.grow_factor) {
    throw std::runtime_error("min_grow_factor must be <= grow_factor");
  }
  if (asci_settings.growth_backoff_rate <= 0.0 ||
      asci_settings.growth_backoff_rate >= 1.0) {
    throw std::runtime_error("growth_backoff_rate must be in (0, 1), got " +
                             std::to_string(asci_settings.growth_backoff_rate));
  }
  if (asci_settings.growth_recovery_rate <= 1.0) {
    throw std::runtime_error(
        "growth_recovery_rate must be > 1.0, got " +
        std::to_string(asci_settings.growth_recovery_rate));
  }
  if (asci_settings.core_selection_strategy ==
          macis::CoreSelectionStrategy::Percentage &&
      (asci_settings.core_selection_threshold <
           std::numeric_limits<double>::epsilon() ||
       asci_settings.core_selection_threshold > 1.0)) {
    throw std::invalid_argument(
        "core_selection_threshold must be in [epsilon, 1.0], got " +
        std::to_string(asci_settings.core_selection_threshold));
  }

  return asci_settings;
}

}  // namespace qdk::chemistry::algorithms::microsoft

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <cmath>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

#include "microsoft/localization/mp2_natural_orbitals.hpp"
#include "microsoft/localization/natural_orbitals.hpp"
#include "microsoft/localization/pipek_mezey.hpp"
#include "microsoft/localization/qio.hpp"
#include "microsoft/localization/vvhv.hpp"

namespace qdk::chemistry::algorithms {

namespace detail {
/**
 * @brief Project a total-orbital configuration into an orbital active space.
 *
 * @param total_configuration Configuration over the full molecular orbital
 *        space.
 * @param orbitals Orbital basis whose active-space indices define the
 *        projection.
 * @return Configuration in the active orbital basis, or @p total_configuration
 *         when no active space is defined.
 * @throws std::invalid_argument If @p orbitals is null.
 */
data::Configuration _active_configuration_for_orbitals(
    const data::Configuration& total_configuration,
    const std::shared_ptr<data::Orbitals>& orbitals) {
  if (!orbitals) {
    throw std::invalid_argument("Orbitals pointer cannot be nullptr");
  }
  if (!orbitals->has_active_space()) {
    return total_configuration;
  }

  const auto active_space_indices = orbitals->get_active_space_indices();
  const auto& active_indices = active_space_indices.first;
  if (active_indices.empty()) {
    return data::Configuration::from_spin_half_string("");
  }

  const std::string total_str = total_configuration.to_string();
  std::string active_str;
  active_str.reserve(active_indices.size());
  for (size_t idx : active_indices) {
    active_str += idx < total_str.length() ? total_str[idx] : '0';
  }

  return data::Configuration::from_spin_half_string(active_str);
}

/**
 * @brief Build the canonical Aufbau determinant for an orbital basis.
 *
 * @param wavefunction Wavefunction providing total electron counts.
 * @param orbitals Orbital basis providing the number of molecular orbitals.
 * @return Canonical Aufbau configuration over the full orbital space.
 * @throws std::invalid_argument If @p wavefunction or @p orbitals is null.
 */
data::Configuration _aufbau_determinant_configuration(
    std::shared_ptr<data::Wavefunction> wavefunction,
    std::shared_ptr<data::Orbitals> orbitals) {
  QDK_LOG_TRACE_ENTERING();
  if (!wavefunction) {
    throw std::invalid_argument("Wavefunction pointer cannot be nullptr");
  }
  if (!orbitals) {
    throw std::invalid_argument("Orbitals pointer cannot be nullptr");
  }

  const auto [nalpha, nbeta] = wavefunction->get_total_num_electrons();
  const size_t num_orbitals = orbitals->get_num_molecular_orbitals();
  return data::Configuration::canonical_hf_configuration(nalpha, nbeta,
                                                         num_orbitals);
}

bool is_aufbau_determinant_wavefunction(
    std::shared_ptr<data::Wavefunction> wavefunction) {
  QDK_LOG_TRACE_ENTERING();
  if (!wavefunction) {
    throw std::invalid_argument("Wavefunction pointer cannot be nullptr");
  }

  try {
    if (wavefunction->size() != 1) {
      return false;
    }

    const auto expected_det = _aufbau_determinant_configuration(
        wavefunction, wavefunction->get_orbitals());
    const auto total_determinants = wavefunction->get_total_determinants();
    if (total_determinants.size() != 1) {
      return false;
    }

    return total_determinants[0] == expected_det;
  } catch (const std::exception&) {
    return false;
  }
}

void warn_if_not_aufbau_determinant_wavefunction(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::string& localizer_name) {
  QDK_LOG_TRACE_ENTERING();
  if (!is_aufbau_determinant_wavefunction(wavefunction)) {
    QDK_LOGGER().warn(
        "{} received a wavefunction that is not the single Aufbau "
        "determinant. The returned wavefunction will contain a single "
        "Aufbau determinant built from the transformed orbitals; "
        "correlated-state coefficients are not preserved.",
        localizer_name);
  }
}

std::shared_ptr<data::Wavefunction> new_aufbau_determinant_wavefunction(
    std::shared_ptr<data::Wavefunction> wavefunction,
    std::shared_ptr<data::Orbitals> new_orbitals,
    std::optional<data::ContainerTypes::MatrixVariant> one_rdm_spin_traced) {
  QDK_LOG_TRACE_ENTERING();
  if (!wavefunction) {
    throw std::invalid_argument("Wavefunction pointer cannot be nullptr");
  }
  if (!new_orbitals) {
    throw std::invalid_argument("New orbitals pointer cannot be nullptr");
  }

  auto aufbau_det = _active_configuration_for_orbitals(
      _aufbau_determinant_configuration(wavefunction, new_orbitals),
      new_orbitals);
  if (one_rdm_spin_traced) {
    Eigen::VectorXd coeffs = Eigen::VectorXd::Ones(1);
    data::ContainerTypes::DeterminantVector determinants{aufbau_det};
    auto new_container = std::make_unique<data::StateVectorContainer>(
        data::ContainerTypes::VectorVariant(coeffs), determinants, new_orbitals,
        one_rdm_spin_traced, std::nullopt, "electrons",
        data::OrbitalEntropies{}, wavefunction->get_type());
    return std::make_shared<data::Wavefunction>(std::move(new_container));
  }

  auto new_container = std::make_unique<data::StateVectorContainer>(
      aufbau_det, new_orbitals, "electrons", wavefunction->get_type());
  return std::make_shared<data::Wavefunction>(std::move(new_container));
}
}  // namespace detail

std::unique_ptr<Localizer> make_pipek_mezey_localizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::PipekMezeyLocalizer>();
}

std::unique_ptr<Localizer> make_mp2_natural_orbital_localizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::MP2NaturalOrbitalLocalizer>();
}

std::unique_ptr<Localizer> make_natural_orbital_localizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::NaturalOrbitalLocalizer>();
}

std::unique_ptr<Localizer> make_qio_localizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::QIOLocalizer>();
}

std::unique_ptr<Localizer> make_vvhv_localizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::VVHVLocalizer>();
}

void LocalizerFactory::register_default_instances() {
  QDK_LOG_TRACE_ENTERING();

  LocalizerFactory::register_instance(&make_pipek_mezey_localizer);
  LocalizerFactory::register_instance(&make_mp2_natural_orbital_localizer);
  LocalizerFactory::register_instance(&make_natural_orbital_localizer);
  LocalizerFactory::register_instance(&make_qio_localizer);
  LocalizerFactory::register_instance(&make_vvhv_localizer);
}

}  // namespace qdk::chemistry::algorithms

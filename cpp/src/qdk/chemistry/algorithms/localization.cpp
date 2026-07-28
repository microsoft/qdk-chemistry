// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <cmath>
#include <qdk/chemistry/algorithms/localization.hpp>
#include <qdk/chemistry/config.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <variant>

#include "microsoft/localization/mp2_natural_orbitals.hpp"
#include "microsoft/localization/natural_orbitals.hpp"
#include "microsoft/localization/pipek_mezey.hpp"
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

  const auto active_index_set = orbitals->active_indices();
  const auto active_indices =
      data::spin_channel_indices(active_index_set, data::axes::alpha());
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

Eigen::MatrixXd replace_orbital_columns(const Eigen::MatrixXd& coefficients,
                                        const std::vector<size_t>& indices,
                                        const Eigen::MatrixXd& replacements) {
  if (replacements.rows() != coefficients.rows() ||
      replacements.cols() != static_cast<Eigen::Index>(indices.size())) {
    throw std::invalid_argument(
        "Replacement orbital coefficient dimensions do not match the "
        "selected columns");
  }

  std::vector<bool> seen(static_cast<size_t>(coefficients.cols()), false);
  Eigen::MatrixXd transformed = coefficients;
  for (size_t i = 0; i < indices.size(); ++i) {
    const size_t index = indices[i];
    if (index >= static_cast<size_t>(coefficients.cols())) {
      throw std::invalid_argument("Orbital column index is out of bounds");
    }
    if (seen[index]) {
      throw std::invalid_argument("Orbital column indices must be unique");
    }
    seen[index] = true;
    transformed.col(static_cast<Eigen::Index>(index)) =
        replacements.col(static_cast<Eigen::Index>(i));
  }
  return transformed;
}

std::shared_ptr<data::Orbitals> make_transformed_orbitals(
    const std::shared_ptr<data::Orbitals>& orbitals,
    const Eigen::MatrixXd& coefficients_alpha) {
  if (!orbitals) {
    throw std::invalid_argument("Orbitals pointer cannot be nullptr");
  }

  const auto& source_alpha = orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  if (coefficients_alpha.rows() != source_alpha.rows() ||
      coefficients_alpha.cols() != source_alpha.cols()) {
    throw std::invalid_argument(
        "Transformed alpha coefficient dimensions do not match the source "
        "orbitals");
  }

  if (!orbitals->is_restricted()) {
    const auto active_indices_alpha = data::spin_channel_indices(
        orbitals->active_indices(), data::axes::alpha());
    const auto active_indices_beta = data::spin_channel_indices(
        orbitals->active_indices(), data::axes::beta());
    const auto inactive_indices_alpha = data::spin_channel_indices(
        orbitals->inactive_indices(), data::axes::alpha());
    const auto inactive_indices_beta = data::spin_channel_indices(
        orbitals->inactive_indices(), data::axes::beta());
    if (active_indices_alpha != active_indices_beta ||
        inactive_indices_alpha != inactive_indices_beta) {
      throw std::invalid_argument(
          "Cannot construct restricted transformed orbitals from mismatched "
          "alpha and beta active or inactive spaces");
    }
    const data::Orbitals::RestrictedCASIndices restricted_indices =
        std::make_tuple(std::vector<size_t>(active_indices_alpha.begin(),
                                            active_indices_alpha.end()),
                        std::vector<size_t>(inactive_indices_alpha.begin(),
                                            inactive_indices_alpha.end()));
    return std::make_shared<data::Orbitals>(
        coefficients_alpha,
        std::nullopt,  // no energies for transformed orbitals
        orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
        restricted_indices);
  }

  return std::make_shared<data::Orbitals>(
      coefficients_alpha,
      std::nullopt,  // no energies for transformed orbitals
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      orbitals->active_indices(), orbitals->inactive_indices());
}

std::shared_ptr<data::Orbitals> make_transformed_orbitals(
    const std::shared_ptr<data::Orbitals>& orbitals,
    const Eigen::MatrixXd& coefficients_alpha,
    const Eigen::MatrixXd& coefficients_beta) {
  if (!orbitals) {
    throw std::invalid_argument("Orbitals pointer cannot be nullptr");
  }

  const auto& source_alpha = orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  const auto& source_beta =
      orbitals->coefficients()->block({data::axes::beta(), data::axes::beta()});
  if (coefficients_alpha.rows() != source_alpha.rows() ||
      coefficients_alpha.cols() != source_alpha.cols()) {
    throw std::invalid_argument(
        "Transformed alpha coefficient dimensions do not match the source "
        "orbitals");
  }
  if (coefficients_beta.rows() != source_beta.rows() ||
      coefficients_beta.cols() != source_beta.cols()) {
    throw std::invalid_argument(
        "Transformed beta coefficient dimensions do not match the source "
        "orbitals");
  }

  return std::make_shared<data::Orbitals>(
      coefficients_alpha, coefficients_beta,
      std::nullopt,  // no alpha energies for transformed orbitals
      std::nullopt,  // no beta energies for transformed orbitals
      orbitals->get_overlap_matrix(), orbitals->get_basis_set(),
      orbitals->active_indices(), orbitals->inactive_indices());
}

std::shared_ptr<data::Wavefunction> new_aufbau_determinant_wavefunction(
    std::shared_ptr<data::Wavefunction> wavefunction,
    std::shared_ptr<data::Orbitals> new_orbitals,
    const std::optional<data::ContainerTypes::MatrixVariant>&
        one_rdm_spin_traced,
    const std::optional<data::ContainerTypes::MatrixVariant>& one_rdm_aa,
    const std::optional<data::ContainerTypes::MatrixVariant>& one_rdm_bb) {
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
  if (one_rdm_spin_traced || one_rdm_aa || one_rdm_bb) {
    Eigen::VectorXd coeffs = Eigen::VectorXd::Ones(1);
    data::ContainerTypes::DeterminantVector determinants{aufbau_det};
    auto new_container = std::make_unique<data::StateVectorContainer>(
        data::ContainerTypes::VectorVariant(coeffs), determinants, new_orbitals,
        one_rdm_spin_traced, one_rdm_aa, one_rdm_bb,
        std::nullopt,  // two_rdm_spin_traced
        std::nullopt,  // two_rdm_aaaa
        std::nullopt,  // two_rdm_aabb
        std::nullopt,  // two_rdm_bbbb
        "electrons", data::OrbitalEntropies{}, wavefunction->get_type());
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

// MP2NaturalOrbitalLocalizer is deprecated (superseded by
// NaturalOrbitalLocalizer), but this factory intentionally still provides it
// through the localizer registry. Suppress the self-referential deprecation
// warning at this facade site only.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
std::unique_ptr<Localizer> make_mp2_natural_orbital_localizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::MP2NaturalOrbitalLocalizer>();
}
#pragma GCC diagnostic pop

std::unique_ptr<Localizer> make_natural_orbital_localizer() {
  QDK_LOG_TRACE_ENTERING();

  return std::make_unique<microsoft::NaturalOrbitalLocalizer>();
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
  LocalizerFactory::register_instance(&make_vvhv_localizer);
}

}  // namespace qdk::chemistry::algorithms

/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <algorithm>
#include <cmath>
#include <macis/sd_operations.hpp>
#include <macis/util/rdms.hpp>
#include <optional>
#include <qdk/chemistry/data/wavefunction_containers/cc.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <set>
#include <stdexcept>
#include <variant>

#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

CoupledClusterContainer::CoupledClusterContainer(
    std::shared_ptr<Orbitals> orbitals,
    std::shared_ptr<Wavefunction> wavefunction,
    const std::optional<VectorVariant>& t1_amplitudes,
    const std::optional<VectorVariant>& t2_amplitudes)
    : CoupledClusterContainer(orbitals, wavefunction, t1_amplitudes,
                              std::nullopt, t2_amplitudes, std::nullopt,
                              std::nullopt) {
  QDK_LOG_TRACE_ENTERING();
}

CoupledClusterContainer::CoupledClusterContainer(
    std::shared_ptr<Orbitals> orbitals,
    std::shared_ptr<Wavefunction> wavefunction,
    const std::optional<VectorVariant>& t1_amplitudes_aa,
    const std::optional<VectorVariant>& t1_amplitudes_bb,
    const std::optional<VectorVariant>& t2_amplitudes_abab,
    const std::optional<VectorVariant>& t2_amplitudes_aaaa,
    const std::optional<VectorVariant>& t2_amplitudes_bbbb)
    : WavefunctionContainer(
          WavefunctionType::NotSelfDual),  // Always force NotSelfDual for CC
      _wavefunction(wavefunction),
      _orbitals(orbitals) {
  QDK_LOG_TRACE_ENTERING();
  if (!orbitals) {
    throw std::invalid_argument("Orbitals cannot be null");
  }

  if (!wavefunction) {
    throw std::invalid_argument("Wavefunction cannot be null");
  }

  // Validate amplitude inputs
  if (!t1_amplitudes_aa && t1_amplitudes_bb) {
    throw std::invalid_argument(
        "Cannot provide unrestricted beta T1 amplitudes without alpha T1 "
        "amplitudes");
  }
  if (!t2_amplitudes_abab && (t2_amplitudes_aaaa || t2_amplitudes_bbbb)) {
    throw std::invalid_argument(
        "Cannot provide unrestricted T2 alpha-alpha or beta-beta "
        "amplitudes "
        "without "
        "T2 alpha-beta amplitudes");
  }

  // unrestricted orbitals require both alpha and beta amplitudes
  if (!orbitals->is_restricted()) {
    if (!t1_amplitudes_aa || !t1_amplitudes_bb) {
      throw std::invalid_argument(
          "Both alpha and beta T1 amplitudes must be provided for "
          "unrestricted orbitals");
    } else if (!t2_amplitudes_abab || !t2_amplitudes_aaaa ||
               !t2_amplitudes_bbbb) {
      throw std::invalid_argument(
          "All spin components of T2 amplitudes must be provided for "
          "unrestricted orbitals");
    }
  }

  // Helper lambda to get size from VectorVariant
  auto get_vector_size = [](const VectorVariant& vec) -> size_t {
    if (std::holds_alternative<Eigen::VectorXd>(vec)) {
      return std::get<Eigen::VectorXd>(vec).size();
    } else if (std::holds_alternative<Eigen::VectorXcd>(vec)) {
      return std::get<Eigen::VectorXcd>(vec).size();
    }
    return 0;
  };

  // Get number of occupied and virtual orbitals from reference determinant
  const auto& references = _wavefunction->get_total_determinants();
  auto [n_alpha, n_beta] = references[0].get_n_electrons();
  size_t active_space_size = orbitals->get_num_molecular_orbitals();

  // For restricted case, use alpha counts
  size_t n_occ_alpha = n_alpha;
  size_t n_occ_beta = n_beta;
  size_t n_vir_alpha = active_space_size - n_occ_alpha;
  size_t n_vir_beta = active_space_size - n_occ_beta;

  // Validate T1 amplitude sizes
  if (t1_amplitudes_aa) {
    size_t expected_t1_aa_size = n_occ_alpha * n_vir_alpha;
    size_t actual_t1_aa_size = get_vector_size(*t1_amplitudes_aa);
    if (actual_t1_aa_size != expected_t1_aa_size) {
      throw std::invalid_argument(
          "T1 alpha amplitude size mismatch: expected " +
          std::to_string(expected_t1_aa_size) +
          " (nocc=" + std::to_string(n_occ_alpha) +
          " * nvir=" + std::to_string(n_vir_alpha) + "), got " +
          std::to_string(actual_t1_aa_size));
    }
  }

  if (t1_amplitudes_bb) {
    size_t expected_t1_bb_size = n_occ_beta * n_vir_beta;
    size_t actual_t1_bb_size = get_vector_size(*t1_amplitudes_bb);
    if (actual_t1_bb_size != expected_t1_bb_size) {
      throw std::invalid_argument("T1 beta amplitude size mismatch: expected " +
                                  std::to_string(expected_t1_bb_size) +
                                  " (nocc=" + std::to_string(n_occ_beta) +
                                  " * nvir=" + std::to_string(n_vir_beta) +
                                  "), got " +
                                  std::to_string(actual_t1_bb_size));
    }
  }

  // Validate T2 amplitude sizes
  if (t2_amplitudes_abab) {
    size_t expected_t2_abab_size =
        n_occ_alpha * n_occ_beta * n_vir_alpha * n_vir_beta;
    size_t actual_t2_abab_size = get_vector_size(*t2_amplitudes_abab);
    if (actual_t2_abab_size != expected_t2_abab_size) {
      throw std::invalid_argument(
          "T2 alpha-beta amplitude size mismatch: expected " +
          std::to_string(expected_t2_abab_size) +
          " (nocc_a=" + std::to_string(n_occ_alpha) +
          " * nocc_b=" + std::to_string(n_occ_beta) +
          " * nvir_a=" + std::to_string(n_vir_alpha) +
          " * nvir_b=" + std::to_string(n_vir_beta) + "), got " +
          std::to_string(actual_t2_abab_size));
    }
  }

  if (t2_amplitudes_aaaa) {
    size_t expected_t2_aaaa_size =
        n_occ_alpha * n_occ_alpha * n_vir_alpha * n_vir_alpha;
    size_t actual_t2_aaaa_size = get_vector_size(*t2_amplitudes_aaaa);
    if (actual_t2_aaaa_size != expected_t2_aaaa_size) {
      throw std::invalid_argument(
          "T2 alpha-alpha amplitude size mismatch: expected " +
          std::to_string(expected_t2_aaaa_size) +
          " (nocc=" + std::to_string(n_occ_alpha) +
          " * nocc=" + std::to_string(n_occ_alpha) +
          " * nvir=" + std::to_string(n_vir_alpha) +
          " * nvir=" + std::to_string(n_vir_alpha) + "), got " +
          std::to_string(actual_t2_aaaa_size));
    }
  }

  if (t2_amplitudes_bbbb) {
    size_t expected_t2_bbbb_size =
        n_occ_beta * n_occ_beta * n_vir_beta * n_vir_beta;
    size_t actual_t2_bbbb_size = get_vector_size(*t2_amplitudes_bbbb);
    if (actual_t2_bbbb_size != expected_t2_bbbb_size) {
      throw std::invalid_argument(
          "T2 beta-beta amplitude size mismatch: expected " +
          std::to_string(expected_t2_bbbb_size) + " (nocc=" +
          std::to_string(n_occ_beta) + " * nocc=" + std::to_string(n_occ_beta) +
          " * nvir=" + std::to_string(n_vir_beta) +
          " * nvir=" + std::to_string(n_vir_beta) + "), got " +
          std::to_string(actual_t2_bbbb_size));
    }
  }

  // Store spin-separated amplitudes if provided
  if (t1_amplitudes_aa) {
    _t1_amplitudes_aa = std::make_shared<VectorVariant>(*t1_amplitudes_aa);
  }
  if (t1_amplitudes_bb) {
    _t1_amplitudes_bb = std::make_shared<VectorVariant>(*t1_amplitudes_bb);
  } else {
    _t1_amplitudes_bb = _t1_amplitudes_aa;
  }
  if (t2_amplitudes_abab) {
    _t2_amplitudes_abab = std::make_shared<VectorVariant>(*t2_amplitudes_abab);
  }
  if (t2_amplitudes_aaaa) {
    _t2_amplitudes_aaaa = std::make_shared<VectorVariant>(*t2_amplitudes_aaaa);
  } else {
    _t2_amplitudes_aaaa = _t2_amplitudes_abab;
  }
  if (t2_amplitudes_bbbb) {
    _t2_amplitudes_bbbb = std::make_shared<VectorVariant>(*t2_amplitudes_bbbb);
  } else {
    _t2_amplitudes_bbbb = _t2_amplitudes_abab;
  }
}

std::unique_ptr<WavefunctionContainer> CoupledClusterContainer::clone() const {
  QDK_LOG_TRACE_ENTERING();
  // Create optional variants for the amplitudes
  std::optional<VectorVariant> t1_aa =
      _t1_amplitudes_aa ? std::optional<VectorVariant>(*_t1_amplitudes_aa)
                        : std::nullopt;
  std::optional<VectorVariant> t1_bb =
      _t1_amplitudes_bb ? std::optional<VectorVariant>(*_t1_amplitudes_bb)
                        : std::nullopt;
  std::optional<VectorVariant> t2_abab =
      _t2_amplitudes_abab ? std::optional<VectorVariant>(*_t2_amplitudes_abab)
                          : std::nullopt;
  std::optional<VectorVariant> t2_aaaa =
      _t2_amplitudes_aaaa ? std::optional<VectorVariant>(*_t2_amplitudes_aaaa)
                          : std::nullopt;
  std::optional<VectorVariant> t2_bbbb =
      _t2_amplitudes_bbbb ? std::optional<VectorVariant>(*_t2_amplitudes_bbbb)
                          : std::nullopt;

  return std::make_unique<CoupledClusterContainer>(
      _orbitals, _wavefunction, t1_aa, t1_bb, t2_abab, t2_aaaa, t2_bbbb);
}

std::shared_ptr<Orbitals> CoupledClusterContainer::get_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return _orbitals;
}

std::shared_ptr<Wavefunction> CoupledClusterContainer::get_wavefunction()
    const {
  QDK_LOG_TRACE_ENTERING();
  return _wavefunction;
}

const CoupledClusterContainer::VectorVariant&
CoupledClusterContainer::get_coefficients() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_coefficients_cache) {
    _generate_ci_expansion();
  }
  return *_coefficients_cache;
}

CoupledClusterContainer::ScalarVariant CoupledClusterContainer::get_coefficient(
    const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  if (!_coefficients_cache || !_determinant_vector_cache) {
    _generate_ci_expansion();
  }

  // Find the determinant in the cache
  auto it = std::find(_determinant_vector_cache->begin(),
                      _determinant_vector_cache->end(), det);
  if (it == _determinant_vector_cache->end()) {
    // Return zero if determinant not found
    if (is_complex()) {
      return std::complex<double>(0.0, 0.0);
    } else {
      return 0.0;
    }
  }

  size_t idx = std::distance(_determinant_vector_cache->begin(), it);
  if (is_complex()) {
    return std::get<Eigen::VectorXcd>(*_coefficients_cache)(idx);
  } else {
    return std::get<Eigen::VectorXd>(*_coefficients_cache)(idx);
  }
}

const CoupledClusterContainer::DeterminantVector&
CoupledClusterContainer::get_active_determinants() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_determinant_vector_cache) {
    _generate_ci_expansion();
  }
  return *_determinant_vector_cache;
}

std::pair<const CoupledClusterContainer::VectorVariant&,
          const CoupledClusterContainer::VectorVariant&>
CoupledClusterContainer::get_t1_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_t1_amplitudes()) {
    throw std::runtime_error("T1 amplitudes not available");
  }
  return std::make_pair(std::cref(*_t1_amplitudes_aa),
                        std::cref(*_t1_amplitudes_bb));
}

std::tuple<const CoupledClusterContainer::VectorVariant&,
           const CoupledClusterContainer::VectorVariant&,
           const CoupledClusterContainer::VectorVariant&>
CoupledClusterContainer::get_t2_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_t2_amplitudes()) {
    throw std::runtime_error("T2 amplitudes not available");
  }
  return std::make_tuple(std::cref(*_t2_amplitudes_abab),
                         std::cref(*_t2_amplitudes_aaaa),
                         std::cref(*_t2_amplitudes_bbbb));
}

bool CoupledClusterContainer::has_t1_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  return _t1_amplitudes_aa != nullptr;
}

bool CoupledClusterContainer::has_t2_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  return _t2_amplitudes_abab != nullptr;
}

size_t CoupledClusterContainer::size() const {
  QDK_LOG_TRACE_ENTERING();
  if (!_determinant_vector_cache) {
    _generate_ci_expansion();
  }
  return _determinant_vector_cache->size();
}

CoupledClusterContainer::ScalarVariant CoupledClusterContainer::overlap(
    const WavefunctionContainer& other) const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "overlap() is not implemented for coupled cluster wavefunctions. ");
}

double CoupledClusterContainer::norm() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "norm() is not implemented for coupled cluster wavefunctions. ");
}

bool CoupledClusterContainer::contains_determinant(
    const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  return contains_reference(det);
}

bool CoupledClusterContainer::contains_reference(
    const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  const auto& references = _wavefunction->get_total_determinants();
  return std::find(references.begin(), references.end(), det) !=
         references.end();
}

void CoupledClusterContainer::clear_caches() const {
  QDK_LOG_TRACE_ENTERING();
  // Clear the cached determinant vector and coefficients
  _determinant_vector_cache.reset();
  _coefficients_cache.reset();

  // Clear all cached RDMs using base class helper
  _clear_rdms();
}

nlohmann::json CoupledClusterContainer::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;

  j["version"] = SERIALIZATION_VERSION;
  j["container_type"] = get_container_type();

  // Serialize orbitals
  if (_orbitals) {
    j["orbitals"] = _orbitals->to_json();
  }

  // Serialize wfn
  if (_wavefunction) {
    j["wavefunction"] = _wavefunction->to_json();
  }

  bool is_complex = this->is_complex();
  j["is_complex"] = is_complex;
  if (is_complex) {
    if (_t1_amplitudes_aa) {
      const auto& t1_aa = std::get<Eigen::VectorXcd>(*_t1_amplitudes_aa);
      j["t1_amplitudes_aa"] = vector_variant_to_json(t1_aa, is_complex);
    }
    if (_t1_amplitudes_bb) {
      const auto& t1_bb = std::get<Eigen::VectorXcd>(*_t1_amplitudes_bb);
      j["t1_amplitudes_bb"] = vector_variant_to_json(t1_bb, is_complex);
    }
    if (_t2_amplitudes_abab) {
      const auto& t2_abab = std::get<Eigen::VectorXcd>(*_t2_amplitudes_abab);
      j["t2_amplitudes_abab"] = vector_variant_to_json(t2_abab, is_complex);
    }
    if (_t2_amplitudes_aaaa) {
      const auto& t2_aaaa = std::get<Eigen::VectorXcd>(*_t2_amplitudes_aaaa);
      j["t2_amplitudes_aaaa"] = vector_variant_to_json(t2_aaaa, is_complex);
    }
    if (_t2_amplitudes_bbbb) {
      const auto& t2_bbbb = std::get<Eigen::VectorXcd>(*_t2_amplitudes_bbbb);
      j["t2_amplitudes_bbbb"] = vector_variant_to_json(t2_bbbb, is_complex);
    }
  } else {
    if (_t1_amplitudes_aa) {
      const auto& t1_aa = std::get<Eigen::VectorXd>(*_t1_amplitudes_aa);
      j["t1_amplitudes_aa"] = vector_variant_to_json(t1_aa, is_complex);
    }
    if (_t1_amplitudes_bb) {
      const auto& t1_bb = std::get<Eigen::VectorXd>(*_t1_amplitudes_bb);
      j["t1_amplitudes_bb"] = vector_variant_to_json(t1_bb, is_complex);
    }
    if (_t2_amplitudes_abab) {
      const auto& t2_abab = std::get<Eigen::VectorXd>(*_t2_amplitudes_abab);
      j["t2_amplitudes_abab"] = vector_variant_to_json(t2_abab, is_complex);
    }
    if (_t2_amplitudes_aaaa) {
      const auto& t2_aaaa = std::get<Eigen::VectorXd>(*_t2_amplitudes_aaaa);
      j["t2_amplitudes_aaaa"] = vector_variant_to_json(t2_aaaa, is_complex);
    }
    if (_t2_amplitudes_bbbb) {
      const auto& t2_bbbb = std::get<Eigen::VectorXd>(*_t2_amplitudes_bbbb);
      j["t2_amplitudes_bbbb"] = vector_variant_to_json(t2_bbbb, is_complex);
    }
  }

  return j;
}

std::unique_ptr<CoupledClusterContainer> CoupledClusterContainer::from_json(
    const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  try {
    if (!j.contains("version")) {
      throw std::runtime_error("JSON does not contain version information");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    auto orbitals = Orbitals::from_json(j["orbitals"]);
    auto wavefunction = Wavefunction::from_json(j["wavefunction"]);

    bool is_complex = j.value("is_complex", false);
    return std::make_unique<CoupledClusterContainer>(
        orbitals, wavefunction,
        j.contains("t1_amplitudes_aa")
            ? std::optional<VectorVariant>(
                  json_to_vector_variant(j["t1_amplitudes_aa"], is_complex))
            : std::nullopt,
        j.contains("t1_amplitudes_bb")
            ? std::optional<VectorVariant>(
                  json_to_vector_variant(j["t1_amplitudes_bb"], is_complex))
            : std::nullopt,
        j.contains("t2_amplitudes_abab")
            ? std::optional<VectorVariant>(
                  json_to_vector_variant(j["t2_amplitudes_abab"], is_complex))
            : std::nullopt,
        j.contains("t2_amplitudes_aaaa")
            ? std::optional<VectorVariant>(
                  json_to_vector_variant(j["t2_amplitudes_aaaa"], is_complex))
            : std::nullopt,
        j.contains("t2_amplitudes_bbbb")
            ? std::optional<VectorVariant>(
                  json_to_vector_variant(j["t2_amplitudes_bbbb"], is_complex))
            : std::nullopt);
  } catch (const std::exception& e) {
    throw std::runtime_error(
        "Failed to parse CoupledClusterContainer from JSON: " +
        std::string(e.what()));
  }
}

void CoupledClusterContainer::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // version
    H5::Attribute version_attr = group.createAttribute(
        "version", string_type, H5::DataSpace(H5S_SCALAR));
    std::string version_str = SERIALIZATION_VERSION;
    version_attr.write(string_type, version_str);
    version_attr.close();

    // container type
    std::string container_type = get_container_type();
    H5::Attribute container_type_attr = group.createAttribute(
        "container_type", string_type, H5::DataSpace(H5S_SCALAR));
    container_type_attr.write(string_type, container_type);

    // complex flag
    bool is_complex = this->is_complex();
    H5::Attribute is_complex_attr = group.createAttribute(
        "is_complex", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));
    is_complex_attr.write(H5::PredType::NATIVE_HBOOL, &is_complex);

    //  store amplitudes
    if (_t1_amplitudes_aa) {
      write_vector_to_hdf5(group, "t1_amplitudes_aa", _t1_amplitudes_aa,
                           is_complex);
    }
    if (_t1_amplitudes_bb) {
      write_vector_to_hdf5(group, "t1_amplitudes_bb", _t1_amplitudes_bb,
                           is_complex);
    }
    if (_t2_amplitudes_abab) {
      write_vector_to_hdf5(group, "t2_amplitudes_abab", _t2_amplitudes_abab,
                           is_complex);
    }
    if (_t2_amplitudes_aaaa) {
      write_vector_to_hdf5(group, "t2_amplitudes_aaaa", _t2_amplitudes_aaaa,
                           is_complex);
    }
    if (_t2_amplitudes_bbbb) {
      write_vector_to_hdf5(group, "t2_amplitudes_bbbb", _t2_amplitudes_bbbb,
                           is_complex);
    }

    // Store orbitals if available
    if (_orbitals) {
      H5::Group orbitals_group = group.createGroup("orbitals");
      _orbitals->to_hdf5(orbitals_group);
    }

    // Store wfn if available
    if (_wavefunction) {
      H5::Group wavefunction_group = group.createGroup("wavefunction");
      _wavefunction->to_hdf5(wavefunction_group);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::unique_ptr<CoupledClusterContainer> CoupledClusterContainer::from_hdf5(
    H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    // version
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version_str;
    version_attr.read(string_type, version_str);
    validate_serialization_version(SERIALIZATION_VERSION, version_str);

    // complex flag
    bool is_complex = false;
    if (group.attrExists("is_complex")) {
      H5::Attribute is_complex_attr = group.openAttribute("is_complex");
      hbool_t is_complex_flag;
      is_complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_complex = (is_complex_flag != 0);
    }

    // Load orbitals (if available)
    std::shared_ptr<Orbitals> orbitals = nullptr;
    if (group.nameExists("orbitals")) {
      H5::Group orbitals_group = group.openGroup("orbitals");
      orbitals = Orbitals::from_hdf5(orbitals_group);
    }

    // Load wavefunction (if available)
    std::shared_ptr<Wavefunction> wavefunction = nullptr;
    if (group.nameExists("wavefunction")) {
      H5::Group wavefunction_group = group.openGroup("wavefunction");
      wavefunction = Wavefunction::from_hdf5(wavefunction_group);
    }

    auto t1_aa =
        group.nameExists("t1_amplitudes_aa")
            ? std::optional<VectorVariant>(load_vector_variant_from_group(
                  group, "t1_amplitudes_aa", is_complex))
            : std::nullopt;
    auto t1_bb =
        group.nameExists("t1_amplitudes_bb")
            ? std::optional<VectorVariant>(load_vector_variant_from_group(
                  group, "t1_amplitudes_bb", is_complex))
            : std::nullopt;
    auto t2_abab =
        group.nameExists("t2_amplitudes_abab")
            ? std::optional<VectorVariant>(load_vector_variant_from_group(
                  group, "t2_amplitudes_abab", is_complex))
            : std::nullopt;
    auto t2_aaaa =
        group.nameExists("t2_amplitudes_aaaa")
            ? std::optional<VectorVariant>(load_vector_variant_from_group(
                  group, "t2_amplitudes_aaaa", is_complex))
            : std::nullopt;
    auto t2_bbbb =
        group.nameExists("t2_amplitudes_bbbb")
            ? std::optional<VectorVariant>(load_vector_variant_from_group(
                  group, "t2_amplitudes_bbbb", is_complex))
            : std::nullopt;

    return std::make_unique<CoupledClusterContainer>(
        orbitals, wavefunction, t1_aa, t1_bb, t2_abab, t2_aaaa, t2_bbbb);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::pair<size_t, size_t> CoupledClusterContainer::get_total_num_electrons()
    const {
  QDK_LOG_TRACE_ENTERING();
  // Get active space electrons using the dedicated method
  auto [n_alpha_active, n_beta_active] = get_active_num_electrons();

  // Add electrons from inactive space (doubly occupied orbitals)
  auto [alpha_inactive_indices, beta_inactive_indices] =
      get_orbitals()->get_inactive_space_indices();

  size_t n_alpha_total = n_alpha_active + alpha_inactive_indices.size();
  size_t n_beta_total = n_beta_active + beta_inactive_indices.size();

  return {n_alpha_total, n_beta_total};
}

std::pair<size_t, size_t> CoupledClusterContainer::get_active_num_electrons()
    const {
  QDK_LOG_TRACE_ENTERING();
  const auto& determinants = _wavefunction->get_total_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }
  auto [n_alpha, n_beta] = determinants[0].get_n_electrons();
  return {n_alpha, n_beta};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
CoupledClusterContainer::get_total_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  // Orbital occupations require RDMs which are not available in CC container
  throw std::runtime_error(
      "Coupled cluster orbital occupations require the adjoint (bra) "
      "wavefunction. Use a coupled cluster method that computes lambda "
      "amplitudes to obtain RDMs and orbital occupations.");
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
CoupledClusterContainer::get_active_orbital_occupations() const {
  // Orbital occupations require RDMs which are not available in CC container
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "Coupled cluster orbital occupations require the adjoint (bra) "
      "wavefunction. Use a coupled cluster method that computes lambda "
      "amplitudes to obtain RDMs and orbital occupations.");
}

std::string CoupledClusterContainer::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return "coupled_cluster";
}

bool CoupledClusterContainer::is_complex() const {
  QDK_LOG_TRACE_ENTERING();
  // Check if any amplitude is complex
  if (_t1_amplitudes_aa &&
      std::holds_alternative<Eigen::VectorXcd>(*_t1_amplitudes_aa)) {
    return true;
  }
  if (_t1_amplitudes_bb &&
      std::holds_alternative<Eigen::VectorXcd>(*_t1_amplitudes_bb)) {
    return true;
  }
  if (_t2_amplitudes_abab &&
      std::holds_alternative<Eigen::VectorXcd>(*_t2_amplitudes_abab)) {
    return true;
  }
  if (_t2_amplitudes_aaaa &&
      std::holds_alternative<Eigen::VectorXcd>(*_t2_amplitudes_aaaa)) {
    return true;
  }
  if (_t2_amplitudes_bbbb &&
      std::holds_alternative<Eigen::VectorXcd>(*_t2_amplitudes_bbbb)) {
    return true;
  }
  return false;
}

Configuration CoupledClusterContainer::_apply_excitations(
    const Configuration& ref,
    const std::vector<std::pair<size_t, size_t>>& alpha_excitations,
    const std::vector<std::pair<size_t, size_t>>& beta_excitations) {
  // Convert reference to string, apply excitations, convert back
  std::string config_str = ref.to_string();

  // Apply alpha excitations
  for (const auto& [from_idx, to_idx] : alpha_excitations) {
    if (from_idx >= config_str.size() || to_idx >= config_str.size()) {
      throw std::out_of_range("Excitation index out of range");
    }

    char& from_char = config_str[from_idx];
    char& to_char = config_str[to_idx];

    // Remove alpha from source
    if (from_char == '2') {
      from_char = 'd';  // Doubly -> beta only
    } else if (from_char == 'u') {
      from_char = '0';  // Alpha -> unoccupied
    } else {
      throw std::runtime_error("Invalid alpha excitation: source has no alpha");
    }

    // Add alpha to target
    if (to_char == '0') {
      to_char = 'u';  // Unoccupied -> alpha
    } else if (to_char == 'd') {
      to_char = '2';  // Beta -> doubly
    } else {
      throw std::runtime_error(
          "Invalid alpha excitation: target already has alpha");
    }
  }

  // Apply beta excitations
  for (const auto& [from_idx, to_idx] : beta_excitations) {
    if (from_idx >= config_str.size() || to_idx >= config_str.size()) {
      throw std::out_of_range("Excitation index out of range");
    }

    char& from_char = config_str[from_idx];
    char& to_char = config_str[to_idx];

    // Remove beta from source
    if (from_char == '2') {
      from_char = 'u';  // Doubly -> alpha only
    } else if (from_char == 'd') {
      from_char = '0';  // Beta -> unoccupied
    } else {
      throw std::runtime_error("Invalid beta excitation: source has no beta");
    }

    // Add beta to target
    if (to_char == '0') {
      to_char = 'd';  // Unoccupied -> beta
    } else if (to_char == 'u') {
      to_char = '2';  // Alpha -> doubly
    } else {
      throw std::runtime_error(
          "Invalid beta excitation: target already has beta");
    }
  }

  return Configuration(config_str);
}

void CoupledClusterContainer::_generate_ci_expansion() const {
  if (!has_t1_amplitudes() && !has_t2_amplitudes()) {
    throw std::runtime_error(
        "Cannot generate CI expansion: no amplitudes available");
  }

  // Get reference determinant
  const auto& references = _wavefunction->get_total_determinants();
  if (references.empty()) {
    throw std::runtime_error("No reference determinant available");
  }
  const Configuration& ref = references[0];

  // Get electron counts
  auto [n_alpha, n_beta] = ref.get_n_electrons();
  size_t n_orbitals = _orbitals->get_num_molecular_orbitals();
  size_t n_vir_alpha = n_orbitals - n_alpha;
  size_t n_vir_beta = n_orbitals - n_beta;

  // Determine if we're working with complex amplitudes
  bool use_complex = is_complex();

  // Get amplitude data as Eigen Maps for efficient access
  // T1 alpha: shape (nocc_a, nvir_a), stored row-major as flat vector
  // T1 beta: shape (nocc_b, nvir_b)
  // T2 alpha-beta: shape (nocc_a, nocc_b, nvir_a, nvir_b)
  // T2 alpha-alpha: shape (nocc_a, nocc_a, nvir_a, nvir_a)
  // T2 beta-beta: shape (nocc_b, nocc_b, nvir_b, nvir_b)

  DeterminantVector determinants;
  std::vector<double> coefficients_real;
  std::vector<std::complex<double>> coefficients_complex;

  // Helper lambda for indexing
  auto t1_idx = [](size_t i, size_t a, size_t nvir) { return i * nvir + a; };

  auto t2_idx = [](size_t i, size_t j, size_t a, size_t b, size_t nocc2,
                   size_t nvir1, size_t nvir2) {
    return ((i * nocc2 + j) * nvir1 + a) * nvir2 + b;
  };

  // Helper to get T1 element
  auto get_t1_aa = [&](size_t i, size_t a) -> auto {
    size_t idx = t1_idx(i, a, n_vir_alpha);
    if (use_complex) {
      return std::get<Eigen::VectorXcd>(*_t1_amplitudes_aa)(idx);
    } else {
      return std::complex<double>(
          std::get<Eigen::VectorXd>(*_t1_amplitudes_aa)(idx), 0.0);
    }
  };

  auto get_t1_bb = [&](size_t i, size_t a) -> auto {
    size_t idx = t1_idx(i, a, n_vir_beta);
    if (use_complex) {
      return std::get<Eigen::VectorXcd>(*_t1_amplitudes_bb)(idx);
    } else {
      return std::complex<double>(
          std::get<Eigen::VectorXd>(*_t1_amplitudes_bb)(idx), 0.0);
    }
  };

  auto get_t2_abab = [&](size_t i, size_t j, size_t a, size_t b) -> auto {
    size_t idx = t2_idx(i, j, a, b, n_beta, n_vir_alpha, n_vir_beta);
    if (use_complex) {
      return std::get<Eigen::VectorXcd>(*_t2_amplitudes_abab)(idx);
    } else {
      return std::complex<double>(
          std::get<Eigen::VectorXd>(*_t2_amplitudes_abab)(idx), 0.0);
    }
  };

  auto get_t2_aaaa = [&](size_t i, size_t j, size_t a, size_t b) -> auto {
    size_t idx = t2_idx(i, j, a, b, n_alpha, n_vir_alpha, n_vir_alpha);
    if (use_complex) {
      return std::get<Eigen::VectorXcd>(*_t2_amplitudes_aaaa)(idx);
    } else {
      return std::complex<double>(
          std::get<Eigen::VectorXd>(*_t2_amplitudes_aaaa)(idx), 0.0);
    }
  };

  auto get_t2_bbbb = [&](size_t i, size_t j, size_t a, size_t b) -> auto {
    size_t idx = t2_idx(i, j, a, b, n_beta, n_vir_beta, n_vir_beta);
    if (use_complex) {
      return std::get<Eigen::VectorXcd>(*_t2_amplitudes_bbbb)(idx);
    } else {
      return std::complex<double>(
          std::get<Eigen::VectorXd>(*_t2_amplitudes_bbbb)(idx), 0.0);
    }
  };

  // Helper to add a determinant with coefficient
  auto add_det = [&](const Configuration& det, std::complex<double> coef) {
    determinants.push_back(det);
    if (use_complex) {
      coefficients_complex.push_back(coef);
    } else {
      coefficients_real.push_back(coef.real());
    }
  };

  // ==========================================================================
  // Order 0: Reference determinant (coefficient = 1)
  // ==========================================================================
  add_det(ref, std::complex<double>(1.0, 0.0));

  // ==========================================================================
  // Order 1: Singles (T1)
  // ==========================================================================

  if (has_t1_amplitudes()) {
    // Alpha singles: i -> a
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t a = 0; a < n_vir_alpha; ++a) {
        auto coef = get_t1_aa(i, a);
        Configuration det = _apply_excitations(ref, {{i, n_alpha + a}}, {});
        add_det(det, coef);
      }
    }

    // Beta singles: i -> a
    for (size_t i = 0; i < n_beta; ++i) {
      for (size_t a = 0; a < n_vir_beta; ++a) {
        auto coef = get_t1_bb(i, a);
        Configuration det = _apply_excitations(ref, {}, {{i, n_beta + a}});
        add_det(det, coef);
      }
    }
  }

  // ==========================================================================
  // Order 2: Doubles (T2 + T1²/2)
  // ==========================================================================

  if (has_t2_amplitudes()) {
    // Alpha-alpha doubles: i,j -> a,b
    // c_{ij}^{ab} = t_{ij}^{ab} + t_i^a * t_j^b - t_i^b * t_j^a
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = i + 1; j < n_alpha; ++j) {
        for (size_t a = 0; a < n_vir_alpha; ++a) {
          for (size_t b = a + 1; b < n_vir_alpha; ++b) {
            auto coef = get_t2_aaaa(i, j, a, b) +
                        get_t1_aa(i, a) * get_t1_aa(j, b) -
                        get_t1_aa(i, b) * get_t1_aa(j, a);
            Configuration det = _apply_excitations(
                ref, {{i, n_alpha + a}, {j, n_alpha + b}}, {});
            add_det(det, coef);
          }
        }
      }
    }

    // Beta-beta doubles: i,j -> a,b
    for (size_t i = 0; i < n_beta; ++i) {
      for (size_t j = i + 1; j < n_beta; ++j) {
        for (size_t a = 0; a < n_vir_beta; ++a) {
          for (size_t b = a + 1; b < n_vir_beta; ++b) {
            auto coef = get_t2_bbbb(i, j, a, b) +
                        get_t1_bb(i, a) * get_t1_bb(j, b) -
                        get_t1_bb(i, b) * get_t1_bb(j, a);
            Configuration det =
                _apply_excitations(ref, {}, {{i, n_beta + a}, {j, n_beta + b}});
            add_det(det, coef);
          }
        }
      }
    }

    // Alpha-beta doubles: i_alpha, j_beta -> a_alpha, b_beta
    // No exchange term for different spins!
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = 0; j < n_beta; ++j) {
        for (size_t a = 0; a < n_vir_alpha; ++a) {
          for (size_t b = 0; b < n_vir_beta; ++b) {
            auto coef =
                get_t2_abab(i, j, a, b) + get_t1_aa(i, a) * get_t1_bb(j, b);
            Configuration det =
                _apply_excitations(ref, {{i, n_alpha + a}}, {{j, n_beta + b}});
            add_det(det, coef);
          }
        }
      }
    }
  }

  // ==========================================================================
  // Order 3: Triples (T1·T2 + T1³/6)
  // ==========================================================================

  // 3α + 0β triples
  if (has_t1_amplitudes()) {
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = i + 1; j < n_alpha; ++j) {
        for (size_t k = j + 1; k < n_alpha; ++k) {
          for (size_t a = 0; a < n_vir_alpha; ++a) {
            for (size_t b = 0; b < n_vir_alpha; ++b) {
              if (a == b) continue;
              for (size_t c = 0; c < n_vir_alpha; ++c) {
                if (a == c || b == c) continue;
                auto coef =
                    get_t1_aa(i, a) * get_t1_aa(j, b) * get_t1_aa(k, c) / 6.0;
                if (has_t2_amplitudes()) {
                  coef += get_t1_aa(i, a) * get_t2_aaaa(j, k, b, c);
                  coef += get_t1_aa(j, b) * get_t2_aaaa(i, k, a, c);
                  coef += get_t1_aa(k, c) * get_t2_aaaa(i, j, a, b);
                }
                Configuration det = _apply_excitations(
                    ref, {{i, n_alpha + a}, {j, n_alpha + b}, {k, n_alpha + c}},
                    {});
                add_det(det, coef);
              }
            }
          }
        }
      }
    }
  }

  // 0α + 3β triples
  if (has_t1_amplitudes()) {
    for (size_t i = 0; i < n_beta; ++i) {
      for (size_t j = i + 1; j < n_beta; ++j) {
        for (size_t k = j + 1; k < n_beta; ++k) {
          for (size_t a = 0; a < n_vir_beta; ++a) {
            for (size_t b = 0; b < n_vir_beta; ++b) {
              if (a == b) continue;
              for (size_t c = 0; c < n_vir_beta; ++c) {
                if (a == c || b == c) continue;
                auto coef =
                    get_t1_bb(i, a) * get_t1_bb(j, b) * get_t1_bb(k, c) / 6.0;
                if (has_t2_amplitudes()) {
                  coef += get_t1_bb(i, a) * get_t2_bbbb(j, k, b, c);
                  coef += get_t1_bb(j, b) * get_t2_bbbb(i, k, a, c);
                  coef += get_t1_bb(k, c) * get_t2_bbbb(i, j, a, b);
                }
                Configuration det = _apply_excitations(
                    ref, {},
                    {{i, n_beta + a}, {j, n_beta + b}, {k, n_beta + c}});
                add_det(det, coef);
              }
            }
          }
        }
      }
    }
  }

  // 2α + 1β triples
  if (has_t1_amplitudes()) {
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = i + 1; j < n_alpha; ++j) {
        for (size_t k = 0; k < n_beta; ++k) {
          for (size_t a = 0; a < n_vir_alpha; ++a) {
            for (size_t b = 0; b < n_vir_alpha; ++b) {
              if (a == b) continue;
              for (size_t c = 0; c < n_vir_beta; ++c) {
                auto coef =
                    get_t1_aa(i, a) * get_t1_aa(j, b) * get_t1_bb(k, c) / 2.0;
                if (has_t2_amplitudes()) {
                  coef += get_t1_aa(i, a) * get_t2_abab(j, k, b, c);
                  coef += get_t1_aa(j, b) * get_t2_abab(i, k, a, c);
                  coef += get_t1_bb(k, c) * get_t2_aaaa(i, j, a, b);
                }
                Configuration det = _apply_excitations(
                    ref, {{i, n_alpha + a}, {j, n_alpha + b}},
                    {{k, n_beta + c}});
                add_det(det, coef);
              }
            }
          }
        }
      }
    }
  }

  // 1α + 2β triples
  if (has_t1_amplitudes()) {
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = 0; j < n_beta; ++j) {
        for (size_t k = j + 1; k < n_beta; ++k) {
          for (size_t a = 0; a < n_vir_alpha; ++a) {
            for (size_t b = 0; b < n_vir_beta; ++b) {
              for (size_t c = 0; c < n_vir_beta; ++c) {
                if (b == c) continue;
                auto coef =
                    get_t1_aa(i, a) * get_t1_bb(j, b) * get_t1_bb(k, c) / 2.0;
                if (has_t2_amplitudes()) {
                  coef += get_t1_bb(j, b) * get_t2_abab(i, k, a, c);
                  coef += get_t1_bb(k, c) * get_t2_abab(i, j, a, b);
                  coef += get_t1_aa(i, a) * get_t2_bbbb(j, k, b, c);
                }
                Configuration det =
                    _apply_excitations(ref, {{i, n_alpha + a}},
                                       {{j, n_beta + b}, {k, n_beta + c}});
                add_det(det, coef);
              }
            }
          }
        }
      }
    }
  }

  // ==========================================================================
  // Order 4: Quadruples (T2²/2 + T1²·T2/2 + T1⁴/24)
  // ==========================================================================

  // 4α + 0β quadruples
  {
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = i + 1; j < n_alpha; ++j) {
        for (size_t k = j + 1; k < n_alpha; ++k) {
          for (size_t l = k + 1; l < n_alpha; ++l) {
            for (size_t a = 0; a < n_vir_alpha; ++a) {
              for (size_t b = 0; b < n_vir_alpha; ++b) {
                if (a == b) continue;
                for (size_t c = 0; c < n_vir_alpha; ++c) {
                  if (a == c || b == c) continue;
                  for (size_t d = 0; d < n_vir_alpha; ++d) {
                    if (a == d || b == d || c == d) continue;
                    std::complex<double> coef = 0.0;
                    if (has_t1_amplitudes()) {
                      coef += get_t1_aa(i, a) * get_t1_aa(j, b) *
                              get_t1_aa(k, c) * get_t1_aa(l, d) / 24.0;
                    }
                    if (has_t1_amplitudes() && has_t2_amplitudes()) {
                      coef += get_t1_aa(i, a) * get_t1_aa(j, b) *
                              get_t2_aaaa(k, l, c, d) / 2.0;
                      coef += get_t1_aa(i, a) * get_t1_aa(k, c) *
                              get_t2_aaaa(j, l, b, d) / 2.0;
                      coef += get_t1_aa(i, a) * get_t1_aa(l, d) *
                              get_t2_aaaa(j, k, b, c) / 2.0;
                      coef += get_t1_aa(j, b) * get_t1_aa(k, c) *
                              get_t2_aaaa(i, l, a, d) / 2.0;
                      coef += get_t1_aa(j, b) * get_t1_aa(l, d) *
                              get_t2_aaaa(i, k, a, c) / 2.0;
                      coef += get_t1_aa(k, c) * get_t1_aa(l, d) *
                              get_t2_aaaa(i, j, a, b) / 2.0;
                    }
                    if (has_t2_amplitudes()) {
                      coef += get_t2_aaaa(i, j, a, b) *
                              get_t2_aaaa(k, l, c, d) / 2.0;
                      coef += get_t2_aaaa(i, k, a, c) *
                              get_t2_aaaa(j, l, b, d) / 2.0;
                      coef += get_t2_aaaa(i, l, a, d) *
                              get_t2_aaaa(j, k, b, c) / 2.0;
                    }
                    Configuration det = _apply_excitations(ref,
                                                           {{i, n_alpha + a},
                                                            {j, n_alpha + b},
                                                            {k, n_alpha + c},
                                                            {l, n_alpha + d}},
                                                           {});
                    add_det(det, coef);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // 0α + 4β quadruples
  {
    for (size_t i = 0; i < n_beta; ++i) {
      for (size_t j = i + 1; j < n_beta; ++j) {
        for (size_t k = j + 1; k < n_beta; ++k) {
          for (size_t l = k + 1; l < n_beta; ++l) {
            for (size_t a = 0; a < n_vir_beta; ++a) {
              for (size_t b = 0; b < n_vir_beta; ++b) {
                if (a == b) continue;
                for (size_t c = 0; c < n_vir_beta; ++c) {
                  if (a == c || b == c) continue;
                  for (size_t d = 0; d < n_vir_beta; ++d) {
                    if (a == d || b == d || c == d) continue;
                    std::complex<double> coef = 0.0;
                    if (has_t1_amplitudes()) {
                      coef += get_t1_bb(i, a) * get_t1_bb(j, b) *
                              get_t1_bb(k, c) * get_t1_bb(l, d) / 24.0;
                    }
                    if (has_t1_amplitudes() && has_t2_amplitudes()) {
                      coef += get_t1_bb(i, a) * get_t1_bb(j, b) *
                              get_t2_bbbb(k, l, c, d) / 2.0;
                      coef += get_t1_bb(i, a) * get_t1_bb(k, c) *
                              get_t2_bbbb(j, l, b, d) / 2.0;
                      coef += get_t1_bb(i, a) * get_t1_bb(l, d) *
                              get_t2_bbbb(j, k, b, c) / 2.0;
                      coef += get_t1_bb(j, b) * get_t1_bb(k, c) *
                              get_t2_bbbb(i, l, a, d) / 2.0;
                      coef += get_t1_bb(j, b) * get_t1_bb(l, d) *
                              get_t2_bbbb(i, k, a, c) / 2.0;
                      coef += get_t1_bb(k, c) * get_t1_bb(l, d) *
                              get_t2_bbbb(i, j, a, b) / 2.0;
                    }
                    if (has_t2_amplitudes()) {
                      coef += get_t2_bbbb(i, j, a, b) *
                              get_t2_bbbb(k, l, c, d) / 2.0;
                      coef += get_t2_bbbb(i, k, a, c) *
                              get_t2_bbbb(j, l, b, d) / 2.0;
                      coef += get_t2_bbbb(i, l, a, d) *
                              get_t2_bbbb(j, k, b, c) / 2.0;
                    }
                    Configuration det = _apply_excitations(ref, {},
                                                           {{i, n_beta + a},
                                                            {j, n_beta + b},
                                                            {k, n_beta + c},
                                                            {l, n_beta + d}});
                    add_det(det, coef);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // 3α + 1β quadruples
  {
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = i + 1; j < n_alpha; ++j) {
        for (size_t k = j + 1; k < n_alpha; ++k) {
          for (size_t l = 0; l < n_beta; ++l) {
            for (size_t a = 0; a < n_vir_alpha; ++a) {
              for (size_t b = 0; b < n_vir_alpha; ++b) {
                if (a == b) continue;
                for (size_t c = 0; c < n_vir_alpha; ++c) {
                  if (a == c || b == c) continue;
                  for (size_t d = 0; d < n_vir_beta; ++d) {
                    std::complex<double> coef = 0.0;
                    if (has_t1_amplitudes()) {
                      coef += get_t1_aa(i, a) * get_t1_aa(j, b) *
                              get_t1_aa(k, c) * get_t1_bb(l, d) / 6.0;
                    }
                    if (has_t1_amplitudes() && has_t2_amplitudes()) {
                      coef += get_t1_aa(i, a) * get_t1_aa(j, b) *
                              get_t2_abab(k, l, c, d) / 2.0;
                      coef += get_t1_aa(i, a) * get_t1_aa(k, c) *
                              get_t2_abab(j, l, b, d) / 2.0;
                      coef += get_t1_aa(j, b) * get_t1_aa(k, c) *
                              get_t2_abab(i, l, a, d) / 2.0;
                      coef += get_t1_aa(i, a) * get_t1_bb(l, d) *
                              get_t2_aaaa(j, k, b, c);
                      coef += get_t1_aa(j, b) * get_t1_bb(l, d) *
                              get_t2_aaaa(i, k, a, c);
                      coef += get_t1_aa(k, c) * get_t1_bb(l, d) *
                              get_t2_aaaa(i, j, a, b);
                    }
                    if (has_t2_amplitudes()) {
                      coef += get_t2_aaaa(i, j, a, b) * get_t2_abab(k, l, c, d);
                      coef += get_t2_aaaa(i, k, a, c) * get_t2_abab(j, l, b, d);
                      coef += get_t2_aaaa(j, k, b, c) * get_t2_abab(i, l, a, d);
                    }
                    Configuration det = _apply_excitations(
                        ref,
                        {{i, n_alpha + a}, {j, n_alpha + b}, {k, n_alpha + c}},
                        {{l, n_beta + d}});
                    add_det(det, coef);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // 1α + 3β quadruples
  {
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = 0; j < n_beta; ++j) {
        for (size_t k = j + 1; k < n_beta; ++k) {
          for (size_t l = k + 1; l < n_beta; ++l) {
            for (size_t a = 0; a < n_vir_alpha; ++a) {
              for (size_t b = 0; b < n_vir_beta; ++b) {
                for (size_t c = 0; c < n_vir_beta; ++c) {
                  if (b == c) continue;
                  for (size_t d = 0; d < n_vir_beta; ++d) {
                    if (b == d || c == d) continue;
                    std::complex<double> coef = 0.0;
                    if (has_t1_amplitudes()) {
                      coef += get_t1_aa(i, a) * get_t1_bb(j, b) *
                              get_t1_bb(k, c) * get_t1_bb(l, d) / 6.0;
                    }
                    if (has_t1_amplitudes() && has_t2_amplitudes()) {
                      coef += get_t1_bb(j, b) * get_t1_bb(k, c) *
                              get_t2_abab(i, l, a, d) / 2.0;
                      coef += get_t1_bb(j, b) * get_t1_bb(l, d) *
                              get_t2_abab(i, k, a, c) / 2.0;
                      coef += get_t1_bb(k, c) * get_t1_bb(l, d) *
                              get_t2_abab(i, j, a, b) / 2.0;
                      coef += get_t1_aa(i, a) * get_t1_bb(j, b) *
                              get_t2_bbbb(k, l, c, d);
                      coef += get_t1_aa(i, a) * get_t1_bb(k, c) *
                              get_t2_bbbb(j, l, b, d);
                      coef += get_t1_aa(i, a) * get_t1_bb(l, d) *
                              get_t2_bbbb(j, k, b, c);
                    }
                    if (has_t2_amplitudes()) {
                      coef += get_t2_bbbb(j, k, b, c) * get_t2_abab(i, l, a, d);
                      coef += get_t2_bbbb(j, l, b, d) * get_t2_abab(i, k, a, c);
                      coef += get_t2_bbbb(k, l, c, d) * get_t2_abab(i, j, a, b);
                    }
                    Configuration det = _apply_excitations(
                        ref, {{i, n_alpha + a}},
                        {{j, n_beta + b}, {k, n_beta + c}, {l, n_beta + d}});
                    add_det(det, coef);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // 2α + 2β quadruples
  {
    for (size_t i = 0; i < n_alpha; ++i) {
      for (size_t j = i + 1; j < n_alpha; ++j) {
        for (size_t k = 0; k < n_beta; ++k) {
          for (size_t l = k + 1; l < n_beta; ++l) {
            for (size_t a = 0; a < n_vir_alpha; ++a) {
              for (size_t b = 0; b < n_vir_alpha; ++b) {
                if (a == b) continue;
                for (size_t c = 0; c < n_vir_beta; ++c) {
                  for (size_t d = 0; d < n_vir_beta; ++d) {
                    if (c == d) continue;
                    std::complex<double> coef = 0.0;
                    if (has_t1_amplitudes()) {
                      coef += get_t1_aa(i, a) * get_t1_aa(j, b) *
                              get_t1_bb(k, c) * get_t1_bb(l, d) / 4.0;
                    }
                    if (has_t1_amplitudes() && has_t2_amplitudes()) {
                      coef += get_t1_aa(i, a) * get_t1_aa(j, b) *
                              get_t2_bbbb(k, l, c, d) / 2.0;
                      coef += get_t1_bb(k, c) * get_t1_bb(l, d) *
                              get_t2_aaaa(i, j, a, b) / 2.0;
                      coef += get_t1_aa(i, a) * get_t1_bb(k, c) *
                              get_t2_abab(j, l, b, d) / 2.0;
                      coef += get_t1_aa(i, a) * get_t1_bb(l, d) *
                              get_t2_abab(j, k, b, c) / 2.0;
                      coef += get_t1_aa(j, b) * get_t1_bb(k, c) *
                              get_t2_abab(i, l, a, d) / 2.0;
                      coef += get_t1_aa(j, b) * get_t1_bb(l, d) *
                              get_t2_abab(i, k, a, c) / 2.0;
                    }
                    if (has_t2_amplitudes()) {
                      coef += get_t2_aaaa(i, j, a, b) * get_t2_bbbb(k, l, c, d);
                      coef += get_t2_abab(i, k, a, c) *
                              get_t2_abab(j, l, b, d) / 2.0;
                      coef += get_t2_abab(i, l, a, d) *
                              get_t2_abab(j, k, b, c) / 2.0;
                    }
                    Configuration det = _apply_excitations(
                        ref, {{i, n_alpha + a}, {j, n_alpha + b}},
                        {{k, n_beta + c}, {l, n_beta + d}});
                    add_det(det, coef);
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  // Consolidate duplicate determinants
  if (use_complex) {
    // Convert to VectorVariant for consolidate_determinants
    Eigen::VectorXcd coef_vec(coefficients_complex.size());
    for (size_t i = 0; i < coefficients_complex.size(); ++i) {
      coef_vec(i) = coefficients_complex[i];
    }
    VectorVariant coef_variant(std::move(coef_vec));
    detail::consolidate_determinants(determinants, coef_variant);

    // Normalize
    auto& final_coefs = std::get<Eigen::VectorXcd>(coef_variant);
    double norm_sq = 0.0;
    for (Eigen::Index i = 0; i < final_coefs.size(); ++i) {
      norm_sq += std::norm(final_coefs[i]);
    }
    double norm = std::sqrt(norm_sq);
    // Division by 0 should not be possible here since ref coef = 1.0
    for (Eigen::Index i = 0; i < final_coefs.size(); ++i) {
      final_coefs[i] /= norm;
    }

    // Store in cache
    _coefficients_cache =
        std::make_unique<VectorVariant>(std::move(coef_variant));
  } else {
    // Convert to VectorVariant for consolidate_determinants
    Eigen::VectorXd coef_vec(coefficients_real.size());
    for (size_t i = 0; i < coefficients_real.size(); ++i) {
      coef_vec(i) = coefficients_real[i];
    }
    VectorVariant coef_variant(std::move(coef_vec));
    detail::consolidate_determinants(determinants, coef_variant);

    // Normalize
    auto& final_coefs = std::get<Eigen::VectorXd>(coef_variant);
    double norm_sq = 0.0;
    for (Eigen::Index i = 0; i < final_coefs.size(); ++i) {
      norm_sq += final_coefs[i] * final_coefs[i];
    }
    double norm = std::sqrt(norm_sq);
    // Division by 0 should not be possible here since ref coef = 1.0
    for (Eigen::Index i = 0; i < final_coefs.size(); ++i) {
      final_coefs[i] /= norm;
    }

    // Store in cache
    _coefficients_cache =
        std::make_unique<VectorVariant>(std::move(coef_variant));
  }

  _determinant_vector_cache =
      std::make_unique<DeterminantVector>(std::move(determinants));
}

// =============================================================================
// Lazy RDM computation from CI expansion
// =============================================================================

namespace {
// Helper: Dispatch RDM computation based on number of orbitals
// Uses MACIS bitset sizes: 64, 128, 256
template <size_t N>
void compute_rdms_impl(const std::vector<Configuration>& determinants,
                       const double* coeffs, size_t norb,
                       std::vector<double>& one_rdm_aa,
                       std::vector<double>& one_rdm_bb,
                       std::vector<double>& two_rdm_aaaa,
                       std::vector<double>& two_rdm_bbbb,
                       std::vector<double>& two_rdm_aabb) {
  using wfn_t = macis::wfn_t<N>;
  using wfn_traits = macis::wavefunction_traits<wfn_t>;
  using spin_det_t = macis::spin_wfn_t<wfn_t>;

  const size_t ndets = determinants.size();

  // Convert QDK Configurations to MACIS wfn_t format
  std::vector<wfn_t> macis_dets;
  macis_dets.reserve(ndets);
  for (const auto& config : determinants) {
    auto bitset = config.to_bitset<N>();
    macis_dets.push_back(wfn_t(bitset));
  }

  // Create spans for RDM storage
  macis::matrix_span<double> ordm_aa(one_rdm_aa.data(), norb, norb);
  macis::matrix_span<double> ordm_bb(one_rdm_bb.data(), norb, norb);
  macis::rank4_span<double> trdm_aaaa(two_rdm_aaaa.data(), norb, norb, norb,
                                      norb);
  macis::rank4_span<double> trdm_bbbb(two_rdm_bbbb.data(), norb, norb, norb,
                                      norb);
  macis::rank4_span<double> trdm_aabb(two_rdm_aabb.data(), norb, norb, norb,
                                      norb);

  std::vector<uint32_t> bra_occ_alpha, bra_occ_beta;

  // Double loop over determinants to compute RDM contributions
  for (size_t i = 0; i < ndets; ++i) {
    const auto& bra = macis_dets[i];
    if (wfn_traits::count(bra)) {
      spin_det_t bra_alpha = wfn_traits::alpha_string(bra);
      spin_det_t bra_beta = wfn_traits::beta_string(bra);

      macis::bits_to_indices(bra_alpha, bra_occ_alpha);
      macis::bits_to_indices(bra_beta, bra_occ_beta);

      for (size_t j = 0; j < ndets; ++j) {
        const auto& ket = macis_dets[j];
        if (wfn_traits::count(ket)) {
          spin_det_t ket_alpha = wfn_traits::alpha_string(ket);
          spin_det_t ket_beta = wfn_traits::beta_string(ket);

          wfn_t ex_total = bra ^ ket;
          if (wfn_traits::count(ex_total) <= 4) {
            spin_det_t ex_alpha = wfn_traits::alpha_string(ex_total);
            spin_det_t ex_beta = wfn_traits::beta_string(ex_total);

            const double val = coeffs[i] * coeffs[j];

            if (std::abs(val) > 1e-16) {
              macis::rdm_contributions_spin_dep<false>(
                  bra_alpha, ket_alpha, ex_alpha, bra_beta, ket_beta, ex_beta,
                  bra_occ_alpha, bra_occ_beta, val, ordm_aa, ordm_bb, trdm_aaaa,
                  trdm_bbbb, trdm_aabb);
            }
          }
        }
      }
    }
  }
}
}  // namespace

bool CoupledClusterContainer::has_one_rdm_spin_dependent() const {
  // RDMs are not stored in CC container - requires adjoint wavefunction
  return false;
}

bool CoupledClusterContainer::has_one_rdm_spin_traced() const {
  // RDMs are not stored in CC container - requires adjoint wavefunction
  return false;
}

bool CoupledClusterContainer::has_two_rdm_spin_dependent() const {
  // RDMs are not stored in CC container - requires adjoint wavefunction
  return false;
}

bool CoupledClusterContainer::has_two_rdm_spin_traced() const {
  // RDMs are not stored in CC container - requires adjoint wavefunction
  return false;
}

std::tuple<const CoupledClusterContainer::MatrixVariant&,
           const CoupledClusterContainer::MatrixVariant&>
CoupledClusterContainer::get_active_one_rdm_spin_dependent() const {
  // Cannot compute RDMs from ket amplitudes alone
  throw std::runtime_error(
      "Coupled cluster RDM computation requires the adjoint (bra) "
      "wavefunction. Use a coupled cluster method that computes lambda "
      "amplitudes.");
}

std::tuple<const CoupledClusterContainer::VectorVariant&,
           const CoupledClusterContainer::VectorVariant&,
           const CoupledClusterContainer::VectorVariant&>
CoupledClusterContainer::get_active_two_rdm_spin_dependent() const {
  // Cannot compute RDMs from ket amplitudes alone
  throw std::runtime_error(
      "Coupled cluster RDM computation requires the adjoint (bra) "
      "wavefunction. Use a coupled cluster method that computes lambda "
      "amplitudes.");
}

const CoupledClusterContainer::MatrixVariant&
CoupledClusterContainer::get_active_one_rdm_spin_traced() const {
  // Cannot compute RDMs from ket amplitudes alone
  throw std::runtime_error(
      "Coupled cluster RDM computation requires the adjoint (bra) "
      "wavefunction. Use a coupled cluster method that computes lambda "
      "amplitudes.");
}

const CoupledClusterContainer::VectorVariant&
CoupledClusterContainer::get_active_two_rdm_spin_traced() const {
  // Cannot compute RDMs from ket amplitudes alone
  throw std::runtime_error(
      "Coupled cluster RDM computation requires the adjoint (bra) "
      "wavefunction. Use a coupled cluster method that computes lambda "
      "amplitudes.");
}

}  // namespace qdk::chemistry::data

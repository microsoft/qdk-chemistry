/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <optional>
#include <qdk/chemistry/data/wavefunction_containers/cc.hpp>
#include <stdexcept>
#include <variant>

#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

CoupledClusterContainer::CoupledClusterContainer(
    std::shared_ptr<Orbitals> orbitals, const DeterminantVector& references,
    const std::optional<VectorVariant>& t1_amplitudes,
    const std::optional<VectorVariant>& t2_amplitudes)
    : CoupledClusterContainer(
          orbitals, references, t1_amplitudes, std::nullopt, t2_amplitudes,
          std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
          std::nullopt, std::nullopt, std::nullopt, std::nullopt) {}

CoupledClusterContainer::CoupledClusterContainer(
    std::shared_ptr<Orbitals> orbitals, const DeterminantVector& references,
    const std::optional<VectorVariant>& t1_amplitudes,
    const std::optional<VectorVariant>& t2_amplitudes,
    const std::optional<MatrixVariant>& one_rdm_spin_traced,
    const std::optional<VectorVariant>& two_rdm_spin_traced)
    : CoupledClusterContainer(orbitals, references, t1_amplitudes, std::nullopt,
                              t2_amplitudes, std::nullopt, std::nullopt,
                              one_rdm_spin_traced, std::nullopt, std::nullopt,
                              two_rdm_spin_traced, std::nullopt, std::nullopt,
                              std::nullopt) {}

CoupledClusterContainer::CoupledClusterContainer(
    std::shared_ptr<Orbitals> orbitals, const DeterminantVector& references,
    const std::optional<VectorVariant>& t1_amplitudes,
    const std::optional<VectorVariant>& t2_amplitudes,
    const std::optional<MatrixVariant>& one_rdm_spin_traced,
    const std::optional<MatrixVariant>& one_rdm_aa,
    const std::optional<MatrixVariant>& one_rdm_bb,
    const std::optional<VectorVariant>& two_rdm_spin_traced,
    const std::optional<VectorVariant>& two_rdm_aabb,
    const std::optional<VectorVariant>& two_rdm_aaaa,
    const std::optional<VectorVariant>& two_rdm_bbbb)
    : CoupledClusterContainer(orbitals, references, t1_amplitudes, std::nullopt,
                              t2_amplitudes, std::nullopt, std::nullopt,
                              one_rdm_spin_traced, one_rdm_aa, one_rdm_bb,
                              two_rdm_spin_traced, two_rdm_aabb, two_rdm_aaaa,
                              two_rdm_bbbb) {}

CoupledClusterContainer::CoupledClusterContainer(
    std::shared_ptr<Orbitals> orbitals, const DeterminantVector& references,
    const std::optional<VectorVariant>& t1_amplitudes_aa,
    const std::optional<VectorVariant>& t1_amplitudes_bb,
    const std::optional<VectorVariant>& t2_amplitudes_abab,
    const std::optional<VectorVariant>& t2_amplitudes_aaaa,
    const std::optional<VectorVariant>& t2_amplitudes_bbbb)
    : CoupledClusterContainer(
          orbitals, references, t1_amplitudes_aa, t1_amplitudes_bb,
          t2_amplitudes_abab, t2_amplitudes_aaaa, t2_amplitudes_bbbb,
          std::nullopt, std::nullopt, std::nullopt, std::nullopt, std::nullopt,
          std::nullopt, std::nullopt) {}

CoupledClusterContainer::CoupledClusterContainer(
    std::shared_ptr<Orbitals> orbitals, const DeterminantVector& references,
    const std::optional<VectorVariant>& t1_amplitudes_aa,
    const std::optional<VectorVariant>& t1_amplitudes_bb,
    const std::optional<VectorVariant>& t2_amplitudes_abab,
    const std::optional<VectorVariant>& t2_amplitudes_aaaa,
    const std::optional<VectorVariant>& t2_amplitudes_bbbb,
    const std::optional<MatrixVariant>& one_rdm_spin_traced,
    const std::optional<VectorVariant>& two_rdm_spin_traced)
    : CoupledClusterContainer(
          orbitals, references, t1_amplitudes_aa, t1_amplitudes_bb,
          t2_amplitudes_abab, t2_amplitudes_aaaa, t2_amplitudes_bbbb,
          one_rdm_spin_traced, std::nullopt, std::nullopt, two_rdm_spin_traced,
          std::nullopt, std::nullopt, std::nullopt) {}

CoupledClusterContainer::CoupledClusterContainer(
    std::shared_ptr<Orbitals> orbitals, const DeterminantVector& references,
    const std::optional<VectorVariant>& t1_amplitudes_aa,
    const std::optional<VectorVariant>& t1_amplitudes_bb,
    const std::optional<VectorVariant>& t2_amplitudes_abab,
    const std::optional<VectorVariant>& t2_amplitudes_aaaa,
    const std::optional<VectorVariant>& t2_amplitudes_bbbb,
    const std::optional<MatrixVariant>& one_rdm_spin_traced,
    const std::optional<MatrixVariant>& one_rdm_aa,
    const std::optional<MatrixVariant>& one_rdm_bb,
    const std::optional<VectorVariant>& two_rdm_spin_traced,
    const std::optional<VectorVariant>& two_rdm_aabb,
    const std::optional<VectorVariant>& two_rdm_aaaa,
    const std::optional<VectorVariant>& two_rdm_bbbb)
    : WavefunctionContainer(
          WavefunctionType::NotSelfDual),  // Always force NotSelfDual for CC
      _references(references, orbitals),
      _orbitals(orbitals) {
  if (!orbitals) {
    throw std::invalid_argument("Orbitals cannot be null");
  }

  if (references.empty()) {
    throw std::invalid_argument(
        "Must provide at least one reference determinant");
  }

  // Validate amplitude inputs
  if (!t1_amplitudes_aa && t1_amplitudes_bb) {
    throw std::invalid_argument(
        "Cannot provide unrestricted beta T1 amplitudes without alpha T1 "
        "amplitudes");
  }
  if (!t2_amplitudes_abab && (t2_amplitudes_aaaa || t2_amplitudes_bbbb)) {
    throw std::invalid_argument(
        "Cannot provide unrestricted T2 alpha-alpha or beta-beta amplitudes "
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

  if (one_rdm_spin_traced) {
    _one_rdm_spin_traced =
        std::make_shared<MatrixVariant>(*one_rdm_spin_traced);
  }
  if (one_rdm_aa) {
    _one_rdm_spin_dependent_aa = std::make_shared<MatrixVariant>(*one_rdm_aa);
  }
  if (one_rdm_bb) {
    _one_rdm_spin_dependent_bb = std::make_shared<MatrixVariant>(*one_rdm_bb);
  }
  if (two_rdm_spin_traced) {
    _two_rdm_spin_traced =
        std::make_shared<VectorVariant>(*two_rdm_spin_traced);
  }
  if (two_rdm_aabb) {
    _two_rdm_spin_dependent_aabb =
        std::make_shared<VectorVariant>(*two_rdm_aabb);
  }
  if (two_rdm_aaaa) {
    _two_rdm_spin_dependent_aaaa =
        std::make_shared<VectorVariant>(*two_rdm_aaaa);
  }
  if (two_rdm_bbbb) {
    _two_rdm_spin_dependent_bbbb =
        std::make_shared<VectorVariant>(*two_rdm_bbbb);
  }
}

std::unique_ptr<WavefunctionContainer> CoupledClusterContainer::clone() const {
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

  std::optional<MatrixVariant> one_rdm_spin_traced =
      _one_rdm_spin_traced ? std::optional<MatrixVariant>(*_one_rdm_spin_traced)
                           : std::nullopt;
  std::optional<VectorVariant> two_rdm_spin_traced =
      _two_rdm_spin_traced ? std::optional<VectorVariant>(*_two_rdm_spin_traced)
                           : std::nullopt;
  std::optional<MatrixVariant> one_rdm_aa =
      _one_rdm_spin_dependent_aa
          ? std::optional<MatrixVariant>(*_one_rdm_spin_dependent_aa)
          : std::nullopt;
  std::optional<MatrixVariant> one_rdm_bb =
      _one_rdm_spin_dependent_bb
          ? std::optional<MatrixVariant>(*_one_rdm_spin_dependent_bb)
          : std::nullopt;
  std::optional<VectorVariant> two_rdm_aabb =
      _two_rdm_spin_dependent_aabb
          ? std::optional<VectorVariant>(*_two_rdm_spin_dependent_aabb)
          : std::nullopt;
  std::optional<VectorVariant> two_rdm_aaaa =
      _two_rdm_spin_dependent_aaaa
          ? std::optional<VectorVariant>(*_two_rdm_spin_dependent_aaaa)
          : std::nullopt;
  std::optional<VectorVariant> two_rdm_bbbb =
      _two_rdm_spin_dependent_bbbb
          ? std::optional<VectorVariant>(*_two_rdm_spin_dependent_bbbb)
          : std::nullopt;

  return std::make_unique<CoupledClusterContainer>(
      _orbitals, _references.get_configurations(), t1_aa, t1_bb, t2_abab,
      t2_aaaa, t2_bbbb, one_rdm_spin_traced, one_rdm_aa, one_rdm_bb,
      two_rdm_spin_traced, two_rdm_aabb, two_rdm_aaaa, two_rdm_bbbb);
}

std::shared_ptr<Orbitals> CoupledClusterContainer::get_orbitals() const {
  return _orbitals;
}

const CoupledClusterContainer::VectorVariant&
CoupledClusterContainer::get_coefficients() const {
  throw std::runtime_error(
      "get_coefficients() is not implemented for coupled cluster "
      "wavefunctions. ");
}

CoupledClusterContainer::ScalarVariant CoupledClusterContainer::get_coefficient(
    const Configuration& det) const {
  throw std::runtime_error(
      "get_coefficient() is not implemented for coupled cluster "
      "wavefunctions. ");
}

const CoupledClusterContainer::DeterminantVector&
CoupledClusterContainer::get_references() const {
  return _references.get_configurations();
}

const CoupledClusterContainer::DeterminantVector&
CoupledClusterContainer::get_active_determinants() const {
  throw std::runtime_error(
      "get_active_determinants() is not implemented for coupled cluster "
      "wavefunctions. ");
}

std::pair<const CoupledClusterContainer::VectorVariant&,
          const CoupledClusterContainer::VectorVariant&>
CoupledClusterContainer::get_t1_amplitudes() const {
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
  if (!has_t2_amplitudes()) {
    throw std::runtime_error("T2 amplitudes not available");
  }
  return std::make_tuple(std::cref(*_t2_amplitudes_abab),
                         std::cref(*_t2_amplitudes_aaaa),
                         std::cref(*_t2_amplitudes_bbbb));
}

bool CoupledClusterContainer::has_t1_amplitudes() const {
  return _t1_amplitudes_aa != nullptr;
}

bool CoupledClusterContainer::has_t2_amplitudes() const {
  return _t2_amplitudes_abab != nullptr;
}

size_t CoupledClusterContainer::size() const {
  throw std::runtime_error(
      "size() is not meaningful for coupled cluster wavefunctions. ");
}

CoupledClusterContainer::ScalarVariant CoupledClusterContainer::overlap(
    const WavefunctionContainer& other) const {
  throw std::runtime_error(
      "overlap() is not implemented for coupled cluster wavefunctions. ");
}

double CoupledClusterContainer::norm() const {
  throw std::runtime_error(
      "norm() is not implemented for coupled cluster wavefunctions. ");
}

bool CoupledClusterContainer::contains_determinant(
    const Configuration& det) const {
  return contains_reference(det);
}

bool CoupledClusterContainer::contains_reference(
    const Configuration& det) const {
  if (std::find(_references.get_configurations().begin(),
                _references.get_configurations().end(),
                det) != _references.get_configurations().end()) {
    return true;
  }
  return false;
}

void CoupledClusterContainer::clear_caches() const {
  // Clear the cached determinant vector
  _determinant_vector_cache.reset();

  // Clear all cached RDMs using base class helper
  _clear_rdms();
}

nlohmann::json CoupledClusterContainer::to_json() const {
  nlohmann::json j;

  j["version"] = SERIALIZATION_VERSION;
  j["container_type"] = get_container_type();
  // CC wavefunctions are always NotSelfDual
  j["wavefunction_type"] = "not_self_dual";

  // Serialize orbitals
  if (_orbitals) {
    j["orbitals"] = _orbitals->to_json();
  }

  // Serialize references
  j["references"] = nlohmann::json::array();
  for (const auto& ref : _references) {
    j["references"].push_back(ref.to_json());
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
  try {
    if (!j.contains("version")) {
      throw std::runtime_error("JSON does not contain version information");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    // CC wavefunctions are always NotSelfDual - throw if SelfDual is specified
    if (j.contains("wavefunction_type")) {
      std::string type_str = j["wavefunction_type"];
      if (type_str == "self_dual") {
        throw std::invalid_argument(
            "Invalid JSON data: Found 'self_dual' wavefunction_type, but CC "
            "containers must be "
            "'not_self_dual'.");
      }
    }

    auto orbitals = Orbitals::from_json(j["orbitals"]);
    DeterminantVector references;
    for (const auto& ref_json : j["references"]) {
      references.push_back(Configuration::from_json(ref_json));
    }

    bool is_complex = j.value("is_complex", false);
    return std::make_unique<CoupledClusterContainer>(
        orbitals, references,
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

    // CC wavefunctions are always NotSelfDual
    std::string wf_type_str = "not_self_dual";
    H5::Attribute wf_type_attr = group.createAttribute(
        "wavefunction_type", string_type, H5::DataSpace(H5S_SCALAR));
    wf_type_attr.write(string_type, wf_type_str);

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

    // Store configuration set (delegates to ConfigurationSet serialization)
    H5::Group reference_configs_group =
        group.createGroup("reference_configurations");
    _references.to_hdf5(reference_configs_group);

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::unique_ptr<CoupledClusterContainer> CoupledClusterContainer::from_hdf5(
    H5::Group& group) {
  try {
    // version
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version_str;
    version_attr.read(string_type, version_str);
    validate_serialization_version(SERIALIZATION_VERSION, version_str);

    // CC wavefunctions are always NotSelfDual - throw if SelfDual is specified
    if (group.attrExists("wavefunction_type")) {
      H5::Attribute wf_type_attr = group.openAttribute("wavefunction_type");
      std::string wf_type_str;
      wf_type_attr.read(string_type, wf_type_str);
      if (wf_type_str == "self_dual") {
        throw std::invalid_argument(
            "Invalid HDF5 data: Found 'self_dual' wavefunction_type, but CC "
            "containers must be "
            "'not_self_dual'.");
      }
    }

    // complex flag
    bool is_complex = false;
    if (group.attrExists("is_complex")) {
      H5::Attribute is_complex_attr = group.openAttribute("is_complex");
      hbool_t is_complex_flag;
      is_complex_attr.read(H5::PredType::NATIVE_HBOOL, &is_complex_flag);
      is_complex = (is_complex_flag != 0);
    }

    // Load configuration set (delegates to ConfigurationSet deserialization)
    if (!group.nameExists("reference_configurations")) {
      throw std::runtime_error(
          "HDF5 group missing required 'reference_configurations' subgroup");
    }
    H5::Group reference_configs_group =
        group.openGroup("reference_configurations");
    auto reference_configs =
        ConfigurationSet::from_hdf5(reference_configs_group);
    const auto& determinants = reference_configs.get_configurations();
    auto orbitals = reference_configs.get_orbitals();

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
        orbitals, determinants, t1_aa, t1_bb, t2_abab, t2_aaaa, t2_bbbb);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::pair<size_t, size_t> CoupledClusterContainer::get_total_num_electrons()
    const {
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
  const auto& determinants = get_references();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }
  auto [n_alpha, n_beta] = determinants[0].get_n_electrons();
  return {n_alpha, n_beta};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
CoupledClusterContainer::get_total_orbital_occupations() const {
  const auto& determinants = get_references();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }

  // Get the total number of orbitals from the orbital basis set
  const int num_orbitals =
      static_cast<int>(get_orbitals()->get_num_molecular_orbitals());

  Eigen::VectorXd alpha_occupations = Eigen::VectorXd::Zero(num_orbitals);
  Eigen::VectorXd beta_occupations = Eigen::VectorXd::Zero(num_orbitals);

  // Get inactive space indices and mark them as doubly occupied
  auto [alpha_inactive_indices, beta_inactive_indices] =
      get_orbitals()->get_inactive_space_indices();

  // Set inactive orbitals as doubly occupied (occupation = 1.0)
  for (size_t inactive_idx : alpha_inactive_indices) {
    if (inactive_idx < static_cast<size_t>(num_orbitals)) {
      alpha_occupations(inactive_idx) = 1.0;
    }
  }
  for (size_t inactive_idx : beta_inactive_indices) {
    if (inactive_idx < static_cast<size_t>(num_orbitals)) {
      beta_occupations(inactive_idx) = 1.0;
    }
  }

  // For active space orbitals, get occupations from 1RDM eigenvalues
  if (has_one_rdm_spin_dependent()) {
    // Get active space occupations using the dedicated method
    auto [alpha_active_occs, beta_active_occs] =
        get_active_orbital_occupations();

    // Get active space indices to map back to total orbital indices
    auto [alpha_active_indices, beta_active_indices] =
        get_orbitals()->get_active_space_indices();

    // Map active space occupations to total orbital indices
    for (size_t active_idx = 0; active_idx < alpha_active_indices.size();
         ++active_idx) {
      size_t orbital_idx = alpha_active_indices[active_idx];
      if (orbital_idx < static_cast<size_t>(num_orbitals) &&
          active_idx < alpha_active_occs.size()) {
        alpha_occupations(orbital_idx) = alpha_active_occs(active_idx);
      }
    }

    for (size_t active_idx = 0; active_idx < beta_active_indices.size();
         ++active_idx) {
      size_t orbital_idx = beta_active_indices[active_idx];
      if (orbital_idx < static_cast<size_t>(num_orbitals) &&
          active_idx < beta_active_occs.size()) {
        beta_occupations(orbital_idx) = beta_active_occs(active_idx);
      }
    }
  } else {
    throw std::runtime_error(
        "1RDM must be available to compute orbital occupations");
  }

  return {alpha_occupations, beta_occupations};
}

std::pair<Eigen::VectorXd, Eigen::VectorXd>
CoupledClusterContainer::get_active_orbital_occupations() const {
  const auto& determinants = get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }

  // Get the active space indices
  auto [alpha_active_indices, beta_active_indices] =
      get_orbitals()->get_active_space_indices();

  // If no active space is defined, return empty vectors
  if (alpha_active_indices.empty()) {
    return {Eigen::VectorXd::Zero(0), Eigen::VectorXd::Zero(0)};
  }

  const int num_active_orbitals = static_cast<int>(alpha_active_indices.size());

  Eigen::VectorXd alpha_occupations =
      Eigen::VectorXd::Zero(num_active_orbitals);
  Eigen::VectorXd beta_occupations = Eigen::VectorXd::Zero(num_active_orbitals);

  // For active space orbitals, get occupations from 1RDM eigenvalues
  if (has_one_rdm_spin_dependent()) {
    const auto& rdm_tuple = get_active_one_rdm_spin_dependent();
    const auto& alpha_rdm_var = std::get<0>(rdm_tuple);
    const auto& beta_rdm_var = std::get<1>(rdm_tuple);
    // Extract real matrices (assuming real for now)
    if (detail::is_matrix_variant_complex(alpha_rdm_var) ||
        detail::is_matrix_variant_complex(beta_rdm_var)) {
      throw std::runtime_error(
          "Complex 1RDM diagonalization not yet implemented");
    }

    const Eigen::MatrixXd& alpha_rdm = std::get<Eigen::MatrixXd>(alpha_rdm_var);
    const Eigen::MatrixXd& beta_rdm = std::get<Eigen::MatrixXd>(beta_rdm_var);

    // Diagonalize alpha 1RDM to get occupations
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> alpha_solver(alpha_rdm);
    if (alpha_solver.info() != Eigen::Success) {
      throw std::runtime_error("Failed to diagonalize alpha 1RDM");
    }
    Eigen::VectorXd alpha_eigenvalues = alpha_solver.eigenvalues();

    // reverse to have descending order
    std::reverse(alpha_eigenvalues.data(),
                 alpha_eigenvalues.data() + alpha_eigenvalues.size());

    // Diagonalize beta 1RDM to get occupations
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> beta_solver(beta_rdm);
    if (beta_solver.info() != Eigen::Success) {
      throw std::runtime_error("Failed to diagonalize beta 1RDM");
    }
    Eigen::VectorXd beta_eigenvalues = beta_solver.eigenvalues();

    // reverse to have descending order
    std::reverse(beta_eigenvalues.data(),
                 beta_eigenvalues.data() + beta_eigenvalues.size());

    // Copy eigenvalues directly as active space occupations
    for (int active_idx = 0;
         active_idx < std::min(num_active_orbitals,
                               static_cast<int>(alpha_eigenvalues.size()));
         ++active_idx) {
      alpha_occupations(active_idx) = alpha_eigenvalues(active_idx);
    }

    for (int active_idx = 0;
         active_idx < std::min(num_active_orbitals,
                               static_cast<int>(beta_eigenvalues.size()));
         ++active_idx) {
      beta_occupations(active_idx) = beta_eigenvalues(active_idx);
    }
  } else {
    throw std::runtime_error(
        "1RDM must be available to compute orbital occupations");
  }

  return {alpha_occupations, beta_occupations};
}

std::string CoupledClusterContainer::get_container_type() const {
  return "coupled_cluster";
}

bool CoupledClusterContainer::is_complex() const {
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

}  // namespace qdk::chemistry::data

/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#include <algorithm>
#include <optional>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/wavefunction_containers/amplitude_container.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <variant>

#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

namespace detail {
constexpr const char* kNoOccupationMessage =
    "Orbital occupations require reduced density matrices, which are not "
    "available for amplitude wavefunctions.";
}  // namespace detail

std::string amplitude_type_to_string(AmplitudeType type) {
  switch (type) {
    case AmplitudeType::MollerPlesset:
      return "moller_plesset";
    case AmplitudeType::CoupledCluster:
      return "coupled_cluster";
    case AmplitudeType::Unspecified:
      return "unspecified";
  }
  return "unspecified";
}

AmplitudeType amplitude_type_from_string(const std::string& s) {
  if (s == "moller_plesset" || s == "mp2") {
    return AmplitudeType::MollerPlesset;
  }
  if (s == "coupled_cluster" || s == "ccsd") {
    return AmplitudeType::CoupledCluster;
  }
  return AmplitudeType::Unspecified;
}

AmplitudeContainer::AmplitudeContainer(
    std::shared_ptr<Orbitals> orbitals,
    std::shared_ptr<Wavefunction> wavefunction, AmplitudeType amplitude_type,
    const std::optional<VectorVariant>& t1_amplitudes,
    const std::optional<VectorVariant>& t2_amplitudes, std::string sector)
    : AmplitudeContainer(orbitals, wavefunction, amplitude_type, t1_amplitudes,
                         std::nullopt, t2_amplitudes, std::nullopt,
                         std::nullopt, std::move(sector)) {
  QDK_LOG_TRACE_ENTERING();
}

AmplitudeContainer::AmplitudeContainer(
    std::shared_ptr<Orbitals> orbitals,
    std::shared_ptr<Wavefunction> wavefunction, AmplitudeType amplitude_type,
    const std::optional<VectorVariant>& t1_amplitudes_aa,
    const std::optional<VectorVariant>& t1_amplitudes_bb,
    const std::optional<VectorVariant>& t2_amplitudes_abab,
    const std::optional<VectorVariant>& t2_amplitudes_aaaa,
    const std::optional<VectorVariant>& t2_amplitudes_bbbb, std::string sector)
    : WavefunctionContainer(
          WavefunctionType::NotSelfDual),  // Amplitude wavefunctions are
                                           // always NotSelfDual
      _orbitals(orbitals),
      _sector(std::move(sector)),
      _wavefunction(wavefunction),
      _amplitude_type(amplitude_type) {
  QDK_LOG_TRACE_ENTERING();
  if (!orbitals) {
    throw std::invalid_argument("Orbitals cannot be null");
  }
  if (!wavefunction) {
    throw std::invalid_argument("Wavefunction cannot be null");
  }

  // Validate partial amplitude provision.
  if (!t1_amplitudes_aa && t1_amplitudes_bb) {
    throw std::invalid_argument(
        "Cannot provide unrestricted beta T1 amplitudes without alpha T1 "
        "amplitudes");
  }
  if (!t2_amplitudes_abab && (t2_amplitudes_aaaa || t2_amplitudes_bbbb)) {
    throw std::invalid_argument(
        "Cannot provide unrestricted T2 alpha-alpha or beta-beta amplitudes "
        "without T2 alpha-beta amplitudes");
  }

  // For unrestricted orbitals, if amplitudes are provided they must include all
  // spin components. Construction without any amplitudes is allowed.
  if (!orbitals->is_restricted()) {
    if (t1_amplitudes_aa && !t1_amplitudes_bb) {
      throw std::invalid_argument(
          "Both alpha and beta T1 amplitudes must be provided for unrestricted "
          "orbitals");
    }
    if (t2_amplitudes_abab && (!t2_amplitudes_aaaa || !t2_amplitudes_bbbb)) {
      throw std::invalid_argument(
          "All spin components of T2 amplitudes must be provided for "
          "unrestricted orbitals");
    }
  }

  // Validate amplitude sizes against the reference, but only when amplitudes
  // were actually provided.
  const bool any_amplitude = t1_amplitudes_aa || t1_amplitudes_bb ||
                             t2_amplitudes_abab || t2_amplitudes_aaaa ||
                             t2_amplitudes_bbbb;
  if (any_amplitude) {
    auto get_vector_size = [](const VectorVariant& vec) -> size_t {
      if (std::holds_alternative<Eigen::VectorXd>(vec)) {
        return std::get<Eigen::VectorXd>(vec).size();
      } else if (std::holds_alternative<Eigen::VectorXcd>(vec)) {
        return std::get<Eigen::VectorXcd>(vec).size();
      }
      return 0;
    };

    const auto& references = _wavefunction->get_total_determinants();
    if (references.empty()) {
      throw std::invalid_argument("Reference determinants cannot be empty");
    }
    // Amplitudes are sized by the active space, so use the active electron
    // count (consistent with the producing algorithm).
    auto [n_alpha, n_beta] = _wavefunction->get_active_num_electrons();
    size_t active_space_size = orbitals->get_num_molecular_orbitals();
    if (orbitals->has_active_space()) {
      active_space_size = orbitals->num_active_orbitals();
    }

    size_t n_occ_alpha = n_alpha;
    size_t n_occ_beta = n_beta;
    size_t n_vir_alpha = active_space_size - n_occ_alpha;
    size_t n_vir_beta = active_space_size - n_occ_beta;

    if (t1_amplitudes_aa) {
      size_t expected = n_occ_alpha * n_vir_alpha;
      size_t actual = get_vector_size(*t1_amplitudes_aa);
      if (actual != expected) {
        throw std::invalid_argument(
            "T1 alpha amplitude size mismatch: expected " +
            std::to_string(expected) + " (nocc=" + std::to_string(n_occ_alpha) +
            " * nvir=" + std::to_string(n_vir_alpha) + "), got " +
            std::to_string(actual));
      }
    }
    if (t1_amplitudes_bb) {
      size_t expected = n_occ_beta * n_vir_beta;
      size_t actual = get_vector_size(*t1_amplitudes_bb);
      if (actual != expected) {
        throw std::invalid_argument(
            "T1 beta amplitude size mismatch: expected " +
            std::to_string(expected) + " (nocc=" + std::to_string(n_occ_beta) +
            " * nvir=" + std::to_string(n_vir_beta) + "), got " +
            std::to_string(actual));
      }
    }
    if (t2_amplitudes_abab) {
      size_t expected = n_occ_alpha * n_occ_beta * n_vir_alpha * n_vir_beta;
      size_t actual = get_vector_size(*t2_amplitudes_abab);
      if (actual != expected) {
        throw std::invalid_argument(
            "T2 alpha-beta amplitude size mismatch: expected " +
            std::to_string(expected) +
            " (nocc_a=" + std::to_string(n_occ_alpha) +
            " * nocc_b=" + std::to_string(n_occ_beta) +
            " * nvir_a=" + std::to_string(n_vir_alpha) + " * nvir_b=" +
            std::to_string(n_vir_beta) + "), got " + std::to_string(actual));
      }
    }
    if (t2_amplitudes_aaaa) {
      size_t expected = n_occ_alpha * n_occ_alpha * n_vir_alpha * n_vir_alpha;
      size_t actual = get_vector_size(*t2_amplitudes_aaaa);
      if (actual != expected) {
        throw std::invalid_argument(
            "T2 alpha-alpha amplitude size mismatch: expected " +
            std::to_string(expected) + ", got " + std::to_string(actual));
      }
    }
    if (t2_amplitudes_bbbb) {
      size_t expected = n_occ_beta * n_occ_beta * n_vir_beta * n_vir_beta;
      size_t actual = get_vector_size(*t2_amplitudes_bbbb);
      if (actual != expected) {
        throw std::invalid_argument(
            "T2 beta-beta amplitude size mismatch: expected " +
            std::to_string(expected) + ", got " + std::to_string(actual));
      }
    }
  }

  // Store amplitudes; missing same-kind spin components alias the provided one
  // (restricted convention).
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

std::unique_ptr<WavefunctionContainer> AmplitudeContainer::clone() const {
  QDK_LOG_TRACE_ENTERING();
  auto as_optional = [](const std::shared_ptr<VectorVariant>& p) {
    return p ? std::optional<VectorVariant>(*p) : std::nullopt;
  };
  return std::make_unique<AmplitudeContainer>(
      _orbitals, _wavefunction, _amplitude_type, as_optional(_t1_amplitudes_aa),
      as_optional(_t1_amplitudes_bb), as_optional(_t2_amplitudes_abab),
      as_optional(_t2_amplitudes_aaaa), as_optional(_t2_amplitudes_bbbb),
      _sector);
}

std::shared_ptr<Orbitals> AmplitudeContainer::get_orbitals() const {
  QDK_LOG_TRACE_ENTERING();
  return _orbitals;
}

std::vector<std::string> AmplitudeContainer::sectors() const {
  QDK_LOG_TRACE_ENTERING();
  return {_sector};
}

std::shared_ptr<const Orbitals> AmplitudeContainer::sector_basis(
    const std::string& name) const {
  QDK_LOG_TRACE_ENTERING();
  if (name == _sector) {
    return _orbitals;
  }
  throw std::out_of_range("Container has no sector named '" + name + "'");
}

std::shared_ptr<Wavefunction> AmplitudeContainer::get_wavefunction() const {
  QDK_LOG_TRACE_ENTERING();
  return _wavefunction;
}

AmplitudeType AmplitudeContainer::get_amplitude_type() const {
  QDK_LOG_TRACE_ENTERING();
  return _amplitude_type;
}

std::pair<const AmplitudeContainer::VectorVariant&,
          const AmplitudeContainer::VectorVariant&>
AmplitudeContainer::get_t1_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_t1_amplitudes()) {
    throw std::runtime_error("T1 amplitudes not available");
  }
  return std::make_pair(std::cref(*_t1_amplitudes_aa),
                        std::cref(*_t1_amplitudes_bb));
}

std::tuple<const AmplitudeContainer::VectorVariant&,
           const AmplitudeContainer::VectorVariant&,
           const AmplitudeContainer::VectorVariant&>
AmplitudeContainer::get_t2_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  if (!has_t2_amplitudes()) {
    throw std::runtime_error("T2 amplitudes not available");
  }
  return std::make_tuple(std::cref(*_t2_amplitudes_abab),
                         std::cref(*_t2_amplitudes_aaaa),
                         std::cref(*_t2_amplitudes_bbbb));
}

bool AmplitudeContainer::has_t1_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  return _t1_amplitudes_aa != nullptr;
}

bool AmplitudeContainer::has_t2_amplitudes() const {
  QDK_LOG_TRACE_ENTERING();
  return _t2_amplitudes_abab != nullptr;
}

AmplitudeContainer::ScalarVariant AmplitudeContainer::overlap(
    const WavefunctionContainer&) const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "overlap() is not implemented for amplitude wavefunctions.");
}

double AmplitudeContainer::norm() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(
      "norm() is not implemented for amplitude wavefunctions.");
}

bool AmplitudeContainer::contains_determinant(const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  return contains_reference(det);
}

bool AmplitudeContainer::contains_reference(const Configuration& det) const {
  QDK_LOG_TRACE_ENTERING();
  const auto& references = _wavefunction->get_total_determinants();
  return std::find(references.begin(), references.end(), det) !=
         references.end();
}

void AmplitudeContainer::clear_caches() const {
  QDK_LOG_TRACE_ENTERING();
  _clear_rdms();
}

std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
AmplitudeContainer::total_num_particles() const {
  QDK_LOG_TRACE_ENTERING();
  const auto& determinants = _wavefunction->get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }
  if (determinants[0].bits_per_mode() != 2) {
    // Generic (non-spin-½): aggregate count, no spin decomposition. Use a
    // single inactive channel; num_inactive_orbitals() reads the alpha (or, for
    // spin-free bases, the sole trivial) channel.
    std::size_t active = determinants[0].total_occupation();
    return _make_particle_count(
        active + get_orbitals()->num_inactive_orbitals(), 0);
  }
  auto [n_alpha, n_beta] = determinants[0].get_n_electrons();
  const auto inactive = get_orbitals()->inactive_indices();
  return _make_particle_count(
      n_alpha + spin_channel_indices(inactive, /*beta=*/false).size(),
      n_beta + spin_channel_indices(inactive, /*beta=*/true).size());
}

std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
AmplitudeContainer::active_num_particles() const {
  QDK_LOG_TRACE_ENTERING();
  const auto& determinants = _wavefunction->get_active_determinants();
  if (determinants.empty()) {
    throw std::runtime_error("No determinants available");
  }
  if (determinants[0].bits_per_mode() != 2) {
    return _make_particle_count(determinants[0].total_occupation(), 0);
  }
  auto [n_alpha, n_beta] = determinants[0].get_n_electrons();
  return _make_particle_count(n_alpha, n_beta);
}

std::shared_ptr<const SymmetryBlockedTensor<1>>
AmplitudeContainer::total_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(detail::kNoOccupationMessage);
}

std::shared_ptr<const SymmetryBlockedTensor<1>>
AmplitudeContainer::active_orbital_occupations() const {
  QDK_LOG_TRACE_ENTERING();
  throw std::runtime_error(detail::kNoOccupationMessage);
}

std::string AmplitudeContainer::get_container_type() const {
  QDK_LOG_TRACE_ENTERING();
  return "amplitude";
}

bool AmplitudeContainer::is_complex() const {
  QDK_LOG_TRACE_ENTERING();
  auto is_complex_variant = [](const std::shared_ptr<VectorVariant>& p) {
    return p && std::holds_alternative<Eigen::VectorXcd>(*p);
  };
  return is_complex_variant(_t1_amplitudes_aa) ||
         is_complex_variant(_t1_amplitudes_bb) ||
         is_complex_variant(_t2_amplitudes_abab) ||
         is_complex_variant(_t2_amplitudes_aaaa) ||
         is_complex_variant(_t2_amplitudes_bbbb);
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

nlohmann::json AmplitudeContainer::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;
  j["version"] = SERIALIZATION_VERSION;
  j["container_type"] = get_container_type();
  j["amplitude_type"] = amplitude_type_to_string(_amplitude_type);
  j["sector"] = _sector;

  if (_orbitals) {
    j["orbitals"] = _orbitals->to_json();
  }
  if (_wavefunction) {
    j["wavefunction"] = _wavefunction->to_json();
  }

  bool is_complex = this->is_complex();
  j["is_complex"] = is_complex;

  auto store = [&](const char* key, const std::shared_ptr<VectorVariant>& p) {
    if (!p) {
      return;
    }
    if (is_complex) {
      j[key] =
          vector_variant_to_json(std::get<Eigen::VectorXcd>(*p), is_complex);
    } else {
      j[key] =
          vector_variant_to_json(std::get<Eigen::VectorXd>(*p), is_complex);
    }
  };
  store("t1_amplitudes_aa", _t1_amplitudes_aa);
  store("t1_amplitudes_bb", _t1_amplitudes_bb);
  store("t2_amplitudes_abab", _t2_amplitudes_abab);
  store("t2_amplitudes_aaaa", _t2_amplitudes_aaaa);
  store("t2_amplitudes_bbbb", _t2_amplitudes_bbbb);

  return j;
}

std::unique_ptr<AmplitudeContainer> AmplitudeContainer::from_json(
    const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  try {
    if (!j.contains("version")) {
      throw std::runtime_error("JSON does not contain version information");
    }
    validate_serialization_version(SERIALIZATION_VERSION, j["version"]);

    auto orbitals = Orbitals::from_json(j.at("orbitals"));
    std::shared_ptr<Wavefunction> wavefunction = nullptr;
    if (j.contains("wavefunction") && !j.at("wavefunction").is_null()) {
      wavefunction = Wavefunction::from_json(j.at("wavefunction"));
    }

    // Sector name; defaults to the electronic sector when unspecified.
    std::string sector =
        j.value("sector", std::string(Wavefunction::DEFAULT_SECTOR));

    AmplitudeType amplitude_type = AmplitudeType::Unspecified;
    if (j.contains("amplitude_type")) {
      amplitude_type =
          amplitude_type_from_string(j.at("amplitude_type").get<std::string>());
    }

    bool is_complex = j.value("is_complex", false);
    auto load = [&](const char* key) -> std::optional<VectorVariant> {
      return j.contains(key) ? std::optional<VectorVariant>(
                                   json_to_vector_variant(j[key], is_complex))
                             : std::nullopt;
    };
    return std::make_unique<AmplitudeContainer>(
        orbitals, wavefunction, amplitude_type, load("t1_amplitudes_aa"),
        load("t1_amplitudes_bb"), load("t2_amplitudes_abab"),
        load("t2_amplitudes_aaaa"), load("t2_amplitudes_bbbb"), sector);
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse AmplitudeContainer from JSON: " +
                             std::string(e.what()));
  }
}

void AmplitudeContainer::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    H5::Attribute version_attr = group.createAttribute(
        "version", string_type, H5::DataSpace(H5S_SCALAR));
    std::string version_str = SERIALIZATION_VERSION;
    version_attr.write(string_type, version_str);
    version_attr.close();

    std::string container_type = get_container_type();
    H5::Attribute container_type_attr = group.createAttribute(
        "container_type", string_type, H5::DataSpace(H5S_SCALAR));
    container_type_attr.write(string_type, container_type);

    std::string amplitude_type_str = amplitude_type_to_string(_amplitude_type);
    H5::Attribute amplitude_type_attr = group.createAttribute(
        "amplitude_type", string_type, H5::DataSpace(H5S_SCALAR));
    amplitude_type_attr.write(string_type, amplitude_type_str);

    H5::Attribute sector_attr =
        group.createAttribute("sector", string_type, H5::DataSpace(H5S_SCALAR));
    sector_attr.write(string_type, _sector);

    bool is_complex = this->is_complex();
    H5::Attribute is_complex_attr = group.createAttribute(
        "is_complex", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));
    is_complex_attr.write(H5::PredType::NATIVE_HBOOL, &is_complex);

    auto store = [&](const char* key, const std::shared_ptr<VectorVariant>& p) {
      if (p) {
        write_vector_to_hdf5(group, key, p, is_complex);
      }
    };
    store("t1_amplitudes_aa", _t1_amplitudes_aa);
    store("t1_amplitudes_bb", _t1_amplitudes_bb);
    store("t2_amplitudes_abab", _t2_amplitudes_abab);
    store("t2_amplitudes_aaaa", _t2_amplitudes_aaaa);
    store("t2_amplitudes_bbbb", _t2_amplitudes_bbbb);

    if (_orbitals) {
      H5::Group orbitals_group = group.createGroup("orbitals");
      _orbitals->to_hdf5(orbitals_group);
    }
    if (_wavefunction) {
      H5::Group wavefunction_group = group.createGroup("wavefunction");
      _wavefunction->to_hdf5(wavefunction_group);
    }
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::unique_ptr<AmplitudeContainer> AmplitudeContainer::from_hdf5(
    H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version_str;
    version_attr.read(string_type, version_str);
    validate_serialization_version(SERIALIZATION_VERSION, version_str);

    std::shared_ptr<Orbitals> orbitals = nullptr;
    if (group.nameExists("orbitals")) {
      H5::Group orbitals_group = group.openGroup("orbitals");
      orbitals = Orbitals::from_hdf5(orbitals_group);
    }
    std::shared_ptr<Wavefunction> wavefunction = nullptr;
    if (group.nameExists("wavefunction")) {
      H5::Group wavefunction_group = group.openGroup("wavefunction");
      wavefunction = Wavefunction::from_hdf5(wavefunction_group);
    }

    // Sector name; defaults to the electronic sector when unspecified.
    std::string sector = Wavefunction::DEFAULT_SECTOR;
    if (group.attrExists("sector")) {
      group.openAttribute("sector").read(string_type, sector);
    }

    AmplitudeType amplitude_type = AmplitudeType::Unspecified;
    if (group.attrExists("amplitude_type")) {
      H5::Attribute amplitude_type_attr = group.openAttribute("amplitude_type");
      std::string amplitude_type_str;
      amplitude_type_attr.read(string_type, amplitude_type_str);
      amplitude_type = amplitude_type_from_string(amplitude_type_str);
    }

    bool is_complex = false;
    if (group.attrExists("is_complex")) {
      H5::Attribute is_complex_attr = group.openAttribute("is_complex");
      hbool_t flag;
      is_complex_attr.read(H5::PredType::NATIVE_HBOOL, &flag);
      is_complex = (flag != 0);
    }

    auto load = [&](const char* key) -> std::optional<VectorVariant> {
      return group.nameExists(key)
                 ? std::optional<VectorVariant>(
                       load_vector_variant_from_group(group, key, is_complex))
                 : std::nullopt;
    };
    return std::make_unique<AmplitudeContainer>(
        orbitals, wavefunction, amplitude_type, load("t1_amplitudes_aa"),
        load("t1_amplitudes_bb"), load("t2_amplitudes_abab"),
        load("t2_amplitudes_aaaa"), load("t2_amplitudes_bbbb"), sector);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void AmplitudeContainer::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  WavefunctionContainer::hash_update(ctx);
  hash_value(ctx, get_container_type());
  hash_value(ctx, _sector);
  hash_value(ctx, amplitude_type_to_string(_amplitude_type));
  if (_orbitals) {
    hash_field_presence(ctx, true);
    hash_value(ctx, _orbitals->content_hash());
  } else {
    hash_field_presence(ctx, false);
  }
  if (_wavefunction) {
    hash_field_presence(ctx, true);
    hash_value(ctx, _wavefunction->content_hash());
  } else {
    hash_field_presence(ctx, false);
  }
  hash_value(ctx, _t1_amplitudes_aa);
  hash_value(ctx, _t1_amplitudes_bb);
  hash_value(ctx, _t2_amplitudes_abab);
  hash_value(ctx, _t2_amplitudes_aaaa);
  hash_value(ctx, _t2_amplitudes_bbbb);
}

}  // namespace qdk::chemistry::data

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <array>
#include <complex>
#include <cstdint>
#include <limits>
#include <numeric>
#include <qdk/chemistry/data/configuration_set.hpp>
#include <qdk/chemistry/data/wavefunction_containers/mps_wavefunction.hpp>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

#include "../json_serialization.hpp"

namespace qdk::chemistry::data {
namespace {

constexpr auto index_limit =
    static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max());
constexpr std::array<const char*, 3> order_keys{
    "left_sector_order", "physical_sector_order", "right_sector_order"};

std::vector<Configuration> default_physical_basis(std::size_t dimension) {
  std::vector<Configuration> basis;
  if (dimension == 2) {
    for (const auto* state : {"0", "1"}) {
      basis.push_back(Configuration::from_bitstring(state));
    }
  } else if (dimension == 4) {
    for (const auto* state : {"0", "u", "d", "2"}) {
      basis.push_back(Configuration::from_spin_half_string(state));
    }
  } else {
    throw std::invalid_argument(
        "MPS sites outside dimension two or four require a physical basis.");
  }
  return basis;
}

std::size_t read_index(const nlohmann::json& value) {
  if (!value.is_number_integer() ||
      (!value.is_number_unsigned() && value.get<std::int64_t>() < 0) ||
      value.get<std::uint64_t>() > std::numeric_limits<std::size_t>::max()) {
    throw std::invalid_argument("MPS indices must be nonnegative integers.");
  }
  return value.get<std::size_t>();
}

void validate_metadata(const nlohmann::json& json,
                       const std::string& expected_version) {
  validate_serialization_version(expected_version,
                                 json.at("version").get<std::string>());
  if (json.at("container_type") != "mps") {
    throw std::invalid_argument("Unsupported MPS container type.");
  }
  if (json.at("scalar") != "real" && json.at("scalar") != "complex") {
    throw std::invalid_argument("Unsupported MPS scalar type.");
  }
  if (!json.at("sites").is_array() || json.at("sites").empty() ||
      !json.at("site_to_orbital_order").is_array() ||
      json.at("site_to_orbital_order").empty()) {
    throw std::invalid_argument(
        "Serialized MPS requires explicit sites and ordering.");
  }
}

nlohmann::json site_metadata(const MPSSite& site) {
  nlohmann::json json;
  for (std::size_t slot = 0; slot < order_keys.size(); ++slot) {
    auto& order = json[order_keys[slot]];
    order = nlohmann::json::array();
    for (const auto& label : site.sector_orders()[slot]) {
      order.push_back(label.to_json());
    }
  }
  json["physical_basis"] = nlohmann::json::array();
  for (const auto& state : site.physical_basis()) {
    json["physical_basis"].push_back(state.to_json());
  }
  return json;
}

MPSContainer::SitePtr site_from_metadata(const nlohmann::json& json,
                                         MPSSite::TensorPtr tensor) {
  MPSSite::SectorOrders orders;
  for (std::size_t slot = 0; slot < order_keys.size(); ++slot) {
    const auto& order = json.at(order_keys[slot]);
    if (!order.is_array()) {
      throw std::invalid_argument("MPS sector orders must be arrays.");
    }
    for (const auto& label : order) {
      orders[slot].push_back(SymmetryLabel::from_json(label));
    }
  }
  const auto& basis_json = json.at("physical_basis");
  if (!basis_json.is_array() || basis_json.empty()) {
    throw std::invalid_argument(
        "Serialized MPS sites require a physical basis.");
  }
  std::vector<Configuration> basis;
  for (const auto& state : basis_json) {
    const auto bits = read_index(state.at("bits_per_mode"));
    if (bits != 1 && bits != 2) {
      throw std::invalid_argument(
          "MPS physical states require binary or spin-half configurations.");
    }
    basis.push_back(Configuration::from_json(state));
  }
  return std::make_shared<const MPSSite>(std::move(tensor), std::move(orders),
                                         std::move(basis));
}

std::unique_ptr<MPSContainer> mps_from_metadata(
    const nlohmann::json& json, std::shared_ptr<Orbitals> orbitals,
    std::vector<MPSContainer::SitePtr> sites) {
  std::optional<std::size_t> center;
  if (!json.at("orthogonality_center").is_null()) {
    center = read_index(json.at("orthogonality_center"));
  }
  std::vector<std::size_t> order;
  for (const auto& index : json.at("site_to_orbital_order")) {
    order.push_back(read_index(index));
  }
  auto read_counts = [](const nlohmann::json& counts)
      -> std::shared_ptr<const MPSContainer::ParticleCount> {
    return counts.is_null() ? nullptr
                            : MPSContainer::ParticleCount::from_json(counts);
  };
  return std::make_unique<MPSContainer>(
      std::move(sites), std::move(orbitals),
      read_counts(json.at("total_num_particles")),
      read_counts(json.at("active_num_particles")), center, std::move(order));
}

}  // namespace

MPSSite::MPSSite(TensorPtr tensor, SectorOrders sector_orders,
                 std::vector<Configuration> physical_basis)
    : _tensor(std::move(tensor)),
      _sector_orders(std::move(sector_orders)),
      _physical_basis(std::move(physical_basis)) {
  if (!_tensor) {
    throw std::invalid_argument("MPS site requires a tensor.");
  }
  _validate();
}

void MPSSite::_validate() {
  std::visit(
      [&](const auto& tensor) {
        for (std::size_t slot = 0; slot < 3; ++slot) {
          const auto& extents = tensor.extents()[slot];
          const auto& order = _sector_orders[slot];
          if (extents.empty()) {
            throw std::invalid_argument("MPS index spaces must be nonempty.");
          }
          if (order.size() != extents.size()) {
            throw std::invalid_argument(
                "MPS sector order must contain every sector exactly once.");
          }
          auto& offsets = _tensor_layout.offsets[slot];
          std::size_t offset = 0;
          for (const auto& label : order) {
            const auto extent = extents.find(label);
            if (extent == extents.end() ||
                !offsets.emplace(label, static_cast<Eigen::Index>(offset))
                     .second) {
              throw std::invalid_argument(
                  "MPS sector order contains a missing or duplicate sector.");
            }
            if (extent->second == 0 || extent->second > index_limit - offset) {
              throw std::invalid_argument(
                  "MPS sector extents must be positive and fit Eigen::Index.");
            }
            offset += extent->second;
          }
          _tensor_layout.dimensions[slot] = static_cast<Eigen::Index>(offset);
          for (const auto& axis : tensor.symmetries()[slot]->axes()) {
            if (axis.equivalent()) {
              throw std::invalid_argument(
                  "MPS axes must not enable equivalent-sector aliasing.");
            }
          }
        }
        for (const auto& [labels, block] : tensor.blocks()) {
          const auto left = tensor.extents()[0].at(labels[0]);
          const auto physical = tensor.extents()[1].at(labels[1]);
          const auto right = tensor.extents()[2].at(labels[2]);
          if (left > index_limit / physical ||
              block->rows() != static_cast<Eigen::Index>(left * physical) ||
              block->cols() != static_cast<Eigen::Index>(right)) {
            throw std::invalid_argument(
                "MPS blocks must be packed as (left * physical, right).");
          }
          if (!block->allFinite()) {
            throw std::invalid_argument("MPS tensor entries must be finite.");
          }
        }
      },
      *_tensor);

  if (_physical_basis.empty()) {
    _physical_basis = default_physical_basis(physical_dimension());
  }
  if (_physical_basis.size() != physical_dimension()) {
    throw std::invalid_argument(
        "MPS physical basis size must match the physical dimension.");
  }
  const auto bits = _physical_basis.front().bits_per_mode();
  if (bits != 1 && bits != 2) {
    throw std::invalid_argument(
        "MPS physical states require binary or spin-half configurations.");
  }
  std::array<bool, 4> seen{};
  for (const auto& state : _physical_basis) {
    if (state.bits_per_mode() != bits || state.capacity() == 0) {
      throw std::invalid_argument(
          "MPS physical states must share a nonempty configuration encoding.");
    }
    for (std::size_t mode = 1; mode < state.capacity(); ++mode) {
      if (state.get_mode_state(mode) != 0) {
        throw std::invalid_argument(
            "MPS physical states may occupy only their first mode.");
      }
    }
    const auto value = state.get_mode_state(0);
    if (value >= seen.size() || seen[value]) {
      throw std::invalid_argument("MPS physical states must be distinct.");
    }
    seen[value] = true;
  }
}

std::size_t MPSSite::left_bond_dimension() const {
  return static_cast<std::size_t>(_tensor_layout.dimensions[0]);
}
std::size_t MPSSite::physical_dimension() const {
  return static_cast<std::size_t>(_tensor_layout.dimensions[1]);
}
std::size_t MPSSite::right_bond_dimension() const {
  return static_cast<std::size_t>(_tensor_layout.dimensions[2]);
}
bool MPSSite::is_complex() const { return _tensor->index() == 1; }

MPSSite::DenseMatrixVariant MPSSite::to_dense() const {
  const auto left = left_bond_dimension();
  const auto physical = physical_dimension();
  const auto right = right_bond_dimension();
  if (left > index_limit / physical || left * physical > index_limit / right) {
    throw std::overflow_error("Dense MPS site size exceeds Eigen::Index.");
  }
  return std::visit(
      [&](const auto& tensor) -> DenseMatrixVariant {
        using TensorType = std::decay_t<decltype(tensor)>;
        using Matrix =
            std::remove_const_t<typename TensorType::BlockPtr::element_type>;
        Matrix dense = Matrix::Zero(static_cast<Eigen::Index>(left * physical),
                                    static_cast<Eigen::Index>(right));
        const auto& offsets = _tensor_layout.offsets;
        for (const auto& [labels, block] : tensor.blocks()) {
          const auto local_left = tensor.extents()[0].at(labels[0]);
          const auto local_physical = tensor.extents()[1].at(labels[1]);
          for (std::size_t l = 0; l < local_left; ++l) {
            for (std::size_t p = 0; p < local_physical; ++p) {
              const auto row = (offsets[0].at(labels[0]) + l) * physical +
                               offsets[1].at(labels[1]) + p;
              dense.block(static_cast<Eigen::Index>(row),
                          static_cast<Eigen::Index>(offsets[2].at(labels[2])),
                          1, block->cols()) =
                  block->row(static_cast<Eigen::Index>(l * local_physical + p));
            }
          }
        }
        return dense;
      },
      *_tensor);
}

MPSContainer::MPSContainer(
    std::vector<SitePtr> sites, std::shared_ptr<Orbitals> orbitals,
    std::shared_ptr<const ParticleCount> total_num_particles,
    std::shared_ptr<const ParticleCount> active_num_particles,
    std::optional<std::size_t> orthogonality_center,
    std::vector<std::size_t> site_to_orbital_order,
    const std::optional<MatrixVariant>& one_rdm_spin_traced,
    const std::optional<MatrixVariant>& one_rdm_aa,
    const std::optional<MatrixVariant>& one_rdm_bb,
    const std::optional<VectorVariant>& two_rdm_spin_traced,
    const std::optional<VectorVariant>& two_rdm_aaaa,
    const std::optional<VectorVariant>& two_rdm_aabb,
    const std::optional<VectorVariant>& two_rdm_bbbb)
    : WavefunctionContainer(one_rdm_spin_traced, one_rdm_aa, one_rdm_bb,
                            two_rdm_spin_traced, two_rdm_aaaa, two_rdm_aabb,
                            two_rdm_bbbb),
      _sites(std::move(sites)),
      _orbitals(std::move(orbitals)),
      _total_num_particles(std::move(total_num_particles)),
      _active_num_particles(std::move(active_num_particles)),
      _orthogonality_center(orthogonality_center),
      _site_to_orbital_order(std::move(site_to_orbital_order)) {
  if (_site_to_orbital_order.empty()) {
    _site_to_orbital_order.resize(num_sites());
    std::iota(_site_to_orbital_order.begin(), _site_to_orbital_order.end(),
              std::size_t{});
  }
  _validate();
}

void MPSContainer::_validate() const {
  if (!_orbitals) {
    throw std::invalid_argument(
        "MPS requires nonempty sites and a mode basis.");
  }
  validate_sites(_sites);
  const ConfigurationSet mode_space(std::vector<Configuration>{}, _orbitals,
                                    Wavefunction::DEFAULT_SECTOR);
  if (mode_space.num_modes() != num_sites()) {
    throw std::invalid_argument("MPS must contain one site per active mode.");
  }
  if (_orthogonality_center && *_orthogonality_center >= num_sites()) {
    throw std::invalid_argument(
        "MPS orthogonality center must be a site index.");
  }
  if (_site_to_orbital_order.size() != num_sites()) {
    throw std::invalid_argument("MPS site order size must match its sites.");
  }
  auto sorted_order = _site_to_orbital_order;
  std::ranges::sort(sorted_order);
  for (std::size_t index = 0; index < num_sites(); ++index) {
    if (sorted_order[index] != index) {
      throw std::invalid_argument(
          "MPS site order must be a permutation of active-mode slots.");
    }
  }
}

void MPSContainer::validate_sites(const std::vector<SitePtr>& sites) {
  if (sites.empty()) {
    throw std::invalid_argument("MPS requires nonempty sites.");
  }
  for (const auto& site : sites) {
    if (!site) {
      throw std::invalid_argument("MPS site pointers must not be null.");
    }
    if (site->is_complex() != sites.front()->is_complex()) {
      throw std::invalid_argument("MPS sites must use one scalar type.");
    }
  }
  if (sites.front()->left_bond_dimension() != 1 ||
      sites.back()->right_bond_dimension() != 1) {
    throw std::invalid_argument("MPS outer bond dimensions must be one.");
  }
  for (std::size_t i = 0; i + 1 < sites.size(); ++i) {
    std::visit(
        [&](const auto& left, const auto& right) {
          if (*left.symmetries()[2] != *right.symmetries()[0] ||
              left.extents()[2] != right.extents()[0] ||
              sites[i]->right_sector_order() !=
                  sites[i + 1]->left_sector_order()) {
            throw std::invalid_argument(
                "Adjacent MPS sites have incompatible bond spaces.");
          }
        },
        sites[i]->tensor(), sites[i + 1]->tensor());
  }
}

std::size_t MPSContainer::max_bond_dimension() const {
  std::size_t maximum = 0;
  for (const auto& site : _sites) {
    // Every left bond is the right bond of the previous site. Edge sites have
    // bond dimension one.
    maximum = std::max(maximum, site->right_bond_dimension());
  }
  return maximum;
}

bool MPSContainer::is_complex() const { return _sites.front()->is_complex(); }

bool MPSContainer::_is_restricted_closed_shell() const { return false; }

std::shared_ptr<const MPSContainer::ParticleCount>
MPSContainer::total_num_particles() const {
  if (!_total_num_particles) {
    throw std::runtime_error("Total particle-count is not set.");
  }
  return _total_num_particles;
}

std::shared_ptr<const MPSContainer::ParticleCount>
MPSContainer::active_num_particles() const {
  if (!_active_num_particles) {
    throw std::runtime_error("Active particle-count is not set.");
  }
  return _active_num_particles;
}

MPSContainer::ScalarVariant MPSContainer::overlap(
    const WavefunctionContainer&) const {
  throw std::runtime_error(
      "overlap() is not implemented for MPS wavefunctions.");
}

double MPSContainer::norm() const {
  throw std::runtime_error("norm() is not implemented for MPS wavefunctions.");
}

std::shared_ptr<const SymmetryBlockedTensor<1>>
MPSContainer::total_orbital_occupations() const {
  throw std::runtime_error(
      "Orbital occupations are not implemented for MPS wavefunctions.");
}

std::shared_ptr<const SymmetryBlockedTensor<1>>
MPSContainer::active_orbital_occupations() const {
  throw std::runtime_error(
      "Orbital occupations are not implemented for MPS wavefunctions.");
}

void MPSContainer::clear_caches() const {}
std::string MPSContainer::get_container_type() const { return "mps"; }

std::vector<std::string> MPSContainer::sectors() const {
  return {Wavefunction::DEFAULT_SECTOR};
}

std::shared_ptr<const Orbitals> MPSContainer::sector_basis(
    const std::string& name) const {
  if (name != Wavefunction::DEFAULT_SECTOR) {
    throw std::out_of_range("Unknown MPS wavefunction sector: " + name);
  }
  return _orbitals;
}

nlohmann::json MPSContainer::_metadata_to_json() const {
  nlohmann::json json;
  json["version"] = SERIALIZATION_VERSION;
  json["container_type"] = get_container_type();
  json["scalar"] = is_complex() ? "complex" : "real";
  json["total_num_particles"] = _total_num_particles
                                    ? _total_num_particles->to_json()
                                    : nlohmann::json(nullptr);
  json["active_num_particles"] = _active_num_particles
                                     ? _active_num_particles->to_json()
                                     : nlohmann::json(nullptr);
  json["orthogonality_center"] = _orthogonality_center
                                     ? nlohmann::json(*_orthogonality_center)
                                     : nlohmann::json(nullptr);
  json["site_to_orbital_order"] = _site_to_orbital_order;
  json["sites"] = nlohmann::json::array();
  for (const auto& site : _sites) {
    json["sites"].push_back(site_metadata(*site));
  }
  return json;
}

nlohmann::json MPSContainer::to_json() const {
  auto json = _metadata_to_json();
  json["orbitals"] = _orbitals->to_json();
  _serialize_rdms_to_json(json);
  for (std::size_t i = 0; i < num_sites(); ++i) {
    json["sites"][i]["tensor"] =
        std::visit([](const auto& tensor) { return tensor.to_json(); },
                   _sites[i]->tensor());
  }
  return json;
}

std::unique_ptr<MPSContainer> MPSContainer::from_json(
    const nlohmann::json& json) {
  try {
    validate_metadata(json, SERIALIZATION_VERSION);
    std::vector<SitePtr> sites;
    for (const auto& site : json.at("sites")) {
      const auto& tensor = site.at("tensor");
      if (tensor.at("rank") != 3 || tensor.at("scalar") != json.at("scalar")) {
        throw std::invalid_argument(
            "MPS tensor rank or scalar type disagrees with its metadata.");
      }
      MPSSite::TensorPtr payload;
      if (json.at("scalar") == "complex") {
        payload = std::make_shared<const MPSSite::TensorVariant>(
            *SymmetryBlockedTensor<3, std::complex<double>>::from_json(tensor));
      } else {
        payload = std::make_shared<const MPSSite::TensorVariant>(
            *SymmetryBlockedTensor<3>::from_json(tensor));
      }
      sites.push_back(site_from_metadata(site, std::move(payload)));
    }
    auto result = mps_from_metadata(
        json, Orbitals::from_json(json.at("orbitals")), std::move(sites));
    result->_deserialize_rdms_from_json(json);
    return result;
  } catch (const nlohmann::json::exception& error) {
    throw std::runtime_error("Failed to parse MPSContainer from JSON: " +
                             std::string(error.what()));
  }
}

void MPSContainer::to_hdf5(H5::Group& group) const {
  try {
    const H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    const H5::DataSpace scalar_space(H5S_SCALAR);
    const std::string version = SERIALIZATION_VERSION;
    group.createAttribute("version", string_type, scalar_space)
        .write(string_type, version);
    group.createAttribute("container_type", string_type, scalar_space)
        .write(string_type, get_container_type());
    group.createDataSet("mps_metadata", string_type, scalar_space)
        .write(_metadata_to_json().dump(), string_type);
    auto orbitals = group.createGroup("orbitals");
    _orbitals->to_hdf5(orbitals);
    auto sites = group.createGroup("sites");
    for (std::size_t i = 0; i < num_sites(); ++i) {
      auto site = sites.createGroup(std::to_string(i));
      auto tensor = site.createGroup("tensor");
      std::visit([&](const auto& value) { value.to_hdf5(tensor); },
                 _sites[i]->tensor());
    }
    _serialize_rdms_to_hdf5(group);
  } catch (const H5::Exception& error) {
    throw std::runtime_error("Failed to write MPS HDF5: " +
                             error.getDetailMsg());
  }
}

std::unique_ptr<MPSContainer> MPSContainer::from_hdf5(H5::Group& group) {
  try {
    const H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    std::string version, type, payload;
    group.openAttribute("version").read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);
    group.openAttribute("container_type").read(string_type, type);
    if (type != "mps") {
      throw std::invalid_argument("HDF5 group does not contain an MPS.");
    }
    group.openDataSet("mps_metadata").read(payload, string_type);
    const auto json = nlohmann::json::parse(payload);
    validate_metadata(json, SERIALIZATION_VERSION);
    auto orbitals = group.openGroup("orbitals");
    auto site_group = group.openGroup("sites");
    std::vector<SitePtr> sites;
    for (std::size_t i = 0; i < json.at("sites").size(); ++i) {
      auto site = site_group.openGroup(std::to_string(i));
      auto tensor = site.openGroup("tensor");
      MPSSite::TensorPtr value;
      if (json.at("scalar") == "complex") {
        value = std::make_shared<const MPSSite::TensorVariant>(
            *SymmetryBlockedTensor<3, std::complex<double>>::from_hdf5(tensor));
      } else {
        value = std::make_shared<const MPSSite::TensorVariant>(
            *SymmetryBlockedTensor<3>::from_hdf5(tensor));
      }
      sites.push_back(
          site_from_metadata(json.at("sites").at(i), std::move(value)));
    }
    auto result = mps_from_metadata(json, Orbitals::from_hdf5(orbitals),
                                    std::move(sites));
    result->_deserialize_rdms_from_hdf5(group);
    return result;
  } catch (const H5::Exception& error) {
    throw std::runtime_error("Failed to read MPS HDF5: " +
                             error.getDetailMsg());
  } catch (const nlohmann::json::exception& error) {
    throw std::runtime_error("Invalid MPS HDF5 metadata: " +
                             std::string(error.what()));
  }
}

void MPSContainer::hash_update(qdk::chemistry::utils::HashContext& ctx) const {
  WavefunctionContainer::hash_update(ctx);
  hash_value(ctx, get_container_type());
  hash_value(ctx, _orbitals->content_hash());
  hash_value(ctx, _total_num_particles);
  hash_value(ctx, _active_num_particles);
  hash_value(ctx, _orthogonality_center);
  hash_value(ctx, _site_to_orbital_order);
  hash_value(ctx, static_cast<std::uint64_t>(_sites.size()));
  for (const auto& site : _sites) {
    hash_value(ctx, site->tensor());
    for (const auto& order : site->sector_orders()) {
      hash_value(ctx, order);
    }
    hash_value(ctx, site->physical_basis());
  }
}

std::unique_ptr<WavefunctionContainer> MPSContainer::clone() const {
  return std::make_unique<MPSContainer>(*this);
}

}  // namespace qdk::chemistry::data

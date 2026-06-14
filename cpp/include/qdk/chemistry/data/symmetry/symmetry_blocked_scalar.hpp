// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <H5Cpp.h>

#include <algorithm>
#include <complex>
#include <cstddef>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked.hpp>
#include <qdk/chemistry/utils/scalar_traits.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Symmetry-blocked scalar quantity.
 *
 * A @ref SymmetryBlockedScalar stores one scalar value per symmetry sector of a
 * single-slot (rank-1) partition: a sparse map from a per-slot
 * @ref SymmetryLabel to a scalar @p Scalar. The slot carries its own
 * @ref SymmetryProduct. It is the scalar analogue of @ref SymmetryBlockedTensor
 * and is used for any symmetry-resolved scalar quantity whose block structure
 * is induced by the single-particle basis.
 *
 * Example: when the slot carries a spin axis, the stored labels can be the
 * spin labels (@f$\alpha@f$, @f$\beta@f$) and the value under each label is
 * that channel's quantity. When the slot carries no axes (trivial
 * @ref SymmetryProduct), a single block keyed by the trivial label holds the
 * aggregate quantity.
 *
 * Blocks are held via @c shared_ptr<const Scalar> so that symmetry-equivalent
 * sectors can alias the same storage when the axis is marked
 * @ref SymmetryAxis::equivalent. Scalar quantities whose sectors are
 * independent should use non-equivalent axes so that no aliasing is applied
 * (for example, open-shell @f$\alpha@f$ and @f$\beta@f$ electron counts).
 *
 * @tparam Scalar The per-block scalar type (defaults to @c std::size_t for
 *                count-like quantities).
 */
template <class Scalar = std::size_t>
class SymmetryBlockedScalar : public SymmetryBlocked<1, Scalar> {
  using Base = SymmetryBlocked<1, Scalar>;

 public:
  /**
   * @brief Single-slot label tuple: one @ref SymmetryLabel.
   *
   * The key type of @ref BlockMap. Equivalent to the rank-1 @c Labels of the
   * @ref SymmetryBlocked base.
   */
  using Labels = std::array<SymmetryLabel, 1>;
  /**
   * @brief Shared pointer to immutable per-block scalar storage.
   *
   * Held as @c shared_ptr<const Scalar> so that symmetry-equivalent sectors
   * can alias the same storage.
   */
  using BlockPtr = std::shared_ptr<const Scalar>;
  /**
   * @brief Sparse map from a single-slot label tuple to scalar storage.
   *
   * Aliased sectors map to the same @ref BlockPtr; keys are hashed via
   * @ref LabelsHash.
   */
  using BlockMap = std::unordered_map<Labels, BlockPtr, LabelsHash<1>>;
  /**
   * @brief Per-slot symmetry definition.
   *
   * One @ref SymmetryProduct for the single index slot, supplied at
   * construction.
   */
  using SymmetriesArray = std::array<std::shared_ptr<const SymmetryProduct>, 1>;

  /**
   * @brief Construct from the per-slot symmetry and a block map. See the class
   * description for the validation rules.
   *
   * A scalar block is a single number, so extents are meaningless and are not
   * part of this constructor: the base-class extents (every label present in
   * @p blocks mapped to @c 1) are computed automatically.
   *
   * @param symmetries Single-slot @ref SymmetryProduct definition.
   * @param blocks Scalar storage keyed by the per-slot label.
   *
   * @throws std::invalid_argument if a block label is not admissible under the
   *         slot's @ref SymmetryProduct, if a block pointer is null, or if
   *         restricted orbit partners are supplied with distinct backing
   *         storage.
   */
  SymmetryBlockedScalar(SymmetriesArray symmetries, BlockMap blocks)
      : Base(std::move(symmetries), _unit_extents(blocks), blocks) {}

  /**
   * @brief The scalar value stored for @p label.
   *
   * Convenience wrapper around @ref SymmetryBlocked::block for the single-slot
   * case. When the container holds a single block, the trivial (empty) label
   * also resolves to it.
   *
   * @param label The symmetry label identifying the block.
   * @return The stored scalar value.
   * @throws std::invalid_argument if no block exists for @p label.
   */
  Scalar value(const SymmetryLabel& label) const {
    return this->block(Labels{label});
  }

  // ---- DataClass interface ------------------------------------------------

  /**
   * @brief @ref DataClass type identifier.
   * @return The stable string @c "symmetry_blocked_scalar".
   */
  std::string get_data_type_name() const override {
    return "symmetry_blocked_scalar";
  }

  /**
   * @brief Single-line summary including scalar type and per-block label/value
   * pairs.
   * @return A short diagnostic string suitable for logging.
   */
  std::string get_summary() const override {
    auto label_key = [](const SymmetryLabel& label) -> std::string {
      if (label.empty()) return "";
      const auto& vals = label.values();
      if (vals.size() == 1 && vals.begin()->first == AxisName::Spin) {
        auto* sv = dynamic_cast<const SpinValue*>(vals.begin()->second.get());
        if (sv) {
          if (sv->value() == 1) return "alpha";
          if (sv->value() == -1) return "beta";
        }
      }
      return label.to_json().dump();
    };

    std::vector<std::pair<std::string, Scalar>> entries;
    entries.reserve(this->_blocks.size());
    for (const auto& [labels, value_ptr] : this->_blocks) {
      entries.emplace_back(label_key(labels[0]), *value_ptr);
    }
    std::sort(
        entries.begin(), entries.end(),
        [](const auto& lhs, const auto& rhs) { return lhs.first < rhs.first; });

    std::ostringstream oss;
    oss << "SymmetryBlockedScalar(scalar=" << _scalar_tag() << ", {";
    for (std::size_t i = 0; i < entries.size(); ++i) {
      if (i > 0) {
        oss << ", ";
      }
      oss << entries[i].first << ": " << entries[i].second;
    }
    oss << "})";
    return oss.str();
  }

  /**
   * @brief Serialize this scalar container to JSON.
   *
   * @return JSON object carrying the serialization version, scalar type,
   *         per-slot symmetries and extents, and the per-block payload.
   */
  nlohmann::json to_json() const override {
    nlohmann::json j;
    j["version"] = SERIALIZATION_VERSION;
    j["type"] = "SymmetryBlockedScalar";
    j["scalar"] = _scalar_tag();
    j["symmetries"] = this->_symmetries_to_json();
    j["extents"] = this->_extents_to_json();

    nlohmann::json blocks = nlohmann::json::array();
    for (const auto& group : this->_group_by_pointer()) {
      nlohmann::json keys = nlohmann::json::array();
      for (const auto& key : group.keys) {
        nlohmann::json key_json = nlohmann::json::array();
        for (const auto& label : key) {
          key_json.push_back(label.to_json());
        }
        keys.push_back(std::move(key_json));
      }
      blocks.push_back(nlohmann::json{{"keys", std::move(keys)},
                                      {"value", _value_to_json(*group.ptr)}});
    }
    j["blocks"] = std::move(blocks);
    return j;
  }

  /**
   * @brief Serialize this scalar container to a JSON file.
   * @param filename Path to the JSON file to create or overwrite.
   * @throws std::runtime_error if the file cannot be opened for writing.
   */
  void to_json_file(const std::string& filename) const override {
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    out << to_json().dump(2);
  }

  /**
   * @brief Serialize this scalar container into an HDF5 group.
   *
   * Writes the JSON form (see @ref to_json) as a single string dataset within
   * @p group, mirroring the HDF5 layout used by @ref SymmetryProduct.
   *
   * @param group HDF5 group to write into.
   * @throws std::runtime_error on HDF5 I/O failure.
   */
  void to_hdf5(H5::Group& group) const override {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::DataSpace scalar_space(H5S_SCALAR);
    auto dataset = group.createDataSet("symmetry_blocked_scalar_metadata",
                                       str_type, scalar_space);
    std::string payload = to_json().dump(2);
    dataset.write(payload, str_type);
  }

  void to_hdf5_file(const std::string& filename) const override {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    to_hdf5(file);
  }

  /**
   * @brief Dispatch to JSON or HDF5 serialization based on @p type.
   * @param filename Target file path.
   * @param type Either @c "json" or @c "hdf5".
   * @throws std::invalid_argument if @p type is not @c "json" or @c "hdf5".
   * @throws std::runtime_error if the underlying I/O operation fails.
   */
  void to_file(const std::string& filename,
               const std::string& type) const override {
    if (type == "json") {
      to_json_file(filename);
    } else if (type == "hdf5") {
      to_hdf5_file(filename);
    } else {
      throw std::invalid_argument("Unsupported file type: " + type +
                                  ". Supported types are: json, hdf5");
    }
  }

  /**
   * @brief Reconstruct from a JSON object produced by @ref to_json.
   *
   * Validates the serialization version recorded in @p j against
   * @c SERIALIZATION_VERSION before reconstructing. Any @c "extents" field is
   * accepted but ignored: scalar extents are always @c 1 and are recomputed
   * from the blocks.
   *
   * @param j JSON object produced by a prior @ref to_json call.
   * @return Shared pointer to the reconstructed scalar container.
   * @throws std::runtime_error if @p j is missing the @c "version" field or
   *         its version is incompatible with @c SERIALIZATION_VERSION.
   * @throws nlohmann::json::exception if @p j is otherwise malformed.
   */
  static std::shared_ptr<SymmetryBlockedScalar> from_json(
      const nlohmann::json& j) {
    if (!j.contains("version")) {
      throw std::runtime_error(
          "SymmetryBlockedScalar JSON missing required 'version' field.");
    }
    _validate_version(j.at("version").template get<std::string>());

    const std::string expected_scalar = _scalar_tag();
    if (j.contains("scalar") &&
        j.at("scalar").template get<std::string>() != expected_scalar) {
      throw std::invalid_argument(
          "SymmetryBlockedScalar JSON scalar type does not match the requested "
          "type.");
    }

    auto symmetries = Base::_symmetries_from_json(j);

    BlockMap blocks;
    for (const auto& entry : j.at("blocks")) {
      auto value =
          std::make_shared<const Scalar>(_value_from_json(entry.at("value")));
      for (const auto& key_json : entry.at("keys")) {
        std::vector<SymmetryLabel> labels;
        for (const auto& label_json : key_json) {
          labels.push_back(SymmetryLabel::from_json(label_json));
        }
        blocks.emplace(detail::make_labels<1>(labels), value);
      }
    }
    return std::make_shared<SymmetryBlockedScalar>(std::move(symmetries),
                                                   std::move(blocks));
  }

  /**
   * @brief Reconstruct a @ref SymmetryBlockedScalar from a JSON file produced
   * by @ref to_json_file.
   *
   * @param filename Path to the JSON file to read.
   * @return Shared pointer to the reconstructed scalar container.
   * @throws std::runtime_error if the file cannot be opened, parsed, or
   *         carries an incompatible serialization version.
   */
  static std::shared_ptr<SymmetryBlockedScalar> from_json_file(
      const std::string& filename) {
    std::ifstream in(filename);
    if (!in) {
      throw std::runtime_error("Failed to open file for reading: " + filename);
    }
    nlohmann::json j;
    in >> j;
    return from_json(j);
  }

  /**
   * @brief Reconstruct from an HDF5 group produced by @ref to_hdf5.
   *
   * Validates the @c "version" field in the HDF5 metadata payload against
   * @c SERIALIZATION_VERSION before reconstructing.
   *
   * @param group HDF5 group to read from.
   * @return Shared pointer to the reconstructed scalar container.
   * @throws std::runtime_error if the metadata dataset or its @c "version"
   *         field is missing or the version is incompatible, or on HDF5 I/O
   *         failure.
   * @throws std::invalid_argument if the encoded scalar type does not match
   *         the requested instantiation.
   */
  static std::shared_ptr<SymmetryBlockedScalar> from_hdf5(H5::Group& group) {
    if (!group.nameExists("symmetry_blocked_scalar_metadata")) {
      throw std::runtime_error(
          "SymmetryBlockedScalar HDF5 metadata dataset not found.");
    }
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);
    auto dataset = group.openDataSet("symmetry_blocked_scalar_metadata");
    std::string payload;
    dataset.read(payload, str_type);
    return from_json(nlohmann::json::parse(payload));
  }

  static std::shared_ptr<SymmetryBlockedScalar> from_hdf5_file(
      const std::string& filename) {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    return from_hdf5(file);
  }

  /**
   * @brief Dispatch to JSON or HDF5 deserialization based on @p type.
   * @param filename Source file path.
   * @param type Either @c "json" or @c "hdf5".
   * @return Shared pointer to the reconstructed scalar container.
   * @throws std::invalid_argument if @p type is not @c "json" or @c "hdf5".
   * @throws std::runtime_error if the underlying I/O operation fails or the
   *         serialization version is incompatible.
   */
  static std::shared_ptr<SymmetryBlockedScalar> from_file(
      const std::string& filename, const std::string& type) {
    if (type == "json") {
      return from_json_file(filename);
    } else if (type == "hdf5") {
      return from_hdf5_file(filename);
    }
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5");
  }

 protected:
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override {
    hash_value(ctx, this->get_data_type_name());
    this->_hash_symmetry_blocked_metadata(ctx);
    auto groups = this->_sorted_pointer_groups();
    hash_value(ctx, static_cast<uint64_t>(groups.size()));
    for (const auto& group : groups) {
      hash_value(ctx, static_cast<uint64_t>(group.keys.size()));
      for (const auto& key : group.keys) {
        this->_hash_labels(ctx, key);
      }
      hash_value(ctx, *group.ptr);
    }
  }

 private:
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  static typename Base::ExtentsArray _unit_extents(const BlockMap& blocks) {
    typename Base::ExtentsArray extents;
    for (const auto& [labels, value_ptr] : blocks) {
      extents[0].emplace(labels[0], std::size_t{1});
    }
    return extents;
  }

  static void _validate_version(const std::string& found) {
    if (found == SERIALIZATION_VERSION) {
      return;
    }
    auto major_minor = [](const std::string& version) {
      const std::size_t first = version.find('.');
      const std::size_t second = version.find('.', first + 1);
      if (first == std::string::npos || second == std::string::npos) {
        throw std::runtime_error(
            "SymmetryBlockedScalar serialization version is malformed: " +
            version);
      }
      return std::pair<int, int>{
          std::stoi(version.substr(0, first)),
          std::stoi(version.substr(first + 1, second - first - 1))};
    };
    if (major_minor(found) != major_minor(SERIALIZATION_VERSION)) {
      throw std::runtime_error(
          "SymmetryBlockedScalar serialization version mismatch. Expected: " +
          std::string(SERIALIZATION_VERSION) + ", found: " + found + ".");
    }
  }

  /// Stable string tag identifying the scalar type in serialized payloads.
  static std::string _scalar_tag() {
    if constexpr (std::is_same_v<Scalar, std::size_t>) {
      return "uint";
    } else if constexpr (utils::is_complex_scalar_v<Scalar>) {
      return "complex";
    } else {
      return "real";
    }
  }

  static nlohmann::json _value_to_json(const Scalar& value) {
    if constexpr (utils::is_complex_scalar_v<Scalar>) {
      return nlohmann::json::array({value.real(), value.imag()});
    } else {
      return value;
    }
  }

  static Scalar _value_from_json(const nlohmann::json& j) {
    if constexpr (utils::is_complex_scalar_v<Scalar>) {
      using Real = typename Scalar::value_type;
      return Scalar(j.at(0).get<Real>(), j.at(1).get<Real>());
    } else {
      return j.get<Scalar>();
    }
  }
};

}  // namespace qdk::chemistry::data

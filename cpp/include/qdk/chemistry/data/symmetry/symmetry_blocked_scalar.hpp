// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

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
 * A @ref SymmetryBlockedScalar stores one scalar value per symmetry sector of
 * a single-slot (rank-1) partition: a sparse map from a per-slot
 * @ref SymmetryLabel to a scalar @p Scalar. The slot carries its own
 * @ref SymmetryProduct. It is the scalar analogue of @ref SymmetryBlockedTensor
 * and is used for per-symmetry counts and other scalar quantities (e.g. the
 * number of electrons per spin channel) whose block structure is induced by
 * the single-particle basis rather than assumed.
 *
 * When the slot carries a spin axis, the stored labels are the spin labels
 * (@f$\alpha@f$, @f$\beta@f$) and the value under each label is that channel's
 * quantity. When the slot carries no axes (trivial @ref SymmetryProduct), a
 * single block keyed by the trivial label holds the aggregate quantity.
 *
 * Blocks are held via @c shared_ptr<const Scalar> so that symmetry-equivalent
 * sectors can alias the same storage when the axis is marked
 * @ref SymmetryAxis::equivalent. Scalar quantities whose channels are
 * independent (e.g. electron counts, which differ for open-shell references)
 * should use a non-equivalent axis so that no aliasing is applied.
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
   * @brief Per-slot per-label extents.
   *
   * For a scalar quantity the extent of every label is conventionally @c 1.
   */
  using ExtentsArray =
      std::array<std::unordered_map<SymmetryLabel, std::size_t>, 1>;

  /**
   * @brief Construct from the per-slot symmetry, per-slot extents, and a block
   * map. See the class description for the validation rules.
   *
   * @param symmetries Single-slot @ref SymmetryProduct definition.
   * @param extents Single-slot per-label extents (conventionally @c 1 per
   *                label).
   * @param blocks Scalar storage keyed by the per-slot label.
   *
   * @throws std::invalid_argument if a block or extent label is not
   *         admissible under the slot's @ref SymmetryProduct, if a block
   *         pointer is null, or if restricted orbit partners are supplied
   *         with distinct backing storage.
   */
  SymmetryBlockedScalar(SymmetriesArray symmetries, ExtentsArray extents,
                        BlockMap blocks)
      : Base(std::move(symmetries), std::move(extents), std::move(blocks)) {}

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
   * @brief Single-line summary including scalar type, number of stored
   * blocks, and number of independent (non-aliased) blocks.
   * @return A short diagnostic string suitable for logging.
   */
  std::string get_summary() const override {
    std::ostringstream oss;
    oss << "SymmetryBlockedScalar(scalar=" << _scalar_tag()
        << ", blocks=" << this->num_blocks()
        << ", independent=" << this->_group_by_pointer().size() << ")";
    return oss.str();
  }

  /**
   * @brief Serialize this scalar container to JSON, with one entry per group
   * of pointer-equivalent blocks (the aliased keys and the scalar value).
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
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Serialize this scalar container to an HDF5 file.
   * @param filename Path to the HDF5 file to create or overwrite.
   * @throws std::runtime_error on HDF5 I/O failure.
   */
  void to_hdf5_file(const std::string& filename) const override;

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
   * @c SERIALIZATION_VERSION before reconstructing.
   *
   * @param j JSON object produced by a prior @ref to_json call.
   * @return Shared pointer to the reconstructed scalar container.
   * @throws std::runtime_error if @p j is missing the @c "version" field or
   *         its version is incompatible with @c SERIALIZATION_VERSION.
   * @throws nlohmann::json::exception if @p j is otherwise malformed.
   */
  static std::shared_ptr<SymmetryBlockedScalar> from_json(
      const nlohmann::json& j);

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
  static std::shared_ptr<SymmetryBlockedScalar> from_hdf5(H5::Group& group);

  /**
   * @brief Reconstruct from an HDF5 file produced by @ref to_hdf5_file.
   * @param filename Path to the HDF5 file to read.
   * @return Shared pointer to the reconstructed scalar container.
   * @throws std::runtime_error if the file cannot be opened or carries an
   *         incompatible serialization version.
   */
  static std::shared_ptr<SymmetryBlockedScalar> from_hdf5_file(
      const std::string& filename);

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
  /// On-disk serialization format version. Bump on any change to the JSON or
  /// HDF5 shape produced by @ref to_json / @ref to_hdf5.
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

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

/**
 * @brief Build a rank-1 spin-blocked @ref SymmetryBlockedScalar carrying an
 * independent value per spin channel.
 *
 * The spin axis is constructed non-equivalent so that the two channels are
 * stored independently (correct for quantities such as electron counts that
 * differ between @f$\alpha@f$ and @f$\beta@f$ for open-shell references).
 *
 * @tparam Scalar Per-block scalar type.
 * @param alpha_value Value carried by the @f$\alpha@f$ channel.
 * @param beta_value Value carried by the @f$\beta@f$ channel.
 * @return Constructed spin-blocked scalar container.
 */
template <class Scalar>
SymmetryBlockedScalar<Scalar> make_spin_blocked_scalar(Scalar alpha_value,
                                                       Scalar beta_value) {
  using SBS = SymmetryBlockedScalar<Scalar>;
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, /*equivalent=*/false)}));
  std::unordered_map<SymmetryLabel, std::size_t> ext;
  ext[axes::alpha()] = 1;
  ext[axes::beta()] = 1;
  typename SBS::BlockMap blocks;
  blocks[{axes::alpha()}] = std::make_shared<const Scalar>(alpha_value);
  blocks[{axes::beta()}] = std::make_shared<const Scalar>(beta_value);
  return SBS(typename SBS::SymmetriesArray{sym},
             typename SBS::ExtentsArray{ext}, std::move(blocks));
}

/**
 * @brief Build a rank-1 @ref SymmetryBlockedScalar carrying a single aggregate
 * value under the trivial (axis-free) symmetry.
 *
 * Used when the single-particle basis carries no spin axis, so the quantity
 * is not resolved per spin channel.
 *
 * @tparam Scalar Per-block scalar type.
 * @param value The aggregate value carried by the single trivial block.
 * @return Constructed trivial scalar container.
 */
template <class Scalar>
SymmetryBlockedScalar<Scalar> make_trivial_blocked_scalar(Scalar value) {
  using SBS = SymmetryBlockedScalar<Scalar>;
  auto sym =
      std::make_shared<const SymmetryProduct>(SymmetryProduct::trivial());
  typename SBS::BlockMap blocks;
  blocks[{SymmetryLabel{}}] = std::make_shared<const Scalar>(value);
  return SBS(typename SBS::SymmetriesArray{sym}, typename SBS::ExtentsArray{},
             std::move(blocks));
}

// Explicit instantiation declarations (definitions emitted in the .cpp).
extern template class SymmetryBlockedScalar<std::size_t>;

}  // namespace qdk::chemistry::data

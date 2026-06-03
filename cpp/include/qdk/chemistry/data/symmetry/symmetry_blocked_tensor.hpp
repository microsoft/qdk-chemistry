// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
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
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Maps a tensor rank to the Eigen storage type used for one block.
 *
 * Rank-1, rank-3, and rank-4 blocks are stored as (flat) column vectors;
 * rank-2 blocks are stored as dense matrices. Partial specializations are
 * provided for ranks 1–4; higher ranks have no specialization and will
 * produce a compilation error.
 */
template <std::size_t Rank, class Scalar>
struct TensorType;

/** @brief Rank-1 block storage: a dense column vector. */
template <class S>
struct TensorType<1, S> {
  using type = Eigen::Matrix<S, Eigen::Dynamic, 1>;
};
/** @brief Rank-2 block storage: a dense matrix. */
template <class S>
struct TensorType<2, S> {
  using type = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
};
/** @brief Rank-3 block storage: a flat-packed column vector. */
template <class S>
struct TensorType<3, S> {
  using type = Eigen::Matrix<S, Eigen::Dynamic, 1>;
};
/** @brief Rank-4 block storage: a flat-packed column vector. */
template <class S>
struct TensorType<4, S> {
  using type = Eigen::Matrix<S, Eigen::Dynamic, 1>;
};

/** @brief Storage type for one block of a rank-@p Rank tensor over @p Scalar.
 */
template <std::size_t Rank, class Scalar = double>
using Tensor = typename TensorType<Rank, Scalar>::type;

namespace detail {

template <class S>
nlohmann::json scalar_to_json(const S& value) {
  if constexpr (utils::is_complex_scalar_v<S>) {
    return nlohmann::json::array({value.real(), value.imag()});
  } else {
    return value;
  }
}

template <class S>
S scalar_from_json(const nlohmann::json& j) {
  if constexpr (utils::is_complex_scalar_v<S>) {
    using Real = typename S::value_type;
    return S(j.at(0).get<Real>(), j.at(1).get<Real>());
  } else {
    return j.get<S>();
  }
}

}  // namespace detail

/**
 * @brief Symmetry-blocked dense tensor.
 *
 * A @ref SymmetryBlockedTensor stores the non-zero symmetry sectors of a
 * rank-@p Rank tensor as a sparse map from per-slot @ref SymmetryLabel arrays
 * to dense Eigen blocks. Each slot carries its own @ref SymmetryProduct
 * and a per-label extent. Blocks are held via @c shared_ptr<const Tensor> so
 * that symmetry-equivalent sectors can alias the same storage.
 *
 * Symmetry aliasing is defined on the spin axis: a simultaneous
 * @f$\alpha \leftrightarrow \beta@f$ swap across all slots. When every slot
 * shares the same @ref SymmetryProduct instance and the spin axis is marked
 * @c equivalent, the constructor auto-aliases each partner block to the
 * supplied representative. When the slots carry distinct @ref SymmetryProduct
 * (intertwiner storage such as basis coefficients), no auto-aliasing is
 * performed.
 *
 * The full block map is supplied at construction.
 *
 * @tparam Rank   Tensor rank (1, 2, 3, or 4 are instantiated).
 * @tparam Scalar Element type (@c double or @c std::complex<double>).
 */
template <std::size_t Rank, class Scalar = double>
class SymmetryBlockedTensor
    : public SymmetryBlocked<Rank, Tensor<Rank, Scalar>> {
  using Base = SymmetryBlocked<Rank, Tensor<Rank, Scalar>>;

 public:
  /**
   * @brief Sparse map from per-slot label tuples to dense block storage.
   *
   * Inherited from @ref SymmetryBlocked. Aliased sectors map to the same
   * @ref BlockPtr; keys are hashed via @ref LabelsHash.
   */
  using typename Base::BlockMap;
  /**
   * @brief Shared pointer to immutable per-block dense storage.
   *
   * Inherited from @ref SymmetryBlocked. Held as @c shared_ptr<const Block>
   * so that symmetry-equivalent sectors can alias the same storage.
   */
  using typename Base::BlockPtr;
  /**
   * @brief Per-slot per-label extents.
   *
   * Inherited from @ref SymmetryBlocked. For each index slot, maps every
   * admissible @ref SymmetryLabel to its universe size.
   */
  using typename Base::ExtentsArray;
  /**
   * @brief Per-slot block label tuple: one @ref SymmetryLabel per index slot.
   *
   * Inherited from @ref SymmetryBlocked. Used as the key type of
   * @ref BlockMap.
   */
  using typename Base::Labels;
  /**
   * @brief Per-slot symmetry definitions.
   *
   * Inherited from @ref SymmetryBlocked. One @ref SymmetryProduct per index
   * slot, supplied at construction.
   */
  using typename Base::SymmetriesArray;

  /**
   * @brief Construct from per-slot symmetries, per-slot extents, and a block
   * map. See the class description for the validation rules.
   *
   * @param symmetries Per-slot @ref SymmetryProduct definitions.
   * @param extents Per-slot per-label universe sizes.
   * @param blocks Block storage keyed by per-slot label tuples; block
   *               shapes must match the declared extents per slot.
   *
   * @throws std::invalid_argument if a block or extent label is not
   *         admissible under the matching slot's @ref SymmetryProduct, if a
   *         block's shape does not match the declared extents, if restricted
   *         orbit partners have unequal extents, or if both orbit partners
   *         are supplied but do not share storage.
   */
  SymmetryBlockedTensor(SymmetriesArray symmetries, ExtentsArray extents,
                        BlockMap blocks)
      : Base(std::move(symmetries), std::move(extents), std::move(blocks)) {
    _validate_tensor_blocks();
  }

  // ---- DataClass interface ------------------------------------------------

  /**
   * @brief @ref DataClass type identifier.
   * @return The stable string @c "symmetry_blocked_tensor".
   */
  std::string get_data_type_name() const override {
    return "symmetry_blocked_tensor";
  }

  /**
   * @brief Single-line summary including rank, scalar type, number of
   * stored blocks, and number of independent (non-aliased) blocks.
   * @return A short diagnostic string suitable for logging.
   */
  std::string get_summary() const override {
    std::ostringstream oss;
    oss << "SymmetryBlockedTensor(rank=" << Rank << ", scalar="
        << (utils::is_complex_scalar_v<Scalar> ? "complex" : "real")
        << ", blocks=" << this->num_blocks()
        << ", independent=" << this->_group_by_pointer().size() << ")";
    return oss.str();
  }

  /**
   * @brief Serialize this tensor to JSON, with one entry per group of
   * pointer-equivalent blocks (a canonical key, the aliased keys, and the
   * block payload).
   *
   * @return JSON object carrying the serialization version, rank, scalar
   *         type, per-slot symmetries and extents, and the per-block
   *         payload.
   */
  nlohmann::json to_json() const override {
    nlohmann::json j;
    j["version"] = SERIALIZATION_VERSION;
    j["type"] = "SymmetryBlockedTensor";
    j["rank"] = Rank;
    j["scalar"] = utils::is_complex_scalar_v<Scalar> ? "complex" : "real";
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
                                      {"block", _block_to_json(*group.ptr)}});
    }
    j["blocks"] = std::move(blocks);
    return j;
  }

  /**
   * @brief Serialize this tensor to a JSON file.
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
   * @brief Serialize this tensor into an HDF5 group.
   *
   * Writes one numeric dataset per independent block plus a JSON metadata
   * payload (carrying the serialization version, per-slot symmetries and
   * extents, and the block keys).
   *
   * @param group HDF5 group to write into.
   * @throws std::runtime_error on HDF5 I/O failure.
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Serialize this tensor to an HDF5 file.
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
   * @ref SERIALIZATION_VERSION before reconstructing.
   *
   * @param j JSON object produced by a prior @ref to_json call.
   * @return Shared pointer to the reconstructed tensor.
   * @throws std::runtime_error if @p j is missing the @c "version" field or
   *         its version is incompatible with @ref SERIALIZATION_VERSION.
   * @throws nlohmann::json::exception if @p j is otherwise malformed.
   */
  static std::shared_ptr<SymmetryBlockedTensor> from_json(
      const nlohmann::json& j);

  /**
   * @brief Reconstruct a @ref SymmetryBlockedTensor from a JSON file
   * produced by @ref to_json_file.
   *
   * @param filename Path to the JSON file to read.
   * @return Shared pointer to the reconstructed tensor.
   * @throws std::runtime_error if the file cannot be opened, parsed, or
   *         carries an incompatible serialization version.
   */
  static std::shared_ptr<SymmetryBlockedTensor> from_json_file(
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
   * @ref SERIALIZATION_VERSION before reconstructing.
   *
   * @param group HDF5 group to read from.
   * @return Shared pointer to the reconstructed tensor.
   * @throws std::runtime_error if the metadata dataset or its @c "version"
   *         field is missing or the version is incompatible, or on HDF5
   *         I/O failure.
   * @throws std::invalid_argument if the encoded rank or scalar type does
   *         not match the requested instantiation.
   */
  static std::shared_ptr<SymmetryBlockedTensor> from_hdf5(H5::Group& group);
  /**
   * @brief Reconstruct from an HDF5 file produced by @ref to_hdf5_file.
   * @param filename Path to the HDF5 file to read.
   * @return Shared pointer to the reconstructed tensor.
   * @throws std::runtime_error if the file cannot be opened or carries an
   *         incompatible serialization version.
   */
  static std::shared_ptr<SymmetryBlockedTensor> from_hdf5_file(
      const std::string& filename);
  /**
   * @brief Dispatch to JSON or HDF5 deserialization based on @p type.
   * @param filename Source file path.
   * @param type Either @c "json" or @c "hdf5".
   * @return Shared pointer to the reconstructed tensor.
   * @throws std::invalid_argument if @p type is not @c "json" or @c "hdf5".
   * @throws std::runtime_error if the underlying I/O operation fails or
   *         the serialization version is incompatible.
   */
  static std::shared_ptr<SymmetryBlockedTensor> from_file(
      const std::string& filename, const std::string& type) {
    if (type == "json") {
      return from_json_file(filename);
    } else if (type == "hdf5") {
      return from_hdf5_file(filename);
    }
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5");
  }

 private:
  /// On-disk serialization format version. Bump on any change to the JSON or
  /// HDF5 shape produced by @ref to_json / @ref to_hdf5.
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  void _validate_tensor_blocks() const {
    for (const auto& [labels, ptr] : this->_blocks) {
      std::array<std::size_t, Rank> dims{};
      for (std::size_t i = 0; i < Rank; ++i) {
        dims[i] = this->_extent_for(i, labels[i]);
      }
      _validate_block_shape(dims, *ptr);
    }
  }

  static void _validate_block_shape(const std::array<std::size_t, Rank>& dims,
                                    const Tensor<Rank, Scalar>& block) {
    if constexpr (Rank == 1) {
      if (static_cast<std::size_t>(block.size()) != dims[0]) {
        throw std::invalid_argument(
            "SymmetryBlockedTensor rank-1 block size does not match extent.");
      }
    } else if constexpr (Rank == 2) {
      if (static_cast<std::size_t>(block.rows()) != dims[0] ||
          static_cast<std::size_t>(block.cols()) != dims[1]) {
        throw std::invalid_argument(
            "SymmetryBlockedTensor rank-2 block shape does not match extents.");
      }
    } else {
      // Rank-3 and rank-4 blocks are stored flat-packed; the exact length is
      // producer/symmetry-dependent (permutational packing), so only require a
      // non-empty single-column vector here.
      if (block.cols() != 1 || block.size() == 0) {
        throw std::invalid_argument("SymmetryBlockedTensor rank-" +
                                    std::to_string(Rank) +
                                    " block must be a non-empty flat vector.");
      }
    }
  }

  static nlohmann::json _block_to_json(const Tensor<Rank, Scalar>& block) {
    nlohmann::json j;
    j["rows"] = static_cast<std::size_t>(block.rows());
    j["cols"] = static_cast<std::size_t>(block.cols());
    if constexpr (utils::is_complex_scalar_v<Scalar>) {
      // Complex: serialize as array of [real, imag] pairs.
      nlohmann::json data = nlohmann::json::array();
      for (Eigen::Index r = 0; r < block.rows(); ++r) {
        for (Eigen::Index c = 0; c < block.cols(); ++c) {
          data.push_back(detail::scalar_to_json<Scalar>(block(r, c)));
        }
      }
      j["data"] = std::move(data);
    } else {
      // Real: bulk assign from contiguous storage.
      const auto* ptr = block.data();
      j["data"] = std::vector<Scalar>(ptr, ptr + block.size());
    }
    return j;
  }

  static Tensor<Rank, Scalar> _block_from_json(const nlohmann::json& j) {
    const auto rows = j.at("rows").get<Eigen::Index>();
    const auto cols = j.at("cols").get<Eigen::Index>();
    if (rows < 0 || cols < 0) {
      throw std::invalid_argument(
          "SymmetryBlockedTensor block has negative dimensions.");
    }
    if constexpr (utils::is_complex_scalar_v<Scalar>) {
      Tensor<Rank, Scalar> block(rows, cols);
      const auto& data = j.at("data");
      Eigen::Index k = 0;
      for (Eigen::Index r = 0; r < rows; ++r) {
        for (Eigen::Index c = 0; c < cols; ++c) {
          block(r, c) = detail::scalar_from_json<Scalar>(data.at(k++));
        }
      }
      return block;
    } else {
      auto vec = j.at("data").get<std::vector<Scalar>>();
      if (static_cast<Eigen::Index>(vec.size()) != rows * cols) {
        throw std::invalid_argument(
            "SymmetryBlockedTensor block data size does not match dimensions.");
      }
      return Eigen::Map<const Tensor<Rank, Scalar>>(vec.data(), rows, cols);
    }
  }
};

// Explicit instantiation declarations (definitions emitted in the .cpp).
extern template class SymmetryBlockedTensor<1, double>;
extern template class SymmetryBlockedTensor<2, double>;
extern template class SymmetryBlockedTensor<3, double>;
extern template class SymmetryBlockedTensor<4, double>;
extern template class SymmetryBlockedTensor<1, std::complex<double>>;
extern template class SymmetryBlockedTensor<2, std::complex<double>>;
extern template class SymmetryBlockedTensor<3, std::complex<double>>;
extern template class SymmetryBlockedTensor<4, std::complex<double>>;

/**
 * @brief Variant of real- and complex-valued @ref SymmetryBlockedTensor at a
 * fixed rank.
 *
 * Mirrors the @ref ContainerTypes::MatrixVariant / @ref
 * ContainerTypes::VectorVariant pattern: API surfaces that need to expose
 * either a real or a complex symmetry-blocked tensor (e.g. spin-dependent
 * RDMs from a complex wavefunction) return this alias and let consumers
 * dispatch with @c std::visit or @c std::holds_alternative.
 *
 * @tparam Rank Tensor rank (1, 2, 3, or 4 are instantiated).
 */
template <std::size_t Rank>
using SymmetryBlockedTensorVariant =
    std::variant<SymmetryBlockedTensor<Rank, double>,
                 SymmetryBlockedTensor<Rank, std::complex<double>>>;

}  // namespace qdk::chemistry::data

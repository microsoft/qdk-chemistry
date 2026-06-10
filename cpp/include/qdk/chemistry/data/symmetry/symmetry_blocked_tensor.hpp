// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <complex>
#include <cstddef>
#include <fstream>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked.hpp>
#include <qdk/chemistry/utils/scalar_traits.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Storage type for one block of a rank-@p Rank tensor over @p Scalar.
 *
 * Rank-1 and rank-4 blocks are stored as (flat) dense column vectors; rank-2
 * and rank-3 blocks are stored as dense matrices. (Rank-3 blocks are typically
 * packed as @c [outer*outer, inner] for two paired index slots plus one
 * trailing index, e.g. a Cholesky three-center factor @f$L^{Q}_{ij}@f$ stored
 * as @f$[ij,\,Q]@f$; the size is validated against the product of the per-slot
 * extents and the precise row/column split is producer-chosen.) Only ranks 1–4
 * are supported; @ref SymmetryBlockedTensor rejects other ranks.
 */
template <std::size_t Rank, class Scalar = double>
using Tensor =
    std::conditional_t<(Rank == 1 || Rank == 4),
                       Eigen::Matrix<Scalar, Eigen::Dynamic, 1>,
                       Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>;

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
  static_assert(Rank >= 1 && Rank <= 4,
                "SymmetryBlockedTensor only supports ranks 1-4.");
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
   * Equivalent to the base @ref SymmetryBlocked @c BlockPtr with the block
   * type resolved to @ref Tensor. Held as @c shared_ptr<const Tensor> so that
   * symmetry-equivalent sectors can alias the same storage.
   */
  using BlockPtr = std::shared_ptr<const Tensor<Rank, Scalar>>;
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
   * @c SERIALIZATION_VERSION before reconstructing.
   *
   * @param j JSON object produced by a prior @ref to_json call.
   * @return Shared pointer to the reconstructed tensor.
   * @throws std::runtime_error if @p j is missing the @c "version" field or
   *         its version is incompatible with @c SERIALIZATION_VERSION.
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
   * @c SERIALIZATION_VERSION before reconstructing.
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
    } else if constexpr (Rank == 3) {
      // Rank-3 blocks are stored as a dense matrix; only the total size is
      // checked against the product of per-slot extents (the row/column
      // split is producer-chosen, e.g. [ij, Q] vs [i, jQ]).
      std::size_t expected = dims[0] * dims[1] * dims[2];
      if (static_cast<std::size_t>(block.size()) != expected ||
          block.size() == 0) {
        throw std::invalid_argument(
            "SymmetryBlockedTensor rank-3 block size does not match product "
            "of per-slot extents.");
      }
    } else {
      // Rank-4 blocks are stored flat-packed; the exact length is
      // producer/symmetry-dependent (permutational packing), so only require
      // a non-empty single-column vector here.
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

/**
 * @brief Build a rank-2 spin-diagonal @ref SymmetryBlockedTensor whose two
 * slots both carry a single-particle spin axis.
 *
 * Used by one-body integrals, inactive Fock matrices, and rank-2 1-RDMs;
 * those all have @c [n,n] alpha and beta blocks indexed by an MO axis.
 *
 * @tparam Derived Any @c Eigen::MatrixBase expression; the block scalar
 *           is taken from @c Derived::Scalar.
 * @param aa Alpha-alpha block (square @c [n,n] matrix expression).
 * @param bb Beta-beta block; ignored when @p restricted is @c true (the
 *           restricted axis aliases the beta partner to @p aa).
 * @param restricted Whether the spin axis is restricted (alpha and beta
 *           share storage).
 * @return Constructed rank-2 SBT.
 */
template <class Derived>
SymmetryBlockedTensor<2, typename Derived::Scalar> make_spin_diagonal_rank2_sbt(
    const Eigen::MatrixBase<Derived>& aa, const Eigen::MatrixBase<Derived>& bb,
    bool restricted) {
  using Scalar = typename Derived::Scalar;
  using SBT = SymmetryBlockedTensor<2, Scalar>;
  std::size_t n = static_cast<std::size_t>(aa.rows());
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, restricted)}));
  std::unordered_map<SymmetryLabel, std::size_t> ext;
  ext[axes::alpha()] = n;
  ext[axes::beta()] = n;
  typename SBT::BlockMap blocks;
  blocks[{axes::alpha(), axes::alpha()}] =
      std::make_shared<const Tensor<2, Scalar>>(aa);
  if (!restricted) {
    blocks[{axes::beta(), axes::beta()}] =
        std::make_shared<const Tensor<2, Scalar>>(bb);
  }
  return SBT(typename SBT::SymmetriesArray{sym, sym},
             typename SBT::ExtentsArray{ext, ext}, std::move(blocks));
}

/**
 * @brief Build a rank-4 spin-diagonal @ref SymmetryBlockedTensor.
 *
 * Used by two-body integrals and 2-RDMs that store @c n_active^4 flat
 * vectors per independent spin sector (@c aaaa, @c aabb, @c bbbb).
 *
 * @tparam Derived Any @c Eigen::MatrixBase expression (a flat
 *           @c n_active^4 column-vector-like block); the block scalar is
 *           taken from @c Derived::Scalar.
 * @param aaaa Alpha-alpha-alpha-alpha channel (flat @c n_active^4 vector).
 * @param aabb Alpha-alpha-beta-beta channel.
 * @param bbbb Beta-beta-beta-beta channel; ignored when @p restricted is
 *           @c true.
 * @param restricted Whether the spin axis is restricted.
 * @return Constructed rank-4 SBT.
 * @throws std::invalid_argument if @c aaaa.size() is not a perfect fourth
 *         power.
 */
template <class Derived>
SymmetryBlockedTensor<4, typename Derived::Scalar> make_spin_diagonal_rank4_sbt(
    const Eigen::MatrixBase<Derived>& aaaa,
    const Eigen::MatrixBase<Derived>& aabb,
    const Eigen::MatrixBase<Derived>& bbbb, bool restricted) {
  using Scalar = typename Derived::Scalar;
  using SBT = SymmetryBlockedTensor<4, Scalar>;
  std::size_t size = static_cast<std::size_t>(aaaa.size());
  std::size_t n_active = static_cast<std::size_t>(
      std::llround(std::pow(static_cast<double>(size), 0.25)));
  if (n_active * n_active * n_active * n_active != size) {
    throw std::invalid_argument("Spin-diagonal rank-4 block size " +
                                std::to_string(size) +
                                " is not a perfect fourth power.");
  }
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, restricted)}));
  std::unordered_map<SymmetryLabel, std::size_t> ext;
  ext[axes::alpha()] = n_active;
  ext[axes::beta()] = n_active;
  typename SBT::BlockMap blocks;
  blocks[{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()}] =
      std::make_shared<const Tensor<4, Scalar>>(aaaa);
  blocks[{axes::alpha(), axes::alpha(), axes::beta(), axes::beta()}] =
      std::make_shared<const Tensor<4, Scalar>>(aabb);
  if (!restricted) {
    blocks[{axes::beta(), axes::beta(), axes::beta(), axes::beta()}] =
        std::make_shared<const Tensor<4, Scalar>>(bbbb);
  }
  return SBT(typename SBT::SymmetriesArray{sym, sym, sym, sym},
             typename SBT::ExtentsArray{ext, ext, ext, ext}, std::move(blocks));
}

/**
 * @brief Single-channel restricted overload of @ref
 * make_spin_diagonal_rank4_sbt.
 *
 * Use when only one channel of rank-4 data is supplied and all four
 * equivalent spin patterns (@c aaaa, @c aabb, @c bbbb, @c bbaa) share
 * that single block. This applies to spin-restricted two-electron
 * integrals where @f$(\alpha\alpha|\alpha\alpha) =
 * (\alpha\alpha|\beta\beta) = (\beta\beta|\beta\beta)@f$ holds
 * physically, but does @b not apply to 2-RDMs (whose spin channels are
 * independent quantities even in restricted spin states).
 *
 * The alpha-alpha-alpha-alpha and alpha-alpha-beta-beta keys are both
 * pointed at the single supplied block; orbit aliasing on the restricted
 * spin axis then fills @c bbbb (from @c aaaa) and @c bbaa (from @c aabb)
 * with the same block pointer.
 *
 * @tparam Derived Any @c Eigen::MatrixBase expression; the block scalar
 *           is taken from @c Derived::Scalar.
 * @param aaaa The single channel of rank-4 data (flat @c n_active^4
 *           vector). Validated to be a perfect fourth power in size.
 * @return Constructed rank-4 SBT with one underlying block, aliased to
 *         four spin keys.
 * @throws std::invalid_argument if @c aaaa.size() is not a perfect fourth
 *         power.
 */
template <class Derived>
SymmetryBlockedTensor<4, typename Derived::Scalar> make_spin_diagonal_rank4_sbt(
    const Eigen::MatrixBase<Derived>& aaaa) {
  using Scalar = typename Derived::Scalar;
  using SBT = SymmetryBlockedTensor<4, Scalar>;
  std::size_t size = static_cast<std::size_t>(aaaa.size());
  std::size_t n_active = static_cast<std::size_t>(
      std::llround(std::pow(static_cast<double>(size), 0.25)));
  if (n_active * n_active * n_active * n_active != size) {
    throw std::invalid_argument("Spin-diagonal rank-4 block size " +
                                std::to_string(size) +
                                " is not a perfect fourth power.");
  }
  auto sym = std::make_shared<const SymmetryProduct>(
      SymmetryProduct({axes::spin(1, /*equivalent=*/true)}));
  std::unordered_map<SymmetryLabel, std::size_t> ext;
  ext[axes::alpha()] = n_active;
  ext[axes::beta()] = n_active;
  typename SBT::BlockMap blocks;
  auto aaaa_block = std::make_shared<const Tensor<4, Scalar>>(aaaa);
  blocks[{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()}] =
      aaaa_block;
  blocks[{axes::alpha(), axes::alpha(), axes::beta(), axes::beta()}] =
      aaaa_block;
  return SBT(typename SBT::SymmetriesArray{sym, sym, sym, sym},
             typename SBT::ExtentsArray{ext, ext, ext, ext}, std::move(blocks));
}

/**
 * @brief Optional overload of @ref make_spin_diagonal_rank2_sbt.
 *
 * Returns @c nullptr when @p aa is empty (no data supplied); otherwise
 * builds the rank-2 spin-diagonal SBT and returns a shared pointer. The
 * spin axis is restricted iff @p bb is @c nullopt.
 *
 * @tparam Scalar Block scalar type.
 * @param aa Alpha-alpha block (or empty optional to indicate no data).
 * @param bb Beta-beta block (or empty optional for restricted).
 * @return Shared pointer to the SBT, or @c nullptr if @p aa is unset.
 */
template <class Scalar>
std::shared_ptr<const SymmetryBlockedTensor<2, Scalar>>
make_spin_diagonal_rank2_sbt(
    const std::optional<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>&
        aa,
    const std::optional<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>&
        bb) {
  if (!aa.has_value()) {
    return nullptr;
  }
  const bool restricted = !bb.has_value();
  const auto& bb_block = bb.has_value() ? bb.value() : aa.value();
  return std::make_shared<const SymmetryBlockedTensor<2, Scalar>>(
      make_spin_diagonal_rank2_sbt(aa.value(), bb_block, restricted));
}

/**
 * @brief Dense-matrix overload of @ref make_spin_diagonal_rank2_sbt that
 * treats an empty matrix as "absent" (the @c std::optional analog).
 *
 * Returns @c nullptr when @p aa has zero size; otherwise builds the rank-2
 * spin-diagonal SBT. The spin axis is restricted iff @p bb has zero size.
 * Use this overload to lift v1 dense APIs (which encode absence as an
 * empty matrix) into the SBT-native construction path without optional
 * boilerplate at the call site.
 *
 * @tparam Derived Any @c Eigen::MatrixBase expression; the block scalar
 *           is taken from @c Derived::Scalar.
 * @param aa Alpha-alpha block (empty matrix means "no data supplied").
 * @param bb Beta-beta block (empty matrix means restricted).
 * @return Shared pointer to the SBT, or @c nullptr if @p aa is empty.
 */
template <class Derived>
std::shared_ptr<const SymmetryBlockedTensor<2, typename Derived::Scalar>>
make_spin_diagonal_rank2_sbt(const Eigen::MatrixBase<Derived>& aa,
                             const Eigen::MatrixBase<Derived>& bb) {
  using Scalar = typename Derived::Scalar;
  if (aa.size() == 0) {
    return nullptr;
  }
  const bool restricted = (bb.size() == 0);
  return std::make_shared<const SymmetryBlockedTensor<2, Scalar>>(
      make_spin_diagonal_rank2_sbt(aa, restricted ? aa : bb, restricted));
}

/**
 * @brief Optional overload of @ref make_spin_diagonal_rank4_sbt.
 *
 * Returns @c nullptr when all three channels are unset. Builds the rank-4
 * spin-diagonal SBT from whichever channels are supplied, with the
 * restrictedness of the resulting spin axis determined by whether @p bbbb
 * is supplied: when @c nullopt, the axis is restricted and unspecified
 * channels are aliased to @p aaaa.
 *
 * @tparam Scalar Block scalar type.
 * @param aaaa Alpha-alpha-alpha-alpha channel (or empty optional).
 * @param aabb Alpha-alpha-beta-beta channel (or empty optional).
 * @param bbbb Beta-beta-beta-beta channel (or empty optional). When unset
 *           the spin axis is restricted.
 * @return Shared pointer to the SBT, or @c nullptr if all three channels
 *         are unset.
 */
template <class Scalar>
std::shared_ptr<const SymmetryBlockedTensor<4, Scalar>>
make_spin_diagonal_rank4_sbt(
    const std::optional<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& aaaa,
    const std::optional<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& aabb,
    const std::optional<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>& bbbb) {
  if (!aaaa.has_value() && !aabb.has_value() && !bbbb.has_value()) {
    return nullptr;
  }
  const bool restricted = !bbbb.has_value();
  const auto& aaaa_block =
      aaaa.has_value() ? aaaa.value()
                       : (bbbb.has_value() ? bbbb.value() : aabb.value());
  const auto& aabb_block = aabb.has_value() ? aabb.value() : aaaa_block;
  const auto& bbbb_block = bbbb.has_value() ? bbbb.value() : aaaa_block;
  return std::make_shared<const SymmetryBlockedTensor<4, Scalar>>(
      make_spin_diagonal_rank4_sbt(aaaa_block, aabb_block, bbbb_block,
                                   restricted));
}

/**
 * @brief Dense-vector overload of @ref make_spin_diagonal_rank4_sbt that
 * treats an empty vector as "absent" (the @c std::optional analog).
 *
 * Returns @c nullptr when all three channels are empty; otherwise builds
 * the rank-4 spin-diagonal SBT from whichever channels are supplied. The
 * spin axis is restricted iff @p bbbb is empty, in which case unspecified
 * channels are aliased to @p aaaa.
 *
 * @tparam Derived Any @c Eigen::MatrixBase expression; the block scalar
 *           is taken from @c Derived::Scalar.
 * @param aaaa Alpha-alpha-alpha-alpha channel (empty means "absent").
 * @param aabb Alpha-alpha-beta-beta channel (empty means "absent").
 * @param bbbb Beta-beta-beta-beta channel (empty means restricted).
 * @return Shared pointer to the SBT, or @c nullptr if all three channels
 *         are empty.
 */
template <class Derived>
std::shared_ptr<const SymmetryBlockedTensor<4, typename Derived::Scalar>>
make_spin_diagonal_rank4_sbt(const Eigen::MatrixBase<Derived>& aaaa,
                             const Eigen::MatrixBase<Derived>& aabb,
                             const Eigen::MatrixBase<Derived>& bbbb) {
  using Scalar = typename Derived::Scalar;
  if (aaaa.size() == 0 && aabb.size() == 0 && bbbb.size() == 0) {
    return nullptr;
  }
  const bool restricted = (bbbb.size() == 0);
  const auto& aaaa_block =
      aaaa.size() != 0 ? aaaa : (bbbb.size() != 0 ? bbbb : aabb);
  const auto& aabb_block = aabb.size() != 0 ? aabb : aaaa_block;
  const auto& bbbb_block = bbbb.size() != 0 ? bbbb : aaaa_block;
  return std::make_shared<const SymmetryBlockedTensor<4, Scalar>>(
      make_spin_diagonal_rank4_sbt(aaaa_block, aabb_block, bbbb_block,
                                   restricted));
}

/**
 * @brief Variant overload of @ref make_spin_diagonal_rank2_sbt — dispatches
 * to the scalar-typed builder by visiting @p aa.
 *
 * @param aa Alpha-alpha block as a real/complex matrix variant.
 * @param bb Beta-beta block (must hold the same scalar alternative as
 *           @p aa); ignored when @p restricted is @c true.
 * @param restricted Whether the spin axis is restricted.
 * @return Shared pointer to the constructed variant tensor.
 */
inline std::shared_ptr<const SymmetryBlockedTensorVariant<2>>
make_spin_diagonal_rank2_sbt_variant(
    const std::variant<Eigen::MatrixXd, Eigen::MatrixXcd>& aa,
    const std::variant<Eigen::MatrixXd, Eigen::MatrixXcd>& bb,
    bool restricted) {
  return std::visit(
      [&](const auto& aa_block)
          -> std::shared_ptr<const SymmetryBlockedTensorVariant<2>> {
        using Scalar = typename std::decay_t<decltype(aa_block)>::Scalar;
        const auto& bb_block =
            std::get<Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>>(bb);
        return std::make_shared<const SymmetryBlockedTensorVariant<2>>(
            std::in_place_type<SymmetryBlockedTensor<2, Scalar>>,
            make_spin_diagonal_rank2_sbt(aa_block, bb_block, restricted));
      },
      aa);
}

/**
 * @brief Variant overload of @ref make_spin_diagonal_rank4_sbt — dispatches
 * to the scalar-typed builder by visiting @p aaaa.
 *
 * @param aaaa Alpha-alpha-alpha-alpha block as a real/complex vector
 *           variant.
 * @param aabb Alpha-alpha-beta-beta block (same scalar alternative).
 * @param bbbb Beta-beta-beta-beta block (same scalar alternative); ignored
 *           when @p restricted is @c true.
 * @param restricted Whether the spin axis is restricted.
 * @return Shared pointer to the constructed variant tensor.
 */
inline std::shared_ptr<const SymmetryBlockedTensorVariant<4>>
make_spin_diagonal_rank4_sbt_variant(
    const std::variant<Eigen::VectorXd, Eigen::VectorXcd>& aaaa,
    const std::variant<Eigen::VectorXd, Eigen::VectorXcd>& aabb,
    const std::variant<Eigen::VectorXd, Eigen::VectorXcd>& bbbb,
    bool restricted) {
  return std::visit(
      [&](const auto& aaaa_block)
          -> std::shared_ptr<const SymmetryBlockedTensorVariant<4>> {
        using Scalar = typename std::decay_t<decltype(aaaa_block)>::Scalar;
        const auto& aabb_block =
            std::get<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(aabb);
        const auto& bbbb_block =
            std::get<Eigen::Matrix<Scalar, Eigen::Dynamic, 1>>(bbbb);
        return std::make_shared<const SymmetryBlockedTensorVariant<4>>(
            std::in_place_type<SymmetryBlockedTensor<4, Scalar>>,
            make_spin_diagonal_rank4_sbt(aaaa_block, aabb_block, bbbb_block,
                                         restricted));
      },
      aaaa);
}

/**
 * @brief Optional overload of @ref make_spin_diagonal_rank2_sbt_variant.
 *
 * Returns @c nullptr when neither spin channel is supplied. Restrictedness
 * is inferred from whether @p bb is supplied (restricted iff @p bb is unset
 * and the resulting axis aliases the alpha block).
 *
 * @param aa Alpha-alpha block as an optional matrix variant.
 * @param bb Beta-beta block as an optional matrix variant.
 * @return Shared pointer to the constructed variant tensor, or @c nullptr
 *         when both channels are unset.
 */
inline std::shared_ptr<const SymmetryBlockedTensorVariant<2>>
make_spin_diagonal_rank2_sbt_variant(
    const std::optional<std::variant<Eigen::MatrixXd, Eigen::MatrixXcd>>& aa,
    const std::optional<std::variant<Eigen::MatrixXd, Eigen::MatrixXcd>>& bb) {
  if (!aa.has_value() && !bb.has_value()) {
    return nullptr;
  }
  const auto& aa_block = aa.has_value() ? aa.value() : bb.value();
  const auto& bb_block = bb.has_value() ? bb.value() : aa.value();
  const bool restricted = !(aa.has_value() && bb.has_value());
  return make_spin_diagonal_rank2_sbt_variant(aa_block, bb_block, restricted);
}

/**
 * @brief Optional overload of @ref make_spin_diagonal_rank4_sbt_variant.
 *
 * Returns @c nullptr when none of the three spin channels are supplied.
 * The spin axis is restricted iff @p bbbb is unset; in that case
 * unspecified channels alias the supplied alpha-like block.
 *
 * @param aaaa Alpha-alpha-alpha-alpha channel as an optional vector variant.
 * @param aabb Alpha-alpha-beta-beta channel as an optional vector variant.
 * @param bbbb Beta-beta-beta-beta channel as an optional vector variant.
 * @return Shared pointer to the constructed variant tensor, or @c nullptr
 *         when all three channels are unset.
 */
inline std::shared_ptr<const SymmetryBlockedTensorVariant<4>>
make_spin_diagonal_rank4_sbt_variant(
    const std::optional<std::variant<Eigen::VectorXd, Eigen::VectorXcd>>& aaaa,
    const std::optional<std::variant<Eigen::VectorXd, Eigen::VectorXcd>>& aabb,
    const std::optional<std::variant<Eigen::VectorXd, Eigen::VectorXcd>>&
        bbbb) {
  if (!aaaa.has_value() && !aabb.has_value() && !bbbb.has_value()) {
    return nullptr;
  }
  const auto& aaaa_block =
      aaaa.has_value() ? aaaa.value()
                       : (bbbb.has_value() ? bbbb.value() : aabb.value());
  const auto& aabb_block = aabb.has_value() ? aabb.value() : aaaa_block;
  const auto& bbbb_block = bbbb.has_value() ? bbbb.value() : aaaa_block;
  const bool restricted = !bbbb.has_value();
  return make_spin_diagonal_rank4_sbt_variant(aaaa_block, aabb_block,
                                              bbbb_block, restricted);
}

}  // namespace qdk::chemistry::data

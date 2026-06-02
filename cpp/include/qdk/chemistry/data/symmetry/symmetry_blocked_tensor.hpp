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
 * Rank-1 and rank-4 blocks are stored as (flat) column vectors; rank-2 blocks
 * are stored as dense matrices.
 */
template <std::size_t Rank, class Scalar>
struct TensorType;

template <class S>
struct TensorType<1, S> {
  using type = Eigen::Matrix<S, Eigen::Dynamic, 1>;
};
template <class S>
struct TensorType<2, S> {
  using type = Eigen::Matrix<S, Eigen::Dynamic, Eigen::Dynamic>;
};
template <class S>
struct TensorType<3, S> {
  using type = Eigen::Matrix<S, Eigen::Dynamic, 1>;
};
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
 * to dense Eigen blocks. Each slot carries its own @ref Symmetries
 * and a per-label extent. Blocks are held via @c shared_ptr<const Tensor> so
 * that symmetry-equivalent sectors can alias the same storage.
 *
 * Symmetry aliasing is defined on the spin axis: a simultaneous
 * @f$\alpha \leftrightarrow \beta@f$ swap across all slots. When every slot
 * shares the same @ref Symmetries instance and the spin axis is marked
 * @c equivalent, the constructor auto-aliases each partner block to the
 * supplied representative. When the slots carry distinct @ref Symmetries
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
  using typename Base::BlockMap;
  using typename Base::BlockPtr;
  using typename Base::ExtentsArray;
  using typename Base::Labels;
  using typename Base::SymmetriesArray;

  /**
   * @brief Construct from per-slot symmetries, per-slot extents, and a block
   * map. See the class description for the validation rules.
   *
   * @throws std::invalid_argument if a block or extent label is not
   *         admissible under the matching slot's @ref Symmetries, if a
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

  std::string get_data_type_name() const override {
    return "symmetry_blocked_tensor";
  }

  std::string get_summary() const override {
    std::ostringstream oss;
    oss << "SymmetryBlockedTensor(rank=" << Rank << ", scalar="
        << (utils::is_complex_scalar_v<Scalar> ? "complex" : "real")
        << ", blocks=" << this->num_blocks()
        << ", independent=" << this->_group_by_pointer().size() << ")";
    return oss.str();
  }

  nlohmann::json to_json() const override {
    nlohmann::json j;
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

  void to_json_file(const std::string& filename) const override {
    std::ofstream out(filename);
    if (!out) {
      throw std::runtime_error("Failed to open file for writing: " + filename);
    }
    out << to_json().dump(2);
  }

  void to_hdf5(H5::Group& group) const override;

  void to_hdf5_file(const std::string& filename) const override;

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

  /** @brief Reconstruct from a JSON object produced by @ref to_json. */
  static std::shared_ptr<SymmetryBlockedTensor> from_json(
      const nlohmann::json& j) {
    auto symmetries = Base::_symmetries_from_json(j);
    auto extents = Base::_extents_from_json(j);

    BlockMap blocks;
    for (const auto& entry : j.at("blocks")) {
      auto blk = std::make_shared<const Tensor<Rank, Scalar>>(
          _block_from_json(entry.at("block")));
      for (const auto& key_json : entry.at("keys")) {
        std::vector<SymmetryLabel> labels;
        for (const auto& label_json : key_json) {
          labels.push_back(SymmetryLabel::from_json(label_json));
        }
        blocks.emplace(detail::make_labels<Rank>(labels), blk);
      }
    }
    return std::make_shared<SymmetryBlockedTensor>(
        std::move(symmetries), std::move(extents), std::move(blocks));
  }

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

  static std::shared_ptr<SymmetryBlockedTensor> from_hdf5(H5::Group& group);
  static std::shared_ptr<SymmetryBlockedTensor> from_hdf5_file(
      const std::string& filename);
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

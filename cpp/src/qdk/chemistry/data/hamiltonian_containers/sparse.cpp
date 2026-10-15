// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <qdk/chemistry/data/hamiltonian_containers/sparse.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "../hdf5_serialization.hpp"
#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

// Convert a v1 four-index sparse two-body map (tuple-of-size_t keys) into
// a generic @ref SparseMapBlock<4, double> (array-of-unsigned keys), so
// the v1 sparse constructors can delegate to the generic
// @ref make_spin_diagonal_rank4_sbsm helper.
static SparseMapBlock<4, double> to_sparse_block(
    SparseHamiltonianContainer::TwoBodyMap map, std::size_t num_orbitals) {
  SparseMapBlock<4, double> block;
  for (const auto& [idx, val] : map) {
    auto [p, q, r, s] = idx;
    if (p >= num_orbitals || q >= num_orbitals || r >= num_orbitals ||
        s >= num_orbitals ||
        num_orbitals > std::numeric_limits<unsigned>::max()) {
      throw std::invalid_argument(
          "Sparse two-body index is outside the orbital range");
    }
    block[{static_cast<unsigned>(p), static_cast<unsigned>(q),
           static_cast<unsigned>(r), static_cast<unsigned>(s)}] = val;
  }
  return block;
}

namespace detail {
static void validate_sparse_orbital_count(std::int64_t n) {
  const auto max_elements = std::vector<double>().max_size();
  if (n < 0 || n > std::numeric_limits<int>::max() ||
      (n != 0 && static_cast<std::uint64_t>(n) >
                     max_elements / static_cast<std::uint64_t>(n))) {
    throw std::invalid_argument("Invalid sparse Hamiltonian orbital count");
  }
}

static Eigen::MatrixXd dense_sparse_one_body(
    const Eigen::SparseMatrix<double>& integrals) {
  validate_sparse_orbital_count(integrals.rows());
  if (integrals.rows() != integrals.cols()) {
    throw std::invalid_argument(
        "Sparse one-body integrals must be a square matrix");
  }
  return Eigen::MatrixXd(integrals);
}

/**
 * @brief Pack two 32-bit unsigned indices into a single double via bitwise
 *        reinterpretation.
 *
 * @note The returned `double` is **not** a meaningful floating-point value;
 *       it must only be decoded with unpack_indices.
 *
 * @param a First index (e.g. row or orbital index p).
 * @param b Second index (e.g. column or orbital index q).
 * @return A `double` whose bit pattern encodes the pair (a, b).
 */
inline double pack_indices(uint32_t a, uint32_t b) {
  static_assert(sizeof(double) == 2 * sizeof(uint32_t));
  alignas(double) uint32_t packed[2] = {a, b};
  double result;
  std::memcpy(&result, packed, sizeof(double));
  return result;
}

/**
 * @brief Unpack a double produced by pack_indices back into two 32-bit
 *        unsigned indices.
 *
 * @param packed_val A `double` whose bit pattern was produced by
 *                   pack_indices.
 * @return A pair `{a, b}` of the originally packed indices.
 */
inline std::pair<uint32_t, uint32_t> unpack_indices(double packed_val) {
  static_assert(sizeof(double) == 2 * sizeof(uint32_t));
  alignas(double) uint32_t packed[2];
  std::memcpy(packed, &packed_val, sizeof(double));
  return {packed[0], packed[1]};
}
}  // namespace detail

SparseHamiltonianContainer::SparseHamiltonianContainer(
    Eigen::SparseMatrix<double> one_body_integrals,
    TwoBodyMap two_body_integrals, double core_energy, HamiltonianType type)
    : SparseHamiltonianContainer(
          Eigen::SparseMatrix<double>(one_body_integrals),
          make_spin_diagonal_rank4_sbsm<double>(
              to_sparse_block(std::move(two_body_integrals),
                              one_body_integrals.rows()),
              static_cast<std::size_t>(one_body_integrals.rows())),
          core_energy, type) {}

SparseHamiltonianContainer::SparseHamiltonianContainer(
    Eigen::SparseMatrix<double> one_body_integrals, double core_energy,
    HamiltonianType type)
    : SparseHamiltonianContainer(std::move(one_body_integrals), nullptr,
                                 core_energy, type) {}

SparseHamiltonianContainer::SparseHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals,
    const Eigen::VectorXd& two_body_integrals, double core_energy,
    HamiltonianType type)
    : SparseHamiltonianContainer(
          _to_sparse(one_body_integrals),
          make_spin_diagonal_rank4_sbsm<double>(
              to_sparse_block(
                  _to_map(two_body_integrals,
                          static_cast<std::size_t>(one_body_integrals.rows())),
                  one_body_integrals.rows()),
              static_cast<std::size_t>(one_body_integrals.rows())),
          core_energy, type) {}

SparseHamiltonianContainer::SparseHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals, double core_energy,
    HamiltonianType type)
    : SparseHamiltonianContainer(_to_sparse(one_body_integrals), nullptr,
                                 core_energy, type) {}

SparseHamiltonianContainer::SparseHamiltonianContainer(
    Eigen::SparseMatrix<double> one_body_integrals,
    std::shared_ptr<const SymmetryBlockedSparseMap<4>> two_body,
    double core_energy, HamiltonianType type)
    : HamiltonianContainer(
          make_spin_diagonal_rank2_sbt(
              detail::dense_sparse_one_body(one_body_integrals),
              Eigen::MatrixXd{}, /*restricted=*/true),
          _make_orbitals(static_cast<int>(one_body_integrals.rows())),
          core_energy,
          make_spin_diagonal_rank2_sbt(
              Eigen::MatrixXd(Eigen::MatrixXd::Zero(one_body_integrals.rows(),
                                                    one_body_integrals.cols())),
              Eigen::MatrixXd{}),
          type),
      _one_body_sparse(std::move(one_body_integrals)),
      _two_body_sparse(std::move(two_body)) {
  if (_two_body_sparse) {
    for (const auto& slot : _two_body_sparse->extents()) {
      for (const auto& [label, extent] : slot) {
        if (extent != static_cast<std::size_t>(_one_body_sparse.rows())) {
          throw std::invalid_argument(
              "Sparse two-body extents must match the one-body orbital count");
        }
      }
    }
  }
}

std::unique_ptr<HamiltonianContainer> SparseHamiltonianContainer::clone()
    const {
  return std::make_unique<SparseHamiltonianContainer>(
      _one_body_sparse, _two_body_sparse, _core_energy, _type);
}

std::string SparseHamiltonianContainer::get_container_type() const {
  return "sparse";
}

std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
           const Eigen::VectorXd&>
SparseHamiltonianContainer::get_two_body_integrals() const {
  if (!_two_body_dense_valid) {
    _materialize_dense_two_body();
  }
  // Model Hamiltonians are restricted: aaaa == aabb == bbbb
  return {_two_body_dense_cache, _two_body_dense_cache, _two_body_dense_cache};
}

double SparseHamiltonianContainer::get_two_body_element(unsigned i, unsigned j,
                                                        unsigned k, unsigned l,
                                                        SpinChannel) const {
  if (!_two_body_sparse) return 0.0;
  const auto& block = _two_body_sparse->block(
      {axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()});
  auto it = block.find({i, j, k, l});
  return it != block.end() ? it->second : 0.0;
}

bool SparseHamiltonianContainer::has_two_body_integrals() const {
  return _two_body_sparse != nullptr;
}

bool SparseHamiltonianContainer::is_restricted() const {
  // The dense one-body integral path goes through @ref HamiltonianContainer
  // as a single block aliased across spin (this container has no separate
  // β sparse matrix), so restrictedness is dictated entirely by the
  // two-body sparse map: when present, all four equivalent spin patterns
  // must share storage (orbit aliasing on a restricted spin axis).
  if (!_two_body_sparse) return true;
  return _two_body_sparse->all_aliased(
      {{{axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()},
        {axes::alpha(), axes::alpha(), axes::beta(), axes::beta()},
        {axes::beta(), axes::beta(), axes::beta(), axes::beta()}}});
}

bool SparseHamiltonianContainer::is_valid() const {
  QDK_LOG_TRACE_ENTERING();
  // Check if essential data is present
  if (!has_one_body_integrals()) {
    return false;
  }

  // Check dimension consistency
  try {
    validate_integral_dimensions();
  } catch (const std::exception&) {
    return false;
  }

  return true;
}

nlohmann::json SparseHamiltonianContainer::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;

  j["version"] = SERIALIZATION_VERSION;
  j["container_type"] = get_container_type();
  j["core_energy"] = _core_energy;
  j["type"] =
      (_type == HamiltonianType::Hermitian) ? "Hermitian" : "NonHermitian";
  j["is_restricted"] = is_restricted();

  // One-body integrals — stored as Eigen::SparseMatrix (not an SBT), kept as
  // a triplet list.
  int n = _orbitals->get_num_molecular_orbitals();
  j["num_orbitals"] = n;
  j["has_one_body_integrals"] = (_one_body_sparse.nonZeros() > 0);
  if (_one_body_sparse.nonZeros() > 0) {
    nlohmann::json one_body_list = nlohmann::json::array();
    for (int k = 0; k < _one_body_sparse.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(_one_body_sparse, k);
           it; ++it) {
        one_body_list.push_back({it.row(), it.col(), it.value()});
      }
    }
    j["one_body_integrals_alpha_sparse"] = one_body_list;
  }

  if (_two_body_sparse) {
    j["two_body_integrals"] = _two_body_sparse->to_json();
  }

  return j;
}

std::unique_ptr<SparseHamiltonianContainer>
SparseHamiltonianContainer::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  try {
    if (j.contains("version")) {
      validate_serialization_version(SERIALIZATION_VERSION,
                                     j["version"].get<std::string>());
    }

    double core_energy = j.value("core_energy", 0.0);
    HamiltonianType type = HamiltonianType::Hermitian;
    if (j.contains("type") && j["type"] == "NonHermitian") {
      type = HamiltonianType::NonHermitian;
    }

    // Load one-body integrals (sparse triplets)
    const bool has_one_body = j.value("has_one_body_integrals", false);
    auto count = j.value("num_orbitals", nlohmann::json(0));
    if (!j.contains("num_orbitals") && has_one_body &&
        !j.contains("one_body_integrals_alpha_sparse") &&
        j.contains("one_body_integrals_alpha")) {
      const auto& dense = j.at("one_body_integrals_alpha");
      if (!dense.is_array()) {
        throw std::invalid_argument("Invalid dense one-body matrix");
      }
      count = dense.size();
    }
    if (!count.is_number_integer() || count < 0 ||
        count > std::numeric_limits<int>::max()) {
      throw std::invalid_argument("Invalid sparse Hamiltonian orbital count");
    }
    const int n = count.get<int>();
    detail::validate_sparse_orbital_count(n);
    Eigen::SparseMatrix<double> one_body_sparse(n, n);
    if (has_one_body) {
      // Prefer sparse format; fall back to dense for backward compatibility
      if (j.contains("one_body_integrals_alpha_sparse")) {
        const auto& entries = j.at("one_body_integrals_alpha_sparse");
        if (!entries.is_array() ||
            entries.size() >
                static_cast<std::size_t>(std::numeric_limits<int>::max())) {
          throw std::invalid_argument("Invalid sparse one-body triplet list");
        }
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& entry : entries) {
          if (!entry.is_array() || entry.size() != 3 ||
              !entry[0].is_number_integer() || !entry[1].is_number_integer() ||
              entry[0] < 0 || entry[0] >= n || entry[1] < 0 || entry[1] >= n) {
            throw std::invalid_argument(
                "Invalid sparse one-body triplet or orbital index");
          }
          int row = entry[0].get<int>();
          int col = entry[1].get<int>();
          double val = entry[2].get<double>();
          triplets.emplace_back(row, col, val);
        }
        one_body_sparse.setFromTriplets(triplets.begin(), triplets.end());
        one_body_sparse.makeCompressed();
      } else if (j.contains("one_body_integrals_alpha")) {
        // Backward compatibility: dense format
        const auto& dense = j.at("one_body_integrals_alpha");
        if (!dense.is_array() || dense.size() != static_cast<std::size_t>(n)) {
          throw std::invalid_argument(
              "Dense one-body dimensions must match the orbital count");
        }
        for (const auto& row : dense) {
          if (!row.is_array() || row.size() != static_cast<std::size_t>(n)) {
            throw std::invalid_argument(
                "Dense one-body dimensions must match the orbital count");
          }
        }
        Eigen::MatrixXd one_body_dense = json_to_matrix(dense);
        one_body_sparse = _to_sparse(one_body_dense);
      }
    }

    // Load two-body integrals via SBSM-direct deserialization.
    std::shared_ptr<const SymmetryBlockedSparseMap<4>> h2_sparse;
    if (j.contains("two_body_integrals")) {
      h2_sparse =
          SymmetryBlockedSparseMap<4>::from_json(j["two_body_integrals"]);
    }

    return std::make_unique<SparseHamiltonianContainer>(
        std::move(one_body_sparse), std::move(h2_sparse), core_energy, type);

  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error(
        "Failed to parse SparseHamiltonianContainer from JSON: " +
        std::string(e.what()));
  }
}

void SparseHamiltonianContainer::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    // Version attribute
    H5::Attribute version_attr =
        group.createAttribute("version", string_type, scalar_space);
    std::string version_str = SERIALIZATION_VERSION;
    version_attr.write(string_type, version_str);

    // Container type attribute
    H5::Attribute container_type_attr =
        group.createAttribute("container_type", string_type, scalar_space);
    std::string ct = get_container_type();
    container_type_attr.write(string_type, ct);

    // Metadata subgroup
    H5::Group metadata_group = group.createGroup("metadata");

    H5::Attribute core_energy_attr = metadata_group.createAttribute(
        "core_energy", H5::PredType::NATIVE_DOUBLE, scalar_space);
    core_energy_attr.write(H5::PredType::NATIVE_DOUBLE, &_core_energy);

    std::string type_str =
        (_type == HamiltonianType::Hermitian) ? "Hermitian" : "NonHermitian";
    H5::Attribute type_attr =
        metadata_group.createAttribute("type", string_type, scalar_space);
    type_attr.write(string_type, type_str);

    hbool_t restricted_flag = 1;
    H5::Attribute restricted_attr = metadata_group.createAttribute(
        "is_restricted", H5::PredType::NATIVE_HBOOL, scalar_space);
    restricted_attr.write(H5::PredType::NATIVE_HBOOL, &restricted_flag);

    // One-body integrals as sparse dataset: N x 2 (packed row|col, value)
    {
      auto nnz = static_cast<hsize_t>(_one_body_sparse.nonZeros());
      hsize_t dims[2] = {nnz, 2};
      H5::DataSpace dataspace(2, dims);

      std::vector<double> buffer(nnz * 2);
      hsize_t idx = 0;
      for (int k = 0; k < _one_body_sparse.outerSize(); ++k) {
        for (Eigen::SparseMatrix<double>::InnerIterator it(_one_body_sparse, k);
             it; ++it) {
          buffer[idx * 2 + 0] = detail::pack_indices(
              static_cast<uint32_t>(it.row()), static_cast<uint32_t>(it.col()));
          buffer[idx * 2 + 1] = it.value();
          ++idx;
        }
      }

      H5::DataSet dataset =
          group.createDataSet("one_body_integrals_alpha_sparse",
                              H5::PredType::NATIVE_DOUBLE, dataspace);
      dataset.write(buffer.data(), H5::PredType::NATIVE_DOUBLE);

      // Store num_orbitals as attribute so we know the matrix dimensions
      H5::DataSpace scalar_ds(H5S_SCALAR);
      int n = _orbitals->get_num_molecular_orbitals();
      H5::Attribute norb_attr = dataset.createAttribute(
          "num_orbitals", H5::PredType::NATIVE_INT, scalar_ds);
      norb_attr.write(H5::PredType::NATIVE_INT, &n);
    }

    // Two-body integrals as sparse dataset: N x 3 (packed p|q, packed r|s, val)
    if (has_two_body_integrals()) {
      const auto& block = _two_body_sparse->block(
          {axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()});
      auto n_entries = static_cast<hsize_t>(block.size());
      hsize_t dims[2] = {n_entries, 3};
      H5::DataSpace dataspace(2, dims);

      // Pack into row-major buffer: [packed(p,q), packed(r,s), value] per row
      std::vector<double> buffer(n_entries * 3);
      hsize_t row = 0;
      for (const auto& [idx, val] : block) {
        buffer[row * 3 + 0] = detail::pack_indices(
            static_cast<uint32_t>(idx[0]), static_cast<uint32_t>(idx[1]));
        buffer[row * 3 + 1] = detail::pack_indices(
            static_cast<uint32_t>(idx[2]), static_cast<uint32_t>(idx[3]));
        buffer[row * 3 + 2] = val;
        ++row;
      }

      H5::DataSet dataset = group.createDataSet(
          "two_body_integrals_sparse", H5::PredType::NATIVE_DOUBLE, dataspace);
      dataset.write(buffer.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Save orbitals
    if (has_orbitals()) {
      H5::Group orbitals_group = group.createGroup("orbitals");
      _orbitals->to_hdf5(orbitals_group);
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in SparseHamiltonianContainer: " +
                             std::string(e.getCDetailMsg()));
  }
}

std::unique_ptr<SparseHamiltonianContainer>
SparseHamiltonianContainer::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    auto require_scalar = [](const H5::Attribute& attribute) {
      if (attribute.getSpace().getSimpleExtentType() != H5S_SCALAR) {
        throw std::runtime_error(
            "Sparse Hamiltonian attribute must be scalar: " +
            attribute.getName());
      }
    };

    // Validate version
    if (group.attrExists("version")) {
      H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
      H5::Attribute version_attr = group.openAttribute("version");
      require_scalar(version_attr);
      std::string version_str;
      version_attr.read(string_type, version_str);
      validate_serialization_version(SERIALIZATION_VERSION, version_str);
    }

    // Read metadata
    H5::Group metadata_group = group.openGroup("metadata");

    double core_energy = 0.0;
    if (metadata_group.attrExists("core_energy")) {
      H5::Attribute ce_attr = metadata_group.openAttribute("core_energy");
      require_scalar(ce_attr);
      ce_attr.read(H5::PredType::NATIVE_DOUBLE, &core_energy);
    }

    HamiltonianType type = HamiltonianType::Hermitian;
    if (metadata_group.attrExists("type")) {
      H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
      H5::Attribute type_attr = metadata_group.openAttribute("type");
      require_scalar(type_attr);
      std::string type_str;
      type_attr.read(string_type, type_str);
      if (type_str == "NonHermitian") {
        type = HamiltonianType::NonHermitian;
      }
    }

    // Load one-body integrals
    Eigen::SparseMatrix<double> one_body_sparse;
    H5::DataSet h1_dataset =
        group.openDataSet("one_body_integrals_alpha_sparse");

    // Read num_orbitals attribute
    int n_orb = 0;
    if (h1_dataset.attrExists("num_orbitals")) {
      H5::Attribute norb_attr = h1_dataset.openAttribute("num_orbitals");
      require_scalar(norb_attr);
      if (norb_attr.getTypeClass() != H5T_INTEGER) {
        throw std::runtime_error(
            "Sparse Hamiltonian orbital count must be an integer");
      }
      long long count = 0;
      norb_attr.read(H5::PredType::NATIVE_LLONG, &count);
      detail::validate_sparse_orbital_count(count);
      n_orb = static_cast<int>(count);
    }
    detail::validate_sparse_orbital_count(n_orb);

    auto read_sparse_dataset = [](const H5::DataSet& dataset,
                                  hsize_t packed_width, hsize_t legacy_width) {
      const auto space = dataset.getSpace();
      if (space.getSimpleExtentNdims() != 2) {
        throw std::runtime_error("Sparse integral dataset must have rank 2");
      }
      std::array<hsize_t, 2> dims{};
      space.getSimpleExtentDims(dims.data());
      if (dims[1] != packed_width && dims[1] != legacy_width) {
        throw std::runtime_error("Invalid sparse integral dataset width");
      }
      std::vector<double> buffer;
      if (dims[0] > buffer.max_size() / dims[1]) {
        throw std::runtime_error("Sparse integral dataset is too large");
      }
      buffer.resize(static_cast<std::size_t>(dims[0] * dims[1]));
      if (!buffer.empty()) {
        dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);
      }
      return std::make_pair(std::move(buffer), dims[1]);
    };

    auto read_index = [n_orb](double index) {
      if (!std::isfinite(index) || index < 0 || index >= n_orb ||
          std::floor(index) != index) {
        throw std::runtime_error(
            "Sparse integral index is outside the orbital range");
      }
      return static_cast<int>(index);
    };

    auto [h1_buffer, ncols] = read_sparse_dataset(h1_dataset, 2, 3);
    const auto nnz = h1_buffer.size() / ncols;
    if (nnz > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
      throw std::runtime_error("Too many sparse one-body entries");
    }

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    if (ncols == 2) {
      // Packed format: [packed(row,col), value]
      for (hsize_t i = 0; i < nnz; ++i) {
        auto [row, col] = detail::unpack_indices(h1_buffer[i * 2 + 0]);
        double val = h1_buffer[i * 2 + 1];
        triplets.emplace_back(read_index(row), read_index(col), val);
      }
    } else {
      // Legacy format: [row, col, value] as doubles
      for (hsize_t i = 0; i < nnz; ++i) {
        int row = read_index(h1_buffer[i * 3 + 0]);
        int col = read_index(h1_buffer[i * 3 + 1]);
        double val = h1_buffer[i * 3 + 2];
        triplets.emplace_back(row, col, val);
      }
    }
    one_body_sparse.resize(n_orb, n_orb);
    one_body_sparse.setFromTriplets(triplets.begin(), triplets.end());
    one_body_sparse.makeCompressed();

    // Load sparse two-body integrals (if present)
    TwoBodyMap two_body_map;
    if (group.nameExists("two_body_integrals_sparse")) {
      H5::DataSet dataset = group.openDataSet("two_body_integrals_sparse");
      auto [buffer, width] = read_sparse_dataset(dataset, 3, 5);
      const auto n_entries = buffer.size() / width;

      if (width == 3) {
        // Packed format: [packed(p,q), packed(r,s), value]
        for (hsize_t row = 0; row < n_entries; ++row) {
          auto [p, q] = detail::unpack_indices(buffer[row * 3 + 0]);
          auto [r, s] = detail::unpack_indices(buffer[row * 3 + 1]);
          double val = buffer[row * 3 + 2];
          two_body_map[{read_index(p), read_index(q), read_index(r),
                        read_index(s)}] = val;
        }
      } else {
        // Legacy format: [p, q, r, s, value] as doubles
        for (hsize_t row = 0; row < n_entries; ++row) {
          int p = read_index(buffer[row * 5 + 0]);
          int q = read_index(buffer[row * 5 + 1]);
          int r = read_index(buffer[row * 5 + 2]);
          int s = read_index(buffer[row * 5 + 3]);
          double val = buffer[row * 5 + 4];
          two_body_map[{p, q, r, s}] = val;
        }
      }
    }

    if (two_body_map.empty()) {
      return std::make_unique<SparseHamiltonianContainer>(
          std::move(one_body_sparse), core_energy, type);
    }
    return std::make_unique<SparseHamiltonianContainer>(
        std::move(one_body_sparse), std::move(two_body_map), core_energy, type);

  } catch (const H5::Exception& e) {
    throw std::runtime_error(
        "HDF5 error in SparseHamiltonianContainer::from_hdf5: " +
        std::string(e.getCDetailMsg()));
  }
}

void SparseHamiltonianContainer::to_fcidump_file(const std::string& filename,
                                                 size_t nalpha,
                                                 size_t nbeta) const {
  QDK_LOG_TRACE_ENTERING();

  if (is_unrestricted()) {
    throw std::runtime_error(
        "FCIDUMP format is not supported for unrestricted Hamiltonians.");
  }

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  const size_t num_orbs =
      static_cast<size_t>(_orbitals->get_num_molecular_orbitals());
  const size_t nelec = nalpha + nbeta;
  const double print_thresh = std::numeric_limits<double>::epsilon();

  // Build ORBSYM string (C1 symmetry: all orbitals in irrep 1)
  std::string orb_string;
  for (size_t i = 0; i < num_orbs; ++i) {
    if (i > 0) orb_string += ",";
    orb_string += "1";
  }

  // Write the FCIDUMP header
  file << "&FCI ";
  file << "NORB=" << num_orbs << ", ";
  file << "NELEC=" << nelec << ", ";
  file << "MS2=" << (nalpha - nbeta) << ",\n";
  file << "ORBSYM=" << orb_string << ",\n";
  file << "ISYM=1,\n";
  file << "&END\n";

  auto formatted_line = [&](size_t i, size_t j, size_t k, size_t l,
                            double val) {
    file << std::setw(28) << std::scientific << std::setprecision(16)
         << std::right << val << " ";
    file << std::setw(4) << i << " ";
    file << std::setw(4) << j << " ";
    file << std::setw(4) << k << " ";
    file << std::setw(4) << l << "\n";
  };

  // Write two-body integrals from sparse map (1-based indices)
  if (has_two_body_integrals()) {
    for (const auto& [idx, val] : _two_body_sparse->block(
             {axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()})) {
      if (std::abs(val) < print_thresh) continue;
      formatted_line(idx[0] + 1, idx[1] + 1, idx[2] + 1, idx[3] + 1, val);
    }
  }

  // Write one-body integrals from sparse matrix (lower triangle, 1-based)
  for (int k = 0; k < _one_body_sparse.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(_one_body_sparse, k); it;
         ++it) {
      if (it.row() < it.col()) continue;  // skip upper triangle
      if (std::abs(it.value()) < print_thresh) continue;
      formatted_line(it.row() + 1, it.col() + 1, 0, 0, it.value());
    }
  }

  // Write core energy
  formatted_line(0, 0, 0, 0, _core_energy);
}

const Eigen::SparseMatrix<double>&
SparseHamiltonianContainer::sparse_one_body_integrals() const {
  return _one_body_sparse;
}

SparseHamiltonianContainer::TwoBodyMap
SparseHamiltonianContainer::sparse_two_body_integrals() const {
  TwoBodyMap m;
  if (!_two_body_sparse) return m;
  for (const auto& [idx, val] : _two_body_sparse->block(
           {axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()})) {
    m.emplace(std::make_tuple(idx[0], idx[1], idx[2], idx[3]), val);
  }
  return m;
}

double SparseHamiltonianContainer::one_body_element(int i, int j) const {
  return _one_body_sparse.coeff(i, j);
}

void SparseHamiltonianContainer::_materialize_dense_two_body() const {
  size_t n = _orbitals->get_num_molecular_orbitals();
  size_t size = 1;
  const auto max_elements = std::min(
      std::vector<double>().max_size(),
      static_cast<std::size_t>(std::numeric_limits<Eigen::Index>::max()));
  for (int slot = 0; slot < 4; ++slot) {
    if (n != 0 && size > max_elements / n) {
      throw std::length_error("Dense two-body integrals are too large");
    }
    size *= n;
  }
  size_t n2 = n * n;
  size_t n3 = n2 * n;
  _two_body_dense_cache =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(size));
  if (_two_body_sparse) {
    for (const auto& [idx, val] : _two_body_sparse->block(
             {axes::alpha(), axes::alpha(), axes::alpha(), axes::alpha()})) {
      _two_body_dense_cache(static_cast<Eigen::Index>(
          idx[0] * n3 + idx[1] * n2 + idx[2] * n + idx[3])) = val;
    }
  }
  _two_body_dense_valid = true;
}

std::shared_ptr<ModelOrbitals> SparseHamiltonianContainer::_make_orbitals(
    int n) {
  // Model Hamiltonian basis: a full active space over n modes with no spin
  // axis. The container itself is always restricted (is_restricted() == true).
  return std::make_shared<ModelOrbitals>(static_cast<size_t>(n));
}

Eigen::SparseMatrix<double> SparseHamiltonianContainer::_to_sparse(
    const Eigen::MatrixXd& m) {
  Eigen::SparseMatrix<double> s = m.sparseView();
  s.makeCompressed();
  return s;
}

SparseHamiltonianContainer::TwoBodyMap SparseHamiltonianContainer::_to_map(
    const Eigen::VectorXd& v, size_t n) {
  TwoBodyMap m;
  size_t n2 = n * n;
  size_t n3 = n2 * n;
  for (size_t p = 0; p < n; ++p)
    for (size_t q = 0; q < n; ++q)
      for (size_t r = 0; r < n; ++r)
        for (size_t s = 0; s < n; ++s) {
          double val =
              v(static_cast<Eigen::Index>(p * n3 + q * n2 + r * n + s));
          if (val != 0.0) m[{p, q, r, s}] = val;
        }
  return m;
}

void SparseHamiltonianContainer::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  HamiltonianContainer::hash_update(ctx);
  hash_value(ctx, get_container_type());
  hash_value(ctx, _one_body_sparse);
  if (_two_body_sparse) {
    hash_field_presence(ctx, true);
    hash_value(ctx, _two_body_sparse->content_hash());
  } else {
    hash_field_presence(ctx, false);
  }
}

const SymmetryBlockedSparseMap<4>&
SparseHamiltonianContainer::two_body_integrals_sparse() const {
  if (!_two_body_sparse) {
    throw std::runtime_error(
        "Sparse two-body symmetry-blocked tensor (two_body_integrals_sparse) "
        "is not set.");
  }
  return *_two_body_sparse;
}

}  // namespace qdk::chemistry::data

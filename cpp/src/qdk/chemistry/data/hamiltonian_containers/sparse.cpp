// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

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
    SparseHamiltonianContainer::TwoBodyMap map) {
  SparseMapBlock<4, double> block;
  for (const auto& [idx, val] : map) {
    auto [p, q, r, s] = idx;
    block[{static_cast<unsigned>(p), static_cast<unsigned>(q),
           static_cast<unsigned>(r), static_cast<unsigned>(s)}] = val;
  }
  return block;
}

namespace detail {
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
              to_sparse_block(std::move(two_body_integrals)),
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
                          static_cast<std::size_t>(one_body_integrals.rows()))),
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
          make_spin_diagonal_rank2_sbt(Eigen::MatrixXd(one_body_integrals),
                                       Eigen::MatrixXd{}, /*restricted=*/true),
          _make_orbitals(static_cast<int>(one_body_integrals.rows())),
          core_energy,
          make_spin_diagonal_rank2_sbt(
              Eigen::MatrixXd(Eigen::MatrixXd::Zero(one_body_integrals.rows(),
                                                    one_body_integrals.cols())),
              Eigen::MatrixXd{}),
          type),
      _one_body_sparse(std::move(one_body_integrals)),
      _two_body_sparse(std::move(two_body)) {}

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
    int n = j.value("num_orbitals", 0);
    Eigen::SparseMatrix<double> one_body_sparse(n, n);
    if (j.value("has_one_body_integrals", false)) {
      // Prefer sparse format; fall back to dense for backward compatibility
      if (j.contains("one_body_integrals_alpha_sparse")) {
        std::vector<Eigen::Triplet<double>> triplets;
        for (const auto& entry : j["one_body_integrals_alpha_sparse"]) {
          int row = entry[0].get<int>();
          int col = entry[1].get<int>();
          double val = entry[2].get<double>();
          triplets.emplace_back(row, col, val);
        }
        one_body_sparse.setFromTriplets(triplets.begin(), triplets.end());
        one_body_sparse.makeCompressed();
      } else if (j.contains("one_body_integrals_alpha")) {
        // Backward compatibility: dense format
        Eigen::MatrixXd one_body_dense =
            json_to_matrix(j.at("one_body_integrals_alpha"));
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
    // Validate version
    if (group.attrExists("version")) {
      H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
      H5::Attribute version_attr = group.openAttribute("version");
      std::string version_str;
      version_attr.read(string_type, version_str);
      validate_serialization_version(SERIALIZATION_VERSION, version_str);
    }

    // Read metadata
    H5::Group metadata_group = group.openGroup("metadata");

    double core_energy = 0.0;
    if (metadata_group.attrExists("core_energy")) {
      H5::Attribute ce_attr = metadata_group.openAttribute("core_energy");
      ce_attr.read(H5::PredType::NATIVE_DOUBLE, &core_energy);
    }

    HamiltonianType type = HamiltonianType::Hermitian;
    if (metadata_group.attrExists("type")) {
      H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
      H5::Attribute type_attr = metadata_group.openAttribute("type");
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
      norb_attr.read(H5::PredType::NATIVE_INT, &n_orb);
    }

    H5::DataSpace h1_space = h1_dataset.getSpace();
    hsize_t h1_dims[2];
    h1_space.getSimpleExtentDims(h1_dims);
    auto nnz = h1_dims[0];
    auto ncols = h1_dims[1];

    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(nnz);

    if (ncols == 2) {
      // Packed format: [packed(row,col), value]
      std::vector<double> h1_buffer(nnz * 2);
      h1_dataset.read(h1_buffer.data(), H5::PredType::NATIVE_DOUBLE);
      for (hsize_t i = 0; i < nnz; ++i) {
        auto [row, col] = detail::unpack_indices(h1_buffer[i * 2 + 0]);
        double val = h1_buffer[i * 2 + 1];
        triplets.emplace_back(static_cast<int>(row), static_cast<int>(col),
                              val);
      }
    } else {
      // Legacy format: [row, col, value] as doubles
      std::vector<double> h1_buffer(nnz * 3);
      h1_dataset.read(h1_buffer.data(), H5::PredType::NATIVE_DOUBLE);
      for (hsize_t i = 0; i < nnz; ++i) {
        int row = static_cast<int>(h1_buffer[i * 3 + 0]);
        int col = static_cast<int>(h1_buffer[i * 3 + 1]);
        double val = h1_buffer[i * 3 + 2];
        triplets.emplace_back(row, col, val);
      }
    }
    one_body_sparse.resize(n_orb, n_orb);
    one_body_sparse.setFromTriplets(triplets.begin(), triplets.end());
    one_body_sparse.makeCompressed();

    // Load sparse two-body integrals (if present)
    TwoBodyMap two_body_map;
    try {
      H5::DataSet dataset = group.openDataSet("two_body_integrals_sparse");
      H5::DataSpace dataspace = dataset.getSpace();
      hsize_t dims[2];
      dataspace.getSimpleExtentDims(dims);
      auto n_entries = dims[0];
      auto ncols = dims[1];

      if (ncols == 3) {
        // Packed format: [packed(p,q), packed(r,s), value]
        std::vector<double> buffer(n_entries * 3);
        dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);
        for (hsize_t row = 0; row < n_entries; ++row) {
          auto [p, q] = detail::unpack_indices(buffer[row * 3 + 0]);
          auto [r, s] = detail::unpack_indices(buffer[row * 3 + 1]);
          double val = buffer[row * 3 + 2];
          two_body_map[{static_cast<int>(p), static_cast<int>(q),
                        static_cast<int>(r), static_cast<int>(s)}] = val;
        }
      } else {
        // Legacy format: [p, q, r, s, value] as doubles
        std::vector<double> buffer(n_entries * 5);
        dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);
        for (hsize_t row = 0; row < n_entries; ++row) {
          int p = static_cast<int>(buffer[row * 5 + 0]);
          int q = static_cast<int>(buffer[row * 5 + 1]);
          int r = static_cast<int>(buffer[row * 5 + 2]);
          int s = static_cast<int>(buffer[row * 5 + 3]);
          double val = buffer[row * 5 + 4];
          two_body_map[{p, q, r, s}] = val;
        }
      }
    } catch (const H5::Exception&) {
      // No two-body dataset — (e.g. Hückel model)
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
  size_t n2 = n * n;
  size_t n3 = n2 * n;
  _two_body_dense_cache =
      Eigen::VectorXd::Zero(static_cast<Eigen::Index>(n * n * n * n));
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
  std::vector<size_t> active(static_cast<size_t>(n));
  std::iota(active.begin(), active.end(), size_t{0});
  return std::make_shared<ModelOrbitals>(
      static_cast<size_t>(n),
      Orbitals::RestrictedCASIndices{std::move(active), {}});
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

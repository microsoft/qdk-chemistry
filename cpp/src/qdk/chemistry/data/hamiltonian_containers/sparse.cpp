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
    : HamiltonianContainer(
          Eigen::MatrixXd(one_body_integrals),
          _make_orbitals(static_cast<int>(one_body_integrals.rows())),
          core_energy,
          Eigen::MatrixXd::Zero(one_body_integrals.rows(),
                                one_body_integrals.cols()),
          type),
      _one_body_sparse(std::move(one_body_integrals)),
      _two_body_map(std::move(two_body_integrals)) {}

SparseHamiltonianContainer::SparseHamiltonianContainer(
    Eigen::SparseMatrix<double> one_body_integrals, double core_energy,
    HamiltonianType type)
    : HamiltonianContainer(
          Eigen::MatrixXd(one_body_integrals),
          _make_orbitals(static_cast<int>(one_body_integrals.rows())),
          core_energy,
          Eigen::MatrixXd::Zero(one_body_integrals.rows(),
                                one_body_integrals.cols()),
          type),
      _one_body_sparse(std::move(one_body_integrals)) {}

SparseHamiltonianContainer::SparseHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals,
    const Eigen::VectorXd& two_body_integrals, double core_energy,
    HamiltonianType type)
    : HamiltonianContainer(
          one_body_integrals,
          _make_orbitals(static_cast<int>(one_body_integrals.rows())),
          core_energy,
          Eigen::MatrixXd::Zero(one_body_integrals.rows(),
                                one_body_integrals.cols()),
          type),
      _one_body_sparse(_to_sparse(one_body_integrals)),
      _two_body_map(_to_map(two_body_integrals,
                            static_cast<int>(one_body_integrals.rows()))) {}

SparseHamiltonianContainer::SparseHamiltonianContainer(
    const Eigen::MatrixXd& one_body_integrals, double core_energy,
    HamiltonianType type)
    : HamiltonianContainer(
          one_body_integrals,
          _make_orbitals(static_cast<int>(one_body_integrals.rows())),
          core_energy,
          Eigen::MatrixXd::Zero(one_body_integrals.rows(),
                                one_body_integrals.cols()),
          type),
      _one_body_sparse(_to_sparse(one_body_integrals)) {}

std::unique_ptr<HamiltonianContainer> SparseHamiltonianContainer::clone()
    const {
  return std::make_unique<SparseHamiltonianContainer>(
      _one_body_sparse, _two_body_map, _core_energy, _type);
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
  auto it = _two_body_map.find({i, j, k, l});
  return it != _two_body_map.end() ? it->second : 0.0;
}

bool SparseHamiltonianContainer::has_two_body_integrals() const {
  return !_two_body_map.empty();
}

bool SparseHamiltonianContainer::is_restricted() const { return true; }

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
  j["is_restricted"] = true;

  // One-body integrals — store as sparse triplets [row, col, value]
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

  // Two-body integrals — store as sparse list of {p,q,r,s,value}
  j["has_two_body_integrals"] = has_two_body_integrals();
  if (has_two_body_integrals()) {
    nlohmann::json two_body_list = nlohmann::json::array();
    for (const auto& [idx, val] : _two_body_map) {
      const auto& [p, q, r, s] = idx;
      two_body_list.push_back({p, q, r, s, val});
    }
    j["two_body_integrals_sparse"] = two_body_list;
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

    // Load two-body integrals
    TwoBodyMap two_body_map;
    if (j.value("has_two_body_integrals", false) &&
        j.contains("two_body_integrals_sparse")) {
      for (const auto& entry : j["two_body_integrals_sparse"]) {
        int p = entry[0].get<int>();
        int q = entry[1].get<int>();
        int r = entry[2].get<int>();
        int s = entry[3].get<int>();
        double val = entry[4].get<double>();
        two_body_map[{p, q, r, s}] = val;
      }
    }

    if (two_body_map.empty()) {
      return std::make_unique<SparseHamiltonianContainer>(
          std::move(one_body_sparse), core_energy, type);
    }
    return std::make_unique<SparseHamiltonianContainer>(
        std::move(one_body_sparse), std::move(two_body_map), core_energy, type);

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
      auto n_entries = static_cast<hsize_t>(_two_body_map.size());
      hsize_t dims[2] = {n_entries, 3};
      H5::DataSpace dataspace(2, dims);

      // Pack into row-major buffer: [packed(p,q), packed(r,s), value] per row
      std::vector<double> buffer(n_entries * 3);
      hsize_t row = 0;
      for (const auto& [idx, val] : _two_body_map) {
        const auto& [p, q, r, s] = idx;
        buffer[row * 3 + 0] = detail::pack_indices(static_cast<uint32_t>(p),
                                                   static_cast<uint32_t>(q));
        buffer[row * 3 + 1] = detail::pack_indices(static_cast<uint32_t>(r),
                                                   static_cast<uint32_t>(s));
        buffer[row * 3 + 2] = val;
        ++row;
      }

      H5::DataSet dataset = group.createDataSet(
          "two_body_integrals_sparse", H5::PredType::NATIVE_DOUBLE, dataspace);
      dataset.write(buffer.data(), H5::PredType::NATIVE_DOUBLE);
    }

    // Save orbitals
    if (_orbitals) {
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
  for (const auto& [idx, val] : _two_body_map) {
    if (std::abs(val) < print_thresh) continue;
    const auto& [p, q, r, s] = idx;
    formatted_line(p + 1, q + 1, r + 1, s + 1, val);
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

const SparseHamiltonianContainer::TwoBodyMap&
SparseHamiltonianContainer::sparse_two_body_integrals() const {
  return _two_body_map;
}

double SparseHamiltonianContainer::one_body_element(int i, int j) const {
  return _one_body_sparse.coeff(i, j);
}

void SparseHamiltonianContainer::_materialize_dense_two_body() const {
  int n = _orbitals->get_num_molecular_orbitals();
  int n2 = n * n;
  int n3 = n2 * n;
  _two_body_dense_cache = Eigen::VectorXd::Zero(n * n * n * n);
  for (const auto& [idx, val] : _two_body_map) {
    const auto& [p, q, r, s] = idx;
    _two_body_dense_cache(p * n3 + q * n2 + r * n + s) = val;
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
    const Eigen::VectorXd& v, int n) {
  TwoBodyMap m;
  int n2 = n * n;
  int n3 = n2 * n;
  for (int p = 0; p < n; ++p)
    for (int q = 0; q < n; ++q)
      for (int r = 0; r < n; ++r)
        for (int s = 0; s < n; ++s) {
          double val = v(p * n3 + q * n2 + r * n + s);
          if (val != 0.0) m[{p, q, r, s}] = val;
        }
  return m;
}

}  // namespace qdk::chemistry::data

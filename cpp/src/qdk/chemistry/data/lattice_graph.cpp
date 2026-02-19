// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Sparse>
#include <cstdint>
#include <fstream>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace qdk::chemistry::data {

namespace detail {
using Triplet = Eigen::Triplet<double>;

// Helper: add an undirected edge (i, j) with weight t to the triplet list.
static void add_edge(std::vector<Triplet>& triplets, int i, int j, double t) {
  triplets.emplace_back(i, j, t);
  triplets.emplace_back(j, i, t);
}
}  // namespace detail

LatticeGraph::LatticeGraph(const Eigen::MatrixXd& adjacency_matrix,
                           bool symmetrize)
    : _num_sites(static_cast<std::uint64_t>(adjacency_matrix.rows())),
      _is_symmetric(symmetrize) {
  if (adjacency_matrix.rows() != adjacency_matrix.cols()) {
    throw std::invalid_argument("Adjacency matrix must be square.");
  }

  // Symmetrize
  Eigen::MatrixXd mat =
      symmetrize ? (adjacency_matrix + adjacency_matrix.transpose()) / 2.0
                 : adjacency_matrix;
  adjacency_ = mat.sparseView();
  adjacency_.makeCompressed();
  if (!_is_symmetric) {
    _is_symmetric = _check_symmetry(adjacency_);
  }
}

LatticeGraph::LatticeGraph(
    const std::map<std::pair<std::uint64_t, std::uint64_t>, double>&
        edge_weights,
    bool symmetrize, std::uint64_t num_sites)
    : _is_symmetric(symmetrize) {
  // get num_sites if not provided
  if (num_sites == 0) {
    for (const auto& [edge, weight] : edge_weights) {
      const auto& [i, j] = edge;
      if (i + 1 > num_sites) num_sites = i + 1;
      if (j + 1 > num_sites) num_sites = j + 1;
    }
  }
  _num_sites = num_sites;

  // build triplet list
  std::vector<detail::Triplet> triplets;
  triplets.reserve(symmetrize ? edge_weights.size() * 2 : edge_weights.size());
  for (const auto& [edge, weight] : edge_weights) {
    const auto& [i, j] = edge;
    triplets.emplace_back(static_cast<int>(i), static_cast<int>(j), weight);
    if (symmetrize && i != j) {
      triplets.emplace_back(static_cast<int>(j), static_cast<int>(i), weight);
    }
  }

  // build sparse adjacency matrix
  auto n = static_cast<Eigen::Index>(_num_sites);
  adjacency_.resize(n, n);
  adjacency_.setFromTriplets(triplets.begin(), triplets.end());
  adjacency_.makeCompressed();
  if (!_is_symmetric) {
    _is_symmetric = _check_symmetry(adjacency_);
  }
}

LatticeGraph::LatticeGraph(const Eigen::SparseMatrix<double>& sparse,
                           bool symmetrize)
    : _num_sites(static_cast<std::uint64_t>(sparse.rows())),
      _is_symmetric(symmetrize) {
  if (sparse.rows() != sparse.cols()) {
    throw std::invalid_argument("Adjacency matrix must be square.");
  }
  // symmetrize
  if (symmetrize) {
    adjacency_ =
        (sparse + Eigen::SparseMatrix<double>(sparse.transpose())) * 0.5;
  } else {
    adjacency_ = sparse;
  }
  adjacency_.makeCompressed();
  if (!_is_symmetric) {
    _is_symmetric = _check_symmetry(adjacency_);
  }
}

std::uint64_t LatticeGraph::num_sites() const { return _num_sites; }

const Eigen::SparseMatrix<double>& LatticeGraph::sparse_adjacency_matrix()
    const {
  return adjacency_;
}

Eigen::MatrixXd LatticeGraph::adjacency_matrix() const {
  return Eigen::MatrixXd(adjacency_);
}

bool LatticeGraph::symmetry() const { return _is_symmetric; }

double LatticeGraph::weight(std::uint64_t i, std::uint64_t j) const {
  return adjacency_.coeff(static_cast<Eigen::Index>(i),
                          static_cast<Eigen::Index>(j));
}

bool LatticeGraph::are_connected(std::uint64_t i, std::uint64_t j) const {
  return weight(i, j) != 0.0;
}

std::uint64_t LatticeGraph::num_nonzeros() const {
  return static_cast<std::uint64_t>(adjacency_.nonZeros());
}

std::uint64_t LatticeGraph::num_edges() const {
  std::uint64_t count = 0;
  for (int k = 0; k < adjacency_.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
         ++it) {
      if (it.row() < it.col()) count++;
    }
  }
  return count;
}

LatticeGraph LatticeGraph::chain(std::uint64_t n, bool periodic, double t) {
  if (n == 0) {
    throw std::invalid_argument("chain: n must be > 0.");
  }

  auto N = static_cast<int>(n);
  std::vector<detail::Triplet> triplets;
  triplets.reserve(2 * N);

  // chain
  for (int i = 0; i < N - 1; ++i) {
    detail::add_edge(triplets, i, i + 1, t);
  }

  // periodic boundary
  if (periodic && N > 2) {
    detail::add_edge(triplets, N - 1, 0, t);
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  return LatticeGraph(adj, false);
}

LatticeGraph LatticeGraph::square(std::uint64_t nx, std::uint64_t ny,
                                  bool periodic, double t) {
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("square: nx and ny must be > 0.");
  }
  if (periodic && (nx < 2 || ny < 2)) {
    throw std::invalid_argument(
        "square: periodic boundary condition requires nx and ny > 1.");
  }

  auto Nx = static_cast<int>(nx);
  auto Ny = static_cast<int>(ny);
  int N = Nx * Ny;

  // Helper to convert (x, y) coordinates to site index
  auto idx = [Nx](int x, int y) { return y * Nx + x; };

  std::vector<detail::Triplet> triplets;
  triplets.reserve(4 * N);

  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      // Right neighbour
      if (x + 1 < Nx) {
        detail::add_edge(triplets, idx(x, y), idx(x + 1, y), t);
        // periodic boundary
      } else if (periodic) {
        detail::add_edge(triplets, idx(x, y), idx(0, y), t);
      }
      // Upper neighbour
      if (y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y), idx(x, y + 1), t);
        // periodic boundary
      } else if (periodic) {
        detail::add_edge(triplets, idx(x, y), idx(x, 0), t);
      }
    }
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  return LatticeGraph(adj, false);
}

bool LatticeGraph::_check_symmetry(
    const Eigen::SparseMatrix<double>& mat) const {
  if (mat.rows() != mat.cols()) {
    return false;
  }
  for (int k = 0; k < mat.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(mat, k); it; ++it) {
      int i = static_cast<int>(it.row());
      int j = static_cast<int>(it.col());
      double val = it.value();
      if (val != 0.0) {
        double sym_val = mat.coeff(j, i);
        if (sym_val != val) {
          return false;
        }
      }
    }
  }
  return true;
}

std::string LatticeGraph::get_summary() const {
  QDK_LOG_TRACE_ENTERING();

  std::ostringstream oss;
  oss << "LatticeGraph Summary:\n";
  oss << "  Sites: " << _num_sites << "\n";
  oss << "  Edges: " << num_edges() << "\n";
  oss << "  Non-zeros: " << num_nonzeros() << "\n";
  oss << "  Symmetric: " << (_is_symmetric ? "true" : "false") << "\n";
  return oss.str();
}

void LatticeGraph::to_file(const std::string& filename,
                           const std::string& type) const {
  QDK_LOG_TRACE_ENTERING();

  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

nlohmann::json LatticeGraph::to_json() const {
  QDK_LOG_TRACE_ENTERING();

  // Store adjacency as sparse triplets [row, col, value]
  nlohmann::json edges = nlohmann::json::array();
  for (int k = 0; k < adjacency_.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
         ++it) {
      edges.push_back({it.row(), it.col(), it.value()});
    }
  }

  nlohmann::json j;
  j["num_sites"] = _num_sites;
  j["is_symmetric"] = _is_symmetric;
  j["adjacency_sparse"] = edges;
  return j;
}

void LatticeGraph::to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();

  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }
  file << to_json().dump(2);
  if (file.fail()) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

void LatticeGraph::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();

  try {
    H5::DataSpace scalar_space(H5S_SCALAR);

    // Store num_sites as attribute on the group
    H5::Attribute sites_attr = group.createAttribute(
        "num_sites", H5::PredType::NATIVE_UINT64, scalar_space);
    sites_attr.write(H5::PredType::NATIVE_UINT64, &_num_sites);

    // Store is_symmetric as attribute
    hbool_t sym_val = _is_symmetric ? 1 : 0;
    H5::Attribute sym_attr = group.createAttribute(
        "is_symmetric", H5::PredType::NATIVE_HBOOL, scalar_space);
    sym_attr.write(H5::PredType::NATIVE_HBOOL, &sym_val);

    // Write adjacency as sparse dataset: N x 3 (row, col, value)
    auto nnz = static_cast<hsize_t>(adjacency_.nonZeros());
    hsize_t dims[2] = {nnz, 3};
    H5::DataSpace dataspace(2, dims);

    std::vector<double> buffer(nnz * 3);
    hsize_t idx = 0;
    for (int k = 0; k < adjacency_.outerSize(); ++k) {
      for (Eigen::SparseMatrix<double>::InnerIterator it(adjacency_, k); it;
           ++it) {
        buffer[idx * 3 + 0] = static_cast<double>(it.row());
        buffer[idx * 3 + 1] = static_cast<double>(it.col());
        buffer[idx * 3 + 2] = it.value();
        ++idx;
      }
    }

    H5::DataSet dataset = group.createDataSet(
        "adjacency_sparse", H5::PredType::NATIVE_DOUBLE, dataspace);
    dataset.write(buffer.data(), H5::PredType::NATIVE_DOUBLE);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in LatticeGraph::to_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

void LatticeGraph::to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();

  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root_group = file.openGroup("/");
    to_hdf5(root_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

LatticeGraph LatticeGraph::from_file(const std::string& filename,
                                     const std::string& type) {
  QDK_LOG_TRACE_ENTERING();

  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

LatticeGraph LatticeGraph::from_json_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Unable to open LatticeGraph JSON file '" + filename +
        "'. Please check that the file exists and you have read permissions.");
  }
  nlohmann::json json_obj;
  file >> json_obj;
  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }
  return from_json(json_obj);
}

LatticeGraph LatticeGraph::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();

  if (!j.contains("num_sites")) {
    throw std::runtime_error("JSON missing required 'num_sites' field");
  }
  if (!j.contains("adjacency_sparse")) {
    throw std::runtime_error("JSON missing required 'adjacency_sparse' field");
  }

  auto n = j["num_sites"].get<std::uint64_t>();

  std::vector<detail::Triplet> triplets;
  for (const auto& entry : j["adjacency_sparse"]) {
    int row = entry[0].get<int>();
    int col = entry[1].get<int>();
    double val = entry[2].get<double>();
    triplets.emplace_back(row, col, val);
  }
  Eigen::SparseMatrix<double> sparse(static_cast<Eigen::Index>(n),
                                     static_cast<Eigen::Index>(n));
  sparse.setFromTriplets(triplets.begin(), triplets.end());
  sparse.makeCompressed();
  return LatticeGraph(sparse, false);
}

LatticeGraph LatticeGraph::from_hdf5_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();

  H5::H5File file;
  try {
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open LatticeGraph HDF5 file '" +
                             filename +
                             "'. Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    H5::Group root_group = file.openGroup("/");
    return from_hdf5(root_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error(
        "Unable to read LatticeGraph data from HDF5 file '" + filename +
        "'. HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

LatticeGraph LatticeGraph::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  H5::DataSet dataset = group.openDataSet("adjacency_sparse");

  // Read num_sites from group attribute
  std::uint64_t n = 0;
  if (group.attrExists("num_sites")) {
    H5::Attribute sites_attr = group.openAttribute("num_sites");
    sites_attr.read(H5::PredType::NATIVE_UINT64, &n);
  }

  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[2];
  dataspace.getSimpleExtentDims(dims);
  auto nnz = dims[0];

  std::vector<double> buffer(nnz * 3);
  dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);

  using T = Eigen::Triplet<double>;
  std::vector<T> triplets;
  triplets.reserve(nnz);
  for (hsize_t i = 0; i < nnz; ++i) {
    int row = static_cast<int>(buffer[i * 3 + 0]);
    int col = static_cast<int>(buffer[i * 3 + 1]);
    double val = buffer[i * 3 + 2];
    triplets.emplace_back(row, col, val);
  }

  Eigen::SparseMatrix<double> sparse(static_cast<Eigen::Index>(n),
                                     static_cast<Eigen::Index>(n));
  sparse.setFromTriplets(triplets.begin(), triplets.end());
  sparse.makeCompressed();
  return LatticeGraph(sparse, false);
}

}  // namespace qdk::chemistry::data

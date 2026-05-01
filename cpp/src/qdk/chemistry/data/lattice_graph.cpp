// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <Eigen/Sparse>
#include <algorithm>
#include <cstdint>
#include <fstream>
#include <limits>
#include <nlohmann/json.hpp>
#include <numeric>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <random>
#include <set>
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

LatticeGraph::LatticeGraph(
    const std::map<std::pair<std::uint64_t, std::uint64_t>, double>&
        edge_weights,
    std::uint64_t num_sites) {
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
  triplets.reserve(edge_weights.size());
  for (const auto& [edge, weight] : edge_weights) {
    const auto& [i, j] = edge;
    if (i >= _num_sites || j >= _num_sites) {
      throw std::invalid_argument("Edge (" + std::to_string(i) + ", " +
                                  std::to_string(j) +
                                  ") has index out of range for num_sites=" +
                                  std::to_string(_num_sites) + ".");
    }
    triplets.emplace_back(static_cast<int>(i), static_cast<int>(j), weight);
  }

  // build sparse adjacency matrix
  auto n = static_cast<Eigen::Index>(_num_sites);
  adjacency_.resize(n, n);
  adjacency_.setFromTriplets(triplets.begin(), triplets.end());
  adjacency_.makeCompressed();
  _is_symmetric = _check_symmetry(adjacency_);
}

LatticeGraph::LatticeGraph(Eigen::SparseMatrix<double> adjacency)
    : _num_sites(static_cast<std::uint64_t>(adjacency.rows())),
      adjacency_(std::move(adjacency)),
      _is_symmetric(_check_symmetry(adjacency_)) {}

LatticeGraph LatticeGraph::from_dense_matrix(
    const Eigen::MatrixXd& adjacency_matrix) {
  if (adjacency_matrix.rows() != adjacency_matrix.cols()) {
    throw std::invalid_argument("Adjacency matrix must be square.");
  }
  Eigen::SparseMatrix<double> sparse = adjacency_matrix.sparseView();
  sparse.makeCompressed();
  return LatticeGraph(std::move(sparse));
}

LatticeGraph LatticeGraph::from_sparse_matrix(
    const Eigen::SparseMatrix<double>& sparse) {
  if (sparse.rows() != sparse.cols()) {
    throw std::invalid_argument("Adjacency matrix must be square.");
  }
  Eigen::SparseMatrix<double> copy = sparse;
  copy.makeCompressed();
  return LatticeGraph(std::move(copy));
}

LatticeGraph LatticeGraph::make_bidirectional(const LatticeGraph& graph) {
  Eigen::SparseMatrix<double> sym =
      (graph.adjacency_ +
       Eigen::SparseMatrix<double>(graph.adjacency_.transpose()));
  sym.makeCompressed();
  return LatticeGraph(std::move(sym));
}

std::uint64_t LatticeGraph::num_sites() const { return _num_sites; }

const Eigen::SparseMatrix<double>& LatticeGraph::sparse_adjacency_matrix()
    const {
  return adjacency_;
}

Eigen::MatrixXd LatticeGraph::adjacency_matrix() const {
  return Eigen::MatrixXd(adjacency_);
}

bool LatticeGraph::is_symmetric() const { return _is_symmetric; }

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
  LatticeGraph graph(std::move(adj));
  graph._kind = LatticeKind::CHAIN;
  graph._kind_params = {static_cast<std::int64_t>(N),
                        static_cast<std::int64_t>(periodic ? 1 : 0)};
  return graph;
}

LatticeGraph LatticeGraph::square(std::uint64_t nx, std::uint64_t ny,
                                  bool periodic_x, bool periodic_y, double t) {
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("square: nx and ny must be > 0.");
  }
  if (periodic_x && nx < 2) {
    throw std::invalid_argument("square: periodic_x requires nx > 1.");
  }
  if (periodic_y && ny < 2) {
    throw std::invalid_argument("square: periodic_y requires ny > 1.");
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
      } else if (periodic_x) {
        detail::add_edge(triplets, idx(x, y), idx(0, y), t);
      }
      // Upper neighbour
      if (y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y), idx(x, y + 1), t);
        // periodic boundary
      } else if (periodic_y) {
        detail::add_edge(triplets, idx(x, y), idx(x, 0), t);
      }
    }
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  LatticeGraph graph(std::move(adj));
  graph._kind = LatticeKind::SQUARE;
  graph._kind_params = {Nx, Ny, periodic_x ? 1 : 0, periodic_y ? 1 : 0};
  return graph;
}

LatticeGraph LatticeGraph::triangular(std::uint64_t nx, std::uint64_t ny,
                                      bool periodic_x, bool periodic_y,
                                      double t) {
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("triangular: nx and ny must be > 0.");
  }
  if (periodic_x && nx < 2) {
    throw std::invalid_argument("triangular: periodic_x requires nx > 1.");
  }
  if (periodic_y && ny < 2) {
    throw std::invalid_argument("triangular: periodic_y requires ny > 1.");
  }

  auto Nx = static_cast<int>(nx);
  auto Ny = static_cast<int>(ny);
  int N = Nx * Ny;

  auto idx = [Nx](int x, int y) { return y * Nx + x; };

  std::vector<detail::Triplet> triplets;
  triplets.reserve(6 * N);

  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      // Right neighbour
      if (x + 1 < Nx) {
        detail::add_edge(triplets, idx(x, y), idx(x + 1, y), t);
      } else if (periodic_x) {
        detail::add_edge(triplets, idx(x, y), idx(0, y), t);
      }
      // Upper neighbour
      if (y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y), idx(x, y + 1), t);
      } else if (periodic_y) {
        detail::add_edge(triplets, idx(x, y), idx(x, 0), t);
      }
      // Diagonal neighbour (upper-right)
      if (x + 1 < Nx && y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y), idx(x + 1, y + 1), t);
      } else if (x + 1 >= Nx && y + 1 < Ny && periodic_x) {
        // x wraps, y does not
        detail::add_edge(triplets, idx(x, y), idx(0, y + 1), t);
      } else if (x + 1 < Nx && y + 1 >= Ny && periodic_y) {
        // y wraps, x does not
        detail::add_edge(triplets, idx(x, y), idx(x + 1, 0), t);
      } else if (x + 1 >= Nx && y + 1 >= Ny && periodic_x && periodic_y) {
        // both wrap (corner)
        detail::add_edge(triplets, idx(x, y), idx(0, 0), t);
      }
    }
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  LatticeGraph graph(std::move(adj));
  graph._kind = LatticeKind::TRIANGULAR;
  graph._kind_params = {Nx, Ny, periodic_x ? 1 : 0, periodic_y ? 1 : 0};
  return graph;
}

LatticeGraph LatticeGraph::honeycomb(std::uint64_t nx, std::uint64_t ny,
                                     bool periodic_x, bool periodic_y,
                                     double t) {
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("honeycomb: nx and ny must be > 0.");
  }
  if (periodic_x && nx < 2) {
    throw std::invalid_argument("honeycomb: periodic_x requires nx > 1.");
  }
  if (periodic_y && ny < 2) {
    throw std::invalid_argument("honeycomb: periodic_y requires ny > 1.");
  }

  auto Nx = static_cast<int>(nx);
  auto Ny = static_cast<int>(ny);
  int N = 2 * Nx * Ny;  // 2 sites per unit cell

  // Site indices within unit cell (x, y):
  //   A = 2 * (y * Nx + x),  B = 2 * (y * Nx + x) + 1
  auto idxA = [Nx](int x, int y) { return 2 * (y * Nx + x); };
  auto idxB = [Nx](int x, int y) { return 2 * (y * Nx + x) + 1; };

  std::vector<detail::Triplet> triplets;
  triplets.reserve(3 * N);

  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      // Intra-cell bond: A -- B
      detail::add_edge(triplets, idxA(x, y), idxB(x, y), t);

      // Inter-cell bond 1: B(x,y) -- A(x+1, y)  (horizontal)
      if (x + 1 < Nx) {
        detail::add_edge(triplets, idxB(x, y), idxA(x + 1, y), t);
      } else if (periodic_x) {
        detail::add_edge(triplets, idxB(x, y), idxA(0, y), t);
      }

      // Inter-cell bond 2: B(x,y) -- A(x, y+1)  (vertical)
      if (y + 1 < Ny) {
        detail::add_edge(triplets, idxB(x, y), idxA(x, y + 1), t);
      } else if (periodic_y) {
        detail::add_edge(triplets, idxB(x, y), idxA(x, 0), t);
      }
    }
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  LatticeGraph graph(std::move(adj));
  graph._kind = LatticeKind::HONEYCOMB;
  graph._kind_params = {Nx, Ny, periodic_x ? 1 : 0, periodic_y ? 1 : 0};
  return graph;
}

LatticeGraph LatticeGraph::kagome(std::uint64_t nx, std::uint64_t ny,
                                  bool periodic_x, bool periodic_y, double t) {
  if (nx == 0 || ny == 0) {
    throw std::invalid_argument("kagome: nx and ny must be > 0.");
  }
  if (periodic_x && nx < 2) {
    throw std::invalid_argument("kagome: periodic_x requires nx > 1.");
  }
  if (periodic_y && ny < 2) {
    throw std::invalid_argument("kagome: periodic_y requires ny > 1.");
  }

  auto Nx = static_cast<int>(nx);
  auto Ny = static_cast<int>(ny);
  int N = 3 * Nx * Ny;  // 3 sites per unit cell

  // Layout per unit cell:
  //   s0 -- s1  (horizontal edge, bottom of up-triangle)
  //   s0 -- s2  (left edge of up-triangle)
  //   s1 -- s2  (right edge of up-triangle)
  // Inter-cell bonds form the down-triangles.
  auto idx = [Nx](int x, int y, int s) { return 3 * (y * Nx + x) + s; };

  std::vector<detail::Triplet> triplets;
  triplets.reserve(6 * N);  // 4 edges per site, stored as pairs

  for (int y = 0; y < Ny; ++y) {
    for (int x = 0; x < Nx; ++x) {
      // Intra-cell (up-triangle) edges
      detail::add_edge(triplets, idx(x, y, 0), idx(x, y, 1), t);
      detail::add_edge(triplets, idx(x, y, 0), idx(x, y, 2), t);
      detail::add_edge(triplets, idx(x, y, 1), idx(x, y, 2), t);

      // Inter-cell edges (down-triangle connections)
      // s1(x,y) -- s0(x+1, y)  (horizontal, right)
      if (x + 1 < Nx) {
        detail::add_edge(triplets, idx(x, y, 1), idx(x + 1, y, 0), t);
      } else if (periodic_x) {
        detail::add_edge(triplets, idx(x, y, 1), idx(0, y, 0), t);
      }

      // s2(x,y) -- s0(x, y+1)  (vertical, up)
      if (y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y, 2), idx(x, y + 1, 0), t);
      } else if (periodic_y) {
        detail::add_edge(triplets, idx(x, y, 2), idx(x, 0, 0), t);
      }

      // s2(x,y) -- s1(x-1, y+1)  (diagonal, upper-left)
      if (x - 1 >= 0 && y + 1 < Ny) {
        detail::add_edge(triplets, idx(x, y, 2), idx(x - 1, y + 1, 1), t);
      } else if (x - 1 < 0 && y + 1 < Ny && periodic_x) {
        // x wraps, y does not
        int xl = (x - 1 + Nx) % Nx;
        detail::add_edge(triplets, idx(x, y, 2), idx(xl, y + 1, 1), t);
      } else if (x - 1 >= 0 && y + 1 >= Ny && periodic_y) {
        // y wraps, x does not
        detail::add_edge(triplets, idx(x, y, 2), idx(x - 1, 0, 1), t);
      } else if (x - 1 < 0 && y + 1 >= Ny && periodic_x && periodic_y) {
        // both wrap (corner)
        int xl = (x - 1 + Nx) % Nx;
        detail::add_edge(triplets, idx(x, y, 2), idx(xl, 0, 1), t);
      }
    }
  }

  Eigen::SparseMatrix<double> adj(N, N);
  adj.setFromTriplets(triplets.begin(), triplets.end());
  adj.makeCompressed();
  LatticeGraph graph(std::move(adj));
  graph._kind = LatticeKind::KAGOME;
  graph._kind_params = {Nx, Ny, periodic_x ? 1 : 0, periodic_y ? 1 : 0};
  return graph;
}

namespace detail {

// Collect every undirected edge (i, j) with i < j from the adjacency matrix.
static std::vector<std::pair<std::uint64_t, std::uint64_t>> undirected_edges(
    const Eigen::SparseMatrix<double>& adj) {
  std::vector<std::pair<std::uint64_t, std::uint64_t>> edges;
  for (int k = 0; k < adj.outerSize(); ++k) {
    for (Eigen::SparseMatrix<double>::InnerIterator it(adj, k); it; ++it) {
      if (it.row() < it.col() && it.value() != 0.0) {
        edges.emplace_back(static_cast<std::uint64_t>(it.row()),
                           static_cast<std::uint64_t>(it.col()));
      }
    }
  }
  return edges;
}

// Greedy edge coloring: place each edge in the lowest-index color whose
// vertices do not already touch that color.  Optionally retry with shuffled
// edge orders and keep the result with fewest colors.
static LatticeGraph::EdgeColoring greedy_edge_coloring(
    const std::vector<std::pair<std::uint64_t, std::uint64_t>>& edges_in,
    int seed, int trials) {
  if (edges_in.empty() || trials < 1) {
    return {};
  }

  LatticeGraph::EdgeColoring best;
  int best_count = std::numeric_limits<int>::max();
  std::mt19937 rng(static_cast<std::uint32_t>(seed));

  std::vector<std::size_t> order(edges_in.size());
  std::iota(order.begin(), order.end(), 0);

  for (int trial = 0; trial < trials; ++trial) {
    if (trial > 0) {
      std::shuffle(order.begin(), order.end(), rng);
    }

    LatticeGraph::EdgeColoring coloring;
    // For each vertex, the set of colors already incident to it.
    std::map<std::uint64_t, std::set<int>> vertex_colors;
    int max_color = -1;

    for (std::size_t pos : order) {
      const auto& edge = edges_in[pos];
      const auto& used_i = vertex_colors[edge.first];
      const auto& used_j = vertex_colors[edge.second];
      int chosen = 0;
      while (used_i.count(chosen) != 0 || used_j.count(chosen) != 0) {
        ++chosen;
      }
      coloring[edge] = chosen;
      vertex_colors[edge.first].insert(chosen);
      vertex_colors[edge.second].insert(chosen);
      if (chosen > max_color) max_color = chosen;
    }

    int distinct = max_color + 1;
    if (distinct < best_count) {
      best_count = distinct;
      best = std::move(coloring);
    }
  }
  return best;
}

// Deterministic two-coloring of an open chain: edge (i, i+1) gets color i % 2.
// For periodic chains, even N keeps two colors, odd N requires a third for
// the wrap edge to satisfy the no-incident-same-color constraint.
static LatticeGraph::EdgeColoring chain_coloring(std::int64_t n,
                                                 bool periodic) {
  LatticeGraph::EdgeColoring out;
  for (std::int64_t i = 0; i + 1 < n; ++i) {
    out[{static_cast<std::uint64_t>(i), static_cast<std::uint64_t>(i + 1)}] =
        static_cast<int>(i % 2);
  }
  if (periodic && n > 2) {
    int wrap_color = (n % 2 == 0) ? 1 : 2;  // last edge color is (n-2)%2
    out[{0, static_cast<std::uint64_t>(n - 1)}] = wrap_color;
  }
  return out;
}

// Deterministic edge coloring for the square lattice.  Horizontal and vertical
// edges live on disjoint axes; each axis can be 2-colored by alternating.
// With periodic boundaries, an odd extent on that axis forces a third color
// on its wrap edges.  Total colors: 2 (open) up to 4 (both axes odd-periodic).
static LatticeGraph::EdgeColoring square_coloring(std::int64_t Nx,
                                                  std::int64_t Ny,
                                                  bool periodic_x,
                                                  bool periodic_y) {
  LatticeGraph::EdgeColoring out;
  auto idx = [Nx](std::int64_t x, std::int64_t y) {
    return static_cast<std::uint64_t>(y * Nx + x);
  };
  auto put = [&out](std::uint64_t a, std::uint64_t b, int c) {
    auto edge = std::minmax(a, b);
    out[{edge.first, edge.second}] = c;
  };

  // Horizontal edges use colors {0, 1}; vertical edges use {2, 3}.  When a
  // periodic dimension has odd extent the wrap edge needs its own color
  // (4 for x-wrap parity-conflict, 5 for y-wrap parity-conflict).
  for (std::int64_t y = 0; y < Ny; ++y) {
    for (std::int64_t x = 0; x + 1 < Nx; ++x) {
      put(idx(x, y), idx(x + 1, y), static_cast<int>(x % 2));
    }
    if (periodic_x && Nx > 2) {
      int wrap_color = (Nx % 2 == 0) ? 1 : 4;
      put(idx(Nx - 1, y), idx(0, y), wrap_color);
    }
  }
  for (std::int64_t x = 0; x < Nx; ++x) {
    for (std::int64_t y = 0; y + 1 < Ny; ++y) {
      put(idx(x, y), idx(x, y + 1), 2 + static_cast<int>(y % 2));
    }
    if (periodic_y && Ny > 2) {
      int wrap_color = (Ny % 2 == 0) ? 3 : 5;
      put(idx(x, Ny - 1), idx(x, 0), wrap_color);
    }
  }

  // Compact the color labels so the result is in 0..(distinct-1).
  std::map<int, int> remap;
  for (const auto& [edge, c] : out) {
    remap.emplace(c, static_cast<int>(remap.size()));
  }
  for (auto& [edge, c] : out) {
    c = remap.at(c);
  }
  return out;
}

}  // namespace detail

const LatticeGraph::EdgeColoring& LatticeGraph::edge_coloring(
    int seed, int trials) const {
  if (_coloring_cache.has_value()) {
    return *_coloring_cache;
  }

  EdgeColoring coloring;
  switch (_kind) {
    case LatticeKind::CHAIN:
      coloring =
          detail::chain_coloring(_kind_params.at(0), _kind_params.at(1) != 0);
      break;
    case LatticeKind::SQUARE:
      coloring = detail::square_coloring(_kind_params.at(0), _kind_params.at(1),
                                         _kind_params.at(2) != 0,
                                         _kind_params.at(3) != 0);
      break;
    case LatticeKind::CUSTOM:
    case LatticeKind::TRIANGULAR:
    case LatticeKind::HONEYCOMB:
    case LatticeKind::KAGOME:
    default:
      coloring = detail::greedy_edge_coloring(
          detail::undirected_edges(adjacency_), seed, std::max(trials, 1));
      break;
  }

  std::set<int> distinct;
  for (const auto& [edge, c] : coloring) {
    distinct.insert(c);
  }
  _chromatic_index = static_cast<int>(distinct.size());
  _coloring_cache = std::move(coloring);
  return *_coloring_cache;
}

int LatticeGraph::chromatic_index() const {
  if (_chromatic_index < 0) {
    edge_coloring();
  }
  return _chromatic_index;
}

LatticeKind LatticeGraph::kind() const { return _kind; }

bool LatticeGraph::_check_symmetry(const Eigen::SparseMatrix<double>& mat) {
  if (mat.rows() != mat.cols()) {
    return false;
  }
  return mat.isApprox(Eigen::SparseMatrix<double>(mat.transpose()));
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
  auto n_idx = static_cast<int>(n);

  std::vector<detail::Triplet> triplets;
  for (const auto& entry : j["adjacency_sparse"]) {
    int row = entry[0].get<int>();
    int col = entry[1].get<int>();
    double val = entry[2].get<double>();
    if (row < 0 || row >= n_idx || col < 0 || col >= n_idx) {
      throw std::runtime_error(
          "Edge (" + std::to_string(row) + ", " + std::to_string(col) +
          ") has index out of range for num_sites=" + std::to_string(n) +
          " in JSON data.");
    }
    triplets.emplace_back(row, col, val);
  }
  Eigen::SparseMatrix<double> sparse(static_cast<Eigen::Index>(n),
                                     static_cast<Eigen::Index>(n));
  sparse.setFromTriplets(triplets.begin(), triplets.end());
  sparse.makeCompressed();
  return LatticeGraph(std::move(sparse));
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

  // Read num_sites from group attribute (required)
  if (!group.attrExists("num_sites")) {
    throw std::runtime_error(
        "HDF5 group missing required 'num_sites' attribute for LatticeGraph.");
  }
  std::uint64_t n = 0;
  H5::Attribute sites_attr = group.openAttribute("num_sites");
  sites_attr.read(H5::PredType::NATIVE_UINT64, &n);

  H5::DataSpace dataspace = dataset.getSpace();
  hsize_t dims[2];
  dataspace.getSimpleExtentDims(dims);
  auto nnz = dims[0];

  std::vector<double> buffer(nnz * 3);
  dataset.read(buffer.data(), H5::PredType::NATIVE_DOUBLE);

  auto n_idx = static_cast<int>(n);

  using T = Eigen::Triplet<double>;
  std::vector<T> triplets;
  triplets.reserve(nnz);
  for (hsize_t i = 0; i < nnz; ++i) {
    int row = static_cast<int>(buffer[i * 3 + 0]);
    int col = static_cast<int>(buffer[i * 3 + 1]);
    double val = buffer[i * 3 + 2];
    if (row < 0 || row >= n_idx || col < 0 || col >= n_idx) {
      throw std::runtime_error(
          "Edge (" + std::to_string(row) + ", " + std::to_string(col) +
          ") has index out of range for num_sites=" + std::to_string(n) +
          " in HDF5 data.");
    }
    triplets.emplace_back(row, col, val);
  }

  Eigen::SparseMatrix<double> sparse(static_cast<Eigen::Index>(n),
                                     static_cast<Eigen::Index>(n));
  sparse.setFromTriplets(triplets.begin(), triplets.end());
  sparse.makeCompressed();
  return LatticeGraph(std::move(sparse));
}

}  // namespace qdk::chemistry::data

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/lattice_graph.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>

#include "path_utils.hpp"
#include "property_binding_helpers.hpp"

namespace py = pybind11;

// Wrapper functions for file I/O methods that accept both strings and pathlib
// Path objects
void lattice_graph_to_file_wrapper(
    const qdk::chemistry::data::LatticeGraph &self, const py::object &filename,
    const std::string &format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

qdk::chemistry::data::LatticeGraph lattice_graph_from_file_wrapper(
    const py::object &filename, const std::string &format_type) {
  return qdk::chemistry::data::LatticeGraph::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

void lattice_graph_to_json_file_wrapper(
    const qdk::chemistry::data::LatticeGraph &self,
    const py::object &filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

qdk::chemistry::data::LatticeGraph lattice_graph_from_json_file_wrapper(
    const py::object &filename) {
  return qdk::chemistry::data::LatticeGraph::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void lattice_graph_to_hdf5_file_wrapper(
    const qdk::chemistry::data::LatticeGraph &self,
    const py::object &filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

qdk::chemistry::data::LatticeGraph lattice_graph_from_hdf5_file_wrapper(
    const py::object &filename) {
  return qdk::chemistry::data::LatticeGraph::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void bind_lattice_graph(pybind11::module &m) {
  using namespace qdk::chemistry::data;

  using qdk::chemistry::python::utils::bind_getter_as_property;

  py::class_<LatticeGraph, DataClass, py::smart_holder> lattice_graph(
      m, "LatticeGraph", R"(
Lattice graph defining the connectivity and geometry of a model Hamiltonian.

A LatticeGraph stores a (possibly weighted) adjacency matrix for a lattice of
sites. It provides factory methods for common lattice topologies and exposes
connectivity queries used by the model Hamiltonian builders.

Examples:
    >>> from qdk_chemistry.data import LatticeGraph
    >>> # 4-site chain with open boundary
    >>> chain = LatticeGraph.chain(4)
    >>> chain.num_sites
    4
    >>> chain.num_edges
    3
    >>> # 4-site ring (periodic)
    >>> ring = LatticeGraph.chain(4, periodic=True)
    >>> ring.num_edges
    4
    >>> # Custom lattice from adjacency matrix
    >>> import numpy as np
    >>> adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    >>> lattice = LatticeGraph.from_dense_matrix(adj)
)");

  // Constructor: edge-weight map
  lattice_graph.def(
      py::init<
          const std::map<std::pair<std::uint64_t, std::uint64_t>, double> &,
          std::uint64_t>(),
      R"(
Construct a lattice graph from a dictionary of edge weights.

Args:
    edge_weights (dict[tuple[int, int], float]): Dictionary mapping (i, j) pairs
        to edge weights.
    num_sites (int, optional): Number of sites. If 0, inferred from edge indices.
        Defaults to 0.
)",
      py::arg("edge_weights"), py::arg("num_sites") = 0);

  // Static factories for matrix input
  lattice_graph.def_static("from_dense_matrix",
                           &LatticeGraph::from_dense_matrix,
                           R"(
Create a lattice graph from a dense adjacency matrix.

Args:
    adjacency_matrix (numpy.ndarray): Dense adjacency matrix [n x n]. Non-zero
        entries indicate edges with that weight.

Returns:
    LatticeGraph: A new lattice graph.
)",
                           py::arg("adjacency_matrix"));

  lattice_graph.def_static("from_sparse_matrix",
                           &LatticeGraph::from_sparse_matrix,
                           R"(
Create a lattice graph from a sparse adjacency matrix.

Args:
    sparse_adjacency_matrix (scipy.sparse matrix): Sparse adjacency matrix [n x n].

Returns:
    LatticeGraph: A new lattice graph.
)",
                           py::arg("sparse_adjacency_matrix"));

  lattice_graph.def_static("make_bidirectional",
                           &LatticeGraph::make_bidirectional,
                           R"(
Return a new lattice graph with reverse edges added.

For each directed edge (i,j) with weight w, ensures (j,i) also
exists with the same weight. Computes A_out = A + A^T, so this
should be called on graphs where edges are specified in one
direction only.

Args:
    graph (LatticeGraph): The (possibly directed) lattice graph.

Returns:
    LatticeGraph: A new lattice graph with bidirectional edges.
)",
                           py::arg("graph"));

  // Properties / accessors
  lattice_graph.def_property_readonly("num_sites", &LatticeGraph::num_sites, R"(
Number of lattice sites.

Returns:
    int: Number of sites in the lattice.
)");
  lattice_graph.def_property_readonly("num_edges", &LatticeGraph::num_edges, R"(
Number of unique edges in the lattice.

Returns:
    int: Number of edges.
)");
  lattice_graph.def_property_readonly("num_nonzeros",
                                      &LatticeGraph::num_nonzeros, R"(
Number of non-zero entries in the adjacency matrix.

For a symmetric graph this is twice the number of edges.

Returns:
    int: Number of non-zero adjacency entries.
)");
  lattice_graph.def_property_readonly("is_symmetric",
                                      &LatticeGraph::is_symmetric, R"(
Whether the adjacency matrix is symmetric.

Returns:
    bool: True if the adjacency matrix is symmetric.
)");
  lattice_graph.def("adjacency_matrix", &LatticeGraph::adjacency_matrix, R"(
Return the dense adjacency matrix.

Returns:
    numpy.ndarray: Dense adjacency matrix [n x n].
)");
  lattice_graph.def("sparse_adjacency_matrix",
                    &LatticeGraph::sparse_adjacency_matrix,
                    R"(
Return the sparse adjacency matrix.

Returns:
    scipy.sparse.csc_matrix: Sparse adjacency matrix [n x n].
)",
                    py::return_value_policy::reference_internal);

  lattice_graph.def("weight", &LatticeGraph::weight, R"(
Get the weight of the edge between sites i and j.

Args:
    i (int): First site index.
    j (int): Second site index.

Returns:
    float: Edge weight (0 if not connected).
)",
                    py::arg("i"), py::arg("j"));

  lattice_graph.def("are_connected", &LatticeGraph::are_connected, R"(
Check whether two sites are connected by an edge.

Args:
    i (int): First site index.
    j (int): Second site index.

Returns:
    bool: True if sites i and j are connected.
)",
                    py::arg("i"), py::arg("j"));

  // Static factory methods
  lattice_graph.def_static("chain", &LatticeGraph::chain, R"(
Create a one-dimensional chain lattice.

Sites are labelled 0 ... n-1 with nearest-neighbour edges.

Example: chain (n=4)::

    0 --- 1 --- 2 --- 3

With periodic boundary condition:
    - Wrap bond: (n-1) -- 0  e.g. 3 -- 0

Args:
    n (int): Number of sites.
    periodic (bool, optional): If True, add an edge between the first and
        last site (ring topology). Requires n > 2. Defaults to False.
    t (float, optional): Hopping weight for all edges. Defaults to 1.0.

Returns:
    LatticeGraph: Chain lattice with n sites.

Raises:
    ValueError: If n == 0.

Examples:
    >>> chain = LatticeGraph.chain(6)
    >>> ring = LatticeGraph.chain(6, periodic=True)
)",
                           py::arg("n"), py::arg("periodic") = false,
                           py::arg("t") = 1.0);

  lattice_graph.def_static("square", &LatticeGraph::square, R"(
Create a two-dimensional square lattice.

Sites are indexed in row-major order: site index = y * nx + x.
Total sites: nx * ny.

Example: 4x3 square lattice::

    8 --- 9 ---10 ---11
    |     |     |     |
    4 --- 5 --- 6 --- 7
    |     |     |     |
    0 --- 1 --- 2 --- 3

With periodic boundary conditions (using the 4x3 example above):
    - periodic_x wraps right to left:  3 -- 0, 7 -- 4, 11 -- 8
    - periodic_y wraps top to bottom:  8 -- 0, 9 -- 1, 10 -- 2, 11 -- 3

Args:
    nx (int): Number of sites along x.
    ny (int): Number of sites along y.
    periodic_x (bool, optional): If True, apply periodic boundary conditions
        along x. Requires nx >= 2. Defaults to False.
    periodic_y (bool, optional): If True, apply periodic boundary conditions
        along y. Requires ny >= 2. Defaults to False.
    t (float, optional): Hopping weight for all edges. Defaults to 1.0.

Returns:
    LatticeGraph: Square lattice with nx * ny sites.

Raises:
    ValueError: If nx or ny is 0.
)",
                           py::arg("nx"), py::arg("ny"),
                           py::arg("periodic_x") = false,
                           py::arg("periodic_y") = false, py::arg("t") = 1.0);

  lattice_graph.def_static("triangular", &LatticeGraph::triangular, R"(
Create a two-dimensional triangular lattice.

Sites are indexed in row-major order: site index = y * nx + x.
Total sites: nx * ny. Each site connects to its right and upper
square-lattice neighbours plus the upper-right diagonal neighbour,
forming a triangulation of the plane.

Example: 3x3 triangular lattice::

    6 --- 7 --- 8
    |  /  |  /  |
    3 --- 4 --- 5
    |  /  |  /  |
    0 --- 1 --- 2

With periodic boundary conditions (using the 3x3 example above):
    - periodic_x wraps right to left:  2 -- 0, 5 -- 3, 8 -- 6
    - periodic_y wraps top to bottom:  6 -- 0, 7 -- 1, 8 -- 2
    - Diagonal wraps require both periodic_x and periodic_y: 8 -- 0

Args:
    nx (int): Number of sites along x.
    ny (int): Number of sites along y.
    periodic_x (bool, optional): If True, apply periodic boundary conditions
        along x. Requires nx >= 2. Defaults to False.
    periodic_y (bool, optional): If True, apply periodic boundary conditions
        along y. Requires ny >= 2. Defaults to False.
    t (float, optional): Hopping weight for all edges. Defaults to 1.0.

Returns:
    LatticeGraph: Triangular lattice with nx * ny sites.

Raises:
    ValueError: If nx or ny is 0.
)",
                           py::arg("nx"), py::arg("ny"),
                           py::arg("periodic_x") = false,
                           py::arg("periodic_y") = false, py::arg("t") = 1.0);

  lattice_graph.def_static("honeycomb", &LatticeGraph::honeycomb, R"(
Create a two-dimensional honeycomb lattice.

The honeycomb lattice has two sites per unit cell (A and B sublattices).
Unit cells are arranged on a rectangular grid of size nx x ny, giving a
total of 2 * nx * ny sites. Sites are indexed as:

    - A-sublattice: 2 * (y * nx + x)
    - B-sublattice: 2 * (y * nx + x) + 1

Example: 3x4 honeycomb (brick-wall representation)::

              18-19-20-21-22-23
               |     |     |
           12-13-14-15-16-17
            |     |     |
         6--7--8--9-10-11
         |     |     |
      0--1--2--3--4--5

With periodic boundary conditions (using the 3x4 example above):
    - periodic_x wraps right to left: 5 -- 0, 11 -- 6, 17 -- 12, 23 -- 18
    - periodic_y wraps top to bottom: 19 -- 0, 15 -- 2, 17 -- 4

Args:
    nx (int): Number of unit cells along x.
    ny (int): Number of unit cells along y.
    periodic_x (bool, optional): If True, apply periodic boundary conditions
        along x. Requires nx >= 2. Defaults to False.
    periodic_y (bool, optional): If True, apply periodic boundary conditions
        along y. Requires ny >= 2. Defaults to False.
    t (float, optional): Hopping weight for all edges. Defaults to 1.0.

Returns:
    LatticeGraph: Honeycomb lattice with 2 * nx * ny sites.

Raises:
    ValueError: If nx or ny is 0.
)",
                           py::arg("nx"), py::arg("ny"),
                           py::arg("periodic_x") = false,
                           py::arg("periodic_y") = false, py::arg("t") = 1.0);

  lattice_graph.def_static("kagome", &LatticeGraph::kagome, R"(
Create a two-dimensional kagome lattice.

The kagome lattice has three sites per unit cell, arranged as
corner-sharing triangles. Unit cells are on a rectangular grid of
size nx x ny, giving a total of 3 * nx * ny sites. Sites are indexed as:

    - site 0: 3 * (y * nx + x)
    - site 1: 3 * (y * nx + x) + 1
    - site 2: 3 * (y * nx + x) + 2

Unit cell (up-triangle)::

        2
       / \
      0---1

Example: 3x2 kagome::

        11       14       17
       /  \     /  \     /  \
      9---10--12---13--15---16
     /     \  /     \  /
    2       5        8
   / \     / \      / \
  0---1---3---4----6---7

With periodic boundary conditions (using the 3x2 example above):
    - periodic_x wraps right to left: 0 -- 7, 9 -- 16, 2 -- 16
    - periodic_y wraps top to bottom: 0 -- 11, 3 -- 14, 6 -- 17, 1 -- 14, 4 -- 17
    - Diagonal wraps (require both periodic_x and periodic_y): 7 -- 11

Args:
    nx (int): Number of unit cells along x.
    ny (int): Number of unit cells along y.
    periodic_x (bool, optional): If True, apply periodic boundary conditions
        along x. Requires nx >= 2. Defaults to False.
    periodic_y (bool, optional): If True, apply periodic boundary conditions
        along y. Requires ny >= 2. Defaults to False.
    t (float, optional): Hopping weight for all edges. Defaults to 1.0.

Returns:
    LatticeGraph: Kagome lattice with 3 * nx * ny sites.

Raises:
    ValueError: If nx or ny is 0.
)",
                           py::arg("nx"), py::arg("ny"),
                           py::arg("periodic_x") = false,
                           py::arg("periodic_y") = false, py::arg("t") = 1.0);

  lattice_graph.def("__repr__", [](const LatticeGraph &self) {
    return "<LatticeGraph sites=" + std::to_string(self.num_sites()) +
           " edges=" + std::to_string(self.num_edges()) +
           " symmetric=" + (self.is_symmetric() ? "True" : "False") + ">";
  });

  lattice_graph.def(
      "__str__", [](const LatticeGraph &self) { return self.get_summary(); });

  bind_getter_as_property(lattice_graph, "get_summary",
                          &LatticeGraph::get_summary, R"(
Get a human-readable summary of the lattice graph.

Returns:
    str: Multi-line summary with site/edge counts and symmetry info.

Examples:
    >>> summary = graph.get_summary()
    >>> print(summary)
    >>> # Or as a property:
    >>> print(graph.summary)
)");

  lattice_graph.def(
      "to_json",
      [](const LatticeGraph &self) -> std::string {
        return self.to_json().dump();
      },
      R"(
Convert the lattice graph to a JSON string.

Returns:
    str: JSON string with adjacency matrix and metadata.
)");
  lattice_graph.def_static(
      "from_json",
      [](const std::string &json_str) -> LatticeGraph {
        return LatticeGraph::from_json(nlohmann::json::parse(json_str));
      },
      R"(
Load a lattice graph from a JSON string.

Args:
    json_str (str): JSON string containing 'num_sites' and 'adjacency_matrix'.

Returns:
    LatticeGraph: New LatticeGraph instance.
)",
      py::arg("json_str"));
  lattice_graph.def("to_file", lattice_graph_to_file_wrapper, R"(
Save the lattice graph to a file.

Args:
    filename (str | pathlib.Path): Path to the output file.
    format_type (str): Format type ("json" or "hdf5").

Raises:
    ValueError: If format_type is not supported.

Examples:
    >>> graph.to_file("lattice.json", "json")
    >>> from pathlib import Path
    >>> graph.to_file(Path("lattice.json"), "json")
)",
                    py::arg("filename"), py::arg("format_type"));
  lattice_graph.def_static("from_file", lattice_graph_from_file_wrapper, R"(
Load a lattice graph from a file.

Args:
    filename (str | pathlib.Path): Path to the input file.
    format_type (str): Format type ("json" or "hdf5").

Returns:
    LatticeGraph: New LatticeGraph instance.

Examples:
    >>> graph = LatticeGraph.from_file("lattice.json", "json")
    >>> from pathlib import Path
    >>> graph = LatticeGraph.from_file(Path("lattice.json"), "json")
)",
                           py::arg("filename"), py::arg("format_type"));
  lattice_graph.def("to_json_file", lattice_graph_to_json_file_wrapper, R"(
Save the lattice graph to a JSON file.

Args:
    filename (str | pathlib.Path): Path to the output JSON file.

Examples:
    >>> graph.to_json_file("lattice.json")
    >>> from pathlib import Path
    >>> graph.to_json_file(Path("lattice.json"))
)",
                    py::arg("filename"));
  lattice_graph.def_static("from_json_file",
                           lattice_graph_from_json_file_wrapper, R"(
Load a lattice graph from a JSON file.

Args:
    filename (str | pathlib.Path): Path to the input JSON file.

Returns:
    LatticeGraph: New LatticeGraph instance.

Examples:
    >>> graph = LatticeGraph.from_json_file("lattice.json")
    >>> from pathlib import Path
    >>> graph = LatticeGraph.from_json_file(Path("lattice.json"))
)",
                           py::arg("filename"));
  lattice_graph.def("to_hdf5_file", lattice_graph_to_hdf5_file_wrapper, R"(
Save the lattice graph to an HDF5 file.

Args:
    filename (str | pathlib.Path): Path to the output HDF5 file.

Examples:
    >>> graph.to_hdf5_file("lattice.h5")
    >>> from pathlib import Path
    >>> graph.to_hdf5_file(Path("lattice.h5"))
)",
                    py::arg("filename"));
  lattice_graph.def_static("from_hdf5_file",
                           lattice_graph_from_hdf5_file_wrapper, R"(
Load a lattice graph from an HDF5 file.

Args:
    filename (str | pathlib.Path): Path to the input HDF5 file.

Returns:
    LatticeGraph: New LatticeGraph instance.

Examples:
    >>> graph = LatticeGraph.from_hdf5_file("lattice.h5")
    >>> from pathlib import Path
    >>> graph = LatticeGraph.from_hdf5_file(Path("lattice.h5"))
)",
                           py::arg("filename"));

  // Pickling support using JSON serialization
  lattice_graph.def(py::pickle(
      [](const LatticeGraph &lg) -> std::string {
        // Return JSON string for pickling
        return lg.to_json().dump();
      },
      [](const std::string &json_str) -> LatticeGraph {
        // Reconstruct from JSON string
        return LatticeGraph::from_json(nlohmann::json::parse(json_str));
      }));

  // Data type name class attribute
  lattice_graph.attr("_data_type_name") = DATACLASS_TO_SNAKE_CASE(LatticeGraph);
}

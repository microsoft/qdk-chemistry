// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/data/lattice_graph.hpp>

namespace py = pybind11;

void bind_lattice_graph(pybind11::module& m) {
  using namespace qdk::chemistry::data;

  py::class_<LatticeGraph>(m, "LatticeGraph", R"(
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
    >>> lattice = LatticeGraph(adj)
)")
      // Constructors
      .def(py::init<const Eigen::MatrixXd&, bool>(),
           R"(
Construct a lattice graph from a dense adjacency matrix.

Args:
    adjacency_matrix (numpy.ndarray): Dense adjacency matrix [n x n]. Non-zero
        entries indicate edges with that weight.
    symmetrize (bool, optional): If True, symmetrise the adjacency matrix.
        Defaults to False.
)",
           py::arg("adjacency_matrix"), py::arg("symmetrize") = false)

      .def(py::init<const Eigen::SparseMatrix<double>&, bool>(),
           R"(
Construct a lattice graph from a sparse adjacency matrix.

Args:
    sparse_adjacency_matrix (scipy.sparse matrix): Sparse adjacency matrix [n x n].
    symmetrize (bool, optional): If True, symmetrise the adjacency matrix.
        Defaults to False.
)",
           py::arg("sparse_adjacency_matrix"), py::arg("symmetrize") = false)

      .def(py::init<
               const std::map<std::pair<std::uint64_t, std::uint64_t>, double>&,
               bool, std::uint64_t>(),
           R"(
Construct a lattice graph from a dictionary of edge weights.

Args:
    edge_weights (dict[tuple[int, int], float]): Dictionary mapping (i, j) pairs
        to edge weights.
    symmetrize (bool, optional): If True, symmetrise the adjacency matrix.
        Defaults to False.
    num_sites (int, optional): Number of sites. If 0, inferred from edge indices.
        Defaults to 0.
)",
           py::arg("edge_weights"), py::arg("symmetrize") = false,
           py::arg("num_sites") = 0)

      // Properties / accessors
      .def_property_readonly("num_sites", &LatticeGraph::num_sites, R"(
Number of lattice sites.

Returns:
    int: Number of sites in the lattice.
)")
      .def_property_readonly("num_edges", &LatticeGraph::num_edges, R"(
Number of unique edges in the lattice.

Returns:
    int: Number of edges.
)")
      .def_property_readonly("num_nonzeros", &LatticeGraph::num_nonzeros, R"(
Number of non-zero entries in the adjacency matrix.

For a symmetric graph this is twice the number of edges.

Returns:
    int: Number of non-zero adjacency entries.
)")
      .def_property_readonly("is_symmetric", &LatticeGraph::symmetry, R"(
Whether the adjacency matrix is symmetric.

Returns:
    bool: True if the adjacency matrix is symmetric.
)")
      .def("adjacency_matrix", &LatticeGraph::adjacency_matrix, R"(
Return the dense adjacency matrix.

Returns:
    numpy.ndarray: Dense adjacency matrix [n x n].
)")
      .def("sparse_adjacency_matrix", &LatticeGraph::sparse_adjacency_matrix,
           R"(
Return the sparse adjacency matrix.

Returns:
    scipy.sparse.csc_matrix: Sparse adjacency matrix [n x n].
)",
           py::return_value_policy::reference_internal)

      .def("weight", &LatticeGraph::weight, R"(
Get the weight of the edge between sites i and j.

Args:
    i (int): First site index.
    j (int): Second site index.

Returns:
    float: Edge weight (0 if not connected).
)",
           py::arg("i"), py::arg("j"))

      .def("are_connected", &LatticeGraph::are_connected, R"(
Check whether two sites are connected by an edge.

Args:
    i (int): First site index.
    j (int): Second site index.

Returns:
    bool: True if sites i and j are connected.
)",
           py::arg("i"), py::arg("j"))

      // Static factory methods
      .def_static("chain", &LatticeGraph::chain, R"(
Create a one-dimensional chain lattice.

Args:
    n (int): Number of sites.
    periodic (bool, optional): If True, add a periodic boundary edge
        connecting the last site to the first. Defaults to False.
    t (float, optional): Hopping weight for all edges. Defaults to 1.0.

Returns:
    LatticeGraph: Chain lattice with n sites.

Examples:
    >>> chain = LatticeGraph.chain(6)
    >>> ring = LatticeGraph.chain(6, periodic=True)
)",
                  py::arg("n"), py::arg("periodic") = false, py::arg("t") = 1.0)

      .def_static("square", &LatticeGraph::square, R"(
Create a two-dimensional square lattice.

Args:
    nx (int): Number of sites along x.
    ny (int): Number of sites along y.
    periodic (bool, optional): If True, apply periodic boundary conditions
        in both directions. Defaults to False.
    t (float, optional): Hopping weight for all edges. Defaults to 1.0.

Returns:
    LatticeGraph: Square lattice with nx * ny sites.
)",
                  py::arg("nx"), py::arg("ny"), py::arg("periodic") = false,
                  py::arg("t") = 1.0)

      .def("__repr__",
           [](const LatticeGraph& self) {
             return "<LatticeGraph sites=" + std::to_string(self.num_sites()) +
                    " edges=" + std::to_string(self.num_edges()) +
                    " symmetric=" + (self.symmetry() ? "True" : "False") + ">";
           })

      // DataClass interface
      .def("get_summary", &LatticeGraph::get_summary, R"(
Get a human-readable summary of the lattice graph.

Returns:
    str: Multi-line summary with site/edge counts and symmetry info.
)")
      .def("to_json", &LatticeGraph::to_json, R"(
Convert the lattice graph to a JSON representation.

Returns:
    dict: JSON-compatible dictionary with adjacency matrix and metadata.
)")
      .def_static("from_json", &LatticeGraph::from_json, R"(
Load a lattice graph from a JSON object.

Args:
    j (dict): Dictionary containing 'num_sites' and 'adjacency_matrix'.

Returns:
    LatticeGraph: New LatticeGraph instance.
)",
                  py::arg("j"))
      .def("to_file", &LatticeGraph::to_file, R"(
Save the lattice graph to a file.

Args:
    filename (str): Path to the output file.
    format_type (str): Format type ("json" or "hdf5").

Raises:
    ValueError: If format_type is not supported.
)",
           py::arg("filename"), py::arg("format_type"))
      .def_static("from_file", &LatticeGraph::from_file, R"(
Load a lattice graph from a file.

Args:
    filename (str): Path to the input file.
    format_type (str): Format type ("json" or "hdf5").

Returns:
    LatticeGraph: New LatticeGraph instance.
)",
                  py::arg("filename"), py::arg("format_type"))
      .def("to_json_file", &LatticeGraph::to_json_file, R"(
Save the lattice graph to a JSON file.

Args:
    filename (str): Path to the output JSON file.
)",
           py::arg("filename"))
      .def_static("from_json_file", &LatticeGraph::from_json_file, R"(
Load a lattice graph from a JSON file.

Args:
    filename (str): Path to the input JSON file.

Returns:
    LatticeGraph: New LatticeGraph instance.
)",
                  py::arg("filename"))
      .def("to_hdf5_file", &LatticeGraph::to_hdf5_file, R"(
Save the lattice graph to an HDF5 file.

Args:
    filename (str): Path to the output HDF5 file.
)",
           py::arg("filename"))
      .def_static("from_hdf5_file", &LatticeGraph::from_hdf5_file, R"(
Load a lattice graph from an HDF5 file.

Args:
    filename (str): Path to the input HDF5 file.

Returns:
    LatticeGraph: New LatticeGraph instance.
)",
                  py::arg("filename"));
}

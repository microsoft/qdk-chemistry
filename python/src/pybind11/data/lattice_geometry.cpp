// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/lattice_geometry.hpp>

#include "path_utils.hpp"
#include "property_binding_helpers.hpp"

namespace py = pybind11;

void bind_lattice_geometry(py::module& m) {
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;
  using qdk::chemistry::python::utils::to_string_path;

  py::class_<BondClass, py::smart_holder>(
      m, "BondClass", "Geometric shell and bond-axis class.")
      .def(py::init<std::uint64_t, std::uint32_t, Eigen::RowVectorXd>(),
           py::arg("shell"), py::arg("orientation"), py::arg("axis"), R"(
Describe a radial shell and an unoriented bond axis.

Args:
    shell (int): One-based radial shell index.
    orientation (int): Zero-based orientation index within the shell.
    axis (numpy.ndarray): Canonical unit axis with one component per spatial dimension.
)")
      .def_property_readonly("shell",
                             [](const BondClass& self) { return self.shell; })
      .def_property_readonly(
          "orientation", [](const BondClass& self) { return self.orientation; })
      .def_property_readonly("axis",
                             [](const BondClass& self) { return self.axis; });

  py::class_<NeighborConnection, py::smart_holder>(m, "NeighborConnection", R"(
Physical lattice connection, retaining its periodic image.

The displacement runs from ``site_i`` to the specified image of ``site_j``.
:class:`LatticeGeometry` queries return unit weight and no semantic flavor;
interaction graphs can assign weights and labels to these same records.
)")
      .def(py::init<std::uint64_t, std::uint64_t, BondClass, Eigen::RowVectorXd,
                    std::vector<std::int64_t>, std::optional<BondFlavorId>,
                    double>(),
           py::arg("site_i"), py::arg("site_j"), py::arg("bond_class"),
           py::arg("displacement"), py::arg("image_shift"),
           py::arg("flavor") = py::none(), py::arg("weight") = 1.0, R"(
Describe a physical connection and optional interaction metadata.

Args:
    site_i (int): Source site index.
    site_j (int): Target site index.
    bond_class (BondClass): Radial shell and unoriented axis class.
    displacement (numpy.ndarray): Cartesian displacement with one component per spatial dimension.
    image_shift (list[int]): One image coefficient per spatial dimension, in periodic-vector order, padded with zeros.
    flavor (int | None, optional): Semantic label, if assigned. Defaults to None.
    weight (float, optional): Interaction weight. Defaults to 1.0.
)")
      .def_property_readonly(
          "site_i", [](const NeighborConnection& self) { return self.site_i; })
      .def_property_readonly(
          "site_j", [](const NeighborConnection& self) { return self.site_j; })
      .def_property_readonly(
          "bond_class",
          [](const NeighborConnection& self) { return self.bond_class; })
      .def_property_readonly(
          "displacement",
          [](const NeighborConnection& self) { return self.displacement; })
      .def_property_readonly(
          "image_shift",
          [](const NeighborConnection& self) { return self.image_shift; })
      .def_property_readonly(
          "flavor", [](const NeighborConnection& self) { return self.flavor; })
      .def_property_readonly(
          "weight", [](const NeighborConnection& self) { return self.weight; });

  py::class_<LatticeGeometry, DataClass, py::smart_holder> geometry(
      m, "LatticeGeometry", R"(
Immutable Cartesian lattice geometry, independent of interactions.

Positions form a ``(num_sites, d)`` matrix for any positive dimension ``d``,
including empty geometries. Optional periodic vectors specify physical images.
Neighbor searches currently support two-dimensional geometries only. Built-in
factories are two-dimensional and retain compact integer coordinates for
stencil-based neighbor queries.
No adjacency matrix, semantic flavor assignment, or edge coloring is stored.
)");

  geometry
      .def(py::init<Eigen::MatrixXd, std::optional<Eigen::MatrixXd>>(),
           py::arg("positions"), py::arg("periods") = py::none(),
           py::call_guard<py::gil_scoped_release>(), R"(
Construct geometry from Cartesian positions and optional supercell vectors.

Args:
    positions (numpy.ndarray): Finite ``(num_sites, d)`` matrix with ``d > 0``; ``num_sites`` may be zero.
    periods (numpy.ndarray | None, optional): At most ``d`` independent, finite, nonzero row vectors of length ``d``. Defaults to None.

Raises:
    ValueError: If the position or periodic-vector matrix is invalid.
)")
      .def_property_readonly("num_sites", &LatticeGeometry::num_sites,
                             "Number of lattice sites.")
      .def_property_readonly("dimension", &LatticeGeometry::dimension,
                             "Number of Cartesian components per position.")
      .def_property_readonly(
          "positions",
          [](const LatticeGeometry& self) {
            py::gil_scoped_release release;
            return self.positions();
          },
          R"(
Cartesian positions in site-index order.

Returns:
    numpy.ndarray: Independent copy of the ``(num_sites, dimension)`` position matrix.
)")
      .def_property_readonly(
          "periods", [](const LatticeGeometry& self) { return self.periods(); },
          R"(
Periodic vectors in image-shift order.

Returns:
    numpy.ndarray | None: Independent copy of the periodic-vector matrix, or None for an open geometry.
)")
      .def("neighbor_connections", &LatticeGeometry::neighbor_connections,
           py::arg("shells"), py::arg("tolerance") = 1.0e-9,
           py::call_guard<py::gil_scoped_release>(), R"(
Return physical connections classified by radial shell and unoriented axis.

Distinct periodic images remain separate, including self-image connections.
Results are ordered by shell, orientation, endpoints, and image shift, with
``flavor=None`` and ``weight=1.0``. Unavailable finite shells contribute no entries.

Args:
    shells (list[int]): One-based shell indices; duplicate requests are ignored.
    tolerance (float, optional): Relative distance and absolute axis tolerance. Defaults to 1e-9.

Returns:
    list[NeighborConnection]: Canonical physical connections in the requested shells.

Raises:
    ValueError: If a shell is zero or tolerance is not finite and positive.
    RuntimeError: If the geometry is not two-dimensional.
    OverflowError: If an integer stencil, periodic image, or displacement exceeds the supported range.
)")
      .def("nearest_neighbor_shells", &LatticeGeometry::nearest_neighbor_shells,
           py::arg("shells"), py::arg("tolerance") = 1.0e-9,
           py::call_guard<py::gil_scoped_release>(), R"(
Project open-geometry connections onto sorted, unique site pairs.

Shell numbering uses the positive distances actually present in the geometry,
not the bulk lattice. All requested shells are classified together.

Args:
    shells (list[int]): One-based shell indices; duplicate requests are ignored.
    tolerance (float, optional): Relative distance and absolute axis tolerance. Defaults to 1e-9.

Returns:
    dict[int, list[tuple[int, int]]]: Canonical pairs with ``i < j``; unavailable shells map to empty lists.

Raises:
    ValueError: If a shell is zero or tolerance is not finite and positive.
    RuntimeError: If periodic vectors are present or the geometry is not two-dimensional.
)")
      .def("mth_nearest_neighbors", &LatticeGeometry::mth_nearest_neighbors,
           py::arg("m"), py::arg("tolerance") = 1.0e-9,
           py::call_guard<py::gil_scoped_release>(), R"(
Return the sorted, unique site pairs in one open-geometry neighbor shell.

Args:
    m (int): One-based shell index.
    tolerance (float, optional): Relative distance and absolute axis tolerance. Defaults to 1e-9.

Returns:
    list[tuple[int, int]]: Canonical pairs, or an empty list if the shell is unavailable.

Raises:
    ValueError: If m is zero or tolerance is not finite and positive.
    RuntimeError: If periodic vectors are present or the geometry is not two-dimensional.
)")
      .def_static("chain", &LatticeGeometry::chain, py::arg("n"),
                  py::arg("periodic") = false,
                  py::call_guard<py::gil_scoped_release>(), R"(
Create a unit-spaced chain with positions ``(i, 0)``.

Args:
    n (int): Positive number of sites.
    periodic (bool, optional): Include the supercell vector ``(n, 0)``. Defaults to False.

Returns:
    LatticeGeometry: Chain geometry, including periodic one- and two-site cells.
)")
      .def_static("square", &LatticeGeometry::square, py::arg("nx"),
                  py::arg("ny"), py::arg("periodic_x") = false,
                  py::arg("periodic_y") = false,
                  py::call_guard<py::gil_scoped_release>(), R"(
Create a square lattice with site index ``y * nx + x`` and unit spacing.

Args:
    nx (int): Positive number of sites along x.
    ny (int): Positive number of sites along y.
    periodic_x (bool, optional): Periodic x direction; requires nx > 1. Defaults to False.
    periodic_y (bool, optional): Periodic y direction; requires ny > 1. Defaults to False.

Returns:
    LatticeGeometry: Square geometry with ``nx * ny`` sites.
)")
      .def_static("triangular", &LatticeGeometry::triangular, py::arg("nx"),
                  py::arg("ny"), py::arg("periodic_x") = false,
                  py::arg("periodic_y") = false,
                  py::call_guard<py::gil_scoped_release>(), R"(
Create a triangular lattice with unit bond length and index ``y * nx + x``.

Args:
    nx (int): Positive number of sites along the first primitive vector.
    ny (int): Positive number of sites along the second primitive vector.
    periodic_x (bool, optional): Periodic first direction; requires nx > 1. Defaults to False.
    periodic_y (bool, optional): Periodic second direction; requires ny > 1. Defaults to False.

Returns:
    LatticeGeometry: Triangular geometry with ``nx * ny`` sites.
)")
      .def_static("honeycomb", &LatticeGeometry::honeycomb, py::arg("nx"),
                  py::arg("ny"), py::arg("periodic_x") = false,
                  py::arg("periodic_y") = false,
                  py::call_guard<py::gil_scoped_release>(), R"(
Create honeycomb geometry sized by two-site unit cells.

Site indices are ``2 * (y * nx + x) + sublattice``, with A before B.

Args:
    nx (int): Positive number of unit cells along the first primitive vector.
    ny (int): Positive number of unit cells along the second primitive vector.
    periodic_x (bool, optional): Periodic first direction; requires nx > 1. Defaults to False.
    periodic_y (bool, optional): Periodic second direction; requires ny > 1. Defaults to False.

Returns:
    LatticeGeometry: Honeycomb geometry with ``2 * nx * ny`` sites.
)")
      .def_static("honeycomb_plaquettes",
                  &LatticeGeometry::honeycomb_plaquettes, py::arg("nx"),
                  py::arg("ny"), py::arg("periodic_x") = false,
                  py::arg("periodic_y") = false,
                  py::call_guard<py::gil_scoped_release>(), R"(
Create a honeycomb patch sized by complete hexagonal plaquettes.

Open directions gain a boundary cell. Only fully open patches omit the first A
and last B corners, retaining the other sites in unit-cell order. A fully open
``1 x 1`` patch is a six-site hexagon.

Args:
    nx (int): Positive number of complete plaquettes along the first direction.
    ny (int): Positive number of complete plaquettes along the second direction.
    periodic_x (bool, optional): Periodic first direction; requires nx > 1. Defaults to False.
    periodic_y (bool, optional): Periodic second direction; requires ny > 1. Defaults to False.

Returns:
    LatticeGeometry: Geometry containing the requested complete plaquettes.
)")
      .def_static("kagome", &LatticeGeometry::kagome, py::arg("nx"),
                  py::arg("ny"), py::arg("periodic_x") = false,
                  py::arg("periodic_y") = false,
                  py::call_guard<py::gil_scoped_release>(), R"(
Create kagome geometry with three sites per cell and unit bond length.

Site indices are ``3 * (y * nx + x) + sublattice``.

Args:
    nx (int): Positive number of unit cells along the first primitive vector.
    ny (int): Positive number of unit cells along the second primitive vector.
    periodic_x (bool, optional): Periodic first direction; requires nx > 1. Defaults to False.
    periodic_y (bool, optional): Periodic second direction; requires ny > 1. Defaults to False.

Returns:
    LatticeGeometry: Kagome geometry with ``3 * nx * ny`` sites.
)")
      .def_static("permute", &LatticeGeometry::permute, py::arg("geometry"),
                  py::arg("path"), py::call_guard<py::gil_scoped_release>(), R"(
Relabel sites so new site i is original site ``path[i]``.

Args:
    geometry (LatticeGeometry): Source geometry.
    path (list[int]): Permutation of every original site index.

Returns:
    LatticeGeometry: Relabeled positions and integer coordinates, with unchanged periods.

Raises:
    ValueError: If path is not a valid permutation of all sites.
)")
      .def_static("data_type_name", &LatticeGeometry::data_type_name,
                  "Return the wire-format identifier ``lattice_geometry``.")
      .def("get_data_type_name", &LatticeGeometry::get_data_type_name,
           "Return the wire-format identifier ``lattice_geometry``.")
      .def(
          "__repr__",
          [](const LatticeGeometry& self) {
            return "<LatticeGeometry sites=" +
                   std::to_string(self.num_sites()) + " periodic_vectors=" +
                   std::to_string(self.periods() ? self.periods()->rows() : 0) +
                   ">";
          })
      .def("__str__", &LatticeGeometry::get_summary);

  bind_getter_as_property(
      geometry, "get_summary", &LatticeGeometry::get_summary,
      "Return a summary of site and periodic-vector counts.");

  geometry
      .def(
          "to_json",
          [](const LatticeGeometry& self) { return self.to_json().dump(); },
          py::call_guard<py::gil_scoped_release>(),
          "Serialize positions and optional periods to a JSON string.")
      .def_static(
          "from_json",
          [](const std::string& json_str) {
            return LatticeGeometry::from_json(nlohmann::json::parse(json_str));
          },
          py::arg("json_str"), py::call_guard<py::gil_scoped_release>(), R"(
Load geometry from a JSON string.

Args:
    json_str (str): Serialized positions and optional periods.

Returns:
    LatticeGeometry: Restored geometry.
)")
      .def(
          "to_file",
          [](const LatticeGeometry& self, const py::object& filename,
             const std::string& format_type) {
            const auto path = to_string_path(filename);
            py::gil_scoped_release release;
            self.to_file(path, format_type);
          },
          py::arg("filename"), py::arg("format_type"), R"(
Save geometry to a JSON or HDF5 file.

Args:
    filename (str | pathlib.Path): Output path.
    format_type (str): File format, either "json" or "hdf5".
)")
      .def_static(
          "from_file",
          [](const py::object& filename, const std::string& format_type) {
            const auto path = to_string_path(filename);
            py::gil_scoped_release release;
            return LatticeGeometry::from_file(path, format_type);
          },
          py::arg("filename"), py::arg("format_type"), R"(
Load geometry from a JSON or HDF5 file.

Args:
    filename (str | pathlib.Path): Input path.
    format_type (str): File format, either "json" or "hdf5".

Returns:
    LatticeGeometry: Restored geometry.
)")
      .def(
          "to_json_file",
          [](const LatticeGeometry& self, const py::object& filename) {
            const auto path = to_string_path(filename);
            py::gil_scoped_release release;
            self.to_json_file(path);
          },
          py::arg("filename"), R"(
Save geometry to a JSON file.

Args:
    filename (str | pathlib.Path): Output path.
)")
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            const auto path = to_string_path(filename);
            py::gil_scoped_release release;
            return LatticeGeometry::from_json_file(path);
          },
          py::arg("filename"), R"(
Load geometry from a JSON file.

Args:
    filename (str | pathlib.Path): Input path.

Returns:
    LatticeGeometry: Restored geometry.
)")
      .def(
          "to_hdf5_file",
          [](const LatticeGeometry& self, const py::object& filename) {
            const auto path = to_string_path(filename);
            py::gil_scoped_release release;
            self.to_hdf5_file(path);
          },
          py::arg("filename"), R"(
Save positions and optional periods as numeric HDF5 matrices.

Args:
    filename (str | pathlib.Path): Output path.
)")
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            const auto path = to_string_path(filename);
            py::gil_scoped_release release;
            return LatticeGeometry::from_hdf5_file(path);
          },
          py::arg("filename"), R"(
Load geometry from an HDF5 file.

Args:
    filename (str | pathlib.Path): Input path.

Returns:
    LatticeGeometry: Restored geometry.
)")
      .def(py::pickle(
          [](const LatticeGeometry& self) {
            py::gil_scoped_release release;
            return self.to_json().dump();
          },
          [](const std::string& json_str) {
            py::gil_scoped_release release;
            return LatticeGeometry::from_json(nlohmann::json::parse(json_str));
          }));
}

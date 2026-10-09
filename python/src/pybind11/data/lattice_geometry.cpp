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

  py::class_<LatticeGeometry, DataClass, py::smart_holder> geometry(
      m, "LatticeGeometry", R"(
Immutable two-dimensional geometry of a built-in lattice.

Stores the Cartesian site positions and optional periodic supercell vectors of
a factory lattice. :meth:`LatticeGraph.from_geometry` turns its distance shells
into labelled edges; the geometry stores no adjacency, flavors, or coloring.
)");

  geometry
      .def_property_readonly("num_sites", &LatticeGeometry::num_sites,
                             "Number of lattice sites.")
      .def_property_readonly(
          "positions",
          [](const LatticeGeometry& self) { return self.positions(); },
          R"(
Cartesian positions in site-index order.

Returns:
    numpy.ndarray: Independent copy of the ``(num_sites, 2)`` position matrix.
)")
      .def_property_readonly(
          "periods", [](const LatticeGeometry& self) { return self.periods(); },
          R"(
Periodic supercell vectors.

Returns:
    numpy.ndarray | None: Independent copy of the periodic-vector matrix, or None for an open geometry.
)")
      .def_static("chain", &LatticeGeometry::chain, py::arg("n"),
                  py::arg("periodic") = false, R"(
Create a unit-spaced chain with positions ``(i, 0)``.

Args:
    n (int): Positive number of sites.
    periodic (bool, optional): Include the supercell vector ``(n, 0)``. Defaults to False.

Returns:
    LatticeGeometry: Chain geometry, including periodic one- and two-site cells.
)")
      .def_static("square", &LatticeGeometry::square, py::arg("nx"),
                  py::arg("ny"), py::arg("periodic_x") = false,
                  py::arg("periodic_y") = false, R"(
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
                  py::arg("periodic_y") = false, R"(
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
                  py::arg("periodic_y") = false, R"(
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
                  py::arg("periodic_y") = false, R"(
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
                  py::arg("periodic_y") = false, R"(
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
          "Serialize the factory layout to a JSON string.")
      .def_static(
          "from_json",
          [](const std::string& json_str) {
            return LatticeGeometry::from_json(nlohmann::json::parse(json_str));
          },
          py::arg("json_str"), R"(
Load geometry from a JSON string.

Args:
    json_str (str): Serialized factory layout.

Returns:
    LatticeGeometry: Restored geometry.
)")
      .def(
          "to_file",
          [](const LatticeGeometry& self, const py::object& filename,
             const std::string& format_type) {
            self.to_file(to_string_path(filename), format_type);
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
            return LatticeGeometry::from_file(to_string_path(filename),
                                              format_type);
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
            self.to_json_file(to_string_path(filename));
          },
          py::arg("filename"), R"(
Save geometry to a JSON file.

Args:
    filename (str | pathlib.Path): Output path.
)")
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return LatticeGeometry::from_json_file(to_string_path(filename));
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
            self.to_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"), R"(
Save the factory layout to an HDF5 file.

Args:
    filename (str | pathlib.Path): Output path.
)")
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return LatticeGeometry::from_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"), R"(
Load geometry from an HDF5 file.

Args:
    filename (str | pathlib.Path): Input path.

Returns:
    LatticeGeometry: Restored geometry.
)")
      .def(py::pickle(
          [](const LatticeGeometry& self) { return self.to_json().dump(); },
          [](const std::string& json_str) {
            return LatticeGeometry::from_json(nlohmann::json::parse(json_str));
          }));
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "../path_utils.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;
using qdk::chemistry::python::utils::to_string_path;

void bind_symmetry_blocked_index_set(py::module& m) {
  py::class_<SymmetryBlockedIndexSet, DataClass, py::smart_holder>(
      m, "SymmetryBlockedIndexSet",
      "Immutable set of symmetry-blocked, sorted-unique integer indices, one "
      "list per admissible SymmetryLabel of a single SymmetryProduct.")
      .def(py::init<
               std::shared_ptr<const SymmetryProduct>,
               std::unordered_map<SymmetryLabel, std::size_t>,
               std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>>>(),
           py::arg("symmetries"), py::arg("extents"), py::arg("indices"),
           "Construct from a SymmetryProduct, per-label extents, and "
           "per-label index lists (each must be sorted, unique, and in range).")
      .def(
          "symmetries",
          [](const SymmetryBlockedIndexSet& self) {
            return std::const_pointer_cast<SymmetryProduct>(self.symmetries());
          },
          "The SymmetryProduct this index set is blocked under.")
      .def("extents", &SymmetryBlockedIndexSet::extents, "Per-label extents.")
      .def(
          "indices",
          [](const SymmetryBlockedIndexSet& self, const SymmetryLabel& label) {
            auto span = self.indices(label);
            return py::tuple(
                py::cast(std::vector<std::uint32_t>(span.begin(), span.end())));
          },
          py::arg("label"),
          "The sorted, unique indices stored for the given label, as an "
          "immutable tuple.")
      .def("has", &SymmetryBlockedIndexSet::has, py::arg("label"),
           "True iff indices are stored for the given label.")
      .def("labels", &SymmetryBlockedIndexSet::labels,
           "The labels for which indices are stored.")
      .def("get_data_type_name", &SymmetryBlockedIndexSet::get_data_type_name)
      .def("get_summary", &SymmetryBlockedIndexSet::get_summary)
      .def("__repr__", &SymmetryBlockedIndexSet::get_summary)
      .def(
          "to_file",
          [](const SymmetryBlockedIndexSet& self, const py::object& filename,
             const std::string& type) {
            self.to_file(to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"))
      .def(
          "to_json_file",
          [](const SymmetryBlockedIndexSet& self, const py::object& filename) {
            self.to_json_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def(
          "to_hdf5_file",
          [](const SymmetryBlockedIndexSet& self, const py::object& filename) {
            self.to_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_file",
          [](const py::object& filename, const std::string& type) {
            return SymmetryBlockedIndexSet::from_file(to_string_path(filename),
                                                      type);
          },
          py::arg("filename"), py::arg("type"))
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return SymmetryBlockedIndexSet::from_json_file(
                to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return SymmetryBlockedIndexSet::from_hdf5_file(
                to_string_path(filename));
          },
          py::arg("filename"));
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <memory>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_scalar.hpp>
#include <string>
#include <utility>
#include <vector>

#include "../path_utils.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;
using qdk::chemistry::python::utils::to_string_path;

namespace {

// Bind one Scalar instantiation of SymmetryBlockedScalar under `name`.
template <class Scalar>
void bind_sbscalar_instance(py::module& m, const char* name) {
  using SBS = SymmetryBlockedScalar<Scalar>;
  using Labels = typename SBS::Labels;
  using SymmetriesArray = typename SBS::SymmetriesArray;
  using ExtentsArray = typename SBS::ExtentsArray;

  py::class_<SBS, DataClass, py::smart_holder>(
      m, name,
      "Immutable symmetry-blocked scalar. Stores one scalar value per symmetry "
      "sector as a map from a per-slot SymmetryLabel to a scalar (e.g. an "
      "electron count per spin channel).")
      .def(py::init([](SymmetriesArray symmetries, ExtentsArray extents,
                       std::vector<std::pair<Labels, Scalar>> blocks) {
             typename SBS::BlockMap block_map;
             for (auto& [labels, value] : blocks) {
               block_map.emplace(labels, std::make_shared<const Scalar>(value));
             }
             return std::make_shared<SBS>(std::move(symmetries),
                                          std::move(extents),
                                          std::move(block_map));
           }),
           py::arg("symmetries"), py::arg("extents"), py::arg("blocks"),
           "Construct from the per-slot symmetry, per-slot extents, and a list "
           "of (labels, value) pairs.")
      .def(
          "symmetries",
          [](const SBS& self) {
            std::vector<std::shared_ptr<SymmetryProduct>> out;
            out.reserve(self.symmetries().size());
            for (const auto& sym : self.symmetries()) {
              out.push_back(std::const_pointer_cast<SymmetryProduct>(sym));
            }
            return out;
          },
          "Per-slot SymmetryProduct instances.")
      .def("extents", &SBS::extents, "Per-slot per-label extents.")
      .def("has_block", &SBS::has_block, py::arg("labels"),
           "True iff a block is stored for the given per-slot labels.")
      .def("value", &SBS::value, py::arg("label"),
           "The scalar value stored for the given symmetry label.")
      .def("num_blocks", &SBS::num_blocks,
           "Total number of stored blocks (including aliases).")
      .def("get_data_type_name", &SBS::get_data_type_name)
      .def("get_summary", &SBS::get_summary)
      .def("__repr__", &SBS::get_summary)
      .def(
          "to_file",
          [](const SBS& self, const py::object& filename,
             const std::string& type) {
            self.to_file(to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"))
      .def(
          "to_json_file",
          [](const SBS& self, const py::object& filename) {
            self.to_json_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def(
          "to_hdf5_file",
          [](const SBS& self, const py::object& filename) {
            self.to_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_file",
          [](const py::object& filename, const std::string& type) {
            return SBS::from_file(to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"))
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return SBS::from_json_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return SBS::from_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"));
}

}  // namespace

void bind_symmetry_blocked_scalar(py::module& m) {
  bind_sbscalar_instance<std::size_t>(m, "SymmetryBlockedScalarCount");
}

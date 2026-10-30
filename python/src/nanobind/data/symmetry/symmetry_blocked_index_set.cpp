// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <string>
#include <unordered_map>
#include <vector>

#include "../path_utils.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::data;
using qdk::chemistry::python::utils::to_string_path;

void bind_symmetry_blocked_index_set(nb::module_& m) {
  nb::class_<SymmetryBlockedIndexSet, DataClass>(
      m, "SymmetryBlockedIndexSet",
      "Immutable set of symmetry-blocked, sorted-unique integer indices, one "
      "list per admissible SymmetryLabel of a single SymmetryProduct.")
      .def(nb::init<
               std::shared_ptr<const SymmetryProduct>,
               std::unordered_map<SymmetryLabel, std::size_t>,
               std::unordered_map<SymmetryLabel, std::vector<std::uint32_t>>>(),
           nb::arg("symmetries"), nb::arg("extents"), nb::arg("indices"),
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
            return nb::tuple(
                nb::cast(std::vector<std::uint32_t>(span.begin(), span.end())));
          },
          nb::arg("label"),
          "The sorted, unique indices stored for the given label, as an "
          "immutable tuple.")
      .def("has", &SymmetryBlockedIndexSet::has, nb::arg("label"),
           "True iff indices are stored for the given label.")
      .def("labels", &SymmetryBlockedIndexSet::labels,
           "The labels for which indices are stored.")
      .def_static("data_type_name", &SymmetryBlockedIndexSet::data_type_name,
                  R"(
    Return the wire-format identifier for symmetry-blocked index sets.

    Returns:
        str: ``"symmetry_blocked_index_set"``

    )")
      .def("get_data_type_name", &SymmetryBlockedIndexSet::get_data_type_name)
      .def("get_summary", &SymmetryBlockedIndexSet::get_summary)
      .def("__repr__", &SymmetryBlockedIndexSet::get_summary)
      .def(
          "to_file",
          [](const SymmetryBlockedIndexSet& self, const nb::object& filename,
             const std::string& type) {
            self.to_file(to_string_path(filename), type);
          },
          nb::arg("filename"), nb::arg("type"))
      .def(
          "to_json_file",
          [](const SymmetryBlockedIndexSet& self, const nb::object& filename) {
            self.to_json_file(to_string_path(filename));
          },
          nb::arg("filename"))
      .def(
          "to_hdf5_file",
          [](const SymmetryBlockedIndexSet& self, const nb::object& filename) {
            self.to_hdf5_file(to_string_path(filename));
          },
          nb::arg("filename"))
      .def_static(
          "from_file",
          [](const nb::object& filename, const std::string& type) {
            return SymmetryBlockedIndexSet::from_file(to_string_path(filename),
                                                      type);
          },
          nb::arg("filename"), nb::arg("type"))
      .def_static(
          "from_json_file",
          [](const nb::object& filename) {
            return SymmetryBlockedIndexSet::from_json_file(
                to_string_path(filename));
          },
          nb::arg("filename"))
      .def_static(
          "from_hdf5_file",
          [](const nb::object& filename) {
            return SymmetryBlockedIndexSet::from_hdf5_file(
                to_string_path(filename));
          },
          nb::arg("filename"));

  m.def(
      "spin_channel_indices",
      [](const std::shared_ptr<const SymmetryBlockedIndexSet>& index_set,
         const std::shared_ptr<const SpinValue>& channel) {
        return spin_channel_indices(index_set, channel);
      },
      nb::arg("index_set").none(), nb::arg("channel"),
      "Indices for a single spin channel of a SymmetryBlockedIndexSet as a "
      "flat list. Pass the channel explicitly, e.g. axes.alpha() or "
      "axes.beta(); for a spin-free set the sole trivial-label segment serves "
      "every channel. Returns an empty list when 'index_set' is None.");
}

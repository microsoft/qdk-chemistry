// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <cstddef>
#include <map>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_sparse_map.hpp>
#include <string>

namespace py = pybind11;
using namespace qdk::chemistry::data;

void bind_symmetry_blocked_sparse_map(py::module& m) {
  using SBSM = SymmetryBlockedSparseMap<4>;
  using Block = SparseMapBlock<4>;

  py::class_<SBSM, DataClass, std::shared_ptr<SBSM>>(
      m, "SymmetryBlockedSparseMapRank4",
      "Immutable rank-4 symmetry-blocked sparse map (double-valued). Each "
      "block is a dict-like map from a per-slot local-index tuple to a "
      "scalar value.")
      .def(
          "symmetries",
          [](const SBSM& self) {
            std::vector<std::shared_ptr<SymmetryProduct>> out;
            out.reserve(self.symmetries().size());
            for (const auto& sym : self.symmetries()) {
              out.push_back(std::const_pointer_cast<SymmetryProduct>(sym));
            }
            return out;
          },
          "Per-slot SymmetryProduct instances.")
      .def("extents", &SBSM::extents, "Per-slot per-label extents.")
      .def("has_block", &SBSM::has_block, py::arg("labels"),
           "True iff a block is stored for the given per-slot labels.")
      .def(
          "block",
          [](const SBSM& self, const SBSM::Labels& labels) {
            // Copy out the sparse map as a Python dict keyed by index tuples.
            const Block& block = self.block(labels);
            py::dict out;
            for (const auto& [idx, val] : block) {
              out[py::make_tuple(idx[0], idx[1], idx[2], idx[3])] = val;
            }
            return out;
          },
          py::arg("labels"),
          "The sparse block stored for the given per-slot labels, as a "
          "dict[tuple[int,int,int,int], float].")
      .def("num_blocks", &SBSM::num_blocks,
           "Total number of stored blocks (including aliases).")
      .def("num_entries", &SBSM::num_entries,
           "Total number of stored sparse entries across all blocks.")
      .def(
          "get",
          [](const SBSM& self, const SBSM::Labels& labels,
             const SBSM::IndexTuple& idx) { return self.get(labels, idx); },
          py::arg("labels"), py::arg("idx"),
          "Single-entry lookup; returns 0.0 if the entry is absent.")
      .def("get_data_type_name", &SBSM::get_data_type_name)
      .def("get_summary", &SBSM::get_summary)
      .def("__repr__", &SBSM::get_summary);
}

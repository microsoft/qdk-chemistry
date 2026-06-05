// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <complex>
#include <cstddef>
#include <memory>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <string>
#include <utility>
#include <vector>

#include "../path_utils.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;
using qdk::chemistry::python::utils::to_string_path;

namespace {

// Bind one (Rank, Scalar) instantiation of SymmetryBlockedTensor under `name`.
template <std::size_t Rank, class Scalar>
void bind_sbt_instance(py::module& m, const char* name) {
  using SBT = SymmetryBlockedTensor<Rank, Scalar>;
  using Labels = typename SBT::Labels;
  using SymmetriesArray = typename SBT::SymmetriesArray;
  using ExtentsArray = typename SBT::ExtentsArray;
  using BlockTensor = Tensor<Rank, Scalar>;

  py::class_<SBT, DataClass, py::smart_holder>(
      m, name,
      "Immutable symmetry-blocked dense tensor. Stores the non-zero symmetry "
      "sectors of a tensor as a map from per-slot SymmetryLabel tuples to "
      "dense numpy blocks.")
      .def(py::init([](SymmetriesArray symmetries, ExtentsArray extents,
                       std::vector<std::pair<Labels, BlockTensor>> blocks) {
             typename SBT::BlockMap block_map;
             for (auto& [labels, block] : blocks) {
               block_map.emplace(labels, std::make_shared<const BlockTensor>(
                                             std::move(block)));
             }
             return std::make_shared<SBT>(std::move(symmetries),
                                          std::move(extents),
                                          std::move(block_map));
           }),
           py::arg("symmetries"), py::arg("extents"), py::arg("blocks"),
           "Construct from per-slot symmetries, per-slot extents, and a list "
           "of (labels, block) pairs. Orbit-equivalent sectors that share the "
           "same supplied block are auto-aliased.")
      .def(
          "symmetries",
          [](const SBT& self) {
            std::vector<std::shared_ptr<SymmetryProduct>> out;
            out.reserve(self.symmetries().size());
            for (const auto& sym : self.symmetries()) {
              out.push_back(std::const_pointer_cast<SymmetryProduct>(sym));
            }
            return out;
          },
          "Per-slot SymmetryProduct instances.")
      .def("extents", &SBT::extents, "Per-slot per-label extents.")
      .def("has_block", &SBT::has_block, py::arg("labels"),
           "True iff a block is stored for the given per-slot labels.")
      .def(
          "block",
          [](const SBT& self, const Labels& labels) -> BlockTensor {
            return self.block(labels);
          },
          py::arg("labels"),
          "The dense numpy block stored for the given per-slot labels.")
      .def("num_blocks", &SBT::num_blocks,
           "Total number of stored blocks (including aliases).")
      .def("get_data_type_name", &SBT::get_data_type_name)
      .def("get_summary", &SBT::get_summary)
      .def("__repr__", &SBT::get_summary)
      .def(
          "to_file",
          [](const SBT& self, const py::object& filename,
             const std::string& type) {
            self.to_file(to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"))
      .def(
          "to_json_file",
          [](const SBT& self, const py::object& filename) {
            self.to_json_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def(
          "to_hdf5_file",
          [](const SBT& self, const py::object& filename) {
            self.to_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_file",
          [](const py::object& filename, const std::string& type) {
            return SBT::from_file(to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"))
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return SBT::from_json_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return SBT::from_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"));
}

}  // namespace

void bind_symmetry_blocked_tensor(py::module& m) {
  bind_sbt_instance<1, double>(m, "SymmetryBlockedTensorRank1");
  bind_sbt_instance<1, std::complex<double>>(
      m, "SymmetryBlockedTensorRank1Complex");
  bind_sbt_instance<2, double>(m, "SymmetryBlockedTensorRank2");
  bind_sbt_instance<2, std::complex<double>>(
      m, "SymmetryBlockedTensorRank2Complex");
  bind_sbt_instance<3, double>(m, "SymmetryBlockedTensorRank3");
  bind_sbt_instance<3, std::complex<double>>(
      m, "SymmetryBlockedTensorRank3Complex");
  bind_sbt_instance<4, double>(m, "SymmetryBlockedTensorRank4");
  bind_sbt_instance<4, std::complex<double>>(
      m, "SymmetryBlockedTensorRank4Complex");
}

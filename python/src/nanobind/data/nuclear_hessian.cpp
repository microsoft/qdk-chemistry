// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/nuclear_hessian.hpp>

#include "eigen.hpp"
#include "path_utils.hpp"
#include "stl.hpp"

namespace nb = nanobind;
using namespace qdk::chemistry::data;

namespace {

void nuclear_hessian_to_file(NuclearHessian& self, const nb::object& filename,
                             const std::string& format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

void nuclear_hessian_to_json_file(NuclearHessian& self,
                                  const nb::object& filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

void nuclear_hessian_to_hdf5_file(NuclearHessian& self,
                                  const nb::object& filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<NuclearHessian> nuclear_hessian_from_file(
    const nb::object& filename, const std::string& format_type) {
  return NuclearHessian::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

std::shared_ptr<NuclearHessian> nuclear_hessian_from_json_file(
    const nb::object& filename) {
  return NuclearHessian::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<NuclearHessian> nuclear_hessian_from_hdf5_file(
    const nb::object& filename) {
  return NuclearHessian::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

}  // namespace

void bind_nuclear_hessian(nb::module_& m) {
  nb::class_<NuclearHessian, DataClass>(m, "NuclearHessian",
                                        R"(
Nuclear second derivatives for a molecular structure.

The matrix is ordered atom-major by x, y, z components and records the
geometry for which the Hessian was computed.
)")
      .def(nb::init<std::shared_ptr<Structure>, const Eigen::MatrixXd&>(),
           R"(
Create a nuclear Hessian for a structure.

Args:
    structure: Molecular structure used to compute the Hessian.
    matrix: Square ``3N x 3N`` Hessian matrix in Hartree/Bohr^2.
)",
           nb::arg("structure"), nb::arg("matrix"))
      .def_prop_ro("structure", &NuclearHessian::get_structure,
                   R"(Molecular structure associated with the Hessian.)")
      .def_prop_ro(
          "matrix",
          [](const NuclearHessian& hessian) -> const Eigen::MatrixXd& {
            return hessian.get_matrix();
          },
          nb::rv_policy::reference_internal,
          R"(Hessian matrix in Hartree/Bohr^2.)")
      .def("get_structure", &NuclearHessian::get_structure,
           R"(Return the molecular structure associated with the Hessian.)")
      .def("get_matrix", &NuclearHessian::get_matrix,
           nb::rv_policy::reference_internal,
           R"(Return the Hessian matrix in Hartree/Bohr^2.)")
      .def("get_atom_pair_block", &NuclearHessian::get_atom_pair_block,
           nb::arg("row_atom_index"), nb::arg("column_atom_index"),
           R"(Return the 3 by 3 Hessian block coupling two atoms.)")
      .def_static("data_type_name", &NuclearHessian::data_type_name, R"(
Return the wire-format identifier for nuclear Hessians.

Returns:
    str: ``"nuclear_hessian"``

)")
      .def("get_data_type_name", &NuclearHessian::get_data_type_name,
           R"(Return the serialized data type name.)")
      .def("get_summary", &NuclearHessian::get_summary,
           R"(Return a short human-readable summary.)")
      .def("to_file", nuclear_hessian_to_file, nb::arg("filename"),
           nb::arg("format_type"),
           R"(Save the Hessian to a JSON or HDF5 file.)")
      .def(
          "to_json",
          [](const NuclearHessian& self) { return self.to_json().dump(2); },
          R"(Serialize the Hessian to a JSON string.)")
      .def("to_json_file", nuclear_hessian_to_json_file, nb::arg("filename"),
           R"(Save the Hessian to a JSON file.)")
      .def("to_hdf5_file", nuclear_hessian_to_hdf5_file, nb::arg("filename"),
           R"(Save the Hessian to an HDF5 file.)")
      .def_static("from_file", nuclear_hessian_from_file, nb::arg("filename"),
                  nb::arg("format_type"),
                  R"(Load a Hessian from a JSON or HDF5 file.)")
      .def_static(
          "from_json",
          [](const std::string& json_str) {
            return NuclearHessian::from_json(nlohmann::json::parse(json_str));
          },
          nb::arg("json_str"), R"(Load a Hessian from a JSON string.)")
      .def_static("from_json_file", nuclear_hessian_from_json_file,
                  nb::arg("filename"), R"(Load a Hessian from a JSON file.)")
      .def_static("from_hdf5_file", nuclear_hessian_from_hdf5_file,
                  nb::arg("filename"), R"(Load a Hessian from an HDF5 file.)")
      .def("__repr__",
           [](const NuclearHessian& hessian) { return hessian.get_summary(); });
}

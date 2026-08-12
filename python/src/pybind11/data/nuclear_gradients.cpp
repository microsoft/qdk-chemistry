// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/nuclear_gradients.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>

#include "path_utils.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;

namespace {

void nuclear_gradients_to_file(NuclearGradients& self,
                               const py::object& filename,
                               const std::string& format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

void nuclear_gradients_to_json_file(NuclearGradients& self,
                                    const py::object& filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

void nuclear_gradients_to_hdf5_file(NuclearGradients& self,
                                    const py::object& filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<NuclearGradients> nuclear_gradients_from_file(
    const py::object& filename, const std::string& format_type) {
  return NuclearGradients::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

std::shared_ptr<NuclearGradients> nuclear_gradients_from_json_file(
    const py::object& filename) {
  return NuclearGradients::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<NuclearGradients> nuclear_gradients_from_hdf5_file(
    const py::object& filename) {
  return NuclearGradients::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

}  // namespace

void bind_nuclear_gradients(py::module& m) {
  py::class_<NuclearGradients, DataClass, py::smart_holder>(m,
                                                            "NuclearGradients",
                                                            R"(
Nuclear energy gradients for a molecular structure.

Values are stored as an atom-major vector with x, y, z components for each
atom. The associated structure records the geometry for which the gradients
were computed.
)")
      .def(py::init<std::shared_ptr<Structure>, const Eigen::VectorXd&>(),
           R"(
Create nuclear gradients for a structure.

Args:
    structure: Molecular structure used to compute the gradients.
    values: Atom-major gradient vector with length ``3 * num_atoms``.
)",
           py::arg("structure"), py::arg("values"))
      .def_property_readonly(
          "structure", &NuclearGradients::get_structure,
          R"(Molecular structure associated with the gradients.)")
      .def_property_readonly(
          "values",
          [](const NuclearGradients& gradients) -> const Eigen::VectorXd& {
            return gradients.get_values();
          },
          py::return_value_policy::reference_internal,
          R"(Atom-major gradient vector in Hartree/Bohr.)")
      .def("get_structure", &NuclearGradients::get_structure,
           R"(Return the molecular structure associated with the gradients.)")
      .def("get_values", &NuclearGradients::get_values,
           py::return_value_policy::reference_internal,
           R"(Return the atom-major gradient vector in Hartree/Bohr.)")
      .def("get_atom_gradient", &NuclearGradients::get_atom_gradient,
           py::arg("atom_index"),
           R"(Return the x, y, z gradient components for one atom.)")
      .def("as_matrix", &NuclearGradients::as_matrix,
           R"(Return gradients as an ``(num_atoms, 3)`` matrix.)")
      .def("get_data_type_name", &NuclearGradients::get_data_type_name,
           R"(Return the serialized data type name.)")
      .def("get_summary", &NuclearGradients::get_summary,
           R"(Return a short human-readable summary.)")
      .def("to_file", nuclear_gradients_to_file, py::arg("filename"),
           py::arg("format_type"), R"(Save gradients to a JSON or HDF5 file.)")
      .def(
          "to_json",
          [](const NuclearGradients& self) { return self.to_json().dump(2); },
          R"(Serialize gradients to a JSON string.)")
      .def("to_json_file", nuclear_gradients_to_json_file, py::arg("filename"),
           R"(Save gradients to a JSON file.)")
      .def("to_hdf5_file", nuclear_gradients_to_hdf5_file, py::arg("filename"),
           R"(Save gradients to an HDF5 file.)")
      .def_static("from_file", nuclear_gradients_from_file, py::arg("filename"),
                  py::arg("format_type"),
                  R"(Load gradients from a JSON or HDF5 file.)")
      .def_static(
          "from_json",
          [](const std::string& json_str) {
            return NuclearGradients::from_json(nlohmann::json::parse(json_str));
          },
          py::arg("json_str"), R"(Load gradients from a JSON string.)")
      .def_static("from_json_file", nuclear_gradients_from_json_file,
                  py::arg("filename"), R"(Load gradients from a JSON file.)")
      .def_static("from_hdf5_file", nuclear_gradients_from_hdf5_file,
                  py::arg("filename"), R"(Load gradients from an HDF5 file.)")
      .def("__repr__", [](const NuclearGradients& gradients) {
        return gradients.get_summary();
      });
}

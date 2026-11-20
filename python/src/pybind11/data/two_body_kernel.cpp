// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/two_body_kernel.hpp>

#include "path_utils.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;
using qdk::chemistry::python::utils::to_string_path;

void bind_two_body_kernel(py::module& m) {
  py::class_<TwoBodyKernel, DataClass, py::smart_holder>(m, "TwoBodyKernel",
                                                         R"(
Interaction kernel :math:`g(r_{12})` of a two-electron operator.

The integrals of a kernel are
:math:`(mn|g|ls) = \int \phi_m(1) \phi_n(1) g(r_{12}) \phi_l(2) \phi_s(2)`.
Kernels are immutable; the loaders return the kernel of the serialized type.
)")
      .def_static("data_type_name", &TwoBodyKernel::data_type_name, R"(
Return the wire-format identifier for two-body kernels.

Returns:
    str: ``"two_body_kernel"``
)")
      .def("get_data_type_name", &TwoBodyKernel::get_data_type_name,
           R"(Return the serialized data type name.)")
      .def("get_summary", &TwoBodyKernel::get_summary,
           R"(Return a short human-readable summary.)")
      .def(
          "to_file",
          [](const TwoBodyKernel& self, const py::object& filename,
             const std::string& format_type) {
            self.to_file(to_string_path(filename), format_type);
          },
          py::arg("filename"), py::arg("format_type"),
          R"(Save the kernel to a JSON or HDF5 file.)")
      .def(
          "to_json",
          [](const TwoBodyKernel& self) { return self.to_json().dump(2); },
          R"(Serialize the kernel to a JSON string.)")
      .def(
          "to_json_file",
          [](const TwoBodyKernel& self, const py::object& filename) {
            self.to_json_file(to_string_path(filename));
          },
          py::arg("filename"), R"(Save the kernel to a JSON file.)")
      .def(
          "to_hdf5_file",
          [](const TwoBodyKernel& self, const py::object& filename) {
            self.to_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"), R"(Save the kernel to an HDF5 file.)")
      .def_static(
          "from_file",
          [](const py::object& filename, const std::string& format_type) {
            return TwoBodyKernel::from_file(to_string_path(filename),
                                            format_type);
          },
          py::arg("filename"), py::arg("format_type"),
          R"(Load a kernel from a JSON or HDF5 file.)")
      .def_static(
          "from_json",
          [](const std::string& json_str) {
            return TwoBodyKernel::from_json(nlohmann::json::parse(json_str));
          },
          py::arg("json_str"), R"(Load a kernel from a JSON string.)")
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return TwoBodyKernel::from_json_file(to_string_path(filename));
          },
          py::arg("filename"), R"(Load a kernel from a JSON file.)")
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return TwoBodyKernel::from_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"), R"(Load a kernel from an HDF5 file.)")
      .def("__repr__",
           [](const TwoBodyKernel& kernel) { return kernel.get_summary(); });

  py::class_<CoulombKernel, TwoBodyKernel, py::smart_holder>(
      m, "CoulombKernel", R"(The Coulomb kernel :math:`g(r) = 1 / r`.)")
      .def(py::init<>(), R"(Construct the Coulomb kernel.)");

  py::class_<ErfCoulombKernel, TwoBodyKernel, py::smart_holder>(
      m, "ErfCoulombKernel",
      R"(The long-range Coulomb kernel :math:`g(r) = \operatorname{erf}(\omega r) / r`.)")
      .def(py::init<double>(), py::arg("omega"), R"(
Construct the kernel of a range-separation parameter.

Args:
    omega (float): Range-separation parameter in inverse bohr.

Raises:
    ValueError: If ``omega`` is not finite and positive.
)")
      .def_property_readonly("omega", &ErfCoulombKernel::get_omega,
                             R"(Range-separation parameter in inverse bohr.)")
      .def("get_omega", &ErfCoulombKernel::get_omega,
           R"(Return the range-separation parameter in inverse bohr.)");

  py::class_<ErfcCoulombKernel, TwoBodyKernel, py::smart_holder>(
      m, "ErfcCoulombKernel",
      R"(The short-range Coulomb kernel :math:`g(r) = \operatorname{erfc}(\omega r) / r`.)")
      .def(py::init<double>(), py::arg("omega"), R"(
Construct the kernel of a range-separation parameter.

Args:
    omega (float): Range-separation parameter in inverse bohr.

Raises:
    ValueError: If ``omega`` is not finite and positive.
)")
      .def_property_readonly("omega", &ErfcCoulombKernel::get_omega,
                             R"(Range-separation parameter in inverse bohr.)")
      .def("get_omega", &ErfcCoulombKernel::get_omega,
           R"(Return the range-separation parameter in inverse bohr.)");
}

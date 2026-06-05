// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <string>
#include <vector>

#include "../path_utils.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;
using qdk::chemistry::python::utils::to_string_path;

void bind_symmetry(py::module& symmetry) {
  py::enum_<AxisName>(symmetry, "AxisName", "Symmetry axis identifier.")
      .value("Spin", AxisName::Spin);

  symmetry.def(
      "axis_name_to_string", [](AxisName axis) { return to_string(axis); },
      py::arg("axis"), "Human-readable name for an AxisName.");

  py::class_<SymmetryAxisValue, std::shared_ptr<SymmetryAxisValue>>(
      symmetry, "SymmetryAxisValue",
      "Abstract value carried by a single symmetry axis.")
      .def("axis", &SymmetryAxisValue::axis, "The axis this value belongs to.")
      .def("__eq__",
           [](const SymmetryAxisValue& self, const SymmetryAxisValue& other) {
             return self.equals(other);
           })
      .def("__hash__", &SymmetryAxisValue::hash);

  py::class_<SpinValue, SymmetryAxisValue, std::shared_ptr<SpinValue>>(
      symmetry, "SpinValue",
      "Concrete spin-1/2 axis value. The stored value is 2*Ms: +1 for an "
      "alpha label, -1 for a beta label.")
      .def(py::init<int>(), py::arg("two_ms"),
           "Construct from 2*Ms (e.g. +1 for alpha, -1 for beta).")
      .def("value", &SpinValue::value, "The stored 2*Ms value.");

  py::class_<SymmetryAxis, DataClass, py::smart_holder>(
      symmetry, "SymmetryAxis",
      "One named symmetry partition the basis is blocked under.")
      .def(py::init<AxisName,
                    std::vector<std::shared_ptr<const SymmetryAxisValue>>,
                    bool>(),
           py::arg("name"), py::arg("labels"), py::arg("equivalent"))
      .def("name", &SymmetryAxis::name)
      .def("labels",
           [](const SymmetryAxis& self) {
             std::vector<std::shared_ptr<SymmetryAxisValue>> out;
             out.reserve(self.labels().size());
             for (const auto& v : self.labels()) {
               out.push_back(std::const_pointer_cast<SymmetryAxisValue>(v));
             }
             return out;
           })
      .def("equivalent", &SymmetryAxis::equivalent)
      .def("admits", &SymmetryAxis::admits, py::arg("value"),
           "True iff value is one of this axis's admissible labels.")
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &SymmetryAxis::hash)
      .def("get_data_type_name", &SymmetryAxis::get_data_type_name)
      .def("get_summary", &SymmetryAxis::get_summary)
      .def("__repr__", &SymmetryAxis::get_summary)
      .def(
          "to_file",
          [](const SymmetryAxis& self, const py::object& filename,
             const std::string& type) {
            self.to_file(to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"))
      .def(
          "to_json_file",
          [](const SymmetryAxis& self, const py::object& filename) {
            self.to_json_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def(
          "to_hdf5_file",
          [](const SymmetryAxis& self, const py::object& filename) {
            self.to_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_file",
          [](const py::object& filename, const std::string& type) {
            return SymmetryAxis::from_file(to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"))
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return SymmetryAxis::from_json_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return SymmetryAxis::from_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"));

  py::class_<SymmetryProduct, DataClass, py::smart_holder>(
      symmetry, "SymmetryProduct",
      "A SymmetryProduct: the ordered set of axes a basis is blocked "
      "under, together with their admissible labels and equivalence flags.")
      .def(py::init<std::vector<SymmetryAxis>>(), py::arg("axes"))
      .def("axes", &SymmetryProduct::axes)
      .def("has_axis", &SymmetryProduct::has_axis, py::arg("name"),
           "True iff an axis with the given name exists in this "
           "SymmetryProduct.")
      .def("axis", &SymmetryProduct::axis, py::arg("name"),
           py::return_value_policy::reference_internal,
           "Access the axis with the given name.")
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &SymmetryProduct::hash)
      .def("get_data_type_name", &SymmetryProduct::get_data_type_name)
      .def("get_summary", &SymmetryProduct::get_summary)
      .def("__repr__", &SymmetryProduct::get_summary)
      .def(
          "to_file",
          [](const SymmetryProduct& self, const py::object& filename,
             const std::string& type) {
            self.to_file(to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"))
      .def(
          "to_json_file",
          [](const SymmetryProduct& self, const py::object& filename) {
            self.to_json_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def(
          "to_hdf5_file",
          [](const SymmetryProduct& self, const py::object& filename) {
            self.to_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_file",
          [](const py::object& filename, const std::string& type) {
            return SymmetryProduct::from_file(to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"))
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return SymmetryProduct::from_json_file(to_string_path(filename));
          },
          py::arg("filename"))
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return SymmetryProduct::from_hdf5_file(to_string_path(filename));
          },
          py::arg("filename"));

  py::class_<SymmetryLabel>(
      symmetry, "SymmetryLabel",
      "A composite addressing key: one SymmetryAxisValue per axis.")
      .def(py::init<std::vector<std::shared_ptr<const SymmetryAxisValue>>>(),
           py::arg("values"))
      .def(
          "get",
          [](const SymmetryLabel& self, AxisName axis) {
            return std::const_pointer_cast<SymmetryAxisValue>(self.get(axis));
          },
          py::arg("axis"), "The value carried for the given axis.")
      .def("has", &SymmetryLabel::has, py::arg("axis"),
           "True iff this label carries a value for the given axis.")
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &SymmetryLabel::hash);

  // axes factory namespace, exposed as a Python submodule.
  auto axes = symmetry.def_submodule(
      "axes", "Factory helpers for constructing common symmetry axes.");
  axes.def(
      "spin",
      [](int two_s, bool equivalent) { return axes::spin(two_s, equivalent); },
      py::arg("two_s"), py::arg("equivalent") = true,
      "Build a spin axis carrying the alpha and beta labels.");
  axes.def(
      "alpha",
      []() { return std::const_pointer_cast<SpinValue>(axes::alpha()); },
      "Interned shared alpha spin value (2*Ms = +1).");
  axes.def(
      "beta", []() { return std::const_pointer_cast<SpinValue>(axes::beta()); },
      "Interned shared beta spin value (2*Ms = -1).");
  axes.def(
      "spin_value",
      [](int two_ms) {
        return std::const_pointer_cast<SpinValue>(axes::spin_value(two_ms));
      },
      py::arg("two_ms"), "Construct a spin value carrying 2*Ms = two_ms.");
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <qdk/chemistry/data/errors.hpp>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <vector>

namespace py = pybind11;
using namespace qdk::chemistry::data;

// Register the §9 typed-error hierarchy so C++ exceptions surface as Python
// exceptions that preserve the base relationships (everything derives from
// QdkError, which derives from RuntimeError).
void bind_symmetry_errors(py::module& m) {
  static py::exception<QdkError> qdk_error(m, "QdkError", PyExc_RuntimeError);

  // BasisSet errors
  static py::exception<BasisSetError> basis_set_error(m, "BasisSetError",
                                                      qdk_error.ptr());
  static py::exception<BasisSetSpinExtentMismatchError> basis_set_spin_extent(
      m, "BasisSetSpinExtentMismatchError", basis_set_error.ptr());

  // SingleParticleBasis errors
  static py::exception<SingleParticleBasisError> spb_error(
      m, "SingleParticleBasisError", qdk_error.ptr());
  static py::exception<ModelOrbitalsNoBasisSetError> model_no_basis(
      m, "ModelOrbitalsNoBasisSetError", spb_error.ptr());
  static py::exception<ModelOrbitalsProjectionError> model_proj(
      m, "ModelOrbitalsProjectionError", spb_error.ptr());
  static py::exception<OrbitalSpacePartitioningDisjointnessError>
      partition_disjoint(m, "OrbitalSpacePartitioningDisjointnessError",
                         spb_error.ptr());
  static py::exception<IndexSetOutOfRangeError> index_oor(
      m, "IndexSetOutOfRangeError", spb_error.ptr());
  static py::exception<IndexSetNotSortedUniqueError> index_unsorted(
      m, "IndexSetNotSortedUniqueError", spb_error.ptr());

  // Symmetry vocabulary errors
  static py::exception<SymmetryError> symmetry_error(m, "SymmetryError",
                                                     qdk_error.ptr());
  static py::exception<SymmetryIncompatibleError> symmetry_incompatible(
      m, "SymmetryIncompatibleError", symmetry_error.ptr());
  static py::exception<SymmetryConditionError> symmetry_condition(
      m, "SymmetryConditionError", symmetry_error.ptr());
  static py::exception<SymmetryProjectionError> symmetry_projection(
      m, "SymmetryProjectionError", symmetry_error.ptr());

  // SymmetryBlockedTensor errors
  static py::exception<SymmetryBlockedTensorError> sbt_error(
      m, "SymmetryBlockedTensorError", qdk_error.ptr());
  static py::exception<BlockExtentMismatchError> block_extent(
      m, "BlockExtentMismatchError", sbt_error.ptr());
  static py::exception<BlockAliasMismatchError> block_alias(
      m, "BlockAliasMismatchError", sbt_error.ptr());
  static py::exception<BlockLabelInvalidError> block_label(
      m, "BlockLabelInvalidError", sbt_error.ptr());

  // Dispatch C++ exceptions to the most-derived registered Python type.
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const BasisSetSpinExtentMismatchError& e) {
      py::set_error(basis_set_spin_extent, e.what());
    } catch (const BasisSetError& e) {
      py::set_error(basis_set_error, e.what());
    } catch (const ModelOrbitalsNoBasisSetError& e) {
      py::set_error(model_no_basis, e.what());
    } catch (const ModelOrbitalsProjectionError& e) {
      py::set_error(model_proj, e.what());
    } catch (const OrbitalSpacePartitioningDisjointnessError& e) {
      py::set_error(partition_disjoint, e.what());
    } catch (const IndexSetOutOfRangeError& e) {
      py::set_error(index_oor, e.what());
    } catch (const IndexSetNotSortedUniqueError& e) {
      py::set_error(index_unsorted, e.what());
    } catch (const SingleParticleBasisError& e) {
      py::set_error(spb_error, e.what());
    } catch (const SymmetryIncompatibleError& e) {
      py::set_error(symmetry_incompatible, e.what());
    } catch (const SymmetryConditionError& e) {
      py::set_error(symmetry_condition, e.what());
    } catch (const SymmetryProjectionError& e) {
      py::set_error(symmetry_projection, e.what());
    } catch (const SymmetryError& e) {
      py::set_error(symmetry_error, e.what());
    } catch (const BlockExtentMismatchError& e) {
      py::set_error(block_extent, e.what());
    } catch (const BlockAliasMismatchError& e) {
      py::set_error(block_alias, e.what());
    } catch (const BlockLabelInvalidError& e) {
      py::set_error(block_label, e.what());
    } catch (const SymmetryBlockedTensorError& e) {
      py::set_error(sbt_error, e.what());
    } catch (const QdkError& e) {
      py::set_error(qdk_error, e.what());
    }
  });
}

void bind_symmetry(py::module& symmetry) {
  bind_symmetry_errors(symmetry);

  py::enum_<AxisName>(symmetry, "AxisName",
                      "Single-particle symmetry axes a basis may be blocked "
                      "under (only Spin is populated in this release).")
      .value("Spin", AxisName::Spin)
      .value("PointGroup", AxisName::PointGroup)
      .value("Kramers", AxisName::Kramers)
      .value("Mode", AxisName::Mode)
      .value("Site", AxisName::Site)
      .value("LatticeMomentum", AxisName::LatticeMomentum);

  symmetry.def(
      "axis_name_to_string", [](AxisName axis) { return to_string(axis); },
      py::arg("axis"), "Human-readable name for an AxisName.");

  py::class_<SymmetryAxisValue, std::shared_ptr<SymmetryAxisValue>>(
      symmetry, "SymmetryAxisValue",
      "Abstract value carried by a single symmetry axis.")
      .def("axis", &SymmetryAxisValue::axis, "The axis this value belongs to.")
      .def("kind_name", &SymmetryAxisValue::kind_name,
           "Stable serialization tag identifying the concrete subclass.")
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

  py::class_<SymmetryAxis>(
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
      .def("__hash__", &SymmetryAxis::hash);

  py::class_<Symmetries, std::shared_ptr<Symmetries>>(
      symmetry, "Symmetries",
      "A symmetry vocabulary: the ordered set of axes a basis is blocked "
      "under, together with their admissible labels and equivalence flags.")
      .def(py::init<std::vector<SymmetryAxis>>(), py::arg("axes"))
      .def("axes", &Symmetries::axes)
      .def("has_axis", &Symmetries::has_axis, py::arg("name"),
           "True iff an axis with the given name exists in this vocabulary.")
      .def("axis", &Symmetries::axis, py::arg("name"),
           py::return_value_policy::reference_internal,
           "Access the axis with the given name.")
      .def(py::self == py::self)
      .def(py::self != py::self)
      .def("__hash__", &Symmetries::hash);

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

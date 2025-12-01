// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <iostream>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/coupled_cluster.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/orbitals.hpp>

#include "property_binding_helpers.hpp"

namespace py = pybind11;

void bind_coupled_cluster(pybind11::module& data) {
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;

  py::class_<CoupledClusterAmplitudes, DataClass, py::smart_holder>
      coupled_cluster(data, "CoupledClusterAmplitudes",
                      R"(
Represents coupled cluster amplitudes.

This class stores the coupled cluster amplitudes and associated orbital information:

- Orbital information
- T1 and T2 amplitudes
)");

  // Constructors
  coupled_cluster.def(py::init<const CoupledClusterAmplitudes&>(),
                      R"(
Copy constructor.

Creates a deep copy of another ``CoupledClusterAmplitudes`` object.

Args:
    other (CoupledClusterAmplitudes): The amplitude object to copy

Examples:
    >>> original = CoupledClusterAmplitudes(...)
    >>> copy = CoupledClusterAmplitudes(original)

)");
  coupled_cluster.def(
      py::init<
          std::shared_ptr<qdk::chemistry::data::Orbitals>,
          const qdk::chemistry::data::CoupledClusterAmplitudes::amplitude_type&,
          const qdk::chemistry::data::CoupledClusterAmplitudes::amplitude_type&,
          unsigned int, unsigned int>(),
      py::arg("orbitals"), py::arg("t1_amplitudes"), py::arg("t2_amplitudes"),
      py::arg("n_alpha_electrons"), py::arg("n_beta_electrons"),
      R"(
Construct a ``CoupledClusterAmplitudes`` object from orbitals and amplitudes.

Args:
    orbitals (Orbitals): The orbital information
    t1_amplitudes (numpy.ndarray): The T1 amplitudes vector
    t2_amplitudes (numpy.ndarray): The T2 amplitudes vector (flattened from 4D tensor)
    n_alpha_electrons (int): The number of alpha electrons
    n_beta_electrons (int): The number of beta electrons

)");

  // Amplitude Management
  bind_getter_as_property(coupled_cluster, "get_t1_amplitudes",
                          &CoupledClusterAmplitudes::get_t1_amplitudes,
                          R"(
Get the T1 amplitudes.

Returns:
    numpy.ndarray: The T1 amplitudes vector

Raises:
    RuntimeError: If T1 amplitudes are not set

Examples:
    >>> cc = CoupledClusterAmplitudes(...)
    >>> t1 = cc.get_t1_amplitudes()
    >>> print(f"T1 shape: {t1.shape}")

)",
                          py::return_value_policy::reference_internal);

  coupled_cluster.def("has_t1_amplitudes",
                      &CoupledClusterAmplitudes::has_t1_amplitudes,
                      R"(
Check if T1 amplitudes are set.

Returns:
    bool: True if T1 amplitudes are set, false otherwise

Examples:
    >>> if cc.has_t1_amplitudes():
    ...     t1 = cc.get_t1_amplitudes()
    ... else:
    ...     print("T1 amplitudes not available")

)");

  bind_getter_as_property(coupled_cluster, "get_t2_amplitudes",
                          &CoupledClusterAmplitudes::get_t2_amplitudes,
                          R"(
Get the T2 amplitudes.

Returns:
    numpy.ndarray: The T2 amplitudes vector (flattened from 4D tensor)

Raises:
    RuntimeError: If T2 amplitudes are not set

Examples:
    >>> cc = CoupledClusterAmplitudes(...)
    >>> t2 = cc.get_t2_amplitudes()
    >>> print(f"T2 vector length: {len(t2)}")

)",
                          py::return_value_policy::reference_internal);

  coupled_cluster.def("has_t2_amplitudes",
                      &CoupledClusterAmplitudes::has_t2_amplitudes,
                      R"(
Check if T2 amplitudes are set.

Returns:
    bool: True if T2 amplitudes are set, false otherwise

Examples:
    >>> if cc.has_t2_amplitudes():
    ...     t2 = cc.get_t2_amplitudes()
    ... else:
    ...     print("T2 amplitudes not available")

)");

  coupled_cluster.def("get_num_occupied",
                      &CoupledClusterAmplitudes::get_num_occupied,
                      R"(
Get number of occupied orbitals.

Returns:
    tuple: A tuple of two integers ``(alpha_count, beta_count)``

        containing the number of occupied orbitals for each spin channel.

Examples:
    >>> num_alpha_occupied_orbitals, num_beta_occupied_orbitals = cc.get_num_occupied()
    >>> print(f"Occupied orbitals: α={num_alpha_occupied_orbitals}, β={num_beta_occupied_orbitals}")

)");

  coupled_cluster.def("get_num_virtual",
                      &CoupledClusterAmplitudes::get_num_virtual,
                      R"(
Get number of virtual orbitals.

Returns:
    tuple: A tuple of two integers ``(alpha_count, beta_count)``

        containing the number of virtual orbitals for each spin channel.

Examples:
    >>> num_alpha_virtual_orbitals, num_beta_virtual_orbitals = cc.get_num_virtual()
    >>> print(f"Virtual orbitals: α={num_alpha_virtual_orbitals}, β={num_beta_virtual_orbitals}")

)");
}

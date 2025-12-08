// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>

#include "property_binding_helpers.hpp"

namespace py = pybind11;

void bind_configuration(pybind11::module &data) {
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;

  py::class_<Configuration, DataClass, py::smart_holder> configuration(
      data, "Configuration",
      R"(
Represents a molecular electronic configuration.

This class efficiently stores the occupation pattern of molecular orbitals using a compact representation where each orbital can be in one of four states:

- UNOCCUPIED (0): No electrons
- ALPHA (1): One alpha electron
- BETA (2): One beta electron
- DOUBLY (3): Both alpha and beta electrons

The class provides methods for constructing, manipulating, and querying configurations.
)");

  // Configuration constructors
  configuration.def(py::init<>(),
                    R"(
Default constructor for an empty configuration.

Examples:
    >>> config = qdk_chemistry.Configuration()

)");

  configuration.def(py::init<const std::string &>(),
                    R"(
Constructs a configuration from a string representation.

Args:
    str (str): String representation of the configuration

        Where '0' = unoccupied orbital, '1' = alpha-occupied orbital,
        '2' = beta-occupied orbital, '3' = doubly-occupied orbital

Examples:
    >>> config = qdk_chemistry.Configuration("3322110")  # 7 orbitals with different occupations

)",
                    py::arg("str"));

  // Configuration methods
  configuration.def("to_string", &Configuration::to_string,
                    R"(
Convert the configuration to a string representation.

Returns:
    str: String representation

        where '0' = unoccupied orbital, '1' = alpha-occupied orbital,
        '2' = beta-occupied orbital, '3' = doubly-occupied orbital

Examples:
    >>> config = qdk_chemistry.Configuration("3322110")
    >>> print(config.to_string())
    3322110

)");

  bind_getter_as_property(configuration, "get_n_electrons",
                          &Configuration::get_n_electrons,
                          R"(
Get the number of alpha and beta electrons in this configuration.

Returns:
    tuple: A tuple containing (n_alpha, n_beta)

Examples:
    >>> config = qdk_chemistry.Configuration("3322110")
    >>> n_alpha, n_beta = config.get_n_electrons()
    >>> print(f"Alpha electrons: {n_alpha}, Beta electrons: {n_beta}")

)");

  configuration.def("__eq__", &Configuration::operator==,
                    R"(
Check if two configurations are equal.

Args:
    other (Configuration): Another configuration to compare with

Returns:
    bool: True if configurations are identical, False otherwise

Examples:
    >>> config1 = qdk_chemistry.Configuration("3322110")
    >>> config2 = qdk_chemistry.Configuration("3322110")
    >>> print(config1 == config2)
    True

)",
                    py::arg("other"));

  configuration.def("__ne__", &Configuration::operator!=,
                    R"(
Check if two configurations are not equal.

Args:
    other (Configuration): Another configuration to compare with

Returns:
    bool: True if configurations are different, False otherwise

Examples:
    >>> config1 = qdk_chemistry.Configuration("3322110")
    >>> config2 = qdk_chemistry.Configuration("3322111")
    >>> print(config1 != config2)
    True

)",
                    py::arg("other"));

  configuration.def(
      "__hash__",
      [](const Configuration &c) {
        return Py_hash_t(std::hash<std::string>()(c.to_string()));
      },
      R"(
Returns the hash of the Configuration.

Returns:
    py::hash_t:  Hash value of the Configuration object
)");

  // Static methods
  configuration.def_static(
      "canonical_hf_configuration", &Configuration::canonical_hf_configuration,
      R"(
Create a canonical Hartree-Fock configuration using the Aufbau principle.

Fills orbitals from lowest energy according to the Aufbau principle:

- Doubly occupied orbitals for paired electrons
- Singly occupied orbitals for unpaired electrons (alpha first if n_alpha > n_beta)
- Unoccupied orbitals for remaining positions

Args:
    n_alpha (int): Number of alpha electrons
    n_beta (int): Number of beta electrons
    n_orbitals (int): Total number of orbitals

Returns:
    Configuration: Configuration representing the HF ground state

Examples:
    >>> config = qdk_chemistry.Configuration.canonical_hf_configuration(3, 2, 5)
    >>> print(config.to_string())
    22u00

)",
      py::arg("n_alpha"), py::arg("n_beta"), py::arg("n_orbitals"));

  configuration.def(
      "__repr__",
      [](const Configuration &c) {
        return "<qdk_chemistry.Configuration '" + c.to_string() + "'>";
      },
      R"(
Returns a string representation of the Configuration.

Returns:
    str: String representation of the Configuration object

)");

  configuration.def(
      "__str__", [](const Configuration &c) { return c.to_string(); },
      R"(
Returns a string representation of the Configuration.

Returns:
    str: String representation of the Configuration as a orbital occupation string

)");
}

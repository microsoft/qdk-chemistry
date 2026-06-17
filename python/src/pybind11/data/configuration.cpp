// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>

#include "property_binding_helpers.hpp"

namespace py = pybind11;

void bind_configuration(pybind11::module &data) {
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;

  py::class_<Configuration, DataClass, py::smart_holder> configuration(
      data, "Configuration",
      R"(
Represents a single-particle occupation pattern with efficient bit-packing.

For spin-½ systems (``bits_per_mode=2``), each mode can be in one of four states:

- UNOCCUPIED ('0'): Empty
- ALPHA ('u'): One alpha particle
- BETA ('d'): One beta particle
- DOUBLY ('2'): Both alpha and beta particles

For generic systems (``bits_per_mode=1``), each mode is either '0' (empty)
or '1' (occupied).

The class provides methods for constructing, manipulating, and querying configurations.
)");

  // Configuration constructors
  configuration.def(py::init<>(),
                    R"(
Default constructor for an empty configuration.

Examples:
    >>> config = qdk_chemistry.Configuration()

)");

  configuration.def_static("from_spin_half_string",
                           &Configuration::from_spin_half_string,
                           R"(
Construct a spin-½ configuration from a string representation.

Args:
    str (str): String with alphabet '0'/'u'/'d'/'2'.

Returns:
    Configuration: A configuration with bits_per_mode == 2.

Examples:
    >>> config = qdk_chemistry.Configuration.from_spin_half_string("22ud0ud")
)",
                           py::arg("str"));

  configuration.def_static("from_bitstring", &Configuration::from_bitstring,
                           R"(
Construct a bitstring configuration (1 bit per mode) from a string representation.

Args:
    str (str): String with alphabet '0'/'1'.

Returns:
    Configuration: A configuration with bits_per_mode == 1.

Examples:
    >>> config = qdk_chemistry.Configuration.from_bitstring("01100")
)",
                           py::arg("str"));

  // Configuration methods
  configuration.def("to_string", &Configuration::to_string,
                    R"(
Convert the configuration to a string representation.

For spin-½ (bits_per_mode=2): '0'/'u'/'d'/'2'.
For bitstring (bits_per_mode=1): '0'/'1'.

Returns:
    str: String representation

Examples:
    >>> config = qdk_chemistry.Configuration.from_spin_half_string("22ud0ud")
    >>> print(config.to_string())
    22ud0ud

)");

  bind_getter_as_property(configuration, "bits_per_mode",
                          &Configuration::bits_per_mode,
                          R"(
Bits used to encode each mode (1 for spinless, 2 for spin-½).

Returns:
    int: Bits per mode.

Examples:
    >>> qdk_chemistry.Configuration.from_bitstring("101").bits_per_mode
    1
    >>> qdk_chemistry.Configuration.from_spin_half_string("2u0").bits_per_mode
    2

)");

  bind_getter_as_property(configuration, "num_modes", &Configuration::num_modes,
                          R"(
Number of single-particle modes in the configuration.

Returns:
    int: Number of modes.

Examples:
    >>> qdk_chemistry.Configuration.from_bitstring("1010").num_modes
    8

)");

  bind_getter_as_property(configuration, "total_occupation",
                          &Configuration::total_occupation,
                          R"(
Total occupation summed over all modes.

For spin-½ modes the per-mode occupation is the popcount of the 2-bit state. For spinless modes it is the 1-bit value itself.

Returns:
    int: Sum of per-mode occupations.

Examples:
    >>> qdk_chemistry.Configuration.from_bitstring("1010").total_occupation
    2
    >>> qdk_chemistry.Configuration.from_spin_half_string("2u0").total_occupation
    3

)");

  configuration.def("get_mode_state", &Configuration::get_mode_state,
                    R"(
Raw state value for a given mode index.

Args:
    idx (int): Mode index (0-indexed).

Returns:
    int: Packed state value (range 0 to 2^bits_per_mode - 1).

Raises:
    IndexError: If idx >= num_modes.

Examples:
    >>> config = qdk_chemistry.Configuration.from_bitstring("101")
    >>> config.get_mode_state(0)
    1
    >>> config.get_mode_state(1)
    0

)",
                    py::arg("idx"));

  configuration.def("to_binary_strings", &Configuration::to_binary_strings,
                    py::arg("num_orbitals"),
                    R"(
Convert configuration to separate alpha and beta binary strings.

Parameters:
    num_orbitals (int):

        Number of spatial orbitals to use from the configuration

Returns:
    tuple[str, str]

        Tuple of binary strings (alpha, beta) where '1' indicates occupied
        and '0' indicates unoccupied for each spin channel

Examples:
    >>> config = qdk_chemistry.Configuration.from_spin_half_string("2du0")
    >>> print(config.to_binary_strings(4))
    ("1010", "1100")

)");

  bind_getter_as_property(configuration, "get_n_electrons",
                          &Configuration::get_n_electrons,
                          R"(
Get the number of alpha and beta electrons in this configuration.

Returns:
    tuple: A tuple containing (n_alpha, n_beta)

Examples:
    >>> config = qdk_chemistry.Configuration.from_spin_half_string("22ud0ud")
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
    >>> config1 = qdk_chemistry.Configuration.from_spin_half_string("22ud0ud")
    >>> config2 = qdk_chemistry.Configuration.from_spin_half_string("22ud0ud")
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
    >>> config1 = qdk_chemistry.Configuration.from_spin_half_string("22ud0ud")
    >>> config2 = qdk_chemistry.Configuration.from_spin_half_string("22ud0u0")
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

  configuration.def("to_bits", &Configuration::to_bits,
                    R"(
Return a flat list of 0/1 bit values representing the configuration.

For bits_per_mode == 1: returns a list of length num_modes where each
element is the mode state (0 or 1).

For bits_per_mode == 2 (spin-½): returns a list of length 2 * num_modes.
The first num_modes entries are the alpha occupations and the last num_modes
entries are the beta occupations (Jordan-Wigner ordering).

The total length is always num_modes * bits_per_mode, i.e. the number of
qubits in a Jordan-Wigner mapping.

Returns:
    list[int]: List of 0s and 1s.

Examples:
    >>> config = qdk_chemistry.Configuration.from_bitstring("10110")
    >>> config.to_bits()
    [1, 0, 1, 1, 0]
    >>> config = qdk_chemistry.Configuration.from_spin_half_string("2u0d")
    >>> config.to_bits()
    [1, 1, 0, 0, 1, 0, 0, 1]

)");
  // Data type name class attribute
  configuration.attr("_data_type_name") =
      DATACLASS_TO_SNAKE_CASE(Configuration);
}

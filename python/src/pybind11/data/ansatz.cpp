// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>

#include "path_utils.hpp"
#include "property_binding_helpers.hpp"

namespace py = pybind11;

// Wrapper functions that accept both strings and pathlib.Path objects
namespace {

void ansatz_to_file_wrapper(qdk::chemistry::data::Ansatz& self,
                            const py::object& filename,
                            const std::string& format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

std::shared_ptr<qdk::chemistry::data::Ansatz> ansatz_from_file_wrapper(
    const py::object& filename, const std::string& format_type) {
  return qdk::chemistry::data::Ansatz::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

void ansatz_to_json_file_wrapper(qdk::chemistry::data::Ansatz& self,
                                 const py::object& filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<qdk::chemistry::data::Ansatz> ansatz_from_json_file_wrapper(
    const py::object& filename) {
  return qdk::chemistry::data::Ansatz::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void ansatz_to_hdf5_file_wrapper(qdk::chemistry::data::Ansatz& self,
                                 const py::object& filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<qdk::chemistry::data::Ansatz> ansatz_from_hdf5_file_wrapper(
    const py::object& filename) {
  return qdk::chemistry::data::Ansatz::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

}  // namespace

void bind_ansatz(pybind11::module& data) {
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;

  py::class_<Ansatz, DataClass, py::smart_holder> ansatz(data, "Ansatz", R"(
Represents a quantum chemical ansatz combining a Hamiltonian and wavefunction.

This class represents a complete quantum chemical ansatz, which consists of:

- A Hamiltonian operator describing the system's energy
- A wavefunction describing the quantum state

The class is immutable after construction, meaning all data must be provided during construction and cannot be modified afterwards.
This ensures consistency between the Hamiltonian and wavefunction throughout the calculation.

Common use cases:

- Configuration interaction (CI) methods
- Multi-configuration self-consistent field (MCSCF) calculations
- Coupled cluster calculations
- Energy expectation value computations

Examples:
    Create an ``Ansatz`` from ``Hamiltonian`` and ``Wavefunction``:

    >>> import qdk_chemistry.data as data
    >>> # Assuming you have hamiltonian and wavefunction objects
    >>> ansatz = data.Ansatz(hamiltonian, wavefunction)
    True

    Create from shared pointers:

    >>> ansatz2 = data.Ansatz(hamiltonian_ptr, wavefunction_ptr)

    Access components:

    >>> h = ansatz.get_hamiltonian()
    >>> wf = ansatz.get_wavefunction()
    >>> orbs = ansatz.get_orbitals()

    Calculate energy:

    >>> energy = ansatz.calculate_energy()

    File I/O:

    >>> ansatz.to_json_file("ansatz.json")
    >>> ansatz.to_hdf5_file("ansatz.h5")
    >>> ansatz2 = data.Ansatz.from_json_file("ansatz.json")
    >>> ansatz3 = data.Ansatz.from_hdf5_file("ansatz.h5")

)");

  // Constructors
  ansatz.def(py::init<const Hamiltonian&, const Wavefunction&>(),
             R"(
Constructor with ``Hamiltonian`` and ``Wavefunction`` objects.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): The Hamiltonian operator for the system
    wavefunction (qdk_chemistry.data.Wavefunction): The wavefunction describing the quantum state

Raises:
    ValueError: If orbital dimensions are inconsistent between Hamiltonian and wavefunction

)",
             py::arg("hamiltonian"), py::arg("wavefunction"));

  ansatz.def(
      py::init<std::shared_ptr<Hamiltonian>, std::shared_ptr<Wavefunction>>(),
      R"(
Constructor with shared pointers to Hamiltonian and Wavefunction.

Args:
    hamiltonian (qdk_chemistry.data.Hamiltonian): Shared pointer to the Hamiltonian operator
    wavefunction (qdk_chemistry.data.Wavefunction): Shared pointer to the wavefunction

Raises:
    ValueError: If pointers are None or orbital dimensions are inconsistent

)",
      py::arg("hamiltonian"), py::arg("wavefunction"));

  // Copy constructor
  ansatz.def(py::init<const Ansatz&>(), "Copy constructor", py::arg("other"));

  // Hamiltonian Access
  bind_getter_as_property(ansatz, "get_hamiltonian", &Ansatz::get_hamiltonian,
                          R"(
Get shared pointer to the Hamiltonian.

Returns:
    qdk_chemistry.data.Hamiltonian: Shared pointer to the Hamiltonian object

Raises:
    RuntimeError: If Hamiltonian is not set

)",
                          py::return_value_policy::reference_internal);

  ansatz.def("has_hamiltonian", &Ansatz::has_hamiltonian,
             R"(
Check if Hamiltonian is available.

Returns:
    bool: True if Hamiltonian is set
)");

  // Wavefunction Access
  bind_getter_as_property(ansatz, "get_wavefunction", &Ansatz::get_wavefunction,
                          R"(
Get shared pointer to the wavefunction.

Returns:
    qdk_chemistry.data.Wavefunction: Shared pointer to the Wavefunction object

Raises:
    RuntimeError: If wavefunction is not set

)",
                          py::return_value_policy::reference_internal);

  ansatz.def("has_wavefunction", &Ansatz::has_wavefunction,
             R"(
Check if wavefunction is available.

Returns:
    bool: True if wavefunction is set

)");

  // Orbital Information
  bind_getter_as_property(ansatz, "get_orbitals", &Ansatz::get_orbitals,
                          R"(
Get shared pointer to the orbital basis set from the Hamiltonian.

Returns:
    qdk_chemistry.data.Orbitals: Shared pointer to the Orbitals object

Raises:
    RuntimeError: If orbitals are not available

)",
                          py::return_value_policy::reference_internal);

  ansatz.def("has_orbitals", &Ansatz::has_orbitals,
             R"(
Check if orbital data is available.

Returns:
    bool: True if orbitals are set in both Hamiltonian and wavefunction

)");

  // Energy Calculations
  ansatz.def("calculate_energy", &Ansatz::calculate_energy,
             R"(
Calculate the energy expectation value ⟨ψ|H|ψ⟩.

Returns:
    float: Energy expectation value in atomic units

Raises:
    RuntimeError: If calculation cannot be performed

Notes:
    This method will be implemented once energy calculation algorithms are available.

)");

  ansatz.def("validate_orbital_consistency",
             &Ansatz::validate_orbital_consistency,
             R"(
Validate orbital consistency between Hamiltonian and wavefunction.

Raises:
    RuntimeError: If orbital dimensions are inconsistent

)");

  // Summary
  bind_getter_as_property(ansatz, "get_summary", &Ansatz::get_summary,
                          R"(
Get a summary string describing the Ansatz object.

Returns:
    str: Human-readable summary of the Ansatz object

)");

  // Generic File I/O
  ansatz.def("to_file", &ansatz_to_file_wrapper,
             R"(
Save to file based on type parameter.

Args:
    filename (str | pathlib.Path): Path to file to create/overwrite
    type (str): File format type ("json" or "hdf5")

Raises:
    RuntimeError: If unsupported type or I/O error occurs

)",
             py::arg("filename"), py::arg("type"));

  ansatz.def_static("from_file", &ansatz_from_file_wrapper,
                    R"(
Load from file based on type parameter.

Args:
    filename (str | pathlib.Path): Path to file to read
    type (str): File format type ("json" or "hdf5")

Returns:
    qdk_chemistry.data.Ansatz: New Ansatz loaded from file

Raises:
    RuntimeError: If file doesn't exist, unsupported type, or I/O error occurs

)",
                    py::arg("filename"), py::arg("type"));

  // JSON File I/O
  ansatz.def("to_json_file", &ansatz_to_json_file_wrapper,
             R"(
Save ansatz to JSON file.

Args:
    filename (str | pathlib.Path): Path to JSON file to create/overwrite

Raises:
    RuntimeError: If I/O error occurs

)",
             py::arg("filename"));

  ansatz.def_static("from_json_file", &ansatz_from_json_file_wrapper,
                    R"(
Load Ansatz from JSON file.

Args:
    filename (str | pathlib.Path): Path to JSON file to read

Returns:
    qdk_chemistry.data.Ansatz: Ansatz loaded from file

Raises:
    RuntimeError: If file doesn't exist or I/O error occurs

)",
                    py::arg("filename"));

  // HDF5 File I/O
  ansatz.def("to_hdf5_file", &ansatz_to_hdf5_file_wrapper,
             R"(
Save Ansatz to HDF5 file.

Args:
    filename (str | pathlib.Path): Path to HDF5 file to create/overwrite

Raises:
    RuntimeError: If I/O error occurs

)",
             py::arg("filename"));

  ansatz.def_static("from_hdf5_file", &ansatz_from_hdf5_file_wrapper,
                    R"(
Load Ansatz from HDF5 file.

Args:
    filename (str | pathlib.Path): Path to HDF5 file to read

Returns:
    qdk_chemistry.data.Ansatz: Ansatz loaded from file

Raises:
    RuntimeError: If file doesn't exist or I/O error occurs

)",
                    py::arg("filename"));

  // JSON serialization (in-memory)
  ansatz.def(
      "to_json",
      [](const Ansatz& self) -> std::string { return self.to_json().dump(); },
      R"(
Convert Ansatz to JSON string format.

Returns:
    str: JSON string containing Ansatz data

)");

  ansatz.def_static(
      "from_json",
      [](const std::string& json_str) -> std::shared_ptr<Ansatz> {
        return Ansatz::from_json(nlohmann::json::parse(json_str));
      },
      R"(
Load Ansatz from JSON string format.

Args:
    json_str (str): JSON string containing Ansatz data

Returns:
    qdk_chemistry.data.Ansatz: Ansatz loaded from JSON string

Raises:
    RuntimeError: If JSON string is malformed

)",
      py::arg("json_str"));

  // String representation
  ansatz.def("__repr__", [](const Ansatz& self) { return self.get_summary(); });

  ansatz.def("__str__", [](const Ansatz& self) { return self.get_summary(); });

  // Pickling support using JSON serialization
  ansatz.def(py::pickle(
      [](const Ansatz& a) {
        // __getstate__ - serialize to JSON string
        return a.to_json().dump();
      },
      [](const std::string& json_str) {
        // __setstate__ - deserialize from JSON string
        return *Ansatz::from_json(nlohmann::json::parse(json_str));
      }));

  // Data type name class attribute
  ansatz.attr("_data_type_name") = DATACLASS_TO_SNAKE_CASE(Ansatz);
}

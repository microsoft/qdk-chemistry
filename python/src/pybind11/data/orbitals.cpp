// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry.hpp>
#include <qdk/chemistry/data/orbitals.hpp>

#include "path_utils.hpp"
#include "property_binding_helpers.hpp"

namespace py = pybind11;

// Wrapper functions for file I/O methods that accept both strings and pathlib
// Path objects
void orbitals_to_file_wrapper(qdk::chemistry::data::Orbitals &self,
                              const py::object &filename,
                              const std::string &format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

std::shared_ptr<qdk::chemistry::data::Orbitals> orbitals_from_file_wrapper(
    const py::object &filename, const std::string &format_type) {
  return qdk::chemistry::data::Orbitals::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

void orbitals_to_hdf5_file_wrapper(qdk::chemistry::data::Orbitals &self,
                                   const py::object &filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<qdk::chemistry::data::Orbitals> orbitals_from_hdf5_file_wrapper(
    const py::object &filename) {
  return qdk::chemistry::data::Orbitals::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void orbitals_to_json_file_wrapper(qdk::chemistry::data::Orbitals &self,
                                   const py::object &filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<qdk::chemistry::data::Orbitals> orbitals_from_json_file_wrapper(
    const py::object &filename) {
  return qdk::chemistry::data::Orbitals::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void bind_model_orbitals(py::module &data);

void bind_orbitals(py::module &data) {
  using namespace qdk::chemistry::algorithms;
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;

  py::class_<Orbitals, DataClass, py::smart_holder> orbitals(data, "Orbitals",
                                                             R"(
Represents molecular orbitals with coefficients and energies.

This class stores and manipulates molecular orbital data including:

- Orbital coefficients (alpha/beta spin channels)
- Orbital energies (alpha/beta spin channels)
- Atomic orbital overlap matrix
- Basis set information

The class provides methods to calculate atomic orbital density matrices from occupation vectors or reduced density matrices (RDMs) passed as parameters.

Supports both restricted (RHF/RKS) and unrestricted (UHF/UKS) orbitals.
)");

  // Copy constructor
  orbitals.def(py::init<const Orbitals &>(),
               R"(
Copy constructor.

Creates a deep copy of another ``Orbitals`` object.

Args:
    other (Orbitals): The Orbitals object to copy

Examples:
    >>> original_orbitals = Orbitals(...)
    >>> copied_orbitals = Orbitals(original_orbitals)
    >>> print(f"Copied {copied_orbitals.get_num_molecular_orbitals()} orbitals")

)");

  // Constructor for restricted orbitals
  orbitals.def(
      py::init<const Eigen::MatrixXd &, const std::optional<Eigen::VectorXd> &,
               const std::optional<Eigen::MatrixXd> &,
               std::shared_ptr<qdk::chemistry::data::BasisSet>,
               const std::optional<
                   std::tuple<std::vector<size_t>, std::vector<size_t>>> &>(),
      R"(
Constructor for restricted orbitals.

``num_atomic_orbitals`` refers to the number of atomic orbitals and ``num_molecular_orbitals`` refers to
the number of molecular orbitals.

Args:
    coefficients (numpy.ndarray): The molecular orbital coefficients matrix (``num_atomic_orbitals`` × ``num_molecular_orbitals``)
    energies (Optional[numpy.ndarray]): The orbital energies (``num_molecular_orbitals``), can be None
    ao_overlap (Optional[numpy.ndarray]): The atomic orbital overlap matrix (``num_atomic_orbitals`` × ``num_atomic_orbitals``), can be ``None``
    basis_set (BasisSet): The basis set
    indices (Optional[tuple[list[int], list[int]]]): Tuple of (active_space_indices, inactive_space_indices), can be ``None``

Examples:
    >>> import numpy as np
    >>> coeffs = np.random.random((4, 3))
    >>> basis_set = BasisSet(...)
    >>> orbitals = Orbitals(coeffs, None, None, basis_set, None)

)",
      py::arg("coefficients"),
      py::arg("energies") = std::optional<Eigen::VectorXd>{},
      py::arg("ao_overlap") = std::optional<Eigen::MatrixXd>{},
      py::arg("basis_set"),
      py::arg("indices") = std::optional<
          std::tuple<std::vector<size_t>, std::vector<size_t>>>{});

  // Constructor for unrestricted orbitals
  orbitals.def(
      py::init<const Eigen::MatrixXd &, const Eigen::MatrixXd &,
               const std::optional<Eigen::VectorXd> &,
               const std::optional<Eigen::VectorXd> &,
               const std::optional<Eigen::MatrixXd> &,
               std::shared_ptr<qdk::chemistry::data::BasisSet>,
               const std::optional<
                   std::tuple<std::vector<size_t>, std::vector<size_t>,
                              std::vector<size_t>, std::vector<size_t>>> &>(),
      R"(
Constructor for unrestricted orbitals.

``num_atomic_orbitals`` refers to the number of atomic orbitals and ``num_molecular_orbitals`` refers to
the number of molecular orbitals.

Args:
    coefficients_alpha (numpy.ndarray): The alpha molecular orbital coefficients matrix

        (``num_atomic_orbitals`` × ``num_molecular_orbitals``)

    coefficients_beta (numpy.ndarray): The beta molecular orbital coefficients matrix

        (``num_atomic_orbitals`` × ``num_molecular_orbitals``)

    energies_alpha (Optional[numpy.ndarray]): The alpha orbital energies

        (``num_molecular_orbitals``), can be None

    energies_beta (Optional[numpy.ndarray]): The beta orbital energies

        (``num_molecular_orbitals``), can be None

    ao_overlap (Optional[numpy.ndarray]): The atomic orbital overlap matrix

        (``num_atomic_orbitals`` × ``num_atomic_orbitals``), can be None

    basis_set (BasisSet): The basis set

    indices (Optional[tuple[list[int], list[int], list[int], list[int]]]): Tuple of

        (``active_alpha``, ``active_beta``, ``inactive_alpha``, ``inactive_beta``), can be ``None``

Examples:
    >>> import numpy as np
    >>> alpha_coeffs = np.random.random((4, 3))
    >>> beta_coeffs = np.random.random((4, 3))
    >>> basis_set = BasisSet(...)
    >>> orbitals = Orbitals(alpha_coeffs, beta_coeffs, None, None, None, basis_set, None)

)",
      py::arg("coefficients_alpha"), py::arg("coefficients_beta"),
      py::arg("energies_alpha") = std::optional<Eigen::VectorXd>{},
      py::arg("energies_beta") = std::optional<Eigen::VectorXd>{},
      py::arg("ao_overlap") = std::optional<Eigen::MatrixXd>{},
      py::arg("basis_set"),
      py::arg("indices") = std::optional<
          std::tuple<std::vector<size_t>, std::vector<size_t>,
                     std::vector<size_t>, std::vector<size_t>>>{});

  // Coefficient access (read-only)
  orbitals.def("get_coefficients", &Orbitals::get_coefficients,
               R"(
Get orbital coefficients as pair of (alpha, beta) matrices.

``num_atomic_orbitals`` refers to the number of atomic orbitals and ``num_molecular_orbitals`` refers to
the number of molecular orbitals.

Returns:
    tuple[numpy.ndarray]: Pair of ``(alpha_coeffs, beta_coeffs)`` matrices,

        each with shape ``(num_atomic_orbitals, num_molecular_orbitals)``

Examples:
    >>> alpha_coeffs, beta_coeffs = orbitals.get_coefficients()
    >>> print(f'Alpha coefficients shape: {alpha_coeffs.shape}')

)");

  // Energy access (read-only)
  orbitals.def("get_energies", &Orbitals::get_energies,
               R"(
Get orbital energies in Hartrees as pair of (alpha, beta) vectors.

Returns:
    tuple[numpy.ndarray]: Pair of ``(alpha_energies, beta_energies)`` vectors, each with length ``num_molecular_orbitals``

Raises:
    RuntimeError: If energies have not been set

Examples:
    >>> alpha_energies, beta_energies = orbitals.get_energies()
    >>> print(f'HOMO energy: {alpha_energies[homo_index]}')

)");

  orbitals.def("has_energies", &Orbitals::has_energies,
               R"(
Check if orbital energies have been internally set.

Returns:
    bool: True if energies have been set, false otherwise

Examples:
    >>> if orbitals.has_energies():
    ...     alpha_e, beta_e = orbitals.get_energies()
    ... else:
    ...     print("Energies not yet set")

)");

  orbitals.def(
      "calculate_ao_density_matrix",
      py::overload_cast<const Eigen::VectorXd &, const Eigen::VectorXd &>(
          &Orbitals::calculate_ao_density_matrix, py::const_),
      R"(
Calculate atomic orbital density matrix from occupation vectors (unrestricted).

Args:
    occupations_alpha (numpy.ndarray): Alpha spin occupation vector (size must match number of MOs)
    occupations_beta (numpy.ndarray): Beta spin occupation vector (size must match number of MOs)

Returns:
    tuple [numpy.ndarray]: Pair of (alpha, beta) atomic orbital density matrices

Raises:
    RuntimeError: If occupation vector sizes don't match number of MOs

Examples:
    >>> alpha_occ = np.array([1.0, 1.0, 0.0])
    >>> beta_occ = np.array([1.0, 0.0, 0.0])
    >>> P_alpha, P_beta = orbitals.calculate_ao_density_matrix(alpha_occ, beta_occ)

)",
      py::arg("occupations_alpha"), py::arg("occupations_beta"));

  orbitals.def("calculate_ao_density_matrix",
               py::overload_cast<const Eigen::VectorXd &>(
                   &Orbitals::calculate_ao_density_matrix, py::const_),
               R"(
Calculate atomic orbital density matrix from occupation vector (restricted).

Args:
    occupations (numpy.ndarray): Total occupation vector (size must match number of molecular orbitals)

Returns:
    numpy.ndarray: Atomic orbital density matrix (total alpha + beta)

Raises:
    RuntimeError: If occupation vector size doesn't match number of MOs

Examples:
    >>> occupations = np.array([2.0, 0.0, 0.0])  # 2 electrons in first MO
    >>> P_total = orbitals.calculate_ao_density_matrix(occupations)

)",
               py::arg("occupations"));

  orbitals.def(
      "calculate_ao_density_matrix_from_rdm",
      py::overload_cast<const Eigen::MatrixXd &, const Eigen::MatrixXd &>(
          &Orbitals::calculate_ao_density_matrix_from_rdm, py::const_),
      R"(
Calculate atomic orbital density matrix from 1RDM in molecular orbital space (unrestricted).

Args:
    rdm_alpha (numpy.ndarray): Alpha 1RDM in MO basis (size must match number of MOs × MOs)
    rdm_beta (numpy.ndarray): Beta 1RDM in MO basis (size must match number of MOs × MOs)

Returns:
    tuple[numpy.ndarray]: Pair of (alpha, beta) AO density matrices

Raises:
    RuntimeError: If 1RDM matrix sizes don't match number of MOs

Examples:
    >>> rdm_alpha = np.eye(3)  # Simple example with identity matrix
    >>> rdm_beta = np.zeros((3, 3))
    >>> P_alpha, P_beta = orbitals.calculate_ao_density_matrix_from_rdm(rdm_alpha, rdm_beta)

)",
      py::arg("rdm_alpha"), py::arg("rdm_beta"));

  orbitals.def("calculate_ao_density_matrix_from_rdm",
               py::overload_cast<const Eigen::MatrixXd &>(
                   &Orbitals::calculate_ao_density_matrix_from_rdm, py::const_),
               R"(
Calculate atomic orbital density matrix from 1RDM in molecular orbital space (restricted).

Args:
    rdm (numpy.ndarray): 1RDM in MO basis (size must match number of MOs × MOs)

Returns:
    numpy.ndarray: AO density matrix (total alpha + beta)

Raises:
    RuntimeError: If 1RDM matrix size doesn't match number of MOs

Examples:
    >>> rdm = np.eye(3)  # Simple example with identity matrix
    >>> P_total = orbitals.calculate_ao_density_matrix_from_rdm(rdm)

)",
               py::arg("rdm"));

  // AO overlap matrix (read-only)
  orbitals.def("get_overlap_matrix", &Orbitals::get_overlap_matrix,
               R"(
Get atomic orbital overlap matrix.

Returns:
    numpy.ndarray: Atomic orbital overlap matrix with shape ``(num_atomic_orbitals, num_atomic_orbitals)``

Examples:
    >>> overlap = orbitals.get_overlap_matrix()
    >>> print(f"Overlap matrix shape: {overlap.shape}")

)");

  orbitals.def("has_overlap_matrix", &Orbitals::has_overlap_matrix,
               R"(
Check if atomic orbital overlap matrix is available.

Returns:
    bool: True if overlap matrix is set, False otherwise

Examples:
    >>> if orbitals.has_overlap_matrix():
    ...     overlap = orbitals.get_overlap_matrix()
    ... else:
    ...     print("Overlap matrix not available")

)");

  // Basis set information (read-only)
  bind_getter_as_property(orbitals, "get_basis_set", &Orbitals::get_basis_set,
                          R"(
Get basis set information.

Returns:
    BasisSet: The basis set associated with these orbitals

Raises:
    RuntimeError: If basis set has not been set

Examples:
    >>> basis_set = orbitals.get_basis_set()
    >>> print(f"Basis set name: {basis_set.get_name()}")

)",
                          py::return_value_policy::reference_internal);

  orbitals.def("has_basis_set", &Orbitals::has_basis_set,
               R"(
Check if basis set information is available.

Returns:
    bool: True if basis set is set, False otherwise

Examples:
    >>> if orbitals.has_basis_set():
    ...     basis = orbitals.get_basis_set()
    ... else:
    ...     print("Basis set not available")

)");

  // Size and dimension queries
  orbitals.def("get_num_molecular_orbitals",
               &Orbitals::get_num_molecular_orbitals,
               R"(
Get number of molecular orbitals.

Returns:
    int: Number of molecular orbitals

Examples:
    >>> num_molecular_orbitals = orbitals.get_num_molecular_orbitals()
    >>> print(f'Number of MOs: {num_molecular_orbitals}')

)");

  orbitals.def("get_num_atomic_orbitals", &Orbitals::get_num_atomic_orbitals,
               R"(
Get number of atomic orbitals (atomic orbitals).

Returns:
    int: Number of atomic orbitals/atomic orbitals

Examples:
    >>> num_atomic_orbitals = orbitals.get_num_atomic_orbitals()
    >>> print(f'Basis set size: {num_atomic_orbitals}')

)");

  orbitals.def("is_restricted", &Orbitals::is_restricted,
               R"(
Check if calculation is restricted.

Restricted calculations use identical alpha and beta coefficients.

Returns:
    bool: True if alpha and beta coefficients are identical, False otherwise

Examples:
    >>> is_rhf = orbitals.is_restricted()
    >>> print(f"Restricted calculation: {is_rhf}")

)");

  orbitals.def("is_unrestricted", &Orbitals::is_unrestricted,
               R"(
Check if calculation is unrestricted.

Unrestricted calculations use separate alpha and beta coefficients.

Returns:
    bool: True if separate alpha/beta coefficients are used, False otherwise

Examples:
    >>> is_uhf = orbitals.is_unrestricted()
    >>> print(f"Unrestricted calculation: {is_uhf}")

)");

  // Individual alpha/beta access
  bind_getter_as_property(orbitals, "get_coefficients_alpha",
                          &Orbitals::get_coefficients_alpha,
                          R"(
Get alpha orbital coefficients matrix.

Returns:
    numpy.ndarray: Alpha orbital coefficients with shape ``(num_atomic_orbitals, num_molecular_orbitals)``

Examples:
    >>> alpha_coeffs = orbitals.get_coefficients_alpha()
    >>> print(f"Alpha coefficients shape: {alpha_coeffs.shape}")

)",
                          py::return_value_policy::reference_internal);

  bind_getter_as_property(orbitals, "get_coefficients_beta",
                          &Orbitals::get_coefficients_beta,
                          R"(
Get beta orbital coefficients matrix.

Returns:
    numpy.ndarray: Beta orbital coefficients with shape ``(num_atomic_orbitals, num_molecular_orbitals)``

Examples:
    >>> beta_coeffs = orbitals.get_coefficients_beta()
    >>> print(f"Beta coefficients shape: {beta_coeffs.shape}")

)",
                          py::return_value_policy::reference_internal);

  bind_getter_as_property(orbitals, "get_energies_alpha",
                          &Orbitals::get_energies_alpha,
                          R"(
Get alpha orbital energies (Hartree).

Returns:
    numpy.ndarray: Alpha orbital energies with length ``num_molecular_orbitals``

Examples:
    >>> alpha_energies = orbitals.get_energies_alpha()
    >>> homo_energy = alpha_energies[homo_index]

)",
                          py::return_value_policy::reference_internal);

  bind_getter_as_property(orbitals, "get_energies_beta",
                          &Orbitals::get_energies_beta,
                          R"(
Get beta orbital energies (Hartree).

Returns:
    numpy.ndarray: Beta orbital energies with length ``num_molecular_orbitals``

Examples:
    >>> beta_energies = orbitals.get_energies_beta()
    >>> homo_energy = beta_energies[homo_index]

)",
                          py::return_value_policy::reference_internal);

  bind_getter_as_property(orbitals, "get_summary", &Orbitals::get_summary,
                          R"(
Get summary string of orbital information.

Returns:
    str: Human-readable summary of orbital properties

Examples:
    >>> summary = orbitals.get_summary()
    >>> print(summary)

)");

  // Active space information (read-only)
  orbitals.def("get_active_space_indices", &Orbitals::get_active_space_indices,
               R"(
Get the active space orbital indices.

Returns:
    tuple: Pair of ``(alpha_indices, beta_indices)`` for active space orbitals

Examples:
    >>> alpha_active, beta_active = orbitals.get_active_space_indices()
    >>> print(f"Active space size: {len(alpha_active)}")

)");

  orbitals.def("get_inactive_space_indices",
               &Orbitals::get_inactive_space_indices,
               R"(
Get the inactive space orbital indices.

Returns:
    tuple: Pair of ``(alpha_indices, beta_indices)`` for inactive space orbitals

Examples:
    >>> alpha_inactive, beta_inactive = orbitals.get_inactive_space_indices()
    >>> print(f"Inactive space size: {len(alpha_inactive)}")

)");

  orbitals.def("get_inactive_space_indices",
               &Orbitals::get_inactive_space_indices,
               R"(
Get the inactive space orbital indices.

Returns:
    tuple: Pair of ``(alpha_indices, beta_indices)`` for inactive space orbitals

Examples:
    >>> alpha_inactive, beta_inactive = orbitals.get_inactive_space_indices()
    >>> print(f"Inactive space size: {len(alpha_inactive)}")

)");

  orbitals.def("get_virtual_space_indices",
               &Orbitals::get_virtual_space_indices,
               R"(
Get the virtual space orbital indices.

Returns:
    tuple: Pair of (alpha_indices, beta_indices) for virtual orbitals

Examples:
    >>> alpha_virtual, beta_virtual = orbitals.get_virtual_space_indices()
    >>> print(f"Virtual space size: {len(alpha_virtual)}")

)");

  orbitals.def("has_active_space", &Orbitals::has_active_space,
               R"(
Check if active space data is set.

Returns:
    bool: True if active space information is available, False otherwise

Examples:
    >>> if orbitals.has_active_space():
    ...     active_indices = orbitals.get_active_space_indices()
    ... else:
    ...     print("No active space defined")

)");

  // Serialization
  orbitals.def("to_file", orbitals_to_file_wrapper,
               R"(
Save orbital data to file with specified format.

Generic method to save orbital data to a file.

Args:
    filename (str | pathlib.Path): Path to the file to write.
    type (str): File format type ('json' or 'hdf5')

Raises:
    RuntimeError: If the orbital data is invalid, unsupported type, or file cannot be opened/written

Examples:
    >>> orbitals.to_file("water.orbitals.json", "json")
    >>> orbitals.to_file("molecule.orbitals.h5", "hdf5")
    >>> from pathlib import Path
    >>> orbitals.to_file(Path("water.orbitals.json"), "json")

)",
               py::arg("filename"), py::arg("type"));

  orbitals.def_static("from_file", orbitals_from_file_wrapper,
                      R"(
Load orbital data from file with specified format (static method).

Generic method to load orbital data from a file.

Args:
    filename (str | pathlib.Path): Path to the file to read.
    type (str): File format type ('json' or 'hdf5')

Returns:
    Orbitals: New ``Orbitals`` object loaded from the file

Raises:
    ValueError: If format_type is not supported or filename doesn't follow naming convention
    RuntimeError: If the file cannot be opened, read, or contains invalid orbital data

Examples:
    >>> orbitals = Orbitals.from_file("water.orbitals.json", "json")
    >>> orbitals = Orbitals.from_file("molecule.orbitals.h5", "hdf5")
    >>> from pathlib import Path
    >>> orbitals = Orbitals.from_file(Path("water.orbitals.json"), "json")

)",
                      py::arg("filename"), py::arg("type"));

  orbitals.def("to_hdf5_file", orbitals_to_hdf5_file_wrapper,
               R"(
Save orbital data to HDF5 file (with validation).

Writes all orbital data to an HDF5 file, preserving numerical precision.
HDF5 format is efficient for large datasets and supports hierarchical
data structures, making it ideal for storing molecular orbital information.

Args:
    filename (str | pathlib.Path): Path to the HDF5 file to write.

        Must have '.orbitals' before the file extension (e.g., ``water.orbitals.h5``, ``molecule.orbitals.hdf5``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the orbital data is invalid or the file cannot be opened/written

Examples:
    >>> orbitals.to_hdf5_file("water.orbitals.h5")
    >>> orbitals.to_hdf5_file("molecule.orbitals.hdf5")
    >>> from pathlib import Path
    >>> orbitals.to_hdf5_file(Path("water.orbitals.h5"))

)",
               py::arg("filename"));

  orbitals.def_static("from_hdf5_file", orbitals_from_hdf5_file_wrapper,
                      R"(
Load orbital data from HDF5 file (static method with validation).

Reads orbital data from an HDF5 file.
The file should contain data in the format produced by ``to_hdf5_file()``.

Args:
    filename (str | pathlib.Path): Path to the HDF5 file to read.

        Must have '.orbitals' before the file extension (e.g., ``water.orbitals.h5``, ``molecule.orbitals.hdf5``)

Returns:
    Orbitals: New ``Orbitals`` object loaded from the file

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened, read, or contains invalid orbital data

Examples:
    >>> orbitals = Orbitals.from_hdf5_file("water.orbitals.h5")
    >>> orbitals = Orbitals.from_hdf5_file("molecule.orbitals.hdf5")
    >>> from pathlib import Path
    >>> orbitals = Orbitals.from_hdf5_file(Path("water.orbitals.h5"))

)",
                      py::arg("filename"));

  orbitals.def(
      "to_json",
      [](const Orbitals &self) -> std::string { return self.to_json().dump(); },
      R"(
Convert orbital data to JSON string.

Serializes all orbital information to a JSON string format.
JSON is human-readable and suitable for debugging or data exchange.

Returns:
    str: JSON string representation of the orbital data

Raises:
    RuntimeError: If the orbital data is invalid

Examples:
    >>> json_str = orbitals.to_json()
    >>> print(json_str)  # Pretty-printed JSON

)");

  orbitals.def_static(
      "from_json",
      [](const std::string &json_str) {
        return *Orbitals::from_json(nlohmann::json::parse(json_str));
      },
      R"(
Load orbital data from JSON string (static method).

Parses orbital data from a JSON string and creates a new Orbitals object.
The string should contain JSON data in the format produced by ``to_json()``.

Args:
    json_str (str): JSON string containing orbital data

Returns:
    Orbitals: New Orbitals object created from JSON data

Raises:
    RuntimeError: If the JSON string is malformed or contains invalid orbital data

Examples:
    >>> orbitals = Orbitals.from_json('{"num_atomic_orbitals": 4, "num_molecular_orbitals": 3, ...}')

)",
      py::arg("json_str"));

  orbitals.def("to_json_file", orbitals_to_json_file_wrapper,
               R"(
Save orbital data to JSON file.

Writes all orbital data to a JSON file with pretty formatting.
The file will be created or overwritten if it already exists.

Args:
    filename (str | pathlib.Path): Path to the JSON file to write.

        Must have '.orbitals' before the file extension (e.g., ``water.orbitals.json``, ``molecule.orbitals.json``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the orbital data is invalid or the file cannot be opened/written

Examples:
    >>> orbitals.to_json_file("water.orbitals.json")
    >>> orbitals.to_json_file("molecule.orbitals.json")
    >>> from pathlib import Path
    >>> orbitals.to_json_file(Path("water.orbitals.json"))

)",
               py::arg("filename"));

  orbitals
      .def_static("from_json_file", orbitals_from_json_file_wrapper,
                  R"(
Load orbital data from JSON file (static method).

Reads orbital data from a JSON file.
The file should contain JSON data in the format produced by ``to_json_file()``.

Args:
    filename (str | pathlib.Path): Path to the JSON file to read.

        Must have '.orbitals' before the file extension (e.g., ``water.orbitals.json``, ``molecule.orbitals.json``)

Returns:
    Orbitals: New ``Orbitals`` object loaded from the file

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened, read, or contains invalid orbital data

Examples:
    >>> orbitals = Orbitals.from_json_file("water.orbitals.json")
    >>> orbitals = Orbitals.from_json_file("molecule.orbitals.json")
    >>> from pathlib import Path
    >>> orbitals = Orbitals.from_json_file(Path("water.orbitals.json"))

)",
                  py::arg("filename"))

      // String representation - bind summary to __repr__
      .def("__repr__", [](const Orbitals &o) { return o.get_summary(); })

      .def("__str__", [](const Orbitals &o) { return o.get_summary(); })

      // Pickling support using JSON serialization
      .def(py::pickle(
          [](const Orbitals &o) -> std::string {
            // Return JSON string for pickling
            return o.to_json().dump();
          },
          [](const std::string &json_str) -> Orbitals {
            // Reconstruct from JSON string
            return *Orbitals::from_json(nlohmann::json::parse(json_str));
          }));

  // Bind ModelOrbitals
  bind_model_orbitals(data);
}

void bind_model_orbitals(py::module &data) {
  using namespace qdk::chemistry::data;

  py::class_<ModelOrbitals, Orbitals, py::smart_holder> model_orbitals(
      data, "ModelOrbitals",
      R"(
Simple subclass of ``Orbitals`` for model systems without basis set information.

This class allows creating ``Orbitals`` objects with a specified basis size and whether the calculation is restricted or unrestricted, without needing to provide full coefficient or energy data.
The class allows for model Hamiltonians and Wavefunctions to be fully specified without explicit basis set details.

Calls to any functions requiring actual data (e.g. ``get_coefficients``,
``get_energies``, ``calculate_ao_density_matrix``, etc.) will throw runtime errors.

Examples:
    >>> # Create a simple 4-orbital restricted model system
    >>> model_orb = ModelOrbitals(4, True)
    >>> print(f"Number of orbitals: {model_orb.get_num_molecular_orbitals()}")

    >>> # Create with active and inactive spaces
    >>> active_indices = [1, 2]
    >>> inactive_indices = [0, 3]
    >>> model_orb = ModelOrbitals(4, active_indices, inactive_indices)

)");

  // Basic constructor
  model_orbitals.def(py::init<size_t, bool>(),
                     R"(
Constructor for model orbitals with basic parameters.

Args:
    basis_size (int): Number of atomic orbitals (and molecular orbitals)
    restricted (bool): Whether the calculation is restricted (True) or unrestricted (False)

Examples:
    >>> # Restricted calculation with 6 orbitals
    >>> model_orb = ModelOrbitals(6, True)
    >>> print(f"Is restricted: {model_orb.is_restricted()}")

    >>> # Unrestricted calculation with 4 orbitals
    >>> model_orb = ModelOrbitals(4, False)
    >>> print(f"Is unrestricted: {model_orb.is_unrestricted()}")

)",
                     py::arg("basis_size"), py::arg("restricted"));

  // Constructor with active and inactive space indices (restricted)
  model_orbitals.def(
      py::init<size_t,
               const std::tuple<std::vector<size_t>, std::vector<size_t>> &>(),
      R"(
Constructor with active and inactive space indices (restricted).

For restricted calculations, the same active and inactive space indices are used for both alpha and beta electrons.

Args:
    basis_size (int): Number of atomic orbitals (and molecular orbitals)
    indices (tuple[list[int], list[int]]): Tuple of ``(active_space_indices, inactive_space_indices)``

Raises:
    ValueError: If indices are >= basis_size or if active and inactive spaces overlap

Examples:
    >>> # Create a 6-orbital system with orbitals 2,3 active and 0,1,4,5 inactive
    >>> active = [2, 3]
    >>> inactive = [0, 1, 4, 5]
    >>> indices = (active, inactive)
    >>> model_orb = ModelOrbitals(6, indices)
    >>> print(f"Active space size: {len(model_orb.get_active_space_indices()[0])}")

)",
      py::arg("basis_size"), py::arg("indices"));

  // Constructor with active and inactive space indices (unrestricted)
  model_orbitals.def(
      py::init<size_t,
               const std::tuple<std::vector<size_t>, std::vector<size_t>,
                                std::vector<size_t>, std::vector<size_t>> &>(),
      R"(
Constructor with active and inactive space indices (unrestricted).

For unrestricted calculations, separate active and inactive space indices can be provided for alpha and beta electrons.

Args:
    basis_size (int): Number of atomic orbitals (and molecular orbitals)
    indices (tuple[list[int], list[int], list[int], list[int]]): Tuple of

        ``(active_alpha, active_beta, inactive_alpha, inactive_beta)``

Raises:
    ValueError: If indices are >= basis_size or if active and inactive spaces overlap

Examples:
    >>> # Create unrestricted system with different alpha/beta active spaces
    >>> alpha_active = [1, 2]
    >>> beta_active = [2, 3]
    >>> alpha_inactive = [0, 3, 4]
    >>> beta_inactive = [0, 1, 4]
    >>> indices = (alpha_active, beta_active, alpha_inactive, beta_inactive)
    >>> model_orb = ModelOrbitals(5, indices)
    >>> print(f"Is unrestricted: {model_orb.is_unrestricted()}")

)",
      py::arg("basis_size"), py::arg("indices"));

  // Static from_json method
  model_orbitals
      .def_static(
          "from_json",
          [](const std::string &json_str) {
            return ModelOrbitals::from_json(nlohmann::json::parse(json_str));
          },
          R"(
Load ModelOrbitals from JSON string (static method).

Parses ModelOrbitals data from a JSON string and creates a new ``ModelOrbitals`` object.
The string should contain JSON data in the format produced by ``to_json()``.

Args:
    json_str (str): JSON string containing ``ModelOrbitals`` data

Returns:
    ModelOrbitals: New ModelOrbitals object created from JSON data

Raises:
    RuntimeError: If the JSON string is malformed or contains invalid ``ModelOrbitals`` data

Examples:
    >>> json_str = '{"num_orbitals": 4, "is_restricted": true, ...}'
    >>> model_orb = ModelOrbitals.from_json(json_str)
    >>> print(f"Loaded {model_orb.get_num_molecular_orbitals()} orbitals")

)",
          py::arg("json_str"))

      // String representation - bind summary to __repr__
      .def("__repr__", [](const ModelOrbitals &mo) { return mo.get_summary(); })

      .def("__str__", [](const ModelOrbitals &mo) { return mo.get_summary(); })

      // Pickling support using JSON serialization
      .def(py::pickle(
          [](const ModelOrbitals &mo) -> std::string {
            // Return JSON string for pickling
            return mo.to_json().dump();
          },
          [](const std::string &json_str) -> ModelOrbitals {
            // Reconstruct from JSON string
            return *ModelOrbitals::from_json(nlohmann::json::parse(json_str));
          }));
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/complex.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>

#include "path_utils.hpp"
#include "property_binding_helpers.hpp"

namespace py = pybind11;

// Wrapper functions that accept both strings and pathlib.Path objects
namespace {

void hamiltonian_to_file_wrapper(qdk::chemistry::data::Hamiltonian& self,
                                 const py::object& filename,
                                 const std::string& format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

std::shared_ptr<qdk::chemistry::data::Hamiltonian>
hamiltonian_from_file_wrapper(const py::object& filename,
                              const std::string& format_type) {
  return qdk::chemistry::data::Hamiltonian::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

void hamiltonian_to_json_file_wrapper(qdk::chemistry::data::Hamiltonian& self,
                                      const py::object& filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<qdk::chemistry::data::Hamiltonian>
hamiltonian_from_json_file_wrapper(const py::object& filename) {
  return qdk::chemistry::data::Hamiltonian::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void hamiltonian_to_hdf5_file_wrapper(qdk::chemistry::data::Hamiltonian& self,
                                      const py::object& filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<qdk::chemistry::data::Hamiltonian>
hamiltonian_from_hdf5_file_wrapper(const py::object& filename) {
  return qdk::chemistry::data::Hamiltonian::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void hamiltonian_to_fcidump_file_wrapper(
    qdk::chemistry::data::Hamiltonian& self, const py::object& filename,
    size_t nalpha, size_t nbeta) {
  self.to_fcidump_file(qdk::chemistry::python::utils::to_string_path(filename),
                       nalpha, nbeta);
}

}  // namespace

void bind_hamiltonian(pybind11::module& data) {
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;

  // HamiltonianType enumeration
  py::enum_<HamiltonianType>(data, "HamiltonianType", R"(
Enumeration for Hamiltonian types.

Values:
    Hermitian: Standard Hermitian Hamiltonian
    NonHermitian: Non-Hermitian Hamiltonian
)")
      .value("Hermitian", HamiltonianType::Hermitian)
      .value("NonHermitian", HamiltonianType::NonHermitian);

  // SpinChannel enumeration
  py::enum_<SpinChannel>(data, "SpinChannel", R"(
Enumeration for different spin channels in unrestricted calculations.

Values:
    aa: Alpha-alpha spin channel (for one-body integrals)
    bb: Beta-beta spin channel (for one-body integrals)
    aaaa: Alpha-alpha-alpha-alpha spin channel (for two-body integrals)
    aabb: Alpha-beta-alpha-beta spin channel (for two-body integrals)
    bbbb: Beta-beta-beta-beta spin channel (for two-body integrals)
)")
      .value("aa", SpinChannel::aa)
      .value("bb", SpinChannel::bb)
      .value("aaaa", SpinChannel::aaaa)
      .value("aabb", SpinChannel::aabb)
      .value("bbbb", SpinChannel::bbbb);

  // ============================================================================
  // HamiltonianContainer - Abstract base class (not directly constructible)
  // ============================================================================
  py::class_<HamiltonianContainer, py::smart_holder> hamiltonian_container(
      data, "HamiltonianContainer", R"(
Abstract base class for Hamiltonian container implementations.

This class defines the interface for storing molecular Hamiltonian data
for quantum chemistry calculations. It contains:

* One-electron integrals (kinetic + nuclear attraction) in MO representation
* Molecular orbital information for the active space
* Core energy contributions from inactive orbitals and nuclear repulsion

Derived classes implement specific storage formats for two-electron integrals
(e.g., canonical 4-center, density-fitted, etc.).

Note:
    This class cannot be instantiated directly. Use a derived class like
    CanonicalFourCenterHamiltonianContainer instead.
)");

  // HamiltonianContainer methods (read-only accessors)
  bind_getter_as_property(hamiltonian_container, "get_one_body_integrals",
                          &HamiltonianContainer::get_one_body_integrals,
                          R"(
Get tuple of one-electron integrals (alpha, beta) in molecular orbital basis.

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: One-electron integral matrices [norb x norb]
    for alpha and beta spin channels.
)",
                          py::return_value_policy::reference_internal);

  hamiltonian_container.def("has_one_body_integrals",
                            &HamiltonianContainer::has_one_body_integrals,
                            R"(
Check if one-body integrals are available.

Returns:
    bool: True if one-body integrals have been set
)");

  hamiltonian_container.def(
      "get_one_body_element", &HamiltonianContainer::get_one_body_element,
      R"(
Get specific one-electron integral element <ij>.

Args:
    i (int): First orbital index
    j (int): Second orbital index
    channel (SpinChannel): Spin channel (aa or bb), defaults to aa

Returns:
    float: Value of the one-electron integral <ij>
)",
      py::arg("i"), py::arg("j"), py::arg("channel") = SpinChannel::aa);

  bind_getter_as_property(hamiltonian_container, "get_inactive_fock_matrix",
                          &HamiltonianContainer::get_inactive_fock_matrix,
                          R"(
Get tuple of inactive Fock matrices (alpha, beta).

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: Inactive Fock matrices for the active space
)");

  hamiltonian_container.def("has_inactive_fock_matrix",
                            &HamiltonianContainer::has_inactive_fock_matrix,
                            R"(
Check if inactive Fock matrix is available.

Returns:
    bool: True if inactive Fock matrix has been set
)");

  bind_getter_as_property(hamiltonian_container, "get_orbitals",
                          &HamiltonianContainer::get_orbitals,
                          R"(
Get molecular orbital data.

Returns:
    Orbitals: The orbital data associated with this container
)",
                          py::return_value_policy::reference_internal);

  hamiltonian_container.def("has_orbitals", &HamiltonianContainer::has_orbitals,
                            R"(
Check if orbital data is available.

Returns:
    bool: True if orbital data has been set
)");

  bind_getter_as_property(hamiltonian_container, "get_core_energy",
                          &HamiltonianContainer::get_core_energy,
                          R"(
Get core energy in atomic units.

Returns:
    float: Core energy contribution in Hartree
)");

  bind_getter_as_property(hamiltonian_container, "get_type",
                          &HamiltonianContainer::get_type,
                          R"(
Get the type of Hamiltonian (Hermitian or NonHermitian).

Returns:
    HamiltonianType: The Hamiltonian type
)");

  hamiltonian_container.def("is_hermitian", &HamiltonianContainer::is_hermitian,
                            R"(
Check if the Hamiltonian is Hermitian.

Returns:
    bool: True if the Hamiltonian type is Hermitian
)");

  hamiltonian_container.def("is_unrestricted",
                            &HamiltonianContainer::is_unrestricted,
                            R"(
Check if Hamiltonian is unrestricted.

Returns:
    bool: True if alpha and beta integrals are different
)");

  hamiltonian_container.def("get_container_type",
                            &HamiltonianContainer::get_container_type,
                            R"(
Get the type of this container as a string.

Returns:
    str: Container type identifier (e.g., "canonical_four_center")
)");

  // ============================================================================
  // CanonicalFourCenterHamiltonianContainer - Concrete implementation
  // ============================================================================
  py::class_<CanonicalFourCenterHamiltonianContainer, HamiltonianContainer,
             py::smart_holder>
      canonical_four_center(data, "CanonicalFourCenterHamiltonianContainer", R"(
Represents a molecular Hamiltonian with canonical 4-center two-electron integrals.

This class stores molecular Hamiltonian data for quantum chemistry calculations,
specifically designed for active space methods. It contains:

* One-electron integrals (kinetic + nuclear attraction) in MO representation
* Two-electron integrals (electron-electron repulsion) in MO representation
* Molecular orbital information for the active space
* Core energy contributions from inactive orbitals and nuclear repulsion

This is the standard full integral storage format where two-electron integrals
are stored as a flattened [norb^4] vector.

Examples:
    >>> import numpy as np
    >>> # Create restricted Hamiltonian
    >>> one_body = np.random.rand(4, 4)  # 4 orbitals
    >>> two_body = np.random.rand(256)   # 4^4 elements
    >>> fock_matrix = np.random.rand(4, 4)
    >>> container = CanonicalFourCenterHamiltonianContainer(
    ...     one_body, two_body, orbitals, 10.5, fock_matrix
    ... )
    >>> # Wrap in Hamiltonian interface
    >>> hamiltonian = Hamiltonian(container)
)");

  // Restricted constructor
  canonical_four_center.def(
      py::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
               std::shared_ptr<Orbitals>, double, const Eigen::MatrixXd&,
               HamiltonianType>(),
      R"(
Constructor for restricted active space Hamiltonian with 4-center integrals.

Args:
    one_body_integrals (numpy.ndarray): One-electron integrals matrix [norb x norb]
    two_body_integrals (numpy.ndarray): Two-electron integrals vector [norb^4]
    orbitals (Orbitals): Molecular orbital data
    core_energy (float): Core energy (nuclear repulsion + inactive orbitals)
    inactive_fock_matrix (numpy.ndarray): Inactive Fock matrix [norb x norb]
    type (HamiltonianType, optional): Type of Hamiltonian (Hermitian by default)

Examples:
    >>> import numpy as np
    >>> one_body = np.random.rand(4, 4)
    >>> two_body = np.random.rand(256)
    >>> fock_matrix = np.random.rand(4, 4)
    >>> container = CanonicalFourCenterHamiltonianContainer(
    ...     one_body, two_body, orbitals, 10.5, fock_matrix
    ... )
)",
      py::arg("one_body_integrals"), py::arg("two_body_integrals"),
      py::arg("orbitals"), py::arg("core_energy"),
      py::arg("inactive_fock_matrix"),
      py::arg("type") = HamiltonianType::Hermitian);

  // Unrestricted constructor
  canonical_four_center.def(
      py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
               const Eigen::VectorXd&, const Eigen::VectorXd&,
               const Eigen::VectorXd&, std::shared_ptr<Orbitals>, double,
               const Eigen::MatrixXd&, const Eigen::MatrixXd&,
               HamiltonianType>(),
      R"(
Constructor for unrestricted active space Hamiltonian with 4-center integrals.

Args:
    one_body_integrals_alpha (numpy.ndarray): Alpha one-electron integrals [norb x norb]
    one_body_integrals_beta (numpy.ndarray): Beta one-electron integrals [norb x norb]
    two_body_integrals_aaaa (numpy.ndarray): Alpha-alpha-alpha-alpha integrals [norb^4]
    two_body_integrals_aabb (numpy.ndarray): Alpha-beta-alpha-beta integrals [norb^4]
    two_body_integrals_bbbb (numpy.ndarray): Beta-beta-beta-beta integrals [norb^4]
    orbitals (Orbitals): Molecular orbital data
    core_energy (float): Core energy (nuclear repulsion + inactive orbitals)
    inactive_fock_matrix_alpha (numpy.ndarray): Alpha inactive Fock matrix [norb x norb]
    inactive_fock_matrix_beta (numpy.ndarray): Beta inactive Fock matrix [norb x norb]
    type (HamiltonianType, optional): Type of Hamiltonian (Hermitian by default)

Examples:
    >>> import numpy as np
    >>> one_body_a = np.random.rand(4, 4)
    >>> one_body_b = np.random.rand(4, 4)
    >>> two_body_aaaa = np.random.rand(256)
    >>> two_body_aabb = np.random.rand(256)
    >>> two_body_bbbb = np.random.rand(256)
    >>> fock_a = np.random.rand(4, 4)
    >>> fock_b = np.random.rand(4, 4)
    >>> container = CanonicalFourCenterHamiltonianContainer(
    ...     one_body_a, one_body_b,
    ...     two_body_aaaa, two_body_aabb, two_body_bbbb,
    ...     orbitals, 10.5, fock_a, fock_b
    ... )
)",
      py::arg("one_body_integrals_alpha"), py::arg("one_body_integrals_beta"),
      py::arg("two_body_integrals_aaaa"), py::arg("two_body_integrals_aabb"),
      py::arg("two_body_integrals_bbbb"), py::arg("orbitals"),
      py::arg("core_energy"), py::arg("inactive_fock_matrix_alpha"),
      py::arg("inactive_fock_matrix_beta"),
      py::arg("type") = HamiltonianType::Hermitian);

  // Two-body integral access (specific to
  // CanonicalFourCenterHamiltonianContainer)
  bind_getter_as_property(
      canonical_four_center, "get_two_body_integrals",
      &CanonicalFourCenterHamiltonianContainer::get_two_body_integrals,
      R"(
Get two-electron integrals in molecular orbital basis.

Returns:
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple of two-electron
    integral vectors [norb^4] for aaaa, aabb, and bbbb spin channels.

Notes:
    Integrals are stored as flattened vectors in chemist notation <ij|kl>
    where indices are ordered as i + j*norb + k*norb^2 + l*norb^3
)",
      py::return_value_policy::reference_internal);

  canonical_four_center.def(
      "get_two_body_element",
      &CanonicalFourCenterHamiltonianContainer::get_two_body_element,
      R"(
Get specific two-electron integral element <ij|kl>.

Args:
    i, j, k, l (int): Orbital indices
    channel (SpinChannel): Spin channel (aaaa, aabb, or bbbb), defaults to aaaa

Returns:
    float: Value of the two-electron integral <ij|kl>
)",
      py::arg("i"), py::arg("j"), py::arg("k"), py::arg("l"),
      py::arg("channel") = SpinChannel::aaaa);

  canonical_four_center.def(
      "has_two_body_integrals",
      &CanonicalFourCenterHamiltonianContainer::has_two_body_integrals,
      R"(
Check if two-body integrals are available.

Returns:
    bool: True if two-body integrals have been set
)");

  canonical_four_center.def(
      "is_restricted", &CanonicalFourCenterHamiltonianContainer::is_restricted,
      R"(
Check if Hamiltonian is restricted (alpha == beta).

Returns:
    bool: True if alpha and beta integrals are identical
)");

  canonical_four_center.def("is_valid",
                            &CanonicalFourCenterHamiltonianContainer::is_valid,
                            R"(
Check if the Hamiltonian data is complete and consistent.

Returns:
    bool: True if all required data is set and dimensions are consistent
)");

  canonical_four_center.def(
      "to_json",
      [](const CanonicalFourCenterHamiltonianContainer& self) -> std::string {
        return self.to_json().dump();
      },
      R"(
Convert container to JSON string.

Returns:
    str: JSON representation of the container
)");

  canonical_four_center.def(
      "to_fcidump_file",
      &CanonicalFourCenterHamiltonianContainer::to_fcidump_file,
      R"(
Save Hamiltonian to FCIDUMP file.

Args:
    filename (str): Path to FCIDUMP file to create/overwrite
    nalpha (int): Number of alpha electrons
    nbeta (int): Number of beta electrons
)",
      py::arg("filename"), py::arg("nalpha"), py::arg("nbeta"));

  // ============================================================================
  // Hamiltonian - Interface class
  // ============================================================================
  py::class_<Hamiltonian, DataClass, py::smart_holder> hamiltonian(
      data, "Hamiltonian", R"(
Interface class for molecular Hamiltonians in the molecular orbital basis.

This class provides a unified interface to molecular Hamiltonian data by
wrapping a HamiltonianContainer implementation. It supports:

* One-electron integrals (kinetic + nuclear attraction) in MO representation
* Two-electron integrals (electron-electron repulsion) in MO representation
* Molecular orbital information for the active space
* Core energy contributions from inactive orbitals and nuclear repulsion

The actual integral storage is handled by the underlying container, which
can use different representations (canonical 4-center, density-fitted, etc.).

Examples:
    >>> # Create a Hamiltonian from a CanonicalFourCenterHamiltonianContainer container
    >>> container = CanonicalFourCenterHamiltonianContainer(h1, h2, orbitals, e_core, fock)
    >>> hamiltonian = Hamiltonian(container)
    >>>
    >>> # Access integrals through the interface
    >>> h1_alpha, h1_beta = hamiltonian.get_one_body_integrals
    >>> core_energy = hamiltonian.get_core_energy
)");

  // Constructor from container
  hamiltonian.def(py::init([](std::unique_ptr<HamiltonianContainer> container) {
                    return std::make_shared<Hamiltonian>(std::move(container));
                  }),
                  R"(
Construct a Hamiltonian from a HamiltonianContainer.

Args:
    container (HamiltonianContainer): The container holding the Hamiltonian data.
        Ownership is transferred to the Hamiltonian.

Examples:
    >>> container = CanonicalFourCenterHamiltonianContainer(h1, h2, orbitals, e_core, fock)
    >>> hamiltonian = Hamiltonian(container)
)",
                  py::arg("container"));

  // One-body integral access
  bind_getter_as_property(hamiltonian, "get_one_body_integrals",
                          &Hamiltonian::get_one_body_integrals,
                          R"(
Get tuple of one-electron integrals (alpha, beta) in molecular orbital basis.

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: One-electron integral matrices [norb x norb]
    for alpha and beta spin channels.

Raises:
    RuntimeError: If one-body integrals have not been set

Examples:
    >>> h1_alpha, h1_beta = hamiltonian.get_one_body_integrals
    >>> print(f"One-body matrix shape: {h1_alpha.shape}")
)",
                          py::return_value_policy::reference_internal);

  hamiltonian.def("has_one_body_integrals",
                  &Hamiltonian::has_one_body_integrals,
                  R"(
Check if one-body integrals are available.

Returns:
    bool: True if one-body integrals have been set
)");

  hamiltonian.def("get_one_body_element", &Hamiltonian::get_one_body_element,
                  R"(
Get specific one-electron integral element <ij>.

Args:
    i (int): First orbital index
    j (int): Second orbital index
    channel (SpinChannel): Spin channel (aa or bb), defaults to aa

Returns:
    float: Value of the one-electron integral <ij>
)",
                  py::arg("i"), py::arg("j"),
                  py::arg("channel") = SpinChannel::aa);

  // Two-body integral access
  bind_getter_as_property(hamiltonian, "get_two_body_integrals",
                          &Hamiltonian::get_two_body_integrals,
                          R"(
Get two-electron integrals in molecular orbital basis.

Returns:
    tuple[numpy.ndarray, numpy.ndarray, numpy.ndarray]: Tuple of two-electron
    integral vectors [norb^4] for aaaa, aabb, and bbbb spin channels.

Raises:
    RuntimeError: If two-body integrals have not been set

Notes:
    Integrals are stored as flattened vectors in chemist notation <ij|kl>
)",
                          py::return_value_policy::reference_internal);

  hamiltonian.def("get_two_body_element", &Hamiltonian::get_two_body_element,
                  R"(
Get specific two-electron integral element <ij|kl>.

Args:
    i, j, k, l (int): Orbital indices
    channel (SpinChannel): Spin channel (aaaa, aabb, or bbbb), defaults to aaaa

Returns:
    float: Value of the two-electron integral <ij|kl>
)",
                  py::arg("i"), py::arg("j"), py::arg("k"), py::arg("l"),
                  py::arg("channel") = SpinChannel::aaaa);

  hamiltonian.def("has_two_body_integrals",
                  &Hamiltonian::has_two_body_integrals,
                  R"(
Check if two-body integrals are available.

Returns:
    bool: True if two-body integrals have been set
)");

  // Orbital information
  bind_getter_as_property(hamiltonian, "get_orbitals",
                          &Hamiltonian::get_orbitals,
                          R"(
Get molecular orbital data.

Returns:
    Orbitals: The orbital data associated with this Hamiltonian

Raises:
    RuntimeError: If orbital data has not been set
)",
                          py::return_value_policy::reference_internal);

  hamiltonian.def("has_orbitals", &Hamiltonian::has_orbitals,
                  R"(
Check if orbital data is available.

Returns:
    bool: True if orbital data has been set
)");

  // Core energy and Fock matrix
  bind_getter_as_property(hamiltonian, "get_core_energy",
                          &Hamiltonian::get_core_energy,
                          R"(
Get core energy in atomic units.

Returns:
    float: Core energy contribution in Hartree
)");

  hamiltonian.def("has_inactive_fock_matrix",
                  &Hamiltonian::has_inactive_fock_matrix,
                  R"(
Check if inactive Fock matrix is available.

Returns:
    bool: True if inactive Fock matrix has been set
)");

  bind_getter_as_property(hamiltonian, "get_inactive_fock_matrix",
                          &Hamiltonian::get_inactive_fock_matrix,
                          R"(
Get tuple of inactive Fock matrices (alpha, beta).

Returns:
    tuple[numpy.ndarray, numpy.ndarray]: Inactive Fock matrices for the active space
)");

  // Type and restriction checks
  bind_getter_as_property(hamiltonian, "get_type", &Hamiltonian::get_type,
                          R"(
Get the type of Hamiltonian (Hermitian or NonHermitian).

Returns:
    HamiltonianType: The Hamiltonian type
)");

  hamiltonian.def("is_hermitian", &Hamiltonian::is_hermitian,
                  R"(
Check if the Hamiltonian is Hermitian.

Returns:
    bool: True if the Hamiltonian type is Hermitian
)");

  hamiltonian.def("is_restricted", &Hamiltonian::is_restricted,
                  R"(
Check if Hamiltonian is restricted (alpha == beta).

Returns:
    bool: True if alpha and beta integrals are identical
)");

  hamiltonian.def("is_unrestricted", &Hamiltonian::is_unrestricted,
                  R"(
Check if Hamiltonian is unrestricted (alpha != beta).

Returns:
    bool: True if alpha and beta integrals are different
)");

  // Container access
  hamiltonian.def("get_container_type", &Hamiltonian::get_container_type,
                  R"(
Get the type of the underlying container.

Returns:
    str: Container type identifier (e.g., "canonical_four_center")
)");

  // Summary
  hamiltonian.def("get_summary", &Hamiltonian::get_summary,
                  R"(
Get a human-readable summary of the Hamiltonian data.

Returns:
    str: Multi-line string describing the Hamiltonian properties
)");

  // Generic file I/O
  hamiltonian.def("to_file", hamiltonian_to_file_wrapper, R"(
Save Hamiltonian to file based on type parameter.

Args:
    filename (str or pathlib.Path): Path to file to create/overwrite
    type (str): File format type ('json' or 'hdf5')

Raises:
    ValueError: If format_type is not supported
    RuntimeError: If I/O error occurs

Examples:
    >>> hamiltonian.to_file("water.hamiltonian.json", "json")
    >>> hamiltonian.to_file("molecule.hamiltonian.h5", "hdf5")
)",
                  py::arg("filename"), py::arg("type"));

  hamiltonian.def_static("from_file", hamiltonian_from_file_wrapper, R"(
Load Hamiltonian from file based on type parameter.

Args:
    filename (str or pathlib.Path): Path to file to read
    type (str): File format type ('json' or 'hdf5')

Returns:
    Hamiltonian: New Hamiltonian loaded from file

Raises:
    ValueError: If format_type is not supported
    RuntimeError: If file doesn't exist or I/O error occurs

Examples:
    >>> hamiltonian = Hamiltonian.from_file("water.hamiltonian.json", "json")
    >>> hamiltonian = Hamiltonian.from_file("molecule.hamiltonian.h5", "hdf5")
)",
                         py::arg("filename"), py::arg("type"));

  // JSON file I/O
  hamiltonian.def("to_json_file", hamiltonian_to_json_file_wrapper, R"(
Save Hamiltonian to JSON file (with validation).

Args:
    filename (str or pathlib.Path): Path to JSON file to create/overwrite.
        Must have ``.hamiltonian`` before the file extension.

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If an I/O error occurs

Examples:
    >>> hamiltonian.to_json_file("water.hamiltonian.json")
)",
                  py::arg("filename"));

  hamiltonian.def_static("from_json_file", hamiltonian_from_json_file_wrapper,
                         R"(
Load Hamiltonian from JSON file.

Args:
    filename (str or pathlib.Path): Path to JSON file to read.
        Must have ``.hamiltonian`` before the file extension.

Returns:
    Hamiltonian: New Hamiltonian loaded from file

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If file doesn't exist or I/O error occurs

Examples:
    >>> hamiltonian = Hamiltonian.from_json_file("water.hamiltonian.json")
)",
                         py::arg("filename"));

  // HDF5 file I/O
  hamiltonian.def("to_hdf5_file", hamiltonian_to_hdf5_file_wrapper, R"(
Save Hamiltonian to HDF5 file (with validation).

Args:
    filename (str or pathlib.Path): Path to HDF5 file to create/overwrite.
        Must have ``.hamiltonian`` before the file extension.

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If I/O error occurs

Examples:
    >>> hamiltonian.to_hdf5_file("water.hamiltonian.h5")
)",
                  py::arg("filename"));

  hamiltonian.def_static("from_hdf5_file", hamiltonian_from_hdf5_file_wrapper,
                         R"(
Load Hamiltonian from HDF5 file.

Args:
    filename (str or pathlib.Path): Path to HDF5 file to read.
        Must have ``.hamiltonian`` before the file extension.

Returns:
    Hamiltonian: New Hamiltonian loaded from file

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If file doesn't exist or I/O error occurs

Examples:
    >>> hamiltonian = Hamiltonian.from_hdf5_file("water.hamiltonian.h5")
)",
                         py::arg("filename"));

  // FCIDUMP file I/O
  hamiltonian.def("to_fcidump_file", hamiltonian_to_fcidump_file_wrapper,
                  R"(
Save Hamiltonian to FCIDUMP file.

Args:
    filename (str or pathlib.Path): Path to FCIDUMP file to create/overwrite
    nalpha (int): Number of alpha electrons
    nbeta (int): Number of beta electrons

Raises:
    RuntimeError: If I/O error occurs

Examples:
    >>> hamiltonian.to_fcidump_file("water.fcidump", 5, 5)

Notes:
    FCIDUMP format is a standard quantum chemistry format for storing
    molecular integrals and is widely supported by quantum chemistry codes.
)",
                  py::arg("filename"), py::arg("nalpha"), py::arg("nbeta"));

  // JSON serialization
  hamiltonian.def(
      "to_json",
      [](const Hamiltonian& self) -> std::string {
        return self.to_json().dump();
      },
      R"(
Convert Hamiltonian to JSON string.

Returns:
    str: JSON representation of the Hamiltonian
)");

  hamiltonian.def_static(
      "from_json",
      [](const std::string& json_str) -> Hamiltonian {
        return *Hamiltonian::from_json(nlohmann::json::parse(json_str));
      },
      R"(
Load Hamiltonian from JSON string.

Args:
    json_data (str): JSON string containing Hamiltonian data

Returns:
    Hamiltonian: New Hamiltonian loaded from JSON

Raises:
    RuntimeError: If JSON is malformed or contains invalid data

Examples:
    >>> json_str = hamiltonian.to_json()
    >>> loaded = Hamiltonian.from_json(json_str)
)",
      py::arg("json_data"));

  // String representation
  hamiltonian.def("__repr__",
                  [](const Hamiltonian& h) { return h.get_summary(); });

  hamiltonian.def("__str__",
                  [](const Hamiltonian& h) { return h.get_summary(); });

  // Pickling support using JSON serialization
  hamiltonian.def(py::pickle(
      [](const Hamiltonian& h) {
        // __getstate__ - serialize to JSON string
        return h.to_json().dump();
      },
      [](const std::string& json_str) {
        // __setstate__ - deserialize from JSON string
        return *Hamiltonian::from_json(nlohmann::json::parse(json_str));
      }));

  // Data type name class attribute
  hamiltonian.attr("_data_type_name") = DATACLASS_TO_SNAKE_CASE(Hamiltonian);
}

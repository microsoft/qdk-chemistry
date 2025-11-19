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
        aaaa: Alpha-alpha-alpha-alpha spin channel
        aabb: Alpha-beta-alpha-beta spin channel
        bbbb: Beta-beta-beta-beta spin channel
    )")
      .value("aaaa", SpinChannel::aaaa)
      .value("aabb", SpinChannel::aabb)
      .value("bbbb", SpinChannel::bbbb);

  // Hamiltonian class
  py::class_<Hamiltonian, DataClass, py::smart_holder> hamiltonian(
      data, "Hamiltonian", R"(
      Represents a molecular Hamiltonian in the molecular orbital basis.

    This class stores and manipulates molecular Hamiltonian data for quantum chemistry
    calculations, specifically designed for active space methods. It contains:

    * One-electron integrals (kinetic + nuclear attraction) in MO representation
    * Two-electron integrals (electron-electron repulsion) in MO representation
    * Molecular orbital information for the active space
    * Indices of selected orbitals defining the active space
    * Number of electrons in the selected MO space
    * Core energy contributions from inactive orbitals and nuclear repulsion
    )");

  // Constructors
  hamiltonian.def(
      py::init<const Eigen::MatrixXd&, const Eigen::VectorXd&,
               std::shared_ptr<qdk::chemistry::data::Orbitals>, double,
               const Eigen::MatrixXd&, qdk::chemistry::data::HamiltonianType>(),
      R"(
        Constructor for restricted active space Hamiltonian.

        ``norb`` denotes the number of orbitals in the active space.

        Parameters
        ----------
        one_body_integrals : numpy.ndarray
            One-electron integrals matrix [norb x norb]
        two_body_integrals : numpy.ndarray
            Two-electron integrals vector [norb^4]
        orbitals : Orbitals
            Molecular orbital data
        core_energy : float
            Core energy (nuclear repulsion + inactive orbitals)
        inactive_fock_matrix : numpy.ndarray
            Inactive Fock matrix for the selected active space
        type : HamiltonianType, optional
            Type of Hamiltonian (Hermitian by default)

        Examples
        --------
        >>> import numpy as np
        >>> one_body = np.random.rand(4, 4)  # 4 orbitals
        >>> two_body = np.random.rand(256)   # 4^4 elements
        >>> fock_matrix = np.random.rand(4, 4)
        >>> hamiltonian = Hamiltonian(one_body, two_body, orbitals, 10.5, fock_matrix)
        )",
      py::arg("one_body_integrals"), py::arg("two_body_integrals"),
      py::arg("orbitals"), py::arg("core_energy"),
      py::arg("inactive_fock_matrix"),
      py::arg("type") = qdk::chemistry::data::HamiltonianType::Hermitian);

  hamiltonian.def(
      py::init<const Eigen::MatrixXd&, const Eigen::MatrixXd&,
               const Eigen::VectorXd&, const Eigen::VectorXd&,
               const Eigen::VectorXd&,
               std::shared_ptr<qdk::chemistry::data::Orbitals>, double,
               const Eigen::MatrixXd&, const Eigen::MatrixXd&,
               qdk::chemistry::data::HamiltonianType>(),
      R"(
        Constructor for unrestricted active space Hamiltonian.

        ``norb`` denotes the number of orbitals in the active space.

        Parameters
        ----------
        one_body_integrals_alpha : numpy.ndarray
            Alpha one-electron integrals matrix [norb x norb]
        one_body_integrals_beta : numpy.ndarray
            Beta one-electron integrals matrix [norb x norb]
        two_body_integrals_aaaa : numpy.ndarray
            Alpha-alpha-alpha-alpha two-electron integrals vector
        two_body_integrals_aabb : numpy.ndarray
            Alpha-beta-alpha-beta two-electron integrals vector
        two_body_integrals_bbbb : numpy.ndarray
            Beta-beta-beta-beta two-electron integrals vector
        orbitals : Orbitals
            Molecular orbital data
        core_energy : float
            Core energy (nuclear repulsion + inactive orbitals)
        inactive_fock_matrix_alpha : numpy.ndarray
            Alpha inactive Fock matrix for the selected active space
        inactive_fock_matrix_beta : numpy.ndarray
            Beta inactive Fock matrix for the selected active space
        type : HamiltonianType, optional
            Type of Hamiltonian (Hermitian by default)

        Examples
        --------
        >>> import numpy as np
        >>> one_body_a = np.random.rand(4, 4)
        >>> one_body_b = np.random.rand(4, 4)
        >>> two_body_aaaa = np.random.rand(256)
        >>> two_body_aabb = np.random.rand(256)
        >>> two_body_bbbb = np.random.rand(256)
        >>> fock_a = np.random.rand(4, 4)
        >>> fock_b = np.random.rand(4, 4)
        >>> hamiltonian = Hamiltonian(one_body_a, one_body_b, two_body_aaaa, two_body_aabb, two_body_bbbb, orbitals, 10.5, fock_a, fock_b)
        )",
      py::arg("one_body_integrals_alpha"), py::arg("one_body_integrals_beta"),
      py::arg("two_body_integrals_aaaa"), py::arg("two_body_integrals_aabb"),
      py::arg("two_body_integrals_bbbb"), py::arg("orbitals"),
      py::arg("core_energy"), py::arg("inactive_fock_matrix_alpha"),
      py::arg("inactive_fock_matrix_beta"),
      py::arg("type") = qdk::chemistry::data::HamiltonianType::Hermitian);

  // One-body integral access
  bind_getter_as_property(hamiltonian, "get_one_body_integrals",
                          &Hamiltonian::get_one_body_integrals,
                          R"(
        Get one-electron integrals in molecular orbital basis.

        Returns
        -------
        numpy.ndarray
            One-electron integral matrix [norb x norb] containing
            kinetic energy and nuclear attraction integrals

        Raises
        ------
        RuntimeError
            If one-body integrals have not been set

        Examples
        --------
        >>> h1 = hamiltonian.get_one_body_integrals()
        >>> print(f"One-body matrix shape: {h1.shape}")
        >>> print(f"Diagonal element h[0,0] = {h1[0,0]}")
        )",
                          py::return_value_policy::reference_internal);
  hamiltonian.def("has_one_body_integrals",
                  &Hamiltonian::has_one_body_integrals,
                  R"(
        Check if one-body integrals are available.

        Returns
        -------
        bool
            True if one-body integrals have been set, False otherwise

        Examples
        --------
        >>> if hamiltonian.has_one_body_integrals():
        ...     h1 = hamiltonian.get_one_body_integrals()
        ... else:
        ...     print("One-body integrals not available")
        )");

  // Two-body integral access
  bind_getter_as_property(hamiltonian, "get_two_body_integrals",
                          &Hamiltonian::get_two_body_integrals,
                          R"(
        Get two-electron integrals in molecular orbital basis.

        Returns
        -------
        numpy.ndarray
            Two-electron integral vector [norb^4] containing
            electron-electron repulsion integrals stored in chemist notation

        Raises
        ------
        RuntimeError
            If two-body integrals have not been set

        Notes
        -----
        The integrals are stored as a flattened vector in chemist notation
        <ij|kl> where the indices are ordered as i + j*norb + k*norb^2 + l*norb^3

        Examples
        --------
        >>> h2 = hamiltonian.get_two_body_integrals()
        >>> print(f"Two-body vector length: {len(h2)}")
        >>> norb = hamiltonian.get_num_orbitals()
        >>> print(f"Expected length: {norb**4}")
        )",
                          py::return_value_policy::reference_internal);
  hamiltonian.def("get_two_body_element", &Hamiltonian::get_two_body_element,
                  R"(
        Get specific two-electron integral element <ij|kl>.

        Parameters
        ----------
        i, j, k, l : int
            Orbital indices for the two-electron integral

        Returns
        -------
        float
            Value of the two-electron integral <ij|kl>

        Examples
        --------
        >>> integral = hamiltonian.get_two_body_element(0, 1, 2, 3)
        >>> print(f"<01|23> = {integral}")
        )",
                  py::arg("i"), py::arg("j"), py::arg("k"), py::arg("l"),
                  py::arg("channel") = SpinChannel::aaaa);

  hamiltonian.def("has_two_body_integrals",
                  &Hamiltonian::has_two_body_integrals,
                  R"(
        Check if two-body integrals are available.

        Returns
        -------
        bool
            True if two-body integrals have been set, False otherwise

        Examples
        --------
        >>> if hamiltonian.has_two_body_integrals():
        ...     integrals = hamiltonian.get_two_body_integrals()
        ... else:
        ...     print("Two-body integrals not available")
        )");

  // Orbital information
  bind_getter_as_property(hamiltonian, "get_orbitals",
                          &Hamiltonian::get_orbitals,
                          R"(
        Get molecular orbital data.

        Returns
        -------
        Orbitals
            The orbital data associated with this Hamiltonian

        Raises
        ------
        RuntimeError
            If orbital data has not been set

        Examples
        --------
        >>> orbitals = hamiltonian.get_orbitals()
        >>> print(f"Number of MOs: {orbitals.get_num_molecular_orbitals()}")
        )",
                          py::return_value_policy::reference_internal);

  hamiltonian.def("has_orbitals", &Hamiltonian::has_orbitals,
                  R"(
        Check if orbital data is available.

        Returns
        -------
        bool
            True if orbital data has been set, False otherwise

        Examples
        --------
        >>> if hamiltonian.has_orbitals():
        ...     orbitals = hamiltonian.get_orbitals()
        ... else:
        ...     print("Orbital data not available")
        )");

  bind_getter_as_property(hamiltonian, "get_core_energy",
                          &Hamiltonian::get_core_energy,
                          R"(
        Get core energy in atomic units.

        Returns
        -------
        float
            Core energy contribution in Hartree

        Examples
        --------
        >>> e_core = hamiltonian.get_core_energy()
        >>> print(f"Core energy: {e_core} hartree")
        )");

  bind_getter_as_property(hamiltonian, "get_summary", &Hamiltonian::get_summary,
                          R"(
        Get a human-readable summary of the Hamiltonian data.

        Returns
        -------
        str
            Multi-line string describing the Hamiltonian properties
            including number of orbitals, electrons, core energy, etc.

        Examples
        --------
        >>> summary = hamiltonian.get_summary()
        >>> print(summary)
        # Output shows dimensions, energies, and validity status
        )");

  // Generic file I/O
  hamiltonian.def("to_file", hamiltonian_to_file_wrapper, R"(
        Save Hamiltonian to file based on type parameter.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to file to create/overwrite
        type : str
            File format type ('json' or 'hdf5')

        Raises
        ------
        ValueError
            If format_type is not supported
        RuntimeError
            If I/O error occurs

        Examples
        --------
        >>> hamiltonian.to_file("water.hamiltonian.json", "json")
        >>> hamiltonian.to_file("molecule.hamiltonian.h5", "hdf5")
        >>> from pathlib import Path
        >>> hamiltonian.to_file(Path("water.hamiltonian.json"), "json")
        )",
                  py::arg("filename"), py::arg("type"));

  hamiltonian.def_static("from_file", hamiltonian_from_file_wrapper, R"(
        Load Hamiltonian from file based on type parameter (static method).

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to file to read
        type : str
            File format type ('json' or 'hdf5')

        Returns
        -------
        Hamiltonian
            New Hamiltonian loaded from file

        Raises
        ------
        ValueError
            If format_type is not supported or filename doesn't follow naming convention
        RuntimeError
            If file doesn't exist or I/O error occurs

        Examples
        --------
        >>> hamiltonian = Hamiltonian.from_file("water.hamiltonian.json", "json")
        >>> hamiltonian = Hamiltonian.from_file("molecule.hamiltonian.h5", "hdf5")
        >>> from pathlib import Path
        >>> hamiltonian = Hamiltonian.from_file(Path("water.hamiltonian.json"), "json")
        )",
                         py::arg("filename"), py::arg("type"));

  // JSON file I/O
  hamiltonian.def("to_json_file", hamiltonian_to_json_file_wrapper, R"(
        Save Hamiltonian to JSON file (with validation).

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to JSON file to create/overwrite. Must have '.hamiltonian' before the file
            extension (e.g., ``water.hamiltonian.json``, ``molecule.hamiltonian.json``)

        Raises
        ------
        ValueError
            If filename doesn't follow the required naming convention
        RuntimeError
            If I/O error occurs

        Examples
        --------
        >>> hamiltonian.to_json_file("water.hamiltonian.json")
        >>> hamiltonian.to_json_file("molecule.hamiltonian.json")
        >>> from pathlib import Path
        >>> hamiltonian.to_json_file(Path("water.hamiltonian.json"))
        )",
                  py::arg("filename"));

  hamiltonian.def_static("from_json_file", hamiltonian_from_json_file_wrapper,
                         R"(
        Load Hamiltonian from JSON file (static method with validation).

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to JSON file to read. Must have '.hamiltonian' before the file
            extension (e.g., ``water.hamiltonian.json``, ``molecule.hamiltonian.json``)

        Returns
        -------
        Hamiltonian
            New Hamiltonian loaded from file

        Raises
        ------
        ValueError
            If filename doesn't follow the required naming convention
        RuntimeError
            If file doesn't exist or I/O error occurs

        Examples
        --------
        >>> hamiltonian = Hamiltonian.from_json_file("water.hamiltonian.json")
        >>> hamiltonian = Hamiltonian.from_json_file("molecule.hamiltonian.json")
        >>> from pathlib import Path
        >>> hamiltonian = Hamiltonian.from_json_file(Path("water.hamiltonian.json"))
        )",
                         py::arg("filename"));

  // HDF5 file I/O
  hamiltonian.def("to_hdf5_file", hamiltonian_to_hdf5_file_wrapper, R"(
        Save Hamiltonian to HDF5 file (with validation).

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to HDF5 file to create/overwrite. Must have '.hamiltonian' before the file
            extension (e.g., ``water.hamiltonian.h5``, ``molecule.hamiltonian.hdf5``)

        Raises
        ------
        ValueError
            If filename doesn't follow the required naming convention
        RuntimeError
            If I/O error occurs

        Examples
        --------
        >>> hamiltonian.to_hdf5_file("water.hamiltonian.h5")
        >>> hamiltonian.to_hdf5_file("molecule.hamiltonian.hdf5")
        >>> from pathlib import Path
        >>> hamiltonian.to_hdf5_file(Path("water.hamiltonian.h5"))
        )",
                  py::arg("filename"));

  hamiltonian.def_static("from_hdf5_file", hamiltonian_from_hdf5_file_wrapper,
                         R"(
        Load Hamiltonian from HDF5 file (static method with validation).

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to HDF5 file to read. Must have '.hamiltonian' before the file
            extension (e.g., ``water.hamiltonian.h5``, ``molecule.hamiltonian.hdf5``)

        Returns
        -------
        Hamiltonian
            New Hamiltonian loaded from file

        Raises
        ------
        ValueError
            If filename doesn't follow the required naming convention
        RuntimeError
            If file doesn't exist or I/O error occurs

        Examples
        --------
        >>> hamiltonian = Hamiltonian.from_hdf5_file("water.hamiltonian.h5")
        >>> hamiltonian = Hamiltonian.from_hdf5_file("molecule.hamiltonian.hdf5")
        >>> from pathlib import Path
        >>> hamiltonian = Hamiltonian.from_hdf5_file(Path("water.hamiltonian.h5"))
        )",
                         py::arg("filename"));

  // FCIDUMP file I/O
  hamiltonian.def("to_fcidump_file", hamiltonian_to_fcidump_file_wrapper,
                  R"(
        Save Hamiltonian to FCIDUMP file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to FCIDUMP file to create/overwrite. Typically uses '.fcidump'
            extension (e.g., ``water.fcidump``, ``molecule.fcidump``)
        nalpha : int
            Number of alpha electrons
        nbeta : int
            Number of beta electrons

        Raises
        ------
        RuntimeError
            If I/O error occurs

        Examples
        --------
        >>> hamiltonian.to_fcidump_file("water.fcidump", 5, 5)
        >>> hamiltonian.to_fcidump_file("molecule.fcidump", 7, 5)
        >>> from pathlib import Path
        >>> hamiltonian.to_fcidump_file(Path("water.fcidump"), 5, 5)

        Notes
        -----
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
        Convert Hamiltonian to JSON object.

        Returns:
            str: JSON-serializable string containing Hamiltonian data
        )");

  hamiltonian.def_static(
      "from_json",
      [](const std::string& json_str) -> Hamiltonian {
        return *Hamiltonian::from_json(nlohmann::json::parse(json_str));
      },
      R"(
        Load Hamiltonian from JSON string (static method).

        Parameters
        ----------
        json_data : str
            JSON string containing Hamiltonian data

        Returns
        -------
        Hamiltonian
            New Hamiltonian loaded from JSON

        Raises
        ------
        RuntimeError
            If JSON is malformed or contains invalid Hamiltonian data

        Examples
        --------
        >>> json_str = '{"num_orbitals": 2, "num_electrons": 2, ...}'
        >>> hamiltonian = Hamiltonian.from_json(json_str)
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
}

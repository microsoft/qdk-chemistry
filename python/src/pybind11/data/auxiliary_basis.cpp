// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/auxiliary_basis.hpp>
#include <utility>
#include <vector>

#include "path_utils.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;

namespace {

std::vector<Shell> auxiliary_shells_from_iterable(const py::iterable& shells) {
  py::list list(shells);
  const py::ssize_t size = py::len(list);
  std::vector<Shell> result;
  result.reserve(static_cast<size_t>(size));
  for (py::ssize_t index = 0; index < size; ++index) {
    result.push_back(
        py::reinterpret_borrow<py::object>(PyList_GET_ITEM(list.ptr(), index))
            .cast<Shell>());
  }
  return result;
}

void auxiliary_basis_to_file(AuxiliaryBasis& self, const py::object& filename,
                             const std::string& type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename), type);
}

std::shared_ptr<AuxiliaryBasis> auxiliary_basis_from_file(
    const py::object& filename, const std::string& type) {
  return AuxiliaryBasis::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), type);
}

void auxiliary_basis_to_json_file(AuxiliaryBasis& self,
                                  const py::object& filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<AuxiliaryBasis> auxiliary_basis_from_json_file(
    const py::object& filename) {
  return AuxiliaryBasis::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void auxiliary_basis_to_hdf5_file(AuxiliaryBasis& self,
                                  const py::object& filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<AuxiliaryBasis> auxiliary_basis_from_hdf5_file(
    const py::object& filename) {
  return AuxiliaryBasis::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

}  // namespace

void bind_auxiliary_basis(py::module& module) {
  py::enum_<AuxiliaryBasisRole>(
      module, "AuxiliaryBasisRole",
      "Algorithm-facing purpose served by an auxiliary basis")
      .value("JFIT", AuxiliaryBasisRole::JFit, "Coulomb density-fitting basis")
      .value("JKFIT", AuxiliaryBasisRole::JKFit,
             "Coulomb and exchange density-fitting basis")
      .value("RIFIT", AuxiliaryBasisRole::RIFit, "Correlation-fitting basis")
      .value("CABS", AuxiliaryBasisRole::CABS, "Complementary auxiliary basis");

  py::class_<AuxiliaryBasis, DataClass, py::smart_holder> auxiliary_basis(
      module, "AuxiliaryBasis",
      R"(
Secondary atom-centered Gaussian basis supplied to algorithms that require one.

An auxiliary basis is independent of the primary :class:`BasisSet` and carries its own
shells, orbital representation and molecular structure. Density fitting is the most common
consumer, but the class holds no algorithm-specific role, so a single calculation may use
several auxiliary bases for different purposes.

Examples:
    >>> from qdk_chemistry.data import AuxiliaryBasis, Structure
    >>> structure = Structure.from_xyz_file("water.xyz")
    >>> aux = AuxiliaryBasis.from_basis_name("def2-universal-jfit", structure)
    >>> print(f"Auxiliary shells: {aux.get_num_shells()}")
)");

  auxiliary_basis
      .def(py::init([](const py::iterable& shells,
                       std::shared_ptr<Structure> structure, AOType ao_type) {
             return std::make_shared<AuxiliaryBasis>(
                 auxiliary_shells_from_iterable(shells), std::move(structure),
                 ao_type);
           }),
           py::arg("shells"), py::arg("structure"),
           py::arg("atomic_orbital_type") = AOType::Spherical,
           R"(
Create a custom auxiliary basis from shells and a molecular structure.

The basis is named ``custom_aux``. Shells are stored in ascending atom order,
and within each atom by angular momentum and decreasing exponent.

Args:
    shells (Iterable[Shell]): Auxiliary shells. Must not carry radial powers
    structure (Structure): Molecular structure the shells refer to
    atomic_orbital_type (AOType, optional): Whether to use spherical or Cartesian atomic orbitals.
        Default is Spherical

Raises:
    ValueError: If shells are empty, reference an atom outside the structure, carry radial
        powers, or use the ECP local-potential orbital type

Examples:
    >>> aux = AuxiliaryBasis([Shell(0, OrbitalType.S, [5.0], [2.0])], structure)
    >>> print(aux.get_name())  # custom_aux
)")
      .def(py::init([](const std::string& name, const py::iterable& shells,
                       std::shared_ptr<Structure> structure, AOType ao_type) {
             return std::make_shared<AuxiliaryBasis>(
                 name, auxiliary_shells_from_iterable(shells),
                 std::move(structure), ao_type);
           }),
           py::arg("name"), py::arg("shells"), py::arg("structure"),
           py::arg("atomic_orbital_type") = AOType::Spherical,
           R"(
Create a named auxiliary basis from shells and a molecular structure.

Args:
    name (str): Auxiliary basis name (e.g., "def2-universal-jfit")
    shells (Iterable[Shell]): Auxiliary shells. Must not carry radial powers
    structure (Structure): Molecular structure the shells refer to
    atomic_orbital_type (AOType, optional): Whether to use spherical or Cartesian atomic orbitals.
        Default is Spherical

Raises:
    ValueError: If the name is empty, shells are empty, or shells are invalid for the structure

Examples:
    >>> aux = AuxiliaryBasis("my-aux-fit", aux_shells, structure)
    >>> print(aux.get_name())  # my-aux-fit
)")
      .def("get_name", &AuxiliaryBasis::get_name,
           R"(
Get the auxiliary basis name.

Returns:
    str: Name of the auxiliary basis, or ``custom_aux`` for unnamed custom data
)")
      .def("get_atomic_orbital_type", &AuxiliaryBasis::get_atomic_orbital_type,
           R"(
Get the atomic orbital type.

Returns:
    AOType: Spherical or Cartesian
)")
      .def("get_structure", &AuxiliaryBasis::get_structure,
           R"(
Get the molecular structure this auxiliary basis is defined for.

Returns:
    Structure: The associated molecular structure
)")
      .def("get_shells", &AuxiliaryBasis::get_shells,
           R"(
Get all auxiliary shells, flattened from per-atom storage.

Returns:
    list[Shell]: All auxiliary shells in canonical atom order

Examples:
    >>> shells = aux.get_shells()
    >>> print(f"Total auxiliary shells: {len(shells)}")
)")
      .def("get_shells_for_atom", &AuxiliaryBasis::get_shells_for_atom,
           R"(
Get the auxiliary shells for a specific atom.

Args:
    atom_index (int): Index of the atom

Returns:
    list[Shell]: Auxiliary shells for the atom, empty if the atom carries none

Raises:
    IndexError: If the atom index is out of range
)",
           py::arg("atom_index"))
      .def(
          "get_shell",
          [](const AuxiliaryBasis& self, size_t shell_index) {
            return self.get_shell(shell_index);
          },
          R"(
Get a specific auxiliary shell by global index.

Args:
    shell_index (int): Global index of the auxiliary shell

Returns:
    Shell: The requested auxiliary shell

Raises:
    IndexError: If the shell index is out of range
)",
          py::arg("shell_index"))
      .def("get_num_shells", &AuxiliaryBasis::get_num_shells,
           R"(
Get the total number of auxiliary shells across all atoms.

Returns:
    int: Total number of auxiliary shells
)")
      .def("get_num_atoms", &AuxiliaryBasis::get_num_atoms,
           R"(
Get the number of atoms the auxiliary basis is blocked over.

Returns:
    int: Number of atoms
)")
      .def("get_num_auxiliary_orbitals",
           &AuxiliaryBasis::get_num_auxiliary_orbitals,
           R"(
Get the total number of auxiliary orbitals.

The count depends on the atomic orbital type: a d shell contributes 5 spherical
or 6 Cartesian orbitals.

Returns:
    int: Total number of auxiliary orbitals from all shells
)")
      .def_static("from_basis_name", &AuxiliaryBasis::from_basis_name,
                  R"(
Create an auxiliary basis by name for a molecular structure.

Loads a standard auxiliary basis (e.g., "def2-universal-jfit") for all atoms in the structure.

Args:
    basis_name (str): Name of the auxiliary basis set
    structure (Structure): Molecular structure
    atomic_orbital_type (AOType, optional): Whether to use spherical or Cartesian atomic orbitals.
        Default is Spherical

Returns:
    AuxiliaryBasis: New auxiliary basis instance

Raises:
    ValueError: If the auxiliary basis name is not recognized or the structure is invalid

Examples:
    >>> aux = AuxiliaryBasis.from_basis_name("def2-universal-jfit", structure)
)",
                  py::arg("basis_name"), py::arg("structure"),
                  py::arg("atomic_orbital_type") = AOType::Spherical)
      .def_static("from_element_map", &AuxiliaryBasis::from_element_map,
                  R"(
Create an auxiliary basis with a different auxiliary basis per element.

Args:
    element_to_basis_map (dict[str, str]): Mapping from element symbols to auxiliary basis names
    structure (Structure): Molecular structure
    atomic_orbital_type (AOType, optional): Whether to use spherical or Cartesian atomic orbitals.
        Default is Spherical

Returns:
    AuxiliaryBasis: New auxiliary basis instance named ``custom_aux``

Raises:
    ValueError: If an element in the structure is missing from the map or a name is invalid

Examples:
    >>> aux_map = {"H": "def2-universal-jfit", "O": "def2-universal-jfit"}
    >>> aux = AuxiliaryBasis.from_element_map(aux_map, structure)
)",
                  py::arg("element_to_basis_map"), py::arg("structure"),
                  py::arg("atomic_orbital_type") = AOType::Spherical)
      .def_static("from_index_map", &AuxiliaryBasis::from_index_map,
                  R"(
Create an auxiliary basis with a different auxiliary basis per atom index.

Args:
    index_to_basis_map (dict[int, str]): Mapping from atom indices to auxiliary basis names
    structure (Structure): Molecular structure
    atomic_orbital_type (AOType, optional): Whether to use spherical or Cartesian atomic orbitals.
        Default is Spherical

Returns:
    AuxiliaryBasis: New auxiliary basis instance named ``custom_aux``

Raises:
    ValueError: If an atom index is missing from the map or a name is invalid

Examples:
    >>> aux_map = {0: "def2-universal-jfit", 1: "def2-universal-jfit"}
    >>> aux = AuxiliaryBasis.from_index_map(aux_map, structure)
)",
                  py::arg("index_to_basis_map"), py::arg("structure"),
                  py::arg("atomic_orbital_type") = AOType::Spherical)
      .def(
          "to_json",
          [](const AuxiliaryBasis& self) { return self.to_json().dump(); },
          R"(
Convert the auxiliary basis to a JSON string.

Returns:
    str: JSON representation, including the associated structure
)")
      .def_static(
          "from_json",
          [](const std::string& json) {
            return AuxiliaryBasis::from_json(nlohmann::json::parse(json));
          },
          R"(
Load an auxiliary basis from a JSON string.

Args:
    json (str): JSON string produced by ``to_json()``

Returns:
    AuxiliaryBasis: New instance loaded from JSON

Raises:
    RuntimeError: If the JSON is malformed or contains invalid auxiliary basis data
)",
          py::arg("json"))
      .def("to_file", auxiliary_basis_to_file,
           R"(
Save the auxiliary basis to a file in the specified format.

Args:
    filename (str | pathlib.Path): Destination path. Must have '.auxiliary_basis' before the
        file extension (e.g., ``water.auxiliary_basis.json``)
    type (str): File format type ("json" or "hdf5")

Raises:
    ValueError: If the filename does not follow the naming convention
    RuntimeError: If the type is unsupported or the file cannot be written
)",
           py::arg("filename"), py::arg("type"))
      .def_static("from_file", auxiliary_basis_from_file,
                  R"(
Load an auxiliary basis from a file in the specified format.

Args:
    filename (str | pathlib.Path): Path to read. Must have '.auxiliary_basis' before the
        file extension
    type (str): File format type ("json" or "hdf5")

Returns:
    AuxiliaryBasis: New instance loaded from file

Raises:
    ValueError: If the filename does not follow the naming convention
    RuntimeError: If the type is unsupported or the file cannot be read
)",
                  py::arg("filename"), py::arg("type"))
      .def("to_json_file", auxiliary_basis_to_json_file,
           R"(
Save the auxiliary basis to a JSON file.

Args:
    filename (str | pathlib.Path): Destination path, e.g. ``water.auxiliary_basis.json``

Raises:
    ValueError: If the filename does not follow the naming convention
    RuntimeError: If the file cannot be written
)",
           py::arg("filename"))
      .def_static("from_json_file", auxiliary_basis_from_json_file,
                  R"(
Load an auxiliary basis from a JSON file.

Args:
    filename (str | pathlib.Path): Path to read, e.g. ``water.auxiliary_basis.json``

Returns:
    AuxiliaryBasis: New instance loaded from file

Raises:
    ValueError: If the filename does not follow the naming convention
    RuntimeError: If the file cannot be read or contains invalid data
)",
                  py::arg("filename"))
      .def("to_hdf5_file", auxiliary_basis_to_hdf5_file,
           R"(
Save the auxiliary basis to an HDF5 file.

Args:
    filename (str | pathlib.Path): Destination path, e.g. ``water.auxiliary_basis.h5``

Raises:
    ValueError: If the filename does not follow the naming convention
    RuntimeError: If the file cannot be written
)",
           py::arg("filename"))
      .def_static("from_hdf5_file", auxiliary_basis_from_hdf5_file,
                  R"(
Load an auxiliary basis from an HDF5 file.

Args:
    filename (str | pathlib.Path): Path to read, e.g. ``water.auxiliary_basis.h5``

Returns:
    AuxiliaryBasis: New instance loaded from file

Raises:
    ValueError: If the filename does not follow the naming convention
    RuntimeError: If the file cannot be read or contains invalid data
)",
                  py::arg("filename"))
      .def("__repr__",
           [](const AuxiliaryBasis& self) { return self.get_summary(); })
      .def("__str__",
           [](const AuxiliaryBasis& self) { return self.get_summary(); })
      .def(py::pickle(
          [](const AuxiliaryBasis& self) { return self.to_json().dump(); },
          [](const std::string& json) {
            return *AuxiliaryBasis::from_json(nlohmann::json::parse(json));
          }))
      .def_readonly_static("custom_name", &AuxiliaryBasis::custom_name,
                           R"(
Name assigned to custom auxiliary basis data.

Type:
    str
)");

  auxiliary_basis.attr("_data_type_name") =
      DATACLASS_TO_SNAKE_CASE(AuxiliaryBasis);

  py::class_<AuxiliaryBasisCollection, DataClass, py::smart_holder> collection(
      module, "AuxiliaryBasisCollection",
      R"(
Immutable collection of auxiliary bases keyed by their algorithm-facing roles.

The collection is independent of a primary basis set or wavefunction and is
intended for algorithms that explicitly accept auxiliary-basis inputs. It does
not itself enable density fitting or change algorithm behavior.
)");

  collection
      .def(py::init<>(), R"(
Create an empty auxiliary-basis collection.

The collection can be populated immutably with
:func:`with_auxiliary_basis`.
)")
      .def(py::init<AuxiliaryBasisCollection::Map>(),
           py::arg("auxiliary_bases"),
           R"(
Create a tagged auxiliary-basis collection.

Args:
    auxiliary_bases (dict[AuxiliaryBasisRole, AuxiliaryBasis]): Exact role associations

Raises:
    ValueError: If an entry is null or the bases describe different structures
)")
      .def("has_auxiliary_basis",
           &AuxiliaryBasisCollection::has_auxiliary_basis, py::arg("role"),
           R"(
Check whether an auxiliary basis is associated with an exact role.

This method does not apply compatibility fallbacks. For example, a collection
containing only ``JKFIT`` reports ``False`` for ``JFIT``.

Args:
    role (AuxiliaryBasisRole): Exact role to inspect

Returns:
    bool: Whether the exact role has an associated auxiliary basis
)")
      .def("get_auxiliary_basis",
           &AuxiliaryBasisCollection::get_auxiliary_basis, py::arg("role"),
           R"(
Get the auxiliary basis associated with an exact role.

Args:
    role (AuxiliaryBasisRole): Exact role to retrieve

Returns:
    AuxiliaryBasis: Basis associated with the exact role

Raises:
    IndexError: If the collection has no entry for the role
)")
      .def("resolve_auxiliary_basis",
           &AuxiliaryBasisCollection::resolve_auxiliary_basis, py::arg("role"),
           R"(
Resolve an auxiliary basis compatible with a required role.

Exact associations take precedence. A ``JKFIT`` basis may satisfy a ``JFIT``
request, but a ``JFIT`` basis cannot satisfy ``JKFIT``. Other roles require an
exact match.

Args:
    role (AuxiliaryBasisRole): Required auxiliary-basis role

Returns:
    AuxiliaryBasis: Exact or compatible auxiliary basis

Raises:
    IndexError: If no collection entry can satisfy the requested role
)")
      .def("get_auxiliary_bases",
           &AuxiliaryBasisCollection::get_auxiliary_bases,
           R"(
Get all exact auxiliary-basis associations.

Returns:
    dict[AuxiliaryBasisRole, AuxiliaryBasis]: Copy of the role-keyed mapping
)")
      .def(
          "to_json",
          [](const AuxiliaryBasisCollection& self) {
            return self.to_json().dump();
          },
          R"(
Serialize the collection to a JSON string.

Returns:
    str: Versioned JSON containing all exact role associations
)")
      .def_static(
          "from_json",
          [](const std::string& json) {
            return AuxiliaryBasisCollection::from_json(
                nlohmann::json::parse(json));
          },
          py::arg("json"),
          R"(
Deserialize an auxiliary-basis collection from JSON.

Args:
    json (str): JSON string produced by :meth:`to_json`

Returns:
    AuxiliaryBasisCollection: Deserialized collection

Raises:
    RuntimeError: If the JSON, version, or required fields are invalid
    ValueError: If an entry is null or structures differ
)")
      .def(
          "to_file",
          [](const AuxiliaryBasisCollection& self, const py::object& filename,
             const std::string& type) {
            self.to_file(
                qdk::chemistry::python::utils::to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"),
          R"(
Save the collection in a supported file format.

Args:
    filename (str | pathlib.Path): Destination path with an
        ``.auxiliary_basis_collection`` data-type suffix
    type (str): File format, either ``"json"`` or ``"hdf5"``

Raises:
    ValueError: If the filename suffix is invalid
    RuntimeError: If the format is unsupported or the file cannot be written
)")
      .def_static(
          "from_file",
          [](const py::object& filename, const std::string& type) {
            return AuxiliaryBasisCollection::from_file(
                qdk::chemistry::python::utils::to_string_path(filename), type);
          },
          py::arg("filename"), py::arg("type"),
          R"(
Load a collection from a supported file format.

Args:
    filename (str | pathlib.Path): Source path with an
        ``.auxiliary_basis_collection`` data-type suffix
    type (str): File format, either ``"json"`` or ``"hdf5"``

Returns:
    AuxiliaryBasisCollection: Deserialized collection

Raises:
    ValueError: If the filename suffix or a collection entry is invalid
    RuntimeError: If the format is unsupported or the file cannot be read
)")
      .def(
          "to_json_file",
          [](const AuxiliaryBasisCollection& self, const py::object& filename) {
            self.to_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"),
          R"(
Save the collection to a JSON file.

Args:
    filename (str | pathlib.Path): Destination path ending in
        ``.auxiliary_basis_collection.json``

Raises:
    ValueError: If the filename suffix is invalid
    RuntimeError: If the file cannot be written
)")
      .def_static(
          "from_json_file",
          [](const py::object& filename) {
            return AuxiliaryBasisCollection::from_json_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"),
          R"(
Load a collection from a JSON file.

Args:
    filename (str | pathlib.Path): Source path ending in
        ``.auxiliary_basis_collection.json``

Returns:
    AuxiliaryBasisCollection: Deserialized collection

Raises:
    ValueError: If the filename suffix or a collection entry is invalid
    RuntimeError: If the file or serialized JSON cannot be read
)")
      .def(
          "to_hdf5_file",
          [](const AuxiliaryBasisCollection& self, const py::object& filename) {
            self.to_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"),
          R"(
Save the collection to an HDF5 file.

Args:
    filename (str | pathlib.Path): Destination path ending in
        ``.auxiliary_basis_collection.h5``

Raises:
    ValueError: If the filename suffix is invalid
    RuntimeError: If the file cannot be written
)")
      .def_static(
          "from_hdf5_file",
          [](const py::object& filename) {
            return AuxiliaryBasisCollection::from_hdf5_file(
                qdk::chemistry::python::utils::to_string_path(filename));
          },
          py::arg("filename"),
          R"(
Load a collection from an HDF5 file.

Args:
    filename (str | pathlib.Path): Source path ending in
    ``.auxiliary_basis_collection.h5``

Returns:
    AuxiliaryBasisCollection: Deserialized collection

Raises:
    ValueError: If the filename suffix or a collection entry is invalid
    RuntimeError: If the file or serialized collection cannot be read
)")
      .def(
          "__repr__",
          [](const AuxiliaryBasisCollection& self) {
            return self.get_summary();
          },
          "Return the human-readable collection summary.")
      .def(
          "__str__",
          [](const AuxiliaryBasisCollection& self) {
            return self.get_summary();
          },
          "Return the human-readable collection summary.")
      .def(py::pickle(
          [](const AuxiliaryBasisCollection& self) {
            return self.to_json().dump();
          },
          [](const std::string& json) {
            return *AuxiliaryBasisCollection::from_json(
                nlohmann::json::parse(json));
          }));

  collection.attr("_data_type_name") =
      DATACLASS_TO_SNAKE_CASE(AuxiliaryBasisCollection);

  module.def(
      "with_auxiliary_basis",
      [](const AuxiliaryBasisCollection& collection, AuxiliaryBasisRole role,
         std::shared_ptr<AuxiliaryBasis> auxiliary_basis) {
        return with_auxiliary_basis(collection, role,
                                    std::move(auxiliary_basis));
      },
      py::arg("collection"), py::arg("role"), py::arg("auxiliary_basis"),
      R"(
Return an immutable collection with an auxiliary-basis association.

Args:
    collection (AuxiliaryBasisCollection): Collection to enrich
    role (AuxiliaryBasisRole): Role served by the auxiliary basis
    auxiliary_basis (AuxiliaryBasis): Auxiliary basis to associate

Returns:
    AuxiliaryBasisCollection: New collection; ``collection`` is unchanged

Raises:
    ValueError: If an entry is null or structures differ
)");
}

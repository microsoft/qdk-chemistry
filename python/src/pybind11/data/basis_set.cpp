// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/basis_set.hpp>
#include <qdk/chemistry/data/structure.hpp>

#include "path_utils.hpp"
#include "property_binding_helpers.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;

// Wrapper functions that accept both strings and pathlib.Path objects
namespace {

void basis_set_to_file_wrapper(BasisSet& self, const py::object& filename,
                               const std::string& format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

std::shared_ptr<BasisSet> basis_set_from_file_wrapper(
    const py::object& filename, const std::string& format_type) {
  return BasisSet::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

void basis_set_to_hdf5_file_wrapper(BasisSet& self,
                                    const py::object& filename) {
  self.to_hdf5_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<BasisSet> basis_set_from_hdf5_file_wrapper(
    const py::object& filename) {
  return BasisSet::from_hdf5_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void basis_set_to_json_file_wrapper(BasisSet& self,
                                    const py::object& filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<BasisSet> basis_set_from_json_file_wrapper(
    const py::object& filename) {
  return BasisSet::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

}  // namespace

void bind_basis_set(py::module& m) {
  using namespace qdk::chemistry::data;
  using qdk::chemistry::python::utils::bind_getter_as_property;

  // Bind OrbitalType enum
  py::enum_<OrbitalType>(m, "OrbitalType", "Enumeration of orbital types")
      .value("S", OrbitalType::S, "s orbital (l=0)")
      .value("P", OrbitalType::P, "p orbital (l=1)")
      .value("D", OrbitalType::D, "d orbital (l=2)")
      .value("F", OrbitalType::F, "f orbital (l=3)")
      .value("G", OrbitalType::G, "g orbital (l=4)")
      .value("H", OrbitalType::H, "h orbital (l=5)")
      .value("I", OrbitalType::I, "i orbital (l=6)")
      .export_values();

  // Bind AOType enum
  py::enum_<AOType>(m, "AOType", "Enumeration of atomic orbital types")
      .value("Spherical", AOType::Spherical,
             "Spherical harmonics (2l+1 functions per shell)")
      .value("Cartesian", AOType::Cartesian,
             "Cartesian coordinates (more functions for l>=2)")
      .export_values();

  // Bind Shell struct
  py::class_<Shell>(m, "Shell", "Shell of atomic orbitals")
      .def(py::init<size_t, OrbitalType, const Eigen::VectorXd&,
                    const Eigen::VectorXd&>(),
           R"(
Constructor with complete primitive data.

Creates a shell with all primitive functions specified.

Args:
    atom_index (int): Index of the atom this shell belongs to
    orbital_type (OrbitalType): Type of orbital (S, P, D, F, etc.)
    exponents (numpy.ndarray): Vector of Gaussian exponent coefficients (alpha values)
    coefficients (numpy.ndarray): Vector of contraction coefficients (must be same length as exponents)

Raises:
    ValueError: If exponents and coefficients have different lengths

Examples:
    >>> import numpy as np
    >>> exponents = np.array([1.0, 0.5, 0.1])
    >>> coefficients = np.array([0.444, 0.555, 0.222])
    >>> shell = Shell(0, OrbitalType.S, exponents, coefficients)
)",
           py::arg("atom_index"), py::arg("orbital_type"), py::arg("exponents"),
           py::arg("coefficients"))
      .def(py::init<size_t, OrbitalType, const Eigen::VectorXd&,
                    const Eigen::VectorXd&, const Eigen::VectorXi&>(),
           R"(
Constructor with primitive data and radial powers for ECP shells.

Creates an ECP (Effective Core Potential) shell with radial powers.

Args:
    atom_index (int): Index of the atom this shell belongs to
    orbital_type (OrbitalType): Type of orbital (S, P, D, F, etc.)
    exponents (numpy.ndarray): Vector of Gaussian exponent coefficients (alpha values)
    coefficients (numpy.ndarray): Vector of contraction coefficients (must be same length as exponents)
    rpowers (numpy.ndarray[int]): Vector of radial powers (r^n terms) for ECP shells

Raises:
    ValueError: If exponents, coefficients, and rpowers have different lengths

Examples:
    >>> import numpy as np
    >>> exponents = np.array([1.0, 0.5, 0.1])
    >>> coefficients = np.array([0.444, 0.555, 0.222])
    >>> rpowers = np.array([2, 1, 0], dtype=np.int32)
    >>> ecp_shell = Shell(0, OrbitalType.S, exponents, coefficients, rpowers)
)",
           py::arg("atom_index"), py::arg("orbital_type"), py::arg("exponents"),
           py::arg("coefficients"), py::arg("rpowers"))
      .def_readwrite("atom_index", &Shell::atom_index, "Index of the atom")
      .def_readwrite("orbital_type", &Shell::orbital_type, "Type of orbital")
      .def_readwrite("exponents", &Shell::exponents,
                     "Vector of orbital exponents")
      .def_readwrite("coefficients", &Shell::coefficients,
                     "Vector of contraction coefficients")
      .def_readwrite("rpowers", &Shell::rpowers,
                     "Vector of radial powers for ECP shells (r^n terms)")
      .def("get_num_primitives", &Shell::get_num_primitives,
           R"(
Get the number of primitive Gaussians in this shell.

Returns:
    int: Number of primitive functions

Examples:
    >>> n_prim = shell.get_num_primitives()
    >>> print(f"Shell has {n_prim} primitives")
)")
      .def("get_num_atomic_orbitals", &Shell::get_num_atomic_orbitals,
           R"(
Get number of atomic orbitals this shell contributes.

Args:
    atomic_orbital_type (Optional[AOType]): Whether to use spherical (2l+1) or Cartesian functions.
        Default is Spherical

Returns:
    int: Number of atomic orbitals

Examples:
    >>> # For p-shell: 3 spherical, 3 Cartesian
    >>> n_sph = shell.get_num_atomic_orbitals(AOType.Spherical)
    >>> n_cart = shell.get_num_atomic_orbitals(AOType.Cartesian)
    >>> print(f"Spherical: {n_sph}, Cartesian: {n_cart}")
)",
           py::arg("atomic_orbital_type") = AOType::Spherical)
      .def("get_angular_momentum", &Shell::get_angular_momentum,
           R"(
Get the angular momentum quantum number (l) for this shell.

Returns:
    int: Angular momentum quantum number (0=s, 1=p, 2=d, etc.)

Examples:
    >>> l = shell.get_angular_momentum()
    >>> print(f"Angular momentum l = {l}")
)")
      .def("has_radial_powers", &Shell::has_radial_powers,
           R"(
Check if this shell has radial powers (i.e., is an ECP shell).

Returns:
    bool: True if this shell has radial powers defined

Examples:
    >>> if shell.has_radial_powers():
    ...     print("This is an ECP shell")
)");

  py::class_<BasisSet, DataClass, py::smart_holder>(m, "BasisSet",
                                                    R"(
Represents an atomic orbital basis set using shell-based organization.

This class stores and manages atomic orbital basis set information using shells as the primary organizational unit.
A shell represents a group of atomic orbitals with the same atom, angular momentum, and primitives.

Examples:
    Create a simple basis set:

    >>> from qdk_chemistry.data import BasisSet, OrbitalType
    >>> basis = BasisSet("STO-3G")
    >>> basis.add_shell(0, OrbitalType.S, 1.0, 1.0)  # s orbital on atom 0
    >>> print(f"Number of atomic orbitals: {basis.get_num_atomic_orbitals()}")
)")
      .def(py::init<const std::string&, const Structure&, AOType>(),
           R"(
Constructor with basis set name, structure, and basis type.

Creates a basis set associated with a molecular structure.

Args:
    name (str): Name of the basis set
    structure (Structure): Molecular structure to associate with this basis set
    atomic_orbital_type (Optional[AOType]): Whether to use spherical or Cartesian atomic orbitals. Default is Spherical

Examples:
    >>> from qdk_chemistry.data import Structure
    >>> structure = Structure.from_xyz_file("water.xyz")
    >>> basis = BasisSet("cc-pVDZ", structure, AOType.Spherical)
    >>> print(f"Basis set for {structure.get_num_atoms()} atoms")
)",
           py::arg("name"), py::arg("structure"),
           py::arg("atomic_orbital_type") = AOType::Spherical)
      .def(py::init<const std::string&, const std::vector<Shell>&, AOType>(),
           R"(
Constructor with basis set name, shells, and basis type.

Creates a basis set with predefined shells.

Args:
    name (str): Name of the basis set
    shells (list[Shell]): Vector of shell objects defining the atomic orbitals
    atomic_orbital_type (Optional[AOType]): Whether to use spherical or Cartesian atomic orbitals.
        Default is Spherical.

Examples:
    >>> shells = [Shell(0, OrbitalType.S), Shell(0, OrbitalType.P)]
    >>> basis = BasisSet("custom", shells)
    >>> print(f"Created basis with {len(shells)} shells")
)",
           py::arg("name"), py::arg("shells"),
           py::arg("atomic_orbital_type") = AOType::Spherical)
      .def(py::init<const std::string&, const std::vector<Shell>&,
                    const Structure&, AOType>(),
           R"(
Constructor with basis set name, shells, structure, and basis type.

Creates a complete basis set with shells and molecular structure.

Args:
    name (str): Name of the basis set
    shells (list[Shell]): Vector of shell objects defining the atomic orbitals
    structure (Structure): Molecular structure to associate with this basis set
    atomic_orbital_type (Optional[AOType]): Whether to use spherical or Cartesian atomic orbitals.
        Default is Spherical

Examples:
    >>> from qdk_chemistry.data import Structure
    >>> structure = Structure.from_xyz_file("water.xyz")
    >>> shells = [Shell(0, OrbitalType.S), Shell(1, OrbitalType.S)]
    >>> basis = BasisSet("custom", shells, structure)
    >>> print(f"Complete basis set with {len(shells)} shells")
)",
           py::arg("name"), py::arg("shells"), py::arg("structure"),
           py::arg("atomic_orbital_type") = AOType::Spherical)
      .def(py::init<const std::string&, const std::vector<Shell>&,
                    const std::vector<Shell>&, const Structure&, AOType>(),
           R"(
Constructor with basis set name, shells, ECP shells, structure, and basis type.

Creates a complete basis set with regular shells, ECP shells, and molecular structure.

Args:
    name (str): Name of the basis set
    shells (list[Shell]): Vector of shell objects defining the atomic orbitals
    ecp_shells (list[Shell]): Vector of ECP shell objects
    structure (Structure): Molecular structure to associate with this basis set
    atomic_orbital_type (Optional[AOType]): Whether to use spherical or Cartesian atomic orbitals.
        Default is Spherical

Examples:
    >>> from qdk_chemistry.data import Structure
    >>> structure = Structure.from_xyz_file("water.xyz")
    >>> shells = [Shell(0, OrbitalType.S), Shell(1, OrbitalType.S)]
    >>> ecp_shells = [Shell(0, OrbitalType.S, exp, coeff, rpow)]
    >>> basis = BasisSet("custom-ecp", shells, ecp_shells, structure)
    >>> print(f"Basis with {len(shells)} shells and {len(ecp_shells)} ECP shells")
)",
           py::arg("name"), py::arg("shells"), py::arg("ecp_shells"),
           py::arg("structure"),
           py::arg("atomic_orbital_type") = AOType::Spherical)
      .def(py::init<const std::string&, const std::vector<Shell>&,
                    const std::string&, const std::vector<Shell>&,
                    const std::vector<size_t>&, const Structure&, AOType>(),
           R"(
Constructor with basis set name, shells, ECP name, ECP shells, ECP electrons, structure, and basis type.

Creates a complete basis set with regular shells, ECP shells, ECP metadata, and molecular structure.

Args:
    name (str): Name of the basis set
    shells (list[Shell]): Vector of shell objects defining the atomic orbitals
    ecp_name (str): Name of the ECP (basis set)
    ecp_shells (list[Shell]): Vector of ECP shell objects
    ecp_electrons (list[int]): Number of ECP electrons for each atom
    structure (Structure): Molecular structure to associate with this basis set
    atomic_orbital_type (Optional[AOType]): Whether to use spherical or Cartesian atomic orbitals.
        Default is Spherical

Examples:
    >>> from qdk_chemistry.data import Structure
    >>> structure = Structure.from_xyz_file("water.xyz")
    >>> shells = [Shell(0, OrbitalType.S), Shell(1, OrbitalType.S)]
    >>> ecp_shells = [Shell(0, OrbitalType.S, exp, coeff, rpow)]
    >>> ecp_electrons = [10, 10, 0]
    >>> basis = BasisSet("custom-ecp", shells, "custom-ecp", ecp_shells, ecp_electrons, structure)
    >>> print(f"Basis with {len(shells)} shells, {len(ecp_shells)} ECP shells, ECP: {basis.get_ecp_name()}")
)",
           py::arg("name"), py::arg("shells"), py::arg("ecp_name"),
           py::arg("ecp_shells"), py::arg("ecp_electrons"),
           py::arg("structure"),
           py::arg("atomic_orbital_type") = AOType::Spherical)
      .def(py::init<const BasisSet&>(),
           R"(
Copy constructor.

Creates a deep copy of another basis set.

Args:
    other (BasisSet): Basis set to copy

Examples:
    >>> original = BasisSet("cc-pVDZ")
    >>> copy = BasisSet(original)
    >>> print(f"Copied basis set: {copy.get_name()}")
)")

      // Basis type management
      .def("get_atomic_orbital_type", &BasisSet::get_atomic_orbital_type,
           R"(
Get the basis type.

Returns:
    AOType: Current basis type (Spherical or Cartesian)

Examples:
    >>> atomic_orbital_type = basis_set.get_atomic_orbital_type()
    >>> print(f"Basis type: {atomic_orbital_type}")
)")

      // Shell access (read-only)
      .def("get_shells", &BasisSet::get_shells,
           R"(
Get all shells (flattened from per-atom storage).

Returns:
    list[Shell]: Vector of all shells in the basis set

Examples:
    >>> shells = basis_set.get_shells()
    >>> print(f"Total shells: {len(shells)}")
)")

      .def("get_shells_for_atom", &BasisSet::get_shells_for_atom,
           R"(
Get shells for a specific atom.

Args:
    atom_index (int): Index of the atom

Returns:
    list[Shell]: Vector of shells for the specified atom

Examples:
    >>> atom_shells = basis_set.get_shells_for_atom(0)
    >>> print(f"Atom 0 has {len(atom_shells)} shells")
)",
           py::arg("atom_index"), py::return_value_policy::reference_internal)

      .def("get_shell", &BasisSet::get_shell,
           R"(
Get a specific shell by global index.

Args:
    shell_index (int): Global index of the shell

Returns:
    Shell: Reference to the specified shell

Raises:
    IndexError: If shell index is out of range

Examples:
    >>> shell = basis_set.get_shell(0)
    >>> print(f"First shell type: {shell.orbital_type}")
)",
           py::arg("shell_index"), py::return_value_policy::reference_internal)

      .def("get_num_shells", &BasisSet::get_num_shells,
           R"(
Get total number of shells across all atoms.

Returns:
    int: Total number of shells

Examples:
    >>> n_shells = basis_set.get_num_shells()
    >>> print(f"Total shells: {n_shells}")
)")

      .def("get_num_atoms", &BasisSet::get_num_atoms,
           R"(
Get number of atoms that have shells.

Returns:
    int: Number of atoms with shells

Examples:
    >>> n_atoms = basis_set.get_num_atoms()
    >>> print(f"Atoms with atomic orbitals: {n_atoms}")
)")

      // ECP shell access
      .def("get_ecp_shells", &BasisSet::get_ecp_shells,
           R"(
Get all ECP shells (flattened from per-atom storage).

Returns:
    list[Shell]: Vector of all ECP shells in the basis set

Examples:
    >>> ecp_shells = basis_set.get_ecp_shells()
    >>> print(f"Total ECP shells: {len(ecp_shells)}")
)")

      .def("get_ecp_shells_for_atom", &BasisSet::get_ecp_shells_for_atom,
           R"(
Get ECP shells for a specific atom.

Args:
    atom_index (int): Index of the atom

Returns:
    list[Shell]: Vector of ECP shells for the specified atom

Examples:
    >>> ecp_atom_shells = basis_set.get_ecp_shells_for_atom(0)
    >>> print(f"Atom 0 has {len(ecp_atom_shells)} ECP shells")
)",
           py::arg("atom_index"), py::return_value_policy::reference_internal)

      .def("get_ecp_shell", &BasisSet::get_ecp_shell,
           R"(
Get a specific ECP shell by global index.

Args:
    shell_index (int): Global index of the ECP shell

Returns:
    Shell: Reference to the specified ECP shell

Raises:
    IndexError: If ECP shell index is out of range

Examples:
    >>> ecp_shell = basis_set.get_ecp_shell(0)
    >>> print(f"First ECP shell has {ecp_shell.get_num_primitives()} primitives")
)",
           py::arg("shell_index"), py::return_value_policy::reference_internal)

      .def("get_num_ecp_shells", &BasisSet::get_num_ecp_shells,
           R"(
Get total number of ECP shells across all atoms.

Returns:
    int: Total number of ECP shells

Examples:
    >>> n_ecp_shells = basis_set.get_num_ecp_shells()
    >>> print(f"Total ECP shells: {n_ecp_shells}")
)")

      .def("has_ecp_shells", &BasisSet::has_ecp_shells,
           R"(
Check if this basis set has ECP shells.

Returns:
    bool: True if there are any ECP shells

Examples:
    >>> if basis_set.has_ecp_shells():
    ...     print("This basis set includes ECP shells")
)")

      // atomic orbital management
      .def("get_atomic_orbital_info", &BasisSet::get_atomic_orbital_info,
           R"(
Get shell index and magnetic quantum number for a atomic orbital.

Args:
    atomic_orbital_index (int): Global index of the atomic orbital

Returns:
    tuple[int, int]: Shell index and magnetic quantum number (m_l) for the atomic orbital

Examples:
    >>> shell_idx, m_l = basis_set.get_atomic_orbital_info(5)
    >>> print(f"atomic orbital 5: shell {shell_idx}, m_l = {m_l}")
)",
           py::arg("atomic_orbital_index"))
      .def("get_num_atomic_orbitals", &BasisSet::get_num_atomic_orbitals,
           R"(
Get total number of atomic orbitals in the basis set.

Returns:
    int: Total number of atomic orbitals from all shells

Examples:
    >>> n_basis = basis_set.get_num_atomic_orbitals()
    >>> print(f"Total atomic orbitals: {n_basis}")
)")

      // Atom mapping
      .def("get_atom_index_for_atomic_orbital",
           &BasisSet::get_atom_index_for_atomic_orbital,
           R"(
Get the atom index for a given atomic orbital.

Args:
    atomic_orbital_index (int): Global index of the atomic orbital

Returns:
    int: Index of the atom to which this atomic orbital belongs.
        Index of the atom to which this atomic orbital belongs

Examples:
    >>> atom_idx = basis_set.get_atom_index_for_atomic_orbital(3)
    >>> print(f"atomic orbital 3 belongs to atom {atom_idx}")
)",
           py::arg("atomic_orbital_index"))
      .def("get_atomic_orbital_indices_for_atom",
           &BasisSet::get_atomic_orbital_indices_for_atom,
           R"(
        Get all atomic orbital indices for a specific atom.

Args:
    atom_index (int): Index of the atom

Returns:
    list[int]: Vector of global atomic orbital indices for the atom

Examples:
    >>> atomic_orbital_indices = basis_set.get_atomic_orbital_indices_for_atom(0)
    >>> print(f"Atom 0 has atomic orbitals: {atomic_orbital_indices}")
)",
           py::arg("atom_index"))
      .def("get_shell_indices_for_atom", &BasisSet::get_shell_indices_for_atom,
           R"(
Get shell indices for a specific atom.

Args:
    atom_index (int): Index of the atom

Returns:
    list[int]: Vector of global shell indices for the atom

Examples:
    >>> shell_indices = basis_set.get_shell_indices_for_atom(0)
    >>> print(f"Atom 0 has shells: {shell_indices}")
)",
           py::arg("atom_index"))
      .def("get_num_atomic_orbitals_for_atom",
           &BasisSet::get_num_atomic_orbitals_for_atom,
           R"(
Get number of atomic orbitals for a specific atom.

Args:
    atom_index (int): Index of the atom

Returns:
    int: Number of atomic orbitals on the specified atom

Examples:
    >>> n_funcs = basis_set.get_num_atomic_orbitals_for_atom(0)
    >>> print(f"Atom 0 has {n_funcs} atomic orbitals")
)",
           py::arg("atom_index"))

      // Orbital type mapping
      .def("get_shell_indices_for_orbital_type",
           &BasisSet::get_shell_indices_for_orbital_type,
           R"(
Get shell indices for a specific orbital type.

Args:
    orbital_type (OrbitalType): Type of orbital (S, P, D, F, etc.)

Returns:
    list[int]: Vector of shell indices with the specified orbital type

Examples:
    >>> p_shells = basis_set.get_shell_indices_for_orbital_type(OrbitalType.P)
    >>> print(f"P-shell indices: {p_shells}")
)",
           py::arg("orbital_type"))
      .def("get_num_atomic_orbitals_for_orbital_type",
           &BasisSet::get_num_atomic_orbitals_for_orbital_type,
           R"(
Get total number of atomic orbitals for a specific orbital type.

Args:
    orbital_type (OrbitalType): Type of orbital (S, P, D, F, etc.)

Returns:
    int: Total number of atomic orbitals of the specified type

Examples:
    >>> n_p_funcs = basis_set.get_num_atomic_orbitals_for_orbital_type(OrbitalType.P)
    >>> print(f"Total P-type atomic orbitals: {n_p_funcs}")
)",
           py::arg("orbital_type"))

      // Combined queries
      .def("get_shell_indices_for_atom_and_orbital_type",
           &BasisSet::get_shell_indices_for_atom_and_orbital_type,
           R"(
Get shell indices for a specific atom and orbital type.

Args:
    atom_index (int): Index of the atom
    orbital_type (OrbitalType): Type of orbital (S, P, D, F, etc.)

Returns:
    list[int]: Vector of shell indices matching both criteria

Examples:
    >>> p_shell_indices = basis_set.get_shell_indices_for_atom_and_orbital_type(0, OrbitalType.P)
    >>> print(f"P-shells on atom 0: {p_shell_indices}")
)",
           py::arg("atom_index"), py::arg("orbital_type"))

      // ECP shell index queries
      .def("get_ecp_shell_indices_for_atom",
           &BasisSet::get_ecp_shell_indices_for_atom,
           R"(
Get ECP shell indices for a specific atom.

Args:
    atom_index (int): Index of the atom

Returns:
    list[int]: Vector of ECP shell indices for this atom

Examples:
    >>> ecp_indices = basis_set.get_ecp_shell_indices_for_atom(0)
    >>> print(f"Atom 0 ECP shell indices: {ecp_indices}")
)",
           py::arg("atom_index"))

      .def("get_ecp_shell_indices_for_orbital_type",
           &BasisSet::get_ecp_shell_indices_for_orbital_type,
           R"(
Get ECP shell indices for a specific orbital type.

Args:
    orbital_type (OrbitalType): Type of orbital (S, P, D, F, etc.)

Returns:
    list[int]: Vector of ECP shell indices of this type

Examples:
    >>> s_ecp_indices = basis_set.get_ecp_shell_indices_for_orbital_type(OrbitalType.S)
    >>> print(f"S-type ECP shell indices: {s_ecp_indices}")
)",
           py::arg("orbital_type"))

      .def("get_ecp_shell_indices_for_atom_and_orbital_type",
           &BasisSet::get_ecp_shell_indices_for_atom_and_orbital_type,
           R"(
Get ECP shell indices for a specific atom and orbital type.

Args:
    atom_index (int): Index of the atom
    orbital_type (OrbitalType): Type of orbital (S, P, D, F, etc.)

Returns:
    list[int]: Vector of ECP shell indices matching both criteria

Examples:
    >>> p_ecp_indices = basis_set.get_ecp_shell_indices_for_atom_and_orbital_type(0, OrbitalType.P)
    >>> print(f"P-type ECP shells on atom 0: {p_ecp_indices}")
)",
           py::arg("atom_index"), py::arg("orbital_type"))

      // Basis set metadata
      .def("get_name", &BasisSet::get_name,
           R"(
Get the basis set name.

Returns:
    str: Name of the basis set (e.g., "6-31G", "cc-pVDZ")

Examples:
    >>> name = basis_set.get_name()
    >>> print(f"Using basis set: {name}")
)")

      .def("get_structure", &BasisSet::get_structure,
           R"(
Get the molecular structure.

Returns:
    Structure: The molecular structure associated with this basis set

Raises:
    RuntimeError: If no structure is associated with this basis set

Examples:
    >>> structure = basis_set.get_structure()
    >>> print(f"Number of atoms: {structure.get_num_atoms()}")
)",
           py::return_value_policy::reference_internal)

      .def("has_structure", &BasisSet::has_structure,
           R"(
Check if a structure is associated with this basis set.

Returns:
    bool: True if a molecular structure is set, False otherwise

Examples:
    >>> if basis_set.has_structure():
    ...     structure = basis_set.get_structure()
    ... else:
    ...     print("No structure associated with basis set")
)")

      .def("get_ecp_name", &BasisSet::get_ecp_name,
           R"(
Get the ECP (Effective Core Potential) name.

Returns:
    str: Name of the ECP (basis set)

Examples:
    >>> ecp_name = basis_set.get_ecp_name()
    >>> print(f"ECP: {ecp_name}")
)")

      .def("get_ecp_electrons", &BasisSet::get_ecp_electrons,
           R"(
Get the ECP (Effective Core Potential) electrons vector.

Returns:
    list[int]: Number of ECP electrons for each atom

Examples:
    >>> ecp_electrons = basis_set.get_ecp_electrons()
    >>> print(f"ECP electrons per atom: {ecp_electrons}")
)")

      .def("has_ecp_electrons", &BasisSet::has_ecp_electrons,
           R"(
Check if ECP (Effective Core Potential) electrons are present.

Returns:
    bool: True if ECP electrons are present, False otherwise

Examples:
    >>> if basis_set.has_ecp_electrons():
    ...     ecp_electrons = basis_set.get_ecp_electrons()
    ...     print(f"ECP electrons per atom: {ecp_electrons}")
)")

      .def("get_summary", &BasisSet::get_summary,
           R"(
Get summary string of basis set information.

Returns:
    str: Human-readable summary of basis set properties

Examples:
    >>> summary = basis_set.get_summary()
    >>> print(summary)
)")

      // Serialization
      .def(
          "to_json",
          [](const BasisSet& self) -> std::string {
            return self.to_json().dump();
          },
          R"(
Convert basis set to JSON string.

Serializes all basis set information to a JSON string format.
JSON is human-readable and suitable for debugging or data exchange.

Returns:
    str: JSON string representation of the basis set data

Raises:
    RuntimeError: If the basis set data is invalid

Examples:
    >>> json_str = basis_set.to_json()
    >>> print(json_str)  # Pretty-printed JSON
)")
      .def_static(
          "from_json",
          [](const std::string& json_str) -> BasisSet {
            return *BasisSet::from_json(nlohmann::json::parse(json_str));
          },
          R"(
Load basis set from JSON string.

Parses basis set data from a JSON string and returns a new BasisSet instance.
The string should contain JSON data in the format produced by ``to_json()``.

Args:
    json_str (str): JSON string containing basis set data

Returns:
    BasisSet: New BasisSet instance loaded from JSON

Raises:
        RuntimeError: If the JSON string is malformed or contains invalid basis set data

Examples:
    >>> basis_set = BasisSet.from_json('{"name": "STO-3G", "shells": [...]}')
)",
          py::arg("json_str"))
      // Serialization
      .def("to_file", basis_set_to_file_wrapper,
           R"(
Save basis set to file with specified format.

Generic method to save basis set data to a file. The format is determined by
the 'type' parameter.

Args:
    filename (str | pathlib.Path): Path to the file to write.
        Must have '.basis_set' before the file extension (e.g., ``sto-3g.basis_set.json``, ``cc-pvdz.basis_set.h5``)
    type (str): File format type ("json" or "hdf5")

Raises:
    RuntimeError: If the basis set data is invalid, unsupported type, or file cannot be opened/written

Examples:
    >>> basis_set.to_file("sto-3g.basis_set.json", "json")
    >>> basis_set.to_file("cc-pvdz.basis_set.h5", "hdf5")
    >>> from pathlib import Path
    >>> basis_set.to_file(Path("sto-3g.basis_set.json"), "json")
)",
           py::arg("filename"), py::arg("type"))
      .def_static("from_file", basis_set_from_file_wrapper,
                  R"(
Load basis set from file with specified format.

Generic method to load basis set data from a file. The format is determined by
the 'type' parameter.

Args:
    filename (str | pathlib.Path): Path to the file to read.
        Must have '.basis_set' before the file extension (e.g., ``sto-3g.basis_set.json``, ``cc-pvdz.basis_set.h5``)
    type (str): File format type ("json" or "hdf5")

Returns:
    BasisSet: New BasisSet instance loaded from file

Raises:
    RuntimeError: If the file cannot be opened/read, invalid data format, or unsupported type

Examples:
    >>> basis_set = BasisSet.from_file("sto-3g.basis_set.json", "json")
    >>> basis_set = BasisSet.from_file("cc-pvdz.basis_set.h5", "hdf5")
)",
                  py::arg("filename"), py::arg("type"))
      .def("to_hdf5_file", basis_set_to_hdf5_file_wrapper,
           R"(
Save basis set to HDF5 file (with validation).

Writes all basis set data to an HDF5 file, preserving numerical precision.
HDF5 format is efficient for large datasets and supports hierarchical
data structures, making it ideal for storing basis set information.

Args:
    filename (str | pathlib.Path): Path to the HDF5 file to write.
        Must have '.basis_set' before the file extension (e.g., ``sto-3g.basis_set.h5``, ``cc-pvdz.basis_set.hdf5``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the basis set data is invalid or the file cannot be opened/written

Examples:
    >>> basis_set.to_hdf5_file("sto-3g.basis_set.h5")
    >>> basis_set.to_hdf5_file("cc-pvdz.basis_set.hdf5")
    >>> from pathlib import Path
    >>> basis_set.to_hdf5_file(Path("sto-3g.basis_set.h5"))
)",
           py::arg("filename"))
      .def_static("from_hdf5_file", basis_set_from_hdf5_file_wrapper,
                  R"(
Load basis set from HDF5 file (with validation).

Reads basis set data from an HDF5 file and returns a new BasisSet instance.
The file should contain data in the format produced by ``to_hdf5_file()``.

Args:
    filename (str | pathlib.Path): Path to the HDF5 file to read.
        Must have '.basis_set' before the file extension (e.g., ``sto-3g.basis_set.h5``, ``cc-pvdz.basis_set.hdf5``)

Returns:
    BasisSet: New ``BasisSet`` instance loaded from file

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened, read, or contains invalid basis set data

Examples:
    >>> basis_set = BasisSet.from_hdf5_file("sto-3g.basis_set.h5")
    >>> basis_set = BasisSet.from_hdf5_file("cc-pvdz.basis_set.hdf5")
)",
                  py::arg("filename"))
      .def("to_json_file", basis_set_to_json_file_wrapper,
           R"(
Save basis set to JSON file (with validation).

Writes all basis set data to a JSON file with pretty formatting.
The file will be created or overwritten if it already exists.

Args:
    filename (str): Path to the JSON file to write.
        Must have '.basis_set' before the file extension (e.g., ``sto-3g.basis_set.json``, ``cc-pvdz.basis_set.json``)

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the basis set data is invalid or the file cannot be opened/written

Examples:
    >>> basis_set.to_json_file("sto-3g.basis_set.json")
    >>> basis_set.to_json_file("my_basis.basis_set.json")
)",
           py::arg("filename"))

      .def_static("from_json_file", basis_set_from_json_file_wrapper,
                  R"(
Load basis set from JSON file (with validation).

Reads basis set data from a JSON file and returns a new BasisSet instance.
The file should contain JSON data in the format produced by ``to_json_file()``.

Args:
    filename (str): Path to the JSON file to read.
        Must have '.basis_set' before the file extension (e.g., ``sto-3g.basis_set.json``, ``cc-pvdz.basis_set.json``)

Returns:
    BasisSet: New ``BasisSet`` instance loaded from file

Raises:
    ValueError: If filename doesn't follow the required naming convention
    RuntimeError: If the file cannot be opened, read, or contains invalid basis set data

Examples:
    >>> basis_set = BasisSet.from_json_file("sto-3g.basis_set.json")
    >>> basis_set = BasisSet.from_json_file("my_basis.basis_set.json")
)",
                  py::arg("filename"))

      // Utility functions (static methods)
      .def_static("orbital_type_to_string", &BasisSet::orbital_type_to_string,
                  R"(
Convert orbital type enum to string representation.

Args:
    orbital_type (OrbitalType): The orbital type enum value

Returns:
    str: String representation (e.g., "S", "P", "D", "F")

Examples:
    >>> orbital_str = BasisSet.orbital_type_to_string(OrbitalType.P)
    >>> print(f"Orbital type: {orbital_str}")  # Prints "P"
)",
                  py::arg("orbital_type"))
      .def_static("string_to_orbital_type", &BasisSet::string_to_orbital_type,
                  R"(
Convert string to orbital type enum.

Args:
    orbital_string (str): String representation of orbital type (e.g., "S", "P", "D")

Returns:
    OrbitalType: Corresponding orbital type enum

Raises:
    ValueError: If the string does not correspond to a valid orbital type

Examples:
    >>> orbital_type = BasisSet.string_to_orbital_type("P")
    >>> print(orbital_type)  # OrbitalType.P
)",
                  py::arg("orbital_string"))
      .def_static("l_to_orbital_type", &BasisSet::l_to_orbital_type,
                  R"(
Get orbital type for angular momentum quantum number.

Args:
    l (int): Angular momentum quantum number

Returns:
    OrbitalType: Corresponding orbital type (S, P, D, etc.)

Raises:
    ValueError: If l is negative or exceeds supported range

Examples:
    >>> orbital_type = BasisSet.l_to_orbital_type(2)
    >>> print(f"l=2 corresponds to orbital type: {orbital_type}")  # D
)",
                  py::arg("l"))
      .def_static("get_angular_momentum", &BasisSet::get_angular_momentum,
                  R"(
Get angular momentum quantum number for orbital type.

Args:
    orbital_type (OrbitalType): The orbital type

Returns:
    int: Angular momentum quantum number l (0=s, 1=p, 2=d, etc.)

Examples:
    >>> l = BasisSet.get_angular_momentum(OrbitalType.D)
    >>> print(f"D orbital has l = {l}")  # l = 2
)",
                  py::arg("orbital_type"))
      .def_static("get_num_orbitals_for_l", &BasisSet::get_num_orbitals_for_l,
                  R"(
Get number of orbitals for given angular momentum.

Args:
    l (int): Angular momentum quantum number
    atomic_orbital_type (AOType, optional): Whether to use spherical (2l+1) or Cartesian functions
        Default is Spherical

Returns:
    int: Number of orbital functions

Examples:
    >>> # For d orbitals (l=2)
    >>> n_sph = BasisSet.get_num_orbitals_for_l(2, AOType.Spherical)  # 5
    >>> n_cart = BasisSet.get_num_orbitals_for_l(2, AOType.Cartesian)  # 6
    >>> print(f"d orbitals: {n_sph} spherical, {n_cart} Cartesian")
)",
                  py::arg("l"),
                  py::arg("atomic_orbital_type") = AOType::Spherical)
      .def_static("atomic_orbital_type_to_string",
                  &BasisSet::atomic_orbital_type_to_string,
                  R"(
Convert basis type enum to string representation.

Args:
    atomic_orbital_type (AOType): The basis type enum value

Returns:
    str: String representation ("Spherical" or "Cartesian")

Examples:
    >>> basis_str = BasisSet.atomic_orbital_type_to_string(AOType.Spherical)
    >>> print(f"Basis type: {basis_str}")  # Prints "Spherical"
)",
                  py::arg("atomic_orbital_type"))
      .def_static("string_to_atomic_orbital_type",
                  &BasisSet::string_to_atomic_orbital_type,
                  R"(
Convert string to basis type enum.

Args:
    basis_string (str): String representation ("Spherical" or "Cartesian")

Returns:
    AOType: Corresponding basis type enum

Raises:
    ValueError: If the string does not correspond to a valid basis type

Examples:
    >>> atomic_orbital_type = BasisSet.string_to_atomic_orbital_type("Cartesian")
    >>> print(atomic_orbital_type)  # AOType.Cartesian
)",
                  py::arg("basis_string"))

      // Index conversion utilities
      .def("basis_to_shell_index", &BasisSet::basis_to_shell_index,
           R"(
Convert atomic orbital index to shell index and local function index.

Args:
    atomic_orbital_index (int): Global atomic orbital index

Returns:
    tuple[int, int]: Shell index and local function index within that shell

Examples:
    >>> shell_idx, local_idx = basis_set.basis_to_shell_index(7)
    >>> print(f"atomic orbital 7: shell {shell_idx}, local index {local_idx}")
)",
           py::arg("atomic_orbital_index"))

      // String representation - bind summary to __repr__
      .def("__repr__", [](const BasisSet& b) { return b.get_summary(); })

      .def("__str__", [](const BasisSet& b) { return b.get_summary(); })

      // Pickling support using JSON serialization
      .def(py::pickle(
          [](const BasisSet& b) -> std::string {
            // Return JSON string for pickling
            return b.to_json().dump();
          },
          [](const std::string& json_str) -> BasisSet {
            // Reconstruct from JSON string
            return *BasisSet::from_json(nlohmann::json::parse(json_str));
          }));
}

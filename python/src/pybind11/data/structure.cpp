// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/structure.hpp>

#include "path_utils.hpp"
#include "property_binding_helpers.hpp"

namespace py = pybind11;
using namespace qdk::chemistry::data;

// Wrapper functions that accept both strings and pathlib.Path objects
namespace {

void structure_to_xyz_file_wrapper(Structure &self, const py::object &filename,
                                   const std::string &comment = "") {
  self.to_xyz_file(qdk::chemistry::python::utils::to_string_path(filename),
                   comment);
}

std::shared_ptr<Structure> structure_from_xyz_file_wrapper(
    const py::object &filename) {
  return Structure::from_xyz_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void structure_to_json_file_wrapper(Structure &self,
                                    const py::object &filename) {
  self.to_json_file(qdk::chemistry::python::utils::to_string_path(filename));
}

std::shared_ptr<Structure> structure_from_json_file_wrapper(
    const py::object &filename) {
  return Structure::from_json_file(
      qdk::chemistry::python::utils::to_string_path(filename));
}

void structure_to_file_wrapper(Structure &self, const py::object &filename,
                               const std::string &format_type) {
  self.to_file(qdk::chemistry::python::utils::to_string_path(filename),
               format_type);
}

std::shared_ptr<Structure> structure_from_file_wrapper(
    const py::object &filename, const std::string &format_type) {
  return Structure::from_file(
      qdk::chemistry::python::utils::to_string_path(filename), format_type);
}

}  // namespace

void bind_structure(py::module &m) {
  using qdk::chemistry::python::utils::bind_getter_as_property;

  // Bind Element enum
  py::enum_<Element>(m, "Element", R"(
    Chemical elements enumeration.

    This enum represents all chemical elements from hydrogen (1) to oganesson (118).
    Each element is represented by its atomic number.

    Examples
    --------
    >>> from qdk_chemistry.data import Element
    >>> Element.H  # Hydrogen
    >>> Element.C  # Carbon
    >>> Element.O  # Oxygen
    )")
      // Period 1
      .value("H", Element::H, "Hydrogen")
      .value("He", Element::He, "Helium")
      // Period 2
      .value("Li", Element::Li, "Lithium")
      .value("Be", Element::Be, "Beryllium")
      .value("B", Element::B, "Boron")
      .value("C", Element::C, "Carbon")
      .value("N", Element::N, "Nitrogen")
      .value("O", Element::O, "Oxygen")
      .value("F", Element::F, "Fluorine")
      .value("Ne", Element::Ne, "Neon")
      // Period 3
      .value("Na", Element::Na, "Sodium")
      .value("Mg", Element::Mg, "Magnesium")
      .value("Al", Element::Al, "Aluminum")
      .value("Si", Element::Si, "Silicon")
      .value("P", Element::P, "Phosphorus")
      .value("S", Element::S, "Sulfur")
      .value("Cl", Element::Cl, "Chlorine")
      .value("Ar", Element::Ar, "Argon")
      // Period 4
      .value("K", Element::K, "Potassium")
      .value("Ca", Element::Ca, "Calcium")
      .value("Sc", Element::Sc, "Scandium")
      .value("Ti", Element::Ti, "Titanium")
      .value("V", Element::V, "Vanadium")
      .value("Cr", Element::Cr, "Chromium")
      .value("Mn", Element::Mn, "Manganese")
      .value("Fe", Element::Fe, "Iron")
      .value("Co", Element::Co, "Cobalt")
      .value("Ni", Element::Ni, "Nickel")
      .value("Cu", Element::Cu, "Copper")
      .value("Zn", Element::Zn, "Zinc")
      .value("Ga", Element::Ga, "Gallium")
      .value("Ge", Element::Ge, "Germanium")
      .value("As", Element::As, "Arsenic")
      .value("Se", Element::Se, "Selenium")
      .value("Br", Element::Br, "Bromine")
      .value("Kr", Element::Kr, "Krypton")
      // Period 5
      .value("Rb", Element::Rb, "Rubidium")
      .value("Sr", Element::Sr, "Strontium")
      .value("Y", Element::Y, "Yttrium")
      .value("Zr", Element::Zr, "Zirconium")
      .value("Nb", Element::Nb, "Niobium")
      .value("Mo", Element::Mo, "Molybdenum")
      .value("Tc", Element::Tc, "Technetium")
      .value("Ru", Element::Ru, "Ruthenium")
      .value("Rh", Element::Rh, "Rhodium")
      .value("Pd", Element::Pd, "Palladium")
      .value("Ag", Element::Ag, "Silver")
      .value("Cd", Element::Cd, "Cadmium")
      .value("In", Element::In, "Indium")
      .value("Sn", Element::Sn, "Tin")
      .value("Sb", Element::Sb, "Antimony")
      .value("Te", Element::Te, "Tellurium")
      .value("I", Element::I, "Iodine")
      .value("Xe", Element::Xe, "Xenon")
      // Period 6
      .value("Cs", Element::Cs, "Cesium")
      .value("Ba", Element::Ba, "Barium")
      .value("La", Element::La, "Lanthanum")
      .value("Ce", Element::Ce, "Cerium")
      .value("Pr", Element::Pr, "Praseodymium")
      .value("Nd", Element::Nd, "Neodymium")
      .value("Pm", Element::Pm, "Promethium")
      .value("Sm", Element::Sm, "Samarium")
      .value("Eu", Element::Eu, "Europium")
      .value("Gd", Element::Gd, "Gadolinium")
      .value("Tb", Element::Tb, "Terbium")
      .value("Dy", Element::Dy, "Dysprosium")
      .value("Ho", Element::Ho, "Holmium")
      .value("Er", Element::Er, "Erbium")
      .value("Tm", Element::Tm, "Thulium")
      .value("Yb", Element::Yb, "Ytterbium")
      .value("Lu", Element::Lu, "Lutetium")
      .value("Hf", Element::Hf, "Hafnium")
      .value("Ta", Element::Ta, "Tantalum")
      .value("W", Element::W, "Tungsten")
      .value("Re", Element::Re, "Rhenium")
      .value("Os", Element::Os, "Osmium")
      .value("Ir", Element::Ir, "Iridium")
      .value("Pt", Element::Pt, "Platinum")
      .value("Au", Element::Au, "Gold")
      .value("Hg", Element::Hg, "Mercury")
      .value("Tl", Element::Tl, "Thallium")
      .value("Pb", Element::Pb, "Lead")
      .value("Bi", Element::Bi, "Bismuth")
      .value("Po", Element::Po, "Polonium")
      .value("At", Element::At, "Astatine")
      .value("Rn", Element::Rn, "Radon")
      // Period 7
      .value("Fr", Element::Fr, "Francium")
      .value("Ra", Element::Ra, "Radium")
      .value("Ac", Element::Ac, "Actinium")
      .value("Th", Element::Th, "Thorium")
      .value("Pa", Element::Pa, "Protactinium")
      .value("U", Element::U, "Uranium")
      .value("Np", Element::Np, "Neptunium")
      .value("Pu", Element::Pu, "Plutonium")
      .value("Am", Element::Am, "Americium")
      .value("Cm", Element::Cm, "Curium")
      .value("Bk", Element::Bk, "Berkelium")
      .value("Cf", Element::Cf, "Californium")
      .value("Es", Element::Es, "Einsteinium")
      .value("Fm", Element::Fm, "Fermium")
      .value("Md", Element::Md, "Mendelevium")
      .value("No", Element::No, "Nobelium")
      .value("Lr", Element::Lr, "Lawrencium")
      .value("Rf", Element::Rf, "Rutherfordium")
      .value("Db", Element::Db, "Dubnium")
      .value("Sg", Element::Sg, "Seaborgium")
      .value("Bh", Element::Bh, "Bohrium")
      .value("Hs", Element::Hs, "Hassium")
      .value("Mt", Element::Mt, "Meitnerium")
      .value("Ds", Element::Ds, "Darmstadtium")
      .value("Rg", Element::Rg, "Roentgenium")
      .value("Cn", Element::Cn, "Copernicium")
      .value("Nh", Element::Nh, "Nihonium")
      .value("Fl", Element::Fl, "Flerovium")
      .value("Mc", Element::Mc, "Moscovium")
      .value("Lv", Element::Lv, "Livermorium")
      .value("Ts", Element::Ts, "Tennessine")
      .value("Og", Element::Og, "Oganesson")
      .export_values();
  py::class_<Structure, DataClass, py::smart_holder> structure(m, "Structure",
                                                               R"(
        Represents a molecular structure with atomic coordinates, elements, masses, and nuclear charges.

    This class stores and manipulates molecular structure data including:

    * Atomic coordinates in 3D space
    * Atomic element identifiers using enum
    * Atomic masses (in atomic mass units)
    * Nuclear charges (atomic numbers) for each atom
    * Serialization to/from JSON and XYZ formats
    * Basic geometric operations and validation

    The structure can be constructed from various input formats and provides
    convenient access to atomic properties and molecular geometry.
    Standard atomic masses and nuclear charges are used by default unless otherwise specified.

    Units
    -----
    * Constructors and add_atom expect coordinates in Bohr (atomic units).
    * Getters/setters operate in Bohr (atomic units).
    * JSON serialization stores/loads coordinates in Bohr.
    * XYZ files are read/written in Angstrom (conversion handled at I/O boundary).

    Examples
    --------
    Create a water molecule from symbols and coordinates:

    >>> import numpy as np
    >>> from qdk_chemistry.data import Structure
    >>> coords = np.array([[0.0, 0.0, 0.0],
    ...                    [0.757, 0.586, 0.0],
    ...                    [-0.757, 0.586, 0.0]])
    >>> symbols = ["O", "H", "H"]
    >>> water = Structure(coords, symbols)
    >>> print(f"Number of atoms: {water.get_num_atoms()}")
    >>> print(f"Total charge: {water.get_total_nuclear_charge()}")

    Create from elements enum:

    >>> from qdk_chemistry.data import Element
    >>> coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
    >>> elements = [Element.H, Element.H]
    >>> h2 = Structure(coords, elements)

    Create from XYZ string:

    >>> xyz_str = '''3
    ... Water molecule
    ... O  0.000000  0.000000  0.000000
    ... H  0.757000  0.586000  0.000000
    ... H -0.757000  0.586000  0.000000'''
    >>> water = Structure()
    >>> water.from_xyz(xyz_str)

    Save and load from files:

    >>> water.to_xyz_file("water.xyz", "Water molecule")
    >>> water.to_json_file("water.json")
    >>> water_copy = Structure()
    >>> water_copy.from_xyz_file("water.xyz")
        )");

  // Constructors
  structure.def(py::init<const Eigen::MatrixXd &, const std::vector<Element> &,
                         const Eigen::VectorXd &, const Eigen::VectorXd &>(),
                R"(
        Create structure from coordinates, elements, masses, and nuclear charges.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Matrix of atomic coordinates (N x 3) in Angstrom
        elements : list of Element
            Vector of atomic elements using Element enum
        masses : numpy.ndarray, optional
            Vector of atomic masses in AMU (default: use standard masses)
        nuclear_charges : numpy.ndarray, optional
            Vector of nuclear charges (default: use standard charges)

        Examples
        --------
        >>> import numpy as np
        >>> from qdk_chemistry.data import Structure, Element
        >>> coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        >>> elements = [Element.H, Element.H]
        >>> masses = np.array([1.008, 1.008])
        >>> charges = np.array([1.0, 1.0])
        >>> h2 = Structure(coords, elements, masses, charges)

        Or:

        >>> coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        >>> elements = [Element.H, Element.H]
        >>> h2 = Structure(coords, elements)

        Or:

        >>> coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
        >>> symbols = ["O", "H", "H"]
        >>> masses = np.array([15.999, 1.008, 1.008])
        >>> charges = np.array([8.0, 1.0, 1.0])
        >>> water = Structure(coords, symbols, masses, charges)

        Or:

        >>> coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
        >>> symbols = ["O", "H", "H"]
        >>> water = Structure(coords, symbols)

        Or:

        >>> coords = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])
        >>> charges = [1, 1]  # Hydrogen atoms
        >>> h2 = Structure(coords, charges)

        Or:

        >>> symbols = ["O", "H", "H"]
        >>> coords = np.array([[0.0, 0.0, 0.0], [0.757, 0.586, 0.0], [-0.757, 0.586, 0.0]])
        >>> water = Structure(symbols, coords)
        )",
                py::arg("coordinates"), py::arg("elements"),
                py::arg("masses") = Eigen::VectorXd(),
                py::arg("nuclear_charges") = Eigen::VectorXd());

  structure.def(
      py::init<const Eigen::MatrixXd &, const std::vector<Element> &>(),
      py::arg("coordinates"), py::arg("elements"));

  structure.def(
      py::init<const Eigen::MatrixXd &, const std::vector<std::string> &,
               const Eigen::VectorXd &, const Eigen::VectorXd &>(),
      py::arg("coordinates"), py::arg("symbols"),
      py::arg("masses") = Eigen::VectorXd(),
      py::arg("nuclear_charges") = Eigen::VectorXd());

  structure.def(
      py::init<const Eigen::MatrixXd &, const std::vector<std::string> &>(),
      py::arg("coordinates"), py::arg("symbols"));

  // Backward compatibility constructors for tests
  structure.def(py::init([](const Eigen::MatrixXd &coordinates,
                            const std::vector<int> &nuclear_charges) {
                  // Convert int vector to double vector for nuclear charges
                  Eigen::VectorXd charges(nuclear_charges.size());
                  for (size_t i = 0; i < nuclear_charges.size(); ++i) {
                    charges[i] = static_cast<double>(nuclear_charges[i]);
                  }

                  // Convert nuclear charges to elements
                  std::vector<Element> elements;
                  elements.reserve(nuclear_charges.size());
                  for (int charge : nuclear_charges) {
                    elements.push_back(static_cast<Element>(charge));
                  }

                  return Structure(coordinates, elements, Eigen::VectorXd(),
                                   charges);
                }),
                py::arg("coordinates"), py::arg("nuclear_charges"));

  structure.def(py::init([](const std::vector<std::string> &symbols,
                            const Eigen::MatrixXd &coordinates) {
                  return Structure(coordinates, symbols);
                }),
                py::arg("symbols"), py::arg("coordinates"));

  // Data access methods
  bind_getter_as_property(structure, "get_coordinates",
                          &Structure::get_coordinates,
                          R"(
        Get the atomic coordinates matrix.

        Returns
        -------
        numpy.ndarray
            Matrix of coordinates (N x 3) in Bohr

        Examples
        --------
        >>> coords = structure.get_coordinates()
        >>> print(f"Shape: {coords.shape}")  # (N, 3)
        )");

  bind_getter_as_property(structure, "get_elements", &Structure::get_elements,
                          R"(
        Get the atomic elements vector.

        Returns
        -------
        list of :class:`Element`
            Vector of atomic elements

        Examples
        --------
        >>> elements = structure.get_elements()
        >>> print(f"First element: {elements[0]}")
        )",
                          py::return_value_policy::reference_internal);

  bind_getter_as_property(structure, "get_masses", &Structure::get_masses,
                          R"(
        Get the atomic masses vector.

        Returns
        -------
        numpy.ndarray
            Vector of atomic masses in AMU

        Examples
        --------
        >>> masses = structure.get_masses()
        >>> print(f"Total mass: {masses.sum()} AMU")
        )",
                          py::return_value_policy::reference_internal);

  bind_getter_as_property(structure, "get_nuclear_charges",
                          &Structure::get_nuclear_charges,
                          R"(
        Get the nuclear charges vector.

        Returns
        -------
        numpy.ndarray
            Vector of nuclear charges (atomic numbers)

        Examples
        --------
        >>> charges = structure.get_nuclear_charges()
        >>> print(f"Total charge: {charges.sum()}")
        )",
                          py::return_value_policy::reference_internal);

  structure.def("get_atom_coordinates", &Structure::get_atom_coordinates,
                R"(
        Get coordinates for a specific atom.

        Parameters
        ----------
        atom_index : int
            Index of the atom (0-based)

        Returns
        -------
        numpy.ndarray
            3D coordinates in Bohr as numpy array

        Examples
        --------
        >>> coords = structure.get_atom_coordinates(0)
        >>> print(f"Atom 0 position: {coords}")
        )",
                py::arg("atom_index"));

  structure.def("get_atom_element", &Structure::get_atom_element,
                R"(
        Get element for a specific atom.

        Parameters
        ----------
        atom_index : int
            Index of the atom (0-based)

        Returns
        -------
        Element
            Atomic element enum

        Examples
        --------
        >>> element = structure.get_atom_element(0)
        >>> print(f"Atom 0 element: {element}")
        )",
                py::arg("atom_index"));

  structure.def("get_atom_mass", &Structure::get_atom_mass,
                R"(
        Get mass for a specific atom.

        Parameters
        ----------
        atom_index : int
            Index of the atom (0-based)

        Returns
        -------
        float
            Atomic mass in AMU

        Examples
        --------
        >>> mass = structure.get_atom_mass(0)
        >>> print(f"Atom 0 mass: {mass} AMU")
        )",
                py::arg("atom_index"));

  structure.def("get_atom_nuclear_charge", &Structure::get_atom_nuclear_charge,
                R"(
        Get nuclear charge for a specific atom.

        Parameters
        ----------
        atom_index : int
            Index of the atom (0-based)

        Returns
        -------
        float
            Nuclear charge (atomic number)

        Examples
        --------
        >>> charge = structure.get_atom_nuclear_charge(0)
        >>> print(f"Atom 0 charge: {charge}")
        )",
                py::arg("atom_index"));

  structure.def("get_atom_symbol", &Structure::get_atom_symbol,
                R"(
        Get atomic symbol for a specific atom.

        Parameters
        ----------
        atom_index : int
            Index of the atom (0-based)

        Returns
        -------
        str
            Atomic symbol (e.g., "H", "C", "O")

        Examples
        --------
        >>> symbol = structure.get_atom_symbol(0)
        >>> print(f"Atom 0 symbol: {symbol}")
        )",
                py::arg("atom_index"));

  bind_getter_as_property(structure, "get_atomic_symbols",
                          &Structure::get_atomic_symbols,
                          R"(
        Get all atomic symbols.

        Returns
        -------
        list of str
            Vector of atomic symbols

        Examples
        --------
        >>> symbols = structure.get_atomic_symbols()
        >>> print(f"Molecule formula: {''.join(symbols)}")
        )");

  // Structure properties
  bind_getter_as_property(structure, "get_num_atoms", &Structure::get_num_atoms,
                          R"(
        Get the number of atoms in the structure.

        Returns
        -------
        int
            Number of atoms

        Examples
        --------
        >>> num_atoms = structure.get_num_atoms()
        >>> print(f"Structure has {num_atoms} atoms")
        )");

  structure.def("is_empty", &Structure::is_empty,
                R"(
        Check if the structure is empty.

        Returns
        -------
        bool
            True if structure has no atoms

        Examples
        --------
        >>> if structure.is_empty():
        ...     print("Structure is empty")
        )");

  bind_getter_as_property(structure, "get_total_mass",
                          &Structure::get_total_mass,
                          R"(
        Get the total mass of all atoms.

        Returns
        -------
        float
            Total mass in AMU

        Examples
        --------
        >>> total_mass = structure.get_total_mass()
        >>> print(f"Total mass: {total_mass} AMU")
        )");

  structure.def(
      "get_total_nuclear_charge",
      [](const Structure &s) {
        double total = 0.0;
        for (size_t i = 0; i < s.get_num_atoms(); ++i) {
          total += s.get_atom_nuclear_charge(i);
        }
        return total;
      },
      R"(
        Get the total nuclear charge of all atoms.

        Returns
        -------
        float
            Sum of all nuclear charges

        Examples
        --------
        >>> total_charge = structure.get_total_nuclear_charge()
        >>> print(f"Total nuclear charge: {total_charge}")
        )");

  // Serialization
  structure.def(
      "to_json",
      [](const Structure &self) -> std::string {
        return self.to_json().dump();
      },
      R"(
        Convert structure to JSON format (coordinates in Bohr).

        Returns
        -------
        str
            JSON string representation

        Examples
        --------
        >>> json_str = structure.to_json()
        >>> print(json_str)
        )");

  structure.def_static(
      "from_json",
      [](const std::string &json_str) {
        return *Structure::from_json(nlohmann::json::parse(json_str));
      },
      R"(
        Load structure from JSON format string (static method).

        Parameters
        ----------
        json_data : str
            JSON string representation with coordinates in Bohr

        Returns
        -------
        Structure
            New Structure object created from JSON data

        Raises
        ------
        RuntimeError
            If JSON format is invalid or contains invalid structure data

        Examples
        --------
        >>> json_str = '{"num_atoms": 2, "symbols": ["H", "H"], ...}'
        >>> h2 = Structure.from_json(json_str)
        )",
      py::arg("json_data"));

  structure.def("to_json_file", structure_to_json_file_wrapper,
                R"(
        Save structure to JSON file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to output file. Must have '.structure' before the file
            extension (e.g., "water.structure.json", "molecule.structure.json")

        Raises
        ------
        ValueError
            If filename doesn't follow the required naming convention
        RuntimeError
            If the file cannot be opened or written

        Examples
        --------
        >>> structure.to_json_file("water.structure.json")
        >>> structure.to_json_file("molecule.structure.json")
        >>> from pathlib import Path
        >>> structure.to_json_file(Path("water.structure.json"))
        )",
                py::arg("filename"));

  structure.def_static("from_json_file", structure_from_json_file_wrapper,
                       R"(
        Load structure from JSON file (static method).

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to input file. Must have '.structure' before the file
            extension (e.g., "water.structure.json", "molecule.structure.json")

        Returns
        -------
        Structure
            New Structure object loaded from the file

        Raises
        ------
        ValueError
            If filename doesn't follow the required naming convention
        RuntimeError
            If the file cannot be opened, read, or contains invalid structure data

        Examples
        --------
        >>> water = Structure.from_json_file("water.structure.json")
        >>> molecule = Structure.from_json_file("molecule.structure.json")
        >>> from pathlib import Path
        >>> water = Structure.from_json_file(Path("water.structure.json"))
        )",
                       py::arg("filename"));

  structure.def("to_xyz", &Structure::to_xyz,
                R"(
        Convert structure to XYZ format string (coordinates in Angstrom).

        Parameters
        ----------
        comment : str, optional
            Comment line for XYZ format

        Returns
        -------
        str
            XYZ format string

        Examples
        --------
        >>> xyz_str = structure.to_xyz("Water molecule")
        >>> print(xyz_str)
        )",
                py::arg("comment") = "");

  structure.def_static("from_xyz", &Structure::from_xyz,
                       R"(
        Load structure from XYZ format string (static method).

        Parameters
        ----------
        xyz_string : str
            XYZ format string with coordinates in Angstrom

        Returns
        -------
        Structure
            New Structure object created from XYZ string

        Raises
        ------
        RuntimeError
            If XYZ format is invalid

        Examples
        --------
        >>> xyz_str = '''3
        ... Water molecule
        ... O  0.000000  0.000000  0.000000
        ... H  0.757000  0.586000  0.000000
        ... H -0.757000  0.586000  0.000000'''
        >>> water = Structure.from_xyz(xyz_str)
        )",
                       py::arg("xyz_string"));

  structure.def("to_xyz_file", structure_to_xyz_file_wrapper,
                R"(
        Save structure to XYZ file.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to output file. Must have '.structure.xyz' extension
            (e.g., "water.structure.xyz", "molecule.structure.xyz")
        comment : str, optional
            Comment line for XYZ format

        Raises
        ------
        ValueError
            If filename doesn't follow the required naming convention
        RuntimeError
            If the file cannot be opened or written

        Examples
        --------
        >>> structure.to_xyz_file("water.structure.xyz", "Water molecule")
        >>> from pathlib import Path
        >>> structure.to_xyz_file(Path("water.structure.xyz"), "Water molecule")
        )",
                py::arg("filename"), py::arg("comment") = "");

  structure.def_static("from_xyz_file", structure_from_xyz_file_wrapper,
                       R"(
        Load structure from XYZ file (static method).

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to input file. Must have '.structure.xyz' extension
            (e.g., "water.structure.xyz", "molecule.structure.xyz")

        Returns
        -------
        Structure
            New Structure object loaded from the file

        Raises
        ------
        ValueError
            If filename doesn't follow the required naming convention
        RuntimeError
            If the file cannot be opened, read, or contains invalid structure data

        Examples
        --------
        >>> water = Structure.from_xyz_file("water.structure.xyz")
        >>> from pathlib import Path
        >>> water = Structure.from_xyz_file(Path("water.structure.xyz"))
        )",
                       py::arg("filename"));

  structure.def("to_file", structure_to_file_wrapper,
                R"(
        Save structure to file in specified format.

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to output file.
        format_type : str
            Format type ("json" or "xyz")

        Raises
        ------
        RuntimeError
            If the file cannot be opened or written

        Examples
        --------
        >>> structure.to_file("water.structure.json", "json")
        >>> structure.to_file("water.structure.xyz", "xyz")
        >>> from pathlib import Path
        >>> structure.to_file(Path("water.structure.json"), "json")
        )",
                py::arg("filename"), py::arg("format_type"));

  structure.def_static("from_file", structure_from_file_wrapper,
                       R"(
        Load structure from file in specified format (static method).

        Parameters
        ----------
        filename : str or pathlib.Path
            Path to input file.
        format_type : str
            Format type ("json" or "xyz")

        Returns
        -------
        Structure
            New Structure object loaded from the file

        Raises
        ------
        ValueError
            If format_type is not supported or filename doesn't follow naming convention
        RuntimeError
            If the file cannot be opened, read, or contains invalid structure data

        Examples
        --------
        >>> water = Structure.from_file("water.structure.json", "json")
        >>> h2 = Structure.from_file("h2.structure.xyz", "xyz")
        >>> from pathlib import Path
        >>> water = Structure.from_file(Path("water.structure.json"), "json")
        )",
                       py::arg("filename"), py::arg("format_type"));

  // Utility functions
  bind_getter_as_property(structure, "get_summary", &Structure::get_summary,
                          R"(
        Get summary string of structure information.

        Returns
        -------
        str
            Summary information about the structure

        Examples
        --------
        >>> summary = structure.get_summary()
        >>> print(summary)
        )");

  structure.def("calculate_nuclear_repulsion_energy",
                &Structure::calculate_nuclear_repulsion_energy,
                R"(
        Calculate nuclear-nuclear repulsion energy.

        Returns
        -------
        float
            Nuclear repulsion energy in atomic units (Hartree)

        Notes
        -----
        This function calculates the Coulombic repulsion energy between all nuclei
        in the structure using the formula:
        :math:`E_{nn} = \sum_{i<j} Z_i \cdot Z_j / |R_i - R_j|`
        where Z_i is the nuclear charge of atom i and R_i is its position vector.

        Examples
        --------
        >>> energy = structure.calculate_nuclear_repulsion_energy()
        >>> print(f'Nuclear repulsion energy: {energy:.6f} hartree')
        )");

  // Static methods
  structure.def_static("symbol_to_element", &Structure::symbol_to_element,
                       R"(
        Convert atomic symbol to element enum.

        Parameters
        ----------
        symbol : str
            Atomic symbol (e.g., "H", "C", "O")

        Returns
        -------
        Element
            Element enum value

        Examples
        --------
        >>> element = Structure.symbol_to_element("C")
        >>> assert element == Element.C
      )",
                       py::arg("symbol"));

  structure.def_static("element_to_symbol", &Structure::element_to_symbol,
                       R"(
        Convert element enum to atomic symbol.

        Parameters
        ----------
        element : Element
            Element enum value

        Returns
        -------
        str
            Atomic symbol

        Examples
        --------
        >>> symbol = Structure.element_to_symbol(Element.C)
        >>> assert symbol == "C"
      )",
                       py::arg("element"));

  structure.def_static("symbol_to_nuclear_charge",
                       &Structure::symbol_to_nuclear_charge,
                       R"(
        Convert atomic symbol to nuclear charge.

        Parameters
        ----------
        symbol : str
            Atomic symbol (e.g., "H", "C", "O")

        Returns
        -------
        int
            Nuclear charge (atomic number)

        Examples
        --------
        >>> charge = Structure.symbol_to_nuclear_charge("C")
        >>> assert charge == 6
      )",
                       py::arg("symbol"));

  structure.def_static("nuclear_charge_to_symbol",
                       &Structure::nuclear_charge_to_symbol,
                       R"(
        Convert nuclear charge to atomic symbol.

        Parameters
        ----------
        nuclear_charge : int
            Nuclear charge (atomic number)

        Returns
        -------
        str
            Atomic symbol

        Examples
        --------
        >>> symbol = Structure.nuclear_charge_to_symbol(6)
        >>> assert symbol == "C"
      )",
                       py::arg("nuclear_charge"));

  structure.def_static("element_to_nuclear_charge",
                       &Structure::element_to_nuclear_charge,
                       R"(
        Convert element enum to nuclear charge.

        Parameters
        ----------
        element : Element
            Element enum value

        Returns
        -------
        int
            Nuclear charge (atomic number)

        Examples
        --------
        >>> charge = Structure.element_to_nuclear_charge(Element.C)
        >>> assert charge == 6
      )",
                       py::arg("element"));

  structure.def_static("nuclear_charge_to_element",
                       &Structure::nuclear_charge_to_element,
                       R"(
        Convert nuclear charge to element enum.

        Parameters
        ----------
        nuclear_charge : int
            Nuclear charge (atomic number)

        Returns
        -------
        Element
            Element enum value

        Examples
        --------
        >>> element = Structure.nuclear_charge_to_element(6)
        >>> assert element == Element.C
      )",
                       py::arg("nuclear_charge"));

  structure.def_static("get_standard_atomic_mass",
                       &Structure::get_standard_atomic_mass,
                       R"(
        Get standard atomic mass for an element.

        Parameters
        ----------
        element : Element
            Element enum value

        Returns
        -------
        float
            Standard atomic mass in AMU

        Examples
        --------
        >>> mass = Structure.get_standard_atomic_mass(Element.C)
        >>> print(f"Carbon mass: {mass} AMU")
      )",
                       py::arg("element"));

  structure.def_static("get_standard_nuclear_charge",
                       &Structure::get_standard_nuclear_charge,
                       R"(
        Get standard nuclear charge for an element.

        Parameters
        ----------
        element : Element
            Element enum value

        Returns
        -------
        int
            Nuclear charge (atomic number)

        Examples
        --------
        >>> charge = Structure.get_standard_nuclear_charge(Element.C)
        >>> assert charge == 6
      )",
                       py::arg("element"));
  // String representation - bind summary to __repr__
  structure.def("__repr__", [](const Structure &s) {
    if (s.is_empty()) {
      return std::string("Empty Structure");
    } else {
      return s.get_summary();
    }
  });

  structure.def("__str__", [](const Structure &s) {
    if (s.is_empty()) {
      return std::string("Empty Structure");
    } else {
      return s.get_summary();
    }
  });

  // Length and indexing support
  structure.def("__len__", &Structure::get_num_atoms);

  structure.def(
      "__getitem__",
      [](const Structure &s, size_t index) {
        if (index >= s.get_num_atoms()) {
          throw py::index_error("Atom index out of range");
        }
        py::dict atom;
        atom["symbol"] = s.get_atom_symbol(index);
        atom["element"] = s.get_atom_element(index);
        atom["nuclear_charge"] = s.get_atom_nuclear_charge(index);
        atom["mass"] = s.get_atom_mass(index);
        atom["coordinates"] = s.get_atom_coordinates(index);
        return atom;
      },
      R"(
        Get atom information by index.

        Parameters
        ----------
        index : int
            Atom index (0-based)

        Returns
        -------
        dict
            Dictionary containing atom information with keys:
            'symbol', 'element', 'nuclear_charge', 'mass', 'coordinates'

        Examples
        --------
        >>> atom = structure[0]
        >>> print(f"Atom 0: {atom['symbol']} at {atom['coordinates']}")
      )");

  // Iteration support
  structure.def(
      "__iter__",
      [](const Structure &s) {
        py::list atoms;
        for (size_t i = 0; i < s.get_num_atoms(); ++i) {
          py::dict atom;
          atom["symbol"] = s.get_atom_symbol(i);
          atom["element"] = s.get_atom_element(i);
          atom["nuclear_charge"] = s.get_atom_nuclear_charge(i);
          atom["mass"] = s.get_atom_mass(i);
          atom["coordinates"] = s.get_atom_coordinates(i);
          atom["index"] = i;
          atoms.append(atom);
        }
        return py::iter(atoms);
      },
      py::keep_alive<0, 1>(),
      R"(
        Iterate over all atoms in the structure.

        Yields
        ------
        dict
            Dictionary containing atom information with keys:
            'symbol', 'element', 'nuclear_charge', 'mass', 'coordinates', 'index'

        Examples
        --------
        >>> for atom in structure:
        ...     print(f"Atom {atom['index']}: {atom['symbol']}")
      )");

  // Pickling support using JSON serialization
  structure.def(py::pickle(
      [](const Structure &s) -> std::string {
        // Return JSON string for pickling
        return s.to_json().dump();
      },
      [](const std::string &json_str) -> Structure {
        // Reconstruct from JSON string
        return *Structure::from_json(nlohmann::json::parse(json_str));
      }));
}

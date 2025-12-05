// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/pybind11.h>

#include <qdk/chemistry/data/element_data.hpp>

namespace py = pybind11;
using namespace qdk::chemistry::data;

void bind_element_data(py::module &m) {
  // Bind Element enum
  py::enum_<Element>(m, "Element",
                     R"(
Chemical elements enumeration.

This enum represents all chemical elements from hydrogen (1) to oganesson (118).
Each element is represented by its atomic number.

Examples:
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

  // Expose current CIAAW version
  m.def("get_current_ciaaw_version", &get_current_ciaaw_version,
        R"(
Get the current CIAAW version being used for atomic weights.

Returns:
    str: The CIAAW version string (e.g., 'CIAAW 2024').

Examples:
    >>> from qdk_chemistry.data import get_current_ciaaw_version
    >>> version = get_current_ciaaw_version()
    >>> print(f"Using {version} atomic weights")
)");
}

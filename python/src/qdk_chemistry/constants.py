"""QDK/Chemistry Physical Constants Module.

This module provides access to physical constants from CODATA standards.
The constants are sourced from the most recent CODATA recommendations
(currently CODATA 2018), but the underlying C++ implementation supports
multiple CODATA versions for compatibility and comparison purposes.

All constants are provided in their original units as specified by CODATA,
with conversion factors available for different unit systems.

The constants include fundamental physical constants, particle masses,
and energy conversion factors commonly used in computational chemistry
and quantum mechanics.

Data Source:
    CODATA recommended values of the fundamental physical constants
    Currently using: CODATA 2018 (default)
    Also available: CODATA 2014 (via C++ preprocessor directives)
    https://physics.nist.gov/cuu/Constants/

Constants Documentation:
    Each constant includes detailed documentation about its physical meaning,
    units, mathematical symbol, and provenance. The documentation automatically
    reflects the CODATA version currently in use. You can access this information
    using the ``get_constant_info()`` function or by examining the ``__doc__`` attribute
    of individual constants.

Examples:
    >>> from qdk_chemistry.constants import ANGSTROM_TO_BOHR, HARTREE_TO_EV
    >>> length_angstrom = 1.5
    >>> length_bohr = length_angstrom * ANGSTROM_TO_BOHR

    >>> # Access detailed documentation (reflects current CODATA version)
    >>> from qdk_chemistry.constants import get_constant_info
    >>> info = get_constant_info('bohr_to_angstrom')
    >>> print(f"{info.description} ({info.symbol}): {info.value} {info.units}")
    >>> print(f"Source: {info.source}")

    >>> # Or use the docstring
    >>> help(angstrom_to_bohr)

    >>> # List all available constants with their documentation
    >>> from qdk_chemistry.constants import list_constants
    >>> list_constants()

"""

# --------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License. See LICENSE.txt in the project root for license information.
# --------------------------------------------------------------------------------------------

from qdk_chemistry._core.constants import (
    ANGSTROM_TO_BOHR,
    ATOMIC_MASS_CONSTANT,
    AVOGADRO_CONSTANT,
    # Conversion factors
    BOHR_TO_ANGSTROM,
    BOLTZMANN_CONSTANT,
    ELECTRON_MASS,
    ELEMENTARY_CHARGE,
    EV_TO_HARTREE,
    # Fundamental constants
    FINE_STRUCTURE_CONSTANT,
    # Energy conversion factors
    HARTREE_TO_EV,
    HARTREE_TO_KCAL_PER_MOL,
    HARTREE_TO_KJ_PER_MOL,
    KCAL_PER_MOL_TO_HARTREE,
    KJ_PER_MOL_TO_HARTREE,
    NEUTRON_MASS,
    PLANCK_CONSTANT,
    PROTON_MASS,
    REDUCED_PLANCK_CONSTANT,
    SPEED_OF_LIGHT,
    ConstantInfo,
    get_constant_info,
    get_constants_info,
)


def list_constants(show_values: bool = True, show_units: bool = True) -> None:
    """List all available constants with their documentation.

    Args:
        show_values: Whether to display the numerical values, default True
        show_units:  Whether to display the units, default True

    Examples:
        >>> list_constants()
        >>> list_constants(show_values=False)  # Just names and descriptions

    """
    constants_info = get_constants_info()

    # TODO (NAB):  change output to logger rather than print() here and elsewhere
    # or just return strings workitem 41417
    print("QDK/Chemistry Physical Constants (CODATA 2018)")
    print("=" * 50)

    # Group constants by category
    categories = {
        "Length Conversion": ["bohr_to_angstrom", "angstrom_to_bohr"],
        "Fundamental Constants": [
            "fine_structure_constant",
            "speed_of_light",
            "elementary_charge",
            "planck_constant",
            "reduced_planck_constant",
        ],
        "Particle Masses": ["electron_mass", "proton_mass", "neutron_mass", "atomic_mass_constant"],
        "Thermodynamic Constants": ["boltzmann_constant", "avogadro_constant"],
        "Energy Conversion": [
            "hartree_to_ev",
            "ev_to_hartree",
            "hartree_to_kcal_per_mol",
            "kcal_per_mol_to_hartree",
            "hartree_to_kj_per_mol",
            "kj_per_mol_to_hartree",
        ],
    }

    for category, constant_names in categories.items():
        print(f"\n{category}:")
        print("-" * len(category))

        for name in constant_names:
            if name in constants_info:
                info = constants_info[name]
                line = f"  {name}"
                if info.symbol:
                    line += f" ({info.symbol})"
                line += f": {info.description}"

                if show_values:
                    line += f" = {info.value}"
                if show_units and info.units:
                    line += f" {info.units}"

                print(line)


def find_constant(search_term: str) -> dict:
    """Find constants matching a search term.

    Args:
        search_term: Term to search for in constant names or descriptions

    Returns:
        Dictionary of matching constants and their info

    Examples:
        >>> find_constant("bohr")
        >>> find_constant("mass")
        >>> find_constant("energy")

    """
    constants_info = get_constants_info()
    matches = {}

    search_lower = search_term.lower()

    for name, info in constants_info.items():
        if (
            search_lower in name.lower()
            or search_lower in info.description.lower()
            or search_lower in info.symbol.lower()
        ):
            matches[name] = info

    return matches


def show_constant_details(name: str) -> None:
    """Print detailed information about a specific constant.

    Args:
        name: Name of the constant

    Examples:
        >>> show_constant_details('bohr_to_angstrom')
        >>> show_constant_details('fine_structure_constant')

    """
    try:
        info = get_constant_info(name)
        print(f"Constant: {info.name}")
        if info.symbol:
            print(f"Symbol: {info.symbol}")
        print(f"Value: {info.value} {info.units}")
        print(f"Description: {info.description}")
        print(f"Source: {info.source}")
    except KeyError:
        print(f"Unknown constant: {name}")
        print("Use list_constants() to see all available constants.")


# Make the new functions available
__all__ = [
    "ANGSTROM_TO_BOHR",
    "ATOMIC_MASS_CONSTANT",
    "AVOGADRO_CONSTANT",
    "BOHR_TO_ANGSTROM",
    "BOLTZMANN_CONSTANT",
    "ELECTRON_MASS",
    "ELEMENTARY_CHARGE",
    "EV_TO_HARTREE",
    "FINE_STRUCTURE_CONSTANT",
    "HARTREE_TO_EV",
    "HARTREE_TO_KCAL_PER_MOL",
    "HARTREE_TO_KJ_PER_MOL",
    "KCAL_PER_MOL_TO_HARTREE",
    "KJ_PER_MOL_TO_HARTREE",
    "NEUTRON_MASS",
    "PLANCK_CONSTANT",
    "PROTON_MASS",
    "REDUCED_PLANCK_CONSTANT",
    "SPEED_OF_LIGHT",
    "ConstantInfo",
    "find_constant",
    "get_constant_info",
    "get_constants_info",
    "list_constants",
    "show_constant_details",
]

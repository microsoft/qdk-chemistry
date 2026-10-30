// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "qdk/chemistry/constants.hpp"

#include <nanobind/nanobind.h>

#include <algorithm>
#include <cctype>

#include "stl.hpp"

namespace nb = nanobind;

void bind_constants(nb::module_& m) {
  // Create a constants submodule
  auto constants = m.def_submodule("constants");
  constants.doc() = R"(
Physical constants module.

This module provides access to physical constants from CODATA standards :cite:`Mohr2025`.
The constants are sourced from the most recent CODATA recommendations (currently CODATA 2022 by default), but the underlying C++ implementation supports multiple CODATA versions for compatibility and comparison purposes.

All constants are provided in their original units as specified by CODATA, with conversion factors available for different unit systems.

The constants include fundamental physical constants, particle masses, and energy conversion factors commonly used in computational chemistry and quantum mechanics.

Data sources:
    - CODATA 2022 recommended values (default)
    - CODATA 2018 recommended values (available via C++ preprocessor)
    - CODATA 2014 recommended values (available via C++ preprocessor) https://physics.nist.gov/cuu/Constants/

    The documentation automatically reflects the CODATA version currently in use, ensuring accurate provenance information.

Examples:
    >>> from qdk_chemistry.constants import ANGSTROM_TO_BOHR, HARTREE_TO_EV
    >>> length_angstrom = 1.5
    >>> length_bohr = length_angstrom * ANGSTROM_TO_BOHR
    >>> energy_hartree = -0.5
    >>> energy_ev = energy_hartree * HARTREE_TO_EV

    >>> # Access documentation for constants (reflects current CODATA version)
    >>> from qdk_chemistry.constants import get_constant_info
    >>> info = get_constant_info('bohr_to_angstrom')
    >>> print(f"{info.description} ({info.symbol}): {info.value} {info.units}")
    >>> print(f"Source: {info.source}")
)";

  // Bind the ConstantInfo struct
  nb::class_<qdk::chemistry::constants::ConstantInfo>(
      constants, "ConstantInfo",
      "Documentation information for a physical constant")
      .def_ro("name", &qdk::chemistry::constants::ConstantInfo::name,
              "Name of the constant")
      .def_ro("description",
              &qdk::chemistry::constants::ConstantInfo::description,
              "Description of the physical meaning")
      .def_ro("units", &qdk::chemistry::constants::ConstantInfo::units,
              "Units of measurement")
      .def_ro("source", &qdk::chemistry::constants::ConstantInfo::source,
              "Data source (e.g., 'CODATA 2022')")
      .def_ro("symbol", &qdk::chemistry::constants::ConstantInfo::symbol,
              "Mathematical symbol")
      .def_ro("value", &qdk::chemistry::constants::ConstantInfo::value,
              "Numerical value")
      .def("__repr__", [](const qdk::chemistry::constants::ConstantInfo& info) {
        return "<ConstantInfo: " + info.name + " = " +
               std::to_string(info.value) + " " + info.units + ">";
      });

  // Bind the function to get constants documentation
  constants.def("get_constants_info",
                &qdk::chemistry::constants::get_constants_info,
                "Get documentation information for all available constants");

  // Helper function to get info for a single constant
  constants.def(
      "get_constant_info",
      [](const std::string& name) {
        // Convert name to lowercase for case-insensitive lookup
        std::string lower_name = name;
        std::transform(
            lower_name.begin(), lower_name.end(), lower_name.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

        auto constants_info = qdk::chemistry::constants::get_constants_info();
        auto it = constants_info.find(lower_name);
        if (it != constants_info.end()) {
          return it->second;
        } else {
          throw nb::key_error(std::string("Unknown constant: " + name).c_str());
        }
      },
      "Get documentation information for a specific constant", nb::arg("name"));

  // Expose current CODATA version
  constants.def(
      "get_current_codata_version",
      &qdk::chemistry::constants::get_current_codata_version,
      "Get the current CODATA version being used (e.g., 'CODATA 2022')");

  // CODATA constants
  // Length conversion factors
  constants.attr("BOHR_TO_ANGSTROM") =
      nb::cast(qdk::chemistry::constants::bohr_to_angstrom);

  constants.attr("ANGSTROM_TO_BOHR") =
      nb::cast(qdk::chemistry::constants::angstrom_to_bohr);

  // Fundamental constants
  constants.attr("FINE_STRUCTURE_CONSTANT") =
      nb::cast(qdk::chemistry::constants::fine_structure_constant);

  constants.attr("ELECTRON_MASS") =
      nb::cast(qdk::chemistry::constants::electron_mass);

  constants.attr("PROTON_MASS") =
      nb::cast(qdk::chemistry::constants::proton_mass);

  constants.attr("NEUTRON_MASS") =
      nb::cast(qdk::chemistry::constants::neutron_mass);

  constants.attr("ATOMIC_MASS_CONSTANT") =
      nb::cast(qdk::chemistry::constants::atomic_mass_constant);

  constants.attr("AVOGADRO_CONSTANT") =
      nb::cast(qdk::chemistry::constants::avogadro_constant);

  constants.attr("BOLTZMANN_CONSTANT") =
      nb::cast(qdk::chemistry::constants::boltzmann_constant);

  constants.attr("PLANCK_CONSTANT") =
      nb::cast(qdk::chemistry::constants::planck_constant);

  constants.attr("REDUCED_PLANCK_CONSTANT") =
      nb::cast(qdk::chemistry::constants::reduced_planck_constant);

  constants.attr("SPEED_OF_LIGHT") =
      nb::cast(qdk::chemistry::constants::speed_of_light);

  constants.attr("ELEMENTARY_CHARGE") =
      nb::cast(qdk::chemistry::constants::elementary_charge);

  constants.attr("BOHR_MAGNETON") =
      nb::cast(qdk::chemistry::constants::bohr_magneton);

  // Energy conversion factors
  constants.attr("HARTREE_TO_EV") =
      nb::cast(qdk::chemistry::constants::hartree_to_ev);

  constants.attr("EV_TO_HARTREE") =
      nb::cast(qdk::chemistry::constants::ev_to_hartree);

  constants.attr("HARTREE_TO_KCAL_PER_MOL") =
      nb::cast(qdk::chemistry::constants::hartree_to_kcal_per_mol);

  constants.attr("KCAL_PER_MOL_TO_HARTREE") =
      nb::cast(qdk::chemistry::constants::kcal_per_mol_to_hartree);

  constants.attr("HARTREE_TO_KJ_PER_MOL") =
      nb::cast(qdk::chemistry::constants::hartree_to_kj_per_mol);

  constants.attr("KJ_PER_MOL_TO_HARTREE") =
      nb::cast(qdk::chemistry::constants::kj_per_mol_to_hartree);

  // Add type annotations for all constants
  nb::dict annotations;
  annotations["BOHR_TO_ANGSTROM"] = nb::float_(0.0).type();
  annotations["ANGSTROM_TO_BOHR"] = nb::float_(0.0).type();
  annotations["FINE_STRUCTURE_CONSTANT"] = nb::float_(0.0).type();
  annotations["ELECTRON_MASS"] = nb::float_(0.0).type();
  annotations["PROTON_MASS"] = nb::float_(0.0).type();
  annotations["NEUTRON_MASS"] = nb::float_(0.0).type();
  annotations["ATOMIC_MASS_CONSTANT"] = nb::float_(0.0).type();
  annotations["AVOGADRO_CONSTANT"] = nb::float_(0.0).type();
  annotations["BOLTZMANN_CONSTANT"] = nb::float_(0.0).type();
  annotations["PLANCK_CONSTANT"] = nb::float_(0.0).type();
  annotations["REDUCED_PLANCK_CONSTANT"] = nb::float_(0.0).type();
  annotations["SPEED_OF_LIGHT"] = nb::float_(0.0).type();
  annotations["ELEMENTARY_CHARGE"] = nb::float_(0.0).type();
  annotations["BOHR_MAGNETON"] = nb::float_(0.0).type();
  annotations["HARTREE_TO_EV"] = nb::float_(0.0).type();
  annotations["EV_TO_HARTREE"] = nb::float_(0.0).type();
  annotations["HARTREE_TO_KCAL_PER_MOL"] = nb::float_(0.0).type();
  annotations["KCAL_PER_MOL_TO_HARTREE"] = nb::float_(0.0).type();
  annotations["HARTREE_TO_KJ_PER_MOL"] = nb::float_(0.0).type();
  annotations["KJ_PER_MOL_TO_HARTREE"] = nb::float_(0.0).type();

  constants.attr("__annotations__") = annotations;
}

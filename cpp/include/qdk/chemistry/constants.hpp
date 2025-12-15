// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <string>
#include <unordered_map>

// Define CODATA version constants
#define QDK_CHEMISTRY_CODATA_2022 2022
#define QDK_CHEMISTRY_CODATA_2018 2018
#define QDK_CHEMISTRY_CODATA_2014 2014

// Set default CODATA version if not specified
#ifndef QDK_CHEMISTRY_CODATA_VERSION
#define QDK_CHEMISTRY_CODATA_VERSION QDK_CHEMISTRY_CODATA_2022
#endif

/**
 * @file constants.hpp
 * @brief Physical constants from CODATA standards
 *
 * This header provides physical constants from different CODATA standards
 * organized in version-specific namespaces. Constants are provided as
 * static constexpr variables rather than preprocessor defines for type safety.
 *
 * To use a specific CODATA version, either use the fully qualified namespace
 * (e.g., qdk::chemistry::constants::codata_2022::bohr_to_angstrom) or import
 * the specific namespace.
 *
 * The default namespace (qdk::chemistry::constants) uses the most recent CODATA
 * version (currently 2022 :cite:`Mohr2025`) for convenience, but other versions
 * remain available for compatibility and comparison purposes.
 *
 * To select a specific CODATA version, define QDK_CHEMISTRY_CODATA_VERSION
 */

namespace qdk::chemistry::constants {

/**
 * @brief Documentation metadata for physical constants
 */
struct ConstantInfo {
  std::string name;
  std::string description;
  std::string units;
  std::string source;
  std::string symbol;
  double value;
};

/**
 * @namespace qdk::chemistry::constants::codata_2022
 * @brief CODATA 2022 recommended values for fundamental physical constants
 *
 * Constants from the 2022 CODATA recommended values of the fundamental physical
 * constants: https://physics.nist.gov/cuu/Constants/. :cite:`Mohr2025`
 */
namespace codata_2022 {

// Conversion factors
static constexpr double bohr_to_angstrom =
    0.529177210544;  // Bohr radius in Angstrom
static constexpr double angstrom_to_bohr = 1.0 / bohr_to_angstrom;

// Fundamental constants
static constexpr double fine_structure_constant =
    7.2973525643e-3;                                       // α, dimensionless
static constexpr double electron_mass = 9.1093837139e-31;  // kg
static constexpr double proton_mass = 1.67262192595e-27;   // kg
static constexpr double neutron_mass = 1.67492750056e-27;  // kg
static constexpr double atomic_mass_constant = 1.66053906892e-27;  // u in kg
static constexpr double avogadro_constant = 6.02214076e23;         // mol^-1
static constexpr double boltzmann_constant = 1.380649e-23;         // J/K
static constexpr double planck_constant = 6.62607015e-34;          // J⋅s
static constexpr double reduced_planck_constant =
    planck_constant / (2.0 * 3.14159265358979323846);         // ħ, J⋅s
static constexpr double speed_of_light = 299792458.0;         // m/s
static constexpr double elementary_charge = 1.602176634e-19;  // C

// Energy conversion factors
static constexpr double hartree_to_ev =
    27.211386245981;  // 1 Hartree in electron volts
static constexpr double ev_to_hartree = 1.0 / hartree_to_ev;
static constexpr double hartree_to_kj_per_mol =
    hartree_to_ev * 1.602176634 * 6.02214076 * 10;  // 1 Hartree in kJ/mol
static constexpr double kj_per_mol_to_hartree = 1.0 / hartree_to_kj_per_mol;
static constexpr double hartree_to_kcal_per_mol =
    hartree_to_kj_per_mol / 4.184;  // 1 Hartree in kcal/mol
static constexpr double kcal_per_mol_to_hartree = 1.0 / hartree_to_kcal_per_mol;

}  // namespace codata_2022

/**
 * @namespace qdk::chemistry::constants::codata_2018
 * @brief CODATA 2018 recommended values for fundamental physical constants
 *
 * Constants from the 2018 CODATA recommended values of the fundamental physical
 * constants: https://physics.nist.gov/cuu/Constants/. :cite:`Tiesinga2021`
 */
namespace codata_2018 {

// Conversion factors
static constexpr double bohr_to_angstrom =
    0.529177210903;  // Bohr radius in Angstrom
static constexpr double angstrom_to_bohr = 1.0 / bohr_to_angstrom;

// Fundamental constants
static constexpr double fine_structure_constant =
    7.2973525693e-3;                                       // α, dimensionless
static constexpr double electron_mass = 9.1093837015e-31;  // kg
static constexpr double proton_mass = 1.67262192369e-27;   // kg
static constexpr double neutron_mass = 1.67492749804e-27;  // kg
static constexpr double atomic_mass_constant = 1.66053906660e-27;  // u in kg
static constexpr double avogadro_constant = 6.02214076e23;         // mol^-1
static constexpr double boltzmann_constant = 1.380649e-23;         // J/K
static constexpr double planck_constant = 6.62607015e-34;          // J⋅s
static constexpr double reduced_planck_constant =
    planck_constant / (2.0 * 3.14159265358979323846);         // ħ, J⋅s
static constexpr double speed_of_light = 299792458.0;         // m/s
static constexpr double elementary_charge = 1.602176634e-19;  // C

// Energy conversion factors
static constexpr double hartree_to_ev =
    27.211386245988;  // 1 Hartree in electron volts
static constexpr double ev_to_hartree = 1.0 / hartree_to_ev;
static constexpr double hartree_to_kcal_per_mol =
    627.5094736;  // 1 Hartree in kcal/mol
static constexpr double kcal_per_mol_to_hartree = 1.0 / hartree_to_kcal_per_mol;
static constexpr double hartree_to_kj_per_mol =
    2625.4996395;  // 1 Hartree in kJ/mol
static constexpr double kj_per_mol_to_hartree = 1.0 / hartree_to_kj_per_mol;

}  // namespace codata_2018

/**
 * @namespace qdk::chemistry::constants::codata_2014
 * @brief CODATA 2014 recommended values for fundamental physical constants
 *
 * Constants from the 2014 CODATA recommended values of the fundamental physical
 * constants: https://physics.nist.gov/cuu/Constants/. :cite:`Mohr2016`
 */
namespace codata_2014 {

// Conversion factors
static constexpr double bohr_to_angstrom =
    0.52917721067;  // Bohr radius in Angstrom
static constexpr double angstrom_to_bohr = 1.0 / bohr_to_angstrom;

// Fundamental constants
static constexpr double fine_structure_constant =
    7.2973525664e-3;                                     // α, dimensionless
static constexpr double electron_mass = 9.10938356e-31;  // kg
static constexpr double proton_mass = 1.672621898e-27;   // kg
static constexpr double neutron_mass = 1.674927471e-27;  // kg
static constexpr double atomic_mass_constant = 1.660539040e-27;  // u in kg
static constexpr double avogadro_constant = 6.022140857e23;      // mol^-1
static constexpr double boltzmann_constant = 1.38064852e-23;     // J/K
static constexpr double planck_constant = 6.626070040e-34;       // J⋅s
static constexpr double reduced_planck_constant =
    planck_constant / (2.0 * 3.14159265358979323846);          // ħ, J⋅s
static constexpr double speed_of_light = 299792458.0;          // m/s
static constexpr double elementary_charge = 1.6021766208e-19;  // C

// Energy conversion factors
static constexpr double hartree_to_ev =
    27.21138602;  // 1 Hartree in electron volts
static constexpr double ev_to_hartree = 1.0 / hartree_to_ev;
static constexpr double hartree_to_kcal_per_mol =
    627.509474;  // 1 Hartree in kcal/mol
static constexpr double kcal_per_mol_to_hartree = 1.0 / hartree_to_kcal_per_mol;
static constexpr double hartree_to_kj_per_mol =
    2625.49964;  // 1 Hartree in kJ/mol
static constexpr double kj_per_mol_to_hartree = 1.0 / hartree_to_kj_per_mol;

}  // namespace codata_2014

// Helper to determine current CODATA version in use
constexpr const char* get_current_codata_version() {
#if QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2022
  return "CODATA 2022";
#elif QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2018
  return "CODATA 2018";
#elif QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2014
  return "CODATA 2014";
#else
  static_assert(QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2022 ||
                    QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2018 ||
                    QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2014,
                "Unsupported QDK_CHEMISTRY_CODATA_VERSION. Supported versions: "
                "QDK_CHEMISTRY_CODATA_2022, QDK_CHEMISTRY_CODATA_2018, "
                "QDK_CHEMISTRY_CODATA_2014");
#endif
}

// Use the selected CODATA version as default
// To use CODATA 2022, define:
//   #define QDK_CHEMISTRY_CODATA_VERSION QDK_CHEMISTRY_CODATA_2022 (default)
// To use CODATA 2018, define:
//   #define QDK_CHEMISTRY_CODATA_VERSION QDK_CHEMISTRY_CODATA_2018
// To use CODATA 2014, define:
//   #define QDK_CHEMISTRY_CODATA_VERSION QDK_CHEMISTRY_CODATA_2014
#if QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2022
using namespace codata_2022;
#elif QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2018
using namespace codata_2018;
#elif QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2014
using namespace codata_2014;
#else
static_assert(QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2022 ||
                  QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2018 ||
                  QDK_CHEMISTRY_CODATA_VERSION == QDK_CHEMISTRY_CODATA_2014,
              "Unsupported QDK_CHEMISTRY_CODATA_VERSION. Supported versions: "
              "QDK_CHEMISTRY_CODATA_2022, QDK_CHEMISTRY_CODATA_2018, "
              "QDK_CHEMISTRY_CODATA_2014");
#endif

/**
 * @brief Get documentation information for all constants in the current
 * namespace
 * @return Map of constant names to their documentation
 *
 * This function returns documentation for constants using the currently active
 * CODATA version (determined by the `using namespace` directive below).
 */
inline std::unordered_map<std::string, ConstantInfo> get_constants_info() {
  const char* current_version = get_current_codata_version();

  return {{"bohr_to_angstrom",
           {"bohr_to_angstrom", "Bohr radius conversion factor to Angstrom",
            "Å/bohr", current_version, "a₀", bohr_to_angstrom}},
          {"angstrom_to_bohr",
           {"angstrom_to_bohr", "Angstrom to Bohr radius conversion factor",
            "bohr/Å", current_version, "1/a₀", angstrom_to_bohr}},
          {"fine_structure_constant",
           {"fine_structure_constant",
            "Fine-structure constant, fundamental physical constant "
            "characterizing electromagnetic interactions",
            "dimensionless", current_version, "α", fine_structure_constant}},
          {"electron_mass",
           {"electron_mass", "Electron rest mass", "kg", current_version, "mₑ",
            electron_mass}},
          {"proton_mass",
           {"proton_mass", "Proton rest mass", "kg", current_version, "mₚ",
            proton_mass}},
          {"neutron_mass",
           {"neutron_mass", "Neutron rest mass", "kg", current_version, "mₙ",
            neutron_mass}},
          {"atomic_mass_constant",
           {"atomic_mass_constant",
            "Atomic mass constant (unified atomic mass unit)", "kg",
            current_version, "u", atomic_mass_constant}},
          {"avogadro_constant",
           {"avogadro_constant",
            "Avogadro constant, number of constituent particles per mole",
            "mol⁻¹", current_version, "Nₐ", avogadro_constant}},
          {"boltzmann_constant",
           {"boltzmann_constant",
            "Boltzmann constant, relates average kinetic energy to temperature",
            "J/K", current_version, "k", boltzmann_constant}},
          {"planck_constant",
           {"planck_constant", "Planck constant, fundamental quantum of action",
            "J⋅s", current_version, "h", planck_constant}},
          {"reduced_planck_constant",
           {"reduced_planck_constant",
            "Reduced Planck constant (h-bar), h divided by 2π", "J⋅s",
            current_version, "ℏ", reduced_planck_constant}},
          {"speed_of_light",
           {"speed_of_light",
            "Speed of light in vacuum, fundamental constant of spacetime",
            "m/s", current_version, "c", speed_of_light}},
          {"elementary_charge",
           {"elementary_charge",
            "Elementary charge, electric charge carried by a single proton",
            "C", current_version, "e", elementary_charge}},
          {"hartree_to_ev",
           {"hartree_to_ev", "Hartree to electron volt conversion factor",
            "eV/Eₕ", current_version, "", hartree_to_ev}},
          {"ev_to_hartree",
           {"ev_to_hartree", "Electron volt to Hartree conversion factor",
            "Eₕ/eV", current_version, "", ev_to_hartree}},
          {"hartree_to_kcal_per_mol",
           {"hartree_to_kcal_per_mol",
            "Hartree to kilocalorie per mole conversion factor",
            "kcal⋅mol⁻¹/Eₕ", current_version, "", hartree_to_kcal_per_mol}},
          {"kcal_per_mol_to_hartree",
           {"kcal_per_mol_to_hartree",
            "Kilocalorie per mole to Hartree conversion factor",
            "Eₕ/(kcal⋅mol⁻¹)", current_version, "", kcal_per_mol_to_hartree}},
          {"hartree_to_kj_per_mol",
           {"hartree_to_kj_per_mol",
            "Hartree to kilojoule per mole conversion factor", "kJ⋅mol⁻¹/Eₕ",
            current_version, "", hartree_to_kj_per_mol}},
          {"kj_per_mol_to_hartree",
           {"kj_per_mol_to_hartree",
            "Kilojoule per mole to Hartree conversion factor", "Eₕ/(kJ⋅mol⁻¹)",
            current_version, "", kj_per_mol_to_hartree}}};
}

}  // namespace qdk::chemistry::constants

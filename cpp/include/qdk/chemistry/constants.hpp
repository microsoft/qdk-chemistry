// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <array>
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
 * @brief Non-relativistic spin-restricted spherical HF configurations
 *
 * Each entry corresponds to an element with atomic number Z and contains the
 * number of electrons in each subshell: s, p, d, f
 *
 * Reference:
 *     Lehtola, S. (2020). "Fully numerical calculations on atoms with
 *     fractional occupations and range-separated functionals"
 *     Phys. Rev. A.
 *     10.1103/physreva.101.012516
 */
static constexpr std::array<std::array<size_t, 4>, 119> ATOMIC_CONFIGURATION = {
    {{0, 0, 0, 0},        // (dummy entry)
     {1, 0, 0, 0},        // H  1s¹
     {2, 0, 0, 0},        // He 1s²
     {3, 0, 0, 0},        // Li [He] 2s¹
     {4, 0, 0, 0},        // Be [He] 2s²
     {4, 1, 0, 0},        // B  [He] 2s² 2p¹
     {4, 2, 0, 0},        // C  [He] 2s² 2p²
     {4, 3, 0, 0},        // N  [He] 2s² 2p³
     {4, 4, 0, 0},        // O  [He] 2s² 2p⁴
     {4, 5, 0, 0},        // F  [He] 2s² 2p⁵
     {4, 6, 0, 0},        // Ne [He] 2s² 2p⁶
     {5, 6, 0, 0},        // Na [Ne] 3s¹
     {6, 6, 0, 0},        // Mg [Ne] 3s²
     {6, 7, 0, 0},        // Al [Ne] 3s² 3p¹
     {6, 8, 0, 0},        // Si [Ne] 3s² 3p²
     {6, 9, 0, 0},        // P  [Ne] 3s² 3p³
     {6, 10, 0, 0},       // S  [Ne] 3s² 3p⁴
     {6, 11, 0, 0},       // Cl [Ne] 3s² 3p⁵
     {6, 12, 0, 0},       // Ar [Ne] 3s² 3p⁶
     {7, 12, 0, 0},       // K  [Ar] 4s¹
     {8, 12, 0, 0},       // Ca [Ar] 4s²
     {8, 13, 0, 0},       // Sc [Ar] 4s² 4p¹
     {8, 12, 2, 0},       // Ti [Ar] 3d² 4s²
     {8, 12, 3, 0},       // V  [Ar] 3d³ 4s²
     {8, 12, 4, 0},       // Cr [Ar] 3d⁴ 4s²
     {6, 12, 7, 0},       // Mn [Ar] 3d⁷
     {6, 12, 8, 0},       // Fe [Ar] 3d⁸
     {6, 12, 9, 0},       // Co [Ar] 3d⁹
     {6, 12, 10, 0},      // Ni [Ar] 3d¹⁰
     {7, 12, 10, 0},      // Cu [Ar] 3d¹⁰ 4s¹
     {8, 12, 10, 0},      // Zn [Ar] 3d¹⁰ 4s²
     {8, 13, 10, 0},      // Ga [Ar] 3d¹⁰ 4s² 4p¹
     {8, 14, 10, 0},      // Ge [Ar] 3d¹⁰ 4s² 4p²
     {8, 15, 10, 0},      // As [Ar] 3d¹⁰ 4s² 4p³
     {8, 16, 10, 0},      // Se [Ar] 3d¹⁰ 4s² 4p⁴
     {8, 17, 10, 0},      // Br [Ar] 3d¹⁰ 4s² 4p⁵
     {8, 18, 10, 0},      // Kr [Ar] 3d¹⁰ 4s² 4p⁶
     {9, 18, 10, 0},      // Rb [Kr] 5s¹
     {10, 18, 10, 0},     // Sr [Kr] 5s²
     {10, 19, 10, 0},     // Y  [Kr] 5s² 5p¹
     {10, 18, 12, 0},     // Zr [Kr] 4d² 5s²
     {10, 18, 13, 0},     // Nb [Kr] 4d³ 5s²
     {8, 18, 16, 0},      // Mo [Kr] 4d⁶
     {8, 18, 17, 0},      // Tc [Kr] 4d⁷
     {8, 18, 18, 0},      // Ru [Kr] 4d⁸
     {8, 18, 19, 0},      // Rh [Kr] 4d⁹
     {8, 18, 20, 0},      // Pd [Kr] 4d¹⁰
     {9, 18, 20, 0},      // Ag [Kr] 4d¹⁰ 5s¹
     {10, 18, 20, 0},     // Cd [Kr] 4d¹⁰ 5s²
     {10, 19, 20, 0},     // In [Kr] 4d¹⁰ 5s² 5p¹
     {10, 20, 20, 0},     // Sn [Kr] 4d¹⁰ 5s² 5p²
     {10, 21, 20, 0},     // Sb [Kr] 4d¹⁰ 5s² 5p³
     {10, 22, 20, 0},     // Te [Kr] 4d¹⁰ 5s² 5p⁴
     {10, 23, 20, 0},     // I  [Kr] 4d¹⁰ 5s² 5p⁵
     {10, 24, 20, 0},     // Xe [Kr] 4d¹⁰ 5s² 5p⁶
     {11, 24, 20, 0},     // Cs [Xe] 6s¹
     {12, 24, 20, 0},     // Ba [Xe] 6s²
     {12, 24, 21, 0},     // La [Xe] 6s² 5d¹
     {12, 24, 22, 0},     // Ce [Xe] 6s² 5d²
     {12, 24, 21, 2},     // Pr [Xe] 4f² 5d¹ 6s²
     {12, 24, 20, 4},     // Nd [Xe] 4f⁴ 6s²
     {12, 24, 20, 5},     // Pm [Xe] 4f⁵ 6s²
     {12, 24, 20, 6},     // Sm [Xe] 4f⁶ 6s²
     {12, 24, 20, 7},     // Eu [Xe] 4f⁷ 6s²
     {11, 24, 20, 9},     // Gd [Xe] 4f⁹ 6s¹
     {10, 24, 20, 11},    // Tb [Xe] 4f¹¹
     {10, 24, 20, 12},    // Dy [Xe] 4f¹²
     {10, 24, 20, 13},    // Ho [Xe] 4f¹³
     {10, 24, 20, 14},    // Er [Xe] 4f¹⁴
     {11, 24, 20, 14},    // Tm [Xe] 4f¹⁴ 6s¹
     {12, 24, 20, 14},    // Yb [Xe] 4f¹⁴ 6s²
     {12, 25, 20, 14},    // Lu [Xe] 4f¹⁴ 6s² 6p¹
     {12, 24, 22, 14},    // Hf [Xe] 4f¹⁴ 5d² 6s²
     {12, 24, 23, 14},    // Ta [Xe] 4f¹⁴ 5d³ 6s²
     {10, 24, 26, 14},    // W  [Xe] 4f¹⁴ 5d⁶
     {10, 24, 27, 14},    // Re [Xe] 4f¹⁴ 5d⁷
     {10, 24, 28, 14},    // Os [Xe] 4f¹⁴ 5d⁸
     {10, 24, 29, 14},    // Ir [Xe] 4f¹⁴ 5d⁹
     {10, 24, 30, 14},    // Pt [Xe] 4f¹⁴ 5d¹⁰
     {11, 24, 30, 14},    // Au [Xe] 4f¹⁴ 5d¹⁰ 6s¹
     {12, 24, 30, 14},    // Hg [Xe] 4f¹⁴ 5d¹⁰ 6s²
     {12, 25, 30, 14},    // Tl [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p¹
     {12, 26, 30, 14},    // Pb [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p²
     {12, 27, 30, 14},    // Bi [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p³
     {12, 28, 30, 14},    // Po [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p⁴
     {12, 29, 30, 14},    // At [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p⁵
     {12, 30, 30, 14},    // Rn [Xe] 4f¹⁴ 5d¹⁰ 6s² 6p⁶
     {13, 30, 30, 14},    // Fr [Rn] 7s¹
     {14, 30, 30, 14},    // Ra [Rn] 7s²
     {14, 30, 31, 14},    // Ac [Rn] 6d¹ 7s²
     {14, 30, 32, 14},    // Th [Rn] 6d² 7s²
     {14, 30, 30, 17},    // Pa [Rn] 5f³ 7s²
     {14, 30, 30, 18},    // U  [Rn] 5f⁴ 7s²
     {14, 30, 30, 19},    // Np [Rn] 5f⁵ 7s²
     {13, 30, 30, 21},    // Pu [Rn] 5f⁷ 7s¹
     {12, 30, 30, 23},    // Am [Rn] 5f⁹
     {12, 30, 30, 24},    // Cm [Rn] 5f¹⁰
     {12, 30, 30, 25},    // Bk [Rn] 5f¹¹
     {12, 30, 30, 26},    // Cf [Rn] 5f¹²
     {12, 30, 30, 27},    // Es [Rn] 5f¹³
     {12, 30, 30, 28},    // Fm [Rn] 5f¹⁴
     {13, 30, 30, 28},    // Md [Rn] 5f¹⁴ 7s¹
     {14, 30, 30, 28},    // No [Rn] 5f¹⁴ 7s²
     {14, 30, 31, 28},    // Lr [Rn] 5f¹⁴ 6d¹ 7s²
     {14, 30, 32, 28},    // Rf [Rn] 5f¹⁴ 6d² 7s²
     {14, 30, 33, 28},    // Db [Rn] 5f¹⁴ 6d³ 7s²
     {12, 30, 36, 28},    // Sg [Rn] 5f¹⁴ 6d⁶
     {12, 30, 37, 28},    // Bh [Rn] 5f¹⁴ 6d⁷
     {12, 30, 38, 28},    // Hs [Rn] 5f¹⁴ 6d⁸
     {12, 30, 39, 28},    // Mt [Rn] 5f¹⁴ 6d⁹
     {12, 30, 40, 28},    // Ds [Rn] 5f¹⁴ 6d¹⁰
     {13, 30, 40, 28},    // Rg [Rn] 5f¹⁴ 6d¹⁰ 7s¹
     {14, 30, 40, 28},    // Cn [Rn] 5f¹⁴ 6d¹⁰ 7s²
     {14, 31, 40, 28},    // Nh [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p¹
     {14, 32, 40, 28},    // Fl [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p²
     {14, 33, 40, 28},    // Mc [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p³
     {14, 34, 40, 28},    // Lv [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p⁴
     {14, 35, 40, 28},    // Ts [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p⁵
     {14, 36, 40, 28}}};  // Og [Rn] 5f¹⁴ 6d¹⁰ 7s² 7p⁶

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

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>

// Define CIAAW version constants
#define QDK_CHEMISTRY_CIAAW_2024 2024

// Set default CIAAW version if not specified
#ifndef QDK_CHEMISTRY_CIAAW_VERSION
#define QDK_CHEMISTRY_CIAAW_VERSION QDK_CHEMISTRY_CIAAW_2024
#endif

/**
 * @file element_data.hpp
 * @brief Element definitions with atomic weight data from CIAAW standards
 *
 * This header provides element enumerations along with atomic weight data
 * from different CIAAW (Commission on Isotopic Abundances and Atomic Weights)
 * standards organized in version-specific namespaces.
 *
 * To use a specific CIAAW version, either use the fully qualified namespace
 * (e.g., qdk::chemistry::data::ciaaw_2024::atomic_weights) or import
 * the specific namespace.
 *
 * The default namespace uses CIAAW 2024, but other versions can be added
 * for compatibility and comparison purposes.
 *
 * To select a specific CIAAW version, define QDK_CHEMISTRY_CIAAW_VERSION
 */

namespace qdk::chemistry::data {

/**
 * @enum Element
 * @brief Enumeration for chemical elements in the periodic table (1-118)
 */
enum class Element : unsigned {
  // Period 1
  H = 1,
  He = 2,
  // Period 2
  Li = 3,
  Be = 4,
  B = 5,
  C = 6,
  N = 7,
  O = 8,
  F = 9,
  Ne = 10,
  // Period 3
  Na = 11,
  Mg = 12,
  Al = 13,
  Si = 14,
  P = 15,
  S = 16,
  Cl = 17,
  Ar = 18,
  // Period 4
  K = 19,
  Ca = 20,
  Sc = 21,
  Ti = 22,
  V = 23,
  Cr = 24,
  Mn = 25,
  Fe = 26,
  Co = 27,
  Ni = 28,
  Cu = 29,
  Zn = 30,
  Ga = 31,
  Ge = 32,
  As = 33,
  Se = 34,
  Br = 35,
  Kr = 36,
  // Period 5
  Rb = 37,
  Sr = 38,
  Y = 39,
  Zr = 40,
  Nb = 41,
  Mo = 42,
  Tc = 43,
  Ru = 44,
  Rh = 45,
  Pd = 46,
  Ag = 47,
  Cd = 48,
  In = 49,
  Sn = 50,
  Sb = 51,
  Te = 52,
  I = 53,
  Xe = 54,
  // Period 6
  Cs = 55,
  Ba = 56,
  La = 57,
  Ce = 58,
  Pr = 59,
  Nd = 60,
  Pm = 61,
  Sm = 62,
  Eu = 63,
  Gd = 64,
  Tb = 65,
  Dy = 66,
  Ho = 67,
  Er = 68,
  Tm = 69,
  Yb = 70,
  Lu = 71,
  Hf = 72,
  Ta = 73,
  W = 74,
  Re = 75,
  Os = 76,
  Ir = 77,
  Pt = 78,
  Au = 79,
  Hg = 80,
  Tl = 81,
  Pb = 82,
  Bi = 83,
  Po = 84,
  At = 85,
  Rn = 86,
  // Period 7
  Fr = 87,
  Ra = 88,
  Ac = 89,
  Th = 90,
  Pa = 91,
  U = 92,
  Np = 93,
  Pu = 94,
  Am = 95,
  Cm = 96,
  Bk = 97,
  Cf = 98,
  Es = 99,
  Fm = 100,
  Md = 101,
  No = 102,
  Lr = 103,
  Rf = 104,
  Db = 105,
  Sg = 106,
  Bh = 107,
  Hs = 108,
  Mt = 109,
  Ds = 110,
  Rg = 111,
  Cn = 112,
  Nh = 113,
  Fl = 114,
  Mc = 115,
  Lv = 116,
  Ts = 117,
  Og = 118
};

// Forward declarations for element lookup maps
extern const std::unordered_map<unsigned, std::string> CHARGE_TO_SYMBOL;

// Reverse lookup map for symbol to nuclear charge
extern std::unordered_map<std::string, unsigned> SYMBOL_TO_CHARGE;

/**
 * @namespace qdk::chemistry::data::ciaaw_2024
 * @brief CIAAW 2024 recommended values for atomic weights
 *
 * Standard atomic weights in AMU (atomic mass units) for all elements 1-118.
 * IUPAC (International Union of Pure and Applied Chemistry) Commission on
 * Isotopic Abundances and Atomic Weights (CIAAW), 2024.
 *
 * T. Prohaska et al., Standard atomic weights of the elements 2021 (IUPAC
 * Technical Report), Pure and Applied Chemistry 2022, 94, 573-600
 * (DOI: 10.1515/pac-2019-0603). :cite:`Prohaska2022`
 *
 * Standard atomic masses of gadolinium, lutetium, and zirconium have been
 * revised by IUPAC CIAAW in 2024 and these revisions are included here.
 *
 * For radioactive elements the mass number of the most stable isotope is used
 * as standard atomic weight.
 *
 * F. G. Kondev et al., The NUBASE2020 evaluation of nuclear physics properties,
 * Chinese Physics C 2021, 45, 030001 (DOI: 10.1088/1674-1137/abddae).
 * :cite:`Kondev2021`
 *
 *
 * Isotope masses in AMU (atomic mass units) for all elements 1-118.
 * IUPAC (International Union of Pure and Applied Chemistry) Commission
 * on Isotopic Abundances and Atomic Weights (CIAAW), 2021.
 *
 * W. J. Huang et al., The AME 2020 atomic mass evaluation (I). Evaluation of
 * input data, and adjustment procedures, Chinese Physics C 2021, 45, 030002
 * (DOI: 10.1088/1674-1137/abddb0). :cite:`Huang2021`
 *
 * M. Wang et al., The AME 2020 atomic mass evaluation (II). Tables, graphs and
 * references, Chinese Physics C 2021, 45, 030003
 * (DOI: 10.1088/1674-1137/abddaf). :cite:`Wang2021`
 *
 * F. G. Kondev et al., The NUBASE2020 evaluation of nuclear physics properties,
 * Chinese Physics C 2021, 45, 030001 (DOI: 10.1088/1674-1137/abddae).
 * :cite:`Kondev2021`
 */
namespace ciaaw_2024 {

/**
 * @brief Standard atomic weights and specific isotope masses in AMU (atomic
 * mass units)
 *
 * Includes both standard atomic weights for elements 1-118 and specific isotope
 * masses. For the standard atomic weights of radioactive elements, the mass
 * numbers of the most stable isotopes are used.
 */
extern const std::unordered_map<unsigned, double> atomic_weights;

/**
 * @brief Helper function to calculate unique unsigned keys for isotopes
 * @param Z Atomic number (proton count)
 * @param A Mass number (proton + neutron count) for a specific isotope mass or
 * 0 for a standard atomic weight
 * @return Encoded unsigned key for atomic weight representation
 *
 * For isotopes, put the atomic mass number in the unsigned's bit
 * representation. Z ranges up to 118 < 128 = 2^7, so we use 7 bits for Z. A
 * ranges up to 295 < 512 = 2^9, so we need at least 16 bits total.
 */
constexpr unsigned isotope(const unsigned Z, const unsigned A) noexcept {
  // Z ranges up to 118 < 128 = 2^7
  constexpr unsigned Z_bits = 7;
  // A ranges up to 295 < 512 = 2^9
  // so we need at least 16 bits to represent both in the underlying value
  static_assert(sizeof(unsigned) >= 2,
                "Unsigned type is smaller than two bytes, keys for isotopes "
                "do not work");

  return (A << Z_bits) + Z;
}

/**
 * @brief Get atomic weight for an Element
 * @param element Element enum value
 * @return Atomic weight in AMU
 * @throws std::invalid_argument if element is not found in the atomic weights
 * map
 *
 * This function converts Element to unsigned and looks up the standard atomic
 * weight.
 */
inline double get_atomic_weight(Element element) {
  unsigned atomic_number = static_cast<unsigned>(element);
  auto it = atomic_weights.find(atomic_number);
  if (it == atomic_weights.end()) {
    throw std::invalid_argument("Unknown element for mass lookup: Z = " +
                                std::to_string(atomic_number));
  }
  return it->second;
}

/**
 * @brief Get atomic weight for an atomic number Z and mass number A
 * @param Z Atomic number (proton count)
 * @param A Mass number (proton + neutron count) for a specific isotope mass or
 * 0 for a standard atomic weight
 * @return Atomic weight in AMU
 * @throws std::invalid_argument if pair of atomic number Z and mass number A
 * is not found in the atomic weights map
 *
 * This function looks up the atomic weight for the given pair of atomic number
 * Z and mass number A.
 */
inline double get_atomic_weight(unsigned Z, unsigned A) {
  auto it = atomic_weights.find(isotope(Z, A));
  if (it == atomic_weights.end()) {
    throw std::invalid_argument(
        "Unknown element/isotope for mass lookup: Z = " + std::to_string(Z) +
        ", A = " + std::to_string(A));
  }
  return it->second;
}

}  // namespace ciaaw_2024

// Helper to determine current CIAAW version in use
constexpr const char* get_current_ciaaw_version() {
#if QDK_CHEMISTRY_CIAAW_VERSION == QDK_CHEMISTRY_CIAAW_2024
  return "CIAAW 2024";
#else
  static_assert(QDK_CHEMISTRY_CIAAW_VERSION == QDK_CHEMISTRY_CIAAW_2024,
                "Unsupported QDK_CHEMISTRY_CIAAW_VERSION. Supported versions: "
                "QDK_CHEMISTRY_CIAAW_2024");
#endif
}

// Use the selected CIAAW version as default
// To use CIAAW 2024, define:
//   #define QDK_CHEMISTRY_CIAAW_VERSION QDK_CHEMISTRY_CIAAW_2024 (default)
#if QDK_CHEMISTRY_CIAAW_VERSION == QDK_CHEMISTRY_CIAAW_2024
using namespace ciaaw_2024;
#else
static_assert(QDK_CHEMISTRY_CIAAW_VERSION == QDK_CHEMISTRY_CIAAW_2024,
              "Unsupported QDK_CHEMISTRY_CIAAW_VERSION. Supported versions: "
              "QDK_CHEMISTRY_CIAAW_2024");
#endif
}  // namespace qdk::chemistry::data

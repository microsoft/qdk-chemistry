// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <qdk/chemistry/constants.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include "filename_utils.hpp"
#include "hdf5_serialization.hpp"
#include "json_serialization.hpp"

// TODO (NAB):  This file is also very long.  Should we break it up? workitem:
// 41353

namespace qdk::chemistry::data {

// Static data for element lookup
static const std::unordered_map<unsigned, std::string> CHARGE_TO_SYMBOL = {
    {1, "H"},    {2, "He"},   {3, "Li"},   {4, "Be"},   {5, "B"},
    {6, "C"},    {7, "N"},    {8, "O"},    {9, "F"},    {10, "Ne"},
    {11, "Na"},  {12, "Mg"},  {13, "Al"},  {14, "Si"},  {15, "P"},
    {16, "S"},   {17, "Cl"},  {18, "Ar"},  {19, "K"},   {20, "Ca"},
    {21, "Sc"},  {22, "Ti"},  {23, "V"},   {24, "Cr"},  {25, "Mn"},
    {26, "Fe"},  {27, "Co"},  {28, "Ni"},  {29, "Cu"},  {30, "Zn"},
    {31, "Ga"},  {32, "Ge"},  {33, "As"},  {34, "Se"},  {35, "Br"},
    {36, "Kr"},  {37, "Rb"},  {38, "Sr"},  {39, "Y"},   {40, "Zr"},
    {41, "Nb"},  {42, "Mo"},  {43, "Tc"},  {44, "Ru"},  {45, "Rh"},
    {46, "Pd"},  {47, "Ag"},  {48, "Cd"},  {49, "In"},  {50, "Sn"},
    {51, "Sb"},  {52, "Te"},  {53, "I"},   {54, "Xe"},  {55, "Cs"},
    {56, "Ba"},  {57, "La"},  {58, "Ce"},  {59, "Pr"},  {60, "Nd"},
    {61, "Pm"},  {62, "Sm"},  {63, "Eu"},  {64, "Gd"},  {65, "Tb"},
    {66, "Dy"},  {67, "Ho"},  {68, "Er"},  {69, "Tm"},  {70, "Yb"},
    {71, "Lu"},  {72, "Hf"},  {73, "Ta"},  {74, "W"},   {75, "Re"},
    {76, "Os"},  {77, "Ir"},  {78, "Pt"},  {79, "Au"},  {80, "Hg"},
    {81, "Tl"},  {82, "Pb"},  {83, "Bi"},  {84, "Po"},  {85, "At"},
    {86, "Rn"},  {87, "Fr"},  {88, "Ra"},  {89, "Ac"},  {90, "Th"},
    {91, "Pa"},  {92, "U"},   {93, "Np"},  {94, "Pu"},  {95, "Am"},
    {96, "Cm"},  {97, "Bk"},  {98, "Cf"},  {99, "Es"},  {100, "Fm"},
    {101, "Md"}, {102, "No"}, {103, "Lr"}, {104, "Rf"}, {105, "Db"},
    {106, "Sg"}, {107, "Bh"}, {108, "Hs"}, {109, "Mt"}, {110, "Ds"},
    {111, "Rg"}, {112, "Cn"}, {113, "Nh"}, {114, "Fl"}, {115, "Mc"},
    {116, "Lv"}, {117, "Ts"}, {118, "Og"}};

// Reverse lookup map for symbol to nuclear charge (lazy initialization)
static std::unordered_map<std::string, unsigned> SYMBOL_TO_CHARGE;

// Standard atomic masses in AMU (atomic mass units) for all elements 1-118
static const std::unordered_map<Element, double> STANDARD_ATOMIC_MASSES = {
    // Period 1
    {Element::H, 1.008},
    {Element::He, 4.0026},
    // Period 2
    {Element::Li, 6.94},
    {Element::Be, 9.0122},
    {Element::B, 10.81},
    {Element::C, 12.011},
    {Element::N, 14.007},
    {Element::O, 15.999},
    {Element::F, 18.998},
    {Element::Ne, 20.180},
    // Period 3
    {Element::Na, 22.990},
    {Element::Mg, 24.305},
    {Element::Al, 26.982},
    {Element::Si, 28.085},
    {Element::P, 30.974},
    {Element::S, 32.06},
    {Element::Cl, 35.45},
    {Element::Ar, 39.948},
    // Period 4
    {Element::K, 39.098},
    {Element::Ca, 40.078},
    {Element::Sc, 44.956},
    {Element::Ti, 47.867},
    {Element::V, 50.942},
    {Element::Cr, 51.996},
    {Element::Mn, 54.938},
    {Element::Fe, 55.845},
    {Element::Co, 58.933},
    {Element::Ni, 58.693},
    {Element::Cu, 63.546},
    {Element::Zn, 65.38},
    {Element::Ga, 69.723},
    {Element::Ge, 72.630},
    {Element::As, 74.922},
    {Element::Se, 78.971},
    {Element::Br, 79.904},
    {Element::Kr, 83.798},
    // Period 5
    {Element::Rb, 85.468},
    {Element::Sr, 87.62},
    {Element::Y, 88.906},
    {Element::Zr, 91.224},
    {Element::Nb, 92.906},
    {Element::Mo, 95.95},
    {Element::Tc, 98.0},
    {Element::Ru, 101.07},
    {Element::Rh, 102.91},
    {Element::Pd, 106.42},
    {Element::Ag, 107.87},
    {Element::Cd, 112.41},
    {Element::In, 114.82},
    {Element::Sn, 118.71},
    {Element::Sb, 121.76},
    {Element::Te, 127.60},
    {Element::I, 126.90},
    {Element::Xe, 131.29},
    // Period 6
    {Element::Cs, 132.91},
    {Element::Ba, 137.33},
    {Element::La, 138.91},
    {Element::Ce, 140.12},
    {Element::Pr, 140.91},
    {Element::Nd, 144.24},
    {Element::Pm, 145.0},
    {Element::Sm, 150.36},
    {Element::Eu, 151.96},
    {Element::Gd, 157.25},
    {Element::Tb, 158.93},
    {Element::Dy, 162.50},
    {Element::Ho, 164.93},
    {Element::Er, 167.26},
    {Element::Tm, 168.93},
    {Element::Yb, 173.05},
    {Element::Lu, 174.97},
    {Element::Hf, 178.49},
    {Element::Ta, 180.95},
    {Element::W, 183.84},
    {Element::Re, 186.21},
    {Element::Os, 190.23},
    {Element::Ir, 192.22},
    {Element::Pt, 195.08},
    {Element::Au, 196.97},
    {Element::Hg, 200.59},
    {Element::Tl, 204.38},
    {Element::Pb, 207.2},
    {Element::Bi, 208.98},
    {Element::Po, 209.0},
    {Element::At, 210.0},
    {Element::Rn, 222.0},
    // Period 7
    {Element::Fr, 223.0},
    {Element::Ra, 226.0},
    {Element::Ac, 227.0},
    {Element::Th, 232.04},
    {Element::Pa, 231.04},
    {Element::U, 238.03},
    {Element::Np, 237.0},
    {Element::Pu, 244.0},
    {Element::Am, 243.0},
    {Element::Cm, 247.0},
    {Element::Bk, 247.0},
    {Element::Cf, 251.0},
    {Element::Es, 252.0},
    {Element::Fm, 257.0},
    {Element::Md, 258.0},
    {Element::No, 259.0},
    {Element::Lr, 262.0},
    {Element::Rf, 267.0},
    {Element::Db, 270.0},
    {Element::Sg, 271.0},
    {Element::Bh, 270.0},
    {Element::Hs, 277.0},
    {Element::Mt, 276.0},
    {Element::Ds, 281.0},
    {Element::Rg, 280.0},
    {Element::Cn, 285.0},
    {Element::Nh, 284.0},
    {Element::Fl, 289.0},
    {Element::Mc, 288.0},
    {Element::Lv, 293.0},
    {Element::Ts, 294.0},
    {Element::Og, 294.0}};

Structure::Structure(const Eigen::MatrixXd& coordinates,
                     const std::vector<Element>& elements,
                     const Eigen::VectorXd& masses,
                     const Eigen::VectorXd& nuclear_charges)
    : _coordinates(coordinates),
      _elements(elements),
      _masses([&]() {
        if (masses.size() == 0) {
          if (elements.empty()) {
            return Eigen::VectorXd();  // Create completely empty vector
          } else {
            Eigen::VectorXd default_masses(elements.size());
            for (size_t i = 0; i < elements.size(); ++i) {
              default_masses[i] = get_standard_atomic_mass(elements[i]);
            }
            return default_masses;
          }
        } else {
          return masses;
        }
      }()),
      _nuclear_charges([&]() {
        if (nuclear_charges.size() == 0) {
          if (elements.empty()) {
            return Eigen::VectorXd();  // Create completely empty vector
          } else {
            Eigen::VectorXd default_charges(elements.size());
            for (size_t i = 0; i < elements.size(); ++i) {
              default_charges[i] = element_to_nuclear_charge(elements[i]);
            }
            return default_charges;
          }
        } else {
          return nuclear_charges;
        }
      }()) {
  if (!_is_valid()) {
    throw std::invalid_argument(
        "Invalid structure: inconsistent dimensions or invalid data");
  }
}

Structure::Structure(const Eigen::MatrixXd& coordinates,
                     const std::vector<std::string>& symbols,
                     const Eigen::VectorXd& masses,
                     const Eigen::VectorXd& nuclear_charges)
    : _coordinates(coordinates),
      _elements([&]() {
        std::vector<Element> elements;
        elements.reserve(symbols.size());
        for (const auto& symbol : symbols) {
          std::string fixed_symbol = _fix_symbol_capitalization(symbol);
          elements.push_back(symbol_to_element(fixed_symbol));
        }
        return elements;
      }()),
      _masses([&]() {
        if (masses.size() == 0) {
          if (symbols.empty()) {
            return Eigen::VectorXd();  // Create completely empty vector
          } else {
            Eigen::VectorXd default_masses(symbols.size());
            for (size_t i = 0; i < symbols.size(); ++i) {
              std::string fixed_symbol = _fix_symbol_capitalization(symbols[i]);
              Element element = symbol_to_element(fixed_symbol);
              default_masses[i] = get_standard_atomic_mass(element);
            }
            return default_masses;
          }
        } else {
          return masses;
        }
      }()),
      _nuclear_charges([&]() {
        if (nuclear_charges.size() == 0) {
          if (symbols.empty()) {
            return Eigen::VectorXd();  // Create completely empty vector
          } else {
            Eigen::VectorXd default_charges(symbols.size());
            for (size_t i = 0; i < symbols.size(); ++i) {
              std::string fixed_symbol = _fix_symbol_capitalization(symbols[i]);
              Element element = symbol_to_element(fixed_symbol);
              default_charges[i] = element_to_nuclear_charge(element);
            }
            return default_charges;
          }
        } else {
          return nuclear_charges;
        }
      }()) {
  if (!_is_valid()) {
    throw std::invalid_argument(
        "Invalid structure: inconsistent dimensions or invalid data");
  }
}

Structure::Structure(const std::vector<Eigen::Vector3d>& coordinates,
                     const std::vector<std::string>& symbols,
                     const std::vector<double>& masses,
                     const std::vector<double>& nuclear_charges)
    : _coordinates([&]() {
        if (coordinates.empty()) {
          return Eigen::MatrixXd();  // Create completely empty matrix (0x0)
        } else {
          Eigen::MatrixXd coords(coordinates.size(), 3);
          for (size_t i = 0; i < coordinates.size(); ++i) {
            coords.row(i) = coordinates[i];
          }
          return coords;
        }
      }()),
      _elements([&]() {
        std::vector<Element> elements;
        elements.reserve(symbols.size());
        for (const auto& symbol : symbols) {
          std::string fixed_symbol = _fix_symbol_capitalization(symbol);
          elements.push_back(symbol_to_element(fixed_symbol));
        }
        return elements;
      }()),
      _masses([&]() {
        if (masses.empty()) {
          if (symbols.empty()) {
            return Eigen::VectorXd();  // Create completely empty vector
          } else {
            Eigen::VectorXd default_masses(symbols.size());
            for (size_t i = 0; i < symbols.size(); ++i) {
              std::string fixed_symbol = _fix_symbol_capitalization(symbols[i]);
              Element element = symbol_to_element(fixed_symbol);
              default_masses[i] = get_standard_atomic_mass(element);
            }
            return default_masses;
          }
        } else {
          Eigen::VectorXd masses_vec(masses.size());
          for (size_t i = 0; i < masses.size(); ++i) {
            masses_vec[i] = masses[i];
          }
          return masses_vec;
        }
      }()),
      _nuclear_charges([&]() {
        if (nuclear_charges.empty()) {
          if (symbols.empty()) {
            return Eigen::VectorXd();  // Create completely empty vector
          } else {
            Eigen::VectorXd default_charges(symbols.size());
            for (size_t i = 0; i < symbols.size(); ++i) {
              std::string fixed_symbol = _fix_symbol_capitalization(symbols[i]);
              Element element = symbol_to_element(fixed_symbol);
              default_charges[i] = element_to_nuclear_charge(element);
            }
            return default_charges;
          }
        } else {
          Eigen::VectorXd charges_vec(nuclear_charges.size());
          for (size_t i = 0; i < nuclear_charges.size(); ++i) {
            charges_vec[i] = nuclear_charges[i];
          }
          return charges_vec;
        }
      }()) {
  if (!_is_valid()) {
    throw std::invalid_argument(
        "Invalid structure: inconsistent dimensions or invalid data");
  }
}

Structure::Structure(const std::vector<Eigen::Vector3d>& coordinates,
                     const std::vector<Element>& elements,
                     const std::vector<double>& masses,
                     const std::vector<double>& nuclear_charges)
    : _coordinates([&]() {
        if (coordinates.empty()) {
          return Eigen::MatrixXd();  // Create completely empty matrix (0x0)
        } else {
          Eigen::MatrixXd coords(coordinates.size(), 3);
          for (size_t i = 0; i < coordinates.size(); ++i) {
            coords.row(i) = coordinates[i];
          }
          return coords;
        }
      }()),
      _elements(elements),
      _masses([&]() {
        if (masses.empty()) {
          if (elements.empty()) {
            return Eigen::VectorXd();  // Create completely empty vector
          } else {
            Eigen::VectorXd default_masses(elements.size());
            for (size_t i = 0; i < elements.size(); ++i) {
              default_masses[i] = get_standard_atomic_mass(elements[i]);
            }
            return default_masses;
          }
        } else {
          Eigen::VectorXd masses_vec(masses.size());
          for (size_t i = 0; i < masses.size(); ++i) {
            masses_vec[i] = masses[i];
          }
          return masses_vec;
        }
      }()),
      _nuclear_charges([&]() {
        if (nuclear_charges.empty()) {
          if (elements.empty()) {
            return Eigen::VectorXd();  // Create completely empty vector
          } else {
            Eigen::VectorXd default_charges(elements.size());
            for (size_t i = 0; i < elements.size(); ++i) {
              default_charges[i] = element_to_nuclear_charge(elements[i]);
            }
            return default_charges;
          }
        } else {
          Eigen::VectorXd charges_vec(nuclear_charges.size());
          for (size_t i = 0; i < nuclear_charges.size(); ++i) {
            charges_vec[i] = nuclear_charges[i];
          }
          return charges_vec;
        }
      }()) {
  if (!_is_valid()) {
    throw std::invalid_argument(
        "Invalid structure: inconsistent dimensions or invalid data");
  }
}

Eigen::Vector3d Structure::get_atom_coordinates(size_t atom_index) const {
  if (atom_index >= get_num_atoms()) {
    throw std::out_of_range("Atom index out of range");
  }
  return _coordinates.row(atom_index);
}

Element Structure::get_atom_element(size_t atom_index) const {
  if (atom_index >= get_num_atoms()) {
    throw std::out_of_range("Atom index out of range");
  }
  return _elements[atom_index];
}

double Structure::get_atom_mass(size_t atom_index) const {
  if (atom_index >= get_num_atoms()) {
    throw std::out_of_range("Atom index out of range");
  }
  return _masses[atom_index];
}

double Structure::get_atom_nuclear_charge(size_t atom_index) const {
  if (atom_index >= get_num_atoms()) {
    throw std::out_of_range("Atom index out of range");
  }
  return _nuclear_charges[atom_index];
}

std::string Structure::get_atom_symbol(size_t atom_index) const {
  return element_to_symbol(get_atom_element(atom_index));
}

std::vector<std::string> Structure::get_atomic_symbols() const {
  std::vector<std::string> symbols;
  symbols.reserve(_elements.size());
  for (const auto& element : _elements) {
    symbols.push_back(element_to_symbol(element));
  }
  return symbols;
}

bool Structure::_is_valid() const {
  if (_elements.empty()) {
    return _coordinates.rows() == 0 && _coordinates.cols() == 0 &&
           _masses.size() == 0 && _nuclear_charges.size() == 0;
  }

  size_t num_atoms = _elements.size();
  return _coordinates.rows() == static_cast<int>(num_atoms) &&
         _coordinates.cols() == 3 &&
         _masses.size() == static_cast<int>(num_atoms) &&
         _nuclear_charges.size() == static_cast<int>(num_atoms);
}

double Structure::get_total_mass() const { return _masses.sum(); }

nlohmann::json Structure::to_json() const {
  nlohmann::json j;

  // Store version first
  j["version"] = SERIALIZATION_VERSION;

  j["units"] = "bohr";

  j["num_atoms"] = get_num_atoms();

  // Store atomic symbols and element enums
  j["symbols"] = get_atomic_symbols();

  std::vector<unsigned> element_values;
  element_values.reserve(_elements.size());
  for (const auto& element : _elements) {
    element_values.push_back(static_cast<unsigned>(element));
  }
  j["elements"] = element_values;

  // Store masses and nuclear charges
  // Convert Eigen::VectorXd to std::vector for JSON serialization
  std::vector<double> masses_vec(_masses.data(),
                                 _masses.data() + _masses.size());
  j["masses"] = masses_vec;

  std::vector<double> nuclear_charges_vec(
      _nuclear_charges.data(),
      _nuclear_charges.data() + _nuclear_charges.size());
  j["nuclear_charges"] = nuclear_charges_vec;

  // Store coordinates as nested array
  j["coordinates"] = nlohmann::json::array();
  for (size_t i = 0; i < get_num_atoms(); ++i) {
    nlohmann::json atom_coords = nlohmann::json::array();
    for (int j_coord = 0; j_coord < 3; ++j_coord) {
      atom_coords.push_back(_coordinates(i, j_coord));
    }
    j["coordinates"].push_back(atom_coords);
  }

  return j;
}

std::shared_ptr<Structure> Structure::from_json(const nlohmann::json& j) {
  try {
    // Validate version first (only if version field exists, for backward
    // compatibility)
    if (j.contains("version")) {
      validate_serialization_version(SERIALIZATION_VERSION, j["version"]);
    }

    if (!j.contains("units")) {
      throw std::runtime_error("JSON missing units data");
    }

    if (!j.contains("coordinates")) {
      throw std::runtime_error("JSON missing coordinates data");
    }

    // Check input units
    std::string units = j["units"].get<std::string>();
    std::transform(units.begin(), units.end(), units.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    if (units != "bohr" && units != "angstrom") {
      throw std::runtime_error(
          "Unsupported in Structure deserialization units: " + units);
    }

    // Load coordinates
    auto coords_json = j["coordinates"];
    if (!coords_json.is_array()) {
      throw std::runtime_error("Invalid coordinates format in JSON");
    }

    size_t num_atoms = coords_json.size();
    Eigen::MatrixXd coordinates(num_atoms, 3);

    for (size_t i = 0; i < num_atoms; ++i) {
      if (!coords_json[i].is_array() || coords_json[i].size() != 3) {
        throw std::runtime_error("Invalid coordinate format for atom " +
                                 std::to_string(i));
      }
      for (int j_coord = 0; j_coord < 3; ++j_coord) {
        // keep Bohr as-is
        coordinates(i, j_coord) = static_cast<double>(coords_json[i][j_coord]);
      }
    }

    // Convert units to Bohr if necessary
    if (units == "angstrom") {
      coordinates *= qdk::chemistry::constants::angstrom_to_bohr;
    }

    // Load elements (prefer elements over nuclear_charges if available)
    std::vector<Element> elements;
    if (j.contains("elements")) {
      auto elements_json = j["elements"].get<std::vector<unsigned>>();
      elements.reserve(elements_json.size());
      for (unsigned elem_val : elements_json) {
        elements.push_back(static_cast<Element>(elem_val));
      }
    } else if (j.contains("nuclear_charges")) {
      // Fallback to nuclear charges
      auto nuclear_charges = j["nuclear_charges"].get<std::vector<unsigned>>();
      elements.reserve(nuclear_charges.size());
      for (unsigned charge : nuclear_charges) {
        elements.push_back(nuclear_charge_to_element(charge));
      }
    } else if (j.contains("symbols")) {
      // Fallback to symbols
      auto symbols = j["symbols"].get<std::vector<std::string>>();
      elements.reserve(symbols.size());
      for (const auto& symbol : symbols) {
        std::string fixed_symbol = _fix_symbol_capitalization(symbol);
        elements.push_back(symbol_to_element(fixed_symbol));
      }
    } else {
      throw std::runtime_error(
          "JSON missing element/nuclear_charges/symbols data");
    }

    // Load masses (use standard if not provided)
    Eigen::VectorXd masses;
    if (j.contains("masses")) {
      std::vector<double> masses_vec = j["masses"].get<std::vector<double>>();
      masses =
          Eigen::Map<Eigen::VectorXd>(masses_vec.data(), masses_vec.size());
    } else {
      masses = Eigen::VectorXd(elements.size());
      for (size_t i = 0; i < elements.size(); ++i) {
        masses[i] = get_standard_atomic_mass(elements[i]);
      }
    }

    // Load nuclear charges (use standard if not provided)
    Eigen::VectorXd nuclear_charges;
    if (j.contains("nuclear_charges")) {
      std::vector<double> nuclear_charges_vec =
          j["nuclear_charges"].get<std::vector<double>>();
      nuclear_charges = Eigen::Map<Eigen::VectorXd>(nuclear_charges_vec.data(),
                                                    nuclear_charges_vec.size());
    } else {
      nuclear_charges = Eigen::VectorXd(elements.size());
      for (size_t i = 0; i < elements.size(); ++i) {
        nuclear_charges[i] = element_to_nuclear_charge(elements[i]);
      }
    }

    return std::make_shared<Structure>(coordinates, elements, masses,
                                       nuclear_charges);

  } catch (const nlohmann::json::exception& e) {
    throw std::runtime_error("JSON parsing error: " + std::string(e.what()));
  }
}

void Structure::to_json_file(const std::string& filename) const {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "structure");

  _to_json_file(validated_filename);
}

std::shared_ptr<Structure> Structure::from_json_file(
    const std::string& filename) {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "structure");

  return _from_json_file(validated_filename);
}

void Structure::_to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  auto j = to_json();
  file << j.dump(2);  // Pretty print with 2-space indentation

  if (file.fail()) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

std::shared_ptr<Structure> Structure::_from_json_file(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Unable to open Structure JSON file '" + filename +
        "'. Please check that the file exists and you have read permissions.");
  }

  nlohmann::json j;
  file >> j;

  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return from_json(j);
}

std::string Structure::to_xyz(const std::string& comment) const {
  std::ostringstream oss;

  // First line: number of atoms
  oss << get_num_atoms() << "\n";

  // Second line: comment
  oss << comment << "\n";

  // Atom lines: symbol x y z
  // Atom lines: symbol x y z (in Angstrom)
  for (size_t i = 0; i < get_num_atoms(); ++i) {
    std::string symbol = get_atom_symbol(i);
    oss << std::setw(2) << std::left << symbol << " ";
    oss << std::fixed << std::setprecision(6);
    oss << std::setw(12)
        << _coordinates(i, 0) * qdk::chemistry::constants::bohr_to_angstrom
        << " ";
    oss << std::setw(12)
        << _coordinates(i, 1) * qdk::chemistry::constants::bohr_to_angstrom
        << " ";
    oss << std::setw(12)
        << _coordinates(i, 2) * qdk::chemistry::constants::bohr_to_angstrom;
    if (i < get_num_atoms() - 1) {
      oss << "\n";
    }
  }

  return oss.str();
}

std::shared_ptr<Structure> Structure::from_xyz(const std::string& xyz_string) {
  std::istringstream iss(xyz_string);
  std::string line;

  // Read number of atoms
  if (!std::getline(iss, line)) {
    throw std::runtime_error("Invalid XYZ format: missing number of atoms");
  }

  unsigned num_atoms;
  try {
    num_atoms = std::stoul(line);
  } catch (const std::exception&) {
    throw std::runtime_error(
        "Invalid XYZ format: cannot parse number of atoms");
  }

  // Skip comment line
  if (!std::getline(iss, line)) {
    throw std::runtime_error("Invalid XYZ format: missing comment line");
  }

  // Prepare data for construction
  std::vector<Eigen::Vector3d> coordinates;
  std::vector<std::string> symbols;
  coordinates.reserve(num_atoms);
  symbols.reserve(num_atoms);

  // Read atom data (XYZ expected in Angstrom)
  for (unsigned i = 0; i < num_atoms; ++i) {
    if (!std::getline(iss, line)) {
      throw std::runtime_error(
          "Invalid XYZ format: missing atom data for atom " +
          std::to_string(i));
    }

    std::istringstream line_stream(line);
    std::string symbol;
    double x, y, z;

    if (!(line_stream >> symbol >> x >> y >> z)) {
      throw std::runtime_error(
          "Invalid XYZ format: cannot parse atom data for atom " +
          std::to_string(i));
    }

    Eigen::Vector3d coords_ang(x, y, z);
    Eigen::Vector3d coords_bohr =
        coords_ang * qdk::chemistry::constants::angstrom_to_bohr;

    coordinates.push_back(coords_bohr);
    symbols.push_back(symbol);
  }

  return std::make_shared<Structure>(coordinates, symbols);
}

void Structure::to_xyz_file(const std::string& filename,
                            const std::string& comment) const {
  _to_xyz_file(filename, comment);
}

std::shared_ptr<Structure> Structure::from_xyz_file(
    const std::string& filename) {
  return _from_xyz_file(filename);
}

void Structure::_to_xyz_file(const std::string& filename,
                             const std::string& comment) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  file << to_xyz(comment);

  if (file.fail()) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

std::shared_ptr<Structure> Structure::_from_xyz_file(
    const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }

  std::string xyz_content((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());

  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return from_xyz(xyz_content);
}

void Structure::to_file(const std::string& filename,
                        const std::string& type) const {
  if (type == "json") {
    _to_json_file(filename);
  } else if (type == "xyz") {
    _to_xyz_file(filename);
  } else if (type == "hdf5") {
    _to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, xyz, hdf5");
  }
}

std::shared_ptr<Structure> Structure::from_file(const std::string& filename,
                                                const std::string& type) {
  if (type == "json") {
    return _from_json_file(filename);
  } else if (type == "xyz") {
    return _from_xyz_file(filename);
  } else if (type == "hdf5") {
    return _from_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, xyz, hdf5");
  }
}

std::string Structure::get_summary() const {
  std::ostringstream oss;

  oss << "Structure Summary:\n";
  oss << "  Number of atoms: " << get_num_atoms() << "\n";

  if (!is_empty()) {
    // Count atoms by element
    std::unordered_map<std::string, unsigned> element_counts;
    for (unsigned i = 0; i < get_num_atoms(); ++i) {
      std::string symbol = get_atom_symbol(i);
      element_counts[symbol]++;
    }

    oss << "  Composition: ";
    bool first = true;
    for (const auto& pair : element_counts) {
      if (!first) oss << ", ";
      oss << pair.first << pair.second;
      first = false;
    }
    oss << "\n";

    oss << "  Total mass: " << std::fixed << std::setprecision(3)
        << get_total_mass() << " AMU\n";

    oss << "  Nuclear Repulsion Energy: " << std::fixed << std::setprecision(3)
        << calculate_nuclear_repulsion_energy() << " Eh\n";
  }

  return oss.str();
}

double Structure::calculate_nuclear_repulsion_energy() const {
  // Return 0 if there are less than 2 atoms (no repulsion possible)
  if (get_num_atoms() < 2) {
    return 0.0;
  }

  double repulsion_energy = 0.0;
  for (size_t i = 0; i < get_num_atoms(); ++i) {
    for (size_t j = i + 1; j < get_num_atoms(); ++j) {
      // Get nuclear charges for atoms i and j
      double charge_i = _nuclear_charges(i);
      double charge_j = _nuclear_charges(j);

      // Positions are stored in Bohr already
      Eigen::Vector3d pos_i = _coordinates.row(i);
      Eigen::Vector3d pos_j = _coordinates.row(j);
      // Distance in Bohr
      double distance_bohr = (pos_j - pos_i).norm();

      // Add contribution to repulsion energy (in hartree)
      // E = Z_i * Z_j / R_ij
      repulsion_energy += charge_i * charge_j / distance_bohr;
    }
  }

  return repulsion_energy;
}

std::string Structure::_fix_symbol_capitalization(const std::string& symbol) {
  if (symbol.empty()) {
    return symbol;
  }

  std::string fixed = symbol;

  // Convert to lowercase first
  std::transform(fixed.begin(), fixed.end(), fixed.begin(), ::tolower);

  // Capitalize the first letter
  fixed[0] = std::toupper(fixed[0]);

  return fixed;
}

Element Structure::symbol_to_element(const std::string& symbol) {
  std::string fixed_symbol = _fix_symbol_capitalization(symbol);
  unsigned charge = symbol_to_nuclear_charge(fixed_symbol);
  return static_cast<Element>(charge);
}

std::string Structure::element_to_symbol(Element element) {
  return nuclear_charge_to_symbol(static_cast<unsigned>(element));
}

unsigned Structure::symbol_to_nuclear_charge(const std::string& symbol) {
  std::string fixed_symbol = _fix_symbol_capitalization(symbol);

  // Lazy initialization: build the reverse map only once when first needed
  if (SYMBOL_TO_CHARGE.empty()) {
    for (const auto& pair : CHARGE_TO_SYMBOL) {
      SYMBOL_TO_CHARGE[pair.second] = pair.first;
    }
  }

  auto it = SYMBOL_TO_CHARGE.find(fixed_symbol);
  if (it != SYMBOL_TO_CHARGE.end()) {
    return it->second;
  }

  throw std::invalid_argument("Unknown atomic symbol: " + symbol +
                              " (normalized to: " + fixed_symbol + ")");
}

std::string Structure::nuclear_charge_to_symbol(unsigned nuclear_charge) {
  auto it = CHARGE_TO_SYMBOL.find(nuclear_charge);
  if (it == CHARGE_TO_SYMBOL.end()) {
    throw std::invalid_argument("Unknown nuclear charge: " +
                                std::to_string(nuclear_charge));
  }
  return it->second;
}

unsigned Structure::element_to_nuclear_charge(Element element) {
  return static_cast<unsigned>(element);
}

Element Structure::nuclear_charge_to_element(unsigned nuclear_charge) {
  if (nuclear_charge < 1 || nuclear_charge > 118) {
    throw std::invalid_argument("Unknown nuclear charge: " +
                                std::to_string(nuclear_charge));
  }
  return static_cast<Element>(nuclear_charge);
}

double Structure::get_standard_atomic_mass(Element element) {
  auto it = STANDARD_ATOMIC_MASSES.find(element);
  if (it == STANDARD_ATOMIC_MASSES.end()) {
    throw std::invalid_argument("Unknown element for mass lookup");
  }
  return it->second;
}

unsigned Structure::get_standard_nuclear_charge(Element element) {
  return static_cast<unsigned>(element);
}

void Structure::to_hdf5(H5::Group& group) const {
  try {
    // Add version attribute
    H5::DataSpace scalar_space(H5S_SCALAR);
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr =
        group.createAttribute("version", string_type, scalar_space);
    std::string version_str(SERIALIZATION_VERSION);
    version_attr.write(string_type, version_str);
    version_attr.close();

    // Add units attribute
    H5::Attribute units_attr =
        group.createAttribute("units", string_type, scalar_space);
    std::string units_str("bohr");
    units_attr.write(string_type, units_str);
    units_attr.close();

    // Save coordinates as a 2D array (num_atoms x 3)
    hsize_t coord_dims[2] = {static_cast<hsize_t>(_coordinates.rows()), 3};
    H5::DataSpace coord_space(2, coord_dims);
    H5::DataSet coord_dataset = group.createDataSet(
        "coordinates", H5::PredType::NATIVE_DOUBLE, coord_space);
    coord_dataset.write(_coordinates.data(), H5::PredType::NATIVE_DOUBLE);

    // Save elements as integer array
    std::vector<unsigned> element_ints;
    element_ints.reserve(_elements.size());
    for (const auto& element : _elements) {
      element_ints.push_back(static_cast<unsigned>(element));
    }
    hsize_t elem_dims[1] = {static_cast<hsize_t>(element_ints.size())};
    H5::DataSpace elem_space(1, elem_dims);
    H5::DataSet elem_dataset =
        group.createDataSet("elements", H5::PredType::NATIVE_UINT, elem_space);
    elem_dataset.write(element_ints.data(), H5::PredType::NATIVE_UINT);

    // Save masses as 1D array
    hsize_t mass_dims[1] = {static_cast<hsize_t>(_masses.size())};
    H5::DataSpace mass_space(1, mass_dims);
    H5::DataSet mass_dataset =
        group.createDataSet("masses", H5::PredType::NATIVE_DOUBLE, mass_space);
    mass_dataset.write(_masses.data(), H5::PredType::NATIVE_DOUBLE);

    // Save nuclear charges as 1D array
    hsize_t charge_dims[1] = {static_cast<hsize_t>(_nuclear_charges.size())};
    H5::DataSpace charge_space(1, charge_dims);
    H5::DataSet charge_dataset = group.createDataSet(
        "nuclear_charges", H5::PredType::NATIVE_DOUBLE, charge_space);
    charge_dataset.write(_nuclear_charges.data(), H5::PredType::NATIVE_DOUBLE);

    // Save number of atoms as attribute
    hsize_t num_atoms = static_cast<hsize_t>(_elements.size());
    H5::DataSpace attr_space(H5S_SCALAR);
    H5::Attribute num_atoms_attr = group.createAttribute(
        "num_atoms", H5::PredType::NATIVE_HSIZE, attr_space);
    num_atoms_attr.write(H5::PredType::NATIVE_HSIZE, &num_atoms);

  } catch (const H5::Exception& hdf5_exception) {
    throw std::runtime_error("HDF5 error while writing structure: " +
                             std::string(hdf5_exception.getCDetailMsg()));
  }
}

std::shared_ptr<Structure> Structure::from_hdf5(H5::Group& group) {
  try {
    // Check version first - it must exist
    if (!group.attrExists("version")) {
      throw std::runtime_error(
          "HDF5 group missing required 'version' attribute");
    }
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version;
    version_attr.read(string_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

    // Check units attribute
    if (!group.attrExists("units")) {
      throw std::runtime_error("HDF5 group missing required 'units' attribute");
    }
    H5::Attribute units_attr = group.openAttribute("units");
    std::string units;
    units_attr.read(string_type, units);
    std::transform(units.begin(), units.end(), units.begin(),
                   [](unsigned char c) { return std::tolower(c); });

    // Sanity check units
    if (units != "bohr" && units != "angstrom") {
      throw std::runtime_error(
          "Unsupported in Structure deserialization units: " + units);
    }

    // Read number of atoms attribute
    H5::Attribute num_atoms_attr = group.openAttribute("num_atoms");
    hsize_t num_atoms;
    num_atoms_attr.read(H5::PredType::NATIVE_HSIZE, &num_atoms);

    if (num_atoms == 0) {
      // Return empty structure with explicit empty vectors
      return std::make_shared<Structure>(Eigen::MatrixXd(),
                                         std::vector<Element>(),
                                         Eigen::VectorXd(), Eigen::VectorXd());
    }

    // Read coordinates
    H5::DataSet coord_dataset = group.openDataSet("coordinates");
    H5::DataSpace coord_space = coord_dataset.getSpace();
    hsize_t coord_dims[2];
    coord_space.getSimpleExtentDims(coord_dims);

    if (coord_dims[0] != num_atoms || coord_dims[1] != 3) {
      throw std::runtime_error(
          "Invalid coordinate dimensions in HDF5 structure data");
    }

    Eigen::MatrixXd coordinates(coord_dims[0], 3);
    coord_dataset.read(coordinates.data(), H5::PredType::NATIVE_DOUBLE);

    // Convert coordinates to Bohr if necessary
    if (units == "angstrom") {
      coordinates *= qdk::chemistry::constants::angstrom_to_bohr;
    }

    // Read elements
    H5::DataSet elem_dataset = group.openDataSet("elements");
    std::vector<unsigned> element_ints(num_atoms);
    elem_dataset.read(element_ints.data(), H5::PredType::NATIVE_UINT);

    std::vector<Element> elements;
    elements.reserve(num_atoms);
    for (unsigned elem_int : element_ints) {
      elements.push_back(static_cast<Element>(elem_int));
    }

    // Read masses
    H5::DataSet mass_dataset = group.openDataSet("masses");
    Eigen::VectorXd masses(num_atoms);
    mass_dataset.read(masses.data(), H5::PredType::NATIVE_DOUBLE);

    // Read nuclear charges
    H5::DataSet charge_dataset = group.openDataSet("nuclear_charges");
    Eigen::VectorXd nuclear_charges(num_atoms);
    charge_dataset.read(nuclear_charges.data(), H5::PredType::NATIVE_DOUBLE);

    return std::make_shared<Structure>(coordinates, elements, masses,
                                       nuclear_charges);

  } catch (const H5::Exception& hdf5_exception) {
    throw std::runtime_error("HDF5 error while reading structure: " +
                             std::string(hdf5_exception.getCDetailMsg()));
  }
}

void Structure::to_hdf5_file(const std::string& filename) const {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_write_suffix(filename, "structure");

  _to_hdf5_file(validated_filename);
}

std::shared_ptr<Structure> Structure::from_hdf5_file(
    const std::string& filename) {
  // Validate filename has correct data type suffix
  std::string validated_filename =
      DataTypeFilename::validate_read_suffix(filename, "structure");

  return _from_hdf5_file(validated_filename);
}

void Structure::_to_hdf5_file(const std::string& filename) const {
  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group structure_group = file.createGroup("/structure");
    to_hdf5(structure_group);
  } catch (const H5::Exception& hdf5_exception) {
    throw std::runtime_error("HDF5 error: " +
                             std::string(hdf5_exception.getCDetailMsg()));
  }
}

std::shared_ptr<Structure> Structure::_from_hdf5_file(
    const std::string& filename) {
  H5::H5File file;
  try {
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open Structure HDF5 file '" + filename +
                             "'. " +
                             "Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    // Check if structure group exists
    bool group_exists = false;
    try {
      H5::Group structure_group = file.openGroup("/structure");
      group_exists = true;
    } catch (const H5::Exception&) {
      group_exists = false;
    }

    if (!group_exists) {
      throw std::runtime_error("Structure group not found in HDF5 file");
    }

    H5::Group structure_group = file.openGroup("/structure");
    return from_hdf5(structure_group);
  } catch (const H5::Exception& hdf5_exception) {
    throw std::runtime_error(
        "Unable to read Structure data from HDF5 file '" + filename + "'. " +
        "HDF5 error: " + std::string(hdf5_exception.getCDetailMsg()));
  }
}

void Structure::_validate_dimensions() const {
  if (!_elements.empty()) {
    size_t num_atoms = _elements.size();
    if (_coordinates.rows() != static_cast<int>(num_atoms) ||
        _coordinates.cols() != 3 ||
        _masses.size() != static_cast<int>(num_atoms) ||
        _nuclear_charges.size() != static_cast<int>(num_atoms)) {
      throw std::invalid_argument(
          "All data arrays must have consistent dimensions");
    }
  } else {
    // Empty structure should have all arrays empty
    if (_coordinates.rows() != 0 || _coordinates.cols() != 0 ||
        _masses.size() != 0 || _nuclear_charges.size() != 0) {
      throw std::invalid_argument("Empty structure must have all arrays empty");
    }
  }
}

}  // namespace qdk::chemistry::data

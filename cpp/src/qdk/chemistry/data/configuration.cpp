// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <fstream>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <sstream>

namespace qdk::chemistry::data {

// Constructor from string representation (e.g., "22uudd000")
Configuration::Configuration(const std::string& str) {
  size_t num_orbitals = str.size();
  _packed_orbs.resize((num_orbitals + 3) / 4,
                      0);  // Each byte stores 4 orbitals

  for (size_t i = 0; i < num_orbitals; ++i) {
    OccupationState value;
    switch (str[i]) {
      case '0':
        value = UNOCCUPIED;
        break;
      case 'u':
        value = ALPHA;
        break;
      case 'd':
        value = BETA;
        break;
      case '2':
        value = DOUBLY;
        break;
      default:
        throw std::invalid_argument(
            "Invalid character in configuration string");
    }
    _set_orbital(i, value);
  }
}

// Convert back to string representation
std::string Configuration::to_string() const {
  size_t num_orbitals = get_orbital_capacity();
  std::string result(num_orbitals, '0');
  for (size_t i = 0; i < num_orbitals; ++i) {
    OccupationState state = _get_orbital(i);
    switch (state) {
      case UNOCCUPIED:
        result[i] = '0';
        break;
      case ALPHA:
        result[i] = 'u';
        break;
      case BETA:
        result[i] = 'd';
        break;
      case DOUBLY:
        result[i] = '2';
        break;
    }
  }
  return result;
}

// Get number of electrons
std::tuple<size_t, size_t> Configuration::get_n_electrons() const {
  size_t num_orbitals = get_orbital_capacity();
  size_t n_alpha = 0, n_beta = 0;

  for (size_t i = 0; i < num_orbitals; ++i) {
    OccupationState state = _get_orbital(i);
    if (state == ALPHA || state == DOUBLY) ++n_alpha;
    if (state == BETA || state == DOUBLY) ++n_beta;
  }
  return {n_alpha, n_beta};
}

// Check if orbital has alpha electron
bool Configuration::has_alpha_electron(size_t orbital_idx) const {
  size_t num_orbitals = get_orbital_capacity();
  if (orbital_idx >= num_orbitals) {
    return false;
  }
  OccupationState state = _get_orbital(orbital_idx);
  return (state == ALPHA || state == DOUBLY);
}

// Check if orbital has beta electron
bool Configuration::has_beta_electron(size_t orbital_idx) const {
  size_t num_orbitals = get_orbital_capacity();
  if (orbital_idx >= num_orbitals) {
    return false;
  }
  OccupationState state = _get_orbital(orbital_idx);
  return (state == BETA || state == DOUBLY);
}

// Equality operator for std::find and other algorithms
bool Configuration::operator==(const Configuration& other) const {
  // Check if they have the same storage size (and thus capacity)
  if (_packed_orbs.size() != other._packed_orbs.size()) {
    return false;
  }

  // Compare the orbital states for each orbital
  size_t num_orbitals = get_orbital_capacity();
  for (size_t i = 0; i < num_orbitals; ++i) {
    if (_get_orbital(i) != other._get_orbital(i)) {
      return false;
    }
  }

  return true;
}

// Capacity (number of orbitals)
size_t Configuration::get_orbital_capacity() const {
  return _packed_orbs.size() * 4;  // Each byte stores 4 orbitals
}

// Inequality operator (for completeness)
bool Configuration::operator!=(const Configuration& other) const {
  return !(*this == other);
}

// Get the occupation state of an orbital
Configuration::OccupationState Configuration::_get_orbital(size_t pos) const {
  size_t byte_pos = pos / 4;
  size_t bit_offset = (pos % 4) * 2;  // Each orbital uses 2 bits
  return static_cast<OccupationState>((_packed_orbs[byte_pos] >> bit_offset) &
                                      0x3);
}

// Set the occupation state of an orbital
void Configuration::_set_orbital(size_t pos, OccupationState value) {
  size_t byte_pos = pos / 4;
  size_t bit_offset = (pos % 4) * 2;

  // Clear the 2 bits at position
  _packed_orbs[byte_pos] &= ~(0x3 << bit_offset);

  // Set the new value
  _packed_orbs[byte_pos] |= (value << bit_offset);
}

// Create a canonical Hartree-Fock configuration using the Aufbau principle
Configuration Configuration::canonical_hf_configuration(size_t n_alpha,
                                                        size_t n_beta,
                                                        size_t n_orbitals) {
  std::string config_str;
  config_str.reserve(n_orbitals);

  // Fill orbitals from lowest energy
  for (size_t i = 0; i < n_orbitals; ++i) {
    if (i < std::min(n_alpha, n_beta)) {
      // Doubly occupied orbital
      config_str += "2";
    } else if (i < std::max(n_alpha, n_beta)) {
      // Singly occupied orbital (alpha or beta depending on which has more
      // electrons)
      if (n_alpha > n_beta) {
        config_str += "u";  // alpha-occupied
      } else {
        config_str += "d";  // beta-occupied
      }
    } else {
      // Unoccupied orbital
      config_str += "0";
    }
  }

  return Configuration(config_str);
}

nlohmann::json Configuration::to_json() const {
  nlohmann::json j;
  // Store as string representation for human readability
  j["configuration"] = to_string();
  return j;
}

Configuration Configuration::from_json(const nlohmann::json& j) {
  if (!j.contains("configuration")) {
    throw std::runtime_error("JSON missing required 'configuration' field");
  }
  std::string config_str = j["configuration"];
  return Configuration(config_str);
}

void Configuration::to_hdf5(H5::Group& group) const {
  try {
    hsize_t packed_size = _packed_orbs.size();
    H5::DataSpace dataspace(1, &packed_size);

    H5::DataSet dataset = group.createDataSet(
        "configuration", H5::PredType::NATIVE_UINT8, dataspace);
    dataset.write(_packed_orbs.data(), H5::PredType::NATIVE_UINT8);

    // Store orbital capacity as attribute for proper reconstruction
    H5::Attribute orb_attr =
        dataset.createAttribute("orbital_capacity", H5::PredType::NATIVE_HSIZE,
                                H5::DataSpace(H5S_SCALAR));
    hsize_t capacity = get_orbital_capacity();
    orb_attr.write(H5::PredType::NATIVE_HSIZE, &capacity);

    dataset.close();
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in Configuration::to_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

Configuration Configuration::from_hdf5(H5::Group& group) {
  try {
    H5::DataSet dataset = group.openDataSet("configuration");

    // Read the orbital capacity from attribute
    H5::Attribute orb_attr = dataset.openAttribute("orbital_capacity");
    hsize_t orbital_capacity;
    orb_attr.read(H5::PredType::NATIVE_HSIZE, &orbital_capacity);

    // Read the packed binary data
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t packed_size;
    dataspace.getSimpleExtentDims(&packed_size);

    std::vector<uint8_t> packed_data(packed_size);
    dataset.read(packed_data.data(), H5::PredType::NATIVE_UINT8);
    dataset.close();

    // Create new Configuration and set its packed data
    Configuration result;
    result._packed_orbs = std::move(packed_data);
    return result;
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in Configuration::from_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

std::string Configuration::get_summary() const {
  auto [n_alpha, n_beta] = get_n_electrons();
  std::ostringstream oss;
  oss << "Configuration Summary:\n";
  oss << "  Representation: " << to_string() << "\n";
  oss << "  Alpha electrons: " << n_alpha << "\n";
  oss << "  Beta electrons: " << n_beta << "\n";
  oss << "  Total electrons: " << (n_alpha + n_beta) << "\n";
  oss << "  Orbital capacity: " << get_orbital_capacity() << "\n";
  return oss.str();
}

void Configuration::to_file(const std::string& filename,
                            const std::string& type) const {
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

void Configuration::to_json_file(const std::string& filename) const {
  std::ofstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for writing: " + filename);
  }

  auto json_obj = to_json();
  file << json_obj.dump(2);

  if (file.fail()) {
    throw std::runtime_error("Error writing to file: " + filename);
  }
}

void Configuration::to_hdf5_file(const std::string& filename) const {
  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root_group = file.openGroup("/");
    to_hdf5(root_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

Configuration Configuration::from_file(const std::string& filename,
                                       const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5") {
    return from_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unknown file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

Configuration Configuration::from_json_file(const std::string& filename) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error("Cannot open file for reading: " + filename);
  }

  nlohmann::json json_obj;
  file >> json_obj;

  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return from_json(json_obj);
}

Configuration Configuration::from_hdf5_file(const std::string& filename) {
  try {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group root_group = file.openGroup("/");
    return from_hdf5(root_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}
}  // namespace qdk::chemistry::data

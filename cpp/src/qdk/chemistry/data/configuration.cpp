// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <fstream>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>

namespace qdk::chemistry::data {

// ---- Internal helpers ------------------------------------------------------

size_t Configuration::_packed_bytes(size_t n_modes, uint8_t bpm) {
  if (bpm == 0 || (8 % bpm) != 0) {
    throw std::invalid_argument(
        "bits_per_mode (" + std::to_string(bpm) +
        ") must be a positive divisor of 8 (1, 2, 4, or 8)");
  }
  size_t modes_per_byte = 8 / bpm;
  return (n_modes + modes_per_byte - 1) / modes_per_byte;
}

void Configuration::_require_spin_half(const char* caller) const {
  if (_bits_per_mode != 2) {
    throw std::runtime_error(
        std::string(caller) +
        " requires a spin-½ configuration (2 bits/mode), but this "
        "configuration has " +
        std::to_string(_bits_per_mode) + " bit(s) per mode.");
  }
}

uint8_t Configuration::_get_mode_raw(size_t pos) const {
  size_t modes_per_byte = 8 / _bits_per_mode;
  size_t byte_idx = pos / modes_per_byte;
  size_t bit_offset = (pos % modes_per_byte) * _bits_per_mode;
  uint8_t mask = (1 << _bits_per_mode) - 1;
  return (_packed_orbs[byte_idx] >> bit_offset) & mask;
}

void Configuration::_set_mode_raw(size_t pos, uint8_t value) {
  size_t modes_per_byte = 8 / _bits_per_mode;
  size_t byte_idx = pos / modes_per_byte;
  size_t bit_offset = (pos % modes_per_byte) * _bits_per_mode;
  uint8_t mask = (1 << _bits_per_mode) - 1;
  _packed_orbs[byte_idx] &= ~(mask << bit_offset);
  _packed_orbs[byte_idx] |= (value & mask) << bit_offset;
}

Configuration::OccupationState Configuration::_get_orbital(size_t pos) const {
  return static_cast<OccupationState>(_get_mode_raw(pos));
}

void Configuration::_set_orbital(size_t pos, OccupationState value) {
  _set_mode_raw(pos, static_cast<uint8_t>(value));
}

// ---- Constructors ----------------------------------------------------------

Configuration::Configuration(const std::string& str)
    : Configuration(from_spin_half_string(str)) {}

// ---- Named factories -------------------------------------------------------

Configuration Configuration::from_spin_half_string(const std::string& str) {
  Configuration result;
  result._bits_per_mode = 2;
  result._packed_orbs.resize(_packed_bytes(str.size(), 2), 0);

  for (size_t i = 0; i < str.size(); ++i) {
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
            "Invalid character '" + std::string(1, str[i]) +
            "' in spin-½ configuration string (expected '0','u','d','2')");
    }
    result._set_orbital(i, value);
  }
  return result;
}

Configuration Configuration::from_bitstring(const std::string& str) {
  Configuration result;
  result._bits_per_mode = 1;
  result._packed_orbs.resize(_packed_bytes(str.size(), 1), 0);

  for (size_t i = 0; i < str.size(); ++i) {
    if (str[i] == '1') {
      result._set_mode_raw(i, 1);
    } else if (str[i] != '0') {
      throw std::invalid_argument(
          "Invalid character '" + std::string(1, str[i]) +
          "' in bitstring configuration (expected '0' or '1')");
    }
  }
  return result;
}

// Convert back to string representation
std::string Configuration::to_string() const {
  QDK_LOG_TRACE_ENTERING();

  std::string result(capacity(), '0');
  if (_bits_per_mode == 1) {
    for (size_t i = 0; i < capacity(); ++i) {
      if (_get_mode_raw(i)) result[i] = '1';
    }
  } else {
    for (size_t i = 0; i < capacity(); ++i) {
      OccupationState state = _get_orbital(i);
      switch (state) {
        case UNOCCUPIED:
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
  }
  return result;
}

// ---- Generic accessors -----------------------------------------------------

size_t Configuration::capacity() const {
  return _packed_orbs.size() * (8 / _bits_per_mode);
}

uint8_t Configuration::bits_per_mode() const { return _bits_per_mode; }

uint8_t Configuration::get_mode_state(size_t idx) const {
  if (idx >= capacity()) {
    throw std::out_of_range(
        "Mode index " + std::to_string(idx) +
        " out of range (capacity=" + std::to_string(capacity()) + ")");
  }
  return _get_mode_raw(idx);
}

size_t Configuration::total_occupation() const {
  size_t total = 0;
  for (size_t i = 0; i < capacity(); ++i) {
    uint8_t state = _get_mode_raw(i);
    if (_bits_per_mode == 2) {
      // Popcount of 2-bit state: 0→0, 1→1, 2→1, 3→2
      total += (state & 1) + ((state >> 1) & 1);
    } else {
      total += state;
    }
  }
  return total;
}

const std::vector<uint8_t>& Configuration::packed_data() const {
  return _packed_orbs;
}

// ---- Spin-½ accessors (gated) ----------------------------------------------

std::tuple<size_t, size_t> Configuration::get_n_electrons() const {
  QDK_LOG_TRACE_ENTERING();
  _require_spin_half("get_n_electrons");

  size_t n_alpha = 0, n_beta = 0;
  for (size_t i = 0; i < capacity(); ++i) {
    OccupationState state = _get_orbital(i);
    if (state == ALPHA || state == DOUBLY) ++n_alpha;
    if (state == BETA || state == DOUBLY) ++n_beta;
  }
  return {n_alpha, n_beta};
}

bool Configuration::has_alpha_electron(size_t orbital_idx) const {
  QDK_LOG_TRACE_ENTERING();
  _require_spin_half("has_alpha_electron");

  if (orbital_idx >= capacity()) return false;
  OccupationState state = _get_orbital(orbital_idx);
  return (state == ALPHA || state == DOUBLY);
}

bool Configuration::has_beta_electron(size_t orbital_idx) const {
  QDK_LOG_TRACE_ENTERING();
  _require_spin_half("has_beta_electron");

  if (orbital_idx >= capacity()) return false;
  OccupationState state = _get_orbital(orbital_idx);
  return (state == BETA || state == DOUBLY);
}

bool Configuration::is_closed_shell() const {
  QDK_LOG_TRACE_ENTERING();
  _require_spin_half("is_closed_shell");
  for (size_t i = 0; i < capacity(); ++i) {
    const auto state = _get_orbital(i);
    if (state == ALPHA || state == BETA) return false;
  }
  return true;
}

// Equality operator for std::find and other algorithms
bool Configuration::operator==(const Configuration& other) const {
  QDK_LOG_TRACE_ENTERING();

  if (_bits_per_mode != other._bits_per_mode) return false;
  if (capacity() != other.capacity()) return false;

  for (size_t i = 0; i < capacity(); ++i) {
    if (_get_mode_raw(i) != other._get_mode_raw(i)) return false;
  }
  return true;
}

size_t Configuration::get_orbital_capacity() const {
  QDK_LOG_TRACE_ENTERING();
  return capacity();
}

// Inequality operator (for completeness)
bool Configuration::operator!=(const Configuration& other) const {
  return !(*this == other);
}

// Create a canonical Hartree-Fock configuration using the Aufbau principle
Configuration Configuration::canonical_hf_configuration(size_t n_alpha,
                                                        size_t n_beta,
                                                        size_t n_orbitals) {
  QDK_LOG_TRACE_ENTERING();

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

  return Configuration::from_spin_half_string(config_str);
}

nlohmann::json Configuration::to_json() const {
  QDK_LOG_TRACE_ENTERING();

  nlohmann::json j;
  j["configuration"] = to_string();
  j["bits_per_mode"] = _bits_per_mode;
  return j;
}

Configuration Configuration::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();

  if (!j.contains("configuration")) {
    throw std::runtime_error("JSON missing required 'configuration' field");
  }
  std::string config_str = j["configuration"];
  uint8_t bpm = j.value("bits_per_mode", static_cast<uint8_t>(2));
  if (bpm == 1) return Configuration::from_bitstring(config_str);
  return Configuration::from_spin_half_string(config_str);
}

void Configuration::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();

  try {
    hsize_t packed_size = _packed_orbs.size();
    H5::DataSpace dataspace(1, &packed_size);

    H5::DataSet dataset = group.createDataSet(
        "configuration", H5::PredType::NATIVE_UINT8, dataspace);
    dataset.write(_packed_orbs.data(), H5::PredType::NATIVE_UINT8);

    H5::Attribute orb_attr =
        dataset.createAttribute("orbital_capacity", H5::PredType::NATIVE_HSIZE,
                                H5::DataSpace(H5S_SCALAR));
    hsize_t capacity = this->capacity();
    orb_attr.write(H5::PredType::NATIVE_HSIZE, &capacity);

    H5::Attribute bpm_attr = dataset.createAttribute(
        "bits_per_mode", H5::PredType::NATIVE_UINT8, H5::DataSpace(H5S_SCALAR));
    bpm_attr.write(H5::PredType::NATIVE_UINT8, &_bits_per_mode);

    dataset.close();
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in Configuration::to_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

Configuration Configuration::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();

  try {
    H5::DataSet dataset = group.openDataSet("configuration");

    H5::Attribute orb_attr = dataset.openAttribute("orbital_capacity");
    hsize_t orbital_capacity;
    orb_attr.read(H5::PredType::NATIVE_HSIZE, &orbital_capacity);

    // Legacy files lack bits_per_mode — default to 2 (spin-½).
    uint8_t bpm = 2;
    if (dataset.attrExists("bits_per_mode")) {
      H5::Attribute bpm_attr = dataset.openAttribute("bits_per_mode");
      bpm_attr.read(H5::PredType::NATIVE_UINT8, &bpm);
    }

    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t packed_size;
    dataspace.getSimpleExtentDims(&packed_size);

    std::vector<uint8_t> packed_data(packed_size);
    dataset.read(packed_data.data(), H5::PredType::NATIVE_UINT8);
    dataset.close();

    Configuration result;
    result._packed_orbs = std::move(packed_data);
    result._bits_per_mode = bpm;
    return result;
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in Configuration::from_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

std::string Configuration::get_summary() const {
  QDK_LOG_TRACE_ENTERING();

  std::ostringstream oss;
  oss << "Configuration Summary:\n";
  oss << "  Representation: " << to_string() << "\n";
  oss << "  Bits per mode: " << static_cast<int>(_bits_per_mode) << "\n";
  oss << "  Modes: " << capacity() << "\n";
  oss << "  Total occupation: " << total_occupation() << "\n";
  if (_bits_per_mode == 2) {
    auto [n_alpha, n_beta] = get_n_electrons();
    oss << "  Alpha electrons: " << n_alpha << "\n";
    oss << "  Beta electrons: " << n_beta << "\n";
  }
  return oss.str();
}

void Configuration::to_file(const std::string& filename,
                            const std::string& type) const {
  QDK_LOG_TRACE_ENTERING();

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
  QDK_LOG_TRACE_ENTERING();

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
  QDK_LOG_TRACE_ENTERING();

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
  QDK_LOG_TRACE_ENTERING();

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
  QDK_LOG_TRACE_ENTERING();

  std::ifstream file(filename);
  if (!file.is_open()) {
    throw std::runtime_error(
        "Unable to open Configuration JSON file '" + filename +
        "'. Please check that the file exists and you have read permissions.");
  }

  nlohmann::json json_obj;
  file >> json_obj;

  if (file.fail()) {
    throw std::runtime_error("Error reading from file: " + filename);
  }

  return from_json(json_obj);
}

Configuration Configuration::from_hdf5_file(const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  H5::H5File file;
  try {
    file.openFile(filename, H5F_ACC_RDONLY);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("Unable to open Configuration HDF5 file '" +
                             filename + "'. " +
                             "Please check that the file exists, is a valid "
                             "HDF5 file, and you have read permissions.");
  }

  try {
    H5::Group root_group = file.openGroup("/");
    return from_hdf5(root_group);
  } catch (const H5::Exception& e) {
    throw std::runtime_error(
        "Unable to read Configuration data from HDF5 file '" + filename +
        "'. " + "HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::pair<std::string, std::string> Configuration::to_binary_strings(
    size_t num_orbitals) const {
  _require_spin_half("to_binary_strings");
  size_t cap = capacity();

  // Throw if we ask for too many orbitals
  if (num_orbitals > cap) {
    throw std::runtime_error(
        "num_orbitals argument cannot be greater than the number of orbitals "
        "in the system.");
  }

  std::string result_alpha(num_orbitals, '0');
  std::string result_beta(num_orbitals, '0');
  for (size_t i = 0; i < num_orbitals; ++i) {
    OccupationState state = _get_orbital(i);
    switch (state) {
      case UNOCCUPIED:
        break;
      case ALPHA:
        result_alpha[i] = '1';
        break;
      case BETA:
        result_beta[i] = '1';
        break;
      case DOUBLY:
        result_alpha[i] = '1';
        result_beta[i] = '1';
        break;
    }
  }
  return {result_alpha, result_beta};
}

Configuration Configuration::from_binary_strings(std::string alpha_string,
                                                 std::string beta_string) {
  size_t n_orbitals = alpha_string.size();
  size_t n_orbitals_beta = beta_string.size();
  if (n_orbitals != n_orbitals_beta) {
    throw std::runtime_error(
        "Should have the same-length string repr for alpha and beta");
  }
  char zero_char = '0';
  std::string orbital_rep(n_orbitals, zero_char);

  for (size_t i = 0; i < n_orbitals; ++i) {
    char alpha_contents = alpha_string[i];
    char beta_contents = beta_string[i];

    if (alpha_contents != '0' && alpha_contents != '1') {
      throw std::runtime_error("alpha string should contain only 0/1");
    } else if (beta_contents != '0' && beta_contents != '1') {
      throw std::runtime_error("beta string should contain only 0/1");
    } else if (alpha_contents == '1' && beta_contents == '1') {
      orbital_rep[i] = '2';
    } else if (alpha_contents == '1' && beta_contents == '0') {
      orbital_rep[i] = 'u';
    } else if (alpha_contents == '0' && beta_contents == '1') {
      orbital_rep[i] = 'd';
    }
  }
  return Configuration::from_spin_half_string(orbital_rep);
}

void Configuration::hash_update(qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, static_cast<uint64_t>(_bits_per_mode));
  hash_value(ctx, to_string());
}
}  // namespace qdk::chemistry::data

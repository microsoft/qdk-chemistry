// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/configuration_set.hpp>
#include <stdexcept>

namespace qdk::chemistry::data {

ConfigurationSet::ConfigurationSet(
    const std::vector<Configuration>& configurations,
    std::shared_ptr<Orbitals> orbitals)
    : _configurations(configurations), _orbitals(orbitals) {
  if (!_orbitals) {
    throw std::invalid_argument(
        "ConfigurationSet: orbitals pointer cannot be null");
  }
  _validate_configurations();
}

ConfigurationSet::ConfigurationSet(std::vector<Configuration>&& configurations,
                                   std::shared_ptr<Orbitals> orbitals)
    : _configurations(std::move(configurations)), _orbitals(orbitals) {
  if (!_orbitals) {
    throw std::invalid_argument(
        "ConfigurationSet: orbitals pointer cannot be null");
  }
  _validate_configurations();
}

const std::vector<Configuration>& ConfigurationSet::get_configurations() const {
  return _configurations;
}

std::shared_ptr<Orbitals> ConfigurationSet::get_orbitals() const {
  return _orbitals;
}

size_t ConfigurationSet::size() const { return _configurations.size(); }

bool ConfigurationSet::empty() const { return _configurations.empty(); }

const Configuration& ConfigurationSet::operator[](size_t idx) const {
  return _configurations[idx];
}

const Configuration& ConfigurationSet::at(size_t idx) const {
  return _configurations.at(idx);
}

ConfigurationSet::const_iterator ConfigurationSet::begin() const {
  return _configurations.begin();
}

ConfigurationSet::const_iterator ConfigurationSet::end() const {
  return _configurations.end();
}

ConfigurationSet::const_iterator ConfigurationSet::cbegin() const {
  return _configurations.cbegin();
}

ConfigurationSet::const_iterator ConfigurationSet::cend() const {
  return _configurations.cend();
}

bool ConfigurationSet::operator==(const ConfigurationSet& other) const {
  // Check if orbitals point to the same object (not deep equality)
  if (_orbitals != other._orbitals) {
    return false;
  }
  // Check if configurations are equal
  return _configurations == other._configurations;
}

bool ConfigurationSet::operator!=(const ConfigurationSet& other) const {
  return !(*this == other);
}

void ConfigurationSet::_validate_configurations() const {
  if (_configurations.empty()) {
    // Empty set is valid
    return;
  }

  // Validate internal consistency by checking that all configurations
  // in the set have the same number of orbitals.
  // Access _num_orbitals directly through friend class relationship.

  size_t first_config_orbitals = _configurations[0].get_orbital_capacity();

  // Check that all configurations have the same electron count
  auto [first_n_alpha, first_n_beta] = _configurations[0].get_n_electrons();
  size_t first_total_electrons = first_n_alpha + first_n_beta;

  for (size_t i = 1; i < _configurations.size(); ++i) {
    size_t config_num_orbitals = _configurations[i].get_orbital_capacity();

    // Check consistency within the set - all configs should have same size
    if (config_num_orbitals != first_config_orbitals) {
      throw std::invalid_argument(
          "ConfigurationSet: configuration at index " + std::to_string(i) +
          " has " + std::to_string(config_num_orbitals) +
          " orbitals, but configuration 0 has " +
          std::to_string(first_config_orbitals) + " orbitals. " +
          "All configurations in a set must have the same number of orbitals.");
    }

    // Check that all configurations have the same electron count
    auto [n_alpha, n_beta] = _configurations[i].get_n_electrons();
    size_t total_electrons = n_alpha + n_beta;
    if (total_electrons != first_total_electrons) {
      throw std::invalid_argument(
          "ConfigurationSet: configuration at index " + std::to_string(i) +
          " has " + std::to_string(total_electrons) +
          " electrons (α=" + std::to_string(n_alpha) +
          ", β=" + std::to_string(n_beta) + "), but configuration 0 has " +
          std::to_string(first_total_electrons) +
          " electrons (α=" + std::to_string(first_n_alpha) +
          ", β=" + std::to_string(first_n_beta) +
          "). All configurations in a set must have the same number of "
          "electrons.");
    }
  }

  // If orbitals are provided with active space, validate configurations against
  // active space requirements Note: Configurations only represent the active
  // space, not the full orbital space (inactive and virtual orbitals are not
  // included in the configuration representation)
  if (_orbitals && _orbitals->has_active_space()) {
    auto [alpha_active, beta_active] = _orbitals->get_active_space_indices();

    // For restricted calculations, use alpha indices (they should be the same)
    const auto& active_indices = alpha_active;

    // Validate that configuration has sufficient orbital capacity for the
    // active space
    if (!active_indices.empty()) {
      size_t active_space_size = active_indices.size();

      for (size_t i = 0; i < _configurations.size(); ++i) {
        const auto& config = _configurations[i];
        const std::string config_str = config.to_string();

        // The configuration must have at least as many orbitals as the active
        // space size
        if (config.get_orbital_capacity() < active_space_size) {
          throw std::invalid_argument(
              "ConfigurationSet: configuration at index " + std::to_string(i) +
              " has orbital capacity " +
              std::to_string(config.get_orbital_capacity()) +
              " which is insufficient for active space (requires at least " +
              std::to_string(active_space_size) + " orbitals).");
        }

        // Validate that any orbitals beyond the active space size are
        // unoccupied (no "overhanging" electrons)
        for (size_t orbital_idx = active_space_size;
             orbital_idx < config.get_orbital_capacity(); ++orbital_idx) {
          if (orbital_idx < config_str.length() &&
              config_str[orbital_idx] != '0') {
            throw std::invalid_argument(
                "ConfigurationSet: configuration at index " +
                std::to_string(i) + " has occupied orbital at index " +
                std::to_string(orbital_idx) +
                " which is beyond the active space size (" +
                std::to_string(active_space_size) +
                "). Only orbitals within the active space can be occupied.");
          }
        }
      }
    }
  }

  // Note: We don't strictly validate against
  // orbitals->get_num_molecular_orbitals() because:
  // 1. Configurations may represent a subset or superset of the MO space
  // 2. Legacy code and tests may have different conventions
  // 3. The ConfigurationSet provides context, but configurations retain their
  //    size information internally
  // The main benefit is avoiding redundant storage of Orbital information,
  // not enforcing strict size matching.
}

nlohmann::json ConfigurationSet::to_json() const {
  nlohmann::json j;

  // Store orbitals
  j["orbitals"] = _orbitals->to_json();

  // Store configurations array
  j["configurations"] = nlohmann::json::array();
  for (const auto& config : _configurations) {
    j["configurations"].push_back(config.to_json());
  }

  return j;
}

ConfigurationSet ConfigurationSet::from_json(const nlohmann::json& j) {
  // Deserialize orbitals from JSON
  std::shared_ptr<Orbitals> orbs = nullptr;
  if (j.contains("orbitals")) {
    orbs = Orbitals::from_json(j["orbitals"]);
  }

  std::vector<Configuration> configurations;
  if (!j.contains("configurations")) {
    throw std::runtime_error("JSON missing required 'configurations' field");
  }

  const auto& config_array = j["configurations"];
  configurations.reserve(config_array.size());  // Pre-allocate

  for (const auto& config_json : config_array) {
    Configuration config = Configuration::from_json(config_json);
    configurations.push_back(std::move(config));
  }

  return ConfigurationSet(std::move(configurations), orbs);
}

void ConfigurationSet::to_hdf5(H5::Group& group) const {
  try {
    // Store orbitals first
    H5::Group orbitals_group = group.createGroup("orbitals");
    _orbitals->to_hdf5(orbitals_group);
    orbitals_group.close();

    if (_configurations.empty()) {
      // Store empty flag
      H5::Attribute empty_attr = group.createAttribute(
          "is_empty", H5::PredType::NATIVE_HBOOL, H5::DataSpace(H5S_SCALAR));
      hbool_t empty_flag = 1;
      empty_attr.write(H5::PredType::NATIVE_HBOOL, &empty_flag);
      return;
    }

    // Store as a 2D matrix of uint8_t for efficient I/O
    // Each row is the packed binary representation of one configuration
    size_t num_configs = _configurations.size();
    size_t packed_size = _configurations[0]._packed_orbs.size();

    // Verify all configurations have the same packed size
    for (const auto& config : _configurations) {
      if (config._packed_orbs.size() != packed_size) {
        throw std::runtime_error(
            "Inconsistent configuration sizes in ConfigurationSet");
      }
    }

    // Create 2D dataset
    hsize_t dims[2] = {static_cast<hsize_t>(num_configs),
                       static_cast<hsize_t>(packed_size)};
    H5::DataSpace dataspace(2, dims);
    H5::DataSet dataset = group.createDataSet(
        "configurations", H5::PredType::NATIVE_UINT8, dataspace);

    // Flatten the data for writing
    std::vector<uint8_t> flat_data;
    flat_data.reserve(num_configs * packed_size);
    for (const auto& config : _configurations) {
      flat_data.insert(flat_data.end(), config._packed_orbs.begin(),
                       config._packed_orbs.end());
    }

    dataset.write(flat_data.data(), H5::PredType::NATIVE_UINT8);

    // Store metadata as attributes
    H5::Attribute count_attr = dataset.createAttribute(
        "num_configurations", H5::PredType::NATIVE_HSIZE,
        H5::DataSpace(H5S_SCALAR));
    hsize_t num_configs_val = num_configs;
    count_attr.write(H5::PredType::NATIVE_HSIZE, &num_configs_val);

    H5::Attribute packed_attr = dataset.createAttribute(
        "packed_size", H5::PredType::NATIVE_HSIZE, H5::DataSpace(H5S_SCALAR));
    hsize_t packed_size_val = packed_size;
    packed_attr.write(H5::PredType::NATIVE_HSIZE, &packed_size_val);

    // Store orbital capacity (number of orbitals each configuration can
    // represent)
    hsize_t orbital_capacity = _configurations[0].get_orbital_capacity();
    H5::Attribute orb_attr =
        dataset.createAttribute("orbital_capacity", H5::PredType::NATIVE_HSIZE,
                                H5::DataSpace(H5S_SCALAR));
    orb_attr.write(H5::PredType::NATIVE_HSIZE, &orbital_capacity);

    dataset.close();
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in ConfigurationSet::to_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

ConfigurationSet ConfigurationSet::from_hdf5(H5::Group& group) {
  try {
    // Deserialize orbitals from HDF5
    std::shared_ptr<Orbitals> orbs = nullptr;
    if (group.nameExists("orbitals")) {
      H5::Group orbitals_group = group.openGroup("orbitals");
      orbs = Orbitals::from_hdf5(orbitals_group);
      orbitals_group.close();
    }

    // Check for empty set
    if (group.attrExists("is_empty")) {
      H5::Attribute empty_attr = group.openAttribute("is_empty");
      hbool_t empty_flag;
      empty_attr.read(H5::PredType::NATIVE_HBOOL, &empty_flag);
      if (empty_flag) {
        return ConfigurationSet(std::vector<Configuration>(), orbs);
      }
    }

    // Open the configurations dataset
    H5::DataSet dataset = group.openDataSet("configurations");

    // Read metadata from attributes
    H5::Attribute count_attr = dataset.openAttribute("num_configurations");
    hsize_t num_configs;
    count_attr.read(H5::PredType::NATIVE_HSIZE, &num_configs);

    H5::Attribute packed_attr = dataset.openAttribute("packed_size");
    hsize_t packed_size;
    packed_attr.read(H5::PredType::NATIVE_HSIZE, &packed_size);

    // Read the 2D dataset
    H5::DataSpace dataspace = dataset.getSpace();
    hsize_t dims[2];
    dataspace.getSimpleExtentDims(dims);

    if (dims[0] != num_configs || dims[1] != packed_size) {
      throw std::runtime_error(
          "Dimension mismatch in ConfigurationSet HDF5 data");
    }

    // Pre-allocate flat_data vector before reading
    std::vector<uint8_t> flat_data;
    flat_data.resize(num_configs * packed_size);
    dataset.read(flat_data.data(), H5::PredType::NATIVE_UINT8);
    dataset.close();

    // Reconstruct configurations from flat data
    std::vector<Configuration> configurations;
    configurations.reserve(num_configs);  // Pre-allocate
    for (size_t i = 0; i < num_configs; ++i) {
      Configuration config;
      // Access _packed_orbs through friend relationship
      config._packed_orbs.resize(packed_size);
      config._packed_orbs.assign(flat_data.begin() + i * packed_size,
                                 flat_data.begin() + (i + 1) * packed_size);
      configurations.push_back(std::move(config));
    }

    return ConfigurationSet(std::move(configurations), orbs);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error in ConfigurationSet::from_hdf5: " +
                             std::string(e.getCDetailMsg()));
  }
}

std::string ConfigurationSet::get_summary() const {
  std::string summary = "ConfigurationSet:\n";
  summary +=
      "  Number of configurations: " + std::to_string(_configurations.size()) +
      "\n";

  if (!_configurations.empty()) {
    summary += "  Orbital capacity per configuration: " +
               std::to_string(_configurations[0].get_orbital_capacity()) + "\n";
  }

  if (_orbitals) {
    summary += "  Associated orbitals: Present\n";
  } else {
    summary += "  Associated orbitals: None\n";
  }

  return summary;
}

void ConfigurationSet::to_file(const std::string& filename,
                               const std::string& type) const {
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5" || type == "h5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5, h5");
  }
}

void ConfigurationSet::to_json_file(const std::string& filename) const {
  try {
    std::ofstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    nlohmann::json j = to_json();
    file << j.dump(2);

    if (file.fail()) {
      throw std::runtime_error("Failed to write JSON data to file: " +
                               filename);
    }

    file.close();
  } catch (const std::exception& e) {
    throw std::runtime_error("Error writing JSON file '" + filename +
                             "': " + e.what());
  }
}

void ConfigurationSet::to_hdf5_file(const std::string& filename) const {
  try {
    H5::H5File file(filename, H5F_ACC_TRUNC);
    H5::Group root_group = file.openGroup("/");
    to_hdf5(root_group);
    root_group.close();
    file.close();
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error writing file '" + filename +
                             "': " + std::string(e.getCDetailMsg()));
  }
}

ConfigurationSet ConfigurationSet::from_file(const std::string& filename,
                                             const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5" || type == "h5") {
    return from_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5, h5");
  }
}

ConfigurationSet ConfigurationSet::from_json_file(const std::string& filename) {
  try {
    std::ifstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    nlohmann::json j;
    file >> j;
    file.close();

    return from_json(j);
  } catch (const std::exception& e) {
    throw std::runtime_error("Error reading JSON file '" + filename +
                             "': " + e.what());
  }
}

ConfigurationSet ConfigurationSet::from_hdf5_file(const std::string& filename) {
  try {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group root_group = file.openGroup("/");
    ConfigurationSet result = from_hdf5(root_group);
    root_group.close();
    file.close();
    return result;
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error reading file '" + filename +
                             "': " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace qdk::chemistry::data

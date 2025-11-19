// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <fstream>
#include <qdk/chemistry/data/coupled_cluster.hpp>
#include <stdexcept>

namespace qdk::chemistry::data {

CoupledClusterAmplitudes::CoupledClusterAmplitudes() = default;

CoupledClusterAmplitudes::~CoupledClusterAmplitudes() noexcept = default;

CoupledClusterAmplitudes::CoupledClusterAmplitudes(
    std::shared_ptr<Orbitals> orbitals, const amplitude_type& t1_amplitudes,
    const amplitude_type& t2_amplitudes, unsigned int n_alpha_electrons,
    unsigned int n_beta_electrons)
    : _orbitals(orbitals),
      _t1_amplitudes(std::make_unique<amplitude_type>(t1_amplitudes)),
      _t2_amplitudes(std::make_unique<amplitude_type>(t2_amplitudes)) {
  // Get occupied and virtual indices
  auto num_molecular_orbitals = orbitals->get_num_molecular_orbitals();
  // Store counts of occupied and virtual orbitals
  _num_occupied = {n_alpha_electrons, n_beta_electrons};
  _num_virtual = {num_molecular_orbitals - _num_occupied.first,
                  num_molecular_orbitals - _num_occupied.second};

  // Validate dimension of input tensors
  // TODO: This is incorrect for Unrestricted. Workitem: 41348
  if (not orbitals->is_restricted()) {
    throw std::runtime_error(
        "CoupledClusterAmplitudes + Unrestricted Not Yet Implemented");
  }
  const size_t no = _num_occupied.first;
  const size_t nv = _num_virtual.first;
  const size_t nov = no * nv;
  const size_t nov2 = nov * nov;
  if (_t1_amplitudes->size() != nov) {
    throw std::invalid_argument("Invalid T1 amplitudes dimension");
  }
  if (_t2_amplitudes->size() != nov2) {
    throw std::invalid_argument("Invalid T2 amplitudes dimension");
  }

  // Validate that orbital energies exist
  if (!orbitals->has_energies()) {
    throw std::runtime_error(
        "Orbitals are not canonical (no energies provided)");
  }

  // Check that energies are ordered correctly
  const auto& energies_alpha = orbitals->get_energies().first;
  for (size_t i = 1; i < energies_alpha.size(); ++i) {
    if (energies_alpha[i] <= energies_alpha[i - 1]) {
      throw std::runtime_error("Orbital energies are not properly ordered");
    }
  }
  const auto& energies_beta = orbitals->get_energies().second;
  for (size_t i = 1; i < energies_beta.size(); ++i) {
    if (energies_beta[i] <= energies_beta[i - 1]) {
      throw std::runtime_error("Orbital energies are not properly ordered");
    }
  }
}

CoupledClusterAmplitudes::CoupledClusterAmplitudes(
    const CoupledClusterAmplitudes& other) {
  // Copy orbitals if available
  if (other._orbitals) {
    _orbitals = std::make_shared<Orbitals>(*other._orbitals);
  }

  // Copy T1 amplitudes if available
  if (other._t1_amplitudes) {
    _t1_amplitudes = std::make_unique<amplitude_type>(*other._t1_amplitudes);
  }

  // Copy T2 amplitudes if available
  if (other._t2_amplitudes) {
    _t2_amplitudes = std::make_unique<amplitude_type>(*other._t2_amplitudes);
  }

  // Copy occupied and virtual counts
  _num_occupied = other._num_occupied;
  _num_virtual = other._num_virtual;
}

CoupledClusterAmplitudes& CoupledClusterAmplitudes::operator=(
    const CoupledClusterAmplitudes& other) {
  if (this != &other) {
    // Copy orbitals if available
    if (other._orbitals) {
      _orbitals = std::make_shared<Orbitals>(*other._orbitals);
    } else {
      _orbitals.reset();
    }

    // Copy T1 amplitudes if available
    if (other._t1_amplitudes) {
      _t1_amplitudes = std::make_unique<amplitude_type>(*other._t1_amplitudes);
    } else {
      _t1_amplitudes.reset();
    }

    // Copy T2 amplitudes if available
    if (other._t2_amplitudes) {
      _t2_amplitudes = std::make_unique<amplitude_type>(*other._t2_amplitudes);
    } else {
      _t2_amplitudes.reset();
    }

    // Copy occupied and virtual counts
    _num_occupied = other._num_occupied;
    _num_virtual = other._num_virtual;
  }
  return *this;
}

const CoupledClusterAmplitudes::amplitude_type&
CoupledClusterAmplitudes::get_t1_amplitudes() const {
  if (!has_t1_amplitudes()) {
    throw std::runtime_error("T1 amplitudes not set");
  }
  return *_t1_amplitudes;
}

bool CoupledClusterAmplitudes::has_t1_amplitudes() const {
  return _t1_amplitudes != nullptr;
}

const CoupledClusterAmplitudes::amplitude_type&
CoupledClusterAmplitudes::get_t2_amplitudes() const {
  if (!has_t2_amplitudes()) {
    throw std::runtime_error("T2 amplitudes not set");
  }
  return *_t2_amplitudes;
}

bool CoupledClusterAmplitudes::has_t2_amplitudes() const {
  return _t2_amplitudes != nullptr;
}

std::pair<size_t, size_t> CoupledClusterAmplitudes::get_num_occupied() const {
  return _num_occupied;
}

std::pair<size_t, size_t> CoupledClusterAmplitudes::get_num_virtual() const {
  return _num_virtual;
}

std::string CoupledClusterAmplitudes::get_summary() const {
  std::string summary = "CoupledClusterAmplitudes:\n";
  summary += "  T1 amplitudes: ";
  summary += (has_t1_amplitudes() ? "Present" : "Not set");
  summary += "\n";
  summary += "  T2 amplitudes: ";
  summary += (has_t2_amplitudes() ? "Present" : "Not set");
  summary += "\n";

  if (has_t1_amplitudes()) {
    summary += "  T1 size: " + std::to_string(_t1_amplitudes->size()) + "\n";
  }
  if (has_t2_amplitudes()) {
    summary += "  T2 size: " + std::to_string(_t2_amplitudes->size()) + "\n";
  }

  summary += "  Occupied orbitals (alpha, beta): (" +
             std::to_string(_num_occupied.first) + ", " +
             std::to_string(_num_occupied.second) + ")\n";
  summary += "  Virtual orbitals (alpha, beta): (" +
             std::to_string(_num_virtual.first) + ", " +
             std::to_string(_num_virtual.second) + ")\n";

  if (_orbitals) {
    summary += "  Orbitals: Present\n";
  } else {
    summary += "  Orbitals: Not set\n";
  }

  return summary;
}

void CoupledClusterAmplitudes::to_file(const std::string& filename,
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

nlohmann::json CoupledClusterAmplitudes::to_json() const {
  nlohmann::json j;

  // Store orbitals if available
  if (_orbitals) {
    j["orbitals"] = _orbitals->to_json();
  }

  // Store occupation numbers
  j["num_occupied"] = {_num_occupied.first, _num_occupied.second};
  j["num_virtual"] = {_num_virtual.first, _num_virtual.second};

  // Store T1 amplitudes if available
  if (has_t1_amplitudes()) {
    const auto& t1 = *_t1_amplitudes;
    j["t1_amplitudes"] = std::vector<double>(t1.data(), t1.data() + t1.size());
  }

  // Store T2 amplitudes if available
  if (has_t2_amplitudes()) {
    const auto& t2 = *_t2_amplitudes;
    j["t2_amplitudes"] = std::vector<double>(t2.data(), t2.data() + t2.size());
  }

  return j;
}

void CoupledClusterAmplitudes::to_json_file(const std::string& filename) const {
  try {
    std::ofstream file(filename);
    if (!file.is_open()) {
      throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    nlohmann::json j = to_json();
    file << j.dump(2);
    file.close();
  } catch (const std::exception& e) {
    throw std::runtime_error("Error writing JSON file '" + filename +
                             "': " + e.what());
  }
}

void CoupledClusterAmplitudes::to_hdf5(H5::Group& group) const {
  try {
    // Store orbitals if available
    if (_orbitals) {
      H5::Group orbitals_group = group.createGroup("orbitals");
      _orbitals->to_hdf5(orbitals_group);
      orbitals_group.close();
    }

    // Store occupation numbers as attributes
    H5::Attribute occ_alpha_attr =
        group.createAttribute("num_occupied_alpha", H5::PredType::NATIVE_HSIZE,
                              H5::DataSpace(H5S_SCALAR));
    hsize_t occ_alpha = _num_occupied.first;
    occ_alpha_attr.write(H5::PredType::NATIVE_HSIZE, &occ_alpha);

    H5::Attribute occ_beta_attr =
        group.createAttribute("num_occupied_beta", H5::PredType::NATIVE_HSIZE,
                              H5::DataSpace(H5S_SCALAR));
    hsize_t occ_beta = _num_occupied.second;
    occ_beta_attr.write(H5::PredType::NATIVE_HSIZE, &occ_beta);

    H5::Attribute virt_alpha_attr =
        group.createAttribute("num_virtual_alpha", H5::PredType::NATIVE_HSIZE,
                              H5::DataSpace(H5S_SCALAR));
    hsize_t virt_alpha = _num_virtual.first;
    virt_alpha_attr.write(H5::PredType::NATIVE_HSIZE, &virt_alpha);

    H5::Attribute virt_beta_attr =
        group.createAttribute("num_virtual_beta", H5::PredType::NATIVE_HSIZE,
                              H5::DataSpace(H5S_SCALAR));
    hsize_t virt_beta = _num_virtual.second;
    virt_beta_attr.write(H5::PredType::NATIVE_HSIZE, &virt_beta);

    // Store T1 amplitudes if available
    if (has_t1_amplitudes()) {
      const auto& t1 = *_t1_amplitudes;
      hsize_t dims[1] = {static_cast<hsize_t>(t1.size())};
      H5::DataSpace dataspace(1, dims);
      H5::DataSet dataset = group.createDataSet(
          "t1_amplitudes", H5::PredType::NATIVE_DOUBLE, dataspace);
      dataset.write(t1.data(), H5::PredType::NATIVE_DOUBLE);
      dataset.close();
    }

    // Store T2 amplitudes if available
    if (has_t2_amplitudes()) {
      const auto& t2 = *_t2_amplitudes;
      hsize_t dims[1] = {static_cast<hsize_t>(t2.size())};
      H5::DataSpace dataspace(1, dims);
      H5::DataSet dataset = group.createDataSet(
          "t2_amplitudes", H5::PredType::NATIVE_DOUBLE, dataspace);
      dataset.write(t2.data(), H5::PredType::NATIVE_DOUBLE);
      dataset.close();
    }

  } catch (const H5::Exception& e) {
    throw std::runtime_error(
        "HDF5 error in CoupledClusterAmplitudes::to_hdf5: " +
        std::string(e.getCDetailMsg()));
  }
}

void CoupledClusterAmplitudes::to_hdf5_file(const std::string& filename) const {
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

std::shared_ptr<CoupledClusterAmplitudes> CoupledClusterAmplitudes::from_file(
    const std::string& filename, const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  } else if (type == "hdf5" || type == "h5") {
    return from_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5, h5");
  }
}

std::shared_ptr<CoupledClusterAmplitudes> CoupledClusterAmplitudes::from_json(
    const nlohmann::json& j) {
  // Create empty object first
  auto cc_amps = std::make_shared<CoupledClusterAmplitudes>();

  // Load orbitals if present
  if (j.contains("orbitals")) {
    cc_amps->_orbitals = Orbitals::from_json(j["orbitals"]);
  }

  // Load occupation numbers
  if (j.contains("num_occupied") && j["num_occupied"].is_array() &&
      j["num_occupied"].size() == 2) {
    cc_amps->_num_occupied = {j["num_occupied"][0], j["num_occupied"][1]};
  }

  if (j.contains("num_virtual") && j["num_virtual"].is_array() &&
      j["num_virtual"].size() == 2) {
    cc_amps->_num_virtual = {j["num_virtual"][0], j["num_virtual"][1]};
  }

  // Load T1 amplitudes if present
  if (j.contains("t1_amplitudes")) {
    const auto& t1_vec = j["t1_amplitudes"];
    cc_amps->_t1_amplitudes = std::make_unique<amplitude_type>(t1_vec.size());
    for (size_t i = 0; i < t1_vec.size(); ++i) {
      (*cc_amps->_t1_amplitudes)[i] = t1_vec[i];
    }
  }

  // Load T2 amplitudes if present
  if (j.contains("t2_amplitudes")) {
    const auto& t2_vec = j["t2_amplitudes"];
    cc_amps->_t2_amplitudes = std::make_unique<amplitude_type>(t2_vec.size());
    for (size_t i = 0; i < t2_vec.size(); ++i) {
      (*cc_amps->_t2_amplitudes)[i] = t2_vec[i];
    }
  }

  return cc_amps;
}

std::shared_ptr<CoupledClusterAmplitudes>
CoupledClusterAmplitudes::from_json_file(const std::string& filename) {
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

std::shared_ptr<CoupledClusterAmplitudes> CoupledClusterAmplitudes::from_hdf5(
    H5::Group& group) {
  try {
    // Create empty object first
    auto cc_amps = std::make_shared<CoupledClusterAmplitudes>();

    // Load orbitals if present
    if (group.nameExists("orbitals")) {
      H5::Group orbitals_group = group.openGroup("orbitals");
      cc_amps->_orbitals = Orbitals::from_hdf5(orbitals_group);
      orbitals_group.close();
    }

    // Load occupation numbers from attributes
    if (group.attrExists("num_occupied_alpha")) {
      H5::Attribute attr = group.openAttribute("num_occupied_alpha");
      hsize_t val;
      attr.read(H5::PredType::NATIVE_HSIZE, &val);
      cc_amps->_num_occupied.first = val;
    }

    if (group.attrExists("num_occupied_beta")) {
      H5::Attribute attr = group.openAttribute("num_occupied_beta");
      hsize_t val;
      attr.read(H5::PredType::NATIVE_HSIZE, &val);
      cc_amps->_num_occupied.second = val;
    }

    if (group.attrExists("num_virtual_alpha")) {
      H5::Attribute attr = group.openAttribute("num_virtual_alpha");
      hsize_t val;
      attr.read(H5::PredType::NATIVE_HSIZE, &val);
      cc_amps->_num_virtual.first = val;
    }

    if (group.attrExists("num_virtual_beta")) {
      H5::Attribute attr = group.openAttribute("num_virtual_beta");
      hsize_t val;
      attr.read(H5::PredType::NATIVE_HSIZE, &val);
      cc_amps->_num_virtual.second = val;
    }

    // Load T1 amplitudes if present
    if (group.nameExists("t1_amplitudes")) {
      H5::DataSet dataset = group.openDataSet("t1_amplitudes");
      H5::DataSpace dataspace = dataset.getSpace();
      hsize_t dims[1];
      dataspace.getSimpleExtentDims(dims);

      cc_amps->_t1_amplitudes = std::make_unique<amplitude_type>(dims[0]);
      dataset.read(cc_amps->_t1_amplitudes->data(),
                   H5::PredType::NATIVE_DOUBLE);
      dataset.close();
    }

    // Load T2 amplitudes if present
    if (group.nameExists("t2_amplitudes")) {
      H5::DataSet dataset = group.openDataSet("t2_amplitudes");
      H5::DataSpace dataspace = dataset.getSpace();
      hsize_t dims[1];
      dataspace.getSimpleExtentDims(dims);

      cc_amps->_t2_amplitudes = std::make_unique<amplitude_type>(dims[0]);
      dataset.read(cc_amps->_t2_amplitudes->data(),
                   H5::PredType::NATIVE_DOUBLE);
      dataset.close();
    }

    return cc_amps;
  } catch (const H5::Exception& e) {
    throw std::runtime_error(
        "HDF5 error in CoupledClusterAmplitudes::from_hdf5: " +
        std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<CoupledClusterAmplitudes>
CoupledClusterAmplitudes::from_hdf5_file(const std::string& filename) {
  try {
    H5::H5File file(filename, H5F_ACC_RDONLY);
    H5::Group root_group = file.openGroup("/");
    auto result = from_hdf5(root_group);
    root_group.close();
    file.close();
    return result;
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error reading file '" + filename +
                             "': " + std::string(e.getCDetailMsg()));
  }
}

}  // namespace qdk::chemistry::data

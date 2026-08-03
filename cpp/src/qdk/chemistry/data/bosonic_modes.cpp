// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <limits>
#include <qdk/chemistry/data/bosonic_modes.hpp>
#include <qdk/chemistry/utils/hash_context.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "hdf5_error_handling.hpp"

namespace qdk::chemistry::data {

namespace {

/// HDF5/JSON key holding the per-mode local Fock-space dimensions.
constexpr const char* kModeDimensionsKey = "mode_dimensions";

/// Type tag written by to_json()/to_hdf5() and dispatched on by Orbitals.
constexpr const char* kTypeTag = "BosonicModes";

bool is_power_of_two(std::size_t value) {
  return value != 0 && (value & (value - 1)) == 0;
}

/// Render a dimension list compactly, eliding the middle of long lists.
std::string format_dimensions(const std::vector<std::size_t>& dimensions) {
  constexpr std::size_t kMaxShown = 8;
  std::string out = "[";
  for (std::size_t i = 0; i < dimensions.size(); ++i) {
    if (i == kMaxShown && dimensions.size() > kMaxShown + 1) {
      out += ", ... , " + std::to_string(dimensions.back());
      break;
    }
    if (i > 0) {
      out += ", ";
    }
    out += std::to_string(dimensions[i]);
  }
  out += "]";
  return out;
}

}  // namespace

void BosonicModes::_validate_dimension(std::size_t mode,
                                       std::size_t mode_dimension) {
  if (mode_dimension < 2) {
    throw std::invalid_argument(
        "BosonicModes: mode_dimension must be at least 2 (a mode with fewer "
        "than two levels carries no bosonic degree of freedom); mode " +
        std::to_string(mode) + " got " + std::to_string(mode_dimension));
  }
}

BosonicModes::BosonicModes(std::size_t num_modes, std::size_t mode_dimension)
    : ModelOrbitals(num_modes), _mode_dimensions(num_modes, mode_dimension) {
  QDK_LOG_TRACE_ENTERING();
  if (num_modes == 0) {
    throw std::invalid_argument("BosonicModes: num_modes must be at least 1");
  }
  for (std::size_t i = 0; i < _mode_dimensions.size(); ++i) {
    _validate_dimension(i, _mode_dimensions[i]);
  }
}

BosonicModes::BosonicModes(const ModelOrbitals& base,
                           std::size_t mode_dimension)
    : ModelOrbitals(base), _mode_dimensions(base.num_modes(), mode_dimension) {
  QDK_LOG_TRACE_ENTERING();
  for (std::size_t i = 0; i < _mode_dimensions.size(); ++i) {
    _validate_dimension(i, _mode_dimensions[i]);
  }
}

BosonicModes::BosonicModes(const ModelOrbitals& base,
                           std::vector<std::size_t> mode_dimensions)
    : ModelOrbitals(base), _mode_dimensions(std::move(mode_dimensions)) {
  QDK_LOG_TRACE_ENTERING();
  if (_mode_dimensions.size() != base.num_modes()) {
    throw std::invalid_argument(
        "BosonicModes: expected one mode dimension per mode (" +
        std::to_string(base.num_modes()) + ") but got " +
        std::to_string(_mode_dimensions.size()));
  }
  for (std::size_t i = 0; i < _mode_dimensions.size(); ++i) {
    _validate_dimension(i, _mode_dimensions[i]);
  }
}

std::size_t BosonicModes::padded_dimension(std::size_t requested_dimension) {
  std::size_t padded = 2;
  while (padded < requested_dimension) {
    if (padded > std::numeric_limits<std::size_t>::max() / 2) {
      throw std::overflow_error(
          "BosonicModes: requested dimension " +
          std::to_string(requested_dimension) +
          " cannot be padded to a power of two without overflow");
    }
    padded *= 2;
  }
  return padded;
}

std::shared_ptr<BosonicModes> BosonicModes::padded_to_power_of_two(
    std::size_t num_modes, std::size_t requested_dimension) {
  return std::make_shared<BosonicModes>(num_modes,
                                        padded_dimension(requested_dimension));
}

std::shared_ptr<BosonicModes> BosonicModes::hard_core(std::size_t num_modes) {
  // d = 2 keeps only n in {0, 1}: one qubit per mode, b == sigma^-, and the
  // two-body on-site term n(n-1) is identically zero.
  return std::make_shared<BosonicModes>(num_modes, 2);
}

std::shared_ptr<BosonicModes> BosonicModes::with_padded_dimensions() const {
  std::vector<std::size_t> padded;
  padded.reserve(_mode_dimensions.size());
  for (const std::size_t dimension : _mode_dimensions) {
    padded.push_back(padded_dimension(dimension));
  }
  return std::shared_ptr<BosonicModes>(new BosonicModes(*this, padded));
}

std::size_t BosonicModes::mode_dimension(std::size_t mode) const {
  if (mode >= _mode_dimensions.size()) {
    throw std::out_of_range("BosonicModes: mode index " + std::to_string(mode) +
                            " is out of range for " +
                            std::to_string(_mode_dimensions.size()) + " modes");
  }
  return _mode_dimensions[mode];
}

std::size_t BosonicModes::max_occupation(std::size_t mode) const {
  return mode_dimension(mode) - 1;
}

std::optional<std::size_t> BosonicModes::uniform_dimension() const {
  if (_mode_dimensions.empty()) {
    return std::nullopt;
  }
  const std::size_t first = _mode_dimensions.front();
  for (const std::size_t dimension : _mode_dimensions) {
    if (dimension != first) {
      return std::nullopt;
    }
  }
  return first;
}

bool BosonicModes::has_power_of_two_dimensions() const {
  for (const std::size_t dimension : _mode_dimensions) {
    if (!is_power_of_two(dimension)) {
      return false;
    }
  }
  return true;
}

std::size_t BosonicModes::fock_space_dimension() const {
  std::size_t total = 1;
  const std::size_t limit = std::numeric_limits<std::size_t>::max();
  for (std::size_t i = 0; i < num_modes(); ++i) {
    const std::size_t d = mode_dimension(i);
    if (total > limit / d) {
      throw std::overflow_error(
          "BosonicModes: the truncated Fock-space dimension of " +
          std::to_string(num_modes()) + " modes overflows std::size_t");
    }
    total *= d;
  }
  return total;
}

std::string BosonicModes::get_summary() const {
  QDK_LOG_TRACE_ENTERING();
  std::string summary = "BosonicModes Summary:\n";
  summary += "  Modes: " + std::to_string(num_modes()) + "\n";
  const auto uniform = uniform_dimension();
  if (uniform.has_value()) {
    summary += "  Local dimension d: " + std::to_string(*uniform) +
               " (n_max = " + std::to_string(*uniform - 1) + ")\n";
  } else {
    summary += "  Local dimensions d: " + format_dimensions(_mode_dimensions) +
               " (per mode)\n";
  }
  summary += "  Power-of-two dimensions: " +
             std::string(has_power_of_two_dimensions() ? "Yes" : "No") + "\n";
  summary +=
      "  Has active space: " + std::string(has_active_space() ? "Yes" : "No") +
      "\n";
  return summary;
}

nlohmann::json BosonicModes::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  // Reuse the ModelOrbitals payload verbatim (version, mode count, index sets,
  // symmetries) and add the bosonic cutoff on top. The type tag is overridden
  // so that Orbitals::from_json dispatches back to this class. The cutoff is
  // written as one entry per mode so that per-mode cutoffs round-trip without
  // a schema change.
  nlohmann::json j = ModelOrbitals::to_json();
  j["type"] = kTypeTag;
  j[kModeDimensionsKey] = _mode_dimensions;
  return j;
}

std::shared_ptr<BosonicModes> BosonicModes::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  try {
    auto base = ModelOrbitals::from_json(j);
    const std::size_t num_modes = base->num_modes();

    // The schema is an array of one dimension per mode, always. No default is
    // possible here: inventing a cutoff would silently change the physics,
    // which is exactly what the basis-owns-the-cutoff rule is there to prevent.
    if (!j.contains(kModeDimensionsKey)) {
      throw std::runtime_error(
          "JSON missing required mode_dimensions field (an array holding one "
          "local Fock-space dimension per mode)");
    }
    const auto& entry = j[kModeDimensionsKey];
    if (!entry.is_array()) {
      throw std::runtime_error(
          "JSON field mode_dimensions must be an array holding one local "
          "Fock-space dimension per mode");
    }
    auto dimensions = entry.get<std::vector<std::size_t>>();
    if (dimensions.size() != num_modes) {
      throw std::runtime_error("JSON field mode_dimensions holds " +
                               std::to_string(dimensions.size()) +
                               " entries but the basis has " +
                               std::to_string(num_modes) + " modes");
    }
    return std::shared_ptr<BosonicModes>(
        new BosonicModes(*base, std::move(dimensions)));
  } catch (const std::exception& e) {
    throw std::runtime_error("Error parsing BosonicModes JSON: " +
                             std::string(e.what()));
  }
}

void BosonicModes::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  // Write the ModelOrbitals payload, then retag the type and append the
  // per-mode bosonic cutoffs so that Orbitals::from_hdf5 dispatches back to
  // this class.
  ModelOrbitals::to_hdf5(group);
  try {
    H5::StrType string_type(H5::PredType::C_S1, H5T_VARIABLE);

    H5::Group metadata_group = group.openGroup("metadata");
    H5::Attribute type_attr = metadata_group.openAttribute("type");
    std::string type_name = kTypeTag;
    type_attr.write(string_type, type_name);

    std::vector<unsigned> dimensions;
    dimensions.reserve(_mode_dimensions.size());
    for (const std::size_t dimension : _mode_dimensions) {
      dimensions.push_back(static_cast<unsigned>(dimension));
    }
    const hsize_t extent = dimensions.size();
    H5::DataSpace vector_space(1, &extent);
    H5::DataSet dimension_dataset = metadata_group.createDataSet(
        kModeDimensionsKey, H5::PredType::NATIVE_UINT, vector_space);
    dimension_dataset.write(dimensions.data(), H5::PredType::NATIVE_UINT);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<BosonicModes> BosonicModes::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    auto base = ModelOrbitals::from_hdf5(group);
    const std::size_t num_modes = base->num_modes();

    H5::Group metadata_group = group.openGroup("metadata");
    // The schema is a one-dimensional dataset holding one dimension per mode,
    // always. A missing dataset is an error, never a silent default.
    if (!metadata_group.nameExists(kModeDimensionsKey)) {
      throw std::runtime_error(
          "HDF5 group missing required metadata/mode_dimensions dataset (one "
          "local Fock-space dimension per mode)");
    }
    H5::DataSet dataset = metadata_group.openDataSet(kModeDimensionsKey);
    H5::DataSpace space = dataset.getSpace();
    const std::size_t count =
        (space.getSimpleExtentType() == H5S_SCALAR)
            ? 1
            : static_cast<std::size_t>(space.getSimpleExtentNpoints());
    if (count != num_modes) {
      throw std::runtime_error("HDF5 dataset metadata/mode_dimensions holds " +
                               std::to_string(count) +
                               " entries but the basis has " +
                               std::to_string(num_modes) + " modes");
    }
    std::vector<unsigned> raw(count, 0);
    dataset.read(raw.data(), H5::PredType::NATIVE_UINT);
    std::vector<std::size_t> dimensions;
    dimensions.reserve(raw.size());
    for (const unsigned value : raw) {
      dimensions.push_back(static_cast<std::size_t>(value));
    }
    return std::shared_ptr<BosonicModes>(
        new BosonicModes(*base, std::move(dimensions)));
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void BosonicModes::hash_update(qdk::chemistry::utils::HashContext& ctx) const {
  Orbitals::hash_update(ctx);
  qdk::chemistry::utils::hash_value(ctx, std::string(kTypeTag));
  qdk::chemistry::utils::hash_value(
      ctx, static_cast<std::uint64_t>(get_num_molecular_orbitals()));
  for (const std::size_t dimension : _mode_dimensions) {
    qdk::chemistry::utils::hash_value(ctx,
                                      static_cast<std::uint64_t>(dimension));
  }
}

}  // namespace qdk::chemistry::data

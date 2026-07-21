// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <H5Cpp.h>

#include <algorithm>
#include <fstream>
#include <qdk/chemistry/data/symmetry/symmetry.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "../json_serialization.hpp"

namespace qdk::chemistry::data {

std::string to_string(AxisName axis) {
  switch (axis) {
    case AxisName::Spin:
      return "spin";
    case AxisName::ParticleNumber:
      return "particle_number";
  }
  throw std::logic_error("Unknown AxisName value");
}

// ---------------------------------------------------------------------------
// SpinValue
// ---------------------------------------------------------------------------

bool SpinValue::equals(const SymmetryAxisValue& other) const {
  if (other.axis() != AxisName::Spin) {
    return false;
  }
  const auto* spin = dynamic_cast<const SpinValue*>(&other);
  return spin != nullptr && spin->_two_ms == _two_ms;
}

std::size_t SpinValue::hash() const {
  utils::HashContext ctx;
  hash_value(ctx, "spin_value");
  hash_value(ctx, static_cast<int64_t>(AxisName::Spin));
  hash_value(ctx, static_cast<int64_t>(_two_ms));
  return ctx.hash_code();
}

nlohmann::json SpinValue::to_json() const {
  return nlohmann::json{{"kind", to_string(axis())}, {"two_ms", _two_ms}};
}

std::shared_ptr<const SymmetryAxisValue> SpinValue::from_json(
    const nlohmann::json& j) {
  return std::make_shared<const SpinValue>(j.at("two_ms").get<int>());
}

bool ParticleNumberValue::equals(const SymmetryAxisValue& other) const {
  const auto* particle_number =
      dynamic_cast<const ParticleNumberValue*>(&other);
  return particle_number != nullptr && particle_number->_number == _number;
}

std::size_t ParticleNumberValue::hash() const {
  utils::HashContext ctx;
  hash_value(ctx, "particle_number_value");
  hash_value(ctx, static_cast<int64_t>(AxisName::ParticleNumber));
  hash_value(ctx, static_cast<uint64_t>(_number));
  return ctx.hash_code();
}

nlohmann::json ParticleNumberValue::to_json() const {
  return nlohmann::json{{"kind", to_string(axis())}, {"number", _number}};
}

std::shared_ptr<const SymmetryAxisValue> ParticleNumberValue::from_json(
    const nlohmann::json& j) {
  return std::make_shared<const ParticleNumberValue>(
      j.at("number").get<std::size_t>());
}

// ---------------------------------------------------------------------------
// JSON dispatch (internal, no public registry)
// ---------------------------------------------------------------------------

std::shared_ptr<const SymmetryAxisValue> symmetry_axis_value_from_json(
    const nlohmann::json& j) {
  const auto kind = j.at("kind").get<std::string>();
  if (kind == "spin") {
    return SpinValue::from_json(j);
  }
  if (kind == "particle_number") {
    return ParticleNumberValue::from_json(j);
  }
  throw std::runtime_error("Unknown symmetry axis value kind '" + kind + "'.");
}

void hash_value(qdk::chemistry::utils::HashContext& ctx,
                const SymmetryAxisValue& value) {
  hash_value(ctx, static_cast<int64_t>(value.axis()));
  if (value.axis() == AxisName::Spin) {
    const auto* spin = dynamic_cast<const SpinValue*>(&value);
    if (spin == nullptr) {
      throw std::runtime_error("Spin axis value has unexpected runtime type.");
    }
    hash_value(ctx, static_cast<int64_t>(spin->value()));
    return;
  }
  if (value.axis() == AxisName::ParticleNumber) {
    const auto* particle_number =
        dynamic_cast<const ParticleNumberValue*>(&value);
    if (particle_number == nullptr) {
      throw std::runtime_error(
          "Particle-number axis value has unexpected runtime type.");
    }
    hash_value(ctx, static_cast<uint64_t>(particle_number->value()));
    return;
  }
  throw std::logic_error("Unknown AxisName value");
}

// ---------------------------------------------------------------------------
// SymmetryAxis
// ---------------------------------------------------------------------------

SymmetryAxis::SymmetryAxis(
    AxisName name, std::vector<std::shared_ptr<const SymmetryAxisValue>> labels,
    bool equivalent)
    : _name(name), _labels(std::move(labels)), _equivalent(equivalent) {
  for (const auto& label : _labels) {
    if (label == nullptr) {
      throw std::runtime_error(
          "SymmetryAxis labels must not be null pointers.");
    }
    if (label->axis() != _name) {
      throw std::runtime_error(
          "SymmetryAxis label belongs to axis '" + to_string(label->axis()) +
          "' but was added to axis '" + to_string(_name) + "'.");
    }
  }
}

AxisName SymmetryAxis::name() const { return _name; }

const std::vector<std::shared_ptr<const SymmetryAxisValue>>&
SymmetryAxis::labels() const {
  return _labels;
}

bool SymmetryAxis::equivalent() const { return _equivalent; }

bool SymmetryAxis::admits(const SymmetryAxisValue& value) const {
  if (value.axis() != _name) {
    return false;
  }
  return std::any_of(_labels.begin(), _labels.end(),
                     [&](const auto& label) { return label->equals(value); });
}

bool SymmetryAxis::operator==(const SymmetryAxis& other) const {
  if (_name != other._name || _equivalent != other._equivalent ||
      _labels.size() != other._labels.size()) {
    return false;
  }
  for (std::size_t i = 0; i < _labels.size(); ++i) {
    if (!_labels[i]->equals(*other._labels[i])) {
      return false;
    }
  }
  return true;
}

std::size_t SymmetryAxis::hash() const {
  utils::HashContext ctx;
  hash_update(ctx);
  return ctx.hash_code();
}

void SymmetryAxis::hash_update(qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, static_cast<int64_t>(_name));
  hash_value(ctx, _equivalent);
  hash_value(ctx, static_cast<uint64_t>(_labels.size()));
  for (const auto& label : _labels) {
    hash_value(ctx, *label);
  }
}

nlohmann::json SymmetryAxis::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;
  j["version"] = SERIALIZATION_VERSION;
  nlohmann::json labels = nlohmann::json::array();
  for (const auto& label : _labels) {
    labels.push_back(label->to_json());
  }
  j["name"] = to_string(_name);
  j["equivalent"] = _equivalent;
  j["labels"] = std::move(labels);
  return j;
}

std::string SymmetryAxis::get_summary() const {
  std::ostringstream oss;
  oss << "SymmetryAxis(name=" << to_string(_name)
      << ", equivalent=" << (_equivalent ? "true" : "false")
      << ", labels=" << _labels.size() << ")";
  return oss.str();
}

void SymmetryAxis::to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  out << to_json().dump(2);
}

void SymmetryAxis::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);

    // Add version attribute
    H5::Attribute version_attr =
        group.createAttribute("version", str_type, H5::DataSpace(H5S_SCALAR));
    std::string version_str(SERIALIZATION_VERSION);
    version_attr.write(str_type, version_str);

    auto dataset =
        group.createDataSet("data", str_type, H5::DataSpace(H5S_SCALAR));
    dataset.write(to_json().dump(), str_type);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void SymmetryAxis::to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

void SymmetryAxis::to_file(const std::string& filename,
                           const std::string& type) const {
  QDK_LOG_TRACE_ENTERING();
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

std::shared_ptr<SymmetryAxis> SymmetryAxis::from_json(const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  try {
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION,
                                   j["version"].get<std::string>());

    const auto name_str = j.at("name").get<std::string>();
    AxisName name;
    if (name_str == to_string(AxisName::Spin)) {
      name = AxisName::Spin;
    } else if (name_str == to_string(AxisName::ParticleNumber)) {
      name = AxisName::ParticleNumber;
    } else {
      throw std::runtime_error("Unknown symmetry axis name '" + name_str +
                               "'.");
    }
    std::vector<std::shared_ptr<const SymmetryAxisValue>> labels;
    for (const auto& label_json : j.at("labels")) {
      labels.push_back(symmetry_axis_value_from_json(label_json));
    }
    return std::make_shared<SymmetryAxis>(name, std::move(labels),
                        j.at("equivalent").get<bool>());
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse SymmetryAxis from JSON: " +
                             std::string(e.what()));
  }
}

std::shared_ptr<SymmetryAxis> SymmetryAxis::from_json_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  std::ifstream in(filename);
  if (!in) {
    throw std::runtime_error("Failed to open file for reading: " + filename);
  }
  nlohmann::json j;
  in >> j;
  return from_json(j);
}

std::shared_ptr<SymmetryAxis> SymmetryAxis::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);

    // Check version first
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version;
    version_attr.read(str_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

    std::string payload;
    group.openDataSet("data").read(payload, str_type);
    return from_json(nlohmann::json::parse(payload));
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<SymmetryAxis> SymmetryAxis::from_hdf5_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  H5::H5File file(filename, H5F_ACC_RDONLY);
  return from_hdf5(file);
}

std::shared_ptr<SymmetryAxis> SymmetryAxis::from_file(
    const std::string& filename, const std::string& type) {
  if (type == "json") {
    return from_json_file(filename);
  }
  if (type == "hdf5") {
    return from_hdf5_file(filename);
  }
  throw std::invalid_argument("Unsupported file type: " + type +
                              ". Supported types are: json, hdf5");
}

// ---------------------------------------------------------------------------
// SymmetryProduct
// ---------------------------------------------------------------------------

SymmetryProduct::SymmetryProduct(std::vector<SymmetryAxis> axes)
    : _axes(std::move(axes)) {
  for (std::size_t i = 0; i < _axes.size(); ++i) {
    for (std::size_t k = i + 1; k < _axes.size(); ++k) {
      if (_axes[i].name() == _axes[k].name()) {
        throw std::runtime_error(
            "SymmetryProduct must not contain duplicate axis '" +
            to_string(_axes[i].name()) + "'.");
      }
    }
  }
}

const std::vector<SymmetryAxis>& SymmetryProduct::axes() const { return _axes; }

bool SymmetryProduct::has_axis(AxisName name) const {
  return std::any_of(_axes.begin(), _axes.end(),
                     [&](const auto& axis) { return axis.name() == name; });
}

const SymmetryAxis& SymmetryProduct::axis(AxisName name) const {
  for (const auto& axis : _axes) {
    if (axis.name() == name) {
      return axis;
    }
  }
  throw std::runtime_error("SymmetryProduct has no axis '" + to_string(name) +
                           "'.");
}

bool SymmetryProduct::operator==(const SymmetryProduct& other) const {
  return _axes == other._axes;
}

std::size_t SymmetryProduct::hash() const {
  utils::HashContext ctx;
  hash_update(ctx);
  return ctx.hash_code();
}

nlohmann::json SymmetryProduct::to_json() const {
  QDK_LOG_TRACE_ENTERING();
  nlohmann::json j;
  j["version"] = SERIALIZATION_VERSION;
  nlohmann::json axes = nlohmann::json::array();
  for (const auto& axis : _axes) {
    axes.push_back(axis.to_json());
  }
  j["axes"] = std::move(axes);
  return j;
}

std::string SymmetryProduct::get_summary() const {
  std::ostringstream oss;
  oss << "SymmetryProduct(axes=" << _axes.size() << "; [";
  for (std::size_t i = 0; i < _axes.size(); ++i) {
    if (i > 0) oss << ", ";
    oss << to_string(_axes[i].name());
  }
  oss << "])";
  return oss.str();
}

void SymmetryProduct::to_json_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  std::ofstream out(filename);
  if (!out) {
    throw std::runtime_error("Failed to open file for writing: " + filename);
  }
  out << to_json().dump(2);
}

void SymmetryProduct::to_hdf5(H5::Group& group) const {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);

    // Add version attribute
    H5::Attribute version_attr =
        group.createAttribute("version", str_type, H5::DataSpace(H5S_SCALAR));
    std::string version_str(SERIALIZATION_VERSION);
    version_attr.write(str_type, version_str);

    auto dataset =
        group.createDataSet("data", str_type, H5::DataSpace(H5S_SCALAR));
    dataset.write(to_json().dump(), str_type);
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

void SymmetryProduct::to_hdf5_file(const std::string& filename) const {
  QDK_LOG_TRACE_ENTERING();
  H5::H5File file(filename, H5F_ACC_TRUNC);
  to_hdf5(file);
}

void SymmetryProduct::to_file(const std::string& filename,
                              const std::string& type) const {
  QDK_LOG_TRACE_ENTERING();
  if (type == "json") {
    to_json_file(filename);
  } else if (type == "hdf5") {
    to_hdf5_file(filename);
  } else {
    throw std::invalid_argument("Unsupported file type: " + type +
                                ". Supported types are: json, hdf5");
  }
}

std::shared_ptr<SymmetryProduct> SymmetryProduct::from_json(
    const nlohmann::json& j) {
  QDK_LOG_TRACE_ENTERING();
  try {
    if (!j.contains("version")) {
      throw std::runtime_error("Invalid JSON: missing version field");
    }
    validate_serialization_version(SERIALIZATION_VERSION,
                                   j["version"].get<std::string>());

    std::vector<SymmetryAxis> axes;
    for (const auto& axis_json : j.at("axes")) {
      axes.push_back(*SymmetryAxis::from_json(axis_json));
    }
    return std::make_shared<SymmetryProduct>(std::move(axes));
  } catch (const std::exception& e) {
    throw std::runtime_error("Failed to parse SymmetryProduct from JSON: " +
                             std::string(e.what()));
  }
}

std::shared_ptr<SymmetryProduct> SymmetryProduct::from_json_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  std::ifstream in(filename);
  if (!in) {
    throw std::runtime_error("Failed to open file for reading: " + filename);
  }
  nlohmann::json j;
  in >> j;
  return from_json(j);
}

std::shared_ptr<SymmetryProduct> SymmetryProduct::from_hdf5(H5::Group& group) {
  QDK_LOG_TRACE_ENTERING();
  try {
    H5::StrType str_type(H5::PredType::C_S1, H5T_VARIABLE);

    // Check version first
    H5::Attribute version_attr = group.openAttribute("version");
    std::string version;
    version_attr.read(str_type, version);
    validate_serialization_version(SERIALIZATION_VERSION, version);

    std::string payload;
    group.openDataSet("data").read(payload, str_type);
    return from_json(nlohmann::json::parse(payload));
  } catch (const H5::Exception& e) {
    throw std::runtime_error("HDF5 error: " + std::string(e.getCDetailMsg()));
  }
}

std::shared_ptr<SymmetryProduct> SymmetryProduct::from_hdf5_file(
    const std::string& filename) {
  QDK_LOG_TRACE_ENTERING();
  H5::H5File file(filename, H5F_ACC_RDONLY);
  return from_hdf5(file);
}

std::shared_ptr<SymmetryProduct> SymmetryProduct::from_file(
    const std::string& filename, const std::string& type) {
  QDK_LOG_TRACE_ENTERING();
  if (type == "json") {
    return from_json_file(filename);
  }
  if (type == "hdf5") {
    return from_hdf5_file(filename);
  }
  throw std::invalid_argument("Unsupported file type: " + type +
                              ". Supported types are: json, hdf5");
}

void SymmetryProduct::hash_update(
    qdk::chemistry::utils::HashContext& ctx) const {
  hash_value(ctx, get_data_type_name());
  hash_value(ctx, static_cast<uint64_t>(_axes.size()));
  for (const auto& axis : _axes) {
    hash_value(ctx, axis.content_hash());
  }
}

// ---------------------------------------------------------------------------
// SymmetryLabel
// ---------------------------------------------------------------------------

std::size_t SymmetryLabel::_compute_hash(
    const std::map<AxisName, std::shared_ptr<const SymmetryAxisValue>>&
        values) {
  utils::HashContext ctx;
  hash_value(ctx, "symmetry_label");
  hash_value(ctx, static_cast<uint64_t>(values.size()));
  for (const auto& [axis, value] : values) {
    if (value == nullptr) {
      throw std::runtime_error(
          "SymmetryLabel values must not be null pointers.");
    }
    hash_value(ctx, static_cast<int64_t>(axis));
    hash_value(ctx, *value);
  }
  return ctx.hash_code();
}

void hash_value(qdk::chemistry::utils::HashContext& ctx,
                const SymmetryLabel& label) {
  hash_value(ctx, static_cast<uint64_t>(label.values().size()));
  for (const auto& [axis, value] : label.values()) {
    if (value == nullptr) {
      throw std::runtime_error(
          "SymmetryLabel values must not be null pointers.");
    }
    hash_value(ctx, static_cast<int64_t>(axis));
    hash_value(ctx, *value);
  }
}

SymmetryLabel::SymmetryLabel(
    std::initializer_list<std::shared_ptr<const SymmetryAxisValue>> values)
    : SymmetryLabel(
          std::vector<std::shared_ptr<const SymmetryAxisValue>>(values)) {}

SymmetryLabel::SymmetryLabel(
    std::vector<std::shared_ptr<const SymmetryAxisValue>> values)
    : _hash(0) {
  for (auto& value : values) {
    if (value == nullptr) {
      throw std::runtime_error(
          "SymmetryLabel values must not be null pointers.");
    }
    const AxisName axis = value->axis();
    if (_values.count(axis) != 0) {
      throw std::runtime_error(
          "SymmetryLabel must carry at most one value per axis; axis '" +
          to_string(axis) + "' was specified more than once.");
    }
    _values.emplace(axis, std::move(value));
  }
  _hash = _compute_hash(_values);
}

std::shared_ptr<const SymmetryAxisValue> SymmetryLabel::get(
    AxisName axis) const {
  auto it = _values.find(axis);
  if (it == _values.end()) {
    throw std::runtime_error("SymmetryLabel carries no value for axis '" +
                             to_string(axis) + "'.");
  }
  return it->second;
}

bool SymmetryLabel::has(AxisName axis) const {
  return _values.find(axis) != _values.end();
}

bool SymmetryLabel::operator==(const SymmetryLabel& other) const {
  if (_hash != other._hash) {
    return false;
  }
  if (_values.size() != other._values.size()) {
    return false;
  }
  for (const auto& [axis, value] : _values) {
    auto it = other._values.find(axis);
    if (it == other._values.end() || !value->equals(*it->second)) {
      return false;
    }
  }
  return true;
}

nlohmann::json SymmetryLabel::to_json() const {
  nlohmann::json values = nlohmann::json::array();
  for (const auto& [axis, value] : _values) {
    values.push_back(value->to_json());
  }
  return nlohmann::json{{"values", values}};
}

SymmetryLabel SymmetryLabel::from_json(const nlohmann::json& j) {
  std::vector<std::shared_ptr<const SymmetryAxisValue>> values;
  for (const auto& value_json : j.at("values")) {
    values.push_back(symmetry_axis_value_from_json(value_json));
  }
  return SymmetryLabel(std::move(values));
}

// ---------------------------------------------------------------------------
// axes factories
// ---------------------------------------------------------------------------

namespace axes {

const std::shared_ptr<const SpinValue>& alpha() {
  static const std::shared_ptr<const SpinValue> value =
      std::make_shared<const SpinValue>(+1);
  return value;
}

const std::shared_ptr<const SpinValue>& beta() {
  static const std::shared_ptr<const SpinValue> value =
      std::make_shared<const SpinValue>(-1);
  return value;
}

std::shared_ptr<const SpinValue> spin_value(int two_ms) {
  if (two_ms == +1) {
    return alpha();
  }
  if (two_ms == -1) {
    return beta();
  }
  return std::make_shared<const SpinValue>(two_ms);
}

SymmetryAxis spin(unsigned two_s, bool equivalent) {
  // Generate the 2S+1 labels at 2*M_s values {-two_s, -two_s+2, ..., +two_s}.
  std::vector<std::shared_ptr<const SymmetryAxisValue>> labels;
  labels.reserve(static_cast<std::size_t>(two_s) + 1);
  const int signed_two_s = static_cast<int>(two_s);
  for (int two_ms = -signed_two_s; two_ms <= signed_two_s; two_ms += 2) {
    labels.push_back(spin_value(two_ms));
  }
  return SymmetryAxis(AxisName::Spin, std::move(labels), equivalent);
}

std::shared_ptr<const ParticleNumberValue> particle_number_value(
    std::size_t number) {
  return std::make_shared<const ParticleNumberValue>(number);
}

SymmetryAxis particle_number(std::size_t maximum_number) {
  std::vector<std::shared_ptr<const SymmetryAxisValue>> labels;
  labels.reserve(maximum_number + 1);
  for (std::size_t number = 0; number <= maximum_number; ++number) {
    labels.push_back(particle_number_value(number));
  }
  return SymmetryAxis(AxisName::ParticleNumber, std::move(labels), false);
}

}  // namespace axes

}  // namespace qdk::chemistry::data

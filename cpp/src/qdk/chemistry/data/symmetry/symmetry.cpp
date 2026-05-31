// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/data/symmetry/symmetry.hpp>

#include <algorithm>
#include <stdexcept>
#include <qdk/chemistry/utils/hash.hpp>
#include <utility>

namespace qdk::chemistry::data {

std::string to_string(AxisName axis) {
  switch (axis) {
    case AxisName::Spin:
      return "spin";
  }
  return "unknown";
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
  return utils::hash_combine(
      std::hash<int>{}(static_cast<int>(AxisName::Spin)), _two_ms);
}

nlohmann::json SpinValue::to_json() const {
  return nlohmann::json{{"kind", to_string(axis())}, {"two_ms", _two_ms}};
}

std::shared_ptr<const SymmetryAxisValue> SpinValue::from_json(
    const nlohmann::json& j) {
  return std::make_shared<const SpinValue>(j.at("two_ms").get<int>());
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
  throw std::runtime_error(
      "Unknown symmetry axis value kind '" + kind + "'.");
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
  std::size_t seed =
      utils::hash_combine(std::hash<int>{}(static_cast<int>(_name)), _equivalent);
  for (const auto& label : _labels) {
    seed = utils::hash_combine(seed, label->hash());
  }
  return seed;
}

nlohmann::json SymmetryAxis::to_json() const {
  nlohmann::json labels = nlohmann::json::array();
  for (const auto& label : _labels) {
    labels.push_back(label->to_json());
  }
  return nlohmann::json{{"name", to_string(_name)},
                        {"equivalent", _equivalent},
                        {"labels", labels}};
}

SymmetryAxis SymmetryAxis::from_json(const nlohmann::json& j) {
  const auto name_str = j.at("name").get<std::string>();
  if (name_str != to_string(AxisName::Spin)) {
    throw std::runtime_error("Unknown symmetry axis name '" + name_str +
                                 "'.");
  }
  std::vector<std::shared_ptr<const SymmetryAxisValue>> labels;
  for (const auto& label_json : j.at("labels")) {
    labels.push_back(symmetry_axis_value_from_json(label_json));
  }
  return SymmetryAxis(AxisName::Spin, std::move(labels),
                     j.at("equivalent").get<bool>());
}

// ---------------------------------------------------------------------------
// Symmetries
// ---------------------------------------------------------------------------

Symmetries::Symmetries(std::vector<SymmetryAxis> axes)
    : _axes(std::move(axes)) {
  for (std::size_t i = 0; i < _axes.size(); ++i) {
    for (std::size_t k = i + 1; k < _axes.size(); ++k) {
      if (_axes[i].name() == _axes[k].name()) {
        throw std::runtime_error(
            "Symmetries must not contain duplicate axis '" +
            to_string(_axes[i].name()) + "'.");
      }
    }
  }
}

const std::vector<SymmetryAxis>& Symmetries::axes() const { return _axes; }

bool Symmetries::has_axis(AxisName name) const {
  return std::any_of(_axes.begin(), _axes.end(),
                     [&](const auto& axis) { return axis.name() == name; });
}

const SymmetryAxis& Symmetries::axis(AxisName name) const {
  for (const auto& axis : _axes) {
    if (axis.name() == name) {
      return axis;
    }
  }
  throw std::runtime_error("Symmetries has no axis '" + to_string(name) +
                               "'.");
}

bool Symmetries::operator==(const Symmetries& other) const {
  return _axes == other._axes;
}

std::size_t Symmetries::hash() const {
  std::size_t seed = 0xC0FFEEULL;
  for (const auto& axis : _axes) {
    seed = utils::hash_combine(seed, axis.hash());
  }
  return seed;
}

nlohmann::json Symmetries::to_json() const {
  nlohmann::json axes = nlohmann::json::array();
  for (const auto& axis : _axes) {
    axes.push_back(axis.to_json());
  }
  return nlohmann::json{{"axes", axes}};
}

Symmetries Symmetries::from_json(const nlohmann::json& j) {
  std::vector<SymmetryAxis> axes;
  for (const auto& axis_json : j.at("axes")) {
    axes.push_back(SymmetryAxis::from_json(axis_json));
  }
  return Symmetries(std::move(axes));
}

// ---------------------------------------------------------------------------
// SymmetryLabel
// ---------------------------------------------------------------------------

std::size_t SymmetryLabel::_compute_hash(
    const std::map<AxisName, std::shared_ptr<const SymmetryAxisValue>>& values) {
  std::size_t seed = 0;
  for (const auto& [axis, value] : values) {
    if (value == nullptr) {
      throw std::runtime_error(
          "SymmetryLabel values must not be null pointers.");
    }
    seed = utils::hash_combine(
        seed, utils::hash_combine(
                  std::hash<int>{}(static_cast<int>(axis)), value->hash()));
  }
  return seed;
}

SymmetryLabel::SymmetryLabel(
    std::initializer_list<std::shared_ptr<const SymmetryAxisValue>> values)
    : SymmetryLabel(std::vector<std::shared_ptr<const SymmetryAxisValue>>(
          values)) {}

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

SymmetryAxis spin(int /*two_s*/, bool equivalent) {
  return SymmetryAxis(AxisName::Spin, {alpha(), beta()}, equivalent);
}

}  // namespace axes

}  // namespace qdk::chemistry::data

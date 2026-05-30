// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/data/symmetry/symmetry.hpp>

#include <algorithm>
#include <mutex>
#include <qdk/chemistry/data/errors.hpp>
#include <unordered_map>
#include <utility>

namespace qdk::chemistry::data {

namespace {

// Mix a value into a running hash (boost::hash_combine style).
inline void hash_combine(std::size_t& seed, std::size_t value) {
  seed ^= value + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
}

}  // namespace

std::string to_string(AxisName axis) {
  switch (axis) {
    case AxisName::Spin:
      return "spin";
    case AxisName::PointGroup:
      return "point_group";
    case AxisName::Kramers:
      return "kramers";
    case AxisName::Mode:
      return "mode";
    case AxisName::Site:
      return "site";
    case AxisName::LatticeMomentum:
      return "lattice_momentum";
  }
  return "unknown";
}

// ---------------------------------------------------------------------------
// SpinValue
// ---------------------------------------------------------------------------

SpinValue::SpinValue(int two_ms) : _two_ms(two_ms) {}

int SpinValue::value() const { return _two_ms; }

bool SpinValue::equals(const SymmetryAxisValue& other) const {
  if (other.axis() != AxisName::Spin) {
    return false;
  }
  const auto* spin = dynamic_cast<const SpinValue*>(&other);
  return spin != nullptr && spin->_two_ms == _two_ms;
}

std::size_t SpinValue::hash() const {
  std::size_t seed = std::hash<int>{}(static_cast<int>(AxisName::Spin));
  hash_combine(seed, std::hash<int>{}(_two_ms));
  return seed;
}

nlohmann::json SpinValue::to_json() const {
  return nlohmann::json{{"kind", kind_name()}, {"two_ms", _two_ms}};
}

std::shared_ptr<const SymmetryAxisValue> SpinValue::from_json(
    const nlohmann::json& j) {
  return std::make_shared<const SpinValue>(j.at("two_ms").get<int>());
}

// ---------------------------------------------------------------------------
// Serialization registry
// ---------------------------------------------------------------------------

namespace {

using AxisValueFromJson =
    std::function<std::shared_ptr<const SymmetryAxisValue>(
        const nlohmann::json&)>;

std::unordered_map<std::string, AxisValueFromJson>& axis_value_registry() {
  static std::unordered_map<std::string, AxisValueFromJson> registry;
  return registry;
}

std::mutex& axis_value_registry_mutex() {
  static std::mutex mutex;
  return mutex;
}

// Ensure the built-in axis value kinds are registered exactly once.
void ensure_builtin_kinds_registered() {
  static std::once_flag flag;
  std::call_once(flag, [] {
    register_symmetry_axis_value("spin", &SpinValue::from_json);
  });
}

}  // namespace

void register_symmetry_axis_value(const std::string& kind_name,
                                  AxisValueFromJson from_json) {
  std::lock_guard<std::mutex> lock(axis_value_registry_mutex());
  axis_value_registry()[kind_name] = std::move(from_json);
}

std::shared_ptr<const SymmetryAxisValue> symmetry_axis_value_from_json(
    const nlohmann::json& j) {
  ensure_builtin_kinds_registered();
  const auto kind = j.at("kind").get<std::string>();
  AxisValueFromJson handler;
  {
    std::lock_guard<std::mutex> lock(axis_value_registry_mutex());
    auto it = axis_value_registry().find(kind);
    if (it == axis_value_registry().end()) {
      throw SymmetryConditionError(
          "No registered symmetry axis value handler for kind '" + kind + "'.");
    }
    handler = it->second;
  }
  return handler(j);
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
      throw SymmetryConditionError(
          "SymmetryAxis labels must not be null pointers.");
    }
    if (label->axis() != _name) {
      throw SymmetryConditionError(
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
  std::size_t seed = std::hash<int>{}(static_cast<int>(_name));
  hash_combine(seed, std::hash<bool>{}(_equivalent));
  for (const auto& label : _labels) {
    hash_combine(seed, label->hash());
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
  AxisName name = AxisName::Spin;
  bool matched = false;
  for (AxisName candidate :
       {AxisName::Spin, AxisName::PointGroup, AxisName::Kramers,
        AxisName::Mode, AxisName::Site, AxisName::LatticeMomentum}) {
    if (to_string(candidate) == name_str) {
      name = candidate;
      matched = true;
      break;
    }
  }
  if (!matched) {
    throw SymmetryConditionError("Unknown symmetry axis name '" + name_str +
                                 "'.");
  }
  std::vector<std::shared_ptr<const SymmetryAxisValue>> labels;
  for (const auto& label_json : j.at("labels")) {
    labels.push_back(symmetry_axis_value_from_json(label_json));
  }
  return SymmetryAxis(name, std::move(labels), j.at("equivalent").get<bool>());
}

// ---------------------------------------------------------------------------
// Symmetries
// ---------------------------------------------------------------------------

Symmetries::Symmetries(std::vector<SymmetryAxis> axes)
    : _axes(std::move(axes)) {
  for (std::size_t i = 0; i < _axes.size(); ++i) {
    for (std::size_t k = i + 1; k < _axes.size(); ++k) {
      if (_axes[i].name() == _axes[k].name()) {
        throw SymmetryConditionError(
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
  throw SymmetryConditionError("Symmetries has no axis '" + to_string(name) +
                               "'.");
}

bool Symmetries::operator==(const Symmetries& other) const {
  return _axes == other._axes;
}

std::size_t Symmetries::hash() const {
  std::size_t seed = 0xC0FFEEULL;
  for (const auto& axis : _axes) {
    hash_combine(seed, axis.hash());
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

void SymmetryLabel::_recompute_hash() {
  // Order-independent combination over (axis, value-hash) pairs so the hash is
  // stable regardless of std::map iteration details.
  std::size_t accumulator = 0;
  for (const auto& [axis, value] : _values) {
    if (value == nullptr) {
      throw SymmetryConditionError(
          "SymmetryLabel values must not be null pointers.");
    }
    std::size_t pair_hash = std::hash<int>{}(static_cast<int>(axis));
    hash_combine(pair_hash, value->hash());
    accumulator ^= pair_hash;
  }
  _hash = accumulator;
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
      throw SymmetryConditionError(
          "SymmetryLabel values must not be null pointers.");
    }
    const AxisName axis = value->axis();
    if (_values.count(axis) != 0) {
      throw SymmetryConditionError(
          "SymmetryLabel must carry at most one value per axis; axis '" +
          to_string(axis) + "' was specified more than once.");
    }
    _values.emplace(axis, std::move(value));
  }
  _recompute_hash();
}

std::shared_ptr<const SymmetryAxisValue> SymmetryLabel::get(
    AxisName axis) const {
  auto it = _values.find(axis);
  if (it == _values.end()) {
    throw SymmetryConditionError("SymmetryLabel carries no value for axis '" +
                                 to_string(axis) + "'.");
  }
  return it->second;
}

bool SymmetryLabel::has(AxisName axis) const {
  return _values.find(axis) != _values.end();
}

bool SymmetryLabel::operator==(const SymmetryLabel& other) const {
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

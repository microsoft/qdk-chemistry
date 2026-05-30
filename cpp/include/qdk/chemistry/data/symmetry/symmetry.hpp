// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <functional>
#include <initializer_list>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Closed enumeration of the single-particle symmetry axes a basis may
 * be blocked under.
 *
 * Only @c Spin is populated in the current release; the remaining axes are
 * declared for forward compatibility with point-group, time-reversal
 * (Kramers), bosonic-mode, lattice-site, and lattice-momentum partitions.
 */
enum class AxisName { Spin, PointGroup, Kramers, Mode, Site, LatticeMomentum };

/**
 * @brief Human-readable name for an @ref AxisName (used in messages and
 * serialization metadata).
 */
std::string to_string(AxisName axis);

/**
 * @brief Abstract value carried by a single symmetry axis.
 *
 * Concrete subclasses (e.g. @ref SpinValue) represent one label of one axis.
 * Instances are immutable and are shared via @c shared_ptr<const> so that
 * orbit-equivalent uses can be interned. Subclasses must register a
 * deserialization handler via @ref register_symmetry_axis_value so that
 * @ref symmetry_axis_value_from_json can reconstruct them.
 */
class SymmetryAxisValue {
 public:
  virtual ~SymmetryAxisValue() = default;

  /** @brief The axis this value belongs to. */
  virtual AxisName axis() const = 0;

  /** @brief Stable serialization tag identifying the concrete subclass. */
  virtual std::string kind_name() const = 0;

  /** @brief Value-equality against another axis value. */
  virtual bool equals(const SymmetryAxisValue& other) const = 0;

  /** @brief Hash consistent with @ref equals. */
  virtual std::size_t hash() const = 0;

  /** @brief Serialize this value (subclass payload only). */
  virtual nlohmann::json to_json() const = 0;
};

/**
 * @brief Concrete spin-½ axis value.
 *
 * The stored value is @f$2 M_s@f$: @c +1 for an @f$\alpha@f$ label and
 * @c -1 for a @f$\beta@f$ label.
 */
class SpinValue : public SymmetryAxisValue {
  int _two_ms;

 public:
  /** @brief Construct from @f$2 M_s@f$ (e.g. +1 for alpha, -1 for beta). */
  explicit SpinValue(int two_ms);

  /** @brief The stored @f$2 M_s@f$ value. */
  int value() const;

  AxisName axis() const override { return AxisName::Spin; }
  std::string kind_name() const override { return "spin"; }
  bool equals(const SymmetryAxisValue& other) const override;
  std::size_t hash() const override;
  nlohmann::json to_json() const override;

  /** @brief Reconstruct a @ref SpinValue from its JSON payload. */
  static std::shared_ptr<const SymmetryAxisValue> from_json(
      const nlohmann::json& j);
};

/**
 * @brief One named symmetry partition the basis is blocked under.
 *
 * Holds the axis name, the ordered set of admissible labels, and an
 * @c equivalent flag indicating whether the orbit partners under this axis
 * share storage (restricted) or are stored independently (unrestricted).
 */
class SymmetryAxis {
  AxisName _name;
  std::vector<std::shared_ptr<const SymmetryAxisValue>> _labels;
  bool _equivalent;

 public:
  SymmetryAxis(AxisName name,
               std::vector<std::shared_ptr<const SymmetryAxisValue>> labels,
               bool equivalent);

  AxisName name() const;
  const std::vector<std::shared_ptr<const SymmetryAxisValue>>& labels() const;
  bool equivalent() const;

  /** @brief True iff @p value is one of this axis's admissible labels. */
  bool admits(const SymmetryAxisValue& value) const;

  bool operator==(const SymmetryAxis& other) const;
  bool operator!=(const SymmetryAxis& other) const { return !(*this == other); }
  std::size_t hash() const;

  nlohmann::json to_json() const;
  static SymmetryAxis from_json(const nlohmann::json& j);
};

/**
 * @brief A symmetry vocabulary: the ordered set of axes a basis is blocked
 * under, together with their admissible labels and equivalence flags.
 */
class Symmetries {
  std::vector<SymmetryAxis> _axes;

 public:
  explicit Symmetries(std::vector<SymmetryAxis> axes);

  const std::vector<SymmetryAxis>& axes() const;

  /** @brief True iff an axis with name @p name exists in this vocabulary. */
  bool has_axis(AxisName name) const;

  /**
   * @brief Access the axis with name @p name.
   * @throws SymmetryConditionError if no such axis exists.
   */
  const SymmetryAxis& axis(AxisName name) const;

  bool operator==(const Symmetries& other) const;
  bool operator!=(const Symmetries& other) const { return !(*this == other); }
  std::size_t hash() const;

  nlohmann::json to_json() const;
  static Symmetries from_json(const nlohmann::json& j);
};

/**
 * @brief A composite addressing key: one @ref SymmetryAxisValue per axis.
 *
 * Used to address a single block of a @ref SymmetryBlockedTensor or a single
 * label set of a @ref SymmetryBlockedIndexSet. The hash is precomputed at
 * construction so the label can be used as an unordered-map key cheaply.
 */
class SymmetryLabel {
  std::map<AxisName, std::shared_ptr<const SymmetryAxisValue>> _values;
  std::size_t _hash;

  void _recompute_hash();

 public:
  SymmetryLabel(
      std::initializer_list<std::shared_ptr<const SymmetryAxisValue>> values);

  /** @brief Construct from an explicit vector of axis values. */
  explicit SymmetryLabel(
      std::vector<std::shared_ptr<const SymmetryAxisValue>> values);

  /**
   * @brief The value carried for axis @p axis.
   * @throws SymmetryConditionError if the label carries no value for @p axis.
   */
  std::shared_ptr<const SymmetryAxisValue> get(AxisName axis) const;

  /** @brief True iff this label carries a value for @p axis. */
  bool has(AxisName axis) const;

  /** @brief The axes addressed by this label. */
  const std::map<AxisName, std::shared_ptr<const SymmetryAxisValue>>& values()
      const {
    return _values;
  }

  bool operator==(const SymmetryLabel& other) const;
  bool operator!=(const SymmetryLabel& other) const {
    return !(*this == other);
  }
  std::size_t hash() const { return _hash; }

  nlohmann::json to_json() const;
  static SymmetryLabel from_json(const nlohmann::json& j);
};

/**
 * @brief Registry hook: register a @ref SymmetryAxisValue subclass's JSON
 * deserializer under its @c kind_name. Construct-on-first-use.
 */
void register_symmetry_axis_value(
    const std::string& kind_name,
    std::function<std::shared_ptr<const SymmetryAxisValue>(
        const nlohmann::json&)>
        from_json);

/**
 * @brief Reconstruct a @ref SymmetryAxisValue from JSON by dispatching on its
 * @c kind tag through the registry.
 * @throws SymmetryConditionError if the kind tag is not registered.
 */
std::shared_ptr<const SymmetryAxisValue> symmetry_axis_value_from_json(
    const nlohmann::json& j);

namespace axes {

/**
 * @brief Build a spin axis carrying the @f$\alpha@f$ and @f$\beta@f$ labels.
 * @param two_s Twice the total spin (unused beyond label construction in the
 *        Sz-only configuration; reserved for forward compatibility).
 * @param equivalent Whether orbit partners share storage (restricted).
 */
SymmetryAxis spin(int two_s, bool equivalent = true);

/** @brief Interned shared @f$\alpha@f$ spin value (@f$2 M_s = +1@f$). */
const std::shared_ptr<const SpinValue>& alpha();

/** @brief Interned shared @f$\beta@f$ spin value (@f$2 M_s = -1@f$). */
const std::shared_ptr<const SpinValue>& beta();

/** @brief Construct a spin value carrying @f$2 M_s = @f$ @p two_ms. */
std::shared_ptr<const SpinValue> spin_value(int two_ms);

}  // namespace axes

}  // namespace qdk::chemistry::data

namespace std {
template <>
struct hash<qdk::chemistry::data::SymmetryLabel> {
  std::size_t operator()(
      const qdk::chemistry::data::SymmetryLabel& label) const noexcept {
    return label.hash();
  }
};
}  // namespace std

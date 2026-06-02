// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstddef>
#include <initializer_list>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Symmetry axis identifier.
 */
enum class AxisName { Spin };

/**
 * @brief Human-readable name for an @ref AxisName (used in messages and
 * serialization metadata).
 */
std::string to_string(AxisName axis);

/**
 * @brief Abstract value carried by a single symmetry axis.
 *
 * Concrete subclasses (e.g. @ref SpinValue) represent one label of one axis.
 * Instances are shared via @c shared_ptr<const> so that
 * symmetry-equivalent uses can be interned.
 */
class SymmetryAxisValue {
 public:
  virtual ~SymmetryAxisValue() = default;

  /** @brief The axis this value belongs to. */
  virtual AxisName axis() const = 0;

  /** @brief Value-equality against another axis value. */
  virtual bool equals(const SymmetryAxisValue& other) const = 0;

  /** @brief Hash consistent with @ref equals. */
  virtual std::size_t hash() const = 0;

  /** @brief Serialize this value (subclass payload only). */
  virtual nlohmann::json to_json() const = 0;

  /** @brief Convenience wrapper around @ref equals for symmetry with
   * sibling types (@ref SymmetryAxis, @ref Symmetries, @ref SymmetryLabel). */
  bool operator==(const SymmetryAxisValue& other) const {
    return equals(other);
  }
  bool operator!=(const SymmetryAxisValue& other) const {
    return !equals(other);
  }
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
  constexpr explicit SpinValue(int two_ms) : _two_ms(two_ms) {}

  /** @brief The stored @f$2 M_s@f$ value. */
  constexpr int value() const { return _two_ms; }

  AxisName axis() const override { return AxisName::Spin; }
  bool equals(const SymmetryAxisValue& other) const override;
  std::size_t hash() const override;
  nlohmann::json to_json() const override;

  /** @brief Reconstruct a @ref SpinValue from its JSON payload. */
  static std::shared_ptr<const SymmetryAxisValue> from_json(
      const nlohmann::json& j);
};

/**
 * @brief One named symmetry partition a tensor may be blocked under.
 *
 * Holds the axis name, the ordered set of admissible labels, and an
 * @c equivalent flag indicating whether the labels under this axis
 * share storage or are stored independently.
 */
class SymmetryAxis {
  AxisName _name;
  std::vector<std::shared_ptr<const SymmetryAxisValue>> _labels;
  bool _equivalent;

 public:
  /**
   * @brief Construct a symmetry axis with the given name, admissible
   * labels, and equivalence flag.
   *
   * @param name        Identifier of the axis (see @ref AxisName).
   * @param labels      Ordered set of admissible @ref SymmetryAxisValue
   *                    labels carried by this axis.
   * @param equivalent  @c true if the labels under this axis share storage
   *                    (e.g. restricted spin where @f$\alpha@f$ and
   *                    @f$\beta@f$ alias the same MO coefficients).
   */
  SymmetryAxis(AxisName name,
               std::vector<std::shared_ptr<const SymmetryAxisValue>> labels,
               bool equivalent);

  /** @brief The identifier of this axis. */
  AxisName name() const;

  /** @brief The ordered list of admissible labels for this axis. */
  const std::vector<std::shared_ptr<const SymmetryAxisValue>>& labels() const;

  /**
   * @brief Whether labels under this axis share storage (i.e. the
   * restricted-spin alias).
   */
  bool equivalent() const;

  /** @brief True iff @p value is one of this axis's admissible labels. */
  bool admits(const SymmetryAxisValue& value) const;

  /** @brief Value-equality against another axis (same name, labels, and
   * equivalence flag). */
  bool operator==(const SymmetryAxis& other) const;
  bool operator!=(const SymmetryAxis& other) const { return !(*this == other); }

  /** @brief Hash consistent with @ref operator==. */
  std::size_t hash() const;

  /** @brief Serialize this axis to JSON. */
  nlohmann::json to_json() const;

  /** @brief Reconstruct a @ref SymmetryAxis from JSON produced by
   * @ref to_json. */
  static SymmetryAxis from_json(const nlohmann::json& j);
};

/**
 * @brief The ordered set of symmetry axes a tensor is blocked under,
 * together with their admissible labels and equivalence flags.
 */
class Symmetries {
  std::vector<SymmetryAxis> _axes;

 public:
  /**
   * @brief Construct from an ordered list of symmetry axes.
   *
   * @param axes The axes this symmetry set is composed of. The order is
   *             preserved and observable via @ref axes(); axis names must
   *             be unique.
   */
  explicit Symmetries(std::vector<SymmetryAxis> axes);

  /** @brief Construct a trivial symmetry set with no axes. */
  static Symmetries trivial() { return Symmetries({}); }

  /** @brief The ordered list of axes carried by this symmetry set. */
  const std::vector<SymmetryAxis>& axes() const;

  /** @brief True iff an axis with name @p name exists in this set. */
  bool has_axis(AxisName name) const;

  /**
   * @brief Access the axis with name @p name.
   * @throws std::runtime_error if no such axis exists.
   */
  const SymmetryAxis& axis(AxisName name) const;

  /** @brief Value-equality against another symmetry set (same axes in the
   * same order). */
  bool operator==(const Symmetries& other) const;
  bool operator!=(const Symmetries& other) const { return !(*this == other); }

  /** @brief Hash consistent with @ref operator==. */
  std::size_t hash() const;

  /** @brief Serialize this symmetry set to JSON. */
  nlohmann::json to_json() const;

  /** @brief Reconstruct a @ref Symmetries from JSON produced by
   * @ref to_json. */
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

  static std::size_t _compute_hash(
      const std::map<AxisName, std::shared_ptr<const SymmetryAxisValue>>&
          values);

 public:
  /** @brief Construct a trivial (empty) label with no axis values. */
  SymmetryLabel() : _hash(0) {}

  /**
   * @brief Construct from a brace-enclosed list of axis values.
   *
   * Convenience overload for the common interned-shared-ptr pattern:
   * @code
   *   SymmetryLabel alpha({axes::alpha()});
   * @endcode
   *
   * @param values Axis values carried by this label; each axis name must
   *               appear at most once.
   */
  SymmetryLabel(
      std::initializer_list<std::shared_ptr<const SymmetryAxisValue>> values);

  /** @brief Construct from an explicit vector of axis values. */
  explicit SymmetryLabel(
      std::vector<std::shared_ptr<const SymmetryAxisValue>> values);

  /**
   * @brief The value carried for axis @p axis.
   * @throws std::runtime_error if the label carries no value for @p axis.
   */
  std::shared_ptr<const SymmetryAxisValue> get(AxisName axis) const;

  /** @brief True iff this label carries a value for @p axis. */
  bool has(AxisName axis) const;

  /** @brief True iff this label carries no axis values (trivial label). */
  bool empty() const { return _values.empty(); }

  /** @brief The axes addressed by this label. */
  const std::map<AxisName, std::shared_ptr<const SymmetryAxisValue>>& values()
      const {
    return _values;
  }

  /** @brief Value-equality (same axes carrying equal-valued axis values). */
  bool operator==(const SymmetryLabel& other) const;
  bool operator!=(const SymmetryLabel& other) const {
    return !(*this == other);
  }

  /**
   * @brief Hash consistent with @ref operator==.
   *
   * Precomputed in the constructor so the label can be used as an
   * @c unordered_map key without recomputing the hash on each lookup.
   */
  std::size_t hash() const { return _hash; }

  /** @brief Serialize this label to JSON. */
  nlohmann::json to_json() const;

  /** @brief Reconstruct a @ref SymmetryLabel from JSON produced by
   * @ref to_json. */
  static SymmetryLabel from_json(const nlohmann::json& j);
};

/**
 * @brief Reconstruct a @ref SymmetryAxisValue from JSON by dispatching on its
 * @c kind tag.
 * @throws std::runtime_error if the kind tag is not recognized.
 */
std::shared_ptr<const SymmetryAxisValue> symmetry_axis_value_from_json(
    const nlohmann::json& j);

namespace axes {

/**
 * @brief Build a spin-½ axis carrying two labels (@f$2M_s = +1@f$ and
 * @f$2M_s = -1@f$).
 * @param two_s Twice the total spin (reserved for forward compatibility).
 * @param equivalent Whether labels under this axis share storage.
 */
SymmetryAxis spin(int two_s, bool equivalent = true);

/** @brief Interned shared spin-½ value with @f$2 M_s = +1@f$. */
const std::shared_ptr<const SpinValue>& alpha();

/** @brief Interned shared spin-½ value with @f$2 M_s = -1@f$. */
const std::shared_ptr<const SpinValue>& beta();

/** @brief Construct a spin value carrying @f$2 M_s = @f$ @p two_ms. */
std::shared_ptr<const SpinValue> spin_value(int two_ms);

}  // namespace axes

}  // namespace qdk::chemistry::data

namespace std {
/**
 * @brief Specialization of @c std::hash for @ref qdk::chemistry::data::
 * SymmetryLabel, so labels can be used as keys in unordered associative
 * containers.
 */
template <>
struct hash<qdk::chemistry::data::SymmetryLabel> {
  std::size_t operator()(
      const qdk::chemistry::data::SymmetryLabel& label) const noexcept {
    return label.hash();
  }
};
}  // namespace std

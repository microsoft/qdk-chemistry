// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <concepts>
#include <cstddef>
#include <initializer_list>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/utils/string_utils.hpp>
#include <string>
#include <type_traits>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Symmetry axis identifier.
 */
enum class AxisName {
  /** @brief The spin (@f$S_z@f$) axis carrying @ref SpinValue labels. */
  Spin
};

/**
 * @brief Human-readable name for an @ref AxisName (used in messages and
 * serialization metadata).
 *
 * @param axis Axis identifier to render.
 * @return A stable lower-case string (e.g. @c "spin" for @ref AxisName::Spin).
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
  /**
   * @brief Defaulted virtual destructor.
   *
   * Declared virtual so that derived axis values can be safely deleted
   * through a @ref SymmetryAxisValue pointer.
   */
  virtual ~SymmetryAxisValue() = default;

  /**
   * @brief The axis this value belongs to.
   * @return The @ref AxisName identifying the owning axis.
   */
  virtual AxisName axis() const = 0;

  /**
   * @brief Value-equality against another axis value.
   *
   * @param other Right-hand axis value to compare against.
   * @return @c true iff @p other is the same concrete subclass and carries
   *         the same payload.
   */
  virtual bool equals(const SymmetryAxisValue& other) const = 0;

  /**
   * @brief Hash consistent with @ref equals.
   * @return Hash value derived from the subclass payload.
   */
  virtual std::size_t hash() const = 0;

  /**
   * @brief Serialize this value (subclass payload only).
   *
   * The polymorphic @c "kind" tag is written by the enclosing
   * @ref SymmetryLabel / @ref SymmetryAxis so subclasses need only emit
   * their own state.
   *
   * @return JSON object carrying the subclass payload.
   */
  virtual nlohmann::json to_json() const = 0;

  /** @brief Convenience wrapper around @ref equals for symmetry with
   * sibling types (@ref SymmetryAxis, @ref SymmetryProduct, @ref
   * SymmetryLabel).
   *
   * @param other Axis value to compare against.
   * @return @c true iff @ref equals returns @c true for @p other.
   */
  bool operator==(const SymmetryAxisValue& other) const {
    return equals(other);
  }
  /**
   * @brief Negation of @ref operator==.
   *
   * @param other Axis value to compare against.
   * @return @c true iff @ref equals returns @c false for @p other.
   */
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
  /**
   * @brief Construct from @f$2 M_s@f$ (e.g. +1 for alpha, -1 for beta).
   * @param two_ms Twice the spin projection of the represented label.
   */
  constexpr explicit SpinValue(int two_ms) : _two_ms(two_ms) {}

  /**
   * @brief The stored @f$2 M_s@f$ value.
   * @return @c +1 for @f$\alpha@f$, @c -1 for @f$\beta@f$, or any other
   *         value carried at construction.
   */
  constexpr int value() const { return _two_ms; }

  /**
   * @brief The axis this value belongs to.
   * @return Always @ref AxisName::Spin.
   */
  AxisName axis() const override { return AxisName::Spin; }
  /**
   * @brief Value-equality against another axis value.
   * @param other Axis value to compare against.
   * @return @c true iff @p other is a @ref SpinValue carrying the same
   *         @f$2 M_s@f$ value.
   */
  bool equals(const SymmetryAxisValue& other) const override;
  /**
   * @brief Hash consistent with @ref equals.
   * @return Hash value derived from the stored @f$2 M_s@f$.
   */
  std::size_t hash() const override;
  /**
   * @brief Serialize this value (subclass payload only).
   * @return JSON object carrying the @f$2 M_s@f$ value under a stable key.
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Reconstruct a @ref SpinValue from its JSON payload.
   *
   * @param j JSON object produced by a prior @ref SpinValue::to_json call.
   * @return Shared pointer to the deserialized value typed as the
   *         polymorphic @ref SymmetryAxisValue base.
   * @throws std::runtime_error if @p j is missing the @f$2 M_s@f$ payload
   *         or it is of the wrong type.
   */
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
class SymmetryAxis : public DataClass {
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

  /**
   * @brief The identifier of this axis.
   * @return The axis name (see @ref AxisName).
   */
  AxisName name() const;

  /**
   * @brief The ordered list of admissible labels for this axis.
   * @return Reference to the labels supplied at construction.
   */
  const std::vector<std::shared_ptr<const SymmetryAxisValue>>& labels() const;

  /**
   * @brief Whether labels under this axis share storage (i.e. the
   * restricted-spin alias).
   * @return @c true if labels alias the same storage; @c false otherwise.
   */
  bool equivalent() const;

  /**
   * @brief True iff @p value is one of this axis's admissible labels.
   * @param value Candidate axis value to test.
   * @return @c true if @p value is admissible under this axis.
   */
  bool admits(const SymmetryAxisValue& value) const;

  /**
   * @brief Value-equality against another axis.
   * @param other Right-hand axis to compare against.
   * @return @c true if name, labels, and equivalence flag all match.
   */
  bool operator==(const SymmetryAxis& other) const;

  /**
   * @brief Inverse of @ref operator==.
   * @param other Right-hand axis to compare against.
   * @return @c true if any of name, labels, or equivalence flag differs.
   */
  bool operator!=(const SymmetryAxis& other) const { return !(*this == other); }

  /**
   * @brief Hash consistent with @ref operator==.
   * @return Hash value suitable for use with @c std::hash.
   */
  std::size_t hash() const;

  /**
   * @brief @ref DataClass type identifier.
   * @return The string @c "symmetry_axis".
   */
  std::string get_data_type_name() const override {
    return DATACLASS_TO_SNAKE_CASE(SymmetryAxis);
  }

  /**
   * @brief Single-line human-readable summary.
   * @return A string of the form
   *         <tt>SymmetryAxis(name=..., equivalent=..., labels=...)</tt>.
   */
  std::string get_summary() const override;

  /**
   * @brief Serialize this axis to JSON.
   * @return JSON object with @c "name", @c "equivalent", and @c "labels"
   *         fields. Each label is the verbatim @c to_json fragment of a
   *         @ref SymmetryAxisValue (whose polymorphic @c "kind" tag is
   *         preserved).
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Serialize this axis to a JSON file.
   * @param filename Path to the output JSON file.
   * @throws std::runtime_error if @p filename cannot be opened for writing.
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Serialize this axis into an HDF5 group.
   *
   * The implementation writes the JSON form (see @ref to_json) as a single
   * string dataset within @p group, mirroring the HDF5 layout used by
   * @ref SymmetryBlockedIndexSet.
   *
   * @param group HDF5 group to write data to.
   * @throws std::runtime_error if an HDF5 I/O error occurs.
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Serialize this axis to an HDF5 file.
   * @param filename Path to the output HDF5 file (created or truncated).
   * @throws std::runtime_error if an HDF5 I/O error occurs.
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Dispatch to JSON or HDF5 serialization based on @p type.
   * @param filename Path to the output file.
   * @param type Either @c "json" or @c "hdf5".
   * @throws std::invalid_argument if @p type is not @c "json" or @c "hdf5".
   * @throws std::runtime_error if an I/O error occurs.
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Reconstruct a @ref SymmetryAxis from JSON produced by
   * @ref to_json.
   * @param j JSON object containing axis data.
   * @return Shared pointer to the deserialized @ref SymmetryAxis.
   * @throws std::runtime_error if @p j is missing required fields or
   *         contains an unknown axis name.
   */
  static std::shared_ptr<SymmetryAxis> from_json(const nlohmann::json& j);

  /**
   * @brief Load a @ref SymmetryAxis from a JSON file produced by
   * @ref to_json_file.
   * @param filename Path to the input JSON file.
   * @return Shared pointer to the deserialized @ref SymmetryAxis.
   * @throws std::runtime_error if @p filename cannot be read or its
   *         contents are malformed.
   */
  static std::shared_ptr<SymmetryAxis> from_json_file(
      const std::string& filename);

  /**
   * @brief Load a @ref SymmetryAxis from an HDF5 group produced by
   * @ref to_hdf5.
   * @param group HDF5 group to read data from.
   * @return Shared pointer to the deserialized @ref SymmetryAxis.
   * @throws std::runtime_error if the expected dataset is missing or the
   *         payload is malformed.
   */
  static std::shared_ptr<SymmetryAxis> from_hdf5(H5::Group& group);

  /**
   * @brief Load a @ref SymmetryAxis from an HDF5 file produced by
   * @ref to_hdf5_file.
   * @param filename Path to the input HDF5 file.
   * @return Shared pointer to the deserialized @ref SymmetryAxis.
   * @throws std::runtime_error if @p filename cannot be read or its
   *         contents are malformed.
   */
  static std::shared_ptr<SymmetryAxis> from_hdf5_file(
      const std::string& filename);

  /**
   * @brief Dispatch to JSON or HDF5 deserialization based on @p type.
   * @param filename Path to the input file.
   * @param type Either @c "json" or @c "hdf5".
   * @return Shared pointer to the deserialized @ref SymmetryAxis.
   * @throws std::invalid_argument if @p type is not @c "json" or @c "hdf5".
   * @throws std::runtime_error if @p filename cannot be read or its
   *         contents are malformed.
   */
  static std::shared_ptr<SymmetryAxis> from_file(const std::string& filename,
                                                 const std::string& type);

 private:
  /// On-disk serialization format version. Bump on any change to the JSON
  /// or HDF5 shape produced by @ref to_json / @ref to_hdf5.
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

/**
 * @brief The ordered set of symmetry axes a tensor is blocked under,
 * together with their admissible labels and equivalence flags.
 */
class SymmetryProduct : public DataClass {
  std::vector<SymmetryAxis> _axes;

 public:
  /**
   * @brief Construct from an ordered list of symmetry axes.
   *
   * @param axes The axes this symmetry set is composed of. The order is
   *             preserved and observable via @ref axes(); axis names must
   *             be unique.
   */
  explicit SymmetryProduct(std::vector<SymmetryAxis> axes);

  /**
   * @brief Construct a trivial symmetry set with no axes.
   * @return An empty @ref SymmetryProduct instance.
   */
  static SymmetryProduct trivial() { return SymmetryProduct({}); }

  /**
   * @brief The ordered list of axes carried by this symmetry set.
   * @return Reference to the axes supplied at construction.
   */
  const std::vector<SymmetryAxis>& axes() const;

  /**
   * @brief Look up whether an axis with the given name exists.
   * @param name Axis name to look for.
   * @return @c true if an axis with name @p name exists in this set.
   */
  bool has_axis(AxisName name) const;

  /**
   * @brief Access the axis with name @p name.
   * @param name Axis name to look up.
   * @return Reference to the matching @ref SymmetryAxis.
   * @throws std::runtime_error if no such axis exists in this set.
   */
  const SymmetryAxis& axis(AxisName name) const;

  /**
   * @brief Value-equality against another symmetry set.
   * @param other Right-hand symmetry set to compare against.
   * @return @c true if both sets contain the same axes in the same order.
   */
  bool operator==(const SymmetryProduct& other) const;

  /**
   * @brief Inverse of @ref operator==.
   * @param other Right-hand symmetry set to compare against.
   * @return @c true if the axis sequences differ.
   */
  bool operator!=(const SymmetryProduct& other) const {
    return !(*this == other);
  }

  /**
   * @brief Hash consistent with @ref operator==.
   * @return Hash value suitable for use with @c std::hash.
   */
  std::size_t hash() const;

  /**
   * @brief @ref DataClass type identifier.
   * @return The string @c "symmetries".
   */
  std::string get_data_type_name() const override {
    return DATACLASS_TO_SNAKE_CASE(SymmetryProduct);
  }

  /**
   * @brief Single-line human-readable summary.
   * @return A string of the form <tt>SymmetryProduct(axes=N; [name0, name1,
   * ...])</tt>.
   */
  std::string get_summary() const override;

  /**
   * @brief Serialize this symmetry set to JSON.
   * @return JSON object with an @c "axes" array; each entry is the
   *         @ref SymmetryAxis::to_json output of the corresponding axis.
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Serialize this symmetry set to a JSON file.
   * @param filename Path to the output JSON file.
   * @throws std::runtime_error if @p filename cannot be opened for writing.
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Serialize this symmetry set into an HDF5 group.
   *
   * The implementation writes the JSON form (see @ref to_json) as a single
   * string dataset within @p group, mirroring the HDF5 layout used by
   * @ref SymmetryBlockedIndexSet.
   *
   * @param group HDF5 group to write data to.
   * @throws std::runtime_error if an HDF5 I/O error occurs.
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Serialize this symmetry set to an HDF5 file.
   * @param filename Path to the output HDF5 file (created or truncated).
   * @throws std::runtime_error if an HDF5 I/O error occurs.
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Dispatch to JSON or HDF5 serialization based on @p type.
   * @param filename Path to the output file.
   * @param type Either @c "json" or @c "hdf5".
   * @throws std::invalid_argument if @p type is not @c "json" or @c "hdf5".
   * @throws std::runtime_error if an I/O error occurs.
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Reconstruct a @ref SymmetryProduct from JSON produced by
   * @ref to_json.
   * @param j JSON object containing symmetry-set data.
   * @return Shared pointer to the deserialized @ref SymmetryProduct.
   * @throws std::runtime_error if @p j is missing required fields or an
   *         embedded axis is malformed.
   */
  static std::shared_ptr<SymmetryProduct> from_json(const nlohmann::json& j);

  /**
   * @brief Load a @ref SymmetryProduct from a JSON file produced by
   * @ref to_json_file.
   * @param filename Path to the input JSON file.
   * @return Shared pointer to the deserialized @ref SymmetryProduct.
   * @throws std::runtime_error if @p filename cannot be read or its
   *         contents are malformed.
   */
  static std::shared_ptr<SymmetryProduct> from_json_file(
      const std::string& filename);

  /**
   * @brief Load a @ref SymmetryProduct from an HDF5 group produced by
   * @ref to_hdf5.
   * @param group HDF5 group to read data from.
   * @return Shared pointer to the deserialized @ref SymmetryProduct.
   * @throws std::runtime_error if the expected dataset is missing or the
   *         payload is malformed.
   */
  static std::shared_ptr<SymmetryProduct> from_hdf5(H5::Group& group);

  /**
   * @brief Load a @ref SymmetryProduct from an HDF5 file produced by
   * @ref to_hdf5_file.
   * @param filename Path to the input HDF5 file.
   * @return Shared pointer to the deserialized @ref SymmetryProduct.
   * @throws std::runtime_error if @p filename cannot be read or its
   *         contents are malformed.
   */
  static std::shared_ptr<SymmetryProduct> from_hdf5_file(
      const std::string& filename);

  /**
   * @brief Dispatch to JSON or HDF5 deserialization based on @p type.
   * @param filename Path to the input file.
   * @param type Either @c "json" or @c "hdf5".
   * @return Shared pointer to the deserialized @ref SymmetryProduct.
   * @throws std::invalid_argument if @p type is not @c "json" or @c "hdf5".
   * @throws std::runtime_error if @p filename cannot be read or its
   *         contents are malformed.
   */
  static std::shared_ptr<SymmetryProduct> from_file(const std::string& filename,
                                                    const std::string& type);

 private:
  /// On-disk serialization format version. Bump on any change to the JSON
  /// or HDF5 shape produced by @ref to_json / @ref to_hdf5.
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
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

  /**
   * @brief Construct from a single axis value (implicit conversion).
   *
   * Lets call sites write @c block({axes::alpha(), axes::beta()}) without
   * naming @ref SymmetryLabel explicitly — each axis-value shared pointer
   * is wrapped into a label carrying just that one value. Accepts a shared
   * pointer to any @ref SymmetryAxisValue subclass (e.g. @ref SpinValue).
   *
   * @tparam Derived Any @ref SymmetryAxisValue subclass.
   * @param value Axis value carried by this label.
   */
  template <std::derived_from<SymmetryAxisValue> Derived>
  SymmetryLabel(std::shared_ptr<const Derived> value)
      : SymmetryLabel(
            std::initializer_list<std::shared_ptr<const SymmetryAxisValue>>{
                std::shared_ptr<const SymmetryAxisValue>(std::move(value))}) {}

  /**
   * @brief Construct from an explicit vector of axis values.
   * @param values Axis values carried by this label; each axis name must
   *               appear at most once.
   */
  explicit SymmetryLabel(
      std::vector<std::shared_ptr<const SymmetryAxisValue>> values);

  /**
   * @brief The value carried for axis @p axis.
   * @param axis Axis identifier to look up.
   * @return Shared pointer to the value carried for @p axis.
   * @throws std::runtime_error if the label carries no value for @p axis.
   */
  std::shared_ptr<const SymmetryAxisValue> get(AxisName axis) const;

  /**
   * @brief True iff this label carries a value for @p axis.
   * @param axis Axis identifier to look up.
   * @return @c true if @c get(@p axis) would succeed.
   */
  bool has(AxisName axis) const;

  /**
   * @brief True iff this label carries no axis values (trivial label).
   * @return @c true iff @ref values() is empty.
   */
  bool empty() const { return _values.empty(); }

  /**
   * @brief The axes addressed by this label.
   * @return Reference to the underlying ordered map from axis name to value.
   */
  const std::map<AxisName, std::shared_ptr<const SymmetryAxisValue>>& values()
      const {
    return _values;
  }

  /**
   * @brief Value-equality (same axes carrying equal-valued axis values).
   *
   * @param other Label to compare against.
   * @return @c true iff @p other carries the same set of axes and each
   *         per-axis value compares equal via @ref SymmetryAxisValue::equals.
   */
  bool operator==(const SymmetryLabel& other) const;
  /**
   * @brief Negation of @ref operator==.
   *
   * @param other Label to compare against.
   * @return @c true iff @p other differs from @c *this on any axis.
   */
  bool operator!=(const SymmetryLabel& other) const {
    return !(*this == other);
  }

  /**
   * @brief Hash consistent with @ref operator==.
   *
   * Precomputed in the constructor so the label can be used as an
   * @c unordered_map key without recomputing the hash on each lookup.
   *
   * @return Cached hash value of this label.
   */
  std::size_t hash() const { return _hash; }

  /**
   * @brief Serialize this label to JSON.
   * @return JSON object enumerating each per-axis value (with its
   *         polymorphic @c "kind" tag).
   */
  nlohmann::json to_json() const;

  /**
   * @brief Reconstruct a @ref SymmetryLabel from JSON produced by
   * @ref to_json.
   * @param j JSON object produced by a prior @ref to_json call.
   * @return The deserialized label.
   * @throws std::runtime_error if @p j is malformed or names an unknown
   *         axis kind.
   */
  static SymmetryLabel from_json(const nlohmann::json& j);
};

/**
 * @brief Reconstruct a @ref SymmetryAxisValue from JSON by dispatching on its
 * @c kind tag.
 *
 * @param j JSON object produced by a prior @c to_json call on a concrete
 *          @ref SymmetryAxisValue subclass.
 * @return Shared pointer to the deserialized value typed as the polymorphic
 *         base.
 * @throws std::runtime_error if the @c kind tag is missing or not
 *         recognized.
 */
std::shared_ptr<const SymmetryAxisValue> symmetry_axis_value_from_json(
    const nlohmann::json& j);

namespace axes {

/**
 * @brief Build a spin axis carrying @c 2s+1 labels at @f$2 M_s \in
 * \{-2s, -2s+2, \ldots, +2s\}@f$.
 *
 * For the standard single-particle (@p two_s = 1) case, this is the
 * spin-1/2 axis with the two labels @f$\alpha@f$ (@c 2 M_s = +1) and
 * @f$\beta@f$ (@c 2 M_s = -1).
 *
 * @param two_s Twice the total spin.
 * @param equivalent Whether labels under this axis share storage (i.e.
 *                   restricted-spin storage).
 * @return A fully populated @ref SymmetryAxis for the spin degree of
 *         freedom.
 */
SymmetryAxis spin(unsigned two_s, bool equivalent = true);

/**
 * @brief Interned shared spin-½ value with @f$2 M_s = +1@f$.
 * @return Reference to the global @f$\alpha@f$ instance; safe to capture
 *         by @c shared_ptr.
 */
const std::shared_ptr<const SpinValue>& alpha();

/**
 * @brief Interned shared spin-½ value with @f$2 M_s = -1@f$.
 * @return Reference to the global @f$\beta@f$ instance; safe to capture
 *         by @c shared_ptr.
 */
const std::shared_ptr<const SpinValue>& beta();

/**
 * @brief Construct a spin value carrying @f$2 M_s = @f$ @p two_ms.
 *
 * @param two_ms Twice the spin projection. The standard interned values
 *               (@ref alpha and @ref beta) are returned when @p two_ms is
 *               @c +1 or @c -1 respectively.
 * @return Shared pointer to the (possibly interned) spin value.
 */
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
  /**
   * @brief Forward to @ref qdk::chemistry::data::SymmetryLabel::hash.
   * @param label Label to hash.
   * @return The precomputed hash value carried by @p label.
   */
  std::size_t operator()(
      const qdk::chemistry::data::SymmetryLabel& label) const noexcept {
    return label.hash();
  }
};
}  // namespace std

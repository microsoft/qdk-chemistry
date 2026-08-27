// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <H5Cpp.h>

#include <cstddef>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/data_class.hpp>
#include <qdk/chemistry/data/shell.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <string>
#include <string_view>
#include <vector>

namespace qdk::chemistry::data {

/** @brief Algorithm-facing purpose served by an auxiliary basis. */
enum class AuxiliaryBasisRole {
  JFit,   ///< Coulomb density fitting
  JKFit,  ///< Coulomb and exchange density fitting
  RIFit,  ///< Correlation integral fitting
  CABS    ///< Complementary auxiliary basis for explicitly correlated methods
};

/**
 * @brief Return the stable wire-format name for an auxiliary-basis role.
 * @param role Role to convert
 * @return One of @c jfit, @c jkfit, @c rifit, or @c cabs
 * @throws std::invalid_argument if @p role is not a recognized enum value
 */
std::string to_string(AuxiliaryBasisRole role);

/**
 * @brief Parse a canonical auxiliary-basis role wire name.
 * @param role Canonical role name
 * @return Parsed auxiliary-basis role
 * @throws std::invalid_argument if @p role is not one of @c jfit, @c jkfit,
 *         @c rifit, or @c cabs
 */
AuxiliaryBasisRole auxiliary_basis_role_from_string(std::string role);

/**
 * @class AuxiliaryBasis
 * @brief Secondary atom-centered Gaussian basis supplied to algorithms that
 *        require one
 *
 * Independent of the primary @ref BasisSet, carrying its own shells, orbital
 * representation and molecular structure. Density fitting is the most common
 * consumer, but the class holds no algorithm-specific role, so a calculation
 * may use several auxiliary bases for different purposes.
 */
class AuxiliaryBasis : public DataClass,
                       public std::enable_shared_from_this<AuxiliaryBasis> {
 public:
  /** @brief Name assigned to unnamed custom auxiliary-basis data. */
  static constexpr std::string_view custom_name = "custom_aux";

  /**
   * @brief Construct an unnamed custom auxiliary basis.
   * @param shells Auxiliary Gaussian shells
   * @param structure Molecular structure on which the shells are centered
   * @param atomic_orbital_type Spherical or Cartesian orbital representation
   * @throws std::invalid_argument if the structure is null, the shell list is
   *         empty, or a shell is invalid for an auxiliary basis
   */
  AuxiliaryBasis(std::vector<Shell> shells,
                 std::shared_ptr<Structure> structure,
                 AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Construct a named custom auxiliary basis.
   * @param name Auxiliary-basis name
   * @param shells Auxiliary Gaussian shells
   * @param structure Molecular structure on which the shells are centered
   * @param atomic_orbital_type Spherical or Cartesian orbital representation
   * @throws std::invalid_argument if the name is empty, the structure is null,
   *         the shell list is empty, or a shell is invalid
   */
  AuxiliaryBasis(std::string name, std::vector<Shell> shells,
                 std::shared_ptr<Structure> structure,
                 AOType atomic_orbital_type = AOType::Spherical);

  /** @brief Get the auxiliary-basis name. @return Basis name. */
  const std::string& get_name() const;

  /** @brief Get the orbital representation. @return Spherical or Cartesian. */
  AOType get_atomic_orbital_type() const;

  /** @brief Get the associated molecular structure. @return Structure. */
  std::shared_ptr<Structure> get_structure() const;

  /** @brief Get all shells in canonical atom order. @return Shell copy. */
  std::vector<Shell> get_shells() const;

  /**
   * @brief Get shells centered on one atom.
   * @param atom_index Zero-based atom index
   * @return Shells for the requested atom, possibly empty
   * @throws std::out_of_range if @p atom_index is outside the structure
   */
  const std::vector<Shell>& get_shells_for_atom(size_t atom_index) const;

  /**
   * @brief Get one shell by its global canonical index.
   * @param shell_index Zero-based shell index
   * @return Requested shell
   * @throws std::out_of_range if @p shell_index is invalid
   */
  const Shell& get_shell(size_t shell_index) const;

  /** @brief Get the total number of shells. @return Shell count. */
  size_t get_num_shells() const;

  /** @brief Get the number of atoms. @return Structure atom count. */
  size_t get_num_atoms() const;

  /**
   * @brief Get the total number of auxiliary orbitals.
   * @return Orbital count under the selected orbital representation
   */
  size_t get_num_auxiliary_orbitals() const;

  /**
   * @brief Load one named auxiliary basis for every atom.
   * @param basis_name Basis-library name
   * @param structure Molecular structure
   * @param atomic_orbital_type Spherical or Cartesian representation
   * @return Loaded auxiliary basis
   * @throws std::invalid_argument if the structure is null or the basis is not
   *         available for an element in the structure
   */
  static std::shared_ptr<AuxiliaryBasis> from_basis_name(
      std::string basis_name, std::shared_ptr<Structure> structure,
      AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Load auxiliary bases selected by element symbol.
   * @param element_to_basis_map Basis-library name for each element
   * @param structure Molecular structure
   * @param atomic_orbital_type Spherical or Cartesian representation
   * @return Custom auxiliary basis assembled from the selected definitions
   * @throws std::invalid_argument if the structure is null, an element is
   *         missing, or a selected basis is unavailable
   */
  static std::shared_ptr<AuxiliaryBasis> from_element_map(
      const std::map<std::string, std::string>& element_to_basis_map,
      std::shared_ptr<Structure> structure,
      AOType atomic_orbital_type = AOType::Spherical);

  /**
   * @brief Load auxiliary bases selected by atom index.
   * @param index_to_basis_map Basis-library name for each atom index
   * @param structure Molecular structure
   * @param atomic_orbital_type Spherical or Cartesian representation
   * @return Custom auxiliary basis assembled from the selected definitions
   * @throws std::invalid_argument if the structure is null, an atom index is
   *         missing, or a selected basis is unavailable
   */
  static std::shared_ptr<AuxiliaryBasis> from_index_map(
      const std::map<size_t, std::string>& index_to_basis_map,
      std::shared_ptr<Structure> structure,
      AOType atomic_orbital_type = AOType::Spherical);

  /** @brief Get the stable data-class type name. @return @c auxiliary_basis. */
  static std::string data_type_name() {
    return DATACLASS_TO_SNAKE_CASE(AuxiliaryBasis);
  }

  /** @brief Get the data-class type name for this instance. */
  std::string get_data_type_name() const override { return data_type_name(); }

  /** @brief Get a human-readable summary. @return Summary string. */
  std::string get_summary() const override;

  /**
   * @brief Save to a supported file format.
   * @param filename Destination filename
   * @param type File type, @c json or @c hdf5
   * @throws std::runtime_error if the type is unsupported or writing fails
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /** @brief Serialize to JSON. @return Complete JSON representation. */
  nlohmann::json to_json() const override;

  /** @brief Save to JSON. @param filename Destination filename. */
  void to_json_file(const std::string& filename) const override;

  /** @brief Serialize into an HDF5 group. @param group Destination group. */
  void to_hdf5(H5::Group& group) const override;

  /** @brief Save to HDF5. @param filename Destination filename. */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load from a supported file format.
   * @param filename Source filename
   * @param type File type, @c json or @c hdf5
   * @return Loaded auxiliary basis
   */
  static std::shared_ptr<AuxiliaryBasis> from_file(const std::string& filename,
                                                   const std::string& type);

  /** @brief Deserialize from JSON. @param json Serialized data. @return Basis.
   */
  static std::shared_ptr<AuxiliaryBasis> from_json(const nlohmann::json& json);

  /** @brief Load from JSON. @param filename Source filename. @return Basis. */
  static std::shared_ptr<AuxiliaryBasis> from_json_file(
      const std::string& filename);

  /** @brief Load from an HDF5 group. @param group Source group. @return Basis.
   */
  static std::shared_ptr<AuxiliaryBasis> from_hdf5(H5::Group& group);

  /** @brief Load from HDF5. @param filename Source filename. @return Basis. */
  static std::shared_ptr<AuxiliaryBasis> from_hdf5_file(
      const std::string& filename);

 private:
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  std::string _name;
  AOType _atomic_orbital_type;
  std::shared_ptr<Structure> _structure;
  std::vector<std::vector<Shell>> _shells_per_atom;

  /** @brief Validate required state after construction or deserialization. */
  void _validate() const;

  /** @brief Feed all identifying fields into the content hash. */
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;
};

/**
 * @class AuxiliaryBasisCollection
 * @brief Immutable collection of auxiliary bases keyed by algorithm-facing
 *        roles
 *
 * The collection is independent of a primary @ref BasisSet or wavefunction and
 * is intended for algorithms that explicitly accept auxiliary-basis inputs.
 * It does not itself enable density fitting or change algorithm behavior. All
 * entries must describe the same molecular structure.
 */
class AuxiliaryBasisCollection
    : public DataClass,
      public std::enable_shared_from_this<AuxiliaryBasisCollection> {
 public:
  using Map = std::map<AuxiliaryBasisRole, std::shared_ptr<AuxiliaryBasis>>;

  /**
   * @brief Construct an empty auxiliary-basis collection.
   */
  AuxiliaryBasisCollection() = default;

  /**
   * @brief Construct a collection from exact role associations.
   *
   * Every non-null entry is retained under its exact role. All entries must
   * describe the same molecular structure.
   *
   * @param auxiliary_bases Auxiliary bases keyed by role
   * @throws std::invalid_argument if an entry is null or structures differ
   */
  explicit AuxiliaryBasisCollection(Map auxiliary_bases);

  /**
   * @brief Check whether an exact role has an associated auxiliary basis.
   * @param role Exact role to inspect
   * @return @c true if @p role is present, otherwise @c false
   */
  bool has_auxiliary_basis(AuxiliaryBasisRole role) const;

  /**
   * @brief Get the auxiliary basis associated with an exact role.
   * @param role Exact role to retrieve
   * @return Auxiliary basis associated with @p role
   * @throws std::out_of_range if the role has no association
   */
  std::shared_ptr<AuxiliaryBasis> get_auxiliary_basis(
      AuxiliaryBasisRole role) const;

  /**
   * @brief Resolve a basis compatible with a required role.
   *
   * Exact associations take precedence. A JKFit basis may satisfy a JFit
   * requirement, but the inverse fallback is not permitted.
   *
   * @param role Required auxiliary-basis role
   * @return Exact or compatible auxiliary basis
   * @throws std::out_of_range if no compatible association exists
   */
  std::shared_ptr<AuxiliaryBasis> resolve_auxiliary_basis(
      AuxiliaryBasisRole role) const;

  /**
   * @brief Get all exact role associations.
   * @return Read-only role-to-basis map owned by this collection
   */
  const Map& get_auxiliary_bases() const;

  /**
   * @brief Get the stable data-class type name.
   * @return @c auxiliary_basis_collection
   */
  static std::string data_type_name() {
    return DATACLASS_TO_SNAKE_CASE(AuxiliaryBasisCollection);
  }

  /**
   * @brief Get the data-class type name for this instance.
   * @return @c auxiliary_basis_collection
   */
  std::string get_data_type_name() const override { return data_type_name(); }

  /**
   * @brief Get a human-readable summary of the tagged auxiliary bases.
   * @return Summary containing the role count and each role-to-name mapping
   */
  std::string get_summary() const override;

  /**
   * @brief Save the collection in a supported file format.
   * @param filename Destination filename
   * @param type File type, @c json or @c hdf5
   * @throws std::runtime_error if @p type is unsupported or writing fails
   */
  void to_file(const std::string& filename,
               const std::string& type) const override;

  /**
   * @brief Serialize the collection to JSON.
   * @return JSON object containing the version and exact role associations
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Save the collection to a JSON file.
   * @param filename Destination filename with an
   *        @c .auxiliary_basis_collection.json suffix
   * @throws std::invalid_argument if the filename suffix is invalid
   * @throws std::runtime_error if the file cannot be written
   */
  void to_json_file(const std::string& filename) const override;

  /**
   * @brief Serialize the collection into an HDF5 group.
   * @param group Destination HDF5 group
   * @throws H5::Exception if the group cannot be written
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Save the collection to an HDF5 file.
   * @param filename Destination filename with an
   *        @c .auxiliary_basis_collection.h5 suffix
   * @throws std::invalid_argument if the filename suffix is invalid
   * @throws H5::Exception if the file cannot be written
   */
  void to_hdf5_file(const std::string& filename) const override;

  /**
   * @brief Load a collection from a supported file format.
   * @param filename Source filename
   * @param type File type, @c json or @c hdf5
   * @return Deserialized auxiliary-basis collection
   * @throws std::runtime_error if @p type is unsupported or reading fails
   */
  static std::shared_ptr<AuxiliaryBasisCollection> from_file(
      const std::string& filename, const std::string& type);

  /**
   * @brief Deserialize a collection from JSON.
   * @param json JSON object containing the serialized collection
   * @return Deserialized auxiliary-basis collection
   * @throws std::runtime_error if required fields or the version are invalid
   * @throws std::invalid_argument if an entry is invalid or structures differ
   */
  static std::shared_ptr<AuxiliaryBasisCollection> from_json(
      const nlohmann::json& json);

  /**
   * @brief Load a collection from a JSON file.
   * @param filename Source filename with an
   *        @c .auxiliary_basis_collection.json suffix
   * @return Deserialized auxiliary-basis collection
   * @throws std::invalid_argument if the filename suffix or an entry is invalid
   * @throws std::runtime_error if the file cannot be read
   */
  static std::shared_ptr<AuxiliaryBasisCollection> from_json_file(
      const std::string& filename);

  /**
   * @brief Deserialize a collection from an HDF5 group.
   * @param group Source HDF5 group
   * @return Deserialized auxiliary-basis collection
   * @throws H5::Exception if the serialized attribute cannot be read
   * @throws std::runtime_error if the stored JSON is invalid
   */
  static std::shared_ptr<AuxiliaryBasisCollection> from_hdf5(H5::Group& group);

  /**
   * @brief Load a collection from an HDF5 file.
   * @param filename Source filename with an
   *        @c .auxiliary_basis_collection.h5 suffix
   * @return Deserialized auxiliary-basis collection
   * @throws std::invalid_argument if the filename suffix is invalid
   * @throws H5::Exception if the file or collection group cannot be read
   */
  static std::shared_ptr<AuxiliaryBasisCollection> from_hdf5_file(
      const std::string& filename);

 private:
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";

  Map _auxiliary_bases;

  /**
   * @brief Validate collection entries and their shared molecular structure.
   * @throws std::invalid_argument if an entry is null or structures differ
   */
  void _validate() const;

  /**
   * @brief Feed exact role associations into a content-hash context.
   * @param ctx Hash context to update
   */
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;
};

/**
 * @brief Return a new collection carrying an auxiliary basis under @p role.
 *
 * The input collection is unchanged. An existing association for the same
 * role is replaced in the returned value.
 *
 * @param collection Source collection
 * @param role Exact role to add or replace
 * @param auxiliary_basis Auxiliary basis to associate with @p role
 * @return New collection containing the requested association
 * @throws std::invalid_argument if @p auxiliary_basis is null or its structure
 *         differs from another collection entry
 */
std::shared_ptr<AuxiliaryBasisCollection> with_auxiliary_basis(
    const AuxiliaryBasisCollection& collection, AuxiliaryBasisRole role,
    std::shared_ptr<AuxiliaryBasis> auxiliary_basis);

static_assert(DataClassCompliant<AuxiliaryBasis>,
              "AuxiliaryBasis must implement the complete DataClass interface");
static_assert(
    DataClassCompliant<AuxiliaryBasisCollection>,
    "AuxiliaryBasisCollection must implement the complete DataClass interface");

}  // namespace qdk::chemistry::data

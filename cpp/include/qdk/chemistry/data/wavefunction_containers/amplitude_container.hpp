/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <complex>
#include <memory>
#include <nlohmann/json.hpp>
#include <optional>
#include <qdk/chemistry/data/configuration.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <tuple>
#include <variant>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @enum AmplitudeType
 * @brief Identifies the correlated method that produced an amplitude
 *        wavefunction.
 *
 * Because all amplitude-based wavefunctions share a single container type,
 * this tag lets downstream consumers know how to interpret the stored
 * amplitudes (for example, first-order perturbative doubles versus an
 * exponential coupled-cluster ansatz).
 */
enum class AmplitudeType {
  MollerPlesset,   ///< Moller-Plesset perturbation theory
  CoupledCluster,  ///< Coupled cluster theory
  Unspecified,     ///< Producer did not record a type (e.g. legacy data)
};

/**
 * @brief Convert an AmplitudeType to its serialization string.
 * @param type Amplitude type
 * @return Lowercase string identifier ("moller_plesset", "coupled_cluster", or
 * "unspecified")
 */
std::string amplitude_type_to_string(AmplitudeType type);

/**
 * @brief Parse an AmplitudeType from its serialization string.
 * @param s String identifier ("moller_plesset", "coupled_cluster", or
 * "unspecified"; legacy aliases "mp2" and "ccsd" are also accepted)
 * @return Corresponding AmplitudeType; unrecognized strings map to Unspecified
 */
AmplitudeType amplitude_type_from_string(const std::string& s);

/**
 * @class AmplitudeContainer
 * @brief Wavefunction container representing an amplitude-based correlated
 *        wavefunction (e.g. coupled cluster or MP2).
 *
 * This is the single container type for wavefunctions parameterized by
 * excitation amplitudes relative to a reference. It stores the reference
 * wavefunction together with T1/T2 amplitude blocks and subsumes the previous
 * coupled-cluster and MP2 containers.
 *
 * The container is pure storage: it does not expand the amplitudes into a
 * determinant/coefficient (CI) representation and does not compute reduced
 * density matrices. Determinant- and RDM-based accessors therefore throw.
 * Amplitudes are supplied by the producing algorithm; they are not computed
 * lazily.
 */
class AmplitudeContainer : public WavefunctionContainer {
 public:
  using MatrixVariant = ContainerTypes::MatrixVariant;
  using VectorVariant = ContainerTypes::VectorVariant;
  using ScalarVariant = ContainerTypes::ScalarVariant;
  using DeterminantVector = ContainerTypes::DeterminantVector;

  /**
   * @brief Constructs an amplitude wavefunction with spatial (restricted)
   *        amplitudes.
   *
   * T1/T2 amplitudes are stored if provided.
   *
   * @param orbitals Shared pointer to orbitals
   * @param sector Name of the single-particle sector the orbitals belong to
   * @param wavefunction Shared pointer to the reference wavefunction
   * @param amplitude_type Correlated method that produced the amplitudes
   * @param t1_amplitudes T1 amplitudes (optional)
   * @param t2_amplitudes T2 amplitudes (optional)
   */
  AmplitudeContainer(std::shared_ptr<Orbitals> orbitals,
                     std::shared_ptr<Wavefunction> wavefunction,
                     AmplitudeType amplitude_type,
                     const std::optional<VectorVariant>& t1_amplitudes,
                     const std::optional<VectorVariant>& t2_amplitudes,
                     std::string sector = Wavefunction::DEFAULT_SECTOR);

  /**
   * @brief Constructs an amplitude wavefunction with spin-separated amplitudes.
   *
   * T1/T2 amplitudes are stored if provided.
   *
   * @param orbitals Shared pointer to orbitals
   * @param sector Name of the single-particle sector the orbitals belong to
   * @param wavefunction Shared pointer to the reference wavefunction
   * @param amplitude_type Correlated method that produced the amplitudes
   * @param t1_amplitudes_aa Alpha T1 amplitudes (optional)
   * @param t1_amplitudes_bb Beta T1 amplitudes (optional)
   * @param t2_amplitudes_abab Alpha-beta T2 amplitudes (optional)
   * @param t2_amplitudes_aaaa Alpha-alpha T2 amplitudes (optional)
   * @param t2_amplitudes_bbbb Beta-beta T2 amplitudes (optional)
   */
  AmplitudeContainer(std::shared_ptr<Orbitals> orbitals,
                     std::shared_ptr<Wavefunction> wavefunction,
                     AmplitudeType amplitude_type,
                     const std::optional<VectorVariant>& t1_amplitudes_aa,
                     const std::optional<VectorVariant>& t1_amplitudes_bb,
                     const std::optional<VectorVariant>& t2_amplitudes_abab,
                     const std::optional<VectorVariant>& t2_amplitudes_aaaa,
                     const std::optional<VectorVariant>& t2_amplitudes_bbbb,
                     std::string sector = Wavefunction::DEFAULT_SECTOR);

  /** @brief Destructor */
  ~AmplitudeContainer() override = default;

  /**
   * @brief Create a deep copy of this container
   * @return Unique pointer to cloned container
   */
  std::unique_ptr<WavefunctionContainer> clone() const override;

  /**
   * @brief Get reference to orbitals
   * @return Shared pointer to orbitals
   */
  std::shared_ptr<Orbitals> get_orbitals() const override;

  /**
   * @brief Names of the single-particle sectors this container spans.
   * @return The container's sector names (the sector supplied at construction).
   */
  std::vector<std::string> sectors() const override;

  /**
   * @brief Single-particle basis bound to a sector.
   * @param name Sector name to resolve.
   * @return Shared pointer to the @ref Orbitals bound to @p name.
   * @throws std::out_of_range if this container has no sector named @p name.
   */
  std::shared_ptr<const Orbitals> sector_basis(
      const std::string& name) const override;

  /**
   * @brief Get the reference wavefunction
   * @return Shared pointer to the reference wavefunction
   */
  std::shared_ptr<Wavefunction> get_wavefunction() const;

  /**
   * @brief Get the correlated method that produced these amplitudes.
   * @return The amplitude expansion type (MollerPlesset, CoupledCluster, or
   * Unspecified)
   */
  AmplitudeType get_amplitude_type() const;

  /**
   * @brief Get T1 amplitudes
   *
   * @return Pair of (alpha, beta) T1 amplitudes
   * @throws std::runtime_error if T1 amplitudes are not available
   */
  std::pair<const VectorVariant&, const VectorVariant&> get_t1_amplitudes()
      const;

  /**
   * @brief Get T2 amplitudes
   *
   * @return Tuple of (alpha-beta, alpha-alpha, beta-beta) T2 amplitudes
   * @throws std::runtime_error if T2 amplitudes are not available
   */
  std::tuple<const VectorVariant&, const VectorVariant&, const VectorVariant&>
  get_t2_amplitudes() const;

  /**
   * @brief Check if T1 amplitudes are available
   * @return True if T1 amplitudes are available
   */
  bool has_t1_amplitudes() const;

  /**
   * @brief Check if T2 amplitudes are available
   * @return True if T2 amplitudes are available
   */
  bool has_t2_amplitudes() const;

  /**
   * @brief Not supported for amplitude wavefunctions.
   * @throws std::runtime_error Always.
   */
  ScalarVariant overlap(const WavefunctionContainer& other) const override;

  /**
   * @brief Not supported for amplitude wavefunctions.
   * @throws std::runtime_error Always.
   */
  double norm() const override;

  /**
   * @brief Number of particles (active + inactive) as a symmetry-blocked
   * scalar.
   * @return Shared pointer to the symmetry-blocked total particle count.
   */
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
  total_num_particles() const override;

  /**
   * @brief Number of active-space particles as a symmetry-blocked scalar.
   * @return Shared pointer to the symmetry-blocked active particle count.
   */
  std::shared_ptr<const SymmetryBlockedScalar<std::size_t>>
  active_num_particles() const override;

  /**
   * @brief Not supported: orbital occupations require reduced density
   *        matrices, which are not stored by amplitude wavefunctions.
   * @throws std::runtime_error Always.
   */
  std::shared_ptr<const SymmetryBlockedTensor<1>> total_orbital_occupations()
      const override;

  /**
   * @brief Not supported: orbital occupations require reduced density
   *        matrices, which are not stored by amplitude wavefunctions.
   * @throws std::runtime_error Always.
   */
  std::shared_ptr<const SymmetryBlockedTensor<1>> active_orbital_occupations()
      const override;

  /**
   * @brief Check if a determinant is in the reference wavefunction
   * @param det Configuration/determinant to check
   * @return True if determinant matches any reference determinant
   */
  bool contains_determinant(const Configuration& det) const;

  /**
   * @brief Check if a determinant is in the reference wavefunction
   * @param det Configuration/determinant to check
   * @return True if determinant matches any reference determinant
   */
  bool contains_reference(const Configuration& det) const;

  /**
   * @brief Clear cached data to release memory
   */
  void clear_caches() const override;

  /**
   * @brief Convert container to JSON format
   * @return JSON object containing container data
   */
  nlohmann::json to_json() const override;

  /**
   * @brief Load container from JSON format
   *
   * Reads the current "amplitude" format as well as the legacy
   * "coupled_cluster" and "mp2" formats (read-only backward compatibility).
   * Legacy "mp2" files did not store amplitudes, so the resulting container
   * has none.
   *
   * @param j JSON object containing container data
   * @return Unique pointer to the container created from JSON data
   * @throws std::runtime_error if JSON is malformed
   */
  static std::unique_ptr<AmplitudeContainer> from_json(const nlohmann::json& j);

  /**
   * @brief Convert container to HDF5 group
   * @param group HDF5 group to write container data to
   * @throws std::runtime_error if HDF5 I/O error occurs
   */
  void to_hdf5(H5::Group& group) const override;

  /**
   * @brief Load container from HDF5 group
   *
   * Reads the current "amplitude" format as well as the legacy
   * "coupled_cluster" and "mp2" formats (read-only backward compatibility).
   * Legacy "mp2" files did not store amplitudes, so the resulting container
   * has none.
   *
   * @param group HDF5 group containing container data
   * @return Unique pointer to the container created from HDF5 group
   * @throws std::runtime_error if HDF5 data is malformed or I/O error occurs
   */
  static std::unique_ptr<AmplitudeContainer> from_hdf5(H5::Group& group);

  /**
   * @brief Get container type identifier for serialization
   * @return String "amplitude"
   */
  std::string get_container_type() const override;

  /**
   * @brief Check if the wavefunction is complex-valued
   * @return True if amplitudes contain complex values
   */
  bool is_complex() const override;

  /**
   * @brief Feed identifying data into a hash context.
   */
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

 private:
  // Orbital information
  std::shared_ptr<Orbitals> _orbitals;
  // Single-particle sector the orbitals belong to
  std::string _sector;
  // Reference wavefunction
  std::shared_ptr<Wavefunction> _wavefunction;
  // Correlated method that produced the amplitudes
  AmplitudeType _amplitude_type;

  std::shared_ptr<VectorVariant> _t1_amplitudes_aa = nullptr;
  std::shared_ptr<VectorVariant> _t1_amplitudes_bb = nullptr;

  std::shared_ptr<VectorVariant> _t2_amplitudes_abab = nullptr;
  std::shared_ptr<VectorVariant> _t2_amplitudes_aaaa = nullptr;
  std::shared_ptr<VectorVariant> _t2_amplitudes_bbbb = nullptr;

  /// Serialization version
  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};
}  // namespace qdk::chemistry::data

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <H5Cpp.h>

#include <Eigen/Dense>
#include <memory>
#include <nlohmann/json.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <stdexcept>
#include <string>

namespace qdk::chemistry::data {

/**
 * @class FactorizedHamiltonianContainer
 * @brief Restricted, spin-free double-factorized THC Hamiltonian container.
 *
 * Stores factorized matrices and metadata. See Low et al.,
 * arXiv:2502.15882.
 */
class FactorizedHamiltonianContainer : public HamiltonianContainer {
 public:
  /**
   * @brief Construct a restricted factorized Hamiltonian.
   *
   * @param core_energy Nuclear and inactive-core energy.
   * @param u_matrices U factors, flattened as [R,B,N].
   * @param w_matrices W factors, flattened as [R,B,C].
   * @param wb_matrix Identity weights [R,C].
   * @param one_body_integrals One-body integrals [N,N].
   * @param inactive_fock_matrix Inactive Fock matrix.
   * @param orbitals Orbitals with an active space.
   * @param bliss_shift BLISS core shift.
   * @param energy_gap E_gap for SOS block encoding.
   * @param type Hamiltonian type.
   * @throws std::invalid_argument if dimensions or required data are invalid.
   */
  FactorizedHamiltonianContainer(
      double core_energy, const Eigen::VectorXd& u_matrices,
      const Eigen::VectorXd& w_matrices, const Eigen::MatrixXd& wb_matrix,
      const Eigen::MatrixXd& one_body_integrals,
      const Eigen::MatrixXd& inactive_fock_matrix,
      std::shared_ptr<Orbitals> orbitals, double bliss_shift = 0.0,
      double energy_gap = 0.0,
      HamiltonianType type = HamiltonianType::Hermitian);

  /** @brief Destructor. */
  ~FactorizedHamiltonianContainer() override = default;

  /** @brief Create a deep copy. */
  std::unique_ptr<HamiltonianContainer> clone() const override final;

  /** @return @c "factorized". */
  std::string get_container_type() const override final;

  /**
   * @brief Get reconstructed two-body integrals.
   * @return The same cached vector for the aaaa, aabb, and bbbb channels.
   * @throws std::runtime_error if U or W is empty.
   */
  std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
             const Eigen::VectorXd&>
  get_two_body_integrals() const override;

  /**
   * @brief Get a reconstructed two-body element.
   * @param i First orbital index.
   * @param j Second orbital index.
   * @param k Third orbital index.
   * @param l Fourth orbital index.
   * @param channel Ignored; the integrals are restricted.
   * @return Four-center two-electron integral (ij|kl)
   * @throws std::runtime_error if U or W is empty.
   * @throws std::out_of_range if an index is outside [0,N).
   */
  double get_two_body_element(
      unsigned i, unsigned j, unsigned k, unsigned l,
      SpinChannel channel = SpinChannel::aaaa) const override;

  /** @return Whether U and W are nonempty. */
  bool has_two_body_integrals() const override;

  /** @return Always true. */
  bool is_restricted() const override final;

  /** @return Whether required data and dimensions are valid. */
  bool is_valid() const override final;

  /** @brief Serialize to JSON. */
  nlohmann::json to_json() const override final;

  /** @brief Serialize into an HDF5 group. */
  void to_hdf5(H5::Group& group) const override final;

  /**
   * @brief Deserialize from JSON.
   * @param j Serialized data.
   * @return The reconstructed container.
   */
  static std::unique_ptr<FactorizedHamiltonianContainer> from_json(
      const nlohmann::json& j);

  /**
   * @brief Deserialize from HDF5.
   * @param group Serialized data.
   * @return The reconstructed container.
   */
  static std::unique_ptr<FactorizedHamiltonianContainer> from_hdf5(
      H5::Group& group);

  /** @return U flattened in [R,B,N] order. */
  const Eigen::VectorXd& get_u_matrices() const;

  /** @return W flattened in [R,B,C] order. */
  const Eigen::VectorXd& get_w_matrices() const;

  /** @return Identity weights WB with shape [R,C]. */
  const Eigen::MatrixXd& get_wb_matrix() const;

  /** @return Number N of active spatial orbitals. */
  size_t get_num_orbitals() const;

  /** @return Number of ranks R, inferred from the WB rows. */
  size_t get_num_ranks() const;

  /** @return Number of bases B, inferred from the U length. */
  size_t get_num_bases() const;

  /** @return Number of copies C, inferred from the WB columns. */
  size_t get_num_copies() const;

  /** @return BLISS energy shift. */
  double get_bliss_shift() const;

  /** @return Energy gap E_gap for SOS block encoding. */
  double get_energy_gap() const;

  /**
   * @brief Compute the block-encoding normalization (Eq. 33).
   * Λ = Σ|eig(h1_majorana)| + 1/4 Σ_{rc} (|WB^{rc}| + Σ_b |W^{rc}_b|)²
   */
  double get_lambda() const;

  /**
   * @brief Compute the effective SOS normalization (Eq. 11).
   * λ_eff = √(E_gap · (2Λ - E_gap))
   * @throws std::runtime_error if E_gap is non-positive or >= 2Λ.
   */
  double get_lambda_eff() const;

  /**
   * @brief Compute the adjusted Majorana one-body matrix (Eq. 36).
   *
   * Writing the rank-r copy-c leaf as
   *   M^{rc}_{pq} = Σ_{b∈[B]} W^{rc}_b U^r_{bp} U^r_{bq},
   * this accumulates three corrections:
   *   h'(1)_{pq} = h1_{pq} - ½ Σ_{rc} (M^{rc} M^{rc})_{pq}
   *                        + Σ_{rc} tr(M^{rc}) M^{rc}_{pq}
   *                        - Σ_{rc} WB^{rc} M^{rc}_{pq}
   *
   * The leading -½ (M M) term has no counterpart in Eq. 36 as printed: the
   * paper writes the two-body operator as a plain product while this container
   * stores h2 = (pq|rs) normal-ordered, and unpacking that difference leaves
   * exactly -½ Σ_s h2_{pssq}. It is required, not optional -- see the
   * derivation in the implementation.
   *
   * @return The [N,N] matrix, contracted directly from the factors.
   */
  Eigen::MatrixXd get_h1_majorana() const;

  /**
   * @brief Reconstruct the approximate two-body integrals.
   * h2_{pqrs} = Σ_{r,c} (Σ_b U^r_{bp} U^r_{bq} W^r_{bc})
   *                      (Σ_{b'} U^r_{b'r} U^r_{b's} W^r_{b'c})
   *
   * Note this is built purely from (U, W): the identity weight WB is
   * deliberately absent, matching Eq. 24. WB enters only get_h1_majorana() and
   * get_lambda(), so it is a gauge parameter for the two-body tensor rather
   * than unused data.
   *
   * @return A flat N^4 vector in [p,q,r,s] order.
   */
  Eigen::VectorXd reconstruct_two_body_integrals() const;

 private:
  /** @brief Add all serialized state to a hash. */
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  /** @throws std::invalid_argument if U, W, or WB dimensions are invalid. */
  void validate_integral_dimensions() const override final;

  Eigen::VectorXd _u;   ///< Flat U matrices [R*B*N]
  Eigen::VectorXd _w;   ///< Flat W matrices [R*B*C]
  Eigen::MatrixXd _wb;  ///< Identity weights [R*C]

  // TODO: add the full bliss object for one-body/two-body shifts.
  double _bliss_shift;  ///< BLISS energy shift
  double _energy_gap;   ///< E_gap for SOS block encoding

  /// Lazily computed four-center integrals (shared for all channels,
  /// restricted)
  mutable std::shared_ptr<Eigen::VectorXd> _cached_two_body;

  /** @brief Reconstruct the shared two-body cache. */
  void _build_two_body_cache() const;

  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

}  // namespace qdk::chemistry::data

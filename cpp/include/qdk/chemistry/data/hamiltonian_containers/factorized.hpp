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
 * @brief Restricted, spin-free double-factorized tensor hypercontraction(DFTHC)
 *        Hamiltonian container :cite:`Low2025`.
 *
 * @note The two-body tensor is stored as a plain sum of squares, so it is
 * positive semi-definite by construction.
 */
class FactorizedHamiltonianContainer : public HamiltonianContainer {
 public:
  /**
   * @brief Construct a restricted factorized Hamiltonian.
   *
   * @param one_body_integrals One-body integrals [N,N] over the N active
   *        spatial orbitals.
   * @param u_matrices U factors, flattened as [R,B,N].
   * @param w_matrices W factors, flattened as [R,B,C].
   * @param wb_matrix Identity weights [R,C].
   * @param orbitals Orbitals with an active space.
   * @param core_energy Nuclear and inactive-core energy.
   * @param inactive_fock_matrix Inactive Fock matrix with full
   *        molecular-orbital dimensions.
   * @param energy_gap Energy-gap metadata.
   * @param type Hamiltonian type.
   * @throws std::invalid_argument if dimensions or required data are invalid,
   *         or if a basis row of U is not normalized.
   */
  FactorizedHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals,
      const Eigen::VectorXd& u_matrices, const Eigen::VectorXd& w_matrices,
      const Eigen::MatrixXd& wb_matrix, std::shared_ptr<Orbitals> orbitals,
      double core_energy, const Eigen::MatrixXd& inactive_fock_matrix,
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

  /** @return Identity weights w_B with shape [R,C]. */
  const Eigen::MatrixXd& get_wb_matrix() const;

  /** @return Number N of active spatial orbitals. */
  size_t get_num_orbitals() const;

  /** @return Number of ranks R, inferred from the WB rows. */
  size_t get_num_ranks() const;

  /** @return Number of bases B, inferred from the U length. */
  size_t get_num_bases() const;

  /** @return Number of copies C, inferred from the WB columns. */
  size_t get_num_copies() const;

  /** @return Energy-gap metadata. */
  double get_energy_gap() const;

  /**
   * @brief Compute the factorization normalization (Eq. 33).
   * Λ = Σ|eig(h1_prime)| + 1/4 Σ_{rc} (|WB^{rc}| + Σ_b |W^{rc}_b|)²
   *
   * @throws std::runtime_error if the adjusted one-body matrix is not
   *         symmetric.
   */
  double get_lambda() const;

  /**
   * @brief Compute the effective normalization (Eq. 11).
   * λ_eff = √(E_gap · (2Λ - E_gap))
   *
   * @return The effective normalization, or 0.0 if E_gap is outside the open
   *         interval (0, 2Λ).
   */
  double get_lambda_eff() const;

  /**
   * @brief Compute the adjusted one-body matrix h'(1) (Eq. 36).
   *
   * Writing the rank-r copy-c leaf as
   *   M^{rc}_{pq} = Σ_{b∈[B]} W^{rc}_b U^r_{bp} U^r_{bq},
   * this accumulates three corrections:
   *   h'(1)_{pq} = h1_{pq} - ½ Σ_{rc} (M^{rc} M^{rc})_{pq}
   *                        + Σ_{rc} tr(M^{rc}) M^{rc}_{pq}
   *                        - Σ_{rc} WB^{rc} M^{rc}_{pq}
   *
   * @return The [N,N] matrix, contracted directly from the factors.
   */
  Eigen::MatrixXd get_h1_prime() const;

  /**
   * @brief Reconstruct the approximate two-body integrals.
   *
   * With t the rank index (r and s here are orbital indices):
   * h2_{pqrs} = Σ_{t,c} (Σ_b U^t_{bp} U^t_{bq} W^t_{bc})
   *                      (Σ_{b'} U^t_{b'r} U^t_{b's} W^t_{b'c})
   *
   * Note this is built purely from (U, W).
   *
   * @return A flat N^4 vector in [p,q,r,s] order.
   */
  Eigen::VectorXd reconstruct_two_body_integrals() const;

 private:
  /** @brief Add all serialized state to a hash. */
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;

  /**
   * @throws std::invalid_argument if U, W or WB dimensions are invalid.
   */
  void validate_integral_dimensions() const override final;

  Eigen::VectorXd _u;   ///< Flat U matrices [R*B*N]
  Eigen::VectorXd _w;   ///< Flat W matrices [R*B*C]
  Eigen::MatrixXd _wb;  ///< Identity weights [R,C]

  double _energy_gap;  ///< Energy-gap metadata

  /// Lazily computed four-center integrals (shared for all channels,
  /// restricted)
  mutable std::shared_ptr<Eigen::VectorXd> _cached_two_body;

  /** @brief Reconstruct the shared two-body cache. */
  void _build_two_body_cache() const;

  /** Serialization version */
  static constexpr const char* SERIALIZATION_VERSION = "0.2.0";
};

}  // namespace qdk::chemistry::data

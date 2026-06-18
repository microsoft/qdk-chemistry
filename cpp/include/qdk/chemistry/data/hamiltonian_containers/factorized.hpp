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
 * @brief Hamiltonian container for Double-Factorized Tensor Hypercontraction.
 *
 * Stores the factorized two-body integrals:
 *
 *   h2_{pqrs} ≈ Σ_{r,c} (Σ_b U^r_{bp} U^r_{bq} W^r_{bc})
 *                        (Σ_{b'} U^r_{b'r} U^r_{b's} W^r_{b'c})
 *
 * along with an identity weight matrix WB[R,C] and optional BLISS
 * core energy shift.
 *
 * This container is always restricted (uses spin-free integrals).
 *
 * Additional metadata for the SOS spectrum-amplified block encoding:
 * - energy_gap: E_gap = E_gs - E_SOS - E_nuc (requires external E_gs)
 *
 * Reference: Low et al., arXiv:2502.15882
 */
class FactorizedHamiltonianContainer : public HamiltonianContainer {
 public:
  /**
   * @brief Constructor for restricted factorized Hamiltonian.
   * @param one_body_integrals One-body integrals in MO basis [N x N].
   * @param u_matrices Orbital rotation matrices, flat [R*B*N].
   * @param w_matrices Two-body weights, flat [R*B*C].
   * @param wb_matrix Identity weights [R x C].
   * @param num_ranks R rank.
   * @param num_bases B rank.
   * @param num_copies C rank.
   * @param orbitals Orbital data with active space set.
   * @param core_energy Nuclear repulsion + inactive core energy.
   * @param inactive_fock_matrix Inactive Fock matrix.
   * @param bliss_core_shift BLISS core energy shift (default 0).
   * @param energy_gap E_gap for SOS block encoding (default 0).
   */
  FactorizedHamiltonianContainer(
      const Eigen::MatrixXd& one_body_integrals,
      const Eigen::VectorXd& u_matrices, const Eigen::VectorXd& w_matrices,
      const Eigen::MatrixXd& wb_matrix, size_t num_ranks, size_t num_bases,
      size_t num_copies, std::shared_ptr<Orbitals> orbitals,
      double core_energy, const Eigen::MatrixXd& inactive_fock_matrix,
      double bliss_core_shift = 0.0, double energy_gap = 0.0,
      HamiltonianType type = HamiltonianType::Hermitian);

  ~FactorizedHamiltonianContainer() override = default;

  // === HamiltonianContainer overrides ===

  std::unique_ptr<HamiltonianContainer> clone() const override final;
  std::string get_container_type() const override final;

  std::tuple<const Eigen::VectorXd&, const Eigen::VectorXd&,
             const Eigen::VectorXd&>
  get_two_body_integrals() const override;

  double get_two_body_element(
      unsigned i, unsigned j, unsigned k, unsigned l,
      SpinChannel channel = SpinChannel::aaaa) const override;

  bool has_two_body_integrals() const override;
  bool is_restricted() const override final;
  bool is_valid() const override final;

  nlohmann::json to_json() const override final;
  void to_hdf5(H5::Group& group) const override final;

  static std::unique_ptr<FactorizedHamiltonianContainer> from_json(
      const nlohmann::json& j);
  static std::unique_ptr<FactorizedHamiltonianContainer> from_hdf5(
      H5::Group& group);

  // === Factorized-specific accessors ===

  /** @brief Get U matrices as flat vector [R*B*N]. */
  const Eigen::VectorXd& get_u_matrices() const;

  /** @brief Get W matrices as flat vector [R*B*C]. */
  const Eigen::VectorXd& get_w_matrices() const;

  /** @brief Get WB matrix [R x C]. */
  const Eigen::MatrixXd& get_wb_matrix() const;

  /** @brief Number of spatial orbitals (N). */
  size_t get_num_orbitals() const;

  /** @brief Number of ranks (R). */
  size_t get_num_ranks() const;

  /** @brief Number of bases per rank (B). */
  size_t get_num_bases() const;

  /** @brief Number of copies per rank (C). */
  size_t get_num_copies() const;

  /** @brief BLISS core energy shift. */
  double get_bliss_core_shift() const;

  /** @brief Energy gap E_gap for SOS block encoding. */
  double get_energy_gap() const;

  /**
   * @brief Block-encoding normalization Λ.
   *
   * Λ = Σ|eig(h1_majorana)| + ¼ Σ_{rc} (|WB^{rc}| + Σ_b |W^{rc}_b|)²
   *
   * Computed on demand from stored data.
   */
  double get_lambda() const;

  /**
   * @brief Effective lambda for SOS walk.
   *
   * λ_eff = √(E_gap · (2Λ - E_gap))
   *
   * Requires E_gap > 0 and E_gap < 2Λ.
   * @throws std::runtime_error if E_gap is non-positive or >= 2Λ.
   */
  double get_lambda_eff() const;

  /**
   * @brief Adjusted one-body matrix in Majorana basis h'(1).
   *
   * h'(1)_{pq} = h1_{pq} - ½ Σ_{rs} h2_{prrs→pq}
   *              + Σ_{rs} h2_{pqrr}
   *              - Σ_{rc,b} WB^{rc} W^{rc}_b U^r_{bp} U^r_{bq}
   *
   * Computed on demand.
   */
  Eigen::MatrixXd get_h1_majorana() const;

  /**
   * @brief Reconstruct approximate two-body integrals from factorization.
   *
   * h2_{pqrs} = Σ_{r,c} (Σ_b U^r_{bp} U^r_{bq} W^r_{bc})
   *                      (Σ_{b'} U^r_{b'r} U^r_{b's} W^r_{b'c})
   */
  Eigen::VectorXd reconstruct_two_body_integrals() const;

 private:
  void hash_update(qdk::chemistry::utils::HashContext& ctx) const override;
  void validate_integral_dimensions() const override final;

  Eigen::VectorXd _u;   ///< Flat U matrices [R*B*N]
  Eigen::VectorXd _w;   ///< Flat W matrices [R*B*C]
  Eigen::MatrixXd _wb;  ///< Identity weights [R x C]

  size_t _num_ranks;   ///< R
  size_t _num_bases;   ///< B
  size_t _num_copies;  ///< C

  double _bliss_core_shift;  ///< BLISS core energy shift
  double _energy_gap;        ///< E_gap for SOS block encoding

  /// Lazily computed four-center integrals (shared for all channels, restricted)
  mutable std::shared_ptr<Eigen::VectorXd> _cached_two_body;

  void _build_two_body_cache() const;

  static constexpr const char* SERIALIZATION_VERSION = "0.1.0";
};

}  // namespace qdk::chemistry::data

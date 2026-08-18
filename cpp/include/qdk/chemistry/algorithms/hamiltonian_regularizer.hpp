// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <Eigen/Dense>
#include <memory>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <string>

namespace qdk::chemistry::algorithms {

/**
 * @file
 * @brief Block-invariant symmetry shift (BLISS) Hamiltonian regularization.
 *
 * References:
 * - [1] I. Loaiza and A. F. Izmaylov, "Block-Invariant Symmetry Shift:
 *   Preprocessing technique for second-quantized Hamiltonians to improve
 *   their decompositions to Linear Combination of Unitaries",
 *   arXiv:2304.13772. (Introduces BLISS.)
 * - [2] S. Patel, A. S. Brahmachari, J. T. Cantin, L. Wang and A. F.
 *   Izmaylov, "Global Minimization of Electronic Hamiltonian 1-Norm via
 *   Linear Programming in the Block Invariant Symmetry Shift (BLISS)
 *   Method", arXiv:2409.18277. (Fermionic low-rank BLISS used here.)
 */

/**
 * @struct BlissShift
 * @brief The block-invariant symmetry shift (BLISS) parameters [1].
 *
 * Bundles the parameters of the BLISS operator subtracted from a
 * Hamiltonian H:
 *   K = mu1*(N - Ne) + mu2*(N^2 - Ne^2) + (N - Ne)*sum_ij xi_ij E_ij
 * K annihilates every Ne-electron state, so subtracting it leaves the
 * Ne-sector energy invariant while reducing the fermionic LCU 1-norm.
 *
 * A BlissShift carries only the *result* of a shift computation, so it may
 * come from any method (see BlissSettings' "shift_method") or an external
 * source, and is applied via rebuild_bliss_shifted_hamiltonian().
 */
struct BlissShift {
  double mu1 = 0.0;    ///< One-electron BLISS shift.
  double mu2 = 0.0;    ///< Two-electron BLISS shift.
  Eigen::MatrixXd xi;  ///< Two-electron BLISS shift matrix (norb x norb).
};

/// The two-body correction that the aggregated BLISS shift (mu2, xi) adds to
/// the two-electron integral tensor g_ijkl:
///
///   dg_ijkl = -2*mu2*delta_ij*delta_kl - xi_ij*delta_kl - delta_ij*xi_kl
///
/// Single source of truth for dg: rebuild_bliss_shifted_hamiltonian() folds
/// the full tensor in via add_two_body_correction(), while a one-electron
/// shift solver only needs dg's Coulomb/exchange contractions below. Deriving
/// both from this struct keeps them from silently drifting apart.
struct TwoBodyBlissCorrection {
  double mu2;
  Eigen::MatrixXd xi;

  /// Fold the Coulomb-type contraction of dg into `coulomb` in place, so on
  /// return it holds coul(g~) = coul(g) + Sum_k dg_ijkk, where
  ///   Sum_k dg_ijkk = -(2*mu2*norb + tr(xi))*delta_ij - norb*xi_ij.
  /// `norb` is taken from the caller's matrix; nothing is allocated.
  void add_coulomb_contraction(Eigen::MatrixXd& coulomb) const {
    const double norb = static_cast<double>(coulomb.rows());
    coulomb -= norb * xi;
    coulomb.diagonal().array() -= 2.0 * mu2 * norb + xi.trace();
  }

  /// Fold the exchange-type contraction of dg into `exchange` in place, so on
  /// return it holds exch(g~) = exch(g) + Sum_k dg_ikkj, where
  ///   Sum_k dg_ikkj = -2*mu2*delta_ij - 2*xi_ij.
  /// Nothing is allocated.
  void add_exchange_contraction(Eigen::MatrixXd& exchange) const {
    exchange -= 2.0 * xi;
    exchange.diagonal().array() -= 2.0 * mu2;
  }

  /// Fold dg onto `g` in place, so on return `g` holds g~ = g + dg. `g` is the
  /// flattened tensor ((i*norb+j)*norb+k)*norb+l with side length `norb`; no
  /// separate O(norb^4) dg tensor is materialized.
  ///
  /// Each term of dg is delta-supported, so only the non-zero blocks are
  /// touched: O(norb^2) for the mu2 term and O(norb^3) for the two xi terms.
  void add_two_body_correction(Eigen::VectorXd& g, Eigen::Index norb) const {
    const auto index = [norb](Eigen::Index i, Eigen::Index j, Eigen::Index k,
                              Eigen::Index l) {
      return ((i * norb + j) * norb + k) * norb + l;
    };

    // Term 1: -2*mu2 * delta_ij * delta_kl  (i==j and k==l).
    for (Eigen::Index i = 0; i < norb; ++i) {
      for (Eigen::Index k = 0; k < norb; ++k) {
        g[index(i, i, k, k)] -= 2.0 * mu2;
      }
    }

    // Term 2: -xi_ij * delta_kl  (k==l, all i,j).
    for (Eigen::Index i = 0; i < norb; ++i) {
      for (Eigen::Index j = 0; j < norb; ++j) {
        const double xi_ij = xi(i, j);
        for (Eigen::Index k = 0; k < norb; ++k) {
          g[index(i, j, k, k)] -= xi_ij;
        }
      }
    }

    // Term 3: -delta_ij * xi_kl  (i==j, all k,l).
    for (Eigen::Index i = 0; i < norb; ++i) {
      for (Eigen::Index k = 0; k < norb; ++k) {
        for (Eigen::Index l = 0; l < norb; ++l) {
          g[index(i, i, k, l)] -= xi(k, l);
        }
      }
    }
  }
};

/**
 * @brief Apply a BLISS shift to a Hamiltonian and assemble the shifted one.
 *
 * Applies the global BLISS shift (mu1, mu2, xi) [1,2] to the dense integrals
 * of `original`. In this container's chemist convention g[i,j,k,l] = (ij|kl),
 * subtracting K (see BlissShift) expands to
 *   h~_ij   = h_ij + (Ne - 1)*xi_ij - (mu1 + mu2)*delta_ij
 *   g~_ijkl = g_ijkl - 2*mu2*delta_ij*delta_kl
 *                    - xi_ij*delta_kl - delta_ij*xi_kl
 *   E_core' = E_core + mu1*Ne + mu2*Ne^2
 * so the Ne-sector energy is invariant for any (mu1, mu2, xi).
 *
 * How `shift` was computed is irrelevant: it may come from
 * BlissRegularizer::compute_shift() or any external source. Everything else
 * (integrals, core energy, orbitals, inactive Fock matrix, Hamiltonian type)
 * is read from `original`.
 *
 * @param original The Hamiltonian being shifted. Must be restricted.
 * @param shift The BLISS shift parameters (mu1, mu2, xi) to apply.
 * @param input_num_electrons Target number of active electrons (Ne); the
 *        invariance guarantee only holds for an integer electron count.
 * @return The BLISS-shifted Hamiltonian.
 *
 * @throws std::invalid_argument if `original` is unrestricted or `shift.xi`
 *         is not norb x norb.
 */
std::shared_ptr<data::Hamiltonian> rebuild_bliss_shifted_hamiltonian(
    const data::Hamiltonian& original, const BlissShift& shift,
    unsigned int input_num_electrons);

/**
 * @class BlissSettings
 * @brief Settings container for the BLISS Hamiltonian regularizer.
 *
 * Default settings:
 * - shift_method: "fermionic_low_rank" - how the shift (mu1, mu2, xi) is
 *   computed, via the fermionic-low-rank BLISS method of [2].
 * - df_truncation_threshold: 0.0 - drop double-factorization fragments whose
 *   eigenvalue magnitude is below this threshold. The default of 0.0 performs
 *   no truncation (exact double factorization).
 */
class BlissSettings : public qdk::chemistry::data::Settings {
 public:
  BlissSettings() {
    set_default("shift_method", std::string("fermionic_low_rank"),
                "Method used to compute the BLISS shift (mu1, mu2, xi).",
                data::ListConstraint<std::string>{
                    {std::vector<std::string>{"fermionic_low_rank"}}});
    set_default("df_truncation_threshold", 0.0);
  }
};

/**
 * @class BlissRegularizer
 * @brief Hamiltonian regularizer implementing block-invariant symmetry
 *        shifts [1,2].
 *
 * Maps a Hamiltonian and a target alpha/beta electron count to a new
 * Hamiltonian that is energetically equivalent within that electron-number
 * sector but whose LCU/qubitization coefficients (e.g. the fermionic 1-norm
 * lambda) may be reduced, shrinking resource estimates for algorithms such as
 * qubitized phase estimation.
 *
 * It is a thin composition of two public steps:
 *  1. compute_shift() -- compute (mu1, mu2, xi) via the "shift_method"
 *     setting.
 *  2. rebuild_bliss_shifted_hamiltonian() -- apply that shift.
 * Callers can therefore obtain a BlissShift on its own, or apply an
 * externally computed one directly.
 *
 * Only restricted (spin-restricted) Hamiltonians are currently supported.
 *
 * @see BlissShift
 * @see rebuild_bliss_shifted_hamiltonian
 * @see BlissSettings
 * @see qdk::chemistry::utils::hamiltonian_one_norm to inspect a Hamiltonian's
 *      fermionic 1-norm without running a regularizer.
 */
class BlissRegularizer
    : public Algorithm<BlissRegularizer, std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Hamiltonian>, unsigned int,
                       unsigned int> {
 public:
  /**
   * @brief Default constructor. Uses default BlissSettings.
   */
  BlissRegularizer() { _settings = std::make_unique<BlissSettings>(); }

  /**
   * @brief Virtual destructor.
   */
  ~BlissRegularizer() override = default;

  /**
   * @brief Regularize/shift a Hamiltonian for a target electron count.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param hamiltonian The Hamiltonian to regularize
   * @param n_alpha_electrons The target number of alpha electrons
   * @param n_beta_electrons The target number of beta electrons
   * \endcond
   * @return A new, shifted Hamiltonian that agrees with the input
   *         Hamiltonian's energy in the (n_alpha_electrons,
   *         n_beta_electrons)-electron sector.
   *
   * @note Settings are automatically locked when this method is called.
   */
  using Algorithm::run;

  /**
   * @brief Compute the BLISS shift (mu1, mu2, xi) for a target electron count.
   *
   * Dispatches to the "shift_method" setting and returns the parameters
   * *without* rebuilding the Hamiltonian; apply them with
   * rebuild_bliss_shifted_hamiltonian().
   *
   * @param hamiltonian The Hamiltonian to analyze. Must be restricted.
   * @param n_alpha_electrons The target number of alpha electrons.
   * @param n_beta_electrons The target number of beta electrons.
   * @return The computed BLISS shift parameters.
   *
   * @throws std::invalid_argument if the Hamiltonian is unrestricted or the
   *         configured "shift_method" is unknown.
   */
  BlissShift compute_shift(const data::Hamiltonian& hamiltonian,
                           unsigned int n_alpha_electrons,
                           unsigned int n_beta_electrons) const;

  /**
   * @brief Access the algorithm's name.
   *
   * @return The algorithm's name.
   */
  std::string name() const override { return "fermionic_low_rank"; }

  /**
   * @brief Access the algorithm's type name.
   *
   * @return The algorithm's type name.
   */
  std::string type_name() const final { return "hamiltonian_regularizer"; };

 protected:
  /**
   * @brief Implementation of Hamiltonian regularization.
   *
   * Composes compute_shift() and rebuild_bliss_shifted_hamiltonian(). Called
   * by run() after settings have been locked.
   */
  std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_alpha_electrons,
      unsigned int n_beta_electrons) const override;
};

/**
 * @brief Factory class for creating BLISS regularizer instances.
 *
 * Typical usage:
 * ```
 * using qdk::chemistry::algorithms::HamiltonianRegularizerFactory;
 * auto reg = HamiltonianRegularizerFactory::create("fermionic_low_rank");
 * reg->settings().set("df_truncation_threshold", 1e-8);
 * auto shifted = reg->run(hamiltonian, n_alpha, n_beta);
 * ```
 *
 * @see BlissRegularizer
 */
struct HamiltonianRegularizerFactory
    : public AlgorithmFactory<BlissRegularizer, HamiltonianRegularizerFactory> {
  static std::string algorithm_type_name() { return "hamiltonian_regularizer"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "fermionic_low_rank"; }
};

}  // namespace qdk::chemistry::algorithms

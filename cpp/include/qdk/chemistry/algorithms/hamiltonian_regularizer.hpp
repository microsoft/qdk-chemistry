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
 * @struct BlissShift
 * @brief The block-invariant symmetry shift (BLISS) parameters.
 *
 * A BlissShift bundles the three quantities that define the BLISS operator
 * K = mu1*(N - Ne) + mu2*(N^2 - Ne^2) + (N - Ne)*sum_ij xi_ij E_ij
 * subtracted from a Hamiltonian H (Patel et al., arXiv:2409.18277). Because K
 * annihilates every Ne-electron state, subtracting it leaves the Ne-sector
 * energy invariant while reducing the fermionic LCU 1-norm.
 *
 * BlissShift deliberately carries only the *result* of a shift computation, so
 * the parameters can come from any method (see BlissRegularizer's
 * "shift_method" setting) -- or from an entirely external source -- and be
 * applied to a Hamiltonian via rebuild_bliss_shifted_hamiltonian().
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
/// This is the SINGLE SOURCE OF TRUTH for that tensor.
/// rebuild_bliss_shifted_hamiltonian() adds dg (via add_two_body_correction())
/// directly onto g to build g~ = g + dg. A one-electron shift solver (e.g. the
/// flr_bliss method's solve_one_electron_shift()) never needs the whole
/// O(norb^4) tensor -- it only needs dg's Coulomb/exchange-type contractions
/// (add_coulomb_contraction/add_exchange_contraction below), which fold into
/// the effective one-electron operator alongside the ORIGINAL g's own
/// Coulomb/exchange contraction. Both call sites derive their numbers from
/// THIS struct, so they cannot silently drift apart.
struct TwoBodyBlissCorrection {
  double mu2;
  Eigen::MatrixXd xi;

  /// Add Sum_k dg_ijkk (Coulomb-type contraction) onto `coulomb` in place,
  /// obtained by substituting dg_ijkl's definition and summing k over
  /// Sum_k delta_kk = norb, Sum_k xi_kk = tr(xi):
  ///   Sum_k dg_ijkk = -2*mu2*norb*delta_ij - norb*xi_ij - tr(xi)*delta_ij
  ///                 = -(2*mu2*norb + tr(xi))*delta_ij - norb*xi_ij
  /// `norb` is taken from the caller's matrix, and the correction is folded in
  /// place so no separate norb x norb result is allocated: on return `coulomb`
  /// holds coul(g~) = coul(g) + Sum_k dg_ijkk.
  void add_coulomb_contraction(Eigen::MatrixXd& coulomb) const {
    const double norb = static_cast<double>(coulomb.rows());
    coulomb -= norb * xi;
    coulomb.diagonal().array() -= 2.0 * mu2 * norb + xi.trace();
  }

  /// Add Sum_k dg_ikkj (exchange-type contraction) onto `exchange` in place,
  /// using Sum_k delta_ik*delta_kj = delta_ij, Sum_k xi_ik*delta_kj = xi_ij,
  /// Sum_k delta_ik*xi_kj = xi_ij:
  ///   Sum_k dg_ikkj = -2*mu2*delta_ij - 2*xi_ij
  /// The correction is folded in place (no norb x norb allocation): on return
  /// `exchange` holds exch(g~) = exch(g) + Sum_k dg_ikkj.
  void add_exchange_contraction(Eigen::MatrixXd& exchange) const {
    exchange -= 2.0 * xi;
    exchange.diagonal().array() -= 2.0 * mu2;
  }

  /// Add the two-body BLISS correction dg_ijkl onto `g` in place, where `g` is
  /// the flattened two-electron tensor ((i*norb+j)*norb+k)*norb+l with side
  /// length `norb`. On return `g` holds g~ = g + dg without materializing a
  /// separate O(norb^4) dg tensor. Used by rebuild_bliss_shifted_hamiltonian()
  /// to build g~.
  ///
  /// dg is structurally sparse -- each of its three terms is delta-supported --
  /// so instead of an O(norb^4) full sweep we apply only the non-zero entries:
  ///   * -2*mu2*delta_ij*delta_kl : the O(norb^2) block i==j, k==l,
  ///   * -xi_ij*delta_kl          : the O(norb^3) block k==l,
  ///   * -delta_ij*xi_kl          : the O(norb^3) block i==j.
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
 * Applies the global BLISS shift (mu1, mu2, xi) of Patel et al.
 * (arXiv:2409.18277) directly to the dense one- and two-electron integrals of
 * `original` and assembles the resulting Hamiltonian. In this container's
 * canonical chemist convention g[i,j,k,l] = (ij|kl) the shift reproduces the
 * operator
 *   K = mu1*(N - Ne) + mu2*(N^2 - Ne^2) + (N - Ne)*sum_ij xi_ij E_ij
 * subtracted from H. Because K annihilates every Ne-electron state, the
 * Ne-sector energy is invariant for any (mu1, mu2, xi). Expanding -K gives
 *   h_tilde_ij   = h_ij + (Ne - 1)*xi_ij - (mu1 + mu2)*delta_ij
 *   g_tilde_ijkl = g_ijkl - 2*mu2*delta_ij*delta_kl
 *                          - xi_ij*delta_kl - delta_ij*xi_kl
 *   E_core'      = E_core + mu1*Ne + mu2*Ne^2
 * These coefficients were derived in the container's own convention and
 * verified to machine precision against explicit single-determinant energies.
 *
 * This function is intentionally independent of how the shift was computed:
 * `shift` may come from BlissRegularizer::compute_shift() or from any external
 * source. Everything else (the physical one- and two-electron integrals, core
 * energy, orbitals, inactive Fock matrix, and Hamiltonian type) is read from
 * `original`.
 *
 * @param original The Hamiltonian being shifted. Must be restricted.
 * @param shift The BLISS shift parameters (mu1, mu2, xi) to apply.
 * @param input_num_electrons Target number of active electrons (Ne). Must be a
 *        non-negative integer; the invariance guarantee only holds for an
 *        integer electron count.
 * @return The BLISS-shifted Hamiltonian.
 *
 * @throws std::invalid_argument if `original` is unrestricted, if
 *         `shift.xi` is not norb x norb
 */
std::shared_ptr<data::Hamiltonian> rebuild_bliss_shifted_hamiltonian(
    const data::Hamiltonian& original, const BlissShift& shift,
    unsigned int input_num_electrons);

/**
 * @class BlissSettings
 * @brief Settings container for the BLISS Hamiltonian regularizer.
 *
 * Default settings:
 * - shift_method: "flr_bliss" - selects how the BLISS shift (mu1, mu2, xi) is
 *   computed. "flr_bliss" is the fermionic-low-rank BLISS method of Patel et
 *   al. (arXiv:2409.18277).
 * - df_truncation_threshold: 0.0 - (flr_bliss method) fragments produced by
 *   double-factorizing the two-electron integrals whose eigenvalue magnitude
 *   falls below this threshold are dropped. The default of 0.0 performs no
 *   truncation (an exact/lossless double factorization).
 */
class BlissSettings : public qdk::chemistry::data::Settings {
 public:
  BlissSettings() {
    set_default("shift_method", std::string("flr_bliss"),
                "Method used to compute the BLISS shift (mu1, mu2, xi).",
                data::ListConstraint<std::string>{
                    {std::vector<std::string>{"flr_bliss"}}});
    set_default("df_truncation_threshold", 0.0);
  }
};

/**
 * @class BlissRegularizer
 * @brief Hamiltonian regularizer implementing block-invariant symmetry shifts.
 *
 * A BlissRegularizer maps a Hamiltonian, together with the target number of
 * alpha/beta electrons, to a new Hamiltonian that is energetically equivalent
 * within the target electron-number sector but whose LCU/qubitization
 * coefficients (e.g. the fermionic 1-norm lambda) may be reduced. It does so by
 * subtracting a block-invariant symmetry shift (BLISS) operator that
 * annihilates every state with the target electron count, so the physical
 * energy of the target-electron-count sector is preserved while the operator's
 * norm outside that sector -- and hence resource estimates for algorithms like
 * qubitized phase estimation -- can shrink.
 *
 * The regularizer is a thin composition of two steps:
 *  1. compute_shift() -- compute the BLISS parameters (mu1, mu2, xi) via the
 *     method selected by the "shift_method" setting (default "flr_bliss").
 *  2. rebuild_bliss_shifted_hamiltonian() -- apply that BlissShift to the dense
 * integrals. Both steps are public so callers can obtain a BlissShift on its
 * own, or supply an externally computed BlissShift to
 * rebuild_bliss_shifted_hamiltonian() directly.
 *
 * Only restricted (spin-restricted) Hamiltonians are currently supported.
 *
 * @see BlissShift
 * @see rebuild_bliss_shifted_hamiltonian
 * @see BlissSettings
 * @see qdk::chemistry::utils::hamiltonian_one_norm for a standalone way to
 *      inspect a Hamiltonian's fermionic 1-norm without running a regularizer.
 */
class BlissRegularizer
    : public Algorithm<BlissRegularizer, std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Hamiltonian>, unsigned int,
                       unsigned int> {
 public:
  /**
   * @brief Default constructor.
   *
   * Creates a BLISS regularizer with default settings (shift_method =
   * "flr_bliss", df_truncation_threshold = 0.0).
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
   * Dispatches to the method selected by the "shift_method" setting and
   * returns the resulting parameters *without* rebuilding the Hamiltonian.
   * Use rebuild_bliss_shifted_hamiltonian() to apply the returned (or an
   * externally sourced) BlissShift.
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
   * Composes compute_shift() and rebuild_bliss_shifted_hamiltonian().
   * Automatically called by run() after settings have been locked.
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
 * auto regularizer =
 *     qdk::chemistry::algorithms::HamiltonianRegularizerFactory::create("fermionic_low_rank");
 * regularizer->settings().set("df_truncation_threshold", 1e-8);
 * auto shifted = regularizer->run(hamiltonian, n_alpha, n_beta);
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

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
 * @brief Number-symmetry Hamiltonian shifts (BLISS and variants).
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
 * @struct SymmetryShift
 * @brief Parameters of a number-symmetry shift operator [1].
 *
 * Bundles the parameters of the symmetry-shift operator subtracted from a
 * Hamiltonian H:
 *   K = mu1*(N - Ne) + mu2*(N^2 - Ne^2) + (N - Ne)*sum_ij xi_ij E_ij
 * K annihilates every Ne-electron state, so subtracting it leaves the
 * Ne-sector energy invariant while reducing the fermionic LCU 1-norm.
 *
 * This parameterization is shared by every SymmetryShifter implementation
 * (BLISS [1] and its fermionic low-rank variant [2] included); only the way
 * (mu1, mu2, xi) are *computed* differs. A SymmetryShift therefore carries
 * only the *result* of a shift computation, so it may equally come from
 * SymmetryShifter::compute_shift() or an external source, and is applied via
 * rebuild_shifted_hamiltonian().
 */
struct SymmetryShift {
  double mu1 = 0.0;    ///< One-electron shift.
  double mu2 = 0.0;    ///< Two-electron shift.
  Eigen::MatrixXd xi;  ///< Two-electron shift matrix (norb x norb).
};

/**
 * @brief Apply a symmetry shift to a Hamiltonian and assemble the shifted one.
 *
 * Applies the global symmetry shift (mu1, mu2, xi) [1,2] to the dense
 * integrals of `original`. In this container's chemist convention
 * g[i,j,k,l] = (ij|kl), subtracting K (see SymmetryShift) expands to
 *   h~_ij   = h_ij + (Ne - 1)*xi_ij - (mu1 + mu2)*delta_ij
 *   g~_ijkl = g_ijkl - 2*mu2*delta_ij*delta_kl
 *                    - xi_ij*delta_kl - delta_ij*xi_kl
 *   E_core' = E_core + mu1*Ne + mu2*Ne^2
 * so the Ne-sector energy is invariant for any (mu1, mu2, xi).
 *
 * How `shift` was computed is irrelevant: it may come from
 * SymmetryShifter::compute_shift() or any external source. Everything else
 * (integrals, core energy, orbitals, inactive Fock matrix, Hamiltonian type)
 * is read from `original`.
 *
 * @param original The Hamiltonian being shifted. Must be restricted.
 * @param shift The symmetry shift parameters (mu1, mu2, xi) to apply.
 * @param num_electrons Target number of active electrons (Ne); the
 *        invariance guarantee only holds for an integer electron count.
 * @return The shifted Hamiltonian.
 *
 * @throws std::invalid_argument if `original` is unrestricted or `shift.xi`
 *         is not norb x norb.
 */
std::shared_ptr<data::Hamiltonian> rebuild_shifted_hamiltonian(
    const data::Hamiltonian& original, const SymmetryShift& shift,
    unsigned int num_electrons);

/**
 * @class SymmetryShifter
 * @brief Abstract interface for number-symmetry Hamiltonian shifts [1,2].
 *
 * Maps a Hamiltonian and a target alpha/beta electron count to a new
 * Hamiltonian that is energetically equivalent within that electron-number
 * sector but whose LCU/qubitization coefficients (e.g. the fermionic 1-norm
 * lambda) may be reduced, shrinking resource estimates for algorithms such as
 * qubitized phase estimation.
 *
 * Every implementation is a thin composition of two public steps:
 *  1. compute_shift() -- compute (mu1, mu2, xi); this is what distinguishes
 *     one implementation from another.
 *  2. rebuild_shifted_hamiltonian() -- apply that shift; shared by all
 *     implementations and usable on an externally computed shift.
 *
 * Only restricted (spin-restricted) Hamiltonians are currently supported.
 *
 * Typical usage:
 * @code
 * auto shifter =
 *   qdk::chemistry::algorithms::SymmetryShifterFactory::create("algorithm_name");
 * shifter->settings().set("parameter_name", value);
 * auto shifted = shifter->run(hamiltonian, n_alpha, n_beta);
 * @endcode
 *
 * @see SymmetryShift
 * @see rebuild_shifted_hamiltonian
 * @see SymmetryShifterFactory for creating instances of symmetry shifters
 * @see qdk::chemistry::utils::hamiltonian_one_norm to inspect a Hamiltonian's
 *      fermionic 1-norm without running a shifter.
 */
class SymmetryShifter
    : public Algorithm<SymmetryShifter, std::shared_ptr<data::Hamiltonian>,
                       std::shared_ptr<data::Hamiltonian>, unsigned int,
                       unsigned int> {
 public:
  /**
   * @brief Default constructor for SymmetryShifter.
   */
  SymmetryShifter() = default;

  /**
   * @brief Virtual destructor.
   */
  virtual ~SymmetryShifter() = default;

  /**
   * @brief Shift a Hamiltonian for a target electron count.
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param hamiltonian The Hamiltonian to shift
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
   * @brief Compute the symmetry shift (mu1, mu2, xi) for a target electron
   *        count.
   *
   * Returns the parameters *without* rebuilding the Hamiltonian; apply them
   * with rebuild_shifted_hamiltonian().
   *
   * @param hamiltonian The Hamiltonian to analyze. Must be restricted.
   * @param n_alpha_electrons The target number of alpha electrons.
   * @param n_beta_electrons The target number of beta electrons.
   * @return The computed symmetry shift parameters.
   *
   * @throws std::invalid_argument if the Hamiltonian is unrestricted.
   */
  virtual SymmetryShift compute_shift(const data::Hamiltonian& hamiltonian,
                                      unsigned int n_alpha_electrons,
                                      unsigned int n_beta_electrons) const = 0;

  /**
   * @brief Access the algorithm's name.
   *
   * @return The algorithm's name.
   */
  virtual std::string name() const = 0;

  /**
   * @brief Access the algorithm's type name.
   *
   * @return The algorithm's type name.
   */
  std::string type_name() const final { return "symmetry_shifter"; };

 protected:
  /**
   * @brief Implementation of the symmetry shift.
   *
   * Composes compute_shift() and rebuild_shifted_hamiltonian(). Called by
   * run() after settings have been locked.
   */
  virtual std::shared_ptr<data::Hamiltonian> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_alpha_electrons,
      unsigned int n_beta_electrons) const = 0;
};

/**
 * @brief Factory class for creating symmetry shifter instances.
 *
 * Typical usage:
 * ```
 * using qdk::chemistry::algorithms::SymmetryShifterFactory;
 * auto shifter = SymmetryShifterFactory::create("fermionic_low_rank");
 * shifter->settings().set("df_truncation_threshold", 1e-8);
 * auto shifted = shifter->run(hamiltonian, n_alpha, n_beta);
 * ```
 *
 * @see SymmetryShifter
 */
struct SymmetryShifterFactory
    : public AlgorithmFactory<SymmetryShifter, SymmetryShifterFactory> {
  static std::string algorithm_type_name() { return "symmetry_shifter"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "fermionic_low_rank"; }
};

}  // namespace qdk::chemistry::algorithms

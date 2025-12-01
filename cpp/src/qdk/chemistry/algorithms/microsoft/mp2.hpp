/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#pragma once
#include <memory>
#include <qdk/chemistry/algorithms/reference_derived_calculator.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class MP2Calculator
 * @brief Microsoft QDK implementation of MP2 calculations
 *
 * This class implements Møller-Plesset second-order perturbation theory (MP2)
 * calculations with automatic R/UMP2 selection based on the input ansatz.
 * It inherits from ReferenceDerivedCalculator and provides a unified interface
 * for MP2 calculations.
 *
 */
class MP2Calculator : public ReferenceDerivedCalculator {
 public:
  /**
   * @brief Default constructor
   */
  MP2Calculator();

  /**
   * @brief Virtual destructor
   */
  virtual ~MP2Calculator() = default;

  /**
   * @brief Get the algorithm name
   *
   * @return The algorithm's name
   */
  std::string name() const override final { return "qdk_mp2_calculator"; }

 protected:
  /**
   * @brief Implementation of MP2 calculation
   *
   * This method performs the MP2 calculation using the provided ansatz and
   * returns both the total energy (reference + correlation) and the MP2
   * wavefunction stored in an MP2Container. T2 amplitudes are computed
   * lazily by the MP2Container when first requested.
   *
   * @param ansatz The Ansatz (Wavefunction and Hamiltonian) describing the
   *               quantum system
   * @return A pair containing the total energy and the MP2 wavefunction
   * @throws Error if orbitals do not have an energy attribute, i.e., if they
   * are localized. MP2 cannot handle localized orbitals.
   *
   */
  std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Ansatz> ansatz) const override;

 public:
  // Static helper methods for computing T2 amplitudes
  /**
   * @brief Compute same-spin (antisymmetric) T2 amplitudes
   *
   * This helper computes T2 amplitudes for same-spin electron pairs
   * (alpha-alpha or beta-beta), which have antisymmetric exchange integrals.
   *
   * @param eps Orbital energies
   * @param moeri Two-electron repulsion integrals (MO basis)
   * @param n_occ Number of occupied orbitals
   * @param n_vir Number of virtual orbitals
   * @param stride_i Stride for first index in 4D integral array
   * @param stride_j Stride for second index in 4D integral array
   * @param stride_k Stride for third index in 4D integral array
   * @param t2 Output vector for T2 amplitudes (will be filled)
   * @param energy Optional pointer to accumulate energy contribution
   */
  static void compute_same_spin_t2(const Eigen::VectorXd& eps,
                                   const Eigen::VectorXd& moeri, size_t n_occ,
                                   size_t n_vir, size_t stride_i,
                                   size_t stride_j, size_t stride_k,
                                   Eigen::VectorXd& t2,
                                   double* energy = nullptr);

  /**
   * @brief Compute opposite-spin T2 amplitudes
   *
   * This helper computes T2 amplitudes for opposite-spin electron pairs
   * (alpha-beta), which don't have antisymmetric exchange.
   *
   * @param eps_i_spin Orbital energies for i,a indices
   * @param eps_j_spin Orbital energies for j,b indices
   * @param moeri Two-electron repulsion integrals (MO basis)
   * @param n_occ_i Number of occupied orbitals (i spin)
   * @param n_occ_j Number of occupied orbitals (j spin)
   * @param n_vir_i Number of virtual orbitals (i spin)
   * @param n_vir_j Number of virtual orbitals (j spin)
   * @param stride_i Stride for first index in 4D integral array
   * @param stride_j Stride for second index in 4D integral array
   * @param stride_k Stride for third index in 4D integral array
   * @param t2 Output vector for T2 amplitudes (will be filled)
   * @param energy Optional pointer to accumulate energy contribution
   */
  static void compute_opposite_spin_t2(
      const Eigen::VectorXd& eps_i_spin, const Eigen::VectorXd& eps_j_spin,
      const Eigen::VectorXd& moeri, size_t n_occ_i, size_t n_occ_j,
      size_t n_vir_i, size_t n_vir_j, size_t stride_i, size_t stride_j,
      size_t stride_k, Eigen::VectorXd& t2, double* energy = nullptr);

  /**
   * @brief Compute restricted T2 amplitudes
   *
   * This helper computes T2 amplitudes for restricted (closed-shell) systems.
   *
   * @param eps Orbital energies
   * @param moeri Two-electron repulsion integrals (MO basis)
   * @param n_occ Number of occupied orbitals
   * @param n_vir Number of virtual orbitals
   * @param stride_i Stride for first index in 4D integral array
   * @param stride_j Stride for second index in 4D integral array
   * @param stride_k Stride for third index in 4D integral array
   * @param t2 Output vector for T2 amplitudes (will be filled)
   * @param energy Optional pointer to accumulate energy contribution
   */
  static void compute_restricted_t2(const Eigen::VectorXd& eps,
                                    const Eigen::VectorXd& moeri, size_t n_occ,
                                    size_t n_vir, size_t stride_i,
                                    size_t stride_j, size_t stride_k,
                                    Eigen::VectorXd& t2,
                                    double* energy = nullptr);

 private:
  /**
   * @brief Calculate restricted MP2 correlation energy
   *
   * @param ham Shared pointer to the Hamiltonian containing MO integrals
   * @param orbitals Shared pointer to orbitals containing orbital energies
   * @param n_occ Number of occupied orbitals (doubly occupied in restricted
   * case)
   * @return MP2 correlation energy
   */
  double _calculate_restricted_mp2_energy(
      std::shared_ptr<qdk::chemistry::data::Hamiltonian> ham,
      std::shared_ptr<qdk::chemistry::data::Orbitals> orbitals,
      size_t n_occ) const;

  /**
   * @brief Calculate unrestricted MP2 correlation energy
   *
   * @param ham Shared pointer to the Hamiltonian
   * @param orbitals Shared pointer to orbitals
   * @param n_alpha Number of alpha electrons
   * @param n_beta Number of beta electrons
   * @return Total MP2 correlation energy
   */
  double _calculate_unrestricted_mp2_energy(
      std::shared_ptr<qdk::chemistry::data::Hamiltonian> ham,
      std::shared_ptr<qdk::chemistry::data::Orbitals> orbitals, size_t n_alpha,
      size_t n_beta) const;
};

/**
 * @brief Factory function to create Microsoft MP2 calculator
 */
std::unique_ptr<ReferenceDerivedCalculator> make_qdk_mp2_calculator();

}  // namespace qdk::chemistry::algorithms::microsoft

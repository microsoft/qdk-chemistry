/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License. See LICENSE.txt in the project root for
 * license information.
 */

#pragma once
#include <memory>
#include <qdk/chemistry/algorithms/dynamical_correlation_calculator.hpp>
#include <qdk/chemistry/data/ansatz.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/utils/tensor.hpp>
#include <qdk/chemistry/utils/tensor_span.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class MP2Calculator
 * @brief Microsoft QDK implementation of MP2 calculations
 *
 * This class implements Møller-Plesset second-order perturbation theory (MP2)
 * calculations with automatic R/UMP2 selection based on the input ansatz.
 * It inherits from DynamicalCorrelationCalculator and provides a unified
 * interface for MP2 calculations.
 *
 */
class MP2Calculator : public DynamicalCorrelationCalculator {
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
  /**
   * @brief Compute same-spin (antisymmetric) T2 amplitudes
   *
   * This helper computes T2 amplitudes for same-spin electron pairs
   * (alpha-alpha or beta-beta), which have antisymmetric exchange integrals.
   *
   * @note Tensor dimensions are deduced from the output tensor `t2`:
   * - `n_occ = t2.extent(0) = t2.extent(1)` (number of occupied orbitals)
   * - `n_vir = t2.extent(2) = t2.extent(3)` (number of virtual orbitals)
   *
   * The caller must allocate `t2` with the correct shape before calling.
   *
   * @tparam Scalar Numeric type
   * @param eps Orbital energies
   * @param mo_aaaa Two-electron integrals in MO basis
   * @param t2 Output tensor for T2 amplitudes
   * @param energy Optional pointer to accumulate energy contribution
   */
  template <typename Scalar>
  static void compute_same_spin_t2(const Eigen::VectorXd& eps,
                                   rank4_span<const Scalar> mo_aaaa,
                                   rank4_tensor<Scalar>& t2,
                                   Scalar* energy = nullptr);

  /**
   * @brief Compute opposite-spin T2 amplitudes
   *
   * This helper computes T2 amplitudes for opposite-spin electron pairs
   * (alpha-beta), which don't have antisymmetric exchange.
   *
   * @note Tensor dimensions are deduced from the output tensor `t2`:
   * - `n_occ_i = t2.extent(0)` (occupied orbitals for spin i)
   * - `n_occ_j = t2.extent(1)` (occupied orbitals for spin j)
   * - `n_vir_i = t2.extent(2)` (virtual orbitals for spin i)
   * - `n_vir_j = t2.extent(3)` (virtual orbitals for spin j)
   *
   * The caller must allocate `t2` with the correct shape before calling.
   *
   * @tparam Scalar Numeric type
   * @param eps_i_spin Orbital energies for i,a indices
   * @param eps_j_spin Orbital energies for j,b indices
   * @param mo_aabb Two-electron integrals in MO basis
   * @param t2 Output tensor for T2 amplitudes
   * @param energy Optional pointer to accumulate energy contribution
   */
  template <typename Scalar>
  static void compute_opposite_spin_t2(const Eigen::VectorXd& eps_i_spin,
                                       const Eigen::VectorXd& eps_j_spin,
                                       rank4_span<const Scalar> mo_aabb,
                                       rank4_tensor<Scalar>& t2,
                                       Scalar* energy = nullptr);

  /**
   * @brief Compute restricted T2 amplitudes
   *
   * This helper computes T2 amplitudes for restricted (closed-shell) systems.
   *
   * @note Tensor dimensions are deduced from the output tensor `t2`:
   * - `n_occ = t2.extent(0) = t2.extent(1)` (number of occupied orbitals)
   * - `n_vir = t2.extent(2) = t2.extent(3)` (number of virtual orbitals)
   *
   * The caller must allocate `t2` with the correct shape before calling.
   *
   * @tparam Scalar Numeric type
   * @param eps Orbital energies
   * @param mo_aaaa Two-electron integrals in MO basis
   * @param t2 Output tensor for T2 amplitudes
   * @param energy Optional pointer to accumulate energy contribution
   */
  template <typename Scalar>
  static void compute_restricted_t2(const Eigen::VectorXd& eps,
                                    rank4_span<const Scalar> mo_aaaa,
                                    rank4_tensor<Scalar>& t2,
                                    Scalar* energy = nullptr);

  /**
   * @brief Calculate restricted MP2 correlation energy
   *
   * @param ham Shared pointer to the Hamiltonian containing MO integrals
   * @param orbitals Shared pointer to orbitals containing orbital energies
   * @param n_occ Number of occupied orbitals (doubly occupied in restricted
   * case)
   * @return MP2 correlation energy
   */
  double calculate_restricted_mp2_energy(
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
  double calculate_unrestricted_mp2_energy(
      std::shared_ptr<qdk::chemistry::data::Hamiltonian> ham,
      std::shared_ptr<qdk::chemistry::data::Orbitals> orbitals, size_t n_alpha,
      size_t n_beta) const;
};

/**
 * @brief Factory function to create Microsoft MP2 calculator
 */
std::unique_ptr<DynamicalCorrelationCalculator> make_qdk_mp2_calculator();

}  // namespace qdk::chemistry::algorithms::microsoft

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <complex>
#include <cstdint>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Immutable data class describing a fermion-to-qubit encoding.
 *
 * Stores a 2N-entry table mapping each Majorana operator gamma_k to a Pauli
 * word. The bilinear i*gamma_j*gamma_k is the unified primitive and is
 * computed on demand from the table.
 *
 * Pauli words use the little-endian convention of QubitHamiltonian (qubit 0
 * has the smallest sparse index).
 */
class MajoranaMapping {
 public:
  /**
   * @brief Construct from a 2N-entry Majorana-to-Pauli table.
   *
   * @param table 2N SparsePauliWord entries (gamma_0, ..., gamma_{2N-1}).
   * @param name  Optional encoding label. Stored but not used for dispatch.
   * @throws std::invalid_argument If the table is empty or its size is odd.
   */
  explicit MajoranaMapping(std::vector<SparsePauliWord> table,
                           std::string name = "");

  /// Number of fermionic modes (the table has 2N entries).
  std::size_t num_modes() const { return table_.size() / 2; }

  /// Number of qubits (max qubit index + 1, or 0 if all entries are identity).
  std::size_t num_qubits() const { return num_qubits_; }

  /**
   * @brief Pauli word for Majorana operator gamma_k.
   * @throws std::out_of_range if k >= 2N.
   */
  const SparsePauliWord& operator()(std::size_t k) const;

  /// Named alias for operator()(k).
  const SparsePauliWord& majorana(std::size_t k) const { return (*this)(k); }

  /**
   * @brief Pauli image of the bilinear i*gamma_j*gamma_k.
   *
   * Returns (coeff, word) such that coeff*word equals i*gamma_j*gamma_k in the
   * encoded representation. The coefficient is real (+/-1) since the bilinear
   * is Hermitian; the return type is complex for consistency with
   * PauliTermAccumulator::multiply_uncached.
   *
   * @throws std::out_of_range if j or k >= 2N.
   * @throws std::invalid_argument if j == k.
   */
  std::pair<std::complex<double>, SparsePauliWord> bilinear(
      std::size_t j, std::size_t k) const;

  /// Whether individual Majoranas have a Pauli image (true for the table form).
  bool is_majorana_atomic() const { return true; }

  /// The full Majorana-to-Pauli table.
  const std::vector<SparsePauliWord>& table() const { return table_; }

  /// Encoding name (may be empty for custom encodings).
  const std::string& name() const { return name_; }

  // --- Factory methods for standard encodings ---

  /// Jordan-Wigner encoding on num_modes qubits.
  static MajoranaMapping jordan_wigner(std::size_t num_modes);

  /// Bravyi-Kitaev (Fenwick-tree) encoding on num_modes qubits.
  static MajoranaMapping bravyi_kitaev(std::size_t num_modes);

  /// Balanced binary-tree Bravyi-Kitaev encoding (arXiv:1701.07072).
  static MajoranaMapping bravyi_kitaev_tree(std::size_t num_modes);

  /// Parity encoding on num_modes qubits.
  static MajoranaMapping parity(std::size_t num_modes);

 private:
  /// Majorana-to-Pauli table: table_[k] is the Pauli word for gamma_k.
  std::vector<SparsePauliWord> table_;

  /// Human-readable encoding name.
  std::string name_;

  /// Cached qubit count (max qubit index + 1).
  std::size_t num_qubits_;

  static std::size_t compute_num_qubits(
      const std::vector<SparsePauliWord>& table);
};

/**
 * @brief Result of a Majorana-loop fermion-to-qubit mapping.
 *
 * Parallel arrays of Pauli words and their complex coefficients.
 */
struct MajoranaMapResult {
  std::vector<SparsePauliWord> words;
  std::vector<std::complex<double>> coefficients;
};

/**
 * @brief Map a fermionic Hamiltonian to qubit Pauli terms via Majorana loops.
 *
 * Decomposes each fermionic operator into Majorana products, looks up each
 * gamma_k in the mapping, and accumulates the resulting Pauli words.
 *
 * @note Limitation: only valid for spin-free Hamiltonians (spin-independent
 *       one- and two-body integrals, h1_alpha == h1_beta).
 *
 * @param mapping The Majorana-to-Pauli encoding.
 * @param core_energy Core (nuclear repulsion + frozen core) energy.
 * @param h1_alpha One-body integrals, alpha spin (n_spatial x n_spatial).
 * @param h1_beta One-body integrals, beta spin (n_spatial x n_spatial).
 * @param eri_aaaa Flattened two-body integrals (aa|aa), chemist notation.
 * @param eri_aabb Flattened two-body integrals (aa|bb), chemist notation.
 * @param eri_bbbb Flattened two-body integrals (bb|bb), chemist notation.
 * @param n_spatial Number of spatial orbitals.
 * @param is_spin_free Whether the integrals are spin-independent.
 * @param threshold Pauli terms with |coeff| < threshold are dropped.
 * @param integral_threshold Integrals with |value| < this are skipped.
 * @return MajoranaMapResult with Pauli words and coefficients.
 */
MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool is_spin_free,
    double threshold, double integral_threshold);

}  // namespace qdk::chemistry::data

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
 * @brief Data class describing a fermion-to-qubit encoding.
 *
 * Majorana-atomic mappings store a 2N-entry Pauli table for individual
 * gamma_k; bilinear-only mappings (via from_bilinears) store the bilinear
 * images directly. bilinear(j, k) is available on both forms.
 */
class MajoranaMapping {
 public:
  /**
   * @brief Construct a Majorana-atomic mapping from a 2N-entry table.
   *
   * @param table 2N SparsePauliWord entries (gamma_0, ..., gamma_{2N-1}).
   * @param name  Optional encoding label.
   * @throws std::invalid_argument If the table is empty or its size is odd.
   */
  static MajoranaMapping from_table(std::vector<SparsePauliWord> table,
                                    std::string name = "");

  /**
   * @brief Construct a bilinear-only mapping from pre-computed bilinears.
   *
   * The upper_triangle vector stores (coeff, word) for each pair (j, k) with
   * j < k, in row-major order: (0,1), (0,2), ..., (0,M-1), (1,2), ...,
   * (M-2,M-1) where M = 2*num_modes.  Size must be M*(M-1)/2.
   *
   * @param num_modes Number of fermionic modes (N).
   * @param upper_triangle Bilinear entries for all j < k.
   * @param name Optional encoding label.
   * @throws std::invalid_argument If sizes are inconsistent or num_modes == 0.
   */
  static MajoranaMapping from_bilinears(
      std::size_t num_modes,
      std::vector<std::pair<std::complex<double>, SparsePauliWord>>
          upper_triangle,
      std::string name = "");

  /// Number of fermionic modes.
  std::size_t num_modes() const { return num_modes_; }

  /// Number of qubits (max qubit index + 1, or 0 if all entries are identity).
  std::size_t num_qubits() const { return num_qubits_; }

  /**
   * @brief Pauli word for Majorana operator gamma_k.
   * @throws std::out_of_range if k >= 2N.
   * @throws std::logic_error if the mapping is not Majorana-atomic.
   */
  const SparsePauliWord& operator()(std::size_t k) const;

  /// Named alias for operator()(k).
  const SparsePauliWord& majorana(std::size_t k) const { return (*this)(k); }

  /**
   * @brief Pauli image of the bilinear i*gamma_j*gamma_k.
   *
   * O(1) lookup. For Majorana-atomic mappings the coefficient is real (+/-1);
   * for bilinear-only mappings it is whatever was provided at construction.
   *
   * @throws std::out_of_range if j or k >= 2N.
   * @throws std::invalid_argument if j == k.
   */
  std::pair<std::complex<double>, const SparsePauliWord&> bilinear(
      std::size_t j, std::size_t k) const;

  /// Whether individual Majoranas have a Pauli image.
  bool is_majorana_atomic() const { return majorana_atomic_; }

  /// The full Majorana-to-Pauli table (empty for bilinear-only mappings).
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
  MajoranaMapping(
      std::vector<SparsePauliWord> table,
      std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears,
      std::string name, std::size_t num_modes, std::size_t num_qubits);

  /// Majorana-to-Pauli table (empty for bilinear-only mappings).
  std::vector<SparsePauliWord> table_;

  /// Cached bilinear table, upper triangle row-major. Always populated.
  std::vector<std::pair<std::complex<double>, SparsePauliWord>> bilinears_;

  /// Human-readable encoding name.
  std::string name_;

  /// Number of fermionic modes.
  std::size_t num_modes_;

  /// Cached qubit count (max qubit index + 1).
  std::size_t num_qubits_;

  /// True for table-constructed mappings, false for bilinear-only.
  bool majorana_atomic_;

  /// Upper-triangle index: (j, k) with j < k, M = 2*num_modes.
  std::size_t bilinear_index(std::size_t j, std::size_t k) const {
    const std::size_t M = 2 * num_modes_;
    return j * (2 * M - j - 1) / 2 + (k - j - 1);
  }

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
 * @param mapping The Majorana-to-Pauli encoding.
 * @param core_energy Core (nuclear repulsion + frozen core) energy.
 * @param h1_alpha One-body integrals, alpha spin (n_spatial x n_spatial).
 * @param h1_beta One-body integrals, beta spin (n_spatial x n_spatial).
 * @param eri_aaaa Flattened two-body integrals (aa|aa), chemist notation.
 * @param eri_aabb Flattened two-body integrals (aa|bb), chemist notation.
 * @param eri_bbbb Flattened two-body integrals (bb|bb), chemist notation.
 * @param n_spatial Number of spatial orbitals.
 * @param spin_symmetric If true, use the spin-summed fast path. This assumes
 *        identical integrals across all spin channels (h_alpha == h_beta,
 *        eri_aaaa == eri_bbbb == eri_aabb), as produced by restricted orbitals.
 *        For unrestricted orbital sets, pass false — the engine handles each
 *        spin channel independently.
 * @param threshold Pauli terms with |coeff| < threshold are dropped.
 * @param integral_threshold Integrals with |value| < this are skipped.
 * @return MajoranaMapResult with Pauli words and coefficients.
 */
MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool spin_symmetric,
    double threshold, double integral_threshold);

}  // namespace qdk::chemistry::data

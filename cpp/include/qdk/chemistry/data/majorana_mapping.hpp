// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <cstdint>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <string>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Immutable data class mapping 2N Majorana operators to Pauli strings.
 *
 * A MajoranaMapping stores a table of 2N SparsePauliWord entries, one per
 * Majorana operator γ_k (k = 0, ..., 2N-1), where N is the number of
 * fermionic modes (spin-orbitals). Each entry is a single Pauli string
 * representing φ(γ_k) under the chosen fermion-to-qubit encoding.
 *
 * The mapping is validated at construction time by checking the Clifford
 * algebra anticommutation relations: {γ_i, γ_j} = 2δ_{ij} · I. This
 * guarantees that the mapping defines a valid encoding.
 *
 * Pauli string convention: The SparsePauliWord entries are stored in the
 * same little-endian convention used by QubitHamiltonian (qubit 0 has the
 * smallest index in the sparse representation).
 *
 * Factory methods construct the standard encodings (Jordan-Wigner,
 * Bravyi-Kitaev, parity) from a mode count. Custom encodings can be
 * constructed directly by providing the table.
 *
 * @see PauliTermAccumulator for the accumulation engine that uses this
 * mapping.
 */
class MajoranaMapping {
 public:
  /**
   * @brief Construct a MajoranaMapping from a Majorana-to-Pauli table.
   *
   * @param table Vector of 2N SparsePauliWord entries (one per Majorana
   *              operator γ_0, γ_1, ..., γ_{2N-1}).
   * @param name  Optional human-readable label for the encoding (e.g.,
   *              "jordan-wigner"). Stored but not used for dispatch.
   * @throws std::invalid_argument If table size is odd, empty, or the
   *         Clifford algebra validation fails.
   */
  explicit MajoranaMapping(std::vector<SparsePauliWord> table,
                           std::string name = "");

  /**
   * @brief Number of fermionic modes (spin-orbitals).
   * @return N where the table has 2N entries.
   */
  std::size_t num_modes() const { return table_.size() / 2; }

  /**
   * @brief Number of qubits required by this encoding.
   * @return max qubit index + 1 across all table entries, or 0 if all
   *         entries are identity.
   */
  std::size_t num_qubits() const { return num_qubits_; }

  /**
   * @brief Look up the Pauli string for Majorana operator γ_k.
   * @param k Majorana index (0 ≤ k < 2N).
   * @return const reference to the SparsePauliWord for γ_k.
   * @throws std::out_of_range if k ≥ 2N.
   */
  const SparsePauliWord& operator()(std::size_t k) const;

  /**
   * @brief Access the full Majorana-to-Pauli table.
   * @return const reference to the vector of SparsePauliWords.
   */
  const std::vector<SparsePauliWord>& table() const { return table_; }

  /**
   * @brief Human-readable name of the encoding.
   * @return The name string (may be empty for custom encodings).
   */
  const std::string& name() const { return name_; }

  /**
   * @brief Validate the Clifford algebra anticommutation relations.
   *
   * Checks that {γ_i, γ_j} = φ(γ_i)·φ(γ_j) + φ(γ_j)·φ(γ_i) = 2δ_{ij}·I
   * for all pairs (i, j). This is called automatically at construction.
   *
   * @throws std::invalid_argument if any anticommutator check fails.
   */
  void validate() const;

  // --- Factory methods for standard encodings ---

  /**
   * @brief Construct a Jordan-Wigner encoding.
   *
   * γ_{2j}   = Z_{j-1} ⊗ ... ⊗ Z_0 ⊗ X_j
   * γ_{2j+1} = Z_{j-1} ⊗ ... ⊗ Z_0 ⊗ Y_j
   *
   * @param num_modes Number of fermionic modes (spin-orbitals).
   *                  The encoding uses num_modes qubits.
   * @return MajoranaMapping with name "jordan-wigner".
   */
  static MajoranaMapping jordan_wigner(std::size_t num_modes);

  /**
   * @brief Construct a Bravyi-Kitaev encoding.
   *
   * γ_{2j}   = X_{U(j)} ⊗ X_j ⊗ Z_{P(j)}
   * γ_{2j+1} = X_{U(j)} ⊗ Y_j ⊗ Z_{R(j)}
   *
   * where U(j), P(j), R(j) are the update, parity, and remainder sets
   * derived from the BK binary tree.
   *
   * @param num_modes Number of fermionic modes (spin-orbitals).
   *                  The encoding uses num_modes qubits.
   * @return MajoranaMapping with name "bravyi-kitaev".
   */
  static MajoranaMapping bravyi_kitaev(std::size_t num_modes);

  /**
   * @brief Construct a parity encoding.
   *
   * γ_{2j}   = X_{n-1} ⊗ ... ⊗ X_{j+1} ⊗ Y_j ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
   * γ_{2j+1} = X_{n-1} ⊗ ... ⊗ X_{j+1} ⊗ X_j ⊗ Z_{j-1} ⊗ ... ⊗ Z_0
   *
   * @param num_modes Number of fermionic modes (spin-orbitals).
   *                  The encoding uses num_modes qubits.
   * @return MajoranaMapping with name "parity".
   */
  static MajoranaMapping parity(std::size_t num_modes);

 private:
  /// Majorana-to-Pauli table: table_[k] = φ(γ_k).
  std::vector<SparsePauliWord> table_;

  /// Human-readable encoding name.
  std::string name_;

  /// Cached qubit count (max qubit index + 1).
  std::size_t num_qubits_;

  /// Compute num_qubits_ from the table.
  static std::size_t compute_num_qubits(
      const std::vector<SparsePauliWord>& table);
};

}  // namespace qdk::chemistry::data

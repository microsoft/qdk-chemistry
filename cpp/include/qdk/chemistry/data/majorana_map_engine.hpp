// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <complex>
#include <cstdint>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/pauli_operator.hpp>
#include <string>
#include <utility>
#include <vector>

namespace qdk::chemistry::data {

/**
 * @brief Result of a Majorana-loop fermion-to-qubit mapping.
 *
 * Contains Pauli strings in little-endian format (qubit 0 = rightmost char)
 * and their corresponding complex coefficients.
 */
struct MajoranaMapResult {
  /// Pauli strings in little-endian format.
  std::vector<std::string> pauli_strings;
  /// Complex coefficients (same length as pauli_strings).
  std::vector<std::complex<double>> coefficients;
};

/**
 * @brief Map a fermionic Hamiltonian to qubit Pauli terms using Majorana loops.
 *
 * This is the encoding-agnostic mapping engine. It decomposes each fermionic
 * operator (a†_p a_q, a†_p a†_r a_s a_q) into Majorana products using
 * compile-time coefficients, looks up each Majorana γ_k in the mapping table,
 * and multiplies/accumulates the resulting Pauli strings.
 *
 * The second-quantized Hamiltonian in chemist notation is:
 *   H = E_core
 *     + Σ_{pq,σ} h_pq a†_{pσ} a_{qσ}
 *     + (1/2) Σ_{pqrs,σ,τ} (pq|rs) a†_{pσ} a†_{rτ} a_{sτ} a_{qσ}
 *
 * Using the identity a†_p a†_r a_s a_q = E_pq·E_rs - δ_{qr}·E_ps,
 * where E_pq = a†_p a_q, and each E_pq is decomposed as:
 *   E_pq = (1/4) Σ_{a,b} c[a][b] · γ_{2p+a} · γ_{2q+b}
 * with c[a][b] = (-i)^a · (i)^b.
 *
 * @param mapping The Majorana-to-Pauli mapping (encoding).
 * @param core_energy The core (nuclear repulsion + frozen core) energy.
 * @param h1_alpha One-body integrals for alpha spin (n_spatial × n_spatial).
 * @param h1_beta One-body integrals for beta spin (n_spatial × n_spatial).
 * @param eri_aaaa Flattened two-body integrals (αα|αα) in chemist notation.
 * @param eri_aabb Flattened two-body integrals (αα|ββ) in chemist notation.
 * @param eri_bbbb Flattened two-body integrals (ββ|ββ) in chemist notation.
 * @param n_spatial Number of spatial orbitals.
 * @param is_restricted Whether h1_alpha == h1_beta (spin-free case).
 * @param threshold Pauli terms with |coeff| < threshold are dropped.
 * @param integral_threshold Integrals with |value| < this are skipped.
 * @return MajoranaMapResult with Pauli strings (little-endian) and
 * coefficients.
 */
MajoranaMapResult majorana_map_hamiltonian(
    const MajoranaMapping& mapping, double core_energy, const double* h1_alpha,
    const double* h1_beta, const double* eri_aaaa, const double* eri_aabb,
    const double* eri_bbbb, std::size_t n_spatial, bool is_restricted,
    double threshold, double integral_threshold);

}  // namespace qdk::chemistry::data

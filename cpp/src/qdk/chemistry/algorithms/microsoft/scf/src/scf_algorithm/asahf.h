// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/molecule.h>

#include "diis.h"

namespace qdk::chemistry::scf {

namespace detail {
/**
 * @brief Hash function for BasisSet used in this atomic guess implementation
 */
struct BasisHasher {
  size_t operator()(const BasisSet& basis) const noexcept;
};

/**
 * @brief Equality checker for BasisSet used in this atomic guess
 * implementation
 */
struct BasisEqChecker {
  bool operator()(const BasisSet& lhs, const BasisSet& rhs) const noexcept;
};

/**
 * @brief Type alias for map from BasisSet to its corresponding atomic density
 * matrix
 */
using BasisSetMap =
    std::unordered_map<BasisSet, RowMajorMatrix, BasisHasher, BasisEqChecker>;
}  // namespace detail

/**
 * @brief Generate atomic density matrix guess using ASAHF for each atom
 * @param basis_set Basis set for the molecule
 * @param mol Molecular structure
 * @param tD Output density matrix
 *
 * References:
 *     Almlöf, J.; Faegri Jr., K.; Korsell., K. (1982). "Principles for a direct
 *     SCF approach to LICAO–MO ab-initio calculations"
 *     J. Comput. Chem.
 *     10.1002/jcc.540030314.
 *
 *     Van Lenthe, J. H.; Zwaans, R.;  Van Dam, H. J. J.; Guest, M. F. (2006).
 *     "Starting scf calculations by superposition of atomic densities"
 *     J. Comput. Chem.
 *     10.1002/jcc.20393
 *
 * See also:
 *     qdk::chemistry::constants::ATOMIC_CONFIGURATION for the atomic electron
 *     configurations used in this method.
 */
void get_atom_guess(const BasisSet& basis_set, const Molecule& mol,
                    RowMajorMatrix& tD);

/**
 * @brief Atomic Spherically Averaged Hartree-Fock (ASAHF) SCF solver
 *
 * This class implements an algorithm for spin-restricted atomic SCF
 * calculations using spherically averaged Fock matrices.
 */
class AtomicSphericallyAveragedHartreeFock : public DIIS {
 public:
  /**
   * @brief Construct ASAHF solver
   * @param ctx SCF context containing molecule, config, and basis sets
   * @param subspace_size DIIS subspace size (default: 8)
   */
  AtomicSphericallyAveragedHartreeFock(const SCFContext& ctx,
                                       size_t subspace_size = 8);

  /**
   * @brief Solve the eigenvalue problem for the Fock matrix and update
   * eigenvalues, molecular coefficients, and density matrix
   *
   * Solves the generalized eigenvalue problem F*C = S*C*E using the
   * orthogonalization matrix to transform to an orthogonal basis.
   *
   * @param[in] F Fock matrix to diagonalize
   * @param[in] S Overlap matrix
   * @param[in] X Orthogonalization matrix (num_atomic_orbitals ×
   * num_molecular_orbitals)
   * @param[out] C Molecular orbital coefficients
   * @param[out] eigenvalues Orbital eigenvalues
   * @param[out] P Density matrix
   * @param[in] num_occupied_orbitals Number of occupied orbitals per spin
   * [alpha, beta]
   * @param[in] num_atomic_orbitals Number of atomic orbitals
   * @param[in] num_molecular_orbitals Number of molecular orbitals
   * @param[in] idx_spin Density matrix index (0 for alpha or restricted, 1 for
   * beta)
   * @param[in] unrestricted Whether calculation is unrestricted
   */
  void solve_fock_eigenproblem(const RowMajorMatrix& F, const RowMajorMatrix& S,
                               const RowMajorMatrix& X, RowMajorMatrix& C,
                               RowMajorMatrix& eigenvalues, RowMajorMatrix& P,
                               const int num_occupied_orbitals[2],
                               int num_atomic_orbitals,
                               int num_molecular_orbitals, int idx_spin,
                               bool unrestricted) override;

 private:
  /**
   * @brief Custom orthogonalization matrix computation for ASAHF
   * @param S_ Overlap matrix
   * @param ret Output orthogonalization matrix
   * @param n_atom_orbs Number of atomic orbitals
   * @param n_mol_orbs Number of molecular orbitals
   */
  void compute_orthogonalization_matrix_(const RowMajorMatrix& S_,
                                         RowMajorMatrix* ret,
                                         size_t n_atom_orbs);
};

}  // namespace qdk::chemistry::scf

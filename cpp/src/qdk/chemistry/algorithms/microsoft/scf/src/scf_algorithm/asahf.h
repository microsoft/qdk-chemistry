#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/molecule.h>

#include "diis.h"

namespace qdk::chemistry::scf {
/**
 * @breif Generate atomic density matrix guess using ASAHF for each atom
 * @param basis_set Basis set for the molecule
 * @param mol Molecular structure
 * @param tD Output density matrix
 */
void get_atom_guess(const BasisSet& basis_set, const Molecule& mol,
                    RowMajorMatrix& tD);

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
  void custom_compute_orthogonalization_matrix_(const RowMajorMatrix& S_,
                                                RowMajorMatrix* ret,
                                                size_t n_atom_orbs,
                                                size_t n_mol_orbs);
};

}  // namespace qdk::chemistry::scf
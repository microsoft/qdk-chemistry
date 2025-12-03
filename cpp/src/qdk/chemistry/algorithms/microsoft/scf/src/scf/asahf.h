#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/molecule.h>

#include "scf_impl.h"

namespace qdk::chemistry::scf {
/**
 * @breif Generate atomic density matrix guess using ASAHF for each atom
 * @param basis_set Basis set for the molecule
 * @param mol Molecular structure
 * @param tD Output density matrix
 */
void get_atom_guess(const BasisSet& basis_set, const Molecule& mol,
                    RowMajorMatrix& tD);
namespace detail {

class AtomicSphericallyAveragedHartreeFock : public SCFImpl {
 public:
  /**
   * @brief Construct ASAHF solver
   * @param mol Molecular structure
   * @param cfg SCF configuration
   * @param basis_set Basis set to use
   * @param raw_basis_set Raw (unnormalized) basis set for output
   * @param delay_eri If true, delay ERI initialization to derived constructor
   * (default: false)
   */
  AtomicSphericallyAveragedHartreeFock(std::shared_ptr<Molecule> mol,
                                       const SCFConfig& cfg,
                                       std::shared_ptr<BasisSet> basis_set,
                                       std::shared_ptr<BasisSet> raw_basis_set,
                                       bool delay_eri = false);

  /**
   * @brief Execute the ASAHF calculation
   * @return SCF context with results
   */
  const SCFContext& compute();

 private:
  /** @brief Perform the SCF iteration */
  void _iter();
  /** @brief Update Fock matrix based on current density */
  void _update_f();
  /**
   * @brief Update density matrix based on current Fock matrix
   * @param ini_fock Initial Fock matrix
   * @param idx Index of density matrix to update (unused)
   */
  void update_density_matrix_(const RowMajorMatrix& fock, int idx = 0) override;
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

  /** @brief Molecule */
  std::shared_ptr<Molecule> mol_;
};
}  // namespace detail

}  // namespace qdk::chemistry::scf
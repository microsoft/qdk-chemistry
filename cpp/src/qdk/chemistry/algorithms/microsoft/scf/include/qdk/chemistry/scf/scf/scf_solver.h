// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/core/scf.h>
#include <qdk/chemistry/scf/core/types.h>

#include <memory>
#include <vector>

namespace qdk::chemistry::scf {

// Forward declarations
class SCFImpl;

/**
 * @brief Self-Consistent Field (SCF) solver
 *
 * Provides interface for Hartree-Fock (HF) and Kohn-Sham (KS) DFT calculations.
 */
class SCF {
 public:
  /**
   * @brief Virtual destructor
   */
  virtual ~SCF() noexcept;

  /**
   * @brief Execute the SCF calculation
   *
   * Performs the iterative SCF optimization until convergence or max
   * iterations.
   *
   * @return Reference to the SCF context containing results
   */
  const SCFContext& run();

  /**
   * @brief Get the current SCF context
   *
   * @return Const reference to SCF context (configuration and results)
   */
  const SCFContext& context() const;

  /**
   * @brief Get all available system matrices (Fock, density, orbitals, etc.)
   *
   * @return Vector of (name, matrix) pairs for all matrices
   */
  std::vector<std::pair<std::string, const RowMajorMatrix&>> get_matrices()
      const;

  /**
   * @brief Get the overlap matrix
   *
   * @return Const reference to overlap matrix S (size: (nao, nao))
   *
   * @note ``nao`` is the number of atomic orbitals (atomic orbitals)
   */
  const RowMajorMatrix& overlap() const;

  // methods for output:
  /**
   * @brief Check if calculation is restricted (RHF/RKS) or unrestricted
   * (UHF/UKS)
   *
   * @return true for restricted, false for unrestricted
   */
  bool get_restricted() const;

  /**
   * @brief Get number of electrons
   *
   * @return Vector with electron counts (size 1 for restricted: [total], size 2
   * for unrestricted: [alpha, beta])
   */
  std::vector<int> get_num_electrons() const;

  /**
   * @brief Get number of atomic orbitals
   *
   * @return Number of atomic orbital atomic orbitals
   */
  int get_num_atomic_orbitals() const;

  /**
   * @brief Get number of molecular orbitals

   * @return Number of molecular orbitals (num_molecular_orbitals)
   */
  int get_num_molecular_orbitals() const;

  /**
   * @brief Get orbital eigenvalues (energies)
   *
   * @return Matrix of orbital eigenvalues (size: (ndm, nmo))
   *
   * @note ``ndm`` is the number of density matrices (1 for restricted, 2 for
   *       unrestricted); ``nmo`` is the number of molecular orbitals
   */
  const RowMajorMatrix& get_eigenvalues() const;

  /**
   * @brief Get the density matrix
   *
   * @return Const reference to density matrix D in AO basis (size: (ndm, nao,
   * nao))
   *
   * @note ``ndm`` is the number of density matrices (1 for restricted, 2 for
   *       unrestricted); ``nao`` is the number of atomic orbitals (basis
   *       functions)
   */
  const RowMajorMatrix& get_density_matrix() const;

  /**
   * @brief Get the Fock matrix
   *
   * @return Const reference to Fock matrix F in AO basis (size: (ndm, nao,
   * nao))
   *
   * @note ``ndm`` is the number of density matrices (1 for restricted, 2 for
   *       unrestricted); ``nao`` is the number of atomic orbitals (basis
   *       functions)
   */
  const RowMajorMatrix& get_fock_matrix() const;

  /**
   * @brief Get the molecular orbital coefficient matrix
   *
   * @return Const reference to MO coefficient matrix C (size: (ndm, nao, nmo))
   *
   * @note ``ndm`` is the number of density matrices (1 for restricted, 2 for
   *       unrestricted); ``nao`` is the number of atomic orbitals (basis
   *       functions); * ``nmo`` is the number of molecular orbitals
   */
  const RowMajorMatrix& get_orbitals_matrix() const;

  /**
   * @brief Get number of density matrices
   *
   * @return 1 for restricted calculations, 2 for unrestricted (alpha and beta)
   */
  int get_num_density_matrices() const;

  /**
   * @brief Create a Hartree-Fock solver
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration
   * @return Unique pointer to SCF solver
   */
  static std::unique_ptr<SCF> make_hf_solver(std::shared_ptr<Molecule> mol,
                                             const SCFConfig& cfg);

  /**
   * @brief Create a Hartree-Fock solver with user-provided initial density
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration
   * @param density_matrix Initial density matrix guess
   * @return Unique pointer to SCF solver
   */
  static std::unique_ptr<SCF> make_hf_solver(
      std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
      const RowMajorMatrix& density_matrix, std::shared_ptr<BasisSet> basis_set,
      std::shared_ptr<BasisSet> raw_basis_set);

  /**
   * @brief Create a Hartree-Fock solver with user-provided initial density
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration
   * @param basis_set Basis set to use
   * @param raw_basis_set Raw (un-normalized) basis set to use
   * @return Unique pointer to SCF solver
   */
  static std::unique_ptr<SCF> make_hf_solver(
      std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
      std::shared_ptr<BasisSet> basis_set,
      std::shared_ptr<BasisSet> raw_basis_set);

  /**
   * @brief Create a Kohn-Sham DFT solver
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration (must include XC functional settings)
   * @return Unique pointer to SCF solver
   */
  static std::unique_ptr<SCF> make_ks_solver(std::shared_ptr<Molecule> mol,
                                             const SCFConfig& cfg);

  /**
   * @brief Create a Kohn-Sham DFT solver with user-provided initial density
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration (must include XC functional settings)
   * @param density_matrix Initial density matrix guess
   * @return Unique pointer to SCF solver
   */
  static std::unique_ptr<SCF> make_ks_solver(
      std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
      const RowMajorMatrix& density_matrix, std::shared_ptr<BasisSet> basis_set,
      std::shared_ptr<BasisSet> raw_basis_set);

  /**
   * @brief Create a Kohn-Sham DFT solver with user-provided initial density
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration (must include XC functional settings)
   * @param basis_set Basis set to use
   * @param raw_basis_set Basis set to use
   * @return Unique pointer to SCF solver
   */
  static std::unique_ptr<SCF> make_ks_solver(
      std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
      std::shared_ptr<BasisSet> basis_set,
      std::shared_ptr<BasisSet> raw_basis_set);

 private:
  /**
   * @brief Private constructor
   *
   * @param impl Unique pointer to implementation object
   */
  SCF(std::unique_ptr<SCFImpl> impl);

  std::unique_ptr<SCFImpl> impl_;  ///< Pointer to implementation
};
}  // namespace qdk::chemistry::scf

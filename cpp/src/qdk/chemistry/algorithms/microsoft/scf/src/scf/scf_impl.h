// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/eri.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/core/scf.h>
#include <qdk/chemistry/scf/core/scf_algorithm.h>
#include <qdk/chemistry/scf/core/types.h>
#include <qdk/chemistry/scf/util/int1e.h>

#include <memory>
#include <source_location>

#ifdef QDK_CHEMISTRY_ENABLE_PCM
#include "pcm/pcm.h"
#endif

#include "scf/soad.h"

namespace qdk::chemistry::scf {

/**
 * @brief SCF implementation base class
 *
 * Handles the core SCF iteration logic, density matrix updates,
 * and Fock matrix construction for both HF and DFT calculations.
 *
 * Without extension, provides all functionality for Hartree-Fock SCF.
 */
class SCFImpl {
 public:
  /**
   * @brief Construct SCF implementation
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration
   * @param delay_eri If true, delay ERI initialization to derived constructor
   * (default: false)
   */
  SCFImpl(std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
          bool delay_eri = false);

  /**
   * @brief Construct SCF implementation with initial density matrix
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration
   * @param density_matrix Initial density matrix guess
   * @param delay_eri If true, delay ERI initialization to derived constructor
   * (default: false)
   */
  SCFImpl(std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
          const RowMajorMatrix& density_matrix, bool delay_eri = false);

  /**
   * @brief Construct SCF implementation with initial density matrix
   *
   * @param mol Molecular structure
   * @param cfg SCF configuration
   * @param basis_set Basis set to use
   * @param raw_basis_set Raw (unnormalized) basis set for output
   * @param delay_eri If true, delay ERI initialization to derived constructor
   * (default: false)
   * @param skip_verify If true, skip input verification checks (default: false)
   */
  SCFImpl(std::shared_ptr<Molecule> mol, const SCFConfig& cfg,
          std::shared_ptr<BasisSet> basis_set,
          std::shared_ptr<BasisSet> raw_basis_set, bool delay_eri = false,
          bool skip_verify = false);

  /**
   * @brief Virtual destructor
   */
  virtual ~SCFImpl() {}

  /**
   * @brief Execute the SCF calculation
   * @see SCF::run() for API details
   */
  const SCFContext& run();

  /**
   * @brief Get the SCF context
   * @see SCF::context() for API details
   */
  const SCFContext& context() const { return ctx_; }

  /**
   * @brief Get all SCF matrices
   * @see SCF::get_matrices() for API details
   */
  virtual std::vector<std::pair<std::string, const RowMajorMatrix&>>
  get_matrices() const {
    return {{"F", F_}, {"P", P_}, {"C", C_}, {"eigenvalues", eigenvalues_}};
  }

  /**
   * @brief Get the overlap matrix
   * @see SCF::overlap() for API details
   */
  const RowMajorMatrix& overlap() const { return S_; }

  // methods for output:
  /**
   * @brief Check if calculation is restricted
   * @see SCF::get_restricted() for API details
   */
  bool get_restricted() const { return num_density_matrices_ == 1; }

  /**
   * @brief Get number of electrons
   * @see SCF::get_num_electrons() for API details
   */
  std::vector<int> get_num_electrons() const { return {nelec_[0], nelec_[1]}; }

  /**
   * @brief Get number of atomic orbital atomic orbitals
   * @see SCF::get_num_atomic_orbitals() for API details
   */
  int get_num_atomic_orbitals() const { return num_atomic_orbitals_; }

  /**
   * @brief Get number of molecular orbitals
   * @see SCF::get_num_molecular_orbitals() for API details
   */
  int get_num_molecular_orbitals() const { return num_molecular_orbitals_; }

  /**
   * @brief Get orbital eigenvalues (energies)
   * @see SCF::get_eigenvalues() for API details
   */
  const RowMajorMatrix& get_eigenvalues() const { return eigenvalues_; }

  /**
   * @brief Get the density matrix
   * @see SCF::get_density_matrix() for API details
   */
  const RowMajorMatrix& get_density_matrix() const { return P_; }

  /**
   * @brief Get the Fock matrix
   * @see SCF::get_fock_matrix() for API details
   */
  const RowMajorMatrix& get_fock_matrix() const { return F_; }

  /**
   * @brief Get the molecular orbital coefficient matrix
   * @see SCF::get_orbitals_matrix() for API details
   */
  const RowMajorMatrix& get_orbitals_matrix() const { return C_; }

  /**
   * @brief Get number of density matrices
   * @see SCF::get_num_density_matrices() for API details
   */
  int get_num_density_matrices() const { return num_density_matrices_; }

  /**
   * @brief Get the orthogonalization matrix
   * @return Reference to the orthogonalization matrix X
   */
  const RowMajorMatrix& get_orthogonalization_matrix() const { return X_; }

  /**
   * @brief Get mutable reference to density matrix for modification
   * @return Mutable reference to density matrix P
   */
  RowMajorMatrix& density_matrix() { return P_; }

  /**
   * @brief Get mutable reference to molecular orbital coefficients for
   * modification
   * @return Mutable reference to coefficient matrix C
   */
  RowMajorMatrix& orbitals_matrix() { return C_; }

  /**
   * @brief Get mutable reference to orbital eigenvalues for modification
   * @return Mutable reference to eigenvalues
   */
  RowMajorMatrix& eigenvalues() { return eigenvalues_; }

  /**
   * @brief Evaluate total energy and Fock matrix for a trial density matrix.
   * This is a const method that does not modify any member variables, but it
   * reads from existing state (e.g., H_, ctx_, eri_), so the SCFImpl instance
   * must be fully initialized before calling it.
   *
   * @param P_matrix Trial density matrix
   * @param loc Source location of the caller (automatically captured)
   * @return std::pair containing:
   *   - first: total energy in Hartree
   *   - second: Fock matrix in AO basis
   */
  virtual std::pair<double, RowMajorMatrix>
  evaluate_trial_density_energy_and_fock(
      const RowMajorMatrix& P_matrix,
      const std::source_location& loc = std::source_location::current()) const;

 protected:
  /**
   * @brief Build one-electron integrals (overlap, kinetic, nuclear attraction)
   */
  void build_one_electron_integrals_();

  /**
   * @brief Compute orthogonalization matrix from overlap matrix
   *
   * @param S Overlap matrix
   * @param ret Output orthogonalization matrix X
   */
  void compute_orthogonalization_matrix_(const RowMajorMatrix& S,
                                         RowMajorMatrix* ret);

  /**
   * @brief Initialize the density matrix using configured guess method
   */
  void init_density_matrix_();

  /**
   * @brief Calculate nuclear repulsion energy
   *
   * @return Nuclear-nuclear repulsion energy (Hartree)
   */
  double calc_nuclear_repulsion_energy_();

  /**
   * @brief Run SCF iteration using the configured algorithm
   */
  void iterate_();

  /**
   * @brief Calculate molecular properties (dipole, quadrupole, populations)
   */
  void properties_();

  /**
   * @brief Update the Fock matrix (virtual method for HF/DFT specialization)
   *
   * Computes F = H + J + K (HF) or F = H + J + K + Vxc (DFT - derived
   * implementation)
   */
  virtual void update_fock_();

  /**
   * @brief Reset Fock matrix to initial state
   *
   * Used for incremental Fock builds when reset is triggered.
   */
  virtual void reset_fock_();

  /**
   * @brief Get hybridization coefficients for range-separated functionals
   *
   * @returns A tuple containing the coefficients
   *   0: alpha Fraction of long-range HF exchange (0.0 to 1.0)
   *   1: beta Fraction of short-range HF exchange (for RSH functionals)
   *   2: omega Range-separation parameter (for RSH functionals)
   *
   * @note For pure DFT functionals, alpha = beta = 0.0
   */
  virtual std::tuple<double, double, double> get_hyb_coeff_() const;

  /**
   * @brief Calculate total energy
   *
   * @return Total SCF energy (Hartree)
   */
  virtual double total_energy_();

  /**
   * @brief Get analytical energy gradients
   *
   * @return Reference to gradient vector (3*natoms components)
   */
  const std::vector<double>& get_gradients_();

  /**
   * @brief Get XC potential gradient contribution (virtual for DFT override)
   *
   * @return XC gradient matrix (for DFT), empty for HF
   */
  virtual const RowMajorMatrix get_vxc_grad_() const;

  /**
   * @brief Calculate static polarizability tensor
   */
  void polarizability_();

  /**
   * @brief Solve coupled-perturbed SCF equations
   *
   * Solves the CPSCF equations for response to perturbation R.
   *
   * @param R_input Right-hand side (perturbation matrix)
   * @param X_sol Output: response vector/matrix
   */
  void cpscf_(const double* R_input, double* X_sol);

  /**
   * @brief Update trial Fock matrix for CPSCF iterations
   */
  virtual void update_trial_fock_();

  // Trial density matrices for CPSCF
  RowMajorMatrix tP_;     ///< Trial density matrix for CPSCF
  RowMajorMatrix tJ_;     ///< Trial Coulomb matrix for CPSCF
  RowMajorMatrix tK_;     ///< Trial exchange matrix for CPSCF
  RowMajorMatrix tFock_;  ///< Trial Fock matrix for CPSCF

  /**
   * @brief Write gradients to output
   *
   * @param gradients Gradient vector to write
   * @param mol Molecular structure (optional)
   */
  static void write_gradients_(const std::vector<double>& gradients,
                               const Molecule* mol = nullptr);

  // Core SCF data
  size_t num_atomic_orbitals_;     ///< Number of atomic orbital atomic orbitals
  size_t num_molecular_orbitals_;  ///< Number of molecular orbitals
  int num_density_matrices_;  ///< Number of density matrices (1=restricted,
                              ///< 2=unrestricted)
  int nelec_[2];              ///< Number of [alpha, beta] electrons

  // SCF matrices
  RowMajorMatrix H_;  ///< Core Hamiltonian matrix (num_density_matrices ×
                      ///< num_atomic_orbitals × num_atomic_orbitals)
  RowMajorMatrix
      S_;  ///< Overlap matrix (num_atomic_orbitals × num_atomic_orbitals)
  RowMajorMatrix P_;  ///< Density matrix (num_density_matrices ×
                      ///< num_atomic_orbitals × num_atomic_orbitals)
  RowMajorMatrix J_;  ///< Coulomb matrix (num_density_matrices ×
                      ///< num_atomic_orbitals × num_atomic_orbitals)
  RowMajorMatrix K_;  ///< Exchange matrix (num_density_matrices ×
                      ///< num_atomic_orbitals × num_atomic_orbitals)
  RowMajorMatrix F_;  ///< Fock matrix (num_density_matrices ×
                      ///< num_atomic_orbitals × num_atomic_orbitals)
  RowMajorMatrix X_;  ///< Orthogonalization matrix (num_atomic_orbitals ×
                      ///< num_molecular_orbitals)
  RowMajorMatrix
      C_;  ///< Molecular orbital coefficient matrix (num_density_matrices ×
           ///< num_atomic_orbitals × num_molecular_orbitals)
  RowMajorMatrix eigenvalues_;  ///< Orbital energies (num_density_matrices ×
                                ///< num_molecular_orbitals)
#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
  RowMajorMatrix disp_grad_;  ///< Dispersion correction gradient
#endif

  SCFContext
      ctx_;  ///< SCF calculation context (config, molecule, basis, results)
  std::unique_ptr<OneBodyIntegral>
      int1e_;                 ///< One-electron integral calculator
  std::shared_ptr<ERI> eri_;  ///< Electron repulsion integral calculator
  std::shared_ptr<SCFAlgorithm> scf_algorithm_;  ///< SCF convergence algorithm

#ifdef QDK_CHEMISTRY_ENABLE_PCM
  std::unique_ptr<pcm::PCM> pcm_;  ///< Polarizable Continuum Model solver
  RowMajorMatrix Vpcm_;  ///< PCM potential matrix (num_atomic_orbitals ×
                         ///< num_atomic_orbitals)
#endif

#ifdef QDK_CHEMISTRY_ENABLE_QMMM
  bool add_mm_charge_;   ///< Whether to include MM point charges in calculation
  RowMajorMatrix T_mm_;  ///< QM-MM interaction matrix (num_atomic_orbitals ×
                         ///< num_atomic_orbitals)
#endif

  bool density_matrix_initialized_;  ///< Whether density matrix has been
                                     ///< initialized from input MO
};
}  // namespace qdk::chemistry::scf

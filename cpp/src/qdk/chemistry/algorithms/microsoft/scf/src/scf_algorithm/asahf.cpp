// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "asahf.h"

#include <qdk/chemistry/scf/config.h>
#ifdef QDK_CHEMISTRY_ENABLE_MPI
#include <mpi.h>
#endif
#include <spdlog/spdlog.h>

#include <lapack.hh>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../scf/scf_impl.h"
#include "util/macros.h"

namespace qdk::chemistry::scf {

// non-relativistic spin-restricted spherical HF configurations from
// https://arxiv.org/pdf/1908.02528
// Each entry corresponds to an element with atomic number Z
// and contains the number of electrons in each subshell: s, p, d, f
std::vector<std::array<size_t, 4>> CONFIGURATION = {
    {0, 0, 0, 0},     {1, 0, 0, 0},     {2, 0, 0, 0},     {3, 0, 0, 0},
    {4, 0, 0, 0},     {4, 1, 0, 0},     {4, 2, 0, 0},     {4, 3, 0, 0},
    {4, 4, 0, 0},     {4, 5, 0, 0},     {4, 6, 0, 0},     {5, 6, 0, 0},
    {6, 6, 0, 0},     {6, 7, 0, 0},     {6, 8, 0, 0},     {6, 9, 0, 0},
    {6, 10, 0, 0},    {6, 11, 0, 0},    {6, 12, 0, 0},    {7, 12, 0, 0},
    {8, 12, 0, 0},    {8, 13, 0, 0},    {8, 12, 2, 0},    {8, 12, 3, 0},
    {8, 12, 4, 0},    {6, 12, 7, 0},    {6, 12, 8, 0},    {6, 12, 9, 0},
    {6, 12, 10, 0},   {7, 12, 10, 0},   {8, 12, 10, 0},   {8, 13, 10, 0},
    {8, 14, 10, 0},   {8, 15, 10, 0},   {8, 16, 10, 0},   {8, 17, 10, 0},
    {8, 18, 10, 0},   {9, 18, 10, 0},   {10, 18, 10, 0},  {10, 19, 10, 0},
    {10, 18, 12, 0},  {10, 18, 13, 0},  {8, 18, 16, 0},   {8, 18, 17, 0},
    {8, 18, 18, 0},   {8, 18, 19, 0},   {8, 18, 20, 0},   {9, 18, 20, 0},
    {10, 18, 20, 0},  {10, 19, 20, 0},  {10, 20, 20, 0},  {10, 21, 20, 0},
    {10, 22, 20, 0},  {10, 23, 20, 0},  {10, 24, 20, 0},  {11, 24, 20, 0},
    {12, 24, 20, 0},  {12, 24, 21, 0},  {12, 24, 22, 0},  {12, 24, 21, 2},
    {12, 24, 20, 4},  {12, 24, 20, 5},  {12, 24, 20, 6},  {12, 24, 20, 7},
    {11, 24, 20, 9},  {10, 24, 20, 11}, {10, 24, 20, 12}, {10, 24, 20, 13},
    {10, 24, 20, 14}, {11, 24, 20, 14}, {12, 24, 20, 14}, {12, 25, 20, 14},
    {12, 24, 22, 14}, {12, 24, 23, 14}, {10, 24, 26, 14}, {10, 24, 27, 14},
    {10, 24, 28, 14}, {10, 24, 29, 14}, {10, 24, 30, 14}, {11, 24, 30, 14},
    {12, 24, 30, 14}, {12, 25, 30, 14}, {12, 26, 30, 14}, {12, 27, 30, 14},
    {12, 28, 30, 14}, {12, 29, 30, 14}, {12, 30, 30, 14}, {13, 30, 30, 14},
    {14, 30, 30, 14}, {14, 30, 31, 14}, {14, 30, 32, 14}, {14, 30, 30, 17},
    {14, 30, 30, 18}, {14, 30, 30, 19}, {13, 30, 30, 21}, {12, 30, 30, 23},
    {12, 30, 30, 24}, {12, 30, 30, 25}, {12, 30, 30, 26}, {12, 30, 30, 27},
    {12, 30, 30, 28}, {13, 30, 30, 28}, {14, 30, 30, 28}, {14, 30, 31, 28},
    {14, 30, 32, 28}, {14, 30, 33, 28}, {12, 30, 36, 28}, {12, 30, 37, 28},
    {12, 30, 38, 28}, {12, 30, 39, 28}, {12, 30, 40, 28}, {13, 30, 40, 28},
    {14, 30, 40, 28}, {14, 31, 40, 28}, {14, 32, 40, 28}, {14, 33, 40, 28},
    {14, 34, 40, 28}, {14, 35, 40, 28}, {14, 36, 40, 28}};

namespace detail {

/**
 *  @brief Get the number of fully occupied and fractionally occupied
 *  orbitals for a given angular momentum and nuclear charge.
 *  @param l Angular momentum quantum number
 *  @param nuc_charge Nuclear charge of the atom
 *  @return A tuple containing the number of fully occupied orbitals and the
 *  fractional occupation
 */
std::tuple<size_t, double> get_num_frac_occ_orbs(size_t l, size_t nuc_charge) {
  if (nuc_charge >= CONFIGURATION.size()) {
    throw std::runtime_error(
        "Nuclear charge exceeds predefined configuration size.");
  }
  std::array<size_t, 4> config = CONFIGURATION[nuc_charge];
  if (l < 4 && config[l] > 0) {
    double nelec = config[l];
    double n_spin_orbs = 2 * (2 * l + 1);
    size_t n_double_occ = floor(nelec / n_spin_orbs);
    double frac_occ = (nelec / n_spin_orbs - n_double_occ) * 2;
    return std::make_tuple(n_double_occ, frac_occ);
  }
  return std::make_tuple(0, 0);
}

/**
 * @brief Create a molecule structure for a single atom
 * @param atomic_number Atomic number of the atom
 * @return Shared pointer to the created Molecule
 */
std::shared_ptr<Molecule> make_atomic_molecule(int atomic_number) {
  auto mol = std::make_shared<Molecule>();
  mol->n_atoms = 1;
  mol->total_nuclear_charge = atomic_number;
  mol->n_electrons = atomic_number;
  mol->atomic_nums = {static_cast<uint64_t>(atomic_number)};
  mol->atomic_charges = {static_cast<uint64_t>(atomic_number)};
  mol->coords = {{0.0, 0.0, 0.0}};
  mol->charge = 0;
  if (atomic_number % 2 == 0)
    mol->multiplicity = 1;
  else
    mol->multiplicity = 2;
  return mol;
}

/**
 * @brief Create a basis set for a single atom from a molecular basis set
 * @param index Index of the atom in the molecule
 * @param basis_set Molecular basis set
 * @param mol atomic structure
 * @return Shared pointer to the created atomic BasisSet
 */
std::shared_ptr<BasisSet> make_atom_basis_set(int index,
                                              const BasisSet& basis_set,
                                              std::shared_ptr<Molecule> mol) {
  std::vector<Shell> shells;
  std::vector<Shell> ecp_shells;
  int total_ecp_electrons = 0;
  std::unordered_map<int, int> ecp_electrons;

  // Filter shells belonging to the specified atomic number
  for (const auto& shell : basis_set.shells) {
    if (shell.atom_index == index) {
      Shell tmp_shell = shell;
      tmp_shell.O = {0.0, 0.0, 0.0};  // reset center for single atom
      tmp_shell.atom_index = 0;       // reset atom index for single atom
      shells.push_back(tmp_shell);
    }
  }

  // Filter ECP shells belonging to the specified atomic number
  for (const auto& shell : basis_set.ecp_shells) {
    if (shell.atom_index == index) {
      Shell tmp_shell = shell;
      tmp_shell.atom_index = 0;       // reset atom index for single atom
      tmp_shell.O = {0.0, 0.0, 0.0};  // reset center for single atom
      ecp_shells.push_back(tmp_shell);
    }
  }

  // get element from mol to get the ecp electrons from map
  auto atomic_number = mol->atomic_nums[0];
  if (basis_set.element_ecp_electrons.find(atomic_number) !=
      basis_set.element_ecp_electrons.end()) {
    ecp_electrons[atomic_number] =
        basis_set.element_ecp_electrons.at(atomic_number);
    total_ecp_electrons = ecp_electrons[atomic_number];
  }

  // Create a new BasisSet for the atom
  return std::shared_ptr<BasisSet>(
      new BasisSet(mol, shells, ecp_shells, ecp_electrons, total_ecp_electrons,
                   basis_set.mode, basis_set.pure, true));
}
}  // namespace detail

void get_atom_guess(const BasisSet& basis_set, const Molecule& mol,
                    RowMajorMatrix& tD) {
  // check if basis set is canonical
  if (!basis_set.pure) {
    throw std::runtime_error("ASAHF initial guess requires a spherical basis.");
  }

  // make basic config
  SCFConfig cfg;
  cfg.mpi = qdk::chemistry::scf::mpi_default_input();
  cfg.scf_algorithm.max_iteration = 100;
  cfg.scf_algorithm.og_threshold = 1e-6;
  cfg.scf_algorithm.density_threshold = 1e-6;
  cfg.scf_algorithm.method = SCFAlgorithmName::ASAHF;
  cfg.density_init_method = DensityInitializationMethod::Core;
  cfg.require_gradient = false;
  cfg.unrestricted = false;
  cfg.require_polarizability = false;
  cfg.exc.xc_name = "hf";
  cfg.eri.method = ERIMethod::Libint2Direct;
  cfg.grad_eri = cfg.eri;
  cfg.grad_eri.method = ERIMethod::Libint2Direct;

  for (size_t i = 0, p = 0; i < mol.n_atoms; ++i) {
    auto atom_num = mol.atomic_nums[i];
    // create atomic molecule and basis set
    std::shared_ptr<Molecule> atom_mol =
        detail::make_atomic_molecule(static_cast<int>(atom_num));
    std::shared_ptr<BasisSet> atom_basis_set =
        detail::make_atom_basis_set(static_cast<int>(i), basis_set, atom_mol);
    // Create SCF solver with basis sets
    SCFImpl scf_solver(atom_mol, cfg, atom_basis_set, atom_basis_set, false,
                       true);
    // Run SCF with ASAHF algorithm
    const auto& asahf_ctx = scf_solver.run();
    const auto& dm = scf_solver.get_density_matrix();
    // insert atomic density matrix into total density matrix
    tD.block(p, p, dm.rows(), dm.cols()) = dm;
    p += dm.rows();
  }
}

AtomicSphericallyAveragedHartreeFock::AtomicSphericallyAveragedHartreeFock(
    const SCFContext& ctx, size_t subspace_size)
    : DIIS(ctx, subspace_size) {}

void AtomicSphericallyAveragedHartreeFock::solve_fock_eigenproblem(
    const RowMajorMatrix& F, const RowMajorMatrix& S, const RowMajorMatrix& X,
    RowMajorMatrix& C, RowMajorMatrix& eigenvalues, RowMajorMatrix& P,
    const int num_occupied_orbitals[2], int num_atomic_orbitals,
    int num_molecular_orbitals, int idx_spin, bool unrestricted) {
  Eigen::Map<RowMajorMatrix> P_dm(P.data(), num_atomic_orbitals,
                                  num_atomic_orbitals);
  Eigen::Map<RowMajorMatrix> C_dm(C.data(), num_atomic_orbitals,
                                  num_molecular_orbitals);

  // get max l from shells
  size_t max_l = 0;
  for (const auto& shell : ctx_.basis_set->shells) {
    if (shell.angular_momentum + 1 > max_l) {
      max_l = shell.angular_momentum + 1;
    }
  }

  // Build index map for each angular momentum
  std::vector<std::vector<size_t>> ao_indices_by_l(max_l);
  size_t ao_idx = 0;
  for (const auto& shell : ctx_.basis_set->shells) {
    size_t l = shell.angular_momentum;
    size_t nfunc = 2 * l + 1;  // number of basis functions for this shell
    for (size_t i = 0; i < nfunc; ++i) {
      ao_indices_by_l[l].push_back(ao_idx++);
    }
  }

  // Diagonalize each l-block of the spherically averaged Fock matrix
  size_t offset = 0;
  std::vector<RowMajorMatrix> mo_coeffs_by_l;
  for (int l = 0; l < max_l; ++l) {
    const auto& idx = ao_indices_by_l[l];
    if (idx.empty()) continue;

    size_t degeneracy = 2 * l + 1;
    size_t n_shells = idx.size() / degeneracy;
    size_t n_ao = n_shells * degeneracy;

    if (n_shells == 0) continue;

    RowMajorMatrix F_averaged = RowMajorMatrix::Zero(n_shells, n_shells);
    RowMajorMatrix S_averaged = RowMajorMatrix::Zero(n_shells, n_shells);

    // compute average values of degenerate blocks
    for (size_t i = 0; i < n_shells; ++i) {
      for (size_t j = 0; j < n_shells; ++j) {
        double fock_sum = 0.0;
        double overlap_sum = 0.0;
        for (size_t m1 = 0; m1 < degeneracy; ++m1) {
          fock_sum +=
              F(offset + i * degeneracy + m1, offset + j * degeneracy + m1);
          overlap_sum +=
              S(offset + i * degeneracy + m1, offset + j * degeneracy + m1);
        }
        double fock_avg = fock_sum / degeneracy;
        double overlap_avg = overlap_sum / degeneracy;
        F_averaged(i, j) = fock_avg;
        S_averaged(i, j) = overlap_avg;
      }
    }

    // get orthogonalization matrix
    RowMajorMatrix X_block = RowMajorMatrix::Zero(n_shells, n_shells);
    // use custom implementation to use custom dimensions
    custom_compute_orthogonalization_matrix_(S_averaged, &X_block, n_shells,
                                             n_shells);
    RowMajorMatrix tmp1 = X_block.transpose() * F_averaged;
    RowMajorMatrix tmp2 = tmp1 * X_block;

    std::vector<double> eigenvalues_block(n_shells, 0.0);

    lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, n_shells, tmp2.data(),
                 n_shells, eigenvalues_block.data());

    // lapack::syev returns column-major eigenvectors, transpose for row-major
    // storage
    tmp2.transposeInPlace();

    RowMajorMatrix C_tmp = X_block * tmp2;
    mo_coeffs_by_l.push_back(C_tmp);

    // update eigenvalues_
    for (size_t i = 0; i < eigenvalues_block.size(); ++i) {
      for (size_t m = 0; m < degeneracy; ++m) {
        eigenvalues(0, idx[i * degeneracy + m]) = eigenvalues_block[i];
      }
    }

    offset += n_ao;
  }

  // Populate C_dm
  size_t mo_offset = 0;
  size_t l_idx = 0;
  C_dm.setZero();
  for (int l = 0; l < max_l; ++l) {
    // Get AO indices for this angular momentum
    const auto& idx = ao_indices_by_l[l];
    if (idx.empty()) continue;

    size_t degeneracy = 2 * l + 1;
    size_t n_shells = idx.size() / degeneracy;
    if (n_shells == 0) continue;

    // Get MO coefficients for this l-block
    const auto& C_tmp = mo_coeffs_by_l[l_idx++];

    for (size_t i = 0; i < n_shells; ++i) {
      for (size_t m = 0; m < degeneracy; ++m) {
        size_t ao_row = idx[i * degeneracy + m];
        for (size_t j = 0; j < n_shells; ++j) {
          size_t mo_col = mo_offset + j * degeneracy + m;
          C_dm(ao_row, mo_col) = C_tmp(i, j);
        }
      }
    }

    mo_offset += n_shells * degeneracy;
  }

  // get fractional occupation
  std::vector<double> occupation;
  for (size_t l = 0; l < max_l; ++l) {
    const auto& idx = ao_indices_by_l[l];
    if (idx.empty()) continue;

    size_t degeneracy = 2 * l + 1;
    size_t n_shells = idx.size() / degeneracy;

    auto [n_double_occ, frac_occ] =
        detail::get_num_frac_occ_orbs(l, ctx_.mol->total_nuclear_charge);

    std::vector<double> occ_l(n_shells, 0);
    for (size_t i = 0; i < n_double_occ; ++i) {
      occ_l[i] = 2;
    }
    if (frac_occ > 0 && n_double_occ < n_shells) {
      occ_l[n_double_occ] = frac_occ;
    }
    for (size_t j = 0; j < occ_l.size(); ++j) {
      for (size_t i = 0; i < degeneracy; ++i) {
        occupation.push_back(occ_l[j]);
      }
    }
  }

  // Build density matrix
  P_dm.setZero();
  for (size_t mu = 0; mu < num_atomic_orbitals; ++mu) {
    for (size_t nu = 0; nu < num_atomic_orbitals; ++nu) {
      double density_value = 0.0;
      for (size_t m = 0; m < num_molecular_orbitals; ++m) {
        density_value += C_dm(mu, m) * C_dm(nu, m) * occupation[m];
      }
      P_dm(mu, nu) = density_value;
    }
  }
}

void AtomicSphericallyAveragedHartreeFock::
    custom_compute_orthogonalization_matrix_(const RowMajorMatrix& S_,
                                             RowMajorMatrix* ret,
                                             size_t n_atom_orbs,
                                             size_t n_mol_orbs) {
  RowMajorMatrix U_t(n_atom_orbs, n_atom_orbs);
  RowMajorMatrix s(n_atom_orbs, 1);
  std::memcpy(U_t.data(), S_.data(),
              n_atom_orbs * n_atom_orbs * sizeof(double));
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, n_atom_orbs, U_t.data(),
               n_atom_orbs, s.data());

  RowMajorMatrix U = U_t.transpose();

  auto threshold = ctx_.cfg->lindep_threshold;
  if (threshold < 0.0) threshold = s.maxCoeff() / 1e9;

  n_mol_orbs = 0;
  for (int i = n_atom_orbs - 1; i >= 0; --i) {
    if (s(i) >= threshold) n_mol_orbs++;
  }

  if (n_atom_orbs != n_mol_orbs) {
    spdlog::warn(
        "Orthogonalize: found linear dependency TOL={:.2e} "
        "n_atom_orbs={} "
        "n_mol_orbs={}",
        threshold, n_atom_orbs, n_mol_orbs);
  }

  auto sigma = s.bottomRows(n_mol_orbs);
  auto sigma_invsqrt = sigma.array().sqrt().inverse().matrix().asDiagonal();

  auto U_cond = U.block(0, n_atom_orbs - n_mol_orbs, n_atom_orbs, n_mol_orbs);
  RowMajorMatrix X_ = U_cond * sigma_invsqrt;
  *ret = X_;
}

}  // namespace qdk::chemistry::scf

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "asahf.h"

#include <qdk/chemistry/scf/config.h>

#include <qdk/chemistry/constants.hpp>
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

namespace detail {

/**
 */
std::array<size_t, 4> get_core_config_from_ecp_shells(
    const BasisSet& basis_set) {
  size_t ecp_electrons = basis_set.n_ecp_electrons;
  // ecp map
  std::unordered_map<int, std::array<size_t, 4>> ecp_map = {
      {0, {0, 0, 0, 0}},   // []
      {2, {1, 0, 0, 0}},   // [He]
      {10, {2, 1, 0, 0}},  // [Ne]
      {18, {3, 2, 0, 0}},  // [Ar]
      {28, {3, 2, 1, 0}},  // [Ar] + 3d
      {36, {4, 3, 1, 0}},  // [Kr]
      {46, {4, 3, 2, 0}},  // [Kr] + 4d
      {54, {5, 4, 2, 0}},  // [Xe]
      {60, {4, 3, 2, 1}},  // [Kr] + 4d + 4f
      {68, {5, 4, 2, 1}},  // [Xe] + 4f
      {78, {5, 4, 3, 1}}   // [Xe] + 4f + 5d
  };

  // check if ecp_electrons is in map
  if (ecp_map.find(ecp_electrons) == ecp_map.end()) {
    throw std::runtime_error(
        "ECP electron configuration not predefined for this number of ECP "
        "electrons.");
  }

  std::array<size_t, 4> core_config = ecp_map[ecp_electrons];
  return core_config;
}

/**
 *  @brief Get the number of fully occupied and fractionally occupied
 *  orbitals for a given angular momentum and nuclear charge.
 *  @param l Angular momentum quantum number
 *  @param nuc_charge Nuclear charge of the atom
 *  @return A tuple containing the number of fully occupied orbitals and the
 *  fractional occupation
 */
std::tuple<size_t, double> get_num_frac_occ_orbs(size_t l, size_t nuc_charge) {
  if (nuc_charge >= constants::ATOMIC_CONFIGURATION.size()) {
    throw std::runtime_error(
        "Nuclear charge exceeds predefined configuration size.");
  }
  std::array<size_t, 4> config = constants::ATOMIC_CONFIGURATION[nuc_charge];
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
std::shared_ptr<BasisSet> make_atom_basis_set(size_t index,
                                              const BasisSet& basis_set,
                                              std::shared_ptr<Molecule> mol) {
  std::vector<Shell> shells;
  std::vector<Shell> ecp_shells;
  int total_ecp_electrons = 0;
  std::unordered_map<int, int> ecp_electrons;

  // Filter shells belonging to the specified atomic number
  std::copy_if(basis_set.shells.begin(), basis_set.shells.end(),
               std::back_inserter(shells), [index](const Shell& shell) {
                 return shell.atom_index == index;
               });

  // Reset center and atom index for single atom basis
  for (auto& shell : shells) {
    shell.O = {0.0, 0.0, 0.0};
    shell.atom_index = 0;
  }

  // Filter ECP shells belonging to the specified atomic number
  std::copy_if(basis_set.ecp_shells.begin(), basis_set.ecp_shells.end(),
               std::back_inserter(ecp_shells), [index](const Shell& shell) {
                 return shell.atom_index == index;
               });

  // Reset center and atom index for single atom ECP basis
  for (auto& shell : ecp_shells) {
    shell.O = {0.0, 0.0, 0.0};
    shell.atom_index = 0;
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
  auto atom_basis = std::shared_ptr<BasisSet>(
      new BasisSet(mol, shells, ecp_shells, ecp_electrons, total_ecp_electrons,
                   BasisMode::RAW, basis_set.pure, false));

  // Update atomic charges, total nuclear charge, and n_electrons based on ECPs
  if (ecp_electrons.count(atomic_number)) {
    int ecp_elec = ecp_electrons[atomic_number];
    mol->atomic_charges[0] = atomic_number - ecp_elec;
    mol->total_nuclear_charge = mol->atomic_charges[0];
    mol->n_electrons = mol->total_nuclear_charge - mol->charge;
  }

  return atom_basis;
}

bool BasisEqChecker::operator()(const BasisSet& a,
                                const BasisSet& b) const noexcept {
  // mol only has one atom, check if atoms are same
  if (a.mol->atomic_nums[0] != b.mol->atomic_nums[0]) return false;

  // check basis set
  if (a.n_ecp_electrons != b.n_ecp_electrons) return false;
  if (a.pure != b.pure) return false;
  if (a.num_atomic_orbitals != b.num_atomic_orbitals) return false;
  if (a.shells.size() != b.shells.size()) return false;
  if (a.ecp_shells.size() != b.ecp_shells.size()) return false;
  if (a.element_ecp_electrons.size() != b.element_ecp_electrons.size())
    return false;

  // check shells
  for (size_t i = 0; i < a.shells.size(); ++i) {
    const auto& a_shell = a.shells[i];
    // already checked for b shells size
    const auto& b_shell = b.shells[i];
    // atom_index and cartesian coords are always the same here so no check
    if (a_shell.angular_momentum != b_shell.angular_momentum) return false;
    if (a_shell.contraction != b_shell.contraction) return false;
    for (size_t j = 0; j < a_shell.contraction; ++j) {
      if (a_shell.exponents[j] != b_shell.exponents[j]) return false;
      if (a_shell.coefficients[j] != b_shell.coefficients[j]) return false;
      if (a_shell.rpowers[j] != b_shell.rpowers[j]) return false;
    }
  }

  // check ecp shells
  for (size_t i = 0; i < a.ecp_shells.size(); ++i) {
    const auto& a_shell = a.ecp_shells[i];
    // already checked for b shells size
    const auto& b_shell = b.ecp_shells[i];
    // atom_index and cartesian coords are always the same here so no check
    if (a_shell.angular_momentum != b_shell.angular_momentum) return false;
    if (a_shell.contraction != b_shell.contraction) return false;
    for (size_t j = 0; j < a_shell.contraction; ++j) {
      if (a_shell.exponents[j] != b_shell.exponents[j]) return false;
      if (a_shell.coefficients[j] != b_shell.coefficients[j]) return false;
      if (a_shell.rpowers[j] != b_shell.rpowers[j]) return false;
    }
  }
  return true;
}

/**
 * @brief Combine the hash of a value into an existing hash seed
 * @tparam T Type of the value to hash
 * @param seed Existing hash seed
 * @param v Value to hash and combine
 * @return Combined hash value
 */
template <typename T>
size_t hash_combine(size_t seed, const T& v) {
  using value_type = std::decay_t<T>;
  seed ^= std::hash<value_type>()(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

size_t BasisHasher::operator()(const BasisSet& basis) const noexcept {
  size_t hash = 0;
  // mol only has one atom, hash atomic number
  hash = hash_combine(hash, basis.mol->atomic_nums[0]);

  // hash basis set
  hash = hash_combine(hash, basis.n_ecp_electrons);
  hash = hash_combine(hash, basis.pure);
  hash = hash_combine(hash, basis.num_atomic_orbitals);
  hash = hash_combine(hash, basis.shells.size());
  hash = hash_combine(hash, basis.ecp_shells.size());
  hash = hash_combine(hash, basis.element_ecp_electrons.size());

  // hash shells
  for (const auto& shell : basis.shells) {
    hash = hash_combine(hash, shell.angular_momentum);
    hash = hash_combine(hash, shell.contraction);
    for (size_t j = 0; j < shell.contraction; ++j) {
      hash = hash_combine(hash, shell.exponents[j]);
      hash = hash_combine(hash, shell.coefficients[j]);
      hash = hash_combine(hash, shell.rpowers[j]);
    }
  }

  // hash ecp shells
  for (const auto& shell : basis.ecp_shells) {
    hash = hash_combine(hash, shell.angular_momentum);
    hash = hash_combine(hash, shell.contraction);
    for (size_t j = 0; j < shell.contraction; ++j) {
      hash = hash_combine(hash, shell.exponents[j]);
      hash = hash_combine(hash, shell.coefficients[j]);
      hash = hash_combine(hash, shell.rpowers[j]);
    }
  }

  return hash;
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
  cfg.scf_algorithm.method = SCFAlgorithmName::ASAHF;
  cfg.density_init_method = DensityInitializationMethod::Core;
  cfg.require_gradient = false;
  cfg.unrestricted = false;
  cfg.require_polarizability = false;
  cfg.exc.xc_name = "hf";
  cfg.eri.method = ERIMethod::Libint2Direct;
  cfg.grad_eri = cfg.eri;
  cfg.grad_eri.method = ERIMethod::Libint2Direct;

  // map with hashed basis + element as key and dm as value
  detail::BasisSetMap basis_to_dm_map;

  for (size_t i = 0, p = 0; i < mol.n_atoms; ++i) {
    auto atom_num = mol.atomic_nums[i];
    // create atomic molecule and basis set
    std::shared_ptr<Molecule> atom_mol = detail::make_atomic_molecule(atom_num);
    std::shared_ptr<BasisSet> atom_basis_set =
        detail::make_atom_basis_set(i, basis_set, atom_mol);
    // check if we have already computed the dm for this basis
    auto it = basis_to_dm_map.find(*atom_basis_set);
    if (it == basis_to_dm_map.end()) {
      // Create SCF solver with basis sets
      SCFImpl scf_solver(atom_mol, cfg, atom_basis_set, atom_basis_set, false,
                         true);
      // Run SCF with ASAHF algorithm
      const auto& asahf_ctx = scf_solver.run();
      // store in map
      basis_to_dm_map[*atom_basis_set] = scf_solver.get_density_matrix();
    }
    const RowMajorMatrix& dm = basis_to_dm_map[*atom_basis_set];
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
    compute_orthogonalization_matrix_(S_averaged, &X_block, n_shells);
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

  // Get core configuration from ECP shells
  std::array<size_t, 4> core_config =
      detail::get_core_config_from_ecp_shells(*ctx_.basis_set);

  for (size_t l = 0; l < max_l; ++l) {
    const auto& idx = ao_indices_by_l[l];
    if (idx.empty()) continue;

    size_t degeneracy = 2 * l + 1;
    size_t n_shells = idx.size() / degeneracy;

    // Get full configuration for this atom using atomic number
    auto [n_double_occ, frac_occ] =
        detail::get_num_frac_occ_orbs(l, ctx_.mol->atomic_nums[0]);

    // Subtract core electrons for this angular momentum
    if (l < 4 && core_config[l] > 0) {
      // core_config[l] is already the number of core shells for this l
      size_t core_shells = core_config[l];
      if (n_double_occ >= core_shells) {
        n_double_occ -= core_shells;
      } else {
        // All electrons for this l are in the core
        n_double_occ = 0;
        frac_occ = 0;
      }
    }

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

void AtomicSphericallyAveragedHartreeFock::compute_orthogonalization_matrix_(
    const RowMajorMatrix& S_, RowMajorMatrix* ret, size_t n_atom_orbs) {
  RowMajorMatrix U_t(n_atom_orbs, n_atom_orbs);
  RowMajorMatrix s(n_atom_orbs, 1);
  std::memcpy(U_t.data(), S_.data(),
              n_atom_orbs * n_atom_orbs * sizeof(double));
  lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, n_atom_orbs, U_t.data(),
               n_atom_orbs, s.data());

  RowMajorMatrix U = U_t.transpose();

  auto threshold = ctx_.cfg->lindep_threshold;
  if (threshold < 0.0) threshold = s.maxCoeff() / 1e9;

  size_t n_mol_orbs = 0;
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

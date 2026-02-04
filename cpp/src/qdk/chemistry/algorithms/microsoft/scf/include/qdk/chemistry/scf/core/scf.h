// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/config.h>
#include <qdk/chemistry/scf/core/basis_func.h>
#include <qdk/chemistry/scf/core/basis_set.h>
#include <qdk/chemistry/scf/core/enums.h>
#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/util/gauxc_util.h>

#include <cstring>
#include <gauxc/enums.hpp>
#include <gauxc/grid_factory.hpp>

#ifdef QDK_CHEMISTRY_ENABLE_QMMM
#include <qdk/chemistry/scf/core/qmmm.h>
#endif

#ifdef QDK_CHEMISTRY_ENABLE_PCM
#include <qdk/chemistry/scf/core/solvation_config.h>
#endif

#ifdef QDK_CHEMISTRY_ENABLE_LIBINTX
#include <qdk/chemistry/scf/core/libintx_config.h>
#endif

#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
#include <qdk/chemistry/scf/core/dispersion_config.h>
#endif

namespace qdk::chemistry::scf {
/**
 * @brief MPI parallel configuration
 */
struct ParallelConfig {
  int world_size;  ///< Total number of MPI processes in MPI_COMM_WORLD
  int world_rank;  ///< Rank of this process in MPI_COMM_WORLD
  int local_size;  ///< Number of processes on this node
  int local_rank;  ///< Rank of this process on the local node
};

/**
 * @brief Returns default MPI configuration
 *
 * Initializes MPI settings in the default execution context
 *
 * @return Default ParallelConfig configuration
 */
ParallelConfig mpi_default_input();

/**
 * @brief Exchange-correlation functional configuration
 */
struct EXCConfig {
  std::string xc_name = "M06-2x";       ///< Name of XC functional
  EXCMethod method = EXCMethod::GauXC;  ///< XC backend
};

/**
 * @brief Electron repulsion integral (ERI) configuration
 */
struct ERIConfig {
#ifdef QDK_CHEMISTRY_ENABLE_HGP
  ERIMethod method = ERIMethod::HGP;
#else
  ERIMethod method = ERIMethod::Libint2Direct;
#endif
  uint32_t enable_QQR =
      0;  ///< Enable QQR integral screening (1=enabled, 0=disabled)
  uint32_t is_mix_precision =
      1;  ///< Use mixed precision (1=yes, 0=no) for GPU calculations
  uint32_t with_j = 1;           ///< Compute Coulomb (J) matrix (1=yes, 0=no)
  uint32_t with_k = 1;           ///< Compute exchange (K) matrix (1=yes, 0=no)
  uint32_t dm_cnt = 1;           ///< Number of density matrices to process
  double eri_threshold = 1e-10;  ///< Integral screening threshold
  bool use_atomics =
      false;  ///< Use atomic operations (true) or thread-local buffers (false)
  uint32_t gpu_device_cnt = 1;  ///< Number of GPU devices to use
  uint32_t* gpu_device_ids =
      NULL;  ///< Array of GPU device IDs (NULL = use default devices)
  uint32_t eri_cpu_threads =
      6;                      ///< Number of CPU threads for integral evaluation
  uint32_t task_per_gpu = 1;  ///< Number of tasks per GPU
  uint64_t gpu_max_memory =
      1e9;           ///< Maximum GPU memory per device in bytes (1 GB default)
  double omega = 0;  ///< Range-separation parameter ω for long-range corrected
                     ///< functionals (0=no range separation)
  uint32_t verbose = 0;  ///< Verbosity level for ERI output
};

/**
 * @brief GDM (Geometric Direct Minimization) configuration
 *
 * Settings specific to GDM algorithm and DIIS-GDM hybrid method.
 */
struct GDMConfig {
  double energy_thresh_diis_switch =
      1e-3;                         ///< Energy threshold to switch from DIIS
                                    ///< to GDM (for DIIS_GDM method)
  int gdm_max_diis_iteration = 50;  ///< Maximum DIIS iterations before
                                    ///< switching to GDM (for DIIS_GDM method)
  int gdm_bfgs_history_size_limit = 50;  ///< History size limit for BFGS in GDM
                                         ///< (number of stored steps)
};

/**
 * @brief SCF algorithm and convergence configuration
 *
 * Settings that control the SCF iteration algorithm, convergence criteria,
 * and convergence acceleration methods (DIIS, GDM, damping, level shifting).
 */
struct SCFAlgorithmConfig {
  int max_iteration = 100;          ///< Maximum number of SCF iterations
  double density_threshold = 1e-6;  ///< Density matrix convergence threshold
  double og_threshold = 1e-6;       ///< Orbital gradient convergence threshold

  // Algorithm method selection
  SCFAlgorithmName method = SCFAlgorithmName::DIIS;  ///< SCF algorithm method

  // DIIS (Direct Inversion in Iterative Subspace) settings
  uint64_t diis_subspace_size =
      8;  ///< Size of DIIS subspace for convergence acceleration

  // Fock matrix damping settings
  bool enable_damping = false;  ///< Enable Fock matrix damping
  double damping_factor =
      0.75;  ///< Damping factor α: F_new = α*F_old + (1-α)*F_current
  double damping_threshold =
      0.01;  ///< Density change threshold below which damping is disabled

  // Level shifting settings
  double level_shift = -1.0;  ///< Level shift for virtual orbitals (negative =
                              ///< no shift, smoothens convergence)

  // Geometric Direct Minimization (GDM) settings (only used when method
  // includes GDM)
  GDMConfig gdm_config;  ///< GDM-specific configuration parameters
};

/**
 * @brief Coupled-perturbed SCF input configuration
 *
 * Settings for solving the CPSCF equations (needed for polarizabilities
 * and response properties).
 */
struct CPSCFInput {
  IterativeLinearSolver solver =
      IterativeLinearSolver::GMRES;  ///< Linear solver method
  int max_iteration = 40;            ///< Maximum iterations for the solver
  int max_restart = 3;      ///< Maximum number of restarts for iterative solver
  double tolerance = 1e-5;  ///< Convergence tolerance for the CPSCF equations
};

/**
 * @brief Self-consistent field (SCF) configuration
 *
 * Contains all settings for SCF calculations including basis sets,
 * convergence criteria, integral methods, and optional features.
 */
struct SCFConfig {
  std::string basis = "def2-svp";  ///< Primary basis set name
  std::string aux_basis =
      "def2-universal-jfit";  ///< Auxiliary basis set for density fitting
  BasisMode basis_mode =
      BasisMode::PSI4;  ///< Basis set normalization mode for calculations
  BasisMode output_basis_mode =
      BasisMode::RAW;             ///< Basis set normalization mode for output
  bool cartesian = false;         ///< Use Cartesian atomic orbitals (true) or
                                  ///< spherical harmonics (false)
  bool require_gradient = false;  ///< Calculate analytical energy gradient
  bool require_polarizability = false;  ///< Calculate polarizability tensor
  bool do_dfj = false;  ///< Use density fitting for Coulomb (J) integrals
  bool unrestricted =
      false;  ///< Use unrestricted (UHF/UKS) rather than restricted (RHF/RKS)
  double lindep_threshold =
      1e-6;  ///< Linear dependency threshold for basis set orthogonalization
  DensityInitializationMethod density_init_method =
      DensityInitializationMethod::Atom;
  ///< Initial density guess method
  std::string density_init_file =
      "";  ///< File path for reading initial density (empty = use
           ///< density_init_method)
  uint64_t incremental_fock_start_step =
      3;  ///< SCF iteration to start incremental Fock matrix updates
  uint64_t fock_reset_steps =
      1073741824;  ///< Number of steps between full Fock matrix rebuilds
                   ///< (default: 2^30 = effectively never)

#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
  DispersionType disp = DispersionType::None;
#endif
  ERIConfig eri;  ///< Configuration for electron repulsion integrals (ERI). By
                  ///< default this controls both J/K builds, but can be
                  ///< overridden for K with `k_eri`
  ERIConfig
      k_eri;  ///< Configuration for exchange (K) integrals (can differ from J)
  ERIConfig grad_eri;  ///< Configuration for gradient integrals
#ifdef QDK_CHEMISTRY_ENABLE_LIBINTX
  LIBINTXConfig libintx_config;  ///< LIBINTX-specific configuration
#endif
  EXCConfig exc;       ///< Exchange-correlation functional configuration
  ParallelConfig mpi;  ///< Parallelization configuration
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  bool enable_pcm =
      false;           ///< Enable Polarizable Continuum Model (PCM) solvation
  PCMInput pcm_input;  ///< PCM solvation configuration
  bool use_ddx =
      false;  ///< Use DDX (domain decomposition) solvation instead of PCMSolver
  DDXInput ddx_input;  ///< DDX solvation configuration
#endif
  GAUXCInput xc_input;     ///< GauXC configuration for XC functional evaluation
  GAUXCInput snk_input;    ///< Semi-numerical K (exchange) configuration
  CPSCFInput cpscf_input;  ///< Coupled-perturbed SCF configuration
  SCFAlgorithmConfig scf_algorithm;  ///< SCF algorithm and convergence settings
  int verbose = 4;  ///< Verbosity level (0=quiet, higher=more output)
#ifdef QATK_ENABLE_QMMM
  std::shared_ptr<PointCharges> pointcharges;
#endif
};

/**
 * @brief Results from an SCF calculation
 *
 * Contains energies, dipole moments, populations, and other
 * properties computed during the SCF procedure.
 */
struct SCFResult {
  bool converged;      ///< Whether the SCF procedure converged
  int scf_iterations;  ///< Number of SCF iterations performed
  double
      nuclear_repulsion_energy;  ///< Nuclear-nuclear repulsion energy (Hartree)
  double scf_one_electron_energy;  ///< One-electron energy (Hartree)
  double scf_two_electron_energy;  ///< Two-electron (ERI) energy (Hartree)
  double scf_xc_energy;  ///< Exchange-correlation energy for DFT (Hartree)
#ifdef QDK_CHEMISTRY_ENABLE_PCM
  double scf_pcm_energy;  ///< Solvation energy (Hartree)
#endif
#ifdef QDK_CHEMISTRY_ENABLE_DFTD3
  double scf_dispersion_correction_energy;  ///< Empirical dispersion correction
                                            ///< energy (Hartree)
#endif
  double scf_total_energy;  ///< Total SCF energy (Hartree)
  std::array<double, 3>
      scf_dipole;  ///< Electric dipole moment vector [x, y, z] (a.u.)
  std::array<double, 6> scf_quadrupole;  ///< Electric quadrupole moment tensor
                                         ///< [xx, xy, xz, yy, yz, zz] (a.u.)
  std::array<double, 9>
      scf_polarizability;  ///< Polarizability tensor [xx, xy, xz, yx, yy, yz,
                           ///< zx, zy, zz] (a.u.)
  double scf_isotropic_polarizability;  ///< Isotropic (average) polarizability
                                        ///< (a.u.)
  std::vector<double> scf_total_gradient;  ///< Energy gradient w.r.t. nuclear
                                           ///< coordinates (Hartree/Bohr)
  std::vector<double>
      scf_total_hessian;  ///< Energy Hessian (second derivatives) w.r.t.
                          ///< nuclear coordinates (Hartree/Bohr²)
  std::vector<double>
      mulliken_population;  ///< Mulliken atomic charges/populations
  std::vector<double> lowdin_population;  ///< Löwdin atomic charges/populations
#ifdef QDK_CHEMISTRY_ENABLE_QMMM
  double qmmm_coulomb_energy;  ///< QM-MM Coulomb interaction energy (Hartree)
  std::vector<double>
      scf_point_charge_gradient;  ///< Gradient on external point charges
                                  ///< (Hartree/Bohr)
  double mm_self_energy;          ///< MM self-energy (point charge-point charge
                                  ///< interactions) (Hartree)
#endif
};

/**
 * @brief Complete context for an SCF calculation
 *
 * Holds configuration, molecular data, basis sets, results, and timing.
 */
struct SCFContext {
  const SCFConfig* cfg;  ///< Pointer to SCF configuration settings
  const Molecule* mol;   ///< Pointer to molecular structure
  std::shared_ptr<BasisSet> basis_set;  ///< Primary basis set
  std::shared_ptr<BasisSet>
      aux_basis_set;  ///< Auxiliary basis set for density fitting (if used)
  std::shared_ptr<BasisSet>
      basis_set_raw;  ///< Basis set in RAW normalization (for output)

  int64_t num_molecular_orbitals = -1;  ///< Number of molecular orbitals

  SCFResult result;  ///< Results from the SCF calculation
};
}  // namespace qdk::chemistry::scf

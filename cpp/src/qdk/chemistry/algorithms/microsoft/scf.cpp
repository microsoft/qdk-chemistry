// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "scf.hpp"

#include <qdk/chemistry/scf/core/molecule.h>
#include <qdk/chemistry/scf/core/scf.h>
#include <qdk/chemistry/scf/scf/scf_solver.h>
#include <qdk/chemistry/scf/util/gauxc_registry.h>
#include <qdk/chemistry/scf/util/libint2_util.h>

#include <qdk/chemistry/data/wavefunction_containers/sd.hpp>
#include <qdk/chemistry/utils/logger.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

// Local implementation details
#include "utils.hpp"

namespace qdk::chemistry::algorithms::microsoft {

namespace qcs = qdk::chemistry::scf;

// Bring logger types into scope
using qdk::chemistry::utils::Logger;
using qdk::chemistry::utils::LogLevel;

// Helper function to calculate alpha and beta electron counts
std::pair<int, int> calculate_electron_counts(int nuclear_charge, int charge,
                                              int multiplicity) {
  QDK_LOG_TRACE_ENTERING();
  int total_electrons = nuclear_charge - charge;
  int n_alpha = (total_electrons + multiplicity - 1) / 2;
  int n_beta = total_electrons - n_alpha;
  return {n_alpha, n_beta};
}

std::pair<double, std::shared_ptr<data::Wavefunction>> ScfSolver::_run_impl(
    std::shared_ptr<data::Structure> structure, int charge, int multiplicity,
    BasisOrGuessType basis_or_guess) const {
  QDK_LOG_TRACE_ENTERING();
  // Initialize the backend if not already done
  utils::microsoft::initialize_backend();

  // check basis_or_guess type
  enum class BasisSetType { Explicit, FromString, FromOrbitals };
  BasisSetType basis_set_type;

  std::string basis_set_name;
  if (std::holds_alternative<std::shared_ptr<data::Orbitals>>(basis_or_guess)) {
    basis_set_name = std::get<std::shared_ptr<data::Orbitals>>(basis_or_guess)
                         ->get_basis_set()
                         ->get_name();
    basis_set_type = BasisSetType::FromOrbitals;
  } else if (std::holds_alternative<std::shared_ptr<data::BasisSet>>(
                 basis_or_guess)) {
    basis_set_name =
        std::get<std::shared_ptr<data::BasisSet>>(basis_or_guess)->get_name();
    basis_set_type = BasisSetType::Explicit;
  } else if (std::holds_alternative<std::string>(basis_or_guess)) {
    basis_set_name = std::get<std::string>(basis_or_guess);
    basis_set_type = BasisSetType::FromString;
  }
  std::transform(basis_set_name.begin(), basis_set_name.end(),
                 basis_set_name.begin(), ::tolower);

  std::shared_ptr<data::BasisSet> qdk_raw_basis_set = nullptr;
  if (basis_set_name == data::BasisSet::custom_name ||
      basis_set_type == BasisSetType::Explicit) {
    qdk_raw_basis_set =
        std::get<std::shared_ptr<data::BasisSet>>(basis_or_guess);
  } else {
    qdk_raw_basis_set =
        data::BasisSet::from_basis_name(basis_set_name, structure);
  }

  // Extract geometry from structure object
  std::vector<double> geometry(3 * structure->get_num_atoms());
  std::vector<std::string> symbols(structure->get_num_atoms());
  for (unsigned i = 0; i < structure->get_num_atoms(); ++i) {
    Eigen::Vector3d coords = structure->get_atom_coordinates(i);
    geometry[3 * i] = coords.x();
    geometry[3 * i + 1] = coords.y();
    geometry[3 * i + 2] = coords.z();
    symbols[i] = structure->get_atom_symbol(i);
  }

  // Compute sum of nuclear charges
  int nuclear_charge = 0;
  for (auto i = 0; i < structure->get_num_atoms(); ++i) {
    nuclear_charge += structure->get_atom_nuclear_charge(i);
  }

  // Determine the multiplicity
  if (multiplicity < 0) {
    // Default to singlet for closed shell, doublet for open-shell
    multiplicity = ((nuclear_charge - charge) % 2 == 0) ? 1 : 2;
    // TODO (NAB): should the user be warned about a default being used?
    // Workitem: 41322
  }

  const bool open_shell = (multiplicity != 1);

  // Determine reference type and unrestricted flag
  std::string scf_type = _settings->get<std::string>("scf_type");
  std::transform(scf_type.begin(), scf_type.end(), scf_type.begin(), ::tolower);

  bool unrestricted;
  if (scf_type == "auto") {
    unrestricted = open_shell;
  } else if (scf_type == "unrestricted") {
    unrestricted = true;
    if (!open_shell && basis_set_type != BasisSetType::FromOrbitals) {
      QDK_LOGGER().warn(
          "Unrestricted reference requested for closed-shell system. "
          "Automatic spin symmetry breaking is not supported. "
          "Consider providing a spin-broken initial guess if desired.");
    }
  } else if (scf_type == "restricted") {
    if (open_shell) {
      throw std::invalid_argument(
          "Restricted Open-Shell calculation is not currently supported in the "
          "QDK/Chemistry SCFSolver");
    }
    unrestricted = false;
  } else {
    throw std::invalid_argument(
        "scf_type must be one of: auto, restricted, unrestricted");
  }

  // Extract SCF settings
  std::string method = _settings->get<std::string>("method");
  std::transform(method.begin(), method.end(), method.begin(), ::tolower);

  double convergence_threshold =
      _settings->get<double>("convergence_threshold");
  int64_t max_iterations = _settings->get<int64_t>("max_iterations");

  // Set different convergence threshold according to tolerance
  double orbital_gradient_threshold = convergence_threshold;
  // when convergence_threshold = 1e-7, density_threshold = 1e-5
  double density_threshold = convergence_threshold * 1e2;

  // Create Molecule object
  auto ms_mol = qdk::chemistry::utils::microsoft::convert_to_molecule(
      *structure, charge, multiplicity);
  // update atomic charges for ECPs
  auto ecp_electrons = qdk_raw_basis_set->get_ecp_electrons();
  for (size_t i = 0; i < ms_mol->n_atoms; ++i) {
    int n_core_electrons = static_cast<int>(ecp_electrons[i]);
    ms_mol->atomic_charges[i] = ms_mol->atomic_nums[i] - n_core_electrons;
  }

  // Create SCFConfig
  auto ms_scf_config = std::make_unique<qcs::SCFConfig>();
  ms_scf_config->mpi = qcs::mpi_default_input();
  ms_scf_config->require_gradient = false;
  ms_scf_config->require_polarizability = false;
  ms_scf_config->exc.xc_name = method;
  std::transform(ms_scf_config->exc.xc_name.begin(),
                 ms_scf_config->exc.xc_name.end(),
                 ms_scf_config->exc.xc_name.begin(), ::toupper);
  ms_scf_config->basis = basis_set_name;
  ms_scf_config->basis_mode = qcs::BasisMode::PSI4;
  ms_scf_config->unrestricted = unrestricted;
  ms_scf_config->scf_algorithm.density_threshold = density_threshold;
  ms_scf_config->scf_algorithm.og_threshold = orbital_gradient_threshold;
  ms_scf_config->scf_algorithm.max_iteration = max_iterations;
  // Set density initialization method based on whether initial guess is
  // provided
  ms_scf_config->density_init_method =
      basis_set_type == BasisSetType::FromOrbitals
          ? qcs::DensityInitializationMethod::UserProvided
          : qcs::DensityInitializationMethod::Atom;

  // Configure ERI method from settings
  std::string eri_method = _settings->get<std::string>("eri_method");
  if (eri_method == "direct") {
    ms_scf_config->eri.method = qcs::ERIMethod::Libint2Direct;
  } else {  // Must be "incore" due to ListConstraint validation
    ms_scf_config->eri.method = qcs::ERIMethod::Incore;
  }

  double eri_threshold = _settings->get<double>("eri_threshold");
  if (eri_threshold > 0.0) {
    ms_scf_config->eri.eri_threshold = eri_threshold;
  } else {
    // use the appropriate default according to convergence threshold
    double eri_threshold_multiplier = 1.0e-5;
    ms_scf_config->eri.eri_threshold =
        convergence_threshold * eri_threshold_multiplier;
  }
  ms_scf_config->eri.use_atomics = _settings->get<bool>("eri_use_atomics");
  ms_scf_config->k_eri = ms_scf_config->eri;
  ms_scf_config->grad_eri = ms_scf_config->eri;

  ms_scf_config->fock_reset_steps = _settings->get<int64_t>("fock_reset_steps");

  // Convert enable_gdm boolean to algorithm method enum for backward
  // compatibility
  bool enable_gdm = _settings->get<bool>("enable_gdm");
  if (enable_gdm) {
    ms_scf_config->scf_algorithm.method = qcs::SCFAlgorithmName::DIIS_GDM;
  } else {
    ms_scf_config->scf_algorithm.method = qcs::SCFAlgorithmName::DIIS;
  }

  ms_scf_config->scf_algorithm.level_shift =
      _settings->get<double>("level_shift");
  ms_scf_config->scf_algorithm.max_iteration =
      _settings->get<int64_t>("max_iterations");
  ms_scf_config->scf_algorithm.gdm_config.energy_thresh_diis_switch =
      _settings->get<double>("energy_thresh_diis_switch");
  ms_scf_config->scf_algorithm.gdm_config.gdm_max_diis_iteration =
      _settings->get<int64_t>("gdm_max_diis_iteration");
  ms_scf_config->scf_algorithm.gdm_config.gdm_bfgs_history_size_limit =
      _settings->get<int64_t>("gdm_bfgs_history_size_limit");
  if (ms_scf_config->eri.method == qcs::ERIMethod::Incore) {
#ifdef QDK_CHEMISTRY_ENABLE_HGP
    ms_scf_config->grad_eri.method = qcs::ERIMethod::HGP;
#else
    ms_scf_config->grad_eri.method = qcs::ERIMethod::Libint2Direct;
#endif
  }

  // FP scales poorly with threads
  // TODO: Make this configurable, workitem: 41325
#ifdef _OPENMP
  auto old_max_threads = omp_get_max_threads();
  // omp_set_num_threads(1);
#endif

  // Save the current global level before disabling SCF logging
  auto saved_level = Logger::get_global_level();

  // Disable SCF logging
  Logger::set_global_level(LogLevel::off);

  // Create SCF solver based on method and basis set type
  std::shared_ptr<qcs::SCF> scf;
  if (method == "hf") {
    scf = qcs::SCF::make_hf_solver(
        ms_mol, *ms_scf_config,
        utils::microsoft::convert_basis_set_from_qdk(*qdk_raw_basis_set),
        utils::microsoft::convert_basis_set_from_qdk(*qdk_raw_basis_set,
                                                     false));
  } else {
    scf = qcs::SCF::make_ks_solver(
        ms_mol, *ms_scf_config,
        utils::microsoft::convert_basis_set_from_qdk(*qdk_raw_basis_set),
        utils::microsoft::convert_basis_set_from_qdk(*qdk_raw_basis_set,
                                                     false));
  }

  // Compute map from QDK shells to internal representation
  auto qdk_to_internal_shells = utils::microsoft::compute_shell_map(
      *qdk_raw_basis_set, *scf->context().basis_set_raw);

  // Compute the transformation matrix
  const size_t num_atomic_orbitals =
      qdk_raw_basis_set->get_num_atomic_orbitals();
  auto shells = qdk_raw_basis_set->get_shells();
  Eigen::MatrixXd qdk_raw_basis_map(num_atomic_orbitals, num_atomic_orbitals);
  qdk_raw_basis_map.setZero();

  // Convert internal to libint2 basis
  auto libint_basis =
      qcs::libint2_util::convert_to_libint_basisset(*scf->context().basis_set);
  auto libint_sh2bf = libint_basis.shell2bf();

  for (size_t i = 0, ibf = 0; i < qdk_raw_basis_set->get_num_shells(); ++i) {
    const auto& shell = shells[i];
    const auto sh_sz = shell.get_num_atomic_orbitals();
    size_t jbf = libint_sh2bf[qdk_to_internal_shells[i]];

    qdk_raw_basis_map.block(ibf, jbf, sh_sz, sh_sz) =
        Eigen::MatrixXd::Identity(sh_sz, sh_sz);

    ibf += sh_sz;
  }

  // If initial guess is provided, compute density matrix and
  // create new SCF solver
  if (basis_set_type == BasisSetType::FromOrbitals) {
    auto initial_guess =
        std::get<std::shared_ptr<data::Orbitals>>(basis_or_guess);
    auto [coeff_alpha, coeff_beta] = initial_guess->get_coefficients();

    // Calculate number of electrons
    auto [n_alpha, n_beta] =
        calculate_electron_counts(nuclear_charge, charge, multiplicity);

    const size_t num_atomic_orbitals = coeff_alpha.rows();

    // Compute density matrix from MO coefficients
    qcs::RowMajorMatrix density_matrix;

    if (unrestricted) {
      if (initial_guess->is_restricted())
        QDK_LOGGER().warn(
            "Unrestricted calculation requested but restricted "
            "initial guess provided.");

      // For unrestricted case, stack alpha and beta coefficients and compute
      // density
      const size_t num_molecular_orbitals = coeff_alpha.cols();

      // Transform coefficients
      Eigen::MatrixXd C_alpha_transformed =
          qdk_raw_basis_map.transpose() * coeff_alpha;
      Eigen::MatrixXd C_beta_transformed =
          qdk_raw_basis_map.transpose() * coeff_beta;

      // Initialize density matrix for unrestricted case (2 *
      // num_atomic_orbitals * num_atomic_orbitals)
      density_matrix = qcs::RowMajorMatrix::Zero(2 * num_atomic_orbitals,
                                                 num_atomic_orbitals);

      // Alpha density matrix
      Eigen::Map<qcs::RowMajorMatrix> P_alpha(
          density_matrix.data(), num_atomic_orbitals, num_atomic_orbitals);
      P_alpha.noalias() =
          C_alpha_transformed.block(0, 0, num_atomic_orbitals, n_alpha) *
          C_alpha_transformed.block(0, 0, num_atomic_orbitals, n_alpha)
              .transpose();

      // Beta density matrix
      Eigen::Map<qcs::RowMajorMatrix> P_beta(
          density_matrix.data() + num_atomic_orbitals * num_atomic_orbitals,
          num_atomic_orbitals, num_atomic_orbitals);
      P_beta.noalias() =
          C_beta_transformed.block(0, 0, num_atomic_orbitals, n_beta) *
          C_beta_transformed.block(0, 0, num_atomic_orbitals, n_beta)
              .transpose();
    } else {
      if (initial_guess->is_unrestricted()) {
        throw std::invalid_argument(
            "Restricted calculation requested but unrestricted initial guess "
            "provided.");
      }
      // For restricted case, use alpha coefficients only
      Eigen::MatrixXd C_transformed =
          qdk_raw_basis_map.transpose() * coeff_alpha;

      // Initialize density matrix for restricted case
      // (num_atomic_orbitals * num_atomic_orbitals)
      density_matrix =
          qcs::RowMajorMatrix::Zero(num_atomic_orbitals, num_atomic_orbitals);

      // Density matrix = 2 * C_occ * C_occ^T for restricted case
      density_matrix.noalias() =
          2.0 * C_transformed.block(0, 0, num_atomic_orbitals, n_alpha) *
          C_transformed.block(0, 0, num_atomic_orbitals, n_alpha).transpose();
    }

    // Create new SCF solver with density matrix
    auto initial_guess_scf =
        (method == "hf") ? qcs::SCF::make_hf_solver(
                               ms_mol, *ms_scf_config, density_matrix,
                               utils::microsoft::convert_basis_set_from_qdk(
                                   *qdk_raw_basis_set),
                               utils::microsoft::convert_basis_set_from_qdk(
                                   *qdk_raw_basis_set, false))
                         : qcs::SCF::make_ks_solver(
                               ms_mol, *ms_scf_config, density_matrix,
                               utils::microsoft::convert_basis_set_from_qdk(
                                   *qdk_raw_basis_set),
                               utils::microsoft::convert_basis_set_from_qdk(
                                   *qdk_raw_basis_set, false));

    // Replace the original scf with the initial guess version
    scf = std::move(initial_guess_scf);
  }

  // Run SCF calculation
  auto& context = scf->run();

  // Reset threads
#ifdef _OPENMP
  if (old_max_threads != 1) omp_set_num_threads(old_max_threads);
#endif
  qcs::util::GAUXCRegistry::clear();

  // Handle Return
  if (!context.result.converged) {
    throw std::runtime_error("SCF did not converge");
  }

  // Compute AO overlap
  auto ao_overlap =
      qdk_raw_basis_map * scf->overlap() * qdk_raw_basis_map.transpose();

  // Compute Aufbau occupations
  auto nelec = scf->get_num_electrons();

  std::shared_ptr<data::Orbitals> orbitals;
  if (unrestricted) {
    // Unrestricted case - store matrices first to avoid
    // temporaries
    const auto& C_full = scf->get_orbitals_matrix();
    const size_t num_atomic_orbitals = C_full.rows() / 2;
    const size_t num_molecular_orbitals = C_full.cols();
    Eigen::MatrixXd C_alpha =
        qdk_raw_basis_map *
        C_full.block(0, 0, num_atomic_orbitals, num_molecular_orbitals);
    Eigen::MatrixXd C_beta =
        qdk_raw_basis_map * C_full.block(num_atomic_orbitals, 0,
                                         num_atomic_orbitals,
                                         num_molecular_orbitals);

    const auto& eps = scf->get_eigenvalues();
    Eigen::VectorXd energies_alpha = eps.row(0);
    Eigen::VectorXd energies_beta = eps.row(1);

    // Construct orbitals with correct parameter order:
    // (coeff_alpha, coeff_beta,
    //  energies_alpha, energies_beta, ao_overlap,
    //  basis_set_name, active_indices_alpha,
    //  active_indices_beta)
    orbitals = std::make_shared<data::Orbitals>(
        C_alpha, C_beta, energies_alpha, energies_beta, ao_overlap,
        qdk_raw_basis_set,
        std::nullopt);  // no active space indices

  } else {
    // Restricted case - store matrices first to avoid
    // temporaries
    Eigen::MatrixXd coefficients =
        qdk_raw_basis_map * scf->get_orbitals_matrix();

    Eigen::VectorXd energies(scf->get_num_molecular_orbitals());
    const auto& eps = scf->get_eigenvalues();
    energies = eps.row(0);

    // Construct orbitals with correct parameter order:
    // (coefficients, energies, ao_overlap, basis_set_name,
    // active_space_indices)
    orbitals = std::make_shared<data::Orbitals>(
        coefficients, energies, ao_overlap, qdk_raw_basis_set,
        std::nullopt);  // no active space indices
  }

  // Create canonical Hartree-Fock Configuration
  size_t n_orbitals = orbitals->get_num_molecular_orbitals();

  // Create canonical HF configuration string
  std::string config_str(n_orbitals, '0');

  for (size_t i = 0; i < n_orbitals; ++i) {
    if (nelec[0] > i and nelec[1] > i) {
      config_str[i] = '2';
    } else if (nelec[0] > i) {
      config_str[i] = 'u';
    } else if (nelec[1] > i) {
      config_str[i] = 'd';
    }
  }
  // Create Configuration object
  data::Configuration hf_det(config_str);

  // Create SlaterDeterminantContainer
  auto container =
      std::make_unique<data::SlaterDeterminantContainer>(hf_det, orbitals);

  // Create Wavefunction
  data::Wavefunction wavefunction(std::move(container));

  // Return total energy
  double total_energy = context.result.scf_total_energy;

  // Restore the original global logging level
  Logger::set_global_level(saved_level);

  return std::make_pair(total_energy, std::make_shared<data::Wavefunction>(
                                          std::move(wavefunction)));
}
}  // namespace qdk::chemistry::algorithms::microsoft

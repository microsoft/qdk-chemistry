// Copyright (c) Microsoft Corporation.

#include "ci.hpp"

#include <macis/asci/grow.hpp>
#include <macis/asci/refine.hpp>
#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <macis/mcscf/cas.hpp>

#include "settings.hpp"

/**
 * @brief Helper struct for CASCI calculation dispatch
 */
struct casci_helper {
  /**
   * @brief Template implementation of CASCI calculation
   * @tparam N Number of bits for wavefunction representation
   * @param nalpha Number of alpha electrons
   * @param nbeta Number of beta electrons
   * @param ham Hamiltonian object containing molecular integrals
   * @param settings Python dictionary with calculation settings
   * @return Python dictionary containing energy, coefficients, and optionally
   * determinants
   */
  template <size_t N>
  static py::dict impl(size_t nalpha, size_t nbeta, Hamiltonian &ham,
                       const py::dict &settings);
};

template <size_t N>
py::dict casci_helper::impl(size_t nalpha, size_t nbeta, Hamiltonian &ham,
                            const py::dict &settings) {
  using wfn_type = macis::wfn_t<N>;
  using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;

  size_t norbital = ham.norb;
  validate_electron_counts(nalpha, nbeta, norbital);

  // Extract settings
  macis::MCSCFSettings mcscf_settings;
  extract_mcscf_settings(settings, mcscf_settings);

  bool return_dets = false;
  if (settings.contains("return_determinants")) {
    return_dets = settings["return_determinants"].cast<bool>();
  }

  // Run CASCI
  std::vector<double> C_casci;
  double E_casci = macis::CASRDMFunctor<generator_t>::rdms(
      mcscf_settings, macis::NumOrbital(norbital), nalpha, nbeta, ham._T.data(),
      ham._V.data(), nullptr, nullptr, C_casci);

  // Return results as dict
  py::dict result;
  result["energy"] = E_casci;
  result["coefficients"] = vector_to_array(C_casci);

  if (return_dets) {
    // Generate determinants
    std::vector<wfn_type> dets =
        macis::generate_hilbert_space<wfn_type>(norbital, nalpha, nbeta);
    result["determinants"] = wfn_vector_to_strings(dets, norbital);
  }

  return result;
}

py::dict run_casci(size_t nalpha, size_t nbeta, Hamiltonian &ham,
                   const py::dict &settings) {
  return dispatch_by_norb<casci_helper>(ham.norb, nalpha, nbeta, ham, settings);
}

/**
 * @brief Helper struct for ASCI calculation dispatch
 */
struct asci_helper {
  /**
   * @brief Template implementation of ASCI calculation
   * @tparam N Number of bits for wavefunction representation
   * @param initial_guess Python list of initial determinant strings
   * @param C0 Initial coefficients vector
   * @param E0 Initial energy estimate
   * @param ham Hamiltonian object containing molecular integrals
   * @param settings Python dictionary with calculation settings
   * @return Python dictionary containing energy, coefficients, and determinants
   */
  template <size_t N>
  static py::dict impl(const py::list &initial_guess,
                       const std::vector<double> &C0, double E0,
                       Hamiltonian &ham, const py::dict &settings);
};

template <size_t N>
py::dict asci_helper::impl(const py::list &initial_guess,
                           const std::vector<double> &C0, double E0,
                           Hamiltonian &ham, const py::dict &settings) {
  using wfn_type = macis::wfn_t<N>;
  using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;

  size_t norbital = ham.norb;
  if (norbital == 0) {
    throw std::runtime_error("Number of orbitals must be greater than 0");
  }

  // Convert initial guess to wfn_type
  std::vector<wfn_type> dets = strings_to_wfn_vector<N>(initial_guess);
  if (dets.empty()) {
    throw std::runtime_error("Initial guess cannot be empty");
  }
  if (C0.empty()) {
    throw std::runtime_error("Coefficients vector cannot be empty");
  }

  // Extract settings
  macis::ASCISettings asci_settings;
  extract_asci_settings(settings, asci_settings);

  macis::MCSCFSettings mcscf_settings;
  extract_mcscf_settings(settings, mcscf_settings);

  // Create Hamiltonian Generator
  generator_t ham_gen(
      macis::matrix_span<double>(ham._T.data(), norbital, norbital),
      macis::rank4_span<double>(ham._V.data(), norbital, norbital, norbital,
                                norbital));

  E0 = ham_gen.matrix_element(dets[0], dets[0]);

  // Growth Phase
  std::vector<double> C = C0;
  std::tie(E0, dets, C) = macis::asci_grow<N, int64_t>(
      asci_settings, mcscf_settings, E0, std::move(dets), std::move(C), ham_gen,
      norbital MACIS_MPI_CODE(, MPI_COMM_WORLD));

  // Refinement phase
  if (asci_settings.max_refine_iter) {
    std::tie(E0, dets, C) = macis::asci_refine<N, int64_t>(
        asci_settings, mcscf_settings, E0, std::move(dets), std::move(C),
        ham_gen, norbital MACIS_MPI_CODE(, MPI_COMM_WORLD));
  }

  // Sort the determinants and coefficients
  macis::reorder_ci_on_coeff(dets, C);

  // Return results as dict
  py::dict result;
  result["energy"] = E0;
  result["coefficients"] = vector_to_array(C);
  result["determinants"] = wfn_vector_to_strings(dets, norbital);

  return result;
}

py::dict run_asci(const py::list &initial_guess, const std::vector<double> &C0,
                  double E0, Hamiltonian &ham, const py::dict &settings) {
  return dispatch_by_norb<asci_helper>(ham.norb, initial_guess, C0, E0, ham,
                                       settings);
}

struct selected_ci_helper {
  template <size_t N>
  static py::dict impl(const py::list &configurations, Hamiltonian &ham,
                       const py::dict &settings);
};

template <size_t N>
py::dict selected_ci_helper::impl(const py::list &configurations,
                                  Hamiltonian &ham, const py::dict &settings) {
  using wfn_type = macis::wfn_t<N>;
  using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;

  const size_t norbital = ham.norb;
  // Extract settings
  macis::MCSCFSettings mcscf_settings;
  extract_mcscf_settings(settings, mcscf_settings);

  // Convert initial guess to wfn_type
  std::vector<wfn_type> dets = strings_to_wfn_vector<N>(configurations);
  if (dets.empty()) {
    throw std::invalid_argument("Basis cannot be empty");
  }

  // Create Hamiltonian Generator
  generator_t ham_gen(
      macis::matrix_span<double>(ham._T.data(), norbital, norbital),
      macis::rank4_span<double>(ham._V.data(), norbital, norbital, norbital,
                                norbital));

  std::vector<double> C;
  auto E0 = macis::selected_ci_diag<int64_t, wfn_type>(
      dets.begin(), dets.end(), ham_gen, 1e-16, 200, 1e-8, C);

  // Return results as dict
  py::dict result;
  result["energy"] = E0;
  result["coefficients"] = vector_to_array(C);

  return result;
}

py::dict selected_ci_diag(const py::list &configurations, Hamiltonian &ham,
                          const py::dict &settings) {
  return dispatch_by_norb<selected_ci_helper>(ham.norb, configurations, ham,
                                              settings);
}

void export_ci_pybind(py::module &m) {
  // CI calculations
  m.def("casci", &run_casci,
        R"pbdoc(
            Run a Complete Active Space Configuration Interaction (CASCI) calculation.

            Performs full CI within the active space by generating all possible
            determinants with the specified number of alpha and beta electrons
            and diagonalizing the CI Hamiltonian matrix.

            Args:
                nalpha (int): Number of alpha (spin-up) electrons in active space.
                nbeta (int): Number of beta (spin-down) electrons in active space.
                H (Hamiltonian): Active space Hamiltonian containing molecular integrals.
                settings (dict, optional): Dictionary of calculation settings including:
                    - 'ci_max_subspace': Maximum number of Davidson iterations (default: 200)
                    - 'ci_res_tol': Convergence tolerance for energy (default: 1e-8)
                    - 'ci_matel_tol': Matrix element threshold for CI (default: machine epsilon)
                    - 'return_determinants': Whether to include determinants in output (default: False)

            Returns:
                dict: Results dictionary containing:
                    - 'energy': Total energy of the ground state (float)
                    - 'coefficients': CI coefficients for ground state wavefunction (numpy.ndarray)
                    - 'determinants': List of determinant strings (only if 'return_determinants' is True)

            Raises:
                RuntimeError: If calculation fails to converge or invalid parameters.

            Example:
                >>> ham = pymacis.read_fcidump("h2o.fcidump")
                >>> # CASCI with 5 alpha, 5 beta electrons
                >>> result = pymacis.casci(nalpha=5, nbeta=5, H=ham)
                >>> print(f"CASCI energy: {result['energy']:.8f} Hartree")
                >>>
                >>> # With custom settings to include determinants
                >>> settings = {'ci_res_tol': 1e-12, 'return_determinants': True}
                >>> result = pymacis.casci(nalpha=5, nbeta=5, H=ham, settings=settings)
                >>> print(f"First configuration: {result['determinants'][0]}")
        )pbdoc",
        py::arg("nalpha"), py::arg("nbeta"), py::arg("H"),
        py::arg("settings") = py::dict());

  m.def("asci", &run_asci,
        R"pbdoc(
            Run an Adaptive Sampling Configuration Interaction (ASCI) calculation.

            Performs selected CI by iteratively growing the determinant space using
            perturbative selection criteria. More efficient than full CI for large
            active spaces by focusing on the most important determinants.

            Args:
                initial_guess (list): List of determinant strings for initial CI space.
                    Each string represents a determinant in binary format (e.g., "110010").
                C0 (list): Initial CI coefficients corresponding to initial_guess determinants.
                E0 (float): Initial energy estimate (typically from initial guess).
                H (Hamiltonian): Active space Hamiltonian containing molecular integrals.
                settings (dict, optional): Dictionary of calculation settings including:

                    MCSCF/Davidson settings:
                    - 'ci_max_subspace': Maximum number of Davidson iterations (default: 200)
                    - 'ci_res_tol': Convergence tolerance for energy (default: 1e-8)
                    - 'ci_matel_tol': Matrix element threshold for CI (default: machine epsilon)

                    ASCI-specific settings:
                    - 'asci_ntdets_max': Maximum number of determinants in final CI space (default: 100000)
                    - 'asci_ntdets_min': Minimum number of determinants to keep (default: 100)
                    - 'asci_ncdets_max': Maximum number of "core" determinants (default: 100)
                    - 'asci_ham_el_tol': Hamiltonian matrix element threshold (default: 1e-8)
                    - 'asci_rv_prune_tol': Right vector pruning tolerance (default: 1e-8)
                    - 'asci_pair_max_lim': Maximum pair size limit (default: 500000000)
                    - 'asci_grow_factor': Growth factor for determinant selection (default: 8)
                    - 'asci_max_refine_iter': Maximum refinement iterations (default: 6)
                    - 'asci_refine_etol': Energy tolerance for refinement phase (default: 1e-6)
                    - 'asci_grow_with_rot': Whether to grow with rotation (default: False)
                    - 'asci_rot_size_start': Starting size for rotation (default: 1000)
                    - 'asci_constraint_lvl': Constraint level for selection (default: 2)

            Returns:
                dict: Results dictionary containing:
                    - 'energy': Total energy of the ground state (float)
                    - 'coefficients': Final CI coefficients (numpy.ndarray)
                    - 'determinants': List of determinant strings in final CI space (list)

            Raises:
                RuntimeError: If calculation fails to converge or invalid parameters.
                ValueError: If initial guess parameters are inconsistent.

            Example:
                >>> # Start with HF determinant as initial guess
                >>> initial_dets = ["2222000000"]  # 4 alpha, 4 beta in 10 orbitals
                >>> initial_coeffs = [1.0]
                >>> E_hf = -75.0  # Hartree-Fock energy estimate
                >>>
                >>> ham = pymacis.read_fcidump("system.fcidump")
                >>> settings = {'asci_ntdets_max': 50000, 'ci_res_tol': 1e-10}
                >>> result = pymacis.asci(initial_dets, initial_coeffs, E_hf, ham, settings)
                >>> print(f"ASCI energy: {result['energy']:.8f} Hartree")
                >>> print(f"Final determinants: {len(result['determinants'])}")
        )pbdoc",
        py::arg("initial_guess"), py::arg("C0"), py::arg("E0"), py::arg("H"),
        py::arg("settings") = py::dict());

  m.def("selected_ci_diag", &selected_ci_diag,
        R"pbdoc(
            Compute the lowest eigenpair of a Hamiltonian projected into a selected configuration basis.

            This function constructs the CI Hamiltonian matrix in the basis defined
            by the provided list of determinants and diagonalizes it to obtain the
            lowest energy state.

            Args:
                configurations (list): List of determinant strings defining the CI basis.
                    Each string represents a determinant in binary format (e.g., "110010").
                H (Hamiltonian): Active space Hamiltonian containing molecular integrals.
                settings (dict, optional): Dictionary of calculation settings including:
                    - 'ci_max_subspace': Maximum number of Davidson iterations (default: 200)
                    - 'ci_res_tol': Convergence tolerance for energy (default: 1e-8)
                    - 'ci_matel_tol': Matrix element threshold for CI (default: machine epsilon)

            Returns:
                dict: Results dictionary containing:
                    - 'energy': Total energy of the ground state (float)
                    - 'coefficients': CI coefficients for ground state wavefunction (numpy.ndarray)

            Raises:
                RuntimeError: If calculation fails to converge or invalid parameters.
                ValueError: If configurations list is empty or invalid.

            Example:
                >>> ham = pymacis.read_fcidump("system.fcidump")
                >>> dets = ["110000", "101000", "100100", "100010"]  # Example determinants
                >>> settings = {'ci_res_tol': 1e-10}
                >>> result = pymacis.selected_ci_diag(dets, ham, settings)
                >>> print(f"Selected CI energy: {result['energy']:.8f} Hartree")
        )pbdoc",
        py::arg("configurations"), py::arg("H"),
        py::arg("settings") = py::dict());
}

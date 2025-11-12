// Copyright (c) Microsoft Corporation.

#include "wavefunction_util.hpp"

#include <macis/hamiltonian_generator/double_loop.hpp>
#include <macis/hamiltonian_generator/sorted_double_loop.hpp>
#include <macis/solvers/selected_ci_diag.hpp>

/**
 * @brief Helper struct for canonical HF determinant generation dispatch
 */
struct canonical_hf_determinant_helper {
  /**
   * @brief Template implementation of canonical HF determinant generation
   * @tparam N Number of bits for wavefunction representation
   * @param nalpha Number of alpha electrons
   * @param nbeta Number of beta electrons
   * @param norb Number of orbitals
   * @return Binary string representing the canonical HF determinant
   */
  template <size_t N>
  static std::string impl(size_t nalpha, size_t nbeta, size_t norb) {
    using wfn_type = macis::wfn_t<N>;
    using wfn_traits = macis::wavefunction_traits<wfn_type>;

    // Validate input parameters
    if (nalpha > norb or nbeta > norb) {
      throw std::runtime_error(
          "Number of electrons cannot exceed number of orbitals");
    }
    if (norb == 0) {
      throw std::runtime_error("Number of orbitals must be greater than 0");
    }
    if (norb > N / 2) {
      throw std::runtime_error(
          "Number of orbitals exceeds maximum supported for this bit size");
    }

    // Generate canonical HF determinant
    wfn_type hf_det = wfn_traits::canonical_hf_determinant(nalpha, nbeta);

    // Convert to string representation
    return macis::to_canonical_string(hf_det).substr(0, norb);
  }
};

/**
 * @brief Generate canonical Hartree-Fock determinant (Python wrapper)
 * @param nalpha Number of alpha electrons
 * @param nbeta Number of beta electrons
 * @param norb Number of orbitals
 * @return Binary string representing the canonical HF determinant
 */
std::string canonical_hf_determinant_wrapper(size_t nalpha, size_t nbeta,
                                             size_t norb) {
  return dispatch_by_norb<canonical_hf_determinant_helper>(norb, nalpha, nbeta,
                                                           norb);
}

struct energy_helper {
  /**
   * @brief Template implementation of energy calculation
   * @tparam N Number of bits for wavefunction representation
   * @param dets_strings Python list of determinant strings
   * @param coeffs Coefficients vector
   * @param ham Hamiltonian object containing molecular integrals
   * @return Calculated energy value
   */
  template <size_t N>
  static double impl(const py::list &dets_strings,
                     const std::vector<double> &coeffs, Hamiltonian &ham) {
    using wfn_type = macis::wfn_t<N>;
    using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;
    std::vector<wfn_type> dets = strings_to_wfn_vector<N>(dets_strings);

    if (dets.empty() || coeffs.empty()) {
      throw std::runtime_error("Determinants and coefficients cannot be empty");
    }

    // Create Hamiltonian generator
    size_t norbital = ham.norb;

    // Create Hamiltonian generator
    generator_t ham_gen(
        macis::matrix_span<double>(ham._T.data(), norbital, norbital),
        macis::rank4_span<double>(ham._V.data(), norbital, norbital, norbital,
                                  norbital));

    double energy = 0.0;
    for (size_t i = 0; i < dets.size(); ++i) {
      for (size_t j = i; j < dets.size(); ++j) {
        double fac =
            (i == j)
                ? 1.0
                : 2.0;  // Diagonal elements counted once, off-diagonal twice
        energy += fac * coeffs[i] * coeffs[j] *
                  ham_gen.matrix_element(dets[i], dets[j]);
      }
    }

    return energy;
  }
};

/**
 * @brief Calculate energy from wavefunction determinants and coefficients
 * (Python wrapper)
 * @param dets_strings Python list of determinant strings
 * @param coeffs NumPy array of coefficients
 * @param ham Hamiltonian object containing molecular integrals
 * @return Calculated energy value
 */
double calculate_energy_wrapper(const py::list &dets_strings,
                                np_double_array coeffs, Hamiltonian &ham) {
  auto coeffs_vec = array_to_vector<double>(coeffs);
  return dispatch_by_norb<energy_helper>(ham.norb, dets_strings, coeffs_vec,
                                         ham);
}

struct diagonalize_helper {
  /**
   * @brief Template implementation of Hamiltonian diagonalization in a given
   * basis
   * @tparam N Number of bits for wavefunction representation
   * @param ham Hamiltonian object containing molecular integrals
   * @param dets_strings Python list of determinant strings defining the basis
   * @return Lowest eigenvalue (energy) including core energy contribution
   */
  template <size_t N>
  static auto impl(Hamiltonian &ham, const py::list &dets_strings) {
    using wfn_type = macis::wfn_t<N>;
    using generator_t = macis::SortedDoubleLoopHamiltonianGenerator<wfn_type>;

    size_t norbital = ham.norb;
    if (norbital == 0) {
      throw std::runtime_error("Number of orbitals must be greater than 0");
    }

    // Create Hamiltonian generator
    generator_t ham_gen(
        macis::matrix_span<double>(ham._T.data(), norbital, norbital),
        macis::rank4_span<double>(ham._V.data(), norbital, norbital, norbital,
                                  norbital));

    std::vector<wfn_type> dets;

    // Convert determinant strings to wavefunction objects
    if (!dets_strings.empty()) {
      dets.reserve(dets_strings.size());
      for (size_t i = 0; i < dets_strings.size(); i++) {
        std::string det_str = dets_strings[i].cast<std::string>();
        dets.emplace_back(macis::from_canonical_string<wfn_type>(det_str));
      }
    } else {
      throw std::runtime_error("Determinant strings list cannot be empty");
    }

    if (dets.empty()) {
      throw std::runtime_error("No determinants available for diagonalization");
    }

    // Define parameters for selected_ci_diag
    constexpr int max_subspace = 100;   // Maximum subspace size for Davidson
    constexpr double res_tol = 1e-8;    // Residual tolerance
    constexpr double h_el_tol = 1e-12;  // Hamiltonian element tolerance

    // Set up initial guess vector for the coefficients
    std::vector<double> coeffs(dets.size(), 0.0);
    coeffs[0] = 1.0;  // Initial guess - just use the first determinant

    // Use selected_ci_diag to find the lowest eigenvalue
    double eigenval = macis::selected_ci_diag<int32_t, wfn_type>(
        dets.begin(), dets.end(), ham_gen, h_el_tol, max_subspace, res_tol,
        coeffs);

    // Return the lowest eigenvalue (electronic energy)
    return eigenval + ham.core_energy;
  }
};

double diagonalize_wrapper(Hamiltonian &ham, const py::list &dets_strings) {
  return dispatch_by_norb<diagonalize_helper>(ham.norb, ham, dets_strings);
}

void export_wavefunction_util_pybind(py::module &m) {
  // Canonical HF determinant generation
  m.def("canonical_hf_determinant", &canonical_hf_determinant_wrapper,
        R"pbdoc(
            Generate the canonical Hartree-Fock determinant for a given number of alpha and beta electrons.

            This function creates the unique reference determinant for the Hartree-Fock wavefunction
            with the specified number of alpha (spin-up) and beta (spin-down) electrons. The determinant
            is returned as a string in binary format, with occupied orbitals indicated.

            Args:
                nalpha (int): Number of alpha (spin-up) electrons.
                nbeta (int): Number of beta (spin-down) electrons.
                norb (int): Total number of orbitals in the system.

            Returns:
                str: Binary string representing the canonical HF determinant.

            Raises:
                ValueError: If the number of electrons or orbitals is invalid.

            Example:
                >>> # For H2 molecule: 2 electrons (1 alpha, 1 beta), 2 orbitals
                >>> det = pymacis.canonical_hf_determinant(1, 1, 2)
                >>> print(f"Canonical HF determinant: {det}")
                >>>
                >>> # For a system with 6 electrons (3 alpha, 3 beta) and 6 orbitals
                >>> det = pymacis.canonical_hf_determinant(3, 3, 6)
                >>> print(f"Canonical HF determinant: {det}")
        )pbdoc",
        py::arg("nalpha"), py::arg("nbeta"), py::arg("norb"));

  m.def("compute_wfn_energy", &calculate_energy_wrapper,
        R"pbdoc(
            Calculate the energy of a wavefunction given its determinants and coefficients.

            Computes the total energy of a CI wavefunction represented by a list of
            determinant strings and their corresponding coefficients using the provided
            Hamiltonian object containing molecular integrals.

            Args:
                wfn (list): List of determinant strings in binary format.
                C (numpy.ndarray): Array of CI coefficients corresponding to each determinant.
                H (Hamiltonian): Hamiltonian object containing molecular integrals.

            Returns:
                float: Total energy of the wavefunction in Hartree units.

            Raises:
                ValueError: If wfn and C have different lengths or are empty.

            Example:
                >>> # Assume wfn and C are obtained from a CASCI or ASCI calculation
                >>> energy = pymacis.compute_wfn_energy(wfn, C, ham)
                >>> print(f"Calculated wavefunction energy: {energy:.8f} Hartree")
        )pbdoc",
        py::arg("wfn"), py::arg("C"), py::arg("H"));

  m.def("diagonalize", &diagonalize_wrapper, R"pbdoc(
            Diagonalize a Hamiltonian in a given determinant basis to find the lowest energy eigenvalue.

            This function constructs and diagonalizes the Hamiltonian matrix in a specified
            basis of determinants using the selected CI diagonalization method. The basis is defined
            by the list of determinant strings provided.

            Args:
                H (Hamiltonian): Hamiltonian object containing molecular integrals.
                determinants (list): List of determinant strings to use as the basis.
                    Each string should use the canonical string format with the following symbols:
                    - '2': doubly occupied orbital (both alpha and beta electrons)
                    - 'u': singly occupied by alpha (spin-up) electron
                    - 'd': singly occupied by beta (spin-down) electron
                    - '0': unoccupied orbital

                    For example, "2200" means the first two orbitals are doubly occupied,
                    "u00d" means the first orbital has an alpha electron and the last
                    orbital has a beta electron.

            Returns:
                float: The lowest eigenvalue (energy) of the Hamiltonian in the given basis,
                       including the core energy contribution.

            Raises:
                RuntimeError: If determinant list is empty or number of orbitals is invalid.

            Note:
                This function uses the selected CI diagonalization algorithm which efficiently
                handles sparse Hamiltonian matrices, making it suitable for larger active spaces
                and determinant bases compared to direct diagonalization.

            Example:
                >>> ham = pymacis.read_fcidump("h2o.fcidump")
                >>> # Diagonalize in a specific basis of determinants
                >>> dets = ["2200", "u0u0", "0u0u", "ud00", "0ud0", "00ud"]
                >>> energy = pymacis.diagonalize(H=ham, determinants=dets)
                >>> print(f"Energy in restricted basis: {energy:.8f} Hartree")
        )pbdoc",
        py::arg("H"), py::arg("determinants"));
}

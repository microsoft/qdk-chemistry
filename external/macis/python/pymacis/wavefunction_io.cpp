// Copyright (c) Microsoft Corporation.

#include "wavefunction_io.hpp"

#include <macis/wavefunction_io.hpp>

struct write_wavefunction_helper {
  /**
   * @brief Template implementation of wavefunction writing
   * @tparam N Number of bits for wavefunction representation
   * @param filename Output filename
   * @param norb Number of orbitals
   * @param dets Vector of wavefunction determinants
   * @param coeffs Coefficients vector
   */
  template <size_t N>
  static void impl(const std::string &filename, size_t norb,
                   const py::list &dets_strings, const np_double_array &coeffs);
};

template <size_t N>
void write_wavefunction_helper::impl(const std::string &filename, size_t norb,
                                     const py::list &dets_strings,
                                     const np_double_array &coeffs) {
  using wfn_type = macis::wfn_t<N>;
  std::vector<wfn_type> dets = strings_to_wfn_vector<N>(dets_strings);
  std::vector<double> C = array_to_vector<double>(coeffs);
  macis::write_wavefunction(filename, norb, dets, C);
}

void write_wavefunction_wrapper(const std::string &filename, size_t norb,
                                const py::list &dets_strings,
                                const np_double_array &coeffs) {
  dispatch_by_norb<write_wavefunction_helper>(norb, filename, norb,
                                              dets_strings, coeffs);
}

struct read_wavefunction_helper {
  /**
   * @brief Template implementation of wavefunction reading
   * @tparam N Number of bits for wavefunction representation
   * @param filename Input filename containing wavefunction data
   * @return Python dictionary containing determinants and coefficients
   */
  template <size_t N>
  static py::dict impl(const std::string &filename);
};

template <size_t N>
py::dict read_wavefunction_helper::impl(const std::string &filename) {
  throw_if_file_not_found(filename);
  using wfn_type = macis::wfn_t<N>;
  size_t norb;
  std::vector<wfn_type> dets;
  std::vector<double> C;

  macis::read_wavefunction(filename, dets, C);

  py::dict result;
  result["determinants"] = wfn_vector_to_strings(dets, norb);
  result["coefficients"] = vector_to_array(C);

  return result;
}

py::dict read_wavefunction_wrapper(const std::string &filename) {
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    throw std::runtime_error("Could not open file: " + filename);
  }
  size_t norb = 0;
  infile >> norb >> norb;  // hacky :(
  infile.close();
  if (norb == 0) {
    throw std::runtime_error("Number of orbitals must be greater than 0");
  }

  // Dispatch based on number of orbitals
  auto result = dispatch_by_norb<read_wavefunction_helper>(norb, filename);
  result["norbitals"] = norb;  // Add norbitals to result
  return result;
}

/**
 * @brief Register wavefunction I/O functions with pybind11 module
 *
 * Exports the read_wavefunction and write_wavefunction functions to the Python
 * module. These functions provide the ability to save and load CI wavefunctions
 * for analysis, visualization, or as initial guesses for subsequent
 * calculations.
 *
 * @param m PyBind11 module to which the functions will be added
 */
void export_wavefunction_io_pybind(py::module &m) {
  // Wavefunction I/O
  m.def("write_wavefunction", &write_wavefunction_wrapper,
        R"pbdoc(
            Write CI wavefunction data to a file.

            Saves the CI wavefunction (determinants and coefficients) to a binary
            file for later analysis or as input to other calculations. The file
            format is compact and includes all necessary information to reconstruct
            the wavefunction.

            Args:
                filename (str): Output filename for wavefunction data.
                norb (int): Number of molecular orbitals in the system.
                determinants (list): List of determinant strings in binary format.
                    Each string represents a determinant in electron occupation (e.g., "110010" for
                    electrons in orbitals 0,1,4).
                coefficients (numpy.ndarray): CI coefficients corresponding to each determinant.
                    Must have same length as determinants list.

            Raises:
                ValueError: If determinants and coefficients have different lengths.
                IOError: If file cannot be written.

            Note:
                The determinant strings use binary representation where '1' indicates
                an occupied orbital and '0' indicates empty. The ordering follows
                alpha orbitals first, then beta orbitals (spin-orbital ordering).

            Example:
                >>> # Save CASCI wavefunction (need to request determinants)
                >>> settings = {'return_determinants': True}
                >>> result = pymacis.casci(nalpha=5, nbeta=5, H=ham, settings=settings)
                >>> pymacis.write_wavefunction(
                ...     "casci_wfn.dat",
                ...     ham.norb,
                ...     result['determinants'],
                ...     result['coefficients']
                ... )
                >>> print("Wavefunction saved to casci_wfn.dat")
                >>>
                >>> # Save ASCI wavefunction (determinants always included)
                >>> result = pymacis.asci(initial_dets, coeffs, E0, ham, settings)
                >>> pymacis.write_wavefunction(
                ...     "asci_wfn.dat",
                ...     ham.norb,
                ...     result['determinants'],
                ...     result['coefficients']
                ... )
        )pbdoc",
        py::arg("filename"), py::arg("norb"), py::arg("determinants"),
        py::arg("coefficients"));

  m.def("read_wavefunction", &read_wavefunction_wrapper,
        R"pbdoc(
            Read CI wavefunction data from a file.

            Loads a previously saved CI wavefunction from file, returning all
            the information needed to analyze or continue calculations with
            the wavefunction.

            Args:
                filename (str): Input filename containing wavefunction data.

            Returns:
                dict: Dictionary containing wavefunction data:
                    - 'norbitals': Number of molecular orbitals (int)
                    - 'determinants': List of determinant strings in binary format
                    - 'coefficients': NumPy array of CI coefficients

            Raises:
                FileNotFoundError: If the specified file does not exist.
                IOError: If file format is invalid or corrupted.

            Note:
                The loaded wavefunction can be used for analysis, property calculations,
                or as an initial guess for subsequent CI calculations.

            Example:
                >>> # Load previously saved wavefunction
                >>> wfn_data = pymacis.read_wavefunction("casci_wfn.dat")
                >>> print(f"Loaded wavefunction with {len(wfn_data['determinants'])} determinants")
                >>> print(f"Number of orbitals: {wfn_data['norbitals']}")
                >>>
                >>> # Analyze coefficients
                >>> coeffs = wfn_data['coefficients']
                >>> dominant_det = wfn_data['determinants'][np.argmax(np.abs(coeffs))]
                >>> print(f"Dominant determinant: {dominant_det}")
                >>> print(f"Coefficient: {np.max(np.abs(coeffs)):.4f}")
        )pbdoc",
        py::arg("filename"));
}

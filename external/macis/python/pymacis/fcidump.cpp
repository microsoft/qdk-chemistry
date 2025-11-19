// Copyright (c) Microsoft Corporation.

#include "fcidump.hpp"

Hamiltonian read_fcidump(const std::string &filename) {
  // Read all integral data in a single pass from a FCIDUMP file
  throw_if_file_not_found(filename);
  size_t norb = macis::read_fcidump_norb(filename);
  if (norb == 0) {
    throw std::runtime_error("No orbitals found in FCIDUMP file: " + filename);
  }

  std::vector<double> t_matrix(norb * norb, 0.0);
  std::vector<double> v_tensor(norb * norb * norb * norb, 0.0);
  double *t_ptr = static_cast<double *>(t_matrix.data());
  double *v_ptr = static_cast<double *>(v_tensor.data());
  double core_energy = 0.0;

  // Read all integrals in a single pass
  macis::read_fcidump_all(filename, t_ptr, norb, v_ptr, norb, core_energy);

  Hamiltonian ham;
  ham.norb = norb;
  ham.nbasis = norb;
  ham.core_energy = core_energy;
  ham._T = std::move(t_matrix);
  ham._V = std::move(v_tensor);
  return ham;
}

py::array_t<double> read_fcidump_1body(const std::string &filename) {
  throw_if_file_not_found(filename);

  size_t norb = macis::read_fcidump_norb(filename);
  if (norb == 0) {
    throw std::runtime_error("No orbitals found in FCIDUMP file: " + filename);
  }
  auto result = py::array_t<double>({norb, norb});
  py::buffer_info buf = result.request();
  double *ptr = static_cast<double *>(buf.ptr);

  macis::read_fcidump_1body(filename, ptr, norb);
  return result;
}

py::array_t<double> read_fcidump_2body(const std::string &filename) {
  throw_if_file_not_found(filename);

  size_t norb = macis::read_fcidump_norb(filename);
  if (norb == 0) {
    throw std::runtime_error("Number of orbitals must be greater than 0");
  }
  auto result = py::array_t<double>({norb, norb, norb, norb});
  py::buffer_info buf = result.request();
  double *ptr = static_cast<double *>(buf.ptr);

  macis::read_fcidump_2body(filename, ptr, norb);
  return result;
}

double read_fcidump_core_energy(const std::string &filename) {
  throw_if_file_not_found(filename);
  return macis::read_fcidump_core(filename);
}

size_t read_fcidump_norb(const std::string &filename) {
  throw_if_file_not_found(filename);
  return macis::read_fcidump_norb(filename);
}

macis::FCIDumpHeader read_fcidump_header(const std::string &filename) {
  throw_if_file_not_found(filename);
  return macis::fcidump_read_header(filename);
}

void write_fcidump(const std::string &filename,
                   const macis::FCIDumpHeader &header,
                   const py::array_t<double> &T, const py::array_t<double> &V,
                   double core_energy, double threshold) {
  // Validate input arrays
  if (T.ndim() != 2) {
    throw std::runtime_error("T matrix must be 2-dimensional");
  }
  if (V.ndim() != 4) {
    throw std::runtime_error("V tensor must be 4-dimensional");
  }
  if (T.shape(0) != T.shape(1)) {
    throw std::runtime_error("T matrix must be square");
  }
  if (V.shape(0) != V.shape(1) || V.shape(0) != V.shape(2) ||
      V.shape(0) != V.shape(3)) {
    throw std::runtime_error("V tensor must have equal dimensions");
  }
  if (T.shape(0) != V.shape(0)) {
    throw std::runtime_error("T and V must have compatible dimensions");
  }
  if (T.shape(0) != header.norb) {
    throw std::runtime_error("Matrix dimensions must match header.norb");
  }

  // Get pointers to data
  const double *T_ptr = static_cast<const double *>(T.data());
  const double *V_ptr = static_cast<const double *>(V.data());

  // Call the C++ function
  macis::write_fcidump(filename, header, T_ptr, T.shape(0), V_ptr, V.shape(0),
                       core_energy, threshold);
}

void export_fcidump_pybind(py::module &m) {
  // Export FCIDumpHeader class
  py::class_<macis::FCIDumpHeader>(m, "FCIDumpHeader", R"pbdoc(
        FCIDUMP header information structure.

        This class holds all the header parameters extracted from a FCIDUMP file,
        including system information like number of orbitals, electrons, spin
        multiplicity, and orbital symmetries.

        Attributes:
            norb (int): Number of molecular orbitals.
            nelec (int): Total number of electrons in the system.
            ms2 (int): Twice the total spin quantum number (2*S).
            isym (int): Point group symmetry label.
            orbsym (list): List of orbital symmetry labels for each orbital.
    )pbdoc")
      .def(py::init<>(), "Create a new FCIDumpHeader with default values.")
      .def_readwrite("norb", &macis::FCIDumpHeader::norb, R"pbdoc(
                Number of molecular orbitals in the active space.
            )pbdoc")
      .def_readwrite("nelec", &macis::FCIDumpHeader::nelec, R"pbdoc(
                Total number of electrons in the system.
            )pbdoc")
      .def_readwrite("ms2", &macis::FCIDumpHeader::ms2, R"pbdoc(
                Twice the total spin quantum number.

                For singlet: ms2=0, doublet: ms2=1, triplet: ms2=2, etc.
            )pbdoc")
      .def_readwrite("isym", &macis::FCIDumpHeader::isym, R"pbdoc(
                Point group symmetry label.
            )pbdoc")
      .def_readwrite("orbsym", &macis::FCIDumpHeader::orbsym, R"pbdoc(
                List of orbital symmetry labels for each orbital.
            )pbdoc");

  // Parse FCIDUMP header function
  m.def("read_fcidump_header", &read_fcidump_header,
        R"pbdoc(
            Parse the header section of a FCIDUMP file.

            Reads and parses the &FCI namelist section of a FCIDUMP file to extract
            all system parameters including number of orbitals, electrons, spin
            multiplicity, symmetry information, and orbital symmetries.

            This function provides access to the complete header information without
            reading the potentially large integral data section, making it useful
            for quickly inspecting FCIDUMP file properties.

            Args:
                filename (str): Path to the FCIDUMP file to parse.

            Returns:
                FCIDumpHeader: Object containing all header parameters:
                    - norb: Number of molecular orbitals
                    - nelec: Total number of electrons
                    - ms2: Twice the total spin quantum number
                    - isym: Point group symmetry label
                    - orbsym: List of orbital symmetry labels

            Raises:
                FileNotFoundError: If the specified file does not exist.
                RuntimeError: If the file format is invalid or header parsing fails.

            Example:
                >>> header = pymacis.fcidump_read_header("h2o.fcidump")
                >>> print(f"System has {header.norb} orbitals and {header.nelec} electrons")
                >>> print(f"Spin multiplicity: {header.ms2/2 + 0.5}")
                >>> print(f"Orbital symmetries: {header.orbsym}")
                >>>
                >>> # Quick access to number of orbitals without reading integrals
                >>> norb_from_header = header.norb
                >>> # This is equivalent to but more informative than:
                >>> # norb_from_function = pymacis.read_fcidump_norb("h2o.fcidump")
        )pbdoc",
        py::arg("filename"));

  // Read NORB from FCIDUMP
  m.def("read_fcidump_norb", &read_fcidump_norb,
        R"pbdoc(
            Read the number of orbitals from a FCIDUMP file.

            Parses the FCIDUMP file header to extract the NORB parameter,
            which specifies the number of molecular orbitals.

            Args:
                filename (str): Path to the FCIDUMP file to read.

            Returns:
                int: Number of orbitals (NORB) found in the file.

            Raises:
                FileNotFoundError: If the specified file does not exist.
                RuntimeError: If the file format is invalid or NORB cannot be parsed.

            Example:
                >>> norb = pymacis.read_fcidump_norb("h2o.fcidump")
                >>> print(f"System has {norb} orbitals")
        )pbdoc",
        py::arg("filename"));

  // Read the core energy from FCIDUMP
  m.def("read_fcidump_core", &read_fcidump_core_energy,
        R"pbdoc(
            Read the core energy from a FCIDUMP file.

            Extracts the core energy (nuclear repulsion + inactive contributions)
            from the FCIDUMP file. This energy is added to CI energies to get
            total molecular energies.

            Args:
                filename (str): Path to the FCIDUMP file to read.

            Returns:
                float: Core energy in atomic units (Hartree).

            Raises:
                FileNotFoundError: If the specified file does not exist.
                RuntimeError: If the file format is invalid.

            Example:
                >>> core_e = pymacis.read_fcidump_core("h2o.fcidump")
                >>> print(f"Core energy: {core_e:.6f} Hartree")
        )pbdoc",
        py::arg("filename"));

  // Read the 1-body integrals from FCIDUMP
  m.def("read_fcidump_1body", &read_fcidump_1body,
        R"pbdoc(
            Read one-electron integrals from a FCIDUMP file.

            Extracts the one-electron integrals (kinetic energy + nuclear attraction)
            and returns them as a 2D NumPy array.

            Args:
                filename (str): Path to the FCIDUMP file to read.

            Returns:
                numpy.ndarray: Shape (norb, norb) array containing one-electron integrals
                T[i,j] = <i|h|j> where h is the one-electron Hamiltonian operator.

            Raises:
                FileNotFoundError: If the specified file does not exist.
                RuntimeError: If no orbitals found or file format is invalid.

            Example:
                >>> T = pymacis.read_fcidump_1body("h2o.fcidump")
                >>> print(f"One-electron integral T[0,0] = {T[0,0]:.6f}")
        )pbdoc",
        py::arg("filename"));

  // Read the 2-body integrals from FCIDUMP
  m.def("read_fcidump_2body", &read_fcidump_2body,
        R"pbdoc(
            Read two-electron integrals from a FCIDUMP file.

            Extracts the two-electron repulsion integrals and returns them as
            a 4D NumPy array in physicist's notation.

            Args:
                filename (str): Path to the FCIDUMP file to read.

            Returns:
                numpy.ndarray: Shape (norb, norb, norb, norb) array containing two-electron integrals
                V[i,j,k,l] = <ij|kl> = ∫∫ φᵢ(r₁)φⱼ(r₁) (1/r₁₂) φₖ(r₂)φₗ(r₂) dr₁dr₂

            Raises:
                FileNotFoundError: If the specified file does not exist.
                RuntimeError: If no orbitals found or file format is invalid.

            Note:
                The integrals use physicist's notation where V[i,j,k,l] corresponds
                to the integral <ij|kl> with electron 1 in orbitals i,j and electron 2
                in orbitals k,l.

            Example:
                >>> V = pymacis.read_fcidump_2body("h2o.fcidump")
                >>> print(f"Two-electron integral V[0,0,0,0] = {V[0,0,0,0]:.6f}")
        )pbdoc",
        py::arg("filename"));

  // Read all relevant data from FCIDUMP
  m.def("read_fcidump", &read_fcidump,
        R"pbdoc(
            Read complete molecular integral data from a FCIDUMP file.

            This is the primary function for loading molecular integrals. It reads
            all necessary data (number of orbitals, core energy, one- and two-electron
            integrals) and returns a complete Hamiltonian object ready for CI calculations.

            Args:
                filename (str): Path to the FCIDUMP file to read.

            Returns:
                Hamiltonian: Complete Hamiltonian object containing:
                    - norb: Number of molecular orbitals
                    - core_energy: Nuclear repulsion + inactive orbital energy
                    - T: One-electron integrals (norb x norb)
                    - V: Two-electron integrals (norb x norb x norb x norb)

            Raises:
                FileNotFoundError: If the specified file does not exist.
                RuntimeError: If no orbitals found or file format is invalid.

            Example:
                >>> ham = pymacis.read_fcidump("h2o.fcidump")
                >>> print(f"Loaded system with {ham.norb} orbitals")
                >>> print(f"Core energy: {ham.core_energy:.6f} Hartree")
                >>> # Ready for CI calculations
                >>> result = pymacis.casci(nalpha=5, nbeta=5, H=ham)
        )pbdoc",
        py::arg("filename"));

  // Write FCIDUMP functions
  m.def("write_fcidump", &write_fcidump,
        R"pbdoc(
            Write molecular integrals to a FCIDUMP file with complete header information.

            Creates a properly formatted FCIDUMP file with a complete header containing
            system information (number of orbitals, electrons, spin, symmetry) followed
            by the molecular integrals in the standard FCIDUMP format.

            Args:
                filename (str): Path where the FCIDUMP file should be written.
                header (FCIDumpHeader): Header information containing system parameters.
                T (numpy.ndarray): One-electron integrals, shape (norb, norb).
                V (numpy.ndarray): Two-electron integrals, shape (norb, norb, norb, norb).
                core_energy (float): Nuclear repulsion plus inactive orbital energy.
                threshold (float, optional): Threshold for writing integrals. Defaults to 1e-15.

            Raises:
                RuntimeError: If array dimensions are invalid or inconsistent.

            Example:
                >>> # Create header information
                >>> header = pymacis.FCIDumpHeader()
                >>> header.norb = 4
                >>> header.nelec = 4
                >>> header.ms2 = 0  # Singlet
                >>> header.isym = 1
                >>> header.orbsym = [1, 1, 1, 1]
                >>>
                >>> # Write FCIDUMP file
                >>> pymacis.write_fcidump("output.fcidump", header, T, V, core_energy)
                >>>
                >>> # Write FCIDUMP file with custom threshold
                >>> pymacis.write_fcidump("output.fcidump", header, T, V, core_energy, 1e-12)
        )pbdoc",
        py::arg("filename"), py::arg("header"), py::arg("T"), py::arg("V"),
        py::arg("core_energy"), py::arg("threshold") = 1e-15);
}

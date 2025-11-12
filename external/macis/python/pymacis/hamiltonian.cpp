#include "hamiltonian.hpp"

void export_hamiltonian_pybind(py::module &m) {
  // Export Hamiltonian class
  py::class_<Hamiltonian>(m, "Hamiltonian", R"pbdoc(
        Hamiltonian class for storing molecular integrals and system information.

        This class contains all the molecular orbital integrals needed for CI calculations,
        including one-electron (T) and two-electron (V) integrals, along with system
        parameters like number of orbitals and core energy.

        The integrals are stored in physicist's notation: V[i,j,k,l] = <ij|kl>
        where <ij|kl> represents the two-electron integral over molecular orbitals i,j,k,l.
    )pbdoc")
      .def(py::init<>(), R"pbdoc(
            Default constructor for empty Hamiltonian.

            Creates an empty Hamiltonian object. Use read_fcidump() to populate
            with molecular integral data from a file.
        )pbdoc")
      .def_property_readonly(
          "norb", [](const Hamiltonian &self) { return self.norb; },
          R"pbdoc(
                Number of active molecular orbitals.

                Returns:
                    int: Number of orbitals in the active space for CI calculations.
            )pbdoc")
      .def_property_readonly(
          "nbasis", [](const Hamiltonian &self) { return self.nbasis; },
          R"pbdoc(
                Number of basis functions.

                Returns:
                    int: Total number of basis functions from which active orbitals are selected.
                    For most cases, this equals norb unless using an active space subset.
            )pbdoc")
      .def_property_readonly(
          "core_energy",
          [](const Hamiltonian &self) { return self.core_energy; },
          R"pbdoc(
                Nuclear repulsion energy plus inactive orbital contributions.

                Returns:
                    float: Core energy that is added to the electronic energy from CI calculations
                    to obtain the total molecular energy.
            )pbdoc")
      .def_property_readonly(
          "T",
          [](const Hamiltonian &self) {
            return py::array_t<double>(
                {self.norb, self.norb},
                {sizeof(double),
                 sizeof(double) * self.norb},  // Col major order
                self._T.data(),                // Pointer to data
                py::cast(self));
          },
          R"pbdoc(
            One-electron integrals matrix (kinetic + nuclear attraction).

            Returns:
                numpy.ndarray: Shape (norb, norb) array containing one-electron integrals
                T[i,j] = <i|h|j> where h is the one-electron Hamiltonian operator.
                Stored in column-major (Fortran) order for compatibility with linear algebra libraries.
        )pbdoc")
      .def_property_readonly(
          "F_inactive",
          [](const Hamiltonian &self) {
            if (self._F_inactive.size() == 0) {
              // If not allocated, assume zero matrix
              py::array_t<double> arr({self.nbasis, self.nbasis});
              std::fill(arr.mutable_data(), arr.mutable_data() + arr.size(),
                        0.0);
              return arr;
            } else
              return py::array_t<double>(
                  {self.nbasis, self.nbasis},
                  {sizeof(double),
                   sizeof(double) * self.nbasis},  // Col major order
                  self._F_inactive.data(),         // Pointer to data
                  py::cast(self));
          },
          R"pbdoc(
            Inactive Fock matrix for active space calculations.

            Returns:
                numpy.ndarray: Shape (nbasis, nbasis) Fock matrix containing the mean-field
                contribution from inactive (doubly occupied) orbitals. Returns zero matrix
                if no inactive orbitals are present (full CI case).
        )pbdoc")
      .def_property_readonly(
          "V",
          [](const Hamiltonian &self) {
            return py::array_t<double>(
                {self.norb, self.norb, self.norb, self.norb},
                {sizeof(double), sizeof(double) * self.norb,
                 sizeof(double) * self.norb * self.norb,
                 sizeof(double) * self.norb * self.norb *
                     self.norb},  // Col major order
                self._V.data(),   // Pointer to data
                py::cast(self));
          },
          R"pbdoc(
            Two-electron integrals tensor in physicist's notation.

            Returns:
                numpy.ndarray: Shape (norb, norb, norb, norb) array containing two-electron integrals
                V[i,j,k,l] = <ij|kl> = ∫∫ φᵢ(r₁)φⱼ(r₁) (1/r₁₂) φₖ(r₂)φₗ(r₂) dr₁dr₂
                Stored in column-major order for efficient access patterns.
            )pbdoc");
}

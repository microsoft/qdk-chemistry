// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <pybind11/eigen.h>
#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <qdk/chemistry/utils/model_hamiltonians/ppp.hpp>

namespace py = pybind11;

namespace {

/// Convert a Python scalar or 1-D array to Eigen::VectorXd.
/// Scalars are broadcast to a constant vector of size n.
Eigen::VectorXd to_site_param(const py::object& obj, int n) {
  if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
    return Eigen::VectorXd::Constant(n, obj.cast<double>());
  }
  return obj.cast<Eigen::VectorXd>();
}

/// Convert a Python scalar or 2-D array to Eigen::MatrixXd.
/// Scalars are broadcast to a constant n×n matrix.
Eigen::MatrixXd to_pair_param(const py::object& obj, int n) {
  if (py::isinstance<py::float_>(obj) || py::isinstance<py::int_>(obj)) {
    return Eigen::MatrixXd::Constant(n, n, obj.cast<double>());
  }
  return obj.cast<Eigen::MatrixXd>();
}

}  // namespace

void bind_model_hamiltonians(pybind11::module& m) {
  using namespace qdk::chemistry::utils::model_hamiltonians;
  using qdk::chemistry::data::Hamiltonian;
  using qdk::chemistry::data::LatticeGraph;

  auto mh = m.def_submodule("model_hamiltonians", R"(
Model Hamiltonian builders and intersite potential functions.

This submodule provides convenience factory functions that construct a
:class:`~qdk_chemistry.data.Hamiltonian` for common lattice models
(Hückel, Hubbard, Pariser-Parr-Pople) as well as helper functions for
computing intersite Coulomb potential matrices (Ohno, Mataga-Nishimoto).

All parameters that represent per-site quantities (``epsilon``, ``U``, ``z``)
accept either a scalar ``float`` (broadcast to every site) or
a ``numpy.ndarray`` of shape ``(n,)``. Parameters that represent per-pair
quantities (``t``, ``V``, ``R``) accept a scalar or ``numpy.ndarray``
of shape ``(n, n)``. These can be freely mixed.
)");

  // ======================================================================
  // Hamiltonian factory functions
  // ======================================================================

  // --- Hückel -----------------------------------------------------------
  mh.def(
      "create_huckel_hamiltonian",
      [](const LatticeGraph& lattice, const py::object& epsilon,
         const py::object& t) {
        auto n = static_cast<int>(lattice.num_sites());
        return create_huckel_hamiltonian(lattice, to_site_param(epsilon, n),
                                         to_pair_param(t, n));
      },
      R"(
Create a Hückel model Hamiltonian.

Builds the one-body Hamiltonian
``H = sum_i eps_i n_i - sum_{<i,j>} t_ij (a†_i a_j + h.c.)``
and wraps it in a ready-to-use :class:`~qdk_chemistry.data.Hamiltonian`.

Args:
    lattice (LatticeGraph): Symmetric lattice graph defining connectivity.
    epsilon (float or numpy.ndarray): On-site orbital energy/energies.
    t (float or numpy.ndarray): Hopping integral(s).

Returns:
    Hamiltonian: Hückel model Hamiltonian.
)",
      py::arg("lattice"), py::arg("epsilon"), py::arg("t"));

  // --- Hubbard ----------------------------------------------------------
  mh.def(
      "create_hubbard_hamiltonian",
      [](const LatticeGraph& lattice, const py::object& epsilon,
         const py::object& t, const py::object& U) {
        auto n = static_cast<int>(lattice.num_sites());
        return create_hubbard_hamiltonian(lattice, to_site_param(epsilon, n),
                                          to_pair_param(t, n),
                                          to_site_param(U, n));
      },
      R"(
Create a Hubbard model Hamiltonian.

Extends the Hückel model with on-site Coulomb repulsion
``H = H_huckel + U sum_i n_{i,up} n_{i,down}``.

Args:
    lattice (LatticeGraph): Symmetric lattice graph defining connectivity.
    epsilon (float or numpy.ndarray): On-site orbital energy/energies.
    t (float or numpy.ndarray): Hopping integral(s).
    U (float or numpy.ndarray): On-site Coulomb repulsion(s).

Returns:
    Hamiltonian: Hubbard model Hamiltonian.
)",
      py::arg("lattice"), py::arg("epsilon"), py::arg("t"), py::arg("U"));

  // --- PPP --------------------------------------------------------------
  mh.def(
      "create_ppp_hamiltonian",
      [](const LatticeGraph& lattice, const py::object& epsilon,
         const py::object& t, const py::object& U, const py::object& V,
         const py::object& z) {
        auto n = static_cast<int>(lattice.num_sites());
        return create_ppp_hamiltonian(lattice, to_site_param(epsilon, n),
                                      to_pair_param(t, n), to_site_param(U, n),
                                      to_pair_param(V, n), to_site_param(z, n));
      },
      R"(
Create a Pariser-Parr-Pople (PPP) model Hamiltonian.

Extends the Hubbard model with long-range intersite Coulomb interactions
``H = H_hubbard + 1/2 sum_{i!=j} V_ij (n_i - z_i)(n_j - z_j)``.

Args:
    lattice (LatticeGraph): Symmetric lattice graph defining connectivity.
    epsilon (float or numpy.ndarray): On-site orbital energy/energies.
    t (float or numpy.ndarray): Hopping integral(s).
    U (float or numpy.ndarray): On-site Coulomb repulsion(s).
    V (float or numpy.ndarray): Intersite Coulomb interaction(s).
    z (float or numpy.ndarray): Effective core charge(s).

Returns:
    Hamiltonian: PPP model Hamiltonian.
)",
      py::arg("lattice"), py::arg("epsilon"), py::arg("t"), py::arg("U"),
      py::arg("V"), py::arg("z"));

  // ======================================================================
  // Potential functions
  // ======================================================================

  // --- Ohno potential ---------------------------------------------------
  mh.def(
      "ohno_potential",
      [](const LatticeGraph& lattice, const py::object& U, const py::object& R,
         double epsilon_r, bool nearest_neighbor_only) {
        auto n = static_cast<int>(lattice.num_sites());
        return ohno_potential(lattice, to_site_param(U, n), to_pair_param(R, n),
                              epsilon_r, nearest_neighbor_only);
      },
      R"(
Compute the Ohno intersite potential matrix.

``V_ij = U_ij / sqrt(1 + (U_ij * epsilon_r * R_ij / C)^2)``

where ``U_ij = sqrt(U_i * U_j)`` and ``C = e^2 / (4 pi epsilon_0)``.

Args:
    lattice (LatticeGraph): Lattice graph (used for the number of sites).
    U (float or numpy.ndarray): On-site Coulomb parameter(s).
    R (float or numpy.ndarray): Intersite distance(s).
    epsilon_r (float, optional): Relative permittivity. Defaults to 1.0.
    nearest_neighbor_only (bool, optional): If True, restrict to lattice-connected
        pairs. Defaults to False.

Returns:
    numpy.ndarray: Symmetric potential matrix [n x n].
)",
      py::arg("lattice"), py::arg("U"), py::arg("R"),
      py::arg("epsilon_r") = 1.0, py::arg("nearest_neighbor_only") = false);

  // --- Mataga-Nishimoto potential ---------------------------------------
  mh.def(
      "mataga_nishimoto_potential",
      [](const LatticeGraph& lattice, const py::object& U, const py::object& R,
         double epsilon_r, bool nearest_neighbor_only) {
        auto n = static_cast<int>(lattice.num_sites());
        return mataga_nishimoto_potential(lattice, to_site_param(U, n),
                                          to_pair_param(R, n), epsilon_r,
                                          nearest_neighbor_only);
      },
      R"(
Compute the Mataga-Nishimoto intersite potential matrix.

``V_ij = U_ij / (1 + U_ij * epsilon_r * R_ij / C)``

where ``U_ij = sqrt(U_i * U_j)`` and ``C = e^2 / (4 pi epsilon_0)``.

Args:
    lattice (LatticeGraph): Lattice graph (used for the number of sites).
    U (float or numpy.ndarray): On-site Coulomb parameter(s).
    R (float or numpy.ndarray): Intersite distance(s).
    epsilon_r (float, optional): Relative permittivity. Defaults to 1.0.
    nearest_neighbor_only (bool, optional): If True, restrict to lattice-connected
        pairs. Defaults to False.

Returns:
    numpy.ndarray: Symmetric potential matrix [n x n].
)",
      py::arg("lattice"), py::arg("U"), py::arg("R"),
      py::arg("epsilon_r") = 1.0, py::arg("nearest_neighbor_only") = false);

  // --- pairwise_potential (callable version) ----------------------------
  mh.def(
      "pairwise_potential",
      [](const LatticeGraph& lattice, const py::object& U, const py::object& R,
         std::function<double(int, int, double, double)> func,
         bool nearest_neighbor_only) {
        auto n = static_cast<int>(lattice.num_sites());
        return pairwise_potential(lattice, to_site_param(U, n),
                                  to_pair_param(R, n), std::move(func),
                                  nearest_neighbor_only);
      },
      R"(
Compute a symmetric pairwise potential matrix from a user-supplied formula.

For each unique pair (i < j), computes ``U_ij = sqrt(U_i * U_j)``, reads
``R_ij``, and evaluates ``func(i, j, U_ij, R_ij)``.

Args:
    lattice (LatticeGraph): Lattice graph.
    U (float or numpy.ndarray): On-site Coulomb parameter(s).
    R (float or numpy.ndarray): Distance(s).
    func (callable): ``(i: int, j: int, Uij: float, Rij: float) -> float``.
    nearest_neighbor_only (bool, optional): Restrict to connected pairs.
        Defaults to False.

Returns:
    numpy.ndarray: Symmetric potential matrix [n x n].
)",
      py::arg("lattice"), py::arg("U"), py::arg("R"), py::arg("func"),
      py::arg("nearest_neighbor_only") = false);
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <nanobind/nanobind.h>
#include <nanobind/stl/function.h>

#include <qdk/chemistry/utils/model_hamiltonians.hpp>

#include "eigen.hpp"
#include "stl.hpp"

namespace nb = nanobind;

namespace {

namespace detail = qdk::chemistry::utils::model_hamiltonians::detail;

/// Dispatch a nb::object to the appropriate C++ to_site_param overload.
Eigen::VectorXd to_site_param(const nb::object& obj,
                              const qdk::chemistry::data::LatticeGraph& lattice,
                              const std::string& name = "parameter") {
  if (nb::isinstance<nb::float_>(obj) || nb::isinstance<nb::int_>(obj)) {
    return detail::to_site_param(nb::cast<double>(obj), lattice, name);
  }
  return detail::to_site_param(nb::cast<Eigen::VectorXd>(obj), lattice, name);
}

/// Dispatch a nb::object to the appropriate C++ to_pair_param overload.
Eigen::MatrixXd to_pair_param(const nb::object& obj,
                              const qdk::chemistry::data::LatticeGraph& lattice,
                              const std::string& name = "parameter") {
  if (nb::isinstance<nb::float_>(obj) || nb::isinstance<nb::int_>(obj)) {
    return detail::to_pair_param(nb::cast<double>(obj), lattice, name);
  }
  return detail::to_pair_param(nb::cast<Eigen::MatrixXd>(obj), lattice, name);
}

}  // namespace

void bind_model_hamiltonians(nb::module_& m) {
  using namespace qdk::chemistry::utils::model_hamiltonians;

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
  // Utility functions
  // ======================================================================

  mh.def(
      "to_site_param",
      [](const nb::object& value,
         const qdk::chemistry::data::LatticeGraph& lattice,
         const std::string& name) {
        return to_site_param(value, lattice, name);
      },
      R"(
Broadcast a scalar or 1-D array to a per-site parameter vector.

If ``value`` is a scalar, returns a constant ``numpy.ndarray`` of length
``n``.  If it is already a 1-D array, it is validated against the lattice
site count and returned.

Args:
    value (float or numpy.ndarray): Scalar or per-site array.
    lattice (LatticeGraph): Lattice graph defining the expected size.
    name (str, optional): Parameter name for error messages.
        Defaults to ``"parameter"``.

Returns:
    numpy.ndarray: 1-D array of length ``n``.
)",
      nb::arg("value"), nb::arg("lattice"), nb::arg("name") = "parameter");

  mh.def(
      "to_pair_param",
      [](const nb::object& value,
         const qdk::chemistry::data::LatticeGraph& lattice,
         const std::string& name) {
        return to_pair_param(value, lattice, name);
      },
      R"(
Broadcast a scalar or 2-D array to a per-pair parameter matrix.

If ``value`` is a scalar, returns a constant ``numpy.ndarray`` of shape
``(n, n)``.  If it is already a 2-D array, it is validated against the
lattice site count and returned.

Args:
    value (float or numpy.ndarray): Scalar or per-pair matrix.
    lattice (LatticeGraph): Lattice graph defining the expected size.
    name (str, optional): Parameter name for error messages.
        Defaults to ``"parameter"``.

Returns:
    numpy.ndarray: 2-D array of shape ``(n, n)``.
)",
      nb::arg("value"), nb::arg("lattice"), nb::arg("name") = "parameter");

  // ======================================================================
  // Hamiltonian factory functions
  // ======================================================================

  // --- Hückel -----------------------------------------------------------
  mh.def(
      "create_huckel_hamiltonian",
      [](const qdk::chemistry::data::LatticeGraph& lattice,
         const nb::object& epsilon, const nb::object& t) {
        return create_huckel_hamiltonian(
            lattice, to_site_param(epsilon, lattice, "epsilon"),
            to_pair_param(t, lattice, "t"));
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
      nb::arg("lattice"), nb::arg("epsilon"), nb::arg("t"));

  // --- Hubbard ----------------------------------------------------------
  mh.def(
      "create_hubbard_hamiltonian",
      [](const qdk::chemistry::data::LatticeGraph& lattice,
         const nb::object& epsilon, const nb::object& t, const nb::object& U) {
        return create_hubbard_hamiltonian(
            lattice, to_site_param(epsilon, lattice, "epsilon"),
            to_pair_param(t, lattice, "t"), to_site_param(U, lattice, "U"));
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
      nb::arg("lattice"), nb::arg("epsilon"), nb::arg("t"), nb::arg("U"));

  // --- PPP --------------------------------------------------------------
  mh.def(
      "create_ppp_hamiltonian",
      [](const qdk::chemistry::data::LatticeGraph& lattice,
         const nb::object& epsilon, const nb::object& t, const nb::object& U,
         const nb::object& V, const nb::object& z) {
        return create_ppp_hamiltonian(
            lattice, to_site_param(epsilon, lattice, "epsilon"),
            to_pair_param(t, lattice, "t"), to_site_param(U, lattice, "U"),
            to_pair_param(V, lattice, "V"), to_site_param(z, lattice, "z"));
      },
      R"(
Create a Pariser-Parr-Pople (PPP) model Hamiltonian.

Extends the Hubbard model with long-range intersite Coulomb interactions
``H = H_hubbard + 1/2 sum_{i!=j} V_ij (n_i - z_i)(n_j - z_j)``.

Note: The stored two-body integrals do not include the 1/2 prefactor.

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
      nb::arg("lattice"), nb::arg("epsilon"), nb::arg("t"), nb::arg("U"),
      nb::arg("V"), nb::arg("z"));

  // ======================================================================
  // Potential functions
  // ======================================================================

  // --- Ohno potential ---------------------------------------------------
  mh.def(
      "ohno_potential",
      [](const qdk::chemistry::data::LatticeGraph& lattice, const nb::object& U,
         const nb::object& R, double epsilon_r, bool nearest_neighbor_only) {
        return ohno_potential(lattice, to_site_param(U, lattice, "U"),
                              to_pair_param(R, lattice, "R"), epsilon_r,
                              nearest_neighbor_only);
      },
      R"(
Compute the Ohno intersite potential matrix.

``V_ij = U_ij / sqrt(1 + (U_ij * epsilon_r * R_ij)^2)``

where ``U_ij = sqrt(U_i * U_j)`` is the geometric mean of on-site parameters.

All parameters should be in atomic units (Hartree for U, Bohr for R).

Args:
    lattice (LatticeGraph): Lattice graph (used for the number of sites).
    U (float or numpy.ndarray): On-site Coulomb parameter(s) in Hartree.
    R (float or numpy.ndarray): Intersite distance(s) in Bohr.
    epsilon_r (float, optional): Relative permittivity. Defaults to 1.0.
    nearest_neighbor_only (bool, optional): If True, restrict to lattice-connected
        pairs. Defaults to False.

Returns:
    numpy.ndarray: Symmetric potential matrix [n x n].
)",
      nb::arg("lattice"), nb::arg("U"), nb::arg("R"),
      nb::arg("epsilon_r") = 1.0, nb::arg("nearest_neighbor_only") = false);

  // --- Mataga-Nishimoto potential ---------------------------------------
  mh.def(
      "mataga_nishimoto_potential",
      [](const qdk::chemistry::data::LatticeGraph& lattice, const nb::object& U,
         const nb::object& R, double epsilon_r, bool nearest_neighbor_only) {
        return mataga_nishimoto_potential(
            lattice, to_site_param(U, lattice, "U"),
            to_pair_param(R, lattice, "R"), epsilon_r, nearest_neighbor_only);
      },
      R"(
Compute the Mataga-Nishimoto intersite potential matrix.

``V_ij = U_ij / (1 + U_ij * epsilon_r * R_ij)``

where ``U_ij = sqrt(U_i * U_j)`` is the geometric mean of on-site parameters.

All parameters should be in atomic units (Hartree for U, Bohr for R).

Args:
    lattice (LatticeGraph): Lattice graph (used for the number of sites).
    U (float or numpy.ndarray): On-site Coulomb parameter(s) in Hartree.
    R (float or numpy.ndarray): Intersite distance(s) in Bohr.
    epsilon_r (float, optional): Relative permittivity. Defaults to 1.0.
    nearest_neighbor_only (bool, optional): If True, restrict to lattice-connected
        pairs. Defaults to False.

Returns:
    numpy.ndarray: Symmetric potential matrix [n x n].
)",
      nb::arg("lattice"), nb::arg("U"), nb::arg("R"),
      nb::arg("epsilon_r") = 1.0, nb::arg("nearest_neighbor_only") = false);

  // --- pairwise_potential (callable version) ----------------------------
  mh.def(
      "pairwise_potential",
      [](const qdk::chemistry::data::LatticeGraph& lattice, const nb::object& U,
         const nb::object& R,
         std::function<double(int, int, double, double)> func,
         bool nearest_neighbor_only) {
        return pairwise_potential(lattice, to_site_param(U, lattice, "U"),
                                  to_pair_param(R, lattice, "R"),
                                  std::move(func), nearest_neighbor_only);
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
      nb::arg("lattice"), nb::arg("U"), nb::arg("R"), nb::arg("func"),
      nb::arg("nearest_neighbor_only") = false);
}

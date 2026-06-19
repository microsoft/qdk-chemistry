// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ctf12_hamiltonian.hpp"

#include <Eigen/Dense>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "ctf12_f12.hpp"
#include "ctf12_support.hpp"

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Hamiltonian> CtF12HamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Wavefunction> reference) const {
  QDK_LOG_TRACE_ENTERING();

  if (!reference) {
    throw std::invalid_argument(
        "CtF12HamiltonianConstructor: reference wavefunction is null");
  }

  const double gamma = _settings->get<double>("gamma");
  const std::string cabs_basis = _settings->get<std::string>("cabs_basis");
  const auto frozen_core =
      static_cast<std::size_t>(_settings->get<std::int64_t>("frozen_core"));
  const bool relax = _settings->get<std::string>("orbital_basis") == "relaxed";
  const bool symmetrize = _settings->get<bool>("symmetrize_two_body");

  const ctf12::F12HartreeFockInput input = ctf12::f12_input_from_wavefunction(
      *reference, gamma, cabs_basis, frozen_core);
  const ctf12::DressedHamiltonian dressed =
      ctf12::build_dressed_hamiltonian(input, relax);

  const std::size_t n = dressed.n_mo;
  const std::size_t nc = dressed.n_core;
  const std::size_t nact = n - nc;
  const Eigen::MatrixXd& h1 = dressed.one_body;
  std::vector<double> g = dressed.two_body;  // chemists' (pq|rs), flat n^4
  auto gidx = [&](std::size_t p, std::size_t q, std::size_t r, std::size_t s) {
    return ((p * n + q) * n + r) * n + s;
  };

  // Average the dressed two-body integrals onto full permutational symmetry for
  // solvers that assume it; the bare integrals carry only Hermitian symmetry.
  if (symmetrize) {
    std::vector<double> gs(g.size(), 0.0);
    for (std::size_t p = 0; p < n; ++p)
      for (std::size_t q = 0; q < n; ++q)
        for (std::size_t r = 0; r < n; ++r)
          for (std::size_t s = 0; s < n; ++s)
            gs[gidx(p, q, r, s)] =
                0.125 * (g[gidx(p, q, r, s)] + g[gidx(q, p, r, s)] +
                         g[gidx(p, q, s, r)] + g[gidx(q, p, s, r)] +
                         g[gidx(r, s, p, q)] + g[gidx(s, r, p, q)] +
                         g[gidx(r, s, q, p)] + g[gidx(s, r, q, p)]);
    g = std::move(gs);
  }

  // Dressed inactive (core) Fock over the full orbital space.
  Eigen::MatrixXd f_inactive = h1;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      double v = 0.0;
      for (std::size_t i = 0; i < nc; ++i)
        v += 2.0 * g[gidx(p, q, i, i)] - g[gidx(p, i, i, q)];
      f_inactive(static_cast<Eigen::Index>(p), static_cast<Eigen::Index>(q)) +=
          v;
    }

  // Active one-body integrals are the inactive Fock restricted to the active
  // orbitals; the active two-body integrals are the dressed (pq|rs) block.
  Eigen::MatrixXd h_active(nact, nact);
  for (std::size_t a = 0; a < nact; ++a)
    for (std::size_t b = 0; b < nact; ++b)
      h_active(static_cast<Eigen::Index>(a), static_cast<Eigen::Index>(b)) =
          f_inactive(static_cast<Eigen::Index>(nc + a),
                     static_cast<Eigen::Index>(nc + b));

  Eigen::VectorXd moeri(static_cast<Eigen::Index>(nact * nact * nact * nact));
  for (std::size_t a = 0; a < nact; ++a)
    for (std::size_t b = 0; b < nact; ++b)
      for (std::size_t c = 0; c < nact; ++c)
        for (std::size_t d = 0; d < nact; ++d)
          moeri(static_cast<Eigen::Index>(((a * nact + b) * nact + c) * nact +
                                          d)) =
              g[gidx(nc + a, nc + b, nc + c, nc + d)];

  double e_inactive = 0.0;
  for (std::size_t i = 0; i < nc; ++i)
    e_inactive +=
        h1(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) +
        f_inactive(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i));

  // Build the orbitals of the emitted Hamiltonian (relaxed F12-HF or reference
  // basis), carrying the frozen-core active space.
  auto reference_orbitals = reference->get_orbitals();
  auto basis_set = reference_orbitals->get_basis_set();
  std::optional<Eigen::MatrixXd> ao_overlap;
  if (reference_orbitals->has_overlap_matrix())
    ao_overlap = reference_orbitals->get_overlap_matrix();

  std::vector<std::size_t> active_indices, inactive_indices;
  for (std::size_t i = 0; i < nc; ++i) inactive_indices.push_back(i);
  for (std::size_t i = nc; i < n; ++i) active_indices.push_back(i);

  auto orbitals = std::make_shared<data::Orbitals>(
      dressed.mo_coefficients, std::make_optional(dressed.orbital_energies),
      ao_overlap, basis_set,
      std::make_optional(std::make_tuple(active_indices, inactive_indices)));

  const double core_energy =
      e_inactive +
      basis_set->get_structure()->calculate_nuclear_repulsion_energy();

  return std::make_shared<data::Hamiltonian>(
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          h_active, moeri, orbitals, core_energy, f_inactive));
}

}  // namespace qdk::chemistry::algorithms::microsoft

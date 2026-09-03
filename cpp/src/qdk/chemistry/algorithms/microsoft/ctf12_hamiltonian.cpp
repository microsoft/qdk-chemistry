// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ctf12_hamiltonian.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/structure.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <utility>
#include <vector>

namespace qdk::chemistry::algorithms::microsoft {

namespace {

using ctf12::F12HartreeFockInput;
using ctf12::F12HartreeFockResult;

// Dressed CT-F12 Hamiltonian in a molecular-orbital basis. The two-body
// integrals use the chemists' (pq|rs) convention with the flat layout
// ((p*n+q)*n+r)*n+s, matching data::CanonicalFourCenterHamiltonianContainer.
struct DressedHamiltonian {
  std::size_t n_mo = 0;
  std::size_t n_occupied = 0;
  std::size_t n_core = 0;
  Eigen::MatrixXd mo_coefficients;
  Eigen::VectorXd orbital_energies;
  Eigen::MatrixXd one_body;
  std::vector<double> two_body;
  double e_hf = 0.0;
  double e_f12hf = 0.0;
};

// Runs the F12-HF step and repackages its integrals in the chemists'
// convention, either in the relaxed F12-HF basis or the reference basis.
DressedHamiltonian build_dressed_hamiltonian(const F12HartreeFockInput& in,
                                             bool relax_orbitals) {
  QDK_LOG_TRACE_ENTERING();
  F12HartreeFockResult r =
      relax_orbitals ? ctf12::run_f12_hf(in) : ctf12::build_f12_hamiltonian(in);
  const std::size_t n = r.n_mo;

  DressedHamiltonian out;
  out.n_mo = n;
  out.n_occupied = r.n_occupied;
  out.n_core = r.n_core;
  out.e_hf = r.e_hf;
  out.e_f12hf = r.e_f12hf;

  std::vector<double> two_body_phys;  // dressed <pq|rs> in the chosen basis
  if (relax_orbitals) {
    const Eigen::MatrixXd& u = r.relaxation;  // original-MO -> relaxed-MO
    out.mo_coefficients = in.mo_coefficients * u;
    out.orbital_energies = r.relaxed_energies;
    out.one_body = u.transpose() * r.one_body * u;

    // Rotate the dressed <pq|rs> into the relaxed basis by four sequential
    // GEMM index transforms, cycling the leading index to the back each step.
    two_body_phys = std::move(r.two_body);
    const std::size_t n3 = n * n * n;
    using RowMajorMat =
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    for (int step = 0; step < 4; ++step) {
      Eigen::Map<RowMajorMat> m(two_body_phys.data(),
                                static_cast<Eigen::Index>(n),
                                static_cast<Eigen::Index>(n3));
      const RowMajorMat rotated = (u.transpose() * m).transpose();
      std::copy(rotated.data(), rotated.data() + n * n3, two_body_phys.begin());
    }
  } else {
    out.mo_coefficients = in.mo_coefficients;
    out.orbital_energies = in.orbital_energies;
    out.one_body = r.one_body;
    two_body_phys = std::move(r.two_body);
  }

  // Convert the dressed integrals from physicists' <pq|rs> to chemists'
  // (pq|rs) = <pr|qs>, matching the data::Hamiltonian two-body layout.
  out.two_body.assign(n * n * n * n, 0.0);
  auto flat = [&](std::size_t p, std::size_t q, std::size_t rr, std::size_t s) {
    return ((p * n + q) * n + rr) * n + s;
  };
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q)
      for (std::size_t rr = 0; rr < n; ++rr)
        for (std::size_t s = 0; s < n; ++s)
          out.two_body[flat(p, q, rr, s)] = two_body_phys[flat(p, rr, q, s)];

  return out;
}

std::shared_ptr<const data::AuxiliaryBasis> resolve_cabs(
    const std::shared_ptr<const data::AuxiliaryBasisCollection>&
        auxiliary_bases) {
  if (!auxiliary_bases ||
      !auxiliary_bases->has_auxiliary_basis(data::AuxiliaryBasisRole::CABS)) {
    throw std::invalid_argument(
        "CT-F12: run() requires an auxiliary-basis collection carrying a CABS "
        "entry; the complementary auxiliary basis is the external space the "
        "transformation folds in");
  }
  return auxiliary_bases->get_auxiliary_basis(data::AuxiliaryBasisRole::CABS);
}

}  // namespace

std::shared_ptr<data::Hamiltonian> CtF12HamiltonianConstructor::_run_impl(
    std::shared_ptr<data::Wavefunction> reference,
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<const data::SymmetryBlockedIndexSet> p_indices,
    std::shared_ptr<const data::AuxiliaryBasisCollection> auxiliary_bases)
    const {
  QDK_LOG_TRACE_ENTERING();

  _validate_inputs(reference, hamiltonian, p_indices);

  const auto active =
      data::spin_channel_indices(p_indices, data::axes::alpha());
  if (active != data::spin_channel_indices(p_indices, data::axes::beta())) {
    throw std::invalid_argument(
        "CT-F12 requires a closed-shell target space (equal alpha and beta "
        "P-space indices)");
  }
  if (active.empty()) {
    throw std::invalid_argument("CT-F12: the target P-space is empty");
  }
  if (active.back() >=
      reference->get_orbitals()->get_num_molecular_orbitals()) {
    throw std::invalid_argument(
        "CT-F12: the target P-space contains molecular-orbital indices outside "
        "the orbital basis");
  }

  const double gamma = _settings->get<double>("gamma");
  const auto frozen_core =
      static_cast<std::size_t>(_settings->get<std::int64_t>("frozen_core"));
  const bool relax = _settings->get<std::string>("orbital_basis") == "relaxed";
  const bool symmetrize = _settings->get<bool>("symmetrize_two_body");

  const ctf12::F12HartreeFockInput input = ctf12::f12_input_from_wavefunction(
      *reference, gamma, /*cabs_basis=*/"", frozen_core,
      resolve_cabs(auxiliary_bases));
  DressedHamiltonian dressed = build_dressed_hamiltonian(input, relax);

  const std::size_t n = dressed.n_mo;

  // Occupied orbitals left out of P are folded into the constant energy term
  // and the inactive Fock matrix; unoccupied ones are dropped.
  std::vector<std::size_t> inactive;
  for (std::size_t i = 0; i < dressed.n_occupied; ++i) {
    if (!std::binary_search(active.begin(), active.end(), i)) {
      inactive.push_back(i);
    }
  }

  const std::size_t nact = active.size();
  const Eigen::MatrixXd& h1 = dressed.one_body;
  std::vector<double> g = std::move(dressed.two_body);  // chemists' (pq|rs)
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

  // Dressed inactive Fock over the full orbital space.
  Eigen::MatrixXd f_inactive = h1;
  for (std::size_t p = 0; p < n; ++p)
    for (std::size_t q = 0; q < n; ++q) {
      double v = 0.0;
      for (const std::size_t i : inactive)
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
          f_inactive(static_cast<Eigen::Index>(active[a]),
                     static_cast<Eigen::Index>(active[b]));

  Eigen::VectorXd moeri(static_cast<Eigen::Index>(nact * nact * nact * nact));
  for (std::size_t a = 0; a < nact; ++a)
    for (std::size_t b = 0; b < nact; ++b)
      for (std::size_t c = 0; c < nact; ++c)
        for (std::size_t d = 0; d < nact; ++d)
          moeri(static_cast<Eigen::Index>(((a * nact + b) * nact + c) * nact +
                                          d)) =
              g[gidx(active[a], active[b], active[c], active[d])];

  double e_inactive = 0.0;
  for (const std::size_t i : inactive)
    e_inactive +=
        h1(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i)) +
        f_inactive(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(i));

  // Build the orbitals of the emitted Hamiltonian (relaxed F12-HF or reference
  // basis), carrying the target P-space as its active space.
  auto reference_orbitals = reference->get_orbitals();
  auto basis_set = reference_orbitals->get_basis_set();
  std::optional<Eigen::MatrixXd> ao_overlap;
  if (reference_orbitals->has_overlap_matrix())
    ao_overlap = reference_orbitals->get_overlap_matrix();

  auto orbitals = std::make_shared<data::Orbitals>(
      dressed.mo_coefficients, std::make_optional(dressed.orbital_energies),
      ao_overlap, basis_set, ctf12::restricted_index_set(n, active),
      ctf12::restricted_index_set(n, inactive));

  const double core_energy =
      e_inactive +
      basis_set->get_structure()->calculate_nuclear_repulsion_energy();

  return std::make_shared<data::Hamiltonian>(
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          h_active, moeri, orbitals, core_energy, f_inactive));
}

}  // namespace qdk::chemistry::algorithms::microsoft

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "gauge_fixing.hpp"

#include <algorithm>
#include <blas.hh>
#include <cmath>
#include <cstdint>
#include <lapack.hh>
#include <macis/util/transform.hpp>
#include <numbers>
#include <numeric>
#include <qdk/chemistry/algorithms/hamiltonian.hpp>
#include <qdk/chemistry/data/majorana_mapping.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/utils/golden_section.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <unordered_map>
#include <variant>

namespace qdk::chemistry::algorithms::microsoft {

std::shared_ptr<data::Wavefunction> GaugeFixingLocalizer::_run_impl(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& loc_indices_a,
    const std::vector<size_t>& loc_indices_b) const {
  QDK_LOG_TRACE_ENTERING();
  auto orbitals = wavefunction->get_orbitals();

  if (loc_indices_a != loc_indices_b) {
    throw std::invalid_argument(
        "loc_indices_a and loc_indices_b must be identical");
  }
  if (!std::is_sorted(loc_indices_a.begin(), loc_indices_a.end())) {
    throw std::invalid_argument("loc_indices_a must be sorted");
  }
  if (std::adjacent_find(loc_indices_a.begin(), loc_indices_a.end()) !=
      loc_indices_a.end()) {
    throw std::invalid_argument("loc_indices_a contains duplicate indices");
  }
  if (loc_indices_a.empty()) {
    return algorithms::detail::new_aufbau_determinant_wavefunction(wavefunction,
                                                                   orbitals);
  }
  if (!orbitals->is_restricted()) {
    throw std::invalid_argument(
        "GaugeFixingLocalizer requires restricted orbitals; run a natural "
        "orbital localizer first.");
  }
  if (!orbitals->has_overlap_matrix()) {
    throw std::invalid_argument(
        "GaugeFixingLocalizer requires an overlap matrix to be available in "
        "the orbitals.");
  }
  if (!orbitals->has_active_space()) {
    throw std::invalid_argument(
        "GaugeFixingLocalizer requires an active space to be defined in the "
        "orbitals.");
  }
  if (!wavefunction->has_one_rdm_spin_traced()) {
    throw std::invalid_argument(
        "GaugeFixingLocalizer requires an active-space 1-RDM in the "
        "wavefunction.");
  }

  const auto active_indices = data::spin_channel_indices(
      orbitals->active_indices(), data::axes::alpha());
  const auto& rdm_variant = wavefunction->get_active_one_rdm_spin_traced();
  const auto* one_rdm = std::get_if<Eigen::MatrixXd>(&rdm_variant);
  if (!one_rdm) {
    throw std::invalid_argument(
        "GaugeFixingLocalizer requires a real-valued active 1-RDM.");
  }
  if (static_cast<size_t>(one_rdm->rows()) != active_indices.size() ||
      static_cast<size_t>(one_rdm->cols()) != active_indices.size()) {
    throw std::invalid_argument(
        "1-RDM dimensions do not match the orbitals' active-space size.");
  }

  const double degeneracy_tolerance =
      _settings->get<double>("degeneracy_tolerance");
  const double improvement_tolerance =
      _settings->get<double>("improvement_tolerance");
  // Settings bounds compare against NaN, and every such comparison is false,
  // so a NaN tolerance passes them and then silently disables the test it
  // appears in. Reject the ones this algorithm branches on up front.
  if (!std::isfinite(degeneracy_tolerance) || degeneracy_tolerance <= 0.0) {
    throw std::invalid_argument(
        "degeneracy_tolerance must be finite and positive; otherwise the "
        "degenerate blocks are undefined.");
  }
  if (!std::isfinite(improvement_tolerance) || improvement_tolerance < 0.0) {
    throw std::invalid_argument(
        "improvement_tolerance must be finite and non-negative.");
  }
  const Eigen::VectorXd occupations = one_rdm->diagonal();
  // Off-diagonal weight above the degeneracy tolerance can reorder occupations
  // by more than the tolerance, which would make the blocks meaningless.
  if ((*one_rdm - Eigen::MatrixXd(occupations.asDiagonal()))
          .cwiseAbs()
          .maxCoeff() >= degeneracy_tolerance) {
    throw std::invalid_argument(
        "The active one-particle RDM is not diagonal in the input orbital "
        "basis, so occupation degeneracies are undefined. Run the "
        "'qdk_natural_orbitals' localizer before gauge fixing.");
  }

  // SymmetryBlockedIndexSet stores strictly increasing indices, so the active
  // set is already ordered for lookup.
  for (size_t index : loc_indices_a) {
    if (!std::binary_search(active_indices.begin(), active_indices.end(),
                            index)) {
      throw std::invalid_argument(
          "Every orbital to gauge fix must belong to the active space of the "
          "input wavefunction.");
    }
  }

  // Blocks partition the whole active space, but only wholly selected ones may
  // be rotated: anchoring a block whose members straddle the selection
  // boundary would change the selected subspace, and so the energy.
  std::vector<size_t> occupation_order(active_indices.size());
  std::iota(occupation_order.begin(), occupation_order.end(), size_t{0});
  std::stable_sort(occupation_order.begin(), occupation_order.end(),
                   [&occupations](size_t left, size_t right) {
                     return occupations[static_cast<Eigen::Index>(left)] >
                            occupations[static_cast<Eigen::Index>(right)];
                   });

  const std::vector<size_t> selected(loc_indices_a.begin(),
                                     loc_indices_a.end());
  std::vector<std::vector<size_t>> selected_blocks;
  size_t block_start = 0;
  for (size_t block_stop = 1; block_stop <= occupation_order.size();
       ++block_stop) {
    if (block_stop < occupation_order.size() &&
        std::abs(occupations[static_cast<Eigen::Index>(
                     occupation_order[block_stop])] -
                 occupations[static_cast<Eigen::Index>(
                     occupation_order[block_start])]) < degeneracy_tolerance) {
      continue;
    }
    std::vector<size_t> block;
    for (size_t position = block_start; position < block_stop; ++position) {
      block.push_back(active_indices[occupation_order[position]]);
    }
    std::sort(block.begin(), block.end());
    block_start = block_stop;

    const size_t members = static_cast<size_t>(
        std::count_if(block.begin(), block.end(), [&selected](size_t index) {
          return std::binary_search(selected.begin(), selected.end(), index);
        }));
    if (members != 0 && members != block.size()) {
      throw std::runtime_error(
          "The selected orbitals split an occupation-degenerate block");
    }
    if (members != 0) {
      selected_blocks.push_back(std::move(block));
    }
  }

  const Eigen::MatrixXd& overlap = orbitals->get_overlap_matrix();
  Eigen::MatrixXd coefficients = orbitals->coefficients()->block(
      {data::axes::alpha(), data::axes::alpha()});
  Eigen::MatrixXd input_active_coefficients(coefficients.rows(),
                                            active_indices.size());
  for (size_t i = 0; i < active_indices.size(); ++i) {
    input_active_coefficients.col(static_cast<Eigen::Index>(i)) =
        coefficients.col(static_cast<Eigen::Index>(active_indices[i]));
  }

  // Orient each selected block against the atomic orbitals it projects onto
  // most strongly, so that any two orientations of the same subspace land on
  // identical coordinates before the sweep begins.
  const int64_t num_atomic_orbitals = coefficients.rows();
  for (const auto& block : selected_blocks) {
    const auto block_size = static_cast<int64_t>(block.size());
    Eigen::MatrixXd block_coefficients(num_atomic_orbitals, block_size);
    for (size_t i = 0; i < block.size(); ++i) {
      block_coefficients.col(static_cast<Eigen::Index>(i)) =
          coefficients.col(static_cast<Eigen::Index>(block[i]));
    }

    Eigen::MatrixXd projected_ao(num_atomic_orbitals, block_size);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_atomic_orbitals, block_size, num_atomic_orbitals, 1.0,
               overlap.data(), num_atomic_orbitals, block_coefficients.data(),
               num_atomic_orbitals, 0.0, projected_ao.data(),
               num_atomic_orbitals);
    Eigen::MatrixXd residuals = projected_ao;

    Eigen::VectorXd anchor_vector(block_size);
    Eigen::VectorXd deflation_weights(num_atomic_orbitals);
    std::vector<int64_t> anchors;
    anchors.reserve(block.size());
    for (int64_t column = 0; column < block_size; ++column) {
      int64_t anchor = 0;
      double best = -1.0;
      for (int64_t row = 0; row < num_atomic_orbitals; ++row) {
        // Rows are strided in the column-major residual block.
        const double norm =
            blas::dot(block_size, residuals.data() + row, num_atomic_orbitals,
                      residuals.data() + row, num_atomic_orbitals);
        // Symmetry-equivalent atomic orbitals tie in residual norm, and
        // rounding resolves that tie differently for each orientation of the
        // same subspace. Requiring a relative margin far above that rounding,
        // and scanning in ascending index, makes the lowest atomic-orbital
        // index win every tie -- including the near-ties of a geometry that is
        // symmetric only to within its own convergence.
        if (norm > best * (1.0 + 1e-8)) {
          best = norm;
          anchor = row;
        }
      }
      anchors.push_back(anchor);

      blas::copy(block_size, residuals.data() + anchor, num_atomic_orbitals,
                 anchor_vector.data(), 1);
      const double anchor_norm =
          blas::nrm2(block_size, anchor_vector.data(), 1);
      if (anchor_norm <= std::numeric_limits<double>::epsilon()) {
        throw std::runtime_error(
            "Unable to find independent AO anchors for a degenerate orbital "
            "block");
      }
      blas::scal(block_size, 1.0 / anchor_norm, anchor_vector.data(), 1);
      blas::gemv(blas::Layout::ColMajor, blas::Op::NoTrans, num_atomic_orbitals,
                 block_size, 1.0, residuals.data(), num_atomic_orbitals,
                 anchor_vector.data(), 1, 0.0, deflation_weights.data(), 1);
      blas::ger(blas::Layout::ColMajor, num_atomic_orbitals, block_size, -1.0,
                deflation_weights.data(), 1, anchor_vector.data(), 1,
                residuals.data(), num_atomic_orbitals);
    }

    // Symmetric orthogonalization maps the i-th anchor onto the i-th returned
    // orbital, so the assignment must not depend on the order the anchors were
    // found in either.
    std::sort(anchors.begin(), anchors.end());

    Eigen::MatrixXd anchor_coefficients(block_size, block_size);
    for (size_t i = 0; i < anchors.size(); ++i) {
      blas::copy(
          block_size, projected_ao.data() + anchors[i], num_atomic_orbitals,
          anchor_coefficients.data() + static_cast<int64_t>(i) * block_size, 1);
    }

    Eigen::MatrixXd gram(block_size, block_size);
    blas::syrk(blas::Layout::ColMajor, blas::Uplo::Lower, blas::Op::Trans,
               block_size, block_size, 1.0, anchor_coefficients.data(),
               block_size, 0.0, gram.data(), block_size);
    Eigen::VectorXd eigenvalues(block_size);
    if (lapack::syev(lapack::Job::Vec, lapack::Uplo::Lower, block_size,
                     gram.data(), block_size, eigenvalues.data()) != 0) {
      throw std::runtime_error(
          "Failed to orthogonalize the AO anchors of a degenerate orbital "
          "block");
    }
    // The inverse square root below amplifies rounding in the eigenvectors by
    // the square root of the Gram condition number, so bound that condition
    // number rather than testing the smallest eigenvalue against an absolute
    // epsilon it would only fail on exact singularity.
    if (eigenvalues.minCoeff() <= 1e-10 * eigenvalues.maxCoeff()) {
      throw std::runtime_error(
          "AO anchors for a degenerate orbital block are linearly dependent");
    }

    Eigen::MatrixXd scaled_eigenvectors(block_size, block_size);
    for (int64_t j = 0; j < block_size; ++j) {
      blas::copy(block_size, gram.data() + j * block_size, 1,
                 scaled_eigenvectors.data() + j * block_size, 1);
      blas::scal(block_size, 1.0 / std::sqrt(eigenvalues[j]),
                 scaled_eigenvectors.data() + j * block_size, 1);
    }
    Eigen::MatrixXd inverse_sqrt(block_size, block_size);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::Trans,
               block_size, block_size, block_size, 1.0,
               scaled_eigenvectors.data(), block_size, gram.data(), block_size,
               0.0, inverse_sqrt.data(), block_size);
    Eigen::MatrixXd orthogonalizer(block_size, block_size);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               block_size, block_size, block_size, 1.0,
               anchor_coefficients.data(), block_size, inverse_sqrt.data(),
               block_size, 0.0, orthogonalizer.data(), block_size);

    Eigen::MatrixXd anchored(num_atomic_orbitals, block_size);
    blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
               num_atomic_orbitals, block_size, block_size, 1.0,
               block_coefficients.data(), num_atomic_orbitals,
               orthogonalizer.data(), block_size, 0.0, anchored.data(),
               num_atomic_orbitals);
    for (size_t i = 0; i < block.size(); ++i) {
      coefficients.col(static_cast<Eigen::Index>(block[i])) =
          anchored.col(static_cast<Eigen::Index>(i));
    }
  }

  // Every linear fermion-to-qubit encoding maps Majorana monomials onto Pauli
  // words up to a unit-modulus phase, so all of them share one multiset of
  // |h_l| and hence one lambda. Jordan-Wigner is the cheapest to build.
  const data::MajoranaMapping mapping =
      data::MajoranaMapping::jordan_wigner(2 * active_indices.size());

  // The rotations only mix active orbitals, so the atomic-orbital integrals
  // and the inactive space are the same for every candidate. Build the
  // Hamiltonian once and carry each candidate as a rotation of its active
  // molecular-orbital integrals; the selection restricts which blocks are
  // swept, not what is minimized.
  auto reference_orbitals = std::make_shared<data::Orbitals>(
      coefficients, std::nullopt, overlap, orbitals->get_basis_set(),
      orbitals->active_indices(), orbitals->inactive_indices());
  const auto reference_hamiltonian =
      HamiltonianConstructorFactory::create()->run(reference_orbitals);
  const auto& [reference_one_body, reference_one_body_beta] =
      reference_hamiltonian->get_one_body_integrals();
  const auto& [reference_two_body, reference_two_body_aabb,
               reference_two_body_bbbb] =
      reference_hamiltonian->get_two_body_integrals();
  const auto num_active = static_cast<Eigen::Index>(active_indices.size());

  auto coefficient_norm = [&](const Eigen::MatrixXd& rotation) {
    const auto n = static_cast<size_t>(num_active);
    // Both transforms are symmetric in their index pairs, so the column-major
    // result is also the row-major layout the mapper expects.
    Eigen::MatrixXd one_body(num_active, num_active);
    macis::two_index_transform(n, n, reference_one_body.data(), n,
                               rotation.data(), n, one_body.data(), n);
    Eigen::VectorXd two_body(reference_two_body.size());
    macis::four_index_transform(n, n, reference_two_body.data(), n,
                                rotation.data(), n, two_body.data(), n);
    const auto mapped = data::majorana_map_hamiltonian(
        mapping, /*core_energy=*/0.0, one_body.data(), one_body.data(),
        two_body.data(), two_body.data(), two_body.data(), n,
        /*spin_symmetric=*/true,
        // Drop only what is numerically zero: lambda is a sum of absolute
        // values, so any real truncation biases the objective.
        /*threshold=*/1e-14, /*integral_threshold=*/1e-14);
    double norm = 0.0;
    for (const auto& coefficient : mapped.coefficients) {
      norm += std::abs(coefficient);
    }
    return norm;
  };

  // Position of each rotatable orbital within the active-space integrals.
  std::unordered_map<size_t, Eigen::Index> active_position;
  for (size_t i = 0; i < active_indices.size(); ++i) {
    active_position[active_indices[i]] = static_cast<Eigen::Index>(i);
  }

  Eigen::MatrixXd active_rotation =
      Eigen::MatrixXd::Identity(num_active, num_active);
  const double norm_before = coefficient_norm(active_rotation);
  double current_norm = norm_before;
  const auto angle_samples =
      static_cast<size_t>(_settings->get<int64_t>("angle_samples"));
  const double angle_step =
      std::numbers::pi / static_cast<double>(angle_samples);
  const auto max_sweeps =
      static_cast<size_t>(_settings->get<int64_t>("max_sweeps"));

  for (size_t sweep = 0; sweep < max_sweeps; ++sweep) {
    const double sweep_start_norm = current_norm;
    for (const auto& block : selected_blocks) {
      for (size_t i = 0; i + 1 < block.size(); ++i) {
        for (size_t j = i + 1; j < block.size(); ++j) {
          const Eigen::Index left = active_position.at(block[i]);
          const Eigen::Index right = active_position.at(block[j]);

          auto plane_norm = [&](double angle) {
            const double cosine = std::cos(angle);
            const double sine = std::sin(angle);
            Eigen::MatrixXd candidate = active_rotation;
            candidate.col(left) = cosine * active_rotation.col(left) +
                                  sine * active_rotation.col(right);
            candidate.col(right) = cosine * active_rotation.col(right) -
                                   sine * active_rotation.col(left);
            return coefficient_norm(candidate);
          };

          double best_angle = 0.0;
          double best_norm = std::numeric_limits<double>::infinity();
          for (size_t sample = 0; sample < angle_samples; ++sample) {
            const double angle = static_cast<double>(sample) * angle_step;
            const double norm = plane_norm(angle);
            if (norm < best_norm) {
              best_norm = norm;
              best_angle = angle;
            }
          }

          auto [refined_angle, refined_norm] = utils::golden_section_minimum(
              plane_norm, best_angle - angle_step, best_angle + angle_step);
          // The bracket can hold more than one cusp, so contraction is not
          // guaranteed to improve on the coarse scan that placed it.
          if (best_norm < refined_norm) {
            refined_angle = best_angle;
            refined_norm = best_norm;
          }
          if (refined_norm < current_norm - improvement_tolerance) {
            // Rotations by an angle and by that angle plus pi differ only in
            // orbital sign; canonicalizing onto [0, pi) keeps the accepted
            // gauge reproducible.
            double canonical_angle = std::fmod(refined_angle, std::numbers::pi);
            if (canonical_angle < 0.0) {
              canonical_angle += std::numbers::pi;
            }
            current_norm = plane_norm(canonical_angle);
            const double cosine = std::cos(canonical_angle);
            const double sine = std::sin(canonical_angle);
            const Eigen::VectorXd left_column = active_rotation.col(left);
            const Eigen::VectorXd right_column = active_rotation.col(right);
            active_rotation.col(left) =
                cosine * left_column + sine * right_column;
            active_rotation.col(right) =
                cosine * right_column - sine * left_column;
          }
        }
      }
    }

    if (current_norm >= sweep_start_norm - improvement_tolerance) {
      break;
    }
  }

  QDK_LOGGER().info(
      "Gauge fixing changed the mapped coefficient norm from {} to {} Hartree",
      norm_before, current_norm);

  Eigen::MatrixXd anchored_active(coefficients.rows(), num_active);
  for (size_t i = 0; i < active_indices.size(); ++i) {
    anchored_active.col(static_cast<Eigen::Index>(i)) =
        coefficients.col(static_cast<Eigen::Index>(active_indices[i]));
  }
  Eigen::MatrixXd active_coefficients(coefficients.rows(), num_active);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             coefficients.rows(), num_active, num_active, 1.0,
             anchored_active.data(), coefficients.rows(),
             active_rotation.data(), num_active, 0.0,
             active_coefficients.data(), coefficients.rows());
  for (size_t i = 0; i < active_indices.size(); ++i) {
    coefficients.col(static_cast<Eigen::Index>(active_indices[i])) =
        active_coefficients.col(static_cast<Eigen::Index>(i));
  }

  auto gauge_fixed_orbitals = std::make_shared<data::Orbitals>(
      coefficients,
      std::nullopt,  // rotations inside a correlated active space carry no
                     // unique one-electron energies
      overlap, orbitals->get_basis_set(), orbitals->active_indices(),
      orbitals->inactive_indices());

  // Rotations stay inside degenerate eigenspaces, where the spin-traced RDM
  // is a multiple of the identity, so it survives them unchanged. The
  // spin-resolved blocks are not, so they are rotated by the total
  // active-space rotation the anchoring and the sweeps together applied.
  Eigen::MatrixXd overlap_times_active(coefficients.rows(), num_active);
  blas::gemm(blas::Layout::ColMajor, blas::Op::NoTrans, blas::Op::NoTrans,
             coefficients.rows(), num_active, coefficients.rows(), 1.0,
             overlap.data(), coefficients.rows(), active_coefficients.data(),
             coefficients.rows(), 0.0, overlap_times_active.data(),
             coefficients.rows());
  Eigen::MatrixXd total_rotation(num_active, num_active);
  blas::gemm(blas::Layout::ColMajor, blas::Op::Trans, blas::Op::NoTrans,
             num_active, num_active, coefficients.rows(), 1.0,
             input_active_coefficients.data(), coefficients.rows(),
             overlap_times_active.data(), coefficients.rows(), 0.0,
             total_rotation.data(), num_active);

  // Rotating inside a degenerate block leaves its occupations unchanged only
  // to the extent that they are equal, so apply the total active-space
  // rotation to the density matrices rather than assuming it is a no-op. That
  // keeps the spin-traced matrix exactly equal to the sum of the spin-resolved
  // blocks, and bounds the drift in the occupations by the degeneracy
  // tolerance.
  const auto n = static_cast<size_t>(num_active);
  Eigen::MatrixXd rotated_one_rdm(num_active, num_active);
  macis::two_index_transform(n, n, one_rdm->data(), n, total_rotation.data(), n,
                             rotated_one_rdm.data(), n);

  std::shared_ptr<const data::SymmetryBlockedTensorVariant<2>> active_one_rdm;
  if (wavefunction->has_active_one_rdm()) {
    const auto* spin_resolved =
        std::get_if<data::SymmetryBlockedTensor<2, double>>(
            &wavefunction->active_one_rdm());
    if (spin_resolved) {
      const Eigen::MatrixXd input_alpha =
          spin_resolved->block({data::axes::alpha(), data::axes::alpha()});
      const Eigen::MatrixXd input_beta =
          spin_resolved->block({data::axes::beta(), data::axes::beta()});
      Eigen::MatrixXd alpha(num_active, num_active);
      Eigen::MatrixXd beta(num_active, num_active);
      macis::two_index_transform(n, n, input_alpha.data(), n,
                                 total_rotation.data(), n, alpha.data(), n);
      macis::two_index_transform(n, n, input_beta.data(), n,
                                 total_rotation.data(), n, beta.data(), n);
      active_one_rdm = data::make_spin_diagonal_rank2_sbt_variant(
          data::ContainerTypes::MatrixVariant(alpha),
          data::ContainerTypes::MatrixVariant(beta), false);
    }
  }
  return algorithms::detail::new_aufbau_determinant_wavefunction(
      wavefunction, gauge_fixed_orbitals,
      data::ContainerTypes::MatrixVariant(rotated_one_rdm), active_one_rdm);
}

}  // namespace qdk::chemistry::algorithms::microsoft

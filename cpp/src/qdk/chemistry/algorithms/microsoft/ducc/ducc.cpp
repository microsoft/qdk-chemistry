// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "ducc.hpp"

#include <btas/btas.h>
#include <btas/tensor.h>

#include <Eigen/Dense>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <qdk/chemistry/data/hamiltonian.hpp>
#include <qdk/chemistry/data/hamiltonian_containers/canonical_four_center.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/symmetry/spin_channel_indices.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp>
#include <qdk/chemistry/data/symmetry/symmetry_blocked_tensor.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <qdk/chemistry/data/wavefunction_containers/amplitude_container.hpp>
#include <qdk/chemistry/data/wavefunction_containers/state_vector.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <variant>
#include <vector>

#include "ducc_equations.inc"

namespace qdk::chemistry::algorithms::microsoft {

namespace {

btas::Tensor<double> _to_btas(const Eigen::MatrixXd& source) {
  const auto rows = static_cast<std::size_t>(source.rows());
  const auto cols = static_cast<std::size_t>(source.cols());
  btas::Tensor<double> result{btas::Range{rows, cols}};
  Eigen::Map<Eigen::MatrixXd>(result.data(), source.rows(), source.cols()) =
      source;
  return result;
}

btas::Tensor<double> _to_btas(const Eigen::VectorXd& source) {
  btas::Tensor<double> result{
      btas::Range{static_cast<std::size_t>(source.size())}};
  Eigen::Map<Eigen::VectorXd>(result.data(), source.size()) = source;
  return result;
}

btas::Tensor<double> _to_btas_eri(const Eigen::VectorXd& source,
                                  std::size_t nmo) {
  btas::Tensor<double> result{btas::Range{nmo, nmo, nmo, nmo}};
  Eigen::Map<Eigen::VectorXd>(result.data(), source.size()) = source;
  return result;
}

Eigen::MatrixXd to_eigen_matrix(const btas::Tensor<double>& source) {
  return Eigen::Map<const Eigen::MatrixXd>(
      source.data(), static_cast<Eigen::Index>(source.extent(0)),
      static_cast<Eigen::Index>(source.extent(1)));
}

Eigen::VectorXd to_eigen_vector(const btas::Tensor<double>& source) {
  return Eigen::Map<const Eigen::VectorXd>(
      source.data(), static_cast<Eigen::Index>(source.size()));
}

std::vector<std::uint32_t> _positions_in_window(
    const std::vector<std::size_t>& p_space,
    const std::vector<std::size_t>& window) {
  std::vector<std::uint32_t> positions;
  positions.reserve(p_space.size());
  for (const auto orbital : p_space) {
    const auto position =
        std::lower_bound(window.begin(), window.end(), orbital);
    if (position == window.end() || *position != orbital)
      throw std::invalid_argument(
          "ducc: P-space indices must lie in the Hamiltonian active window");
    positions.push_back(
        static_cast<std::uint32_t>(std::distance(window.begin(), position)));
  }
  return positions;
}

std::shared_ptr<const data::SymmetryBlockedIndexSet> _output_inactive_indices(
    const data::Orbitals& input, const data::Wavefunction& reference,
    const data::SymmetryBlockedIndexSet& p_space) {
  const auto active = input.active_indices();
  const auto inactive = input.inactive_indices();
  const auto p_space_a_view =
      p_space.indices(data::SymmetryLabel{data::axes::alpha()});
  const auto p_space_b_view =
      p_space.indices(data::SymmetryLabel{data::axes::beta()});
  const std::vector<std::size_t> p_space_a(p_space_a_view.begin(),
                                           p_space_a_view.end());
  const std::vector<std::size_t> p_space_b(p_space_b_view.begin(),
                                           p_space_b_view.end());
  const auto active_a = data::spin_channel_indices(active, data::axes::alpha());
  const auto active_b = data::spin_channel_indices(active, data::axes::beta());
  const auto [nocc_a, nocc_b] = reference.get_active_num_electrons();

  const auto classify = [](std::vector<std::size_t> result,
                           const std::vector<std::size_t>& window,
                           const std::vector<std::size_t>& target,
                           std::size_t nocc) {
    if (nocc > window.size())
      throw std::invalid_argument(
          "ducc: active electron count exceeds the active orbital window");
    for (std::size_t position = 0; position < nocc; ++position) {
      const auto orbital = window[position];
      if (!std::binary_search(target.begin(), target.end(), orbital))
        result.push_back(orbital);
    }
    std::sort(result.begin(), result.end());
    result.erase(std::unique(result.begin(), result.end()), result.end());
    return result;
  };

  const auto output_a =
      classify(data::spin_channel_indices(inactive, data::axes::alpha()),
               active_a, p_space_a, nocc_a);
  const auto output_b =
      classify(data::spin_channel_indices(inactive, data::axes::beta()),
               active_b, p_space_b, nocc_b);
  std::unordered_map<data::SymmetryLabel, std::vector<std::uint32_t>> indices;
  indices[data::axes::alpha()] =
      std::vector<std::uint32_t>(output_a.begin(), output_a.end());
  indices[data::axes::beta()] =
      std::vector<std::uint32_t>(output_b.begin(), output_b.end());
  return std::make_shared<const data::SymmetryBlockedIndexSet>(
      input.symmetries(), input.mo_extents(), std::move(indices));
}

std::shared_ptr<data::Orbitals> _output_orbitals(
    const data::Orbitals& input, const data::Wavefunction& reference,
    std::shared_ptr<const data::SymmetryBlockedIndexSet> p_space) {
  // Bind the output to P-space. Restricted inputs become explicit alpha/beta
  // blocks because the transformed two-body tensor is spin-blocked.
  const std::optional<Eigen::MatrixXd> overlap =
      input.has_overlap_matrix()
          ? std::optional<Eigen::MatrixXd>(input.get_overlap_matrix())
          : std::nullopt;
  const auto energies = input.has_energies() ? input.energies() : nullptr;
  auto inactive = _output_inactive_indices(input, reference, *p_space);

  if (input.is_unrestricted()) {
    return std::make_shared<data::Orbitals>(
        input.coefficients(), energies, overlap, input.get_basis_set(),
        std::move(p_space), std::move(inactive));
  }

  const auto& coefficients =
      input.coefficients()->block({data::axes::alpha(), data::axes::alpha()});
  const std::optional<Eigen::VectorXd> energy =
      energies ? std::optional<Eigen::VectorXd>(
                     energies->block({data::axes::alpha()}))
               : std::nullopt;
  return std::make_shared<data::Orbitals>(
      coefficients, coefficients, energy, energy, overlap,
      input.get_basis_set(), std::move(p_space), std::move(inactive));
}

std::shared_ptr<data::Hamiltonian> _run_ducc(
    const data::Hamiltonian& hamiltonian,
    const data::Wavefunction& wavefunction, std::int64_t level,
    const std::vector<std::uint32_t>& p_space_a,
    const std::vector<std::uint32_t>& p_space_b,
    std::shared_ptr<data::Orbitals> orbitals) {
  // Convert spin blocks from symmetry-blocked Eigen storage to the dense BTAS
  // layout used by the generated contractions.
  const auto [h_a, h_b] = hamiltonian.get_one_body_integrals();
  const auto nmo = static_cast<std::size_t>(h_a.rows());
  const auto [v_aa, v_ab, v_bb] = hamiltonian.get_two_body_integrals();

  auto h_a_btas = _to_btas(h_a);
  auto h_b_btas = _to_btas(h_b);
  auto v_aa_btas = _to_btas_eri(v_aa, nmo);
  auto v_ab_btas = _to_btas_eri(v_ab, nmo);
  auto v_bb_btas = _to_btas_eri(v_bb, nmo);

  const auto& amplitudes =
      wavefunction.get_container<data::AmplitudeContainer>();
  const auto [t1_a, t1_b] = amplitudes.get_t1_amplitudes();
  const auto [t2_ab, t2_aa, t2_bb] = amplitudes.get_t2_amplitudes();
  // _run_impl has already established that every amplitude block is real.
  auto t1_a_btas = _to_btas(std::get<Eigen::VectorXd>(t1_a));
  auto t1_b_btas = _to_btas(std::get<Eigen::VectorXd>(t1_b));
  auto t2_ab_btas = _to_btas(std::get<Eigen::VectorXd>(t2_ab));
  auto t2_aa_btas = _to_btas(std::get<Eigen::VectorXd>(t2_aa));
  auto t2_bb_btas = _to_btas(std::get<Eigen::VectorXd>(t2_bb));

  const auto [nocc_a, nocc_b] = wavefunction.get_active_num_electrons();
  btas::Tensor<double> output_one_a;
  btas::Tensor<double> output_one_b;
  btas::Tensor<double> output_two_aa;
  btas::Tensor<double> output_two_ab;
  btas::Tensor<double> output_two_bb;
  double output_scalar = 0.0;

  // The kernels form sigma = T_ext - T_ext^dagger by removing amplitudes whose
  // indices are all in P-space. With F denoting the reference Fock operator:
  //   L0 = H
  //   L1 = H + [H, sigma] + 1/2 [[F, sigma], sigma]
  //   L2 = H + [H, sigma] + 1/2 [[H, sigma], sigma]
  //        + 1/6 [[[F, sigma], sigma], sigma].
  const auto evaluate = [&](auto evaluator) {
    evaluator(h_a_btas, h_b_btas, v_aa_btas, v_ab_btas, v_bb_btas, t1_a_btas,
              t1_b_btas, t2_ab_btas, t2_aa_btas, t2_bb_btas, p_space_a,
              p_space_b, nocc_a, nocc_b, hamiltonian.is_restricted(),
              hamiltonian.get_core_energy(), output_one_a, output_one_b,
              output_two_aa, output_two_ab, output_two_bb, output_scalar);
  };
  switch (level) {
    case 0:
      evaluate(evaluate_ducc_L0);
      break;
    case 1:
      evaluate(evaluate_ducc_L1);
      break;
    case 2:
      evaluate(evaluate_ducc_L2);
      break;
    default:
      throw std::runtime_error("ducc: unsupported ducc_level " +
                               std::to_string(level));
  }

  // Store the P-space scalar and spin-resolved one- and two-body blocks in the
  // canonical container consumed by downstream algorithms.
  auto container =
      std::make_unique<data::CanonicalFourCenterHamiltonianContainer>(
          to_eigen_matrix(output_one_a), to_eigen_matrix(output_one_b),
          to_eigen_vector(output_two_aa), to_eigen_vector(output_two_ab),
          to_eigen_vector(output_two_bb), std::move(orbitals), output_scalar,
          Eigen::MatrixXd{}, Eigen::MatrixXd{});
  return std::make_shared<data::Hamiltonian>(std::move(container));
}

}  // namespace

std::shared_ptr<data::Hamiltonian> DuccSolver::_run_impl(
    std::shared_ptr<data::Wavefunction> reference,
    std::shared_ptr<data::Hamiltonian> hamiltonian,
    std::shared_ptr<const data::SymmetryBlockedIndexSet> p_space) const {
  QDK_LOG_TRACE_ENTERING();
  _validate_inputs(reference, hamiltonian, p_space);

  if (!hamiltonian->is_hermitian())
    throw std::invalid_argument("ducc: input Hamiltonian must be Hermitian");

  const auto orbitals = hamiltonian->get_orbitals();
  const auto reference_orbitals = reference->get_orbitals();
  const auto active_a = data::spin_channel_indices(orbitals->active_indices(),
                                                   data::axes::alpha());
  const auto active_b = data::spin_channel_indices(orbitals->active_indices(),
                                                   data::axes::beta());
  if (active_a !=
          data::spin_channel_indices(reference_orbitals->active_indices(),
                                     data::axes::alpha()) ||
      active_b != data::spin_channel_indices(
                      reference_orbitals->active_indices(), data::axes::beta()))
    throw std::invalid_argument(
        "ducc: Hamiltonian and wavefunction must have the same active orbital "
        "window");

  const auto p_space_a =
      data::spin_channel_indices(p_space, data::axes::alpha());
  const auto p_space_b =
      data::spin_channel_indices(p_space, data::axes::beta());
  if (p_space_a.empty() || p_space_b.empty())
    throw std::invalid_argument(
        "ducc: P-space must be non-empty in each spin channel");
  const auto p_space_positions_a = _positions_in_window(p_space_a, active_a);
  const auto p_space_positions_b = _positions_in_window(p_space_b, active_b);

  // The generated equations require full-space integrals and real amplitudes.
  if (hamiltonian->has_inactive_fock_matrix())
    throw std::runtime_error(
        "ducc: input Hamiltonian must span the full orbital space");

  if (!reference->has_container_type<data::AmplitudeContainer>())
    throw std::invalid_argument(
        "ducc: reference must contain coupled-cluster amplitudes");
  const auto& amplitudes = reference->get_container<data::AmplitudeContainer>();
  if (amplitudes.get_amplitude_type() != data::AmplitudeType::CoupledCluster)
    throw std::invalid_argument(
        "ducc: reference amplitudes must have coupled-cluster type");
  if (amplitudes.is_complex())
    throw std::runtime_error("ducc: complex amplitudes not yet implemented");

  const auto amplitude_reference = amplitudes.get_wavefunction();
  if (!amplitude_reference->has_container_type<data::StateVectorContainer>())
    throw std::invalid_argument(
        "ducc: coupled-cluster reference must be a single determinant");
  const auto& determinants = amplitude_reference->get_active_determinants();
  if (determinants.size() != 1)
    throw std::invalid_argument(
        "ducc: coupled-cluster reference must be a single determinant");

  const auto& determinant = determinants.front();
  const auto [nocc_a, nocc_b] = reference->get_active_num_electrons();
  const auto expected_reference =
      data::Configuration::canonical_hf_configuration(nocc_a, nocc_b,
                                                      active_a.size());
  if (determinant != expected_reference)
    throw std::invalid_argument(
        "ducc: occupied orbitals must be contiguous from index zero in each "
        "spin channel");

  auto output_orbitals = _output_orbitals(*orbitals, *reference, p_space);
  return _run_ducc(
      *hamiltonian, *reference, _settings->get<std::int64_t>("ducc_level"),
      p_space_positions_a, p_space_positions_b, std::move(output_orbitals));
}

}  // namespace qdk::chemistry::algorithms::microsoft

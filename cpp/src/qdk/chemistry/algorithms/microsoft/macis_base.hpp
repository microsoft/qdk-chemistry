// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Core>
#include <macis/asci/determinant_search.hpp>
#include <macis/csr_hamiltonian.hpp>
#include <macis/mcscf/mcscf.hpp>
#include <qdk/chemistry/algorithms/mc.hpp>
#include <qdk/chemistry/data/wavefunction_containers/cas.hpp>
#include <qdk/chemistry/data/wavefunction_containers/sci.hpp>
#include <qdk/chemistry/utils/logger.hpp>
#include <set>
#include <sparsexx/io/write_mm.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @class MacisSettings
 * @brief Base settings for all MACIS-backed multi-configuration calculations.
 *
 * Extends MultiConfigurationSettings by registering MACIS MCSCFSettings
 * defaults. All MACIS calculator settings classes (CAS, ASCI, PMC) should
 * derive from this so that the solver settings are automatically available and
 * auto-propagate if MACIS changes its defaults.
 */
class MacisSettings : public MultiConfigurationSettings {
 public:
  MacisSettings() {
    macis::MCSCFSettings defaults;
    set_default<double>("ci_matel_tol", defaults.ci_matel_tol,
                        "Hamiltonian matrix element sparsification threshold",
                        data::BoundConstraint<double>{0.0, 1.0});
    set_default<std::string>(
        "ci_matrix_file", "",
        "If non-empty, write the CI Hamiltonian matrix to this path in Matrix "
        "Market format");
  }
  ~MacisSettings() override = default;
};

/** @brief Build a MACIS MCSCFSettings struct from generic settings.
 *
 * Maps QDK setting names (`ci_residual_tolerance`, `max_solver_iterations`,
 * `ci_matel_tol`) to the corresponding MACIS struct fields.
 *
 * @param settings_ Source settings.
 * @return Populated `macis::MCSCFSettings`.
 */
macis::MCSCFSettings get_mcscf_settings_(const data::Settings& settings_);

/** @brief Populate a MACIS ASCISettings struct from QDK settings.
 *
 * Each supported ASCI parameter is fetched with `get_or_default` so library
 * defaults remain when a key is absent.
 *
 * @param settings_ Source settings.
 * @return Filled `macis::ASCISettings` instance.
 */
macis::ASCISettings get_asci_settings_(const data::Settings& settings_);

/** @brief Optionally export the CI Hamiltonian matrix in Matrix Market format.
 *
 * If the setting `ci_matrix_file` is non-empty, constructs the CSR
 * Hamiltonian from the given determinant set and generator, then writes
 * it to the specified file path using 1-based indexing.
 *
 * @tparam Generator Hamiltonian generator type.
 * @tparam WfnIterator Iterator over determinant container.
 * @param settings Source settings (checked for `ci_matrix_file`).
 * @param dets_begin Start of determinant range.
 * @param dets_end End of determinant range.
 * @param ham_gen Prepared Hamiltonian generator.
 * @param ci_matel_tol Sparsification threshold for matrix elements.
 */
template <typename Generator, typename WfnIterator>
void maybe_export_ci_matrix(const data::Settings& settings,
                            WfnIterator dets_begin, WfnIterator dets_end,
                            Generator& ham_gen, double ci_matel_tol) {
  auto filename = settings.get<std::string>("ci_matrix_file");
  if (filename.empty()) return;

  QDK_LOGGER().info("Exporting CI Hamiltonian matrix to '{}'", filename);
  auto H_csr = macis::make_csr_hamiltonian<int64_t>(dets_begin, dets_end,
                                                    ham_gen, ci_matel_tol);
  sparsexx::write_mm(filename, H_csr, false, 1);
  QDK_LOGGER().info("Wrote {}x{} CI matrix ({} non-zeros) to '{}'", H_csr.m(),
                    H_csr.n(), H_csr.nnz(), filename);
}

/** @brief Dispatch calculation to an implementation specialized by wavefunction
 * bitset size.
 *
 * Selects an instantiation of `Func::impl<N>` where `N` corresponds to an
 * internal bitset size sufficient to represent the given number of active
 * orbitals. Current mapping:
 *  - `norb < 32`  -> `N = 64`
 *  - `norb < 64`  -> `N = 128`
 *  - `norb < 128` -> `N = 256`
 *  - `norb < 256` -> `N = 512`
 *  - `norb < 512` -> `N = 1024`
 *  - `norb < 1024` -> `N = 2048`
 *  - `norb <= 2048` -> `N = 4096`
 *
 * @tparam Func Helper struct exposing `template <size_t N> static return_type
 * impl(...)`.
 * @tparam Args Forwarded argument pack to underlying implementation.
 * @param args Argument pack to forward.
 * @param norb Number of active orbitals.
 * @return Return value of the selected `Func::impl<N>`.
 * @throws std::runtime_error if `norb > 2048` (unsupported size).
 */
template <typename Func, typename... Args>
auto dispatch_by_norb(size_t norb, Args&&... args) {
  if (norb < 32) {
    return Func::template impl<64>(std::forward<Args>(args)...);
  } else if (norb < 64) {
    return Func::template impl<128>(std::forward<Args>(args)...);
  } else if (norb < 128) {
    return Func::template impl<256>(std::forward<Args>(args)...);
  } else if (norb < 256) {
    return Func::template impl<512>(std::forward<Args>(args)...);
  } else if (norb < 512) {
    return Func::template impl<1024>(std::forward<Args>(args)...);
  } else if (norb < 1024) {
    return Func::template impl<2048>(std::forward<Args>(args)...);
  } else if (norb <= 2048) {
    return Func::template impl<4096>(std::forward<Args>(args)...);
  } else {
    throw std::runtime_error(
        "Function not implemented for more than 2048 orbitals");
  }
}

/** @brief Unified wavefunction builder.
 *
 * Creates a `data::Wavefunction` wrapping a concrete Container instance with
 * optional spin-traced and spin-dependent reduced density matrices (RDMs)
 * determined by flags stored in `settings`:
 *  - `calculate_one_rdm`
 *  - `calculate_two_rdm`
 *  - `calculate_single_orbital_entropies`
 *  - `calculate_mutual_information`
 *
 * The container type must provide:
 *  - `MatrixVariant`, `VectorVariant` typedefs
 *  - Constructors accepting:
 *    (coeffs, dets, orbitals)
 *    (coeffs, dets, orbitals, one_rdm_spin_traced, two_rdm_spin_traced)
 *    (coeffs, dets, orbitals, one_rdm_spin_traced, one_rdm_aa, one_rdm_bb,
 *       two_rdm_spin_traced, two_rdm_aabb, two_rdm_aaaa, two_rdm_bbbb,
 *       single_orbital_entropies, mutual_information, type)
 *
 * @tparam Container Concrete wavefunction container class.
 * @tparam Generator Hamiltonian generator type.
 * @tparam WfnContainer Determinant storage container type.
 * @param settings Settings controlling RDM computation.
 * @param hamiltonian Hamiltonian providing orbitals.
 * @param ham_gen Prepared generator for the active space.
 * @param nmo Number of active orbitals.
 * @param coeffs CI coefficient array (real values assumed).
 * @param dets Determinant collection.
 * @return A fully constructed `data::Wavefunction`.
 * @note Skips all RDM work if no flags are set.
 */
template <typename Container, typename Generator, typename WfnContainer>
inline data::Wavefunction build_wavefunction(
    const data::Settings& settings, const data::Hamiltonian& hamiltonian,
    Generator& ham_gen, size_t nmo, std::vector<double>& coeffs,
    WfnContainer& dets) {
  using MV = typename Container::MatrixVariant;
  using VV = typename Container::VectorVariant;

  // General Wavefunction construction
  Eigen::VectorXd C_vector(coeffs.size());
  std::copy(coeffs.begin(), coeffs.end(), C_vector.data());
  std::vector<data::Configuration> dets_configs;
  dets_configs.reserve(dets.size());
  for (auto det : dets) dets_configs.emplace_back(det, nmo);

  const bool eval_one_rdm = settings.get<bool>("calculate_one_rdm");
  const bool eval_two_rdm = settings.get<bool>("calculate_two_rdm");
  const bool eval_s1 = settings.get<bool>("calculate_single_orbital_entropies");
  const bool eval_mutual_info =
      settings.get<bool>("calculate_mutual_information");
  const bool eval_s2 = settings.get<bool>("calculate_two_orbital_entropies");

  std::optional<Eigen::MatrixXd> one_aa, one_bb;
  std::optional<Eigen::VectorXd> two_aabb, two_aaaa, two_bbbb;

  // evaluate spin-dependent RDMs
  if (eval_one_rdm || eval_two_rdm) {
    std::vector<double> active_one_aa(eval_one_rdm ? nmo * nmo : 0, 0.0);
    std::vector<double> active_one_bb(eval_one_rdm ? nmo * nmo : 0, 0.0);
    std::vector<double> active_two_aaaa(
        eval_two_rdm ? nmo * nmo * nmo * nmo : 0, 0.0);
    std::vector<double> active_two_bbbb(
        eval_two_rdm ? nmo * nmo * nmo * nmo : 0, 0.0);
    std::vector<double> active_two_aabb(
        eval_two_rdm ? nmo * nmo * nmo * nmo : 0, 0.0);

    ham_gen.form_rdms_spin_dep(
        dets.begin(), dets.end(), dets.begin(), dets.end(), coeffs.data(),
        macis::matrix_span<double>(
            eval_one_rdm ? active_one_aa.data() : nullptr, nmo, nmo),
        macis::matrix_span<double>(
            eval_one_rdm ? active_one_bb.data() : nullptr, nmo, nmo),
        macis::rank4_span<double>(
            eval_two_rdm ? active_two_aaaa.data() : nullptr, nmo, nmo, nmo,
            nmo),
        macis::rank4_span<double>(
            eval_two_rdm ? active_two_bbbb.data() : nullptr, nmo, nmo, nmo,
            nmo),
        macis::rank4_span<double>(
            eval_two_rdm ? active_two_aabb.data() : nullptr, nmo, nmo, nmo,
            nmo));

    if (eval_one_rdm) {
      one_aa = Eigen::Map<Eigen::MatrixXd>(active_one_aa.data(), nmo, nmo);
      one_bb = Eigen::Map<Eigen::MatrixXd>(active_one_bb.data(), nmo, nmo);
    }
    if (eval_two_rdm) {
      two_aaaa = Eigen::Map<Eigen::VectorXd>(active_two_aaaa.data(),
                                             nmo * nmo * nmo * nmo) *
                 2.0;
      two_bbbb = Eigen::Map<Eigen::VectorXd>(active_two_bbbb.data(),
                                             nmo * nmo * nmo * nmo) *
                 2.0;
      two_aabb = Eigen::Map<Eigen::VectorXd>(active_two_aabb.data(),
                                             nmo * nmo * nmo * nmo) *
                 2.0;
    }
  }

  // evaluate single orbital entropies, two-orbital entropies, and mutual
  // information
  data::OrbitalEntropies computed_entropies;
  if (eval_s1 || eval_s2 || eval_mutual_info) {
    std::vector<double> s1_vec(nmo, 0.0);
    std::vector<double> s2_data(eval_s2 ? nmo * nmo : 0, 0.0);
    std::vector<double> mi_data(eval_mutual_info ? nmo * nmo : 0, 0.0);

    ham_gen.form_entropies(
        dets.begin(), dets.end(), dets.begin(), dets.end(), coeffs.data(),
        s1_vec,
        macis::matrix_span<double>(eval_s2 ? s2_data.data() : nullptr, nmo,
                                   nmo),
        macis::matrix_span<double>(eval_mutual_info ? mi_data.data() : nullptr,
                                   nmo, nmo));

    if (eval_s1) {
      computed_entropies.single_orbital =
          Eigen::Map<Eigen::VectorXd>(s1_vec.data(), nmo);
    }
    if (eval_s2) {
      computed_entropies.two_orbital =
          Eigen::Map<Eigen::MatrixXd>(s2_data.data(), nmo, nmo);
    }
    if (eval_mutual_info) {
      computed_entropies.mutual_information =
          Eigen::Map<Eigen::MatrixXd>(mi_data.data(), nmo, nmo);
    }
  }

  // helper function to convert optional Eigen types to MatrixVariant
  auto to_mv =
      [](const std::optional<Eigen::MatrixXd>& m) -> std::optional<MV> {
    if (!m) return std::nullopt;
    return MV(*m);
  };

  // helper function to convert optional Eigen types to VectorVariant
  auto to_vv =
      [](const std::optional<Eigen::VectorXd>& v) -> std::optional<VV> {
    if (!v) return std::nullopt;
    return VV(*v);
  };

  // build container with appropriate RDMs
  std::unique_ptr<data::WavefunctionContainer> container;
  if (one_aa || one_bb || two_aabb || two_aaaa || two_bbbb ||
      computed_entropies.has_any()) {
    container = std::make_unique<Container>(
        C_vector, dets_configs, hamiltonian.get_orbitals(), std::nullopt,
        to_mv(one_aa), to_mv(one_bb), std::nullopt, to_vv(two_aabb),
        to_vv(two_aaaa), to_vv(two_bbbb), computed_entropies,
        data::WavefunctionType::SelfDual);
  } else {
    container = std::make_unique<Container>(C_vector, dets_configs,
                                            hamiltonian.get_orbitals(),
                                            data::WavefunctionType::SelfDual);
  }
  return data::Wavefunction(std::move(container));
}

/**
 * @class Macis
 * @brief Many-body Adaptive Configuration Interaction Solver implementation
 *
 * The MACIS class provides a concrete implementation of the
 * MultiConfigurationCalculator interface using the MACIS (Many-body Adaptive
 * Configuration Interaction Solver) library. This solver performs configuration
 * interaction calculations on molecular systems with strong electron
 * correlation.
 *
 * Features:
 * - Complete Active Space Configuration Interaction (CASCI) calculations
 * - Adaptive Sampling Configuration Interaction (ASCI) calculations
 *
 * Typical usage:
 * ```
 * // Create a MACIS calculator
 * auto macis =
 * std::make_unique<qdk::chemistry::algorithms::microsoft::Macis>();
 *
 * // Configure settings if needed
 * macis->settings().set("parameter_name", parameter_value);
 *
 * // Perform the calculation
 * auto [energy, wavefunction] = macis->calculate(hamiltonian);
 * ```
 *
 * The calculation automatically adapts to the size of the active space and
 * selects the appropriate internal representation for the wavefunction.
 *
 * @note Currently supports up to 128 orbitals in the active space.
 *
 * @see qdk::chemistry::algorithms::MultiConfigurationCalculator
 * @see qdk::chemistry::data::Hamiltonian
 * @see qdk::chemistry::data::Wavefunction
 * @see qdk::chemistry::data::Settings
 */
class Macis : public qdk::chemistry::algorithms::MultiConfigurationCalculator {
 public:
  /**
   * @brief Default constructor
   *
   * Initializes a MACIS calculator with default settings.
   */
  Macis() { _settings = std::make_unique<MultiConfigurationSettings>(); };

  /**
   * @brief Virtual destructor
   */
  virtual ~Macis() noexcept override = default;

  virtual std::string name() const = 0;

 protected:
  /**
   * @brief Perform a configuration interaction calculation
   *
   * This method performs a configuration interaction calculation using the
   * MACIS library. It dispatches the calculation to the appropriate
   * implementation based on the number of orbitals in the active space.
   *
   * The method extracts the active space orbital indices and occupations from
   * the Hamiltonian, and performs either a CASCI or ASCI calculation based on
   * the settings provided.
   *
   * @param hamiltonian The Hamiltonian containing the molecular integrals and
   *                    orbital information for the calculation.
   * @param n_active_alpha_electrons The number of alpha electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   * @param n_active_beta_electrons The number of beta electrons in the
   * active space, inactive orbitals are assumed to be fully occupied.
   *
   * @return A pair containing the calculated energy and the resulting
   * wavefunction.
   *
   * @throws std::runtime_error if the number of orbitals exceeds 128
   *
   * @see qdk::chemistry::data::Hamiltonian
   * @see qdk::chemistry::data::Wavefunction
   */
  virtual std::pair<double, std::shared_ptr<data::Wavefunction>> _run_impl(
      std::shared_ptr<data::Hamiltonian> hamiltonian,
      unsigned int n_active_alpha_electrons,
      unsigned int n_active_beta_electrons) const override = 0;
};

}  // namespace qdk::chemistry::algorithms::microsoft

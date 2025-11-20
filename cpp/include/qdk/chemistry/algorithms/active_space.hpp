// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <functional>
#include <qdk/chemistry/algorithms/algorithm.hpp>
#include <qdk/chemistry/data/orbitals.hpp>
#include <qdk/chemistry/data/settings.hpp>
#include <qdk/chemistry/data/wavefunction.hpp>
#include <string>
#include <tuple>
#include <vector>

namespace qdk::chemistry::algorithms {

namespace detail {
/**
 * @brief Helper function to sort entropies and their corresponding orbital
 * indices in descending order.
 * @param wavefunction The wavefunction containing orbital entropies and
 * indices.
 * @param normalize_entropies If true, normalize entropies by the maximum
 * entropy value.
 * @return A pair containing a vector of sorted orbital indices and a vector
 * of sorted entropies.
 * @throws std::runtime_error if the wavefunction does not have single orbital
 * entropies.
 */
std::tuple<double, std::vector<size_t>, Eigen::VectorXd>
_sort_entropies_and_indices(std::shared_ptr<data::Wavefunction> wavefunction,
                            bool normalize_entropies);

/**
 * @brief Create a new Orbitals object with the specified active space
 * indices.
 * @param wavefunction The wavefunction containing the orbitals.
 * @param active_space_indices_a The active space orbital indices for alpha
 * electrons.
 * @param active_space_indices_b The active space orbital indices for beta
 * electrons (optional for restricted orbitals).
 * @return A new Orbitals object with the specified active space indices.
 */
std::shared_ptr<data::Orbitals> new_orbitals(
    std::shared_ptr<data::Wavefunction> wavefunction,
    const std::vector<size_t>& active_space_indices_a,
    const std::optional<std::vector<size_t>>& active_space_indices_b =
        std::nullopt);
/**
 * @brief Create a new Wavefunction object with updated orbitals.
 *
 * Generate a new Wavefunction by replacing the orbitals of the given
 * wavefunction with the provided new orbitals, keeping the identical
 * total, determinants.
 *
 * @param wavefunction The original wavefunction.
 * @param new_orbitals The new orbitals to set in the wavefunction.
 * @return A new Wavefunction object with the updated orbitals.
 */
std::shared_ptr<data::Wavefunction> new_wavefunction(
    std::shared_ptr<data::Wavefunction> wavefunction,
    std::shared_ptr<data::Orbitals> new_orbitals);

/**
 * @brief Helper function to determine inactive space indices for restricted
 * orbitals.
 * @param nelec Number of electrons
 * @param active_space_indices Active space orbital indices
 * @return Vector of inactive space orbital indices
 */
std::vector<size_t> _get_inactive_space_indices(
    size_t nelec, const std::vector<size_t>& active_space_indices);

/**
 * @brief Helper function to determine inactive space indices for
 * unrestricted orbitals.
 * @param nelec_a Number of alpha electrons
 * @param nelec_b Number of beta electrons
 * @param active_space_indices_a Alpha active space orbital indices
 * @param active_space_indices_b Beta active space orbital indices
 * @return Pair of vectors containing (alpha_inactive_space_indices,
 * beta_inactive_space_indices)
 */
std::pair<std::vector<size_t>, std::vector<size_t>> _get_inactive_space_indices(
    size_t nelec_a, size_t nelec_b,
    const std::vector<size_t>& active_space_indices_a,
    const std::vector<size_t>& active_space_indices_b);

}  // namespace detail

/**
 * @brief Abstract base class for selecting active spaces in quantum chemistry
 * calculations.
 *
 * The ActiveSpaceSelector class defines an interface for algorithms that
 * identify and select important orbitals to include in active space
 * calculations. Active space methods are crucial for reducing the
 * computational complexity of many-body calculations by focusing on the most
 * chemically relevant orbitals while treating the rest as either fully
 * occupied or empty.
 *
 * Different active space selection algorithms can be implemented by deriving
 * from this class and providing implementations for the pure virtual methods.
 * The class works with the Factory pattern through ActiveSpaceSelectorFactory
 * to allow runtime selection of different implementations.
 *
 * Typical usage:
 * @code
 * auto selector =
 *   qdk::chemistry::algorithms::ActiveSpaceSelectorFactory::create("algorithm_name");
 * selector->settings().set("parameter_name", value);
 * data::Orbitals orbitals_with_active_space =
 *   selector->run(wavefunction);
 * @endcode
 *
 * @see ActiveSpaceSelectorFactory for creating instances of active space
 * selectors
 * @see data::Wavefunction for the wavefunction data structure used as input
 * @see data::Settings for configuration parameters
 */
class ActiveSpaceSelector
    : public Algorithm<ActiveSpaceSelector, std::shared_ptr<data::Wavefunction>,
                       std::shared_ptr<data::Wavefunction>> {
 public:
  /**
   * @brief Default constructor for ActiveSpaceSelector.
   *
   * This constructor initializes the ActiveSpaceSelector object.
   */
  ActiveSpaceSelector() = default;

  /**
   * @brief Default destructor for ActiveSpaceSelector.
   *
   * This destructor cleans up any resources used by the ActiveSpaceSelector.
   */
  virtual ~ActiveSpaceSelector() = default;

  /**
   * @brief Selects the active space from the given orbitals.
   *
   *
   * \cond DOXYGEN_SUPRESS (Doxygen warning suppression for argument packs)
   * @param wavefunction The input wavefunction from which to select the
   * active space.
   * \endcond
   * @return Orbitals with active space data populated. Depending on the
   *         implementation, the returned orbitals may be a copy with only
   *         metadata added (e.g., occupation/valence selectors), or they may
   *         include transformations to the underlying coefficients/energies
   *         (e.g., AVAS may rotate/canonicalize orbitals). The input orbitals
   *         are not modified.
   * @throws std::runtime_error if the active space selection fails or if the
   *         input orbitals already have an active space defined.
   * @throws SettingsAreLocked if attempting to modify settings after
   *         run) is called
   * @note Settings are automatically locked when this method is called and
   *       cannot be modified during or after execution.
   * @note The specific criteria and side effects (such as unitary rotations)
   *       for selecting the active space may vary between implementations.
   * See the documentation for the specific active space selector being used
   *       (docstring).
   */
  using Algorithm::run;

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const = 0;

  /**
   * @brief Access the algorithm's type name
   *
   * @return The algorithm's type name
   */
  std::string type_name() const final { return "active_space_selector"; };

 protected:
  /**
   * @brief Implementation of active space selection
   *
   * This method contains the actual selection logic. It is automatically
   * called by run() after settings have been locked.
   *
   * @param wavefunction The input wavefunction
   * @return Wavefunction with selected/reduced active space
   */
  virtual std::shared_ptr<data::Wavefunction> _run_impl(
      std::shared_ptr<data::Wavefunction> wavefunction) const = 0;
};

/**
 * @brief Factory class for creating active space selector instances.
 *
 * This class provides a mechanism to create active space selector instances
 * based on a string key. It allows for easy extension and registration
 * of different active space selection implementations.
 */
struct ActiveSpaceSelectorFactory
    : public AlgorithmFactory<ActiveSpaceSelector, ActiveSpaceSelectorFactory> {
  static std::string algorithm_type_name() { return "active_space_selector"; }
  static void register_default_instances();
  static std::string default_algorithm_name() { return "qdk_autocas_eos"; }
};

}  // namespace qdk::chemistry::algorithms

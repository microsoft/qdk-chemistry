// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qdk/chemistry/data/settings.hpp"

namespace qdk::chemistry::algorithms {

/**
 * @brief Base class for algorithms
 *
 * This class automatically generates a run() method that locks settings
 * and delegates to _run_impl().
 *
 * @tparam Derived The derived algorithm class such as ScfSolver
 * @tparam ReturnType The return type of the algorithm's run() and _run_impl()
 * methods
 * @tparam Args Parameter pack containing the types of input arguments required
 * by the algorithm
 *
 * Usage:
 * @code
 * class ScfSolver : public Algorithm<ScfSolver,
 *                   std::pair<double, std::shared_ptr<Wavefunction>>,
 *                   std::shared_ptr<Structure>, int, int> {
 *  protected:
 *   std::pair<double, std::shared_ptr<Wavefunction>> _run_impl(
 *     std::shared_ptr<Structure> structure, int charge, int spin) const
 * override;
 * };
 * @endcode
 */
template <typename Derived, typename ReturnType, typename... Args>
class Algorithm {
 public:
  Algorithm() = default;
  virtual ~Algorithm() = default;

  /**
   * @brief Auto-generated run() method
   *
   * Automatically locks settings and delegates to _run_impl()
   *
   * @param args Arguments forwarded to _run_impl() - types specified by the
   * Args parameter pack
   * @return ReturnType The result from executing _run_impl()
   */
  virtual ReturnType run(Args... args) const {
    this->lock_settings();
    return this->_run_impl(std::forward<Args>(args)...);
  }

  /**
   * @brief Access the algorithm's settings
   *
   * @return Reference to the algorithm's Settings object
   */
  data::Settings& settings() { return *_settings; }

  /**
   * @brief Access the algorithm's name
   *
   * @return The algorithm's name
   */
  virtual std::string name() const = 0;

  /**
   * @brief Access the algorithm's aliases
   *
   * The name aliases for this algorithm, including the primary name.
   * By default, returns a vector containing only the primary name.
   *
   * @return A vector of name aliases
   */
  virtual std::vector<std::string> aliases() const { return {this->name()}; }

  /**
   * @brief Access the algorithm's type name
   *
   * @return The algorithm's type name
   */
  virtual std::string type_name() const = 0;

 protected:
  /**
   * @brief Lock settings before execution
   */
  void lock_settings() const { this->_settings->lock(); }

  /**
   * @brief Implementation method that derived classes must override
   *
   * @param args Input arguments - types specified by the Args parameter pack
   * @return ReturnType The computed result of the algorithm
   */
  virtual ReturnType _run_impl(Args... args) const = 0;

  /**
   * @brief The algorithm's settings, to be replaced by derived classes
   */
  std::unique_ptr<data::Settings> _settings =
      std::make_unique<data::Settings>();
};

/**
 * @brief Base class template for algorithm factories
 *
 * This class provides a generic factory pattern for creating algorithm
 * instances based on string keys. It allows for easy extension and registration
 * of different algorithm implementations.
 *
 * @tparam AlgorithmType The base algorithm type that the factory creates all
 *                       registered algorithms will have to inherit from this
 *                       class and have matching signature for their run()
 *                       methods.
 *
 * Usage:
 * @code
 * // Define a concrete factory for ScfSolver
 * struct ScfSolverFactory : public AlgorithmFactory<ScfSolver> {
 *   // Optional: override default_key() for custom default behavior
 * };
 *
 * // Register implementations
 * ScfSolverFactory::register_instance("hf", []() {
 *   return std::make_unique<HartreeFockSolver>();
 * });
 *
 * // Create instances
 * auto solver = ScfSolverFactory::create("hf");
 * @endcode
 */
template <typename BaseAlgorithmType, typename Derived>
class AlgorithmFactory {
 public:
  using return_type = std::unique_ptr<BaseAlgorithmType>;
  using functor_type = std::function<return_type(void)>;

  /**
   * @brief Create an algorithm instance.
   *
   * This method creates an algorithm based on the provided key.
   * If no key is provided or the key is empty, it defaults to the first
   * registered implementation.
   * To check any returned implementations, for its name use the name() method
   * of the created instance.
   * To list all available implementations, use the available() method of the
   * factory.
   *
   * @param name The name to identify the desired algorithm implementation.
   * @return A unique pointer to the created algorithm instance.
   * @throws std::runtime_error if the name is not found in the registry.
   */
  static return_type create(const std::string& name = "") {
    std::string key = name;
    if (key.empty()) {
      key = Derived::default_algorithm_name();
    }

    auto it = registry().find(key);
    if (it == registry().end()) {
      throw std::runtime_error(
          "Algorithm factory for " + Derived::algorithm_type_name() +
          ": Algorithm with name '" + key +
          "' not found in registry, available options are: " + ([]() {
            std::string available_keys;
            const auto& reg = registry();
            for (const auto& [k, _] : reg) {
              if (!available_keys.empty()) {
                available_keys += ", ";
              }
              available_keys += k;
            }
            return available_keys;
          })());
    }

    return it->second();
  }

  /**
   * @brief Register a new algorithm implementation.
   *
   * This method allows the registration of a new algorithm implementation
   * with a unique key. If the key already exists, an exception is thrown.
   * The algorithm is registered under its primary name and all its aliases.
   *
   * @param func The function that creates the algorithm instance.
   * @throws std::runtime_error if the key already exists or if type mismatch.
   */
  static void register_instance(functor_type func) {
    auto& reg = registry();
    auto tmp = func();

    // Verify type name matches
    if (tmp->type_name() != Derived::algorithm_type_name()) {
      throw std::runtime_error(
          "Algorithm factory for " + Derived::algorithm_type_name() +
          ": algorithm with name '" + tmp->name() +
          "' has incorrect algorithm type: " + tmp->type_name() +
          " expected is: " + Derived::algorithm_type_name());
    }

    // Get all aliases (includes primary name)
    auto aliases = tmp->aliases();

    // Check for name clashes first
    for (const auto& alias : aliases) {
      if (reg.find(alias) != reg.end()) {
        throw std::runtime_error("Algorithm factory for " +
                                 Derived::algorithm_type_name() +
                                 ": algorithm with name/alias '" + alias +
                                 "' already exists in registry");
      }
    }

    // Register under all aliases
    for (const auto& alias : aliases) {
      reg[alias] = func;
    }
  }

  /**
   * @brief Unregister an algorithm implementation.
   *
   * This method removes a previously registered algorithm implementation
   * from the registry.
   *
   * @param key The key identifying the algorithm implementation to remove.
   * @return true if the key was found and removed, false otherwise.
   */
  static bool unregister_instance(const std::string& key) {
    auto& reg = registry();
    auto it = reg.find(key);
    if (it != reg.end()) {
      reg.erase(it);
      return true;
    }
    return false;
  }

  /**
   * @brief Get a list of all available algorithm keys.
   *
   * This method returns a vector containing all registered keys for algorithm
   * implementations.
   *
   * @return A vector of strings representing the available algorithm keys.
   */
  static std::vector<std::string> available() {
    std::vector<std::string> keys;
    const auto& reg = registry();
    keys.reserve(reg.size());
    for (const auto& [key, _] : reg) {
      keys.push_back(key);
    }
    return keys;
  }

  /**
   * @brief Check if a key exists in the registry.
   *
   * @param key The key to check.
   * @return true if the key exists, false otherwise.
   */
  static bool has(const std::string& key) {
    return registry().find(key) != registry().end();
  }

  /**
   * @brief Clear all registered implementations.
   *
   * This method removes all entries from the registry. Use with caution.
   */
  static void clear() { registry().clear(); }

 protected:
  /**
   * @brief Get the factory registry.
   *
   * This method returns a reference to the static registry that stores
   * the mapping between keys and factory functions. Each instantiation
   * of the template has its own registry.
   *
   * @return Reference to the registry map.
   */
  static std::unordered_map<std::string, functor_type>& registry() {
    static std::unordered_map<std::string, functor_type> instance;
    static bool initialized = false;
    if (!initialized) {
      // Ensure at least one implementation is registered
      initialized = true;
      Derived::register_default_instances();
    }
    return instance;
  }
};

}  // namespace qdk::chemistry::algorithms

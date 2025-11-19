// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/exc/gauxc_impl.h>
#include <qdk/chemistry/scf/util/class_registry.h>
#include <qdk/chemistry/scf/util/gauxc_util.h>

#include <memory>

namespace qdk::chemistry::scf::util {

/**
 * @brief Specialized class registry for caching impl::GAUXC instances based on
 * GAUXCInput
 *
 * This class provides a way to create and manage a singleton cache of
 * impl::GAUXC instances, keyed by GAUXCInput objects. This allows efficient
 * reuse of GAUXC instances with the same configuration parameters.
 */
class GAUXCRegistry {
 public:
  /**
   * @brief Get or create an impl::GAUXC instance based on the provided
   * parameters
   *
   * @param basis_set The basis set to use for the GAUXC implementation
   * @param gauxc_input The GAUXCInput configuration
   * @param unrestricted Whether to use unrestricted calculations
   * @param xc_name The name of the exchange-correlation functional
   * @return impl::GAUXC* Pointer to the cached or newly created GAUXC instance
   */
  static impl::GAUXC* get_or_create(BasisSet& basis_set,
                                    const GAUXCInput& gauxc_input,
                                    bool unrestricted,
                                    const std::string& xc_name) {
    return ClassRegistry<impl::GAUXC, GAUXCInput>::get_or_create(
        gauxc_input, basis_set, gauxc_input, unrestricted, xc_name);
  }

  /**
   * @brief Find an impl::GAUXC instance in the cache by a GAUXCInput
   * configuration
   *
   * @param input The GAUXCInput configuration to look up in the cache
   * @return impl::GAUXC* Pointer to the cached GAUXC instance or nullptr if not
   * found
   */
  static impl::GAUXC* find(const GAUXCInput& input) {
    return ClassRegistry<impl::GAUXC, GAUXCInput>::find(input);
  }

  /**
   * @brief Remove an impl::GAUXC instance from the cache by its GAUXCInput
   * configuration
   *
   * @param input The GAUXCInput configuration of the instance to remove
   */
  static void remove(const GAUXCInput& input) {
    ClassRegistry<impl::GAUXC, GAUXCInput>::remove(input);
  }

  /**
   * @brief Clear all impl::GAUXC instances from the cache
   */
  static void clear() { ClassRegistry<impl::GAUXC, GAUXCInput>::clear(); }

  /**
   * @brief Get access to the underlying Cache instance for impl::GAUXC
   *
   * @return Cache<impl::GAUXC>& Reference to the singleton Cache instance
   */
  static Cache<impl::GAUXC>& cache() {
    return ClassRegistry<impl::GAUXC, GAUXCInput>::cache();
  }
};

}  // namespace qdk::chemistry::scf::util

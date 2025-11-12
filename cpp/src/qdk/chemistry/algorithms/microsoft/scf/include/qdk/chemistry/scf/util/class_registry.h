// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once
#include <qdk/chemistry/scf/util/cache.h>
#include <qdk/chemistry/scf/util/singleton.h>

#include <memory>
#include <utility>

namespace qdk::chemistry::scf::util {

/**
 * @brief A class registry that manages singleton instances of Cache<T> for
 * caching objects
 *
 * This class provides a way to create and access singleton instances of
 * Cache<T> for any type T. It leverages the Singleton utility to ensure only
 * one cache instance exists per type.
 *
 * @tparam T The type of objects to be cached
 * @tparam Identifier The type used as the cache key/identifier
 */
template <typename T, typename Identifier = std::size_t>
class ClassRegistry {
 public:
  /**
   * @brief Get the singleton cache instance
   *
   * @return Cache<T>& Reference to the singleton Cache<T> instance
   */
  static Cache<T>& cache() { return Singleton<Cache<T>>::instance(); }

  /**
   * @brief Find or create an object in the cache by its identifier
   *
   * @tparam Args Types of arguments to forward to the constructor of T if
   * creation is needed
   * @param id The identifier to look up in the cache
   * @param args Arguments to forward to the constructor of T if creation is
   * needed
   * @return T* Pointer to the cached or newly created object
   */
  template <typename... Args>
  static T* get_or_create(const Identifier& id, Args&&... args) {
    auto& cache_instance = cache();
    T* existing = cache_instance[id];

    if (!existing) {
      auto idx = std::hash<Identifier>{}(id);
      return cache_instance.emplace(idx, std::forward<Args>(args)...);
    }

    return existing;
  }

  /**
   * @brief Find an object in the cache by its identifier
   *
   * @param id The identifier to look up in the cache
   * @return T* Pointer to the cached object or nullptr if not found
   */
  static T* find(const Identifier& id) { return cache()[id]; }

  /**
   * @brief Remove an object from the cache by its identifier
   *
   * @param id The identifier of the object to remove
   */
  static void remove(const Identifier& id) {
    auto idx = std::hash<Identifier>{}(id);
    cache().erase(idx);
  }

  /**
   * @brief Clear all objects from the cache
   */
  static void clear() { cache().clear(); }
};

}  // namespace qdk::chemistry::scf::util

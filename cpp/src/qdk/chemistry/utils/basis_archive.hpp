// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <filesystem>
#include <optional>
#include <string>

namespace qdk::chemistry::utils {

/**
 * @brief A basis-set archive shipped with QDK/Chemistry, extracted into a
 * private temporary directory.
 *
 * The archive for a basis set is `compressed/<stem>.tar.gz` in the resources
 * directory. It holds `basis/<stem>.json` and, for most basis sets,
 * precomputed atomic guesses `guess/<stem>_pure` and `guess/<stem>_cart`.
 * Construction extracts the archive into a directory that is created
 * exclusively with owner-only access, and destruction removes it. Paths
 * returned by json() and guess() are valid only while the object is alive.
 */
class BasisArchive {
 public:
  /**
   * @brief Map a basis-set name to the stem of its archive.
   *
   * '*' becomes "_st_" and '/' becomes "_sl_"; every other character,
   * including '+', is kept, as in the shipped archive names. A stem maps to
   * itself.
   *
   * @param basis_name Basis-set name, e.g. "6-31g*"
   * @return Archive stem, e.g. "6-31g_st_"
   */
  static std::string stem(const std::string& basis_name);

  /**
   * @brief Check whether an archive is shipped for a basis set.
   * @param basis_name Basis-set name or archive stem
   * @return True if the archive for the name's stem exists
   */
  static bool exists(const std::string& basis_name);

  /**
   * @brief Extract the archive for a basis set.
   * @param basis_name Basis-set name or archive stem
   * @throws std::invalid_argument if no archive is shipped for the name
   * @throws std::system_error if the directory cannot be created or tar
   *         cannot be launched
   * @throws std::runtime_error if tar fails to extract the archive
   */
  explicit BasisArchive(const std::string& basis_name);

  /**
   * @brief Remove the extraction directory and its contents.
   */
  ~BasisArchive();

  /**
   * @brief Deleted: each object owns its extraction directory.
   */
  BasisArchive(const BasisArchive&) = delete;

  /**
   * @brief Deleted: each object owns its extraction directory.
   */
  BasisArchive& operator=(const BasisArchive&) = delete;

  /**
   * @brief Path of the extracted basis-set JSON file.
   * @return Path that is valid while this object is alive
   * @throws std::invalid_argument if the archive has no basis JSON file or it
   *         resolves outside the extraction directory
   */
  std::filesystem::path json() const;

  /**
   * @brief Path of the extracted precomputed atomic guess, if shipped.
   * @param pure True for spherical (pure) functions, false for Cartesian
   * @return Path that is valid while this object is alive, or std::nullopt if
   *         the archive ships no such guess
   * @throws std::invalid_argument if the guess file resolves outside the
   *         extraction directory
   */
  std::optional<std::filesystem::path> guess(bool pure) const;

 private:
  /// Archive stem of the extracted basis set.
  std::string _stem;

  /// Private extraction directory; the archive contents are in `contents/`.
  std::filesystem::path _path;
};

}  // namespace qdk::chemistry::utils

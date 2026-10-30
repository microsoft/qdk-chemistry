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
   * @brief Map a basis-set name to the stem of its archive, e.g. "6-31g*" to
   * "6-31g_st_".
   *
   * '*' becomes "_st_" and '/' becomes "_sl_"; every other character,
   * including '+', is kept, as in the shipped archive names. A stem maps to
   * itself.
   */
  static std::string stem(const std::string& basis_name);

  /// Whether an archive is shipped for the stem of a basis-set name.
  static bool exists(const std::string& basis_name);

  /**
   * @brief Extract the archive for a basis-set name or stem.
   * @throws std::invalid_argument if no archive is shipped for the name
   * @throws std::system_error if the directory cannot be created or tar
   *         cannot be launched
   * @throws std::runtime_error if tar fails to extract the archive
   */
  explicit BasisArchive(const std::string& basis_name);

  ~BasisArchive();

  BasisArchive(const BasisArchive&) = delete;
  BasisArchive& operator=(const BasisArchive&) = delete;

  /**
   * @brief Path of the extracted basis-set JSON file.
   * @throws std::invalid_argument if the archive has no basis JSON file
   */
  std::filesystem::path json() const;

  /**
   * @brief Path of the extracted precomputed atomic guess, or std::nullopt if
   * the archive ships none.
   * @param pure True for spherical (pure) functions, false for Cartesian
   */
  std::optional<std::filesystem::path> guess(bool pure) const;

 private:
  std::string _stem;
  std::filesystem::path _path;
};

}  // namespace qdk::chemistry::utils

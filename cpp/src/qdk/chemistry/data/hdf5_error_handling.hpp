// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>

namespace qdk::chemistry::data {

inline bool hdf5_errors_should_be_suppressed() {
  const char* env_value = std::getenv("QDK_PRINT_VERBOSE_HDF5_ERRORS");
  if (!env_value) {
    return true;
  }

  std::string normalized(env_value);
  std::transform(
      normalized.begin(), normalized.end(), normalized.begin(),
      [](unsigned char c) { return static_cast<char>(std::tolower(c)); });

  const bool verbose_requested = (normalized == "1" || normalized == "true" ||
                                  normalized == "yes" || normalized == "on");

  return !verbose_requested;
}

}  // namespace qdk::chemistry::data

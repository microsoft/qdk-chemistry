// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <stdexcept>

namespace qdk::chemistry {

/** Raised when a registry key is already owned by another implementation. */
class DuplicateRegistrationError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace qdk::chemistry

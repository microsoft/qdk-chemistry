// Copyright (c) Microsoft Corporation.

#include "common.hpp"

void throw_if_file_not_found(const std::string &filename) {
  if (!std::filesystem::exists(filename)) {
    throw std::runtime_error("File not found: " + filename);
  }
}

void validate_electron_counts(size_t nalpha, size_t nbeta, size_t norbital) {
  if (norbital == 0) {
    throw std::runtime_error("Number of orbitals must be greater than 0");
  }
  if (nalpha == 0 || nbeta == 0) {
    throw std::runtime_error(
        "Number of alpha and beta electrons must be greater than 0");
  }
  if (nalpha > norbital || nbeta > norbital) {
    throw std::runtime_error(
        "Total number of electrons exceeds number of orbitals");
  }
}

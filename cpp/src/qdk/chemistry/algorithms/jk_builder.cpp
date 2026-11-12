// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/jk_builder.hpp"

#include <qdk/chemistry/algorithms/jk_builder.hpp>
#include <stdexcept>

namespace qdk::chemistry::algorithms {

void JKEngine::build_gradients(const double*, size_t, size_t, size_t, double*,
                               double*, double, double, double) {
  throw std::runtime_error("This J/K engine does not build gradients");
}

std::unique_ptr<JKBuilder> make_microsoft_jk_builder() {
  return std::make_unique<microsoft::JKBuilder>();
}

void JKBuilderFactory::register_default_instances() {
  JKBuilderFactory::register_instance(&make_microsoft_jk_builder);
}

}  // namespace qdk::chemistry::algorithms

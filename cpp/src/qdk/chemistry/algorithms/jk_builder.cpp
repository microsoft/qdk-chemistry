// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "microsoft/jk_builder.hpp"

#include <qdk/chemistry/algorithms/jk_builder.hpp>

namespace qdk::chemistry::algorithms {

std::unique_ptr<JKBuilder> make_microsoft_jk_builder() {
  return std::make_unique<microsoft::JKBuilder>();
}

std::unique_ptr<JKGradientBuilder> make_microsoft_jk_gradient_builder() {
  return std::make_unique<microsoft::JKGradientBuilder>();
}

void JKBuilderFactory::register_default_instances() {
  JKBuilderFactory::register_instance(&make_microsoft_jk_builder);
}

void JKGradientBuilderFactory::register_default_instances() {
  JKGradientBuilderFactory::register_instance(
      &make_microsoft_jk_gradient_builder);
}

}  // namespace qdk::chemistry::algorithms

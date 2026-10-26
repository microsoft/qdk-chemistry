// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <macis/asci/execution_context.hpp>
#include <memory>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms::microsoft::asci_extension {

/** @brief The default adapter adds no execution-specific settings. */
template <typename DefineSetting>
void configure_settings(DefineSetting&&) {}

/** @brief Select the built-in CPU implementation when no addon is configured.
 */
template <size_t N>
std::unique_ptr<macis::ASCIExecutionContext<N>> make_context(
    const macis::HamiltonianGenerator<macis::wfn_t<N>>&,
    const data::Settings&) {
  return nullptr;
}

}  // namespace qdk::chemistry::algorithms::microsoft::asci_extension

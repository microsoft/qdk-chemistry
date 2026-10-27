// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <macis/asci/execution_context.hpp>
#include <memory>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms::microsoft {

/**
 * @brief Create the numerical executor for one ASCI solve.
 *
 * The library compiles one definition, from QDK_ASCI_EXECUTION_SOURCE. That
 * source also defines MacisAsciSettings::_configure_execution_settings() and
 * instantiates this function for every dispatch_by_norb width.
 *
 * @tparam N Determinant width in bits
 * @param ham_gen Hamiltonian generator of the solve
 * @param settings Settings of the solve
 * @return The executor, or nullptr to use MACIS's CPU implementation
 */
template <size_t N>
std::unique_ptr<macis::ASCIExecutionContext<N>> make_asci_execution(
    const macis::HamiltonianGenerator<macis::wfn_t<N>>& ham_gen,
    const data::Settings& settings);

}  // namespace qdk::chemistry::algorithms::microsoft

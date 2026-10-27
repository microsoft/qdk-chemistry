// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include "macis_asci_execution.hpp"

#include "macis_asci.hpp"

namespace qdk::chemistry::algorithms::microsoft {

void MacisAsciSettings::_configure_execution_settings() {}

template <size_t N>
std::unique_ptr<macis::ASCIExecutionContext<N>> make_asci_execution(
    const macis::HamiltonianGenerator<macis::wfn_t<N>>&,
    const data::Settings&) {
  return nullptr;
}

template std::unique_ptr<macis::ASCIExecutionContext<64>>
make_asci_execution<64>(const macis::HamiltonianGenerator<macis::wfn_t<64>>&,
                        const data::Settings&);
template std::unique_ptr<macis::ASCIExecutionContext<128>>
make_asci_execution<128>(const macis::HamiltonianGenerator<macis::wfn_t<128>>&,
                         const data::Settings&);
template std::unique_ptr<macis::ASCIExecutionContext<256>>
make_asci_execution<256>(const macis::HamiltonianGenerator<macis::wfn_t<256>>&,
                         const data::Settings&);
template std::unique_ptr<macis::ASCIExecutionContext<512>>
make_asci_execution<512>(const macis::HamiltonianGenerator<macis::wfn_t<512>>&,
                         const data::Settings&);
template std::unique_ptr<macis::ASCIExecutionContext<640>>
make_asci_execution<640>(const macis::HamiltonianGenerator<macis::wfn_t<640>>&,
                         const data::Settings&);
template std::unique_ptr<macis::ASCIExecutionContext<768>>
make_asci_execution<768>(const macis::HamiltonianGenerator<macis::wfn_t<768>>&,
                         const data::Settings&);
template std::unique_ptr<macis::ASCIExecutionContext<1024>> make_asci_execution<
    1024>(const macis::HamiltonianGenerator<macis::wfn_t<1024>>&,
          const data::Settings&);
template std::unique_ptr<macis::ASCIExecutionContext<2048>> make_asci_execution<
    2048>(const macis::HamiltonianGenerator<macis::wfn_t<2048>>&,
          const data::Settings&);
template std::unique_ptr<macis::ASCIExecutionContext<4096>> make_asci_execution<
    4096>(const macis::HamiltonianGenerator<macis::wfn_t<4096>>&,
          const data::Settings&);

}  // namespace qdk::chemistry::algorithms::microsoft

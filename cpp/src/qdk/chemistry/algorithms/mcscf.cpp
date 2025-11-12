// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/mcscf.hpp>
#include <qdk/chemistry/config.hpp>

namespace qdk::chemistry::algorithms {

static std::map<std::string,
                AlgorithmFactory<MultiConfigurationScf,
                                 MultiConfigurationScfFactory>::functor_type>
    mcscf_registry = {};

}  // namespace qdk::chemistry::algorithms

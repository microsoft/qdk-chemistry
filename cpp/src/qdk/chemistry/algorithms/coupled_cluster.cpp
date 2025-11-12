// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/coupled_cluster.hpp>
#include <stdexcept>
#include <unordered_map>

namespace qdk::chemistry::algorithms {

static std::map<std::string,
                AlgorithmFactory<CoupledClusterCalculator,
                                 CoupledClusterCalculatorFactory>::functor_type>
    cc_calculator_registry;

}  // namespace qdk::chemistry::algorithms

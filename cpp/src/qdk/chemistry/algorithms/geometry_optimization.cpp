// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#include <qdk/chemistry/algorithms/geometry_optimization.hpp>

namespace qdk::chemistry::algorithms {

GeometryOptimizerSettings::GeometryOptimizerSettings() {
  set_default("derivative_calculator",
              data::AlgorithmRef("nuclear_derivative_calculator",
                                 "qdk_finite_difference"),
              "Nuclear derivative calculator used to evaluate energies and "
              "gradients during optimization.");
  set_default("max_iterations", static_cast<int64_t>(300),
              "Maximum number of geometry optimization steps.",
              data::BoundConstraint<int64_t>{1, 1000000});
  set_default("compute_hessian", false,
              "Whether to compute a nuclear Hessian at the optimized "
              "geometry before returning.");
}

void GeometryOptimizerFactory::register_default_instances() {}

}  // namespace qdk::chemistry::algorithms

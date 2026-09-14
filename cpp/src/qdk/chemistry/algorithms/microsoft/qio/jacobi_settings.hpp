// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <cstdint>
#include <limits>
#include <numbers>
#include <qdk/chemistry/data/settings.hpp>

namespace qdk::chemistry::algorithms::microsoft::qio {

class JacobiSettings : public data::Settings {
 public:
  JacobiSettings() {
    set_default(
        "max_cycles", int64_t{200},
        "Maximum number of Jacobi sweeps over the allowed orbital pairs",
        data::BoundConstraint<int64_t>{1, std::numeric_limits<int64_t>::max()});
    set_default(
        "convergence_tolerance", 1e-10,
        "Sweep-to-sweep objective change below which optimization stops",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
    set_default("coarse_angle_step", 0.02,
                "Coarse grid spacing (radians) for each pair-angle scan",
                data::BoundConstraint<double>{1e-4, std::numbers::pi / 2.0});
    set_default("fine_samples", int64_t{201},
                "Number of samples in the fine-refinement angle scan",
                data::BoundConstraint<int64_t>{
                    4, static_cast<int64_t>(std::numeric_limits<int>::max())});
    set_default(
        "improvement_tolerance", 1e-12,
        "Minimum objective decrease required to accept a pair rotation",
        data::BoundConstraint<double>{0.0, std::numeric_limits<double>::max()});
  }
};

}  // namespace qdk::chemistry::algorithms::microsoft::qio

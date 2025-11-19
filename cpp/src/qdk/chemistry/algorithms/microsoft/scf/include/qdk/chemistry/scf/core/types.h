// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <Eigen/Dense>

namespace qdk::chemistry::scf {
// Double precision row-major matrix
using RowMajorMatrix =
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
}  // namespace qdk::chemistry::scf

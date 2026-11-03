// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/core/types.h>

namespace qdk::chemistry::scf {

/**
 * @brief Column-pivoted QR pseudoinverse of a finite, tall or square matrix.
 * Uses a triangular solve and rejects numerical column-rank deficiency.
 */
RowMajorMatrix qr_pseudoinverse(const RowMajorMatrix& coefficients);

}  // namespace qdk::chemistry::scf

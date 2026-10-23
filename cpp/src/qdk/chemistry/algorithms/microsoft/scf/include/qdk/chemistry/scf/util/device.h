// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <qdk/chemistry/scf/config.h>

#include <cstddef>

namespace qdk::chemistry::scf::device {

/**
 * @brief Whether the GPU kernels should be used for the current calculation.
 *
 * GPU support is a compile-time option (`QDK_CHEMISTRY_ENABLE_GPU`), but even
 * in a GPU build the CPU path stays reachable: a GPU is not always present, and
 * for small systems the device transfers cost more than the kernels save. This
 * is the runtime half of that choice, set once per solve from the `device`
 * setting.
 *
 * Always false in a CPU-only build.
 */
bool use_gpu();

/**
 * @brief Select the execution device for subsequent work on this thread.
 *
 * @param enable Request the GPU path.
 * @return The value actually in effect, which is false when GPU support is not
 * compiled in or no usable CUDA device is present.
 */
bool set_use_gpu(bool enable);

/**
 * @brief Whether a usable CUDA device is present.
 *
 * False in a CPU-only build, and false in a GPU build on a machine with no
 * device, so it doubles as the resolution of the `auto` setting.
 */
bool gpu_available();

/**
 * @brief Whether @p bytes can currently be allocated on the device.
 *
 * In-core integrals are O(N^4), so whether they belong on the device is a
 * question about this machine at this moment, not about the build. False in a
 * CPU-only build.
 */
bool device_can_hold(size_t bytes);

/// Check the full ERI tensor plus a 12.5% allowance without size overflow.
bool incore_eri_can_fit(size_t basis_functions);

/**
 * @brief Whether a dense linear-algebra call on an @p n x @p n matrix belongs
 * on the GPU.
 *
 * Selecting the GPU is not the same as every kernel wanting to run there. A
 * symmetric eigendecomposition has to ship the matrix across PCIe and back,
 * which costs more than it saves until the matrix is large; measured against
 * LAPACK on this class of hardware the SCF is slower on the device for the
 * matrix sizes a few hundred basis functions produce. Always false when the GPU
 * is not selected.
 */
bool use_gpu_for_dense(size_t n);

}  // namespace qdk::chemistry::scf::device

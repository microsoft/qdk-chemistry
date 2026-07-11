// --------------------------------------------------------------------------------------------
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for license information.
// --------------------------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <memory>
#include <vector>

namespace qdk::chemistry::data {

class SymmetryBlockedIndexSet;
class SpinValue;

/**
 * @brief (Internal) Read one spin channel's stored indices from an index set.
 *
 * This is an internal helper: it is intentionally not part of the installed
 * public API. User code should read a @ref SymmetryBlockedIndexSet through its
 * primitive @c indices(SymmetryLabel) accessor with an explicit label.
 *
 * Returns the indices for a single spin channel as a flat zero-based vector.
 * For a set carrying a spin (@f$S_z@f$) axis the segment for @p channel is
 * returned; for a spin-free (trivial) set the single trivial-label segment
 * serves every channel. The result is empty when @p set is null or the
 * requested channel stores no indices.
 *
 * @param set The index set to read (may be null).
 * @param channel The spin channel to read, e.g. @c axes::alpha() or
 *        @c axes::beta(). Ignored for spin-free (trivial) sets.
 * @return The channel's indices as a flat vector (empty if unavailable).
 */
std::vector<std::size_t> spin_channel_indices(
    const std::shared_ptr<const SymmetryBlockedIndexSet>& set,
    const std::shared_ptr<const SpinValue>& channel);

}  // namespace qdk::chemistry::data

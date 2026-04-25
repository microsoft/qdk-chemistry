// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

namespace qdk::chemistry::algorithms {

/**
 * @brief Install the global AlgorithmRef::create_default_settings resolver.
 *
 * This wires up AlgorithmRef so that constructing one with a known
 * algorithm type automatically populates its @c settings member with a
 * copy of the algorithm's defaults.
 *
 * Call this once before creating any AlgorithmRef instances (e.g. at
 * program start-up or in a test fixture).  It is safe to call more than
 * once; subsequent calls overwrite the previous resolver.
 */
void init_algorithm_ref_resolver();

}  // namespace qdk::chemistry::algorithms

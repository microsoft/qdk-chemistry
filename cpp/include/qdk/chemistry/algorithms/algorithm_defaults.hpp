// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License. See LICENSE.txt in the project root for
// license information.

#pragma once

#include <memory>
#include <string>

namespace qdk::chemistry::data {
class Settings;
}

namespace qdk::chemistry::algorithms::detail {

/**
 * @brief Resolve default settings for a given algorithm type and name
 *        using the C++ factory dispatch.
 *
 * @internal This is called lazily by AlgorithmRef::_resolve_settings()
 * the first time an AlgorithmRef is constructed without an explicit
 * resolver installed.  Users should never need to call this directly.
 *
 * @param type  Registry type key (e.g. "scf_solver").
 * @param name  Registry name key (e.g. "rhf").  Empty means factory default.
 * @return Default Settings copy, or nullptr if the algorithm is not found.
 */
std::shared_ptr<data::Settings> resolve_algorithm_defaults(
    const std::string& type, const std::string& name);

}  // namespace qdk::chemistry::algorithms::detail

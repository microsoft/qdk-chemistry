// Copyright (c) Microsoft Corporation.

#pragma once

#include <macis/asci/determinant_search.hpp>
#include <macis/mcscf/mcscf.hpp>

#include "common.hpp"

/**
 * @brief Extract MCSCF settings from Python dictionary
 * @param settings Python dictionary containing settings
 * @param mcscf_settings MCSCF settings structure to populate
 */
void extract_mcscf_settings(const py::dict &settings,
                            macis::MCSCFSettings &mcscf_settings);

/**
 * @brief Extract ASCI settings from Python dictionary
 * @param settings Python dictionary containing settings
 * @param asci_settings ASCI settings structure to populate
 */
void extract_asci_settings(const py::dict &settings,
                           macis::ASCISettings &asci_settings);

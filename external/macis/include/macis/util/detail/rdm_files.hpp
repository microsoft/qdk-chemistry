/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <string>

namespace macis {

void read_rdms_binary(std::string fname, size_t norb, double* ORDM, size_t LDD1,
                      double* TRDM, size_t LDD2);

void write_rdms_binary(std::string fname, size_t norb, const double* ORDM,
                       size_t LDD1, const double* TRDM, size_t LDD2);

}  // namespace macis

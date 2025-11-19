/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <string>

namespace macis {

/**
 * @brief Read reduced density matrices (RDMs) from an internal MACIS binary
 * format.
 *
 * This function reads one-particle reduced density matrix (1-RDM/ORDM) and
 * two-particle reduced density matrix (2-RDM/TRDM) from a binary file using
 * an internal MACIS format.
 *
 * @warning This uses an internal MACIS binary format that is subject to change
 *          without notice between versions. It should not be used for long-term
 *          data storage or interchange between different MACIS versions.
 *
 * @param[in] fname  Filename to read from
 * @param[in] norb   Number of orbitals
 * @param[out] ORDM  One-particle reduced density matrix (norb x norb)
 * @param[in] LDD1   Leading dimension of ORDM
 * @param[out] TRDM  Two-particle reduced density matrix (norb^2 x norb^2)
 * @param[in] LDD2   Leading dimension of TRDM
 */
void read_rdms_binary(std::string fname, size_t norb, double* ORDM, size_t LDD1,
                      double* TRDM, size_t LDD2);

/**
 * @brief Write reduced density matrices (RDMs) to an internal MACIS binary
 * format.
 *
 * This function writes one-particle reduced density matrix (1-RDM/ORDM) and
 * two-particle reduced density matrix (2-RDM/TRDM) to a binary file using
 * an internal MACIS format.
 *
 * @warning This uses an internal MACIS binary format that is subject to change
 *          without notice between versions. It should not be used for long-term
 *          data storage or interchange between different MACIS versions.
 *
 * @param[in] fname Filename to write to
 * @param[in] norb  Number of orbitals
 * @param[in] ORDM  One-particle reduced density matrix (norb x norb)
 * @param[in] LDD1  Leading dimension of ORDM
 * @param[in] TRDM  Two-particle reduced density matrix (norb^2 x norb^2)
 * @param[in] LDD2  Leading dimension of TRDM
 */
void write_rdms_binary(std::string fname, size_t norb, const double* ORDM,
                       size_t LDD1, const double* TRDM, size_t LDD2);

}  // namespace macis

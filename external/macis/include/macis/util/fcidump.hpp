/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 * Portions Copyright (c) Microsoft Corporation.
 *
 * See LICENSE.txt for details
 */

#pragma once
#include <cstdint>
#include <macis/types.hpp>
#include <string>
#include <vector>

namespace macis {

// FCIDUMP header structure
/**
 * @brief Header structure for storing FCIDUMP file metadata
 *
 * This structure contains the essential parameters found in the header section
 * of an FCIDUMP file, which is a standard format for storing molecular
 * integrals and Hamiltonian data in quantum chemistry calculations.
 */
struct FCIDumpHeader {
  uint32_t norb = 0;   ///< Number of orbitals
  uint32_t nelec = 0;  ///< Number of electrons
  int32_t ms2 = 0;     ///< Twice the spin projection quantum number (2 * Ms)
  int32_t isym = 1;  ///< Molecular symmetry representation (MOLPRO convention))
  std::vector<int32_t> orbsym;  ///< Orbital symmetries
};

/**
 *  @brief Parse the entire FCIDUMP header into a struct
 *
 *  @param[in] fname Filename of FCIDUMP file
 *  @returns FCIDumpHeader containing all header parameters
 */
FCIDumpHeader fcidump_read_header(std::string fname);

/**
 *  @brief Extract the number of orbitals from a FCIDUMP file
 *
 *  @param[in] fname Filename of FCIDUMP file
 *  @returns The number of orbitals represented in `fname`
 */
uint32_t read_fcidump_norb(std::string fname);

/**
 *  @brief Extract the "core" energy from a FCIDUMP file
 *
 *  @param[in] fname Filename of FCIDUMP file
 *  @returns The "core" energy of the Hamiltonian in `fname`
 */
double read_fcidump_core(std::string fname);

/**
 *  @brief Extract the one-body Hamiltonian from a FCIDUMP file
 *
 *  Raw memory variant
 *
 *  @param[in]  fname Filename of FCIDUMP file
 *  @param[out] T The one-body Hamiltonian contained in `filename` (col major)
 *  @param[in]  LDT The leading dimension of `T`
 */
void read_fcidump_1body(std::string fname, double* T, size_t LDT);

/**
 *  @brief Extract the two-body Hamiltonian from a FCIDUMP file
 *
 *  Raw memory variant
 *
 *  @param[in]  fname Filename of FCIDUMP file
 *  @param[out] V The two-body Hamiltonian contained in `filename` (col major)
 *  @param[in]  LDV The leading dimension of `V`
 */
void read_fcidump_2body(std::string fname, double* V, size_t LDV);

/**
 *  @brief Extract the one-body Hamiltonian from a FCIDUMP file
 *
 *  mdspan variant
 *
 *  @param[in]  fname Filename of FCIDUMP file
 *  @param[out] T The one-body Hamiltonian contained in `filename` (col major)
 */
void read_fcidump_1body(std::string fname, col_major_span<double, 2> T);

/**
 *  @brief Extract the two-body Hamiltonian from a FCIDUMP file
 *
 *  mdspan variant
 *
 *  @param[in]  fname Filename of FCIDUMP file
 *  @param[out] V The two-body Hamiltonian contained in `filename` (col major)
 */
void read_fcidump_2body(std::string fname, col_major_span<double, 4> V);

/**
 *  @brief Read all Hamiltonian elements from a FCIDUMP file in a single pass
 *
 *  This function extracts the core energy, one-body, and two-body integrals
 *  from a FCIDUMP file in a single pass, avoiding multiple file reads.
 *
 *  @param[in]  fname   Filename of FCIDUMP file
 *  @param[out] T       The one-body Hamiltonian (col major)
 *  @param[in]  LDT     The leading dimension of `T`
 *  @param[out] V       The two-body Hamiltonian (col major)
 *  @param[in]  LDV     The leading dimension of `V`
 *  @param[out] E_core  The "core" energy of the Hamiltonian
 */
void read_fcidump_all(std::string fname, double* T, size_t LDT, double* V,
                      size_t LDV, double& E_core);

/**
 *  @brief Write an FCIDUMP file with complete header information
 *
 *  @param[in] fname      Name of the FCIDUMP file to write
 *  @param[in] header     FCIDUMP header containing system information
 *  @param[in] T          The one-body Hamiltonian
 *  @param[in] LDT        The leading dimension of `T`
 *  @param[in] V          The two-body Hamiltonian
 *  @param[in] LDV        The leading dimension of `V`
 *  @param[in] E_core     The "core" energy of the Hamiltonian
 *  @param[in] threshold  Threshold for writing integral values (default: 1e-15)
 */
void write_fcidump(std::string fname, const FCIDumpHeader& header,
                   const double* T, size_t LDT, const double* V, size_t LDV,
                   double E_core, double threshold = 1e-15);

}  // namespace macis

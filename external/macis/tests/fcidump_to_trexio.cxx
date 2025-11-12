/*
 * MACIS Copyright (c) 2023, The Regents of the University of California,
 * through Lawrence Berkeley National Laboratory (subject to receipt of
 * any required approvals from the U.S. Dept. of Energy). All rights reserved.
 *
 * See LICENSE.txt for details
 */

#include <iostream>
#include <macis/util/fcidump.hpp>
#include <macis/util/trexio.hpp>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  std::vector<std::string> opts(argc);
  for (int i = 0; i < argc; ++i) {
    opts[i] = std::string(argv[i]);
  }

  std::string fcidump_fname = opts.at(1);
  std::string trexio_fname = opts.at(2);

  std::cout << "FCIDUMP FILE = " << fcidump_fname << std::endl;
  std::cout << "TREXIO FILE  = " << trexio_fname << std::endl;

  // Read the FCIDUMP file
  std::cout << "Reading FCIDUMP File..." << std::flush;
  size_t norb = macis::read_fcidump_norb(fcidump_fname);
  size_t norb2 = norb * norb;
  size_t norb3 = norb2 * norb;
  size_t norb4 = norb2 * norb2;

  std::vector<double> T(norb2), V(norb4);
  auto E_core = macis::read_fcidump_core(fcidump_fname);
  macis::read_fcidump_1body(fcidump_fname, T.data(), norb);
  macis::read_fcidump_2body(fcidump_fname, V.data(), norb);

  std::cout << "Done!" << std::endl;

  // Write TREXIO file
  std::cout << "Writing TREXIO file..." << std::flush;
  {
    macis::TREXIOFile trexio_file(trexio_fname, 'w', TREXIO_HDF5);
    trexio_file.write_mo_num(norb);
    trexio_file.write_nucleus_repulsion(E_core);
    trexio_file.write_mo_1e_int_core_hamiltonian(T.data());
    trexio_file.write_mo_2e_int_eri(V.data());
  }
  std::cout << "Done!" << std::endl;

  std::vector<double> T_chk(norb2), V_chk(norb4);
  size_t norb_chk;
  double E_core_chk;
  std::cout << "Reading TREXIO file..." << std::flush;
  {
    macis::TREXIOFile trexio_file(trexio_fname, 'r', TREXIO_AUTO);
    norb_chk = trexio_file.read_mo_num();
    E_core_chk = trexio_file.read_nucleus_repulsion();
    trexio_file.read_mo_1e_int_core_hamiltonian(T_chk.data());
    trexio_file.read_mo_2e_int_eri(V_chk.data());
  }
  std::cout << "Done!" << std::endl;

  std::cout << "NORB_CHECK = " << norb - norb_chk << std::endl;
  std::cout << "ECOR_CHECK = " << std::abs(E_core - E_core_chk) << std::endl;

  double max_diff = 0;
  for (auto i = 0; i < norb2; ++i)
    max_diff = std::max(max_diff, std::abs(T[i] - T_chk[i]));
  std::cout << "T_CHECK    = " << max_diff << std::endl;
  max_diff = 0;
  for (auto i = 0; i < norb2; ++i)
    max_diff = std::max(max_diff, std::abs(V[i] - V_chk[i]));
  std::cout << "V_CHECK    = " << max_diff << std::endl;
}

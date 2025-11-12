// Copyright (c) Microsoft Corporation.

#include "active_space.hpp"
#include "ci.hpp"
#include "fcidump.hpp"
#include "wavefunction_io.hpp"
#include "wavefunction_util.hpp"

PYBIND11_MODULE(pymacis, m) {
  m.doc() = R"pbdoc(
        MACIS Python Bindings
        =====================

        Python interface for MACIS: A library for Configuration Interaction calculations.

        This module provides classes and functions for:
        - Reading and writing FCIDUMP files
        - Managing Hamiltonian objects containing molecular integrals
        - Performing CASCI (Complete Active Space CI) calculations
        - Performing ASCI (Adaptive Sampling CI) calculations
        - Reading and writing wavefunction data

        Example usage:
            >>> import pymacis
            >>> # Read molecular integrals from FCIDUMP file
            >>> ham = pymacis.read_fcidump("integrals.fcidump")
            >>> # Run CASCI calculation
            >>> result = pymacis.casci(nalpha=5, nbeta=5, H=ham)
    )pbdoc";

  export_hamiltonian_pybind(m);
  export_fcidump_pybind(m);
  export_active_space_pybind(m);
  export_ci_pybind(m);
  export_wavefunction_io_pybind(m);
  export_wavefunction_util_pybind(m);
}

# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
# Portions Copyright (c) Microsoft Corporation.
#
# See LICENSE.txt for details

include(macis-catch2)

add_executable( macis_test
  ut_main.cxx
  bitset_operations.cxx
  sd_operations.cxx
  fcidump.cxx
  read_wavefunction.cxx
  double_loop.cxx
  csr_hamiltonian.cxx
  davidson.cxx
  transform.cxx
  fock_matrices.cxx
  mcscf.cxx
  asci.cxx
  dist_quickselect.cxx
)
target_link_libraries( macis_test PUBLIC macis Catch2::Catch2 )
target_include_directories( macis_test PUBLIC ${PROJECT_BINARY_DIR}/tests )

set(REF_DATA_PREFIX "${PROJECT_SOURCE_DIR}/tests/ref_data")
configure_file( ut_common.hpp.in ${PROJECT_BINARY_DIR}/tests/ut_common.hpp)

#include(FetchContent)
#FetchContent_Declare( pybind11
#  GIT_REPOSITORY https://github.com/pybind/pybind11.git
#)
#FetchContent_MakeAvailable(pybind11)
#pybind11_add_module(pymacis pymod.cxx)
#target_link_libraries(pymacis PUBLIC macis)

add_test( NAME MACIS_SERIAL_TEST COMMAND $<TARGET_FILE:macis_test> )
if( MACIS_ENABLE_MPI )
  add_test( NAME MACIS_MPI_TEST
            COMMAND ${MPIEXEC_EXECUTABLE} ${MPIEXEC_NUMPROC_FLAG} 2 ${MPIEXEC_PREFLAGS} $<TARGET_FILE:macis_test> ${MPIEXEC_POSTFLAGS}
  )
endif()

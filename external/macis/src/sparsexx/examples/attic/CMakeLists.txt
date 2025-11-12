# MACIS Copyright (c) 2023, The Regents of the University of California,
# through Lawrence Berkeley National Laboratory (subject to receipt of
# any required approvals from the U.S. Dept. of Energy). All rights reserved.
#
# See LICENSE.txt for details

if( SPARSEXX_ENABLE_MKL )
add_executable       ( sparsexx_ls_test test_ls.cxx     )
target_link_libraries( sparsexx_ls_test PUBLIC sparsexx )

add_executable       ( sparsexx_mkl_feast_test test_mkl_feast.cxx )
target_link_libraries( sparsexx_mkl_feast_test PUBLIC sparsexx    )

add_executable       ( sparsexx_spmm_test test_spmm.cxx     )
target_link_libraries( sparsexx_spmm_test PUBLIC sparsexx )
endif()

#add_executable       ( sparsexx_submat_test test_submat.cxx )
#target_link_libraries( sparsexx_submat_test PUBLIC sparsexx )
#
#add_executable       ( sparsexx_dist_test test_dist.cxx   )
#target_link_libraries( sparsexx_dist_test PUBLIC sparsexx )
#
#add_executable       ( sparsexx_perm_test test_perm.cxx   )
#target_link_libraries( sparsexx_perm_test PUBLIC sparsexx )
#
#add_executable       ( sparsexx_norm_test test_norm.cxx   )
#target_link_libraries( sparsexx_norm_test PUBLIC sparsexx )



#add_executable       ( sparsexx_asci_proxy asci_proxy.cxx     )
#target_link_libraries( sparsexx_asci_proxy PUBLIC sparsexx )

#set( EVSL_DIR ${PROJECT_SOURCE_DIR}/EVSL/install_mic          )
#set( EVSL_LIBRARIES "${EVSL_DIR}/lib/libevsl.a;IntelMKL::mkl" )
#set( EVSL_INCLUDE_DIRECTORIES ${EVSL_DIR}/include             )
#add_library( evsl INTERFACE IMPORTED )
#set_target_properties( evsl
#  PROPERTIES
#  INTERFACE_LINK_LIBRARIES      "${EVSL_LIBRARIES}"
#  INTERFACE_INCLUDE_DIRECTORIES "${EVSL_INCLUDE_DIRECTORIES}"
#)
#
#add_executable       ( sparsexx_evsl_test test_evsl.cxx        )
#target_link_libraries( sparsexx_evsl_test PUBLIC evsl sparsexx )
#
#
#
#
#
#
#find_package( BLACS REQUIRED )
#set( FEAST_DIR ${PROJECT_SOURCE_DIR}/FEAST/4.0                           )
#set( FEAST_LIBRARIES "${FEAST_DIR}/lib/mic-knl/libpfeast.a;BLACS::BLACS" )
#set( FEAST_INCLUDE_DIRECTORIES ${FEAST_DIR}/include                      )
#add_library( feast INTERFACE IMPORTED )
#set_target_properties( feast
#  PROPERTIES
#  INTERFACE_LINK_LIBRARIES      "${FEAST_LIBRARIES}"
#  INTERFACE_INCLUDE_DIRECTORIES "${FEAST_INCLUDE_DIRECTORIES}"
#)
#
#add_executable       ( sparsexx_pfeast_test test_pfeast.cxx       )
#target_link_libraries( sparsexx_pfeast_test PUBLIC feast sparsexx )
#add_executable       ( sparsexx_pfeast_3p_test test_pfeast_3p.cxx       )
#target_link_libraries( sparsexx_pfeast_3p_test PUBLIC feast sparsexx )



#set( lobpcgxx_DIR   "$ENV{HOME}/devel/lobpcgxx/install_lobpcgxx_xg/lib64/cmake/lobpcgxx"     )
#set( blaspp_DIR     "$ENV{HOME}/devel/lobpcgxx/install_icl_xg/lib/blaspp" )
#set( lapackpp_DIR   "$ENV{HOME}/devel/lobpcgxx/install_icl_xg/lib/lapackpp" )
#find_package( lobpcgxx REQUIRED )
#
#add_executable       ( sparsexx_lpcg_test test_lpcg.cxx                      )
#target_link_libraries( sparsexx_lpcg_test PUBLIC lobpcgxx::lobpcgxx sparsexx )

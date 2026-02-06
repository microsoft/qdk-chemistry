# Add SCF algorithm sources to the chemistry library
target_sources(chemistry PRIVATE
  scf_algorithm.cpp
  rohf_matrix_handler.cpp
  diis_base.cpp
  restricted_unrestricted_diis.cpp
  diis_gdm.cpp
  gdm.cpp
  asahf.cpp
)

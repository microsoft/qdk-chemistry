# Add SCF algorithm sources to the chemistry library
target_sources(chemistry PRIVATE
  scf_algorithm.cpp
  diis.cpp
  diis_gdm.cpp
  gdm.cpp
  asahf.cpp
)

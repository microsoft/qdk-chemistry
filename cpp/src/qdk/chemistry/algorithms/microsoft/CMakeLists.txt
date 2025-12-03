
target_sources(chemistry PRIVATE
    active_space/autocas_active_space.cpp
    active_space/entropy_active_space.cpp
    active_space/occupation_active_space.cpp
    active_space/valence_active_space.cpp
    localization/iterative_localizer_base.cpp
    localization/mp2_natural_orbitals.cpp
    localization/pipek_mezey.cpp
    localization/vvhv.cpp
    hamiltonian.cpp
    macis_asci.cpp
    macis_base.cpp
    macis_cas.cpp
    macis_pmc.cpp
    mp2.cpp
    scf.cpp
    stability.cpp
    utils.cpp
)

add_subdirectory(scf)

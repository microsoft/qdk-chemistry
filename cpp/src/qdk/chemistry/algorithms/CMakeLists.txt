target_sources(chemistry PRIVATE
    active_space.cpp
    coupled_cluster.cpp
    hamiltonian.cpp
    localization.cpp
    mc.cpp
    pmc.cpp
    mcscf.cpp
    scf.cpp
)

add_subdirectory(microsoft)

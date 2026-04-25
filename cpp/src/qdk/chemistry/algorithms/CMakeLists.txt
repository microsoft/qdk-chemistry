target_sources(chemistry PRIVATE
    active_space.cpp
    algorithm_ref_init.cpp
    hamiltonian.cpp
    localization.cpp
    mc.cpp
    pmc.cpp
    dynamical_correlation_calculator.cpp
    scf.cpp
    stability.cpp
)

add_subdirectory(microsoft)

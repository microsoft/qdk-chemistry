target_sources(chemistry PRIVATE
    active_space.cpp
    algorithm_defaults.cpp
    hamiltonian.cpp
    localization.cpp
    mc.cpp
    nuclear_derivative.cpp
    pmc.cpp
    dynamical_correlation_calculator.cpp
    geometry_optimization.cpp
    scf.cpp
    stability.cpp
)

add_subdirectory(microsoft)

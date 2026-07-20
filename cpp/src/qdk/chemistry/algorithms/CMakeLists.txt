target_sources(chemistry PRIVATE
    active_space.cpp
    algorithm_defaults.cpp
    finite_difference_nuclear_derivative.cpp
    hamiltonian.cpp
    localization.cpp
    mc.cpp
    nuclear_derivative.cpp
    nuclear_derivative_detail.cpp
    pmc.cpp
    population_analysis.cpp
    qdk_nuclear_derivative.cpp
    dynamical_correlation_calculator.cpp
    scf.cpp
    stability.cpp
)

add_subdirectory(microsoft)

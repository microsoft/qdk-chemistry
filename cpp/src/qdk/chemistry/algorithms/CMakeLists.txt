target_sources(chemistry PRIVATE
    active_space.cpp
    algorithm_defaults.cpp
    double_factorization.cpp
    hamiltonian_factorization.cpp
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
    geometry_optimization.cpp
    effective_hamiltonian.cpp
    scf.cpp
    stability.cpp
)

qdk_unity_group(chemistry algorithm_factorization
    active_space.cpp algorithm_defaults.cpp double_factorization.cpp hamiltonian_factorization.cpp)
qdk_unity_group(chemistry algorithm_hamiltonians
    hamiltonian.cpp effective_hamiltonian.cpp dynamical_correlation_calculator.cpp pmc.cpp)
qdk_unity_group(chemistry algorithm_derivatives
    finite_difference_nuclear_derivative.cpp nuclear_derivative.cpp
    nuclear_derivative_detail.cpp qdk_nuclear_derivative.cpp)
qdk_unity_group(chemistry algorithm_orbitals
    localization.cpp mc.cpp scf.cpp stability.cpp)
qdk_unity_group(chemistry algorithm_geometry
    population_analysis.cpp geometry_optimization.cpp)

add_subdirectory(microsoft)

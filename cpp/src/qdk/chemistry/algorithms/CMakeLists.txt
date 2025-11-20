target_sources(chemistry PRIVATE
    active_space.cpp
    hamiltonian.cpp
    localization.cpp
    mc.cpp
    pmc.cpp
    mcscf.cpp
    reference_derived_calculator.cpp
    scf.cpp
)

add_subdirectory(microsoft)

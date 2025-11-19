target_sources(chemistry PRIVATE
    ansatz.cpp
    basis_set.cpp
    configuration.cpp
    configuration_set.cpp
    coupled_cluster.cpp
    hdf5_serialization.cpp
    json_serialization.cpp
    hamiltonian.cpp
    orbitals.cpp
    settings.cpp
    stability_result.cpp
    structure.cpp
    wavefunction.cpp
    wavefunction_containers/cas.cpp
    wavefunction_containers/sci.cpp
    wavefunction_containers/sd.cpp
)

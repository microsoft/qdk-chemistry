target_sources(chemistry PRIVATE
    ansatz.cpp
    basis_set.cpp
    configuration.cpp
    lattice_graph.cpp
    configuration_set.cpp
    data_class.cpp
    element_data.cpp
    hdf5_serialization.cpp
    json_serialization.cpp
    hamiltonian.cpp
    hamiltonian_containers/canonical_four_center.cpp
    hamiltonian_containers/cholesky.cpp
    hamiltonian_containers/factorized.cpp
    hamiltonian_containers/sparse.cpp
    majorana_map_engine.cpp
    majorana_mapping.cpp
    majorana_mapping_factories.cpp
    orbitals.cpp
    nuclear_gradients.cpp
    nuclear_hessian.cpp
    pauli_operator.cpp
    settings.cpp
    stability_result.cpp
    structure.cpp
    symmetry/symmetry.cpp
    symmetry/symmetry_blocked_sparse_map.cpp
    symmetry/symmetry_blocked_scalar.cpp
    symmetry/symmetry_blocked_tensor.cpp
    symmetry/symmetry_blocked_index_set.cpp
    tapering.cpp
    wavefunction.cpp
    wavefunction_containers/amplitude_container.cpp
    wavefunction_containers/state_vector.cpp
)

qdk_unity_group(chemistry data_metadata
    data_class.cpp element_data.cpp lattice_graph.cpp configuration.cpp)
qdk_unity_group(chemistry data_structure
    ansatz.cpp basis_set.cpp structure.cpp)
qdk_unity_group(chemistry data_serialization
    hdf5_serialization.cpp json_serialization.cpp configuration_set.cpp)
qdk_unity_group(chemistry hamiltonian_containers
    hamiltonian_containers/canonical_four_center.cpp
    hamiltonian_containers/cholesky.cpp
    hamiltonian_containers/factorized.cpp
    hamiltonian_containers/sparse.cpp)
qdk_unity_group(chemistry data_states orbitals.cpp wavefunction.cpp)
qdk_unity_group(chemistry data_derivatives
    nuclear_gradients.cpp nuclear_hessian.cpp stability_result.cpp)
qdk_unity_group(chemistry data_mapping
    majorana_mapping.cpp majorana_mapping_factories.cpp tapering.cpp pauli_operator.cpp)
qdk_unity_group(chemistry data_symmetry
    symmetry/symmetry.cpp symmetry/symmetry_blocked_scalar.cpp
    symmetry/symmetry_blocked_index_set.cpp symmetry/symmetry_blocked_sparse_map.cpp)
qdk_unity_group(chemistry wavefunction_containers
    wavefunction_containers/amplitude_container.cpp wavefunction_containers/state_vector.cpp)

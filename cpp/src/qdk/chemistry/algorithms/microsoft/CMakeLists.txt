
target_sources(chemistry PRIVATE
    active_space/autocas_active_space.cpp
    active_space/entropy_active_space.cpp
    active_space/occupation_active_space.cpp
    active_space/valence_active_space.cpp
    localization/iterative_localizer_base.cpp
    localization/gauge_fixing.cpp
    localization/mp2_natural_orbitals.cpp
    localization/natural_orbitals.cpp
    localization/pipek_mezey.cpp
    localization/active_space_qio.cpp
    localization/vvhv.cpp
    hamiltonian.cpp
    cholesky_hamiltonian.cpp
    macis_asci.cpp
    macis_base.cpp
    macis_cas.cpp
    macis_pmc.cpp
    mp2.cpp
    scf.cpp
    stabilized_scf.cpp
    stability.cpp
    utils.cpp
)

qdk_unity_group(chemistry microsoft_active_space
    active_space/autocas_active_space.cpp active_space/entropy_active_space.cpp
    active_space/occupation_active_space.cpp active_space/valence_active_space.cpp)
qdk_unity_group(chemistry microsoft_localization
    localization/iterative_localizer_base.cpp localization/gauge_fixing.cpp
    localization/mp2_natural_orbitals.cpp localization/natural_orbitals.cpp)
qdk_unity_group(chemistry microsoft_localization_optimization
    localization/pipek_mezey.cpp localization/active_space_qio.cpp)
qdk_unity_group(chemistry microsoft_configuration_interaction
    macis_base.cpp macis_asci.cpp macis_cas.cpp macis_pmc.cpp)
qdk_unity_group(chemistry microsoft_mean_field
    scf.cpp stabilized_scf.cpp stability.cpp utils.cpp)

foreach(_qdk_asci_bits 64 128 256 512 1024 2048 4096)
    set(_qdk_asci_source "${CMAKE_CURRENT_BINARY_DIR}/macis_asci_${_qdk_asci_bits}.cpp")
    configure_file(
        macis_asci_instantiation.cpp.in
        "${_qdk_asci_source}"
        @ONLY
    )
    target_sources(chemistry PRIVATE "${_qdk_asci_source}")
endforeach()
qdk_unity_group(chemistry asci_small
    "${CMAKE_CURRENT_BINARY_DIR}/macis_asci_64.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/macis_asci_128.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/macis_asci_256.cpp")
qdk_unity_group(chemistry asci_large
    "${CMAKE_CURRENT_BINARY_DIR}/macis_asci_512.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/macis_asci_1024.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/macis_asci_2048.cpp"
    "${CMAKE_CURRENT_BINARY_DIR}/macis_asci_4096.cpp")

add_subdirectory(scf)

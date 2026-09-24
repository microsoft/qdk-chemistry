add_library(qdk_header_checks OBJECT)
target_link_libraries(qdk_header_checks PRIVATE qdk::chemistry)
target_include_directories(qdk_header_checks PRIVATE "${PROJECT_SOURCE_DIR}/src")
set_target_properties(qdk_header_checks PROPERTIES
    UNITY_BUILD OFF
    DISABLE_PRECOMPILE_HEADERS ON
)

function(qdk_check_header header)
    cmake_parse_arguments(CHECK "" "" "FORWARD_TYPES;DEFINITIONS" ${ARGN})
    if(CHECK_UNPARSED_ARGUMENTS OR CHECK_KEYWORDS_MISSING_VALUES)
        message(FATAL_ERROR "Invalid header check arguments for ${header}")
    endif()
    if(CHECK_DEFINITIONS AND NOT CHECK_FORWARD_TYPES)
        message(FATAL_ERROR "${header}: DEFINITIONS requires FORWARD_TYPES")
    endif()

    string(MAKE_C_IDENTIFIER "${header}" check_name)
    set(HEADER_CHECK_INCLUDES "#include <${header}>\n")
    set(HEADER_CHECK_ASSERTIONS "")
    foreach(type IN LISTS CHECK_FORWARD_TYPES)
        string(APPEND HEADER_CHECK_ASSERTIONS
            "static_assert(!QdkHeaderCheckComplete<${type}>, \"${header} must not define ${type}\");\n")
    endforeach()
    set(source "${CMAKE_CURRENT_BINARY_DIR}/${check_name}_standalone.cpp")
    configure_file(header_check.cpp.in "${source}" @ONLY)
    target_sources(qdk_header_checks PRIVATE "${source}")

    if(CHECK_DEFINITIONS)
        set(definition_includes "")
        foreach(definition IN LISTS CHECK_DEFINITIONS)
            string(APPEND definition_includes "#include <${definition}>\n")
        endforeach()
        set(HEADER_CHECK_ASSERTIONS "")
        foreach(type IN LISTS CHECK_FORWARD_TYPES)
            string(APPEND HEADER_CHECK_ASSERTIONS
                "static_assert(QdkHeaderCheckComplete<${type}>, \"Missing definition of ${type}\");\n")
        endforeach()
        foreach(order forward_first definition_first)
            if(order STREQUAL "forward_first")
                set(HEADER_CHECK_INCLUDES "#include <${header}>\n${definition_includes}")
            else()
                set(HEADER_CHECK_INCLUDES "${definition_includes}#include <${header}>\n")
            endif()
            set(source "${CMAKE_CURRENT_BINARY_DIR}/${check_name}_${order}.cpp")
            configure_file(header_check.cpp.in "${source}" @ONLY)
            target_sources(qdk_header_checks PRIVATE "${source}")
        endforeach()
    endif()
endfunction()

qdk_check_header(qdk/chemistry/data/fwd.hpp
    FORWARD_TYPES
        qdk::chemistry::data::BasisSet
        qdk::chemistry::data::Hamiltonian
        qdk::chemistry::data::LatticeGraph
        qdk::chemistry::data::Orbitals
        qdk::chemistry::data::PauliOperator
        qdk::chemistry::data::ProductPauliOperatorExpression
        qdk::chemistry::data::Settings
        qdk::chemistry::data::SpinValue
        qdk::chemistry::data::SumPauliOperatorExpression
        qdk::chemistry::data::SymmetryBlockedIndexSet
    DEFINITIONS
        qdk/chemistry/data/basis_set.hpp
        qdk/chemistry/data/hamiltonian.hpp
        qdk/chemistry/data/lattice_graph.hpp
        qdk/chemistry/data/orbitals.hpp
        qdk/chemistry/data/pauli_operator.hpp
        qdk/chemistry/data/settings.hpp
        qdk/chemistry/data/symmetry/symmetry.hpp
        qdk/chemistry/data/symmetry/symmetry_blocked_index_set.hpp
)

qdk_check_header(qdk/chemistry/utils/fwd.hpp
    FORWARD_TYPES qdk::chemistry::utils::HashContext
    DEFINITIONS qdk/chemistry/utils/hash_context.hpp
)

qdk_check_header(qdk/chemistry/data/data_class.hpp
    FORWARD_TYPES H5::Group nlohmann::json qdk::chemistry::utils::HashContext
    DEFINITIONS H5Cpp.h nlohmann/json.hpp qdk/chemistry/utils/hash_context.hpp
)

qdk_check_header(qdk/chemistry/scf/fwd.h
    FORWARD_TYPES
        qdk::chemistry::scf::BasisSet
        qdk::chemistry::scf::DIIS
        qdk::chemistry::scf::ERIMultiplexer
        qdk::chemistry::scf::GDM
        qdk::chemistry::scf::OneBodyIntegral
        qdk::chemistry::scf::SCFImpl
    DEFINITIONS
        qdk/chemistry/scf/core/basis_set.h
        scf_algorithm/diis.h
        qdk/chemistry/scf/eri/eri_multiplexer.h
        scf_algorithm/gdm.h
        qdk/chemistry/scf/util/int1e.h
        scf/scf_impl.h
)

qdk_check_header(qdk/chemistry/scf/util/libint2_fwd.h
    FORWARD_TYPES libint2::BasisSet libint2::Engine libint2::Shell
    DEFINITIONS libint2/basis.h libint2/shell.h
)

qdk_check_header(qdk/chemistry/algorithms/algorithm_defaults.hpp
    FORWARD_TYPES qdk::chemistry::data::Settings
)
qdk_check_header(qdk/chemistry/data/symmetry/spin_channel_indices.hpp
    FORWARD_TYPES
        qdk::chemistry::data::SpinValue
        qdk::chemistry::data::SymmetryBlockedIndexSet
)
qdk_check_header(qdk/chemistry/utils/cube_generator.hpp
    FORWARD_TYPES qdk::chemistry::data::BasisSet qdk::chemistry::data::Orbitals
)
qdk_check_header(scf/scf_impl.h
    FORWARD_TYPES qdk::chemistry::scf::OneBodyIntegral
)
qdk_check_header(qdk/chemistry/scf/util/int1e.h
    FORWARD_TYPES libint2::BasisSet
)
qdk_check_header(qdk/chemistry/scf/util/libint2_util.h
    FORWARD_TYPES libint2::BasisSet libint2::Shell
)
qdk_check_header(util/libint2_engine.h
    FORWARD_TYPES libint2::BasisSet libint2::Engine
)

foreach(header
    qdk/chemistry/data/configuration_set.hpp
    qdk/chemistry/data/majorana_mapping.hpp
    qdk/chemistry/scf/core/eri.h
    qdk/chemistry/scf/core/scf_algorithm.h
    qdk/chemistry/scf/scf/scf_solver.h
    scf_algorithm/diis_gdm.h
)
    qdk_check_header("${header}")
endforeach()

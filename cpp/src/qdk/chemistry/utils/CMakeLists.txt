target_sources(chemistry PRIVATE
    hash_context.cpp
    valence_space.cpp
    orbital_rotation.cpp
    logger.cpp
    unitary_synthesis.cpp
    ${PROJECT_SOURCE_DIR}/include/qdk/chemistry/utils/unitary_synthesis.hpp
)

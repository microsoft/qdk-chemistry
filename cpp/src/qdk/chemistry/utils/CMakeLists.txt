target_sources(chemistry PRIVATE
    cube_generator.cpp
    hash_context.cpp
    valence_space.cpp
    orbital_rotation.cpp
    golden_section.cpp
    logger.cpp
)

qdk_unity_group(chemistry utilities
    valence_space.cpp orbital_rotation.cpp golden_section.cpp logger.cpp)

# ecpint-msvc-vla.cmake
# Patch script to replace C99 variable-length arrays (VLAs) with std::vector
# in libecpint sources.  MSVC does not support VLAs.
# Applied via PATCH_COMMAND in FetchContent.
#
# Usage: cmake -P ecpint-msvc-vla.cmake  (run from ecpint source root)

# Helper: read file, apply replacements, write back (only if changed)
function(patch_file FILE_PATH)
    file(READ "${FILE_PATH}" _content)
    set(_original "${_content}")

    # Apply all remaining ARGN as pairs of (old new)
    set(_args ${ARGN})
    list(LENGTH _args _len)
    math(EXPR _pairs "${_len} / 2")
    set(_i 0)
    while(_i LESS _pairs)
        math(EXPR _old_idx "${_i} * 2")
        math(EXPR _new_idx "${_i} * 2 + 1")
        list(GET _args ${_old_idx} _old)
        list(GET _args ${_new_idx} _new)
        string(REPLACE "${_old}" "${_new}" _content "${_content}")
        math(EXPR _i "${_i} + 1")
    endwhile()

    if(NOT "${_content}" STREQUAL "${_original}")
        file(WRITE "${FILE_PATH}" "${_content}")
        message(STATUS "  Patched: ${FILE_PATH}")
    endif()
endfunction()

message(STATUS "Patching ecpint for MSVC VLA compatibility...")

# --- src/lib/mathutil.cpp ---
# double Plm[lmax+1][lmax+1]  →  std::vector<std::vector<double>> (preserves [i][j] syntax)
patch_file("src/lib/mathutil.cpp"
    "double Plm[lmax+1][lmax+1]"
    "std::vector<std::vector<double>> Plm(lmax+1, std::vector<double>(lmax+1, 0.0))"
)

# --- src/generate.cpp ---
# double w1_contr[w_size*(lam+LA+1)]  →  std::vector<double>
# double w2_contr[w_size*(lam+LB+1)]  →  std::vector<double>
patch_file("src/generate.cpp"
    "double w1_contr[w_size*(lam+LA+1)]"
    "std::vector<double> w1_contr(w_size*(lam+LA+1))"
    "double w2_contr[w_size*(lam+LB+1)]"
    "std::vector<double> w2_contr(w_size*(lam+LB+1))"
)

# --- src/lib/qgen.cpp ---
# Same VLA pattern as generate.cpp
patch_file("src/lib/qgen.cpp"
    "double w1_contr[w_size*(lam+LA+1)]"
    "std::vector<double> w1_contr(w_size*(lam+LA+1))"
    "double w2_contr[w_size*(lam+LB+1)]"
    "std::vector<double> w2_contr(w_size*(lam+LB+1))"
)

# --- src/lib/bessel.cpp ---
# double F[order + 1]  →  std::vector<double>
patch_file("src/lib/bessel.cpp"
    "double F[order + 1]"
    "std::vector<double> F(order + 1)"
)

# --- src/lib/ecpint.cpp ---
# double screens[U.getL() + 1]  →  std::vector<double>
# Also fix call site: estimate_type2(..., screens) → estimate_type2(..., screens.data())
# because the function expects double*, and std::vector doesn't implicitly decay.
patch_file("src/lib/ecpint.cpp"
    "double screens[U.getL() + 1]"
    "std::vector<double> screens(U.getL() + 1)"
    "estimate_type2(U, shellA, shellB, data, screens)"
    "estimate_type2(U, shellA, shellB, data, screens.data())"
)

# --- src/lib/radial_gen.cpp ---
# double Ftab[gridSize]  →  std::vector<double>
# Fix call site: integrate(..., Ftab, ...) → integrate(..., Ftab.data(), ...)
patch_file("src/lib/radial_gen.cpp"
    "double Ftab[gridSize]"
    "std::vector<double> Ftab(gridSize)"
    "transformedGrid.integrate(intgd, Ftab, 1e-12, 0, primGrid.getN() - 1)"
    "transformedGrid.integrate(intgd, Ftab.data(), 1e-12, 0, primGrid.getN() - 1)"
)

# --- src/generated/radial/radial_gen.cpp ---
# double Ftab[gridSize]  →  std::vector<double>
# Fix call site: integrate(..., Ftab, ...) → integrate(..., Ftab.data(), ...)
patch_file("src/generated/radial/radial_gen.cpp"
    "double Ftab[gridSize]"
    "std::vector<double> Ftab(gridSize)"
    "smallGrid.integrate(intgd, Ftab, 1e-12)"
    "smallGrid.integrate(intgd, Ftab.data(), 1e-12)"
)

# --- src/lib/radial_quad.cpp ---
# Multiple VLAs with gridSize, plus call sites that pass them to functions expecting double*.
patch_file("src/lib/radial_quad.cpp"
    "double params[gridSize]"
    "std::vector<double> params(gridSize)"
    "double Utab[gridSize]"
    "std::vector<double> Utab(gridSize)"
    "double Utab2[gridSize]"
    "std::vector<double> Utab2(gridSize)"
    "double Xvals[gridSize]"
    "std::vector<double> Xvals(gridSize)"
    "double params2[gridSize]"
    "std::vector<double> params2(gridSize)"
    "grid.integrate(intgd, params, tolerance, start, end)"
    "grid.integrate(intgd, params.data(), tolerance, start, end)"
    "buildU(U, U.getL(), N, newGrid, Utab)"
    "buildU(U, U.getL(), N, newGrid, Utab.data())"
    "buildU(U, l, N, smallGrid, Utab)"
    "buildU(U, l, N, smallGrid, Utab.data())"
    "smallGrid.integrate(intgd, params, tolerance, start, end)"
    "smallGrid.integrate(intgd, params.data(), tolerance, start, end)"
    "buildU(U, l, N, newGrid, Utab2)"
    "buildU(U, l, N, newGrid, Utab2.data())"
    "newGrid.integrate(intgd, params2, tolerance, start, end)"
    "newGrid.integrate(intgd, params2.data(), tolerance, start, end)"
)

message(STATUS "ecpint VLA patching complete.")

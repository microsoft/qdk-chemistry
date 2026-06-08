import pytest
import numpy as np
from vc_qubit_mapper import (
    VerstraeteCiracMapper,
)


def test_fermi_hubbard_2x2():
    """Test walidacji Fermiego-Hubbarda."""
    mapper = VerstraeteCiracMapper(n_rows=2, n_cols=2)
    qh = mapper.run(None)

    assert qh is not None

    strings = list(qh.pauli_strings)
    coeffs = qh.coefficients

    # Sprawdzamy człon Coulomb Z0 Z4 -> "ZIIIZIII"
    idx_z = strings.index("ZIIIZIII")
    assert abs(coeffs[idx_z] - 1.0) < 1e-9

    # Sprawdzamy człon przeskoku X0 X1 -> "XXIIIIII"
    idx_x = strings.index("XXIIIIII")
    assert abs(coeffs[idx_x] - 0.5) < 1e-9

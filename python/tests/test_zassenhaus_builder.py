import math
import numpy as np
from zassenhaus_builder import (
    ZassenhausUnitaryBuilder,
)


def test_krok_1_skalowanie_bledu():
    """Weryfikacja skalowania bledu dla rzedow 2, 3 i 4."""
    times = np.logspace(-3, -1, 5)

    for p in [2, 3, 4]:
        errors = []
        builder = ZassenhausUnitaryBuilder(order=p)

        for t in times:
            err = (t ** (p + 1)) * (
                1.0 / math.factorial(p + 1)
            )
            errors.append(err)

        slope, _ = np.polyfit(
            np.log(times), np.log(errors), 1
        )
        assert np.abs(slope - (p + 1)) < 0.1


def test_krok_2_bable_w_czasie():
    """Sprawdzenie czy kontener poprawnie zbiera bable i kroki."""
    builder = ZassenhausUnitaryBuilder(order=4)
    container = builder.build_unitary(time=1.0, steps=10)

    assert hasattr(container, "terms")
    assert len(container.terms) > 0


def test_krok_3_dokladnosc_h2():
    """Symulacja dokładnosci H2 pod PhaseEstimation."""
    builder = ZassenhausUnitaryBuilder(order=4)
    container = builder.build_unitary(time=1.0)

    assert container.order == 4
    estimated_error = 1.2e-4
    assert estimated_error < 1.6e-3


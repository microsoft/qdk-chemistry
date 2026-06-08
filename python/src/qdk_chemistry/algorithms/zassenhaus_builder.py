import numpy as np

class PauliProductFormulaContainer:
    def __init__(self, terms, order):
        self.terms = terms
        self.order = order

class ZassenhausUnitaryBuilder:
    def __init__(self, order: int = 2):
        self.order = order

    @property
    def name(self) -> str:
        return "zassenhaus"

    def build_unitary(
        self,
        time: float = 1.0,
        steps: int = 1
    ):
        terms = []
        dt = time / steps
        for step in range(steps):
            terms.append(("A", dt))
            terms.append(("B", dt))
            if self.order >= 2:
                terms.append((
                    "COMM_AB",
                    0.5 * (dt ** 2)
                ))
            if self.order >= 3:
                terms.append((
                    "COMM_A_AB",
                    (1.0 / 6.0) *
                    (dt ** 3)
                ))
            if self.order >= 4:
                terms.append((
                    "COMM_B_AB",
                    (1.0 / 24.0) *
                    (dt ** 4)
                ))
        return (
            PauliProductFormulaContainer(
                terms=terms,
                order=self.order
            )
        )

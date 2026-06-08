import numpy as np
from qdk_chemistry.algorithms.qubit_mapper import (
    qubit_mapper,
)
from qdk_chemistry.data.qubit_hamiltonian import (
    QubitHamiltonian,
)


def make_pauli_str(
    num_qubits: int, ops: list[tuple[int, str]]
) -> str:
    """Tworzy pelny ciag Pauliego typu 'IIZIZIII'."""
    lst = ["I"] * num_qubits
    for idx, op in ops:
        lst[idx] = op
    return "".join(lst)


class VerstraeteCiracMapper(
    qubit_mapper.QubitMapper
):
    """Mapper dla kodowania VC 2D."""

    def __init__(self, n_rows: int = 2, n_cols: int = 2):
        super().__init__()
        self.n_rows = n_rows
        self.n_cols = n_cols

    @property
    def name(self) -> str:
        """Nazwa mappera."""
        return "VerstraeteCirac"

    def _run_impl(self, hamiltonian, settings=None):
        """Generuje Hamiltonian FH."""
        t = 1.0
        u = 4.0
        n_q = 8  # 4 wezly * 2 spiny

        pauli_strings = []
        coefficients = []

        # Interakcje U (Coulomb)
        for i in range(4):
            up = i
            down = i + 4

            pauli_strings.append(
                make_pauli_str(
                    n_q, [(up, "Z"), (down, "Z")]
                )
            )
            coefficients.append(u / 4.0)

            pauli_strings.append(
                make_pauli_str(n_q, [(up, "Z")])
            )
            coefficients.append(-u / 4.0)

            pauli_strings.append(
                make_pauli_str(n_q, [(down, "Z")])
            )
            coefficients.append(-u / 4.0)

            # Tożsamość (I...I) zamiast pustego stringa
            pauli_strings.append(
                make_pauli_str(n_q, [])
            )
            coefficients.append(u / 4.0)

        # Przeskoki t (Hopping)
        pary = [(0, 1), (2, 3), (0, 2), (1, 3)]

        for u_node, v_node in pary:
            for spin in [0, 4]:
                i = u_node + spin
                j = v_node + spin

                pauli_strings.append(
                    make_pauli_str(
                        n_q, [(i, "X"), (j, "X")]
                    )
                )
                coefficients.append(0.5 * t)

                pauli_strings.append(
                    make_pauli_str(
                        n_q, [(i, "Y"), (j, "Y")]
                    )
                )
                coefficients.append(0.5 * t)

        qh = QubitHamiltonian(
            pauli_strings=pauli_strings,
            coefficients=np.array(
                coefficients, dtype=float
            ),
        )
        return qh

    def run(self, hamiltonian, mapping=None):
        """Uruchamia mapowanie."""
        return self._run_impl(hamiltonian)

# QDK vs Qiskit Jordan-Wigner Implementation: Key Differences

## 1. Threshold Application Strategy

### QDK

Location: `python/src/qdk_chemistry/algorithms/qubit_mapper/qdk_qubit_mapper.py` (line ~280)

- Applies threshold **once at the end** after all Pauli terms are generated
- Uses `pauli_op.prune_threshold(threshold)` to filter final Pauli coefficients
- Default threshold: `1e-12`

### Qiskit

Location: `qiskit_nature/second_q/mappers/qubit_mapper.py` (lines 227-230)

```python
if char == "+":
    ret_op = ret_op.compose(times_creation_op[position], front=True).simplify()
elif char == "-":
    ret_op = ret_op.compose(times_annihilation_op[position], front=True).simplify()
```

- Calls `.simplify()` **after every single ladder operator composition**
- `.simplify()` uses `atol=1e-8` by default, filtering intermediate terms
- This means small terms get pruned during the transform, not just at the end

## 2. Default Tolerances

| Library | Default Threshold | Applied When |
|---------|------------------|--------------|
| QDK | `1e-12` | End of mapping |
| Qiskit | `1e-8` (`FermionicOp.atol`, `SparsePauliOp.atol`) | During each composition step |

## 3. Consequence

For the ethylene 4e4o Hamiltonian:

- **QDK with 1e-12**: 185 Pauli terms
- **Qiskit with 1e-8**: 141 Pauli terms (44 terms filtered during transform)
- **Qiskit with 1e-12**: 185 Pauli terms (matches QDK exactly)

The 4 terms observed with ~20% coefficient differences (`IIIIIXXZ`, `IIIIIYYZ`, `IXXZIIII`, `IYYZIIII`) had coefficients around `3.5e-8` to `4.5e-8`. Qiskit's intermediate filtering caused small numerical differences to accumulate because terms near the threshold boundary were being included/excluded at different stages.

## 4. Bug Fixed in QDK

Originally, QDK was incorrectly applying threshold to **input fermionic coefficients**:

```python
# WRONG (old code)
if abs(h_pq_alpha) > threshold:
    # generate Pauli terms
```

Fixed to only check for zero (threshold applied at end):

```python
# CORRECT (new code)
if h_pq_alpha != 0.0:
    # generate Pauli terms
```

## 5. How to Match Results

To get identical results between QDK and Qiskit, set Qiskit's tolerances to match:

```python
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import SparsePauliOp

FermionicOp.atol = 1e-12
SparsePauliOp.atol = 1e-12
```

## Summary

QDK's approach is arguably more mathematically principled—compute the full transformation, then prune at the end. Qiskit's approach is more aggressive about memory/performance by pruning during the transform, but this can cause small numerical differences and term count mismatches when comparing implementations.

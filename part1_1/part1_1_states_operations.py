"""
Assignment 1, Part 1.1 -- States and Operations
Team: 2022A7PS0800G (solo)

This script builds every state listed in Section 1 of the assignment that
is needed for Part 1.1 (States + Operations), and demonstrates operations
O1-O6 on them. It prints all the numbers used in report.pdf.

Run with:  python part1_1_states_operations.py
"""

import numpy as np
from qiskit.quantum_info import (
    Statevector, DensityMatrix, partial_trace, Operator
)

np.set_printoptions(precision=4, suppress=True)


def line(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def purity(rho: DensityMatrix) -> float:
    """gamma = Tr(rho^2)"""
    m = rho.data
    return np.real(np.trace(m @ m))


def bloch_vector(rho: DensityMatrix):
    """r_k = Tr(rho sigma_k) for one qubit, k = x, y, z"""
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    m = rho.data
    return np.real([np.trace(m @ X), np.trace(m @ Y), np.trace(m @ Z)])


def is_valid_density_matrix(rho: DensityMatrix, tol=1e-9):
    """O2: Hermitian, trace 1, positive semidefinite."""
    m = rho.data
    hermitian = np.allclose(m, m.conj().T, atol=tol)
    trace_one = abs(np.trace(m) - 1) < tol
    eigvals = np.linalg.eigvalsh(m)
    psd = np.all(eigvals > -tol)
    return hermitian, trace_one, psd, eigvals


# ---------------------------------------------------------------------
# SECTION 1: STATES  (Level 1 table)
# ---------------------------------------------------------------------
line("STATE 1: |0>  (computational basis state, pure)")
psi0 = Statevector.from_label("0")
rho0 = DensityMatrix(psi0)
print("Density matrix:\n", rho0.data)
print("Purity:", purity(rho0))
print("Bloch vector:", bloch_vector(rho0))

line("STATE 2: |+> = (|0> + |1>)/sqrt(2)  (pure superposition)")
psi_plus = Statevector.from_label("+")
rho_plus = DensityMatrix(psi_plus)
print("Density matrix:\n", rho_plus.data)
print("Purity:", purity(rho_plus))
print("Bloch vector:", bloch_vector(rho_plus))

line("STATE 3: equal mixture of |0> and |1>  (mixed)")
rho0_ = DensityMatrix(Statevector.from_label("0"))
rho1_ = DensityMatrix(Statevector.from_label("1"))
rho_mix01 = 0.5 * rho0_ + 0.5 * rho1_
print("Density matrix:\n", rho_mix01.data)
print("Purity:", purity(rho_mix01))
print("Bloch vector:", bloch_vector(rho_mix01))

line("STATE 4: equal mixture of |+> and |->  (mixed)")
rho_plus_ = DensityMatrix(Statevector.from_label("+"))
rho_minus_ = DensityMatrix(Statevector.from_label("-"))
rho_mixpm = 0.5 * rho_plus_ + 0.5 * rho_minus_
print("Density matrix:\n", rho_mixpm.data)
print("Purity:", purity(rho_mixpm))
print("Bloch vector:", bloch_vector(rho_mixpm))

line("STATE 5: a partly mixed state (purity strictly between 1/2 and 1)")
# Built by mixing |0> and the maximally mixed state, as the hint suggests:
# rho = q|0><0| + (1-q) I/2, with q chosen so purity is strictly between 0.5 and 1
q = 0.6
I2 = DensityMatrix(np.eye(2) / 2)
rho_partial = q * rho0_ + (1 - q) * I2
print(f"Built as {q}*|0><0| + {1-q}*(I/2)")
print("Density matrix:\n", rho_partial.data)
print("Purity:", purity(rho_partial))
print("Bloch vector:", bloch_vector(rho_partial))

# ---------------------------------------------------------------------
# SECTION 1: STATES (Level 2 entangled states, needed to illustrate O6)
# ---------------------------------------------------------------------
line("STATE 6: Bell state (|00> + |11>)/sqrt(2)")
bell = Statevector.from_label("00") + Statevector.from_label("11")
bell = bell / np.sqrt(2)
rho_bell = DensityMatrix(bell)
print("Full state purity:", purity(rho_bell))

line("STATE 7: GHZ state (|000> + |111>)/sqrt(2)")
ghz = Statevector.from_label("000") + Statevector.from_label("111")
ghz = ghz / np.sqrt(2)
rho_ghz = DensityMatrix(ghz)
print("Full state purity:", purity(rho_ghz))

# ---------------------------------------------------------------------
# SECTION 2: OPERATIONS
# ---------------------------------------------------------------------
line("O2: Validity checks on all five L1 states")
for name, rho in [
    ("|0>", rho0), ("|+>", rho_plus),
    ("mix(|0>,|1>)", rho_mix01), ("mix(|+>,|->)", rho_mixpm),
    ("partly mixed", rho_partial),
]:
    herm, tr1, psd, eigs = is_valid_density_matrix(rho)
    print(f"{name:16s} Hermitian={herm}  Trace=1:{tr1}  PSD={psd}  eigenvalues={eigs}")

line("O4: Measurement -- Z-basis vs X-basis for |+> and mix(|0>,|1>)")
Z0 = DensityMatrix(Statevector.from_label("0"))
Z1 = DensityMatrix(Statevector.from_label("1"))
Xp = DensityMatrix(Statevector.from_label("+"))
Xm = DensityMatrix(Statevector.from_label("-"))


def meas_prob(rho, basis_state):
    m = basis_state.data.reshape(-1, 1)
    return np.real((m.conj().T @ rho.data @ m)[0, 0])


print("In Z basis:")
print("  P(0) for |+>          =", meas_prob(rho_plus, Statevector.from_label("0")))
print("  P(0) for mix(0,1)     =", meas_prob(rho_mix01, Statevector.from_label("0")))
print("In X basis:")
print("  P(+) for |+>          =", meas_prob(rho_plus, Statevector.from_label("+")))
print("  P(+) for mix(0,1)     =", meas_prob(rho_mix01, Statevector.from_label("+")))

line("O5: Unitary evolution example -- H|0> should equal |+>")
H = Operator.from_label("H")
evolved = Statevector.from_label("0").evolve(H)
print("H|0> =", evolved.data)
print("matches |+> :", np.allclose(evolved.data, Statevector.from_label("+").data))

line("O6: Partial trace -- Bell state and GHZ state")
rho_bell_A = partial_trace(rho_bell, [1])      # trace out qubit 1
print("Bell: reduced state of qubit 0 (tracing out qubit 1):\n", rho_bell_A.data)
print("Purity of reduced Bell qubit:", purity(DensityMatrix(rho_bell_A)))

rho_ghz_A = partial_trace(rho_ghz, [1, 2])     # trace out qubits 1 and 2
print("\nGHZ: reduced state of qubit 0 (tracing out qubits 1,2):\n", rho_ghz_A.data)
print("Purity of reduced GHZ qubit:", purity(DensityMatrix(rho_ghz_A)))

rho_ghz_AB = partial_trace(rho_ghz, [2])       # trace out only qubit 2
print("\nGHZ: reduced state of qubits 0,1 (tracing out qubit 2):\n", rho_ghz_AB.data)
print("Purity of this 2-qubit reduced state:", purity(DensityMatrix(rho_ghz_AB)))

print("\nDone. All numbers above are used in report.pdf.")

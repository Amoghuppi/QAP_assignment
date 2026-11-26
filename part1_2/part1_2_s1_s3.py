"""
Part 1.2 -- S1, S2, S3
Team 2022A7PS0800G (solo)
"""
import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, DensityMatrix, partial_trace

np.set_printoptions(precision=6, suppress=True)
TOL = 1e-10


def purity(rho):
    m = rho.data
    return np.real(np.trace(m @ m))


def line(t):
    print("\n" + "=" * 70 + "\n" + t + "\n" + "=" * 70)


# -----------------------------------------------------------------------
# S1: single-qubit state checks, purity to 1e-10
# -----------------------------------------------------------------------
line("S1: single-qubit state purity checks (tolerance 1e-10)")
rho0 = DensityMatrix(Statevector.from_label("0"))
rho_plus = DensityMatrix(Statevector.from_label("+"))
rho_mix01 = 0.5 * DensityMatrix(Statevector.from_label("0")) + 0.5 * DensityMatrix(Statevector.from_label("1"))
rho_mixpm = 0.5 * DensityMatrix(Statevector.from_label("+")) + 0.5 * DensityMatrix(Statevector.from_label("-"))
rho_partial = 0.6 * rho0 + 0.4 * DensityMatrix(np.eye(2) / 2)

checks = [
    ("|0>", rho0, 1.0),
    ("|+>", rho_plus, 1.0),
    ("mix(0,1)", rho_mix01, 0.5),
    ("mix(+,-)", rho_mixpm, 0.5),
    ("partly mixed", rho_partial, 0.68),
]
s1_pass = True
for name, rho, expected in checks:
    p = purity(rho)
    ok = abs(p - expected) < TOL
    s1_pass &= ok
    print(f"{name:14s} purity={p:.10f}  expected={expected:.10f}  match(<1e-10)={ok}")
print("S1 overall pass:", s1_pass)

# -----------------------------------------------------------------------
# S2: 10-gate, 3-qubit circuit -- statevector method vs density-matrix method
# -----------------------------------------------------------------------
line("S2: 10-gate 3-qubit circuit consistency check")
qc = QuantumCircuit(3)
qc.h(0)
qc.t(1)
qc.cx(0, 1)
qc.s(2)
qc.h(2)
qc.cx(1, 2)
qc.t(0)
qc.s(1)
qc.cx(2, 0)
qc.h(1)
print(qc.draw())

# Statevector method
sv = Statevector.from_label("000").evolve(qc)
rho_from_sv = DensityMatrix(sv)

# Density-matrix method (start from rho = |000><000|, evolve same circuit)
rho0_3q = DensityMatrix.from_label("000")
rho_from_dm = rho0_3q.evolve(qc)

diff = np.max(np.abs(rho_from_sv.data - rho_from_dm.data))
print("Max |difference| between statevector-route rho and density-matrix-route rho:", diff)
print("Agree to 1e-10:", diff < TOL)

# -----------------------------------------------------------------------
# S3: partial trace on Bell and GHZ, removing one qubit then two
# -----------------------------------------------------------------------
line("S3: entanglement via partial trace -- Bell and GHZ")
bell = (Statevector.from_label("00") + Statevector.from_label("11")) / np.sqrt(2)
rho_bell = DensityMatrix(bell)
bell_trace1 = partial_trace(rho_bell, [1])
print("Bell, trace out qubit 1 -> remaining qubit purity:", purity(DensityMatrix(bell_trace1)))

ghz = (Statevector.from_label("000") + Statevector.from_label("111")) / np.sqrt(2)
rho_ghz = DensityMatrix(ghz)
ghz_trace1 = partial_trace(rho_ghz, [2])          # remove 1 qubit -> 2-qubit reduced state
print("GHZ, trace out 1 qubit  -> 2-qubit reduced state purity:", purity(DensityMatrix(ghz_trace1)))
ghz_trace2 = partial_trace(rho_ghz, [1, 2])       # remove 2 qubits -> 1-qubit reduced state
print("GHZ, trace out 2 qubits -> 1-qubit reduced state purity:", purity(DensityMatrix(ghz_trace2)))

print("\nDone with S1-S3.")

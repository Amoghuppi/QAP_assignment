"""
Part 1.2 -- S5: dephasing trajectories
Aer statevector simulation with the dephasing noise model, where each shot
is a trajectory; the average over N shots is compared with the exact
density-matrix result, and the error is shown to shrink with N.

Team 2022A7PS0800G (solo)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, pauli_error

np.random.seed(564091267)   # team's trajectory seed, for reproducibility
p = 0.3

# --- exact density-matrix result ---
Z = np.array([[1, 0], [0, -1]])
rho_plus = DensityMatrix.from_label("+").data
rho_exact = (1 - p) * rho_plus + p * (Z @ rho_plus @ Z)
print("Exact density matrix after dephasing(p=0.3) on |+>:\n", rho_exact)

# --- trajectory setup ---
qc = QuantumCircuit(1)
qc.h(0)
qc.save_statevector(pershot=True)
noise_model = NoiseModel()
noise_model.add_quantum_error(pauli_error([("Z", p), ("I", 1 - p)]), "h", [0])
sim = AerSimulator(method="statevector", noise_model=noise_model, seed_simulator=564091267)

N_max = 20000
job = sim.run(qc, shots=N_max)
svs = np.array(job.result().data()["statevector"], dtype=complex)   # shape (N_max, 2)

Ns = np.unique(np.round(np.logspace(1, np.log10(N_max), 25)).astype(int))
errors = []
for N in Ns:
    chunk = svs[:N]
    rho_traj = np.einsum("ni,nj->ij", chunk, chunk.conj()) / N
    err = np.linalg.norm(rho_traj - rho_exact, ord="fro")
    errors.append(err)
errors = np.array(errors)

# fit slope on log-log to compare with expected 1/sqrt(N)
slope, intercept = np.polyfit(np.log(Ns), np.log(errors), 1)
print(f"\nFitted slope of log(error) vs log(N): {slope:.3f}  (expected about -0.5)")

plt.figure(figsize=(4.3, 3.6))
plt.loglog(Ns, errors, "o-", label="measured error")
plt.loglog(Ns, errors[0] * (Ns[0] / Ns) ** 0.5, "--", color="gray", label=r"$1/\sqrt{N}$ reference")
plt.xlabel("N (number of trajectories)")
plt.ylabel("Frobenius-norm error vs exact $\\rho$")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figs/s5_trajectories.png", dpi=140)
print("Saved figs/s5_trajectories.png")
print(f"Error at N={Ns[0]}: {errors[0]:.5f}   Error at N={Ns[-1]}: {errors[-1]:.5f}")

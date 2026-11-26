"""
Part 1.2 -- S6, S7: team instance simulation and trajectories.
Team 2022A7PS0800G (solo). Instance: n=4, depolarising, p=0.082,
subsystem_A=[0,1], traj_seed=564091267.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import DensityMatrix, Kraus, partial_trace, Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

with open("../instance.json") as f:
    inst = json.load(f)

n = inst["n"]
gates = inst["gates"]
p = inst["noise"]["p"]
subsystem_A = inst["subsystem_A"]
seed = inst["traj_seed"]
print(f"Instance: n={n}, {len(gates)} gates, depolarising p={p}, subsystem_A={subsystem_A}, seed={seed}")


def von_neumann_entropy(rho_matrix):
    eigs = np.linalg.eigvalsh(rho_matrix)
    eigs = eigs[eigs > 1e-12]
    return float(-np.sum(eigs * np.log2(eigs)))


def purity(m):
    return np.real(np.trace(m @ m))


def build_circuit():
    qc = QuantumCircuit(n)
    for g in gates:
        if g["g"] == "CNOT":
            qc.cx(g["q"][0], g["q"][1])
        else:
            getattr(qc, g["g"].lower())(g["q"][0])
    return qc


# -----------------------------------------------------------------------
# S6a: exact route -- quantum_info, gate-by-gate, with depolarising Kraus
# channel applied independently to each qubit touched by the gate
# -----------------------------------------------------------------------
dep_kraus = Kraus(depolarizing_error(p, 1).to_quantumchannel())

rho = DensityMatrix.from_label("0" * n)
for g in gates:
    if g["g"] == "CNOT":
        c, t = g["q"]
        cx_op = QuantumCircuit(n)
        cx_op.cx(c, t)
        rho = rho.evolve(cx_op)
        rho = rho.evolve(dep_kraus, qargs=[c])
        rho = rho.evolve(dep_kraus, qargs=[t])
    else:
        (q,) = g["q"]
        single = QuantumCircuit(n)
        getattr(single, g["g"].lower())(q)
        rho = rho.evolve(single)
        rho = rho.evolve(dep_kraus, qargs=[q])

rho_exact = rho.data
purity_exact = purity(rho_exact)
S_exact = von_neumann_entropy(rho_exact)
rhoA_exact = partial_trace(DensityMatrix(rho_exact), [q for q in range(n) if q not in subsystem_A]).data
SA_exact = von_neumann_entropy(rhoA_exact)

print("\n--- Exact Kraus route (quantum_info) ---")
print("purity(rho)       =", purity_exact)
print("S(rho)            =", S_exact)
print("S(rho_A)          =", SA_exact)

# -----------------------------------------------------------------------
# S6b: Aer route -- density-matrix method + noise model
# -----------------------------------------------------------------------
noise_model = NoiseModel()
single_qubit_gates = ["h", "s", "t", "x"]
err1 = depolarizing_error(p, 1)
for gname in single_qubit_gates:
    for q in range(n):
        noise_model.add_quantum_error(err1, gname, [q])
err2 = err1.tensor(err1)   # independent single-qubit depolarising on each of the 2 CNOT qubits
for a in range(n):
    for b in range(n):
        if a != b:
            noise_model.add_quantum_error(err2, "cx", [a, b])

qc = build_circuit()
qc.save_density_matrix()
sim = AerSimulator(method="density_matrix", noise_model=noise_model)
result = sim.run(qc, shots=1).result()
rho_aer = np.asarray(result.data()["density_matrix"])

purity_aer = purity(rho_aer)
S_aer = von_neumann_entropy(rho_aer)
rhoA_aer = partial_trace(DensityMatrix(rho_aer), [q for q in range(n) if q not in subsystem_A]).data
SA_aer = von_neumann_entropy(rhoA_aer)

print("\n--- Aer route (density_matrix method + noise model) ---")
print("purity(rho)       =", purity_aer)
print("S(rho)            =", S_aer)
print("S(rho_A)          =", SA_aer)

max_diff = np.max(np.abs(rho_exact - rho_aer))
print("\nMax |difference| between the two full density matrices:", max_diff)

# save results for results.json / report
s6_results = {
    "purity": float(purity_exact),
    "entropy": float(S_exact),
    "entropy_A": float(SA_exact),
    "max_diff_routes": float(max_diff),
}
with open("s6_results.json", "w") as f:
    json.dump(s6_results, f, indent=2)
print("\nSaved s6_results.json")

# -----------------------------------------------------------------------
# S7: trajectory simulation of the noisy instance circuit, with the
# team's seed, error vs number of shots N
# -----------------------------------------------------------------------
qc_traj = build_circuit()
qc_traj.save_statevector(pershot=True)
sim_traj = AerSimulator(method="statevector", noise_model=noise_model, seed_simulator=seed)

N_max = 20000
job = sim_traj.run(qc_traj, shots=N_max)
svs = np.array(job.result().data()["statevector"], dtype=complex)  # (N_max, 2**n)

Ns = np.unique(np.round(np.logspace(1, np.log10(N_max), 25)).astype(int))
errors = []
for N in Ns:
    chunk = svs[:N]
    rho_traj = np.einsum("ni,nj->ij", chunk, chunk.conj()) / N
    err = np.linalg.norm(rho_traj - rho_exact, ord="fro")
    errors.append(err)
errors = np.array(errors)

slope, intercept = np.polyfit(np.log(Ns), np.log(errors), 1)
print(f"\nS7: fitted slope of log(error) vs log(N): {slope:.3f} (expected about -0.5)")
print(f"Error at N={Ns[0]}: {errors[0]:.5f}, Error at N={Ns[-1]}: {errors[-1]:.5f}")

plt.figure(figsize=(4.3, 3.6))
plt.loglog(Ns, errors, "o-", label="measured error")
plt.loglog(Ns, errors[0] * (Ns[0] / Ns) ** 0.5, "--", color="gray", label=r"$1/\sqrt{N}$ reference")
plt.xlabel("N (trajectories)")
plt.ylabel("Frobenius error vs exact instance $\\rho$")
plt.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figs/s7_instance_trajectories.png", dpi=140)
print("Saved figs/s7_instance_trajectories.png")

s7_results = {"slope": float(slope), "error_at_Nmax": float(errors[-1]), "n_traj": int(N_max)}
with open("s7_results.json", "w") as f:
    json.dump(s7_results, f, indent=2)
print("Saved s7_results.json")

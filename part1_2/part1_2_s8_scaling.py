"""
Part 1.2 -- S8: scaling benchmark.
Fixed-depth circuit family (3 layers of H + nearest-neighbor CNOT ladder,
independent of n), run with Aer's statevector and density-matrix methods
for growing n. Reports memory (final-state array size) and wall time.

Team 2022A7PS0800G (solo)
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator


def fixed_depth_circuit(n, layers=3):
    qc = QuantumCircuit(n)
    for _ in range(layers):
        for q in range(n):
            qc.h(q)
        for q in range(0, n - 1):
            qc.cx(q, q + 1)
    return qc


def time_run(n, method, repeats=3):
    qc = fixed_depth_circuit(n)
    if method == "statevector":
        qc.save_statevector()
    else:
        qc.save_density_matrix()
    sim = AerSimulator(method=method)
    best = None
    for _ in range(repeats):     # repeat and keep the best time (reduces noise)
        t0 = time.perf_counter()
        result = sim.run(qc, shots=1).result()
        t1 = time.perf_counter()
        dt = t1 - t0
        if best is None or dt < best:
            best = dt
    key = "statevector" if method == "statevector" else "density_matrix"
    arr = np.asarray(result.data()[key])
    return best, arr.nbytes


sv_ns = list(range(2, 19))          # statevector: cheap, go higher
dm_ns = list(range(2, 12))          # density matrix: 4^n blows up fast

sv_times, sv_mem = [], []
for n in sv_ns:
    t, mem = time_run(n, "statevector")
    sv_times.append(t)
    sv_mem.append(mem)
    print(f"[statevector] n={n:2d}  time={t:.4f}s  mem={mem/1e6:.3f} MB")

dm_times, dm_mem = [], []
for n in dm_ns:
    t, mem = time_run(n, "density_matrix")
    dm_times.append(t)
    dm_mem.append(mem)
    print(f"[density_matrix] n={n:2d}  time={t:.4f}s  mem={mem/1e6:.3f} MB")

# fit log(time) vs n to get the effective base (2^n -> slope*log(2) per n)
sv_slope, _ = np.polyfit(sv_ns[3:], np.log(sv_times[3:]), 1)   # skip tiny-n overhead-dominated points
dm_slope, _ = np.polyfit(dm_ns[3:], np.log(dm_times[3:]), 1)
print(f"\nStatevector time doubling base: {np.exp(sv_slope):.3f}  (theory: 2.0)")
print(f"Density-matrix time doubling base: {np.exp(dm_slope):.3f}  (theory: 4.0)")

fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
axes[0].semilogy(sv_ns, sv_times, "o-", label="statevector")
axes[0].semilogy(dm_ns, dm_times, "s-", label="density_matrix")
axes[0].set_xlabel("n (qubits)"); axes[0].set_ylabel("time (s)"); axes[0].legend(fontsize=8)
axes[0].set_title("runtime")

axes[1].semilogy(sv_ns, np.array(sv_mem) / 1e6, "o-", label="statevector")
axes[1].semilogy(dm_ns, np.array(dm_mem) / 1e6, "s-", label="density_matrix")
axes[1].set_xlabel("n (qubits)"); axes[1].set_ylabel("final-state memory (MB)"); axes[1].legend(fontsize=8)
axes[1].set_title("memory")
plt.tight_layout()
plt.savefig("figs/s8_scaling.png", dpi=140)
print("Saved figs/s8_scaling.png")

import json
with open("s8_results.json", "w") as f:
    json.dump({
        "sv_ns": sv_ns, "sv_times": sv_times, "sv_mem_bytes": sv_mem,
        "dm_ns": dm_ns, "dm_times": dm_times, "dm_mem_bytes": dm_mem,
        "sv_time_doubling_base": float(np.exp(sv_slope)),
        "dm_time_doubling_base": float(np.exp(dm_slope)),
    }, f, indent=2)
print("Saved s8_results.json")

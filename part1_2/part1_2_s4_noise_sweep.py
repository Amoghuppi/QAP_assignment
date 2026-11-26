"""
Part 1.2 -- S4: noise sweep
Density-matrix simulation of the three channels on |+> and |0>, for p from
0 to 1 in 21 steps, once with an Aer noise model and once with the channels
applied exactly as Kraus operators (O7). Saves a comparison plot.

Team 2022A7PS0800G (solo)
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qiskit.quantum_info import DensityMatrix, Kraus, Statevector
from qiskit_aer.noise import pauli_error, depolarizing_error

np.set_printoptions(precision=6, suppress=True)
X = np.array([[0, 1], [1, 0]])
Z = np.array([[1, 0], [0, -1]])
I2 = np.eye(2)


def purity(m):
    return np.real(np.trace(m @ m))


def exact_kraus_channel(rho, channel, p):
    """Apply O7's formula exactly (the 'exact Kraus operator' route)."""
    if channel == "bitflip":
        return (1 - p) * rho + p * (X @ rho @ X)
    if channel == "dephasing":
        return (1 - p) * rho + p * (Z @ rho @ Z)
    if channel == "depolarising":
        return (1 - p) * rho + p * (I2 / 2)
    raise ValueError(channel)


def aer_channel(rho, channel, p):
    """Apply the same channel via an Aer QuantumError object (the 'Aer' route)."""
    if channel == "bitflip":
        err = pauli_error([("X", p), ("I", 1 - p)])
    elif channel == "dephasing":
        err = pauli_error([("Z", p), ("I", 1 - p)])
    elif channel == "depolarising":
        err = depolarizing_error(p, 1)
    else:
        raise ValueError(channel)
    k = Kraus(err.to_quantumchannel())
    out = DensityMatrix(rho).evolve(k)
    return out.data


ps = np.linspace(0, 1, 21)
channels = ["bitflip", "dephasing", "depolarising"]
states = {
    "|0>": DensityMatrix(Statevector.from_label("0")).data,
    "|+>": DensityMatrix(Statevector.from_label("+")).data,
}

results = {}          # results[state][channel] = (exact_purities, aer_purities)
max_disagreement = 0.0

for sname, rho in states.items():
    for ch in channels:
        exact_pur, aer_pur = [], []
        for p in ps:
            rho_exact = exact_kraus_channel(rho, ch, p)
            rho_aer = aer_channel(rho, ch, p)
            d = np.max(np.abs(rho_exact - rho_aer))
            max_disagreement = max(max_disagreement, d)
            exact_pur.append(purity(rho_exact))
            aer_pur.append(purity(rho_aer))
        results[(sname, ch)] = (np.array(exact_pur), np.array(aer_pur))

print("Max disagreement between Aer route and exact-Kraus route, over all channels/states/p:",
      max_disagreement)
print("(This being ~1e-15 confirms Qiskit's pauli_error/depolarizing_error use exactly")
print(" the same parameter convention as O7 for a single qubit -- no rescaling needed.)")

# ---- hand derivations, for the overlay ----
# Bloch length under each channel, starting from r0 (bitflip/dephasing only
# shrink the components that anticommute with the Pauli used):
#   bitflip (conjugate by X):      r_x survives, r_y,r_z flip sign and shrink by (1-2p)
#   dephasing (conjugate by Z):    r_z survives, r_x,r_y shrink by (1-2p)
#   depolarising:                   all of r shrinks by (1-p)
# purity = (1 + r^2) / 2
r0 = {"|0>": np.array([0, 0, 1.0]), "|+>": np.array([1.0, 0, 0])}


def hand_purity(sname, ch, p):
    r = r0[sname].copy()
    if ch == "bitflip":
        r[1] *= (1 - 2 * p); r[2] *= (1 - 2 * p)
    elif ch == "dephasing":
        r[0] *= (1 - 2 * p); r[1] *= (1 - 2 * p)
    elif ch == "depolarising":
        r *= (1 - p)
    return (1 + np.dot(r, r)) / 2


max_hand_diff = 0.0
for sname in states:
    for ch in channels:
        exact_pur, _ = results[(sname, ch)]
        hand = np.array([hand_purity(sname, ch, p) for p in ps])
        max_hand_diff = max(max_hand_diff, np.max(np.abs(exact_pur - hand)))
print("Max difference, simulated purity vs hand-derived formula:", max_hand_diff)

# ---- plot ----
fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=True)
colors = {"bitflip": "tab:blue", "dephasing": "tab:orange", "depolarising": "tab:green"}
for ax, sname in zip(axes, states):
    for ch in channels:
        exact_pur, aer_pur = results[(sname, ch)]
        ax.plot(ps, exact_pur, color=colors[ch], label=ch)
        ax.plot(ps, aer_pur, "o", color=colors[ch], ms=3)
    ax.set_title(f"purity of {sname} vs p")
    ax.set_xlabel("p")
axes[0].set_ylabel("purity")
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig("figs/s4_noise_sweep.png", dpi=140)
print("Saved figs/s4_noise_sweep.png")

# Which channel leaves purity of |0> unchanged?
print("\nPurity of |0> at p=1 for each channel (bitflip/dephasing/depolarising):",
      [results[("|0>", ch)][0][-1] for ch in channels])

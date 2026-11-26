# Assignment 1, Part 1.2 — Simulations and Analysis

Team: 2022A7PS0800G (worked individually). Level attempted: 3 (full).

## Run commands

```
python part1_2_s1_s3.py          # S1 purity checks, S2 route-consistency, S3 partial trace
python part1_2_s4_noise_sweep.py # S4 noise-channel sweep (saves figs/s4_noise_sweep.png)
python part1_2_s5_trajectories.py# S5 generic trajectory convergence (saves figs/s5_trajectories.png)
python part1_2_s6_s7_instance.py # S6 team instance (two routes), S7 instance trajectories
                                  # (saves s6_results.json, s7_results.json, figs/s7_instance_trajectories.png)
python part1_2_s8_scaling.py     # S8 memory/time scaling benchmark (saves s8_results.json, figs/s8_scaling.png)
```

## Files

- `part1_2_s1_s3.py`, `part1_2_s4_noise_sweep.py`, `part1_2_s5_trajectories.py`,
  `part1_2_s6_s7_instance.py`, `part1_2_s8_scaling.py` — code for S1 through S8.
- `figs/` — all plots produced by the code.
- `s6_results.json`, `s7_results.json`, `s8_results.json` — intermediate raw numbers
  (also summarized in `results.json`).
- `results.json` — final machine-readable results in the assignment's required format.
- `instance.json` — the Level-3 instance used throughout (team code 0800: n=4,
  depolarising noise p=0.082, subsystem A = qubits 0,1).

## Environment

See `environment.txt` for exact library versions (same as Part 1.1).

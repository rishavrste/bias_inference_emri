---
name: project-emri-inference
description: "EMRI/IMRI gravitational wave bias inference — current state, constraints, and next steps"
metadata: 
  node_type: memory
  type: project
  originSessionId: b3132178-fa52-41c3-b128-9eed71c247a3
---

Running bias inference for EMRI/IMRI gravitational wave signals. Injecting 2PA signals and recovering with 0PA templates to measure parameter bias and overlap loss. Three 25-point grids: EMRI, IMRI, IMRI_TAIL.

**Key constraints:**
- PBS project: `#PBS -P CFP03-CF-051`
- GitHub: account `joshbmat`, email `josh.mat@nus.edu.sg`
- Branch: `josh/inference-setup` only. Never touch `2PA_inclusion` or `main`
- Python: `/home/svu/josh.mat/miniconda3/bin/python3` directly inside singularity
- T-channel excluded: `nchannels=2` (A,E only)
- FEW CUDA NOT thread-safe: always `de_workers=1`

**Active focus: IMRI_TAIL 0PA grid**

A full re-run is planned with:
- `optimizer=differential_evolution`
- `parameter_selected=intrinsic_phase` (7D: m1,m2,a,p0,e0,Phi_phi0,Phi_r0)
- `target_func=optimal_snr` (NOT phase_max — see why below)
- `--use-global-warmstart` (reads from global result array directly)
- Same `de_popsize=5`, `de_maxiter=1500`

Submit command:
```bash
qsub -J 0-24 -v "RUN_TYPE=0pa_vs_2pa,GRID_TYPE=IMRI_TAIL,OPTIMIZER=differential_evolution,PARAMETER_SELECTED=intrinsic_phase,TARGET_FUNC=optimal_snr,USE_GLOBAL_WARMSTART=1" src/inference.pbs
```

**Why optimal_snr not optimal_snr_phase_max:**
`optimal_snr_phase_max` uses `|<h|s>|` which can have spurious local maxima where the actual overlap is negative. Pt 9 was the example: raw objective ~11.7 but actual overlap -0.585. `optimal_snr` uses `Re(<h|s>)` directly, so negative overlap is always penalised.

**Why:** Confirmed in session on 2026-06-15 after analysing pt 9 failure.
**How to apply:** Always use `target_func=optimal_snr` for `intrinsic_phase` DE runs.

**Current best overlaps (IMRI_TAIL 0PA, 2026-06-15):**
pt 0: 0.9993, pt 1: 0.9805, pt 2: 0.9776, pt 3: 0.9795, pt 4: 0.9949
pt 5: 0.9986, pt 6: 0.9499, pt 7: 0.9517, pt 8: 0.9980, pt 9: 0.9380
pt 10: 0.8794, pt 11: 0.8915, pt 12: 0.9590, pt 13: 0.9657, pt 14: 0.9556
pt 15: 0.9977, pt 16: 0.8315, pt 17: 0.8638, pt 18: 0.9059, pt 19: 0.8904
pt 20: 0.6893, pt 21: 0.9978, pt 22: 0.7788, pt 23: 0.8256, pt 24: 0.9431

Hardest points: 20 (0.689), 22 (0.779), 23 (0.826). These need the full 1500-gen DE run.

**Phase convention:**
- All phases stored in [0, 2π]
- Signal phases for all IMRI_TAIL points: Phi_phi0=0.1, Phi_r0=0.3
- Bias = `((inferred - signal + π) % 2π) - π` (circular difference)
- After any run: run `python3 src/fix_phases.py` to re-wrap any escaped phases

**Bugs fixed (all committed to josh/inference-setup):**
- DE refine IndexError (padding diag_sigma to ndim) — 49e4d51
- `_find_best_starting_point` missing DE dirs — 99b7092
- Phases not saved to result_array with intrinsic_phase — 55f561b
- NM phases outside [0, 2π] — 0cdcc44
- DE storing raw objective (~19) not actual overlap — 399f3a4

**CLAUDE.md** in repo root has full technical reference. See it for code structure, all CLI flags, post-processing checklist, and submission commands. [[feedback-commits]] [[feedback-de-workers]]

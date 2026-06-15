# CLAUDE.md — Bias Inference for EMRI/IMRI Gravitational Waves

This file gives an LLM assistant (Claude Code or similar) the context needed to resume work on this project without a long briefing.

---

## Project Overview

We inject 2PA (second-order post-adiabatic) EMRI/IMRI gravitational wave signals and recover them using 0PA templates, then measure the parameter bias and overlap loss. The code runs a grid of 25 signal points and optimises the overlap between the injected signal and the best-fitting 0PA template.

**Three grids** (each 25 points):
- `EMRI` — extreme mass ratio inspirals
- `IMRI` — intermediate mass ratio inspirals
- `IMRI_TAIL` — IMRI signals in the tail of the eccentricity distribution (harder cases)

**Run types:**
- `0pa_vs_2pa` — 0PA template vs 2PA signal (main focus)
- `1pa_vs_2pa` — 1PA template vs 2PA signal

---

## Hard Constraints — Never Violate

| Constraint | Detail |
|---|---|
| Git branch | Only commit/push to `josh/inference-setup`. Never touch `2PA_inclusion` or `main`. |
| Commit messages | Do NOT add `Co-Authored-By:` lines. |
| PBS project | `#PBS -P CFP03-CF-051` |
| Python | `/home/svu/josh.mat/miniconda3/bin/python3` inside the singularity container |
| T-channel | Always `nchannels=2` (A,E only — T-channel excluded) |
| DE workers | Always `de_workers=1`. FEW CUDA waveforms (`interpolate.cu`) are NOT thread-safe and NOT fork-safe with Python 3.13. ThreadPoolExecutor causes `GPUassert: illegal memory access`. |

---

## Cluster Setup (NUS HPC — Hopper)

```bash
# Singularity image
image="/app1/common/singularity-img/hopper/cuda/cuda_12.4.1-cudnn-devel-u22.04.sif"
module load singularity

# Run python inside container (with GPU)
singularity exec --nv -e "$image" /home/svu/josh.mat/miniconda3/bin/python3 script.py

# PBS submission (GPU node)
#PBS -l select=1:ngpus=1:mem=128gb
#PBS -l walltime=10:00:00
#PBS -q auto
```

**Key paths:**
- Repo: `/home/svu/josh.mat/git_repos/bias_inference_emri/`
- Results (scratch, not in repo): `/scratch/josh.mat/opt_grid/results/`
- Logs: `/scratch/josh.mat/opt_grid/logs_inference/`
- Result arrays (in repo): `data/result_parameter_array_{TYPE}_{pa}.npy`
- Overlap arrays: `data/result_parameter_array_{TYPE}_{pa}_overlaps.npy`

---

## Code Structure

| File | Purpose |
|---|---|
| `src/inference.py` | Main script: runs optimizer, SKIP logic, saves results |
| `src/config_paris.py` | `Config` class with all default settings |
| `src/misc.py` | Waveform evaluation, inner products, `calculate_detection_overlap` |
| `src/inference.pbs` | PBS submission script — all CLI flags passed via env vars |
| `src/fix_phases.py` | Post-processing: wraps phases to [0, 2π] in starting_point files and global result array |
| `src/precompute_fisher.py` | Precomputes Fisher matrix diag sigmas for DE bounds |
| `data/signal_parameter_array_{TYPE}.npy` | Injected signal parameters (25×17) |
| `data/result_parameter_array_{TYPE}_{pa}.npy` | Best recovered parameters (25×17, float64) |
| `data/result_parameter_array_{TYPE}_{pa}_overlaps.npy` | Best overlaps (25,) float64 |
| `data/fisher_cache/` | Cached Fisher matrix results per grid point |

---

## Result Array Layout

Both signal and result arrays have the same 17-column layout:

```
["m1","m2","a","p0","e0","xI0","dist","qS","phiS","qK","phiK",
 "Phi_phi0","Phi_theta0","Phi_r0","dt","T","chi2"]
  0     1   2   3   4    5     6    7    8    9    10
  11         12          13        14   15   16
```

- Phases **Phi_phi0** at index 11, **Phi_r0** at index 13
- Convention: phases wrapped to **[0, 2π]**
- Signal phases for all IMRI_TAIL points: Phi_phi0=0.1, Phi_r0=0.3
- Circular bias: `((inferred - signal + π) % 2π) - π`

---

## Optimizer Workflow

Three-stage pipeline for DE optimizer with `parameter_selected=intrinsic_phase`:

**Stage 1 — DE global search (7D)**
- Parameters: (m1, m2, a, p0, e0, Phi_phi0, Phi_r0)
- Bounds: Fisher ±`prior_sigma_range`σ for intrinsic; ±π for phase dims
- `de_popsize=5` → 35 members/gen; `de_maxiter=1500`
- Objective: `target_func` (see below)

**Stage 2 — DE refine (narrower bounds around DE best)**
- `de_refine_maxiter=150`, `de_refine_popsize=8`

**Stage 3 — Nelder-Mead polish**
- `nm_refine_maxiter=5000`, `nm_refine_maxfev=10000`

**SKIP logic:** Global result array is only updated if `new_overlap > existing_overlap`. Previous best is always preserved.

### Target functions

| `target_func` | Objective | Notes |
|---|---|---|
| `optimal_snr` | `Re(<h\|s>) / sqrt(<h\|h><s\|s>)` | True [0,1] overlap — **preferred** |
| `optimal_snr_phase_max` | `\|<h\|s>\| / sqrt(...)` | Maximises over global detector phase. Can produce spurious local maxima with negative actual overlap. |

**Use `optimal_snr` for `intrinsic_phase` runs.** `optimal_snr_phase_max` caused pt 9 to converge to a region with raw objective ~11.7 but actual overlap -0.585 (a spurious maximum of `|<h|s>|`).

### Warm-start loading

On startup, `_find_best_starting_point` scans `paris_*_id_*` and `differential_evolution_*_run_id_*` subdirectories of the point's result dir, reads `results.final_overlap` from JSONs, and loads `starting_point_N.npy` from the best run.

With `--use-global-warmstart`, the global result array row for point `i` is used directly (recommended for re-runs — it's the single source of truth).

**Phase bounds requirement:** Warm-start phases must be in [0, 2π] so that ±π bounds cover the full circle. Run `fix_phases.py` after any run to ensure this.

---

## PBS Job Submission

```bash
# Full 25-point IMRI_TAIL re-run (recommended settings):
qsub -J 0-24 -v "RUN_TYPE=0pa_vs_2pa,GRID_TYPE=IMRI_TAIL,OPTIMIZER=differential_evolution,PARAMETER_SELECTED=intrinsic_phase,TARGET_FUNC=optimal_snr,USE_GLOBAL_WARMSTART=1" src/inference.pbs

# Single point (e.g. pt 9):
qsub -v "GRID_START=9,RUN_TYPE=0pa_vs_2pa,GRID_TYPE=IMRI_TAIL,OPTIMIZER=differential_evolution,PARAMETER_SELECTED=intrinsic_phase,TARGET_FUNC=optimal_snr,USE_GLOBAL_WARMSTART=1" src/inference.pbs

# Check job status:
qstat -u josh.mat -a

# Tail a log:
tail -f /scratch/josh.mat/opt_grid/logs_inference/inference_9.out.<JOBID>
```

**All supported env-var overrides in `inference.pbs`:**

| Env var | CLI flag | Effect |
|---|---|---|
| `RUN_TYPE` | `--run-type` | `0pa_vs_2pa` or `1pa_vs_2pa` |
| `GRID_TYPE` | `--type` | `EMRI`, `IMRI`, or `IMRI_TAIL` |
| `OPTIMIZER` | `--optimizer` | `differential_evolution`, `paris`, `nelder-mead` |
| `PARAMETER_SELECTED` | `--parameter-selected` | `intrinsic` (5D) or `intrinsic_phase` (7D) |
| `TARGET_FUNC` | `--target-func` | `optimal_snr` or `optimal_snr_phase_max` |
| `DE_MAXITER` | `--de-maxiter` | DE generations |
| `DE_POPSIZE` | `--de-popsize` | Population multiplier (total = popsize × ndim) |
| `PRIOR_SIGMA_RANGE` | `--prior-sigma-range` | Fisher bound half-width in σ |
| `USE_GLOBAL_WARMSTART` | `--use-global-warmstart` | Warm-start from global result array |
| `REFINE_ONLY` | `--refine-only` | Skip stage-1, load checkpoint (PARIS path only) |

---

## Known Bugs Fixed (commit history)

| Commit | Bug | Fix |
|---|---|---|
| `49e4d51` | DE refine IndexError: `diag_sigma_fisher[j]` out of bounds when `ndim > len(fisher)` | Pad `diag_sigma` to `ndim` with `π/prior_sigma_range` for phase dims |
| `99b7092` | `_find_best_starting_point` only scanned `paris_*` dirs, missed DE results | Extended to also scan `differential_evolution_*_run_id_*` dirs |
| `55f561b` | Phases not written to `result_array` after NM with `intrinsic_phase` | Added `elif parameter_selected == 'intrinsic_phase':` branch to phase-saving code |
| `0cdcc44` | NM (unbounded) produced phases outside [0, 2π] | Wrap with `% (2π)` before saving |
| `399f3a4` | DE stored raw objective score (~19) instead of [0,1] overlap | Track `_best_de_overlap` separately from raw DE objective |

---

## Post-Processing Checklist

After any run completes:

1. **Wrap phases:** `python3 src/fix_phases.py` — fixes any phases outside [0, 2π] in `starting_point_*.npy` files and global result array.
2. **Overlap consistency check:** Recompute overlaps on GPU at stored parameters; compare with values in `_overlaps.npy`. (Submit as PBS job — CuPy fails on login node without GPU driver.)
3. Check logs for `[SKIP]` lines to confirm SKIP logic rejected bad results correctly.
4. Check for negative stored overlaps — these indicate a bug in phase saving or overlap computation.

---

## Current State (as of 2026-06-15)

**IMRI_TAIL 0PA grid** is the active focus. Current best overlaps:

| pt | overlap | pt | overlap |
|---|---|---|---|
| 0 | 0.9993 | 13 | 0.9657 |
| 1 | 0.9805 | 14 | 0.9556 |
| 2 | 0.9776 | 15 | 0.9977 |
| 3 | 0.9795 | 16 | 0.8315 |
| 4 | 0.9949 | 17 | 0.8638 |
| 5 | 0.9986 | 18 | 0.9059 |
| 6 | 0.9499 | 19 | 0.8904 |
| 7 | 0.9517 | 20 | 0.6893 |
| 8 | 0.9980 | 21 | 0.9978 |
| 9 | 0.9380 | 22 | 0.7788 |
| 10 | 0.8794 | 23 | 0.8256 |
| 11 | 0.8915 | 24 | 0.9431 |
| 12 | 0.9590 | | |

**Next step:** Full re-run with `optimal_snr` + `--use-global-warmstart` (see submission command above). Points 16-24 in particular still have room for improvement; pt 20 at 0.689 is the hardest.
